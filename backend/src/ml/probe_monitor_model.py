"""The probe head and its six score-combining rules (032 FR-6, BR-006).

A probe monitor is deliberately the simplest thing that can work: a linear
function of ONE layer's residual stream,

    score_t = w · standardise(z_t) + b

evaluated at every token, and then a **rule** that combines those per-token
scores into one score for the row. The rules are where the design lives, so they
are pure functions over tensors here, with no model, no database and no I/O.

EVERY RULE HAS TWO FORMS AND THEY MUST AGREE. The batch form scores a whole
sequence at once, which is what training and evaluation use. The online form
consumes one token at a time, which is what a serving runtime does while tokens
stream. Two implementations of one definition is exactly the shape that drifts —
so `tests/unit/test_probe_monitor_model.py` runs them against each other to 1e-6
on every rule, and the online form of `softmax`/`attention` uses the standard
running-max rescaling rather than a naive sum of exponentials, because that is
where the two forms part company first.

WHY `last` IS NOT STREAMABLE, when keeping the newest score is trivial. A
streaming monitor has to be able to answer "what is the verdict NOW" at every
token. `last` is not defined until generation stops: the score at the current
token is not the row's score, it is a guess that the row is about to end. Every
other rule's partial value IS its value over the tokens seen so far. That is the
distinction `STREAMABLE` records, and it is a property of the definition rather
than of the implementation — which is why the online form REFUSES rather than
returning the newest score and letting a caller believe it is monitoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

#: Every combining rule, in the order FR-6 lists them.
RULES: Tuple[str, ...] = ("mean", "max", "last", "softmax", "attention", "rolling_mean_max")

#: Rules whose value over the tokens seen so far IS their value — so a serving
#: runtime can emit a verdict at every token. `last` is absent BY DEFINITION,
#: not by omission; see the module docstring.
STREAMABLE: frozenset = frozenset(RULES) - {"last"}

#: Default softmax temperature. 1.0 makes `softmax` a self-weighted mean; as
#: tau -> 0 it approaches `max`, which is the relationship worth knowing when
#: reading a sweep that picked one over the other.
DEFAULT_TAU: float = 1.0

#: Default rolling window, in tokens.
DEFAULT_WINDOW: int = 16


def is_streamable(rule: str) -> bool:
    """Whether `rule` can be computed while tokens stream.

    Raises on an unknown rule rather than returning False: a typo that reads as
    "not streamable" would silently disable streaming for a rule that supports
    it, and the caller would see a plausible answer.
    """
    if rule not in RULES:
        raise ValueError(f"unknown combining rule {rule!r}; known: {', '.join(RULES)}")
    return rule in STREAMABLE


@dataclass(frozen=True)
class ProbeHead:
    """`w`, `b` and the standardisation the probe was TRAINED with.

    The normalisation travels with the head because it is part of the function.
    A head applied to unstandardised activations is not a worse probe, it is a
    different one — the same class of silent-wrong-basis error as encoding with
    an SAE's normalisation dropped.

    `std` is not trusted: a constant feature has std 0 and dividing by it makes
    every score infinite. A DEGENERATE CHANNEL (std at or below `eps`) is ZEROED
    instead — it varied not at all in training, so the weight fitted for it means
    nothing and it must contribute nothing.

    ⚠ CLAMPING WAS THE WRONG FIX AND AMPLIFIED INSTEAD. The first version did
    `std.clamp_min(eps)`, so a channel with std 0 divided by 1e-6: a drift of
    0.001 at serving time arrived at the dot product as **1000.0**, dominating
    every other channel and firing the monitor on a channel that carries no
    signal at all. The docstring claimed the clamped dimension "contributes
    nothing" while the code made it contribute more than all the rest combined —
    the dangerous direction, since a probe that fires is a probe that is believed.
    """

    weight: torch.Tensor            # (d_model,)
    bias: float = 0.0
    mean: Optional[torch.Tensor] = None   # (d_model,)
    std: Optional[torch.Tensor] = None    # (d_model,)
    #: The `attention` rule's learned query `q`, REQUIRED by that rule and unused
    #: by the other five. It lives on the head because IDL-53 requires it to
    #: travel inline in the exported definition: a probe whose weighting cannot be
    #: reconstructed is unserveable, and the first version of this module had
    #: nowhere to put it at all — so `attention` was a rule that could be swept
    #: and then never exported.
    attention_query: Optional[torch.Tensor] = None   # (d_model,)
    #: WHICH LAYER this readout reads. Part of the probe's IDENTITY, not a caller's
    #: bookkeeping: the same weights over a different layer are a different detector,
    #: and `forward_scores` has to know where to hook. It lives here for the reason
    #: `attention_query` does — a review found that rule had nowhere to put its query,
    #: so `attention` could be swept and then never exported. A head whose layer is
    #: only known to its caller is a head that cannot be serialised or served.
    layer: Optional[int] = None
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.weight.ndim != 1:
            raise ValueError(f"weight must be 1-D (d_model,), got {tuple(self.weight.shape)}")
        for name in ("mean", "std", "attention_query"):
            vector = getattr(self, name)
            if vector is not None and vector.shape != self.weight.shape:
                raise ValueError(
                    f"{name} has shape {tuple(vector.shape)} but weight is "
                    f"{tuple(self.weight.shape)} — a probe's normalisation must match its weights"
                )

    def standardise(self, activations: torch.Tensor) -> torch.Tensor:
        """(..., d_model) -> (..., d_model), using the TRAINING statistics.

        A channel whose training std is at or below `eps` is zeroed rather than
        divided by the floor — see the class docstring for the 1000x amplification
        that clamping produced. On training data the two agree exactly (a centred
        constant is 0); they differ only off it, which is where it matters.
        """
        out = activations
        if self.mean is not None:
            out = out - self.mean
        if self.std is not None:
            degenerate = self.std.abs() <= self.eps
            divisor = torch.where(degenerate, torch.ones_like(self.std), self.std)
            out = torch.where(degenerate, torch.zeros_like(out), out / divisor)
        return out

    def attention_logits(self, activations: torch.Tensor) -> torch.Tensor:
        """(..., T, d_model) -> (..., T). `q · standardise(z_t)`, for the `attention` rule.

        Standardised with the SAME statistics as the score, because the query was
        trained in that space. Refuses when no query is set rather than falling
        back to the scores — that fallback would silently turn `attention` into
        `softmax` at tau=1.
        """
        if self.attention_query is None:
            raise ValueError(
                "this probe has no attention_query, so the `attention` rule cannot be "
                "computed; using the scores as the weighting would silently make it "
                "`softmax` at tau=1"
            )
        return self.standardise(activations) @ self.attention_query

    def token_scores(self, activations: torch.Tensor) -> torch.Tensor:
        """(..., T, d_model) -> (..., T). The per-token score, before any rule."""
        if activations.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"activations are d_model={activations.shape[-1]} but this probe is "
                f"d_model={self.weight.shape[0]}"
            )
        return self.standardise(activations) @ self.weight + self.bias


# ── masking ────────────────────────────────────────────────────────────────────
# A mask is 1 for a real token and 0 for padding. It is REQUIRED reasoning, not
# an optimisation: ~48% of a padded batch here has been padding before, and a
# rule that folds pad positions into a mean or a max reports a number about the
# padding. Every rule below takes the mask and every test exercises it.

def _require_mask(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(scores, dtype=torch.bool)
    if mask.shape != scores.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match scores {tuple(scores.shape)}"
        )
    return mask.bool()


def _require_any_token(valid: torch.Tensor) -> None:
    """A row with no real tokens has no score. Refuse rather than invent one.

    `mean` over an all-pad row is 0/0 and `max` is -inf; both look like numbers.
    """
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("a row has no unmasked tokens, so it has no score to combine")


# ── the six rules, batch form ─────────────────────────────────────────────────

def combine(
    rule: str,
    scores: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    attention_logits: Optional[torch.Tensor] = None,
    tau: float = DEFAULT_TAU,
    window: int = DEFAULT_WINDOW,
) -> torch.Tensor:
    """Combine per-token scores into one score per row.

    `scores` is (..., T). The return is (...,) — one score per row.

    `attention_logits` is (..., T) and REQUIRED by `attention` only: that rule
    weights tokens by `softmax(q · z_t)` while the value stays `w · z_t`, so the
    weighting is a learned function of the activation and NOT of the score. They
    are different tensors on purpose; passing the scores as the logits silently
    turns `attention` into `softmax` at tau=1.
    """
    valid = _require_mask(scores, mask)
    _require_any_token(valid)

    if rule == "mean":
        return (scores * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)

    if rule == "max":
        return scores.masked_fill(~valid, float("-inf")).amax(dim=-1)

    if rule == "last":
        index = _last_valid_index(valid)
        return scores.gather(-1, index.unsqueeze(-1)).squeeze(-1)

    if rule == "softmax":
        if tau <= 0:
            raise ValueError(f"softmax temperature must be > 0, got {tau}")
        weights = _masked_softmax(scores / tau, valid)
        return (weights * scores).sum(dim=-1)

    if rule == "attention":
        if attention_logits is None:
            raise ValueError(
                "the `attention` rule needs `attention_logits` (q · z_t); passing the "
                "scores instead would make it `softmax` at tau=1 under another name"
            )
        if attention_logits.shape != scores.shape:
            raise ValueError(
                f"attention_logits {tuple(attention_logits.shape)} must match scores "
                f"{tuple(scores.shape)}"
            )
        weights = _masked_softmax(attention_logits, valid)
        return (weights * scores).sum(dim=-1)

    if rule == "rolling_mean_max":
        return _rolling_mean_max(scores, valid, window)

    raise ValueError(f"unknown combining rule {rule!r}; known: {', '.join(RULES)}")


def _last_valid_index(valid: torch.Tensor) -> torch.Tensor:
    """Index of the last unmasked token per row, for ANY padding side.

    The first version counted real tokens and used the count minus one, which is
    only the last real token under RIGHT padding. Under left padding
    (`mask=[0,0,1,1]`) it returned index 1 — a PAD position — and the docstring
    claimed left-padding correctness while every test used right padding. A
    review caught it; `test_last_is_correct_under_left_padding` pins it now.

    Searching from the right is what actually answers the question: the last
    position whose mask is 1, wherever the padding sits.
    """
    length = valid.shape[-1]
    reversed_first_valid = valid.flip(-1).to(torch.int8).argmax(dim=-1)
    return (length - 1) - reversed_first_valid


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax over unmasked positions only, max-shifted for stability."""
    filled = logits.masked_fill(~valid, float("-inf"))
    shifted = filled - filled.amax(dim=-1, keepdim=True)
    weights = shifted.exp() * valid
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


def _rolling_mean_max(scores: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
    """Maximum over windows of `window` tokens of the windowed mean.

    A SEQUENCE SHORTER THAN THE WINDOW is scored as one window over all of its
    real tokens, which makes the rule equal `mean` there. The alternative —
    refusing, or padding the window — would make short rows unscoreable or
    score them against padding, and a probe dataset's rows vary in length by
    design (the length bands in the metrics exist because of it).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    flat_scores = scores.reshape(-1, scores.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1])
    out = torch.empty(flat_scores.shape[0], dtype=scores.dtype, device=scores.device)

    for row in range(flat_scores.shape[0]):
        real = flat_scores[row][flat_valid[row]]
        n = real.numel()
        if n <= window:
            out[row] = real.mean()
            continue
        cumulative = torch.cat([torch.zeros(1, dtype=real.dtype, device=real.device), real.cumsum(0)])
        window_sums = cumulative[window:] - cumulative[:-window]
        out[row] = (window_sums / window).max()

    return out.reshape(scores.shape[:-1])


# ── the six rules, online form ────────────────────────────────────────────────

class OnlineRule:
    """Incremental combiner: `update(score)` then read `value`.

    One instance is ONE row. It exists so a serving runtime can answer "what is
    the verdict now" at every token without keeping the sequence, and so that
    ability can be tested against the batch form rather than assumed.
    """

    def __init__(self, rule: str, *, tau: float = DEFAULT_TAU, window: int = DEFAULT_WINDOW):
        if not is_streamable(rule):       # raises on an unknown rule
            raise ValueError(
                f"the {rule!r} rule cannot be computed while tokens stream: its value is "
                f"not defined until the sequence ends, so a partial answer would be a "
                f"guess that the row is about to finish, not a verdict"
            )
        if rule == "softmax" and tau <= 0:
            raise ValueError(f"softmax temperature must be > 0, got {tau}")
        if rule == "rolling_mean_max" and window < 1:
            raise ValueError(f"window must be >= 1, got {window}")

        self.rule = rule
        self.tau = tau
        self.window = window
        self.count = 0
        self._sum = 0.0
        self._max = -math.inf
        # Running softmax/attention state, max-shifted: `_shift` is the largest
        # logit seen, `_den` is sum(exp(logit - shift)), `_num` is
        # sum(exp(logit - shift) * score). Rescaling on a new maximum is what
        # keeps this equal to the batch form instead of overflowing.
        self._shift = -math.inf
        self._num = 0.0
        self._den = 0.0
        self._recent: list[float] = []
        self._window_sum = 0.0
        self._window_best = -math.inf

    def update(self, score: float, *, attention_logit: Optional[float] = None) -> float:
        """Consume one token's score and return the combined value SO FAR."""
        score = float(score)
        self.count += 1

        if self.rule == "mean":
            self._sum += score
        elif self.rule == "max":
            self._max = max(self._max, score)
        elif self.rule in ("softmax", "attention"):
            if self.rule == "attention":
                if attention_logit is None:
                    raise ValueError(
                        "the `attention` rule needs `attention_logit` (q · z_t) per token"
                    )
                logit = float(attention_logit)
            else:
                logit = score / self.tau
            if logit > self._shift:
                # Rebase onto the new maximum.
                factor = math.exp(self._shift - logit) if self._shift > -math.inf else 0.0
                self._num *= factor
                self._den *= factor
                self._shift = logit
            weight = math.exp(logit - self._shift)
            self._num += weight * score
            self._den += weight
        elif self.rule == "rolling_mean_max":
            self._recent.append(score)
            self._window_sum += score
            if len(self._recent) > self.window:
                self._window_sum -= self._recent.pop(0)
            if len(self._recent) == self.window:
                self._window_best = max(self._window_best, self._window_sum / self.window)

        return self.value

    @property
    def value(self) -> float:
        """The combined score over the tokens consumed so far."""
        if self.count == 0:
            raise ValueError("no tokens consumed yet, so there is no score")
        if self.rule == "mean":
            return self._sum / self.count
        if self.rule == "max":
            return self._max
        if self.rule in ("softmax", "attention"):
            return self._num / self._den
        if self.rule == "rolling_mean_max":
            if self._window_best > -math.inf:
                return self._window_best
            # Fewer tokens than the window: one window over everything seen, which
            # is the same rule the batch form applies to a short sequence.
            return self._window_sum / len(self._recent)
        raise AssertionError(f"unhandled streamable rule {self.rule!r}")


def rule_parameters(rule: str, *, tau: float = DEFAULT_TAU, window: int = DEFAULT_WINDOW) -> Dict[str, float]:
    """The parameters that must travel with a probe for this rule (IDL-53).

    Only the ones the rule actually uses: recording a window for `mean` would
    invite a reader to believe it mattered.
    """
    is_streamable(rule)          # raises on an unknown rule
    if rule == "softmax":
        return {"tau": float(tau)}
    if rule == "rolling_mean_max":
        return {"window": int(window)}
    # mean, max, last and attention take no SCALAR parameter. `attention`'s query
    # is a vector and travels on the head (`ProbeHead.attention_query`), not here;
    # the two dead `return {}` branches this replaced implied otherwise.
    return {}


def combine_sequence(
    rule: str,
    scores: Sequence[float],
    *,
    attention_logits: Optional[Sequence[float]] = None,
    tau: float = DEFAULT_TAU,
    window: int = DEFAULT_WINDOW,
) -> float:
    """Batch-combine a single unpadded sequence. A convenience for one row."""
    tensor = torch.as_tensor(list(scores), dtype=torch.float64)
    logits = (
        torch.as_tensor(list(attention_logits), dtype=torch.float64)
        if attention_logits is not None else None
    )
    return float(combine(rule, tensor, attention_logits=logits, tau=tau, window=window))
