"""
Fit a Jacobian lens for any loadable model.

WHAT IS BEING COMPUTED, in the source paper's own terms:

    J_l = E_t [ sum_{t' >= t} d h_target,t' / d h_l,t ]

an expectation over SOURCE positions `t` of the total downstream effect on every
subsequent target position `t'`, averaged again over a corpus. Read out as
`softmax(W_U norm(J_l h_l))`, with the logit lens being the degenerate case
`J_l = I`.

FOUR RECIPE CHOICES CHANGE THE ARTIFACT and are recorded with it (BRD A.2):
target layer (default PENULTIMATE — the last block is specialised for
next-token calibration and adds noise), attention-gradient treatment (default
FULL backward; frozen Q/K is an ablation, recommended only for intervention
artifacts), target-position scope (all subsequent), and aggregation (mean).

FREEZING IS NOT THE DEFAULT. An earlier version of this module froze attention
patterns and norm statistics unconditionally, which makes the map exactly affine
— convenient, and not what the paper computes. Its J is a local linearisation
and the departure is REPORTED rather than engineered away.

MODEL-AGNOSTIC BY CONSTRUCTION (BR-032, PADR IDL-41). Structure comes from
`discover_transformer_structure`; freezing is applied by patching the operations
themselves, not by knowing which modules an architecture happens to use. There
is deliberately no architecture name in the executable path.

THE HOOK TARGET IS `structure.layers_module[L]`, NEVER A NORM MODULE. On a
hybrid model `residual_norm_module` resolves to a post-attention RMSNorm, and a
lens fitted there is renormalised away — plausible numbers with the signal
scaled out, and no error anywhere. This project has already paid for that
confusion once in steering (PADR IDL-38).

`W_U J` IS NEVER FORMED (BR-006, PADR IDL-42). This module produces `J` alone.

CONVERGENCE IS MEASURED ON `J` ITSELF, never on a readout-quality proxy. Any
such proxy drifts toward next-token agreement, which BR-004 forbids as a quality
metric anywhere in the product — the J-lens is deliberately WORSE on that
measure than the logit lens through most of the network, so a fitter that
optimises for it is optimising for the wrong thing.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Appendix A.2. v0.1 of the BRD assumed ~10 sequences; both reference
# implementations disagree and this is the corrected figure. Fitting fewer is
# refused rather than warned about — an under-fitted lens is indistinguishable
# from a fitted one by inspection.
MIN_PROMPTS = 100

# Output dimensions differentiated in one batched backward.
#
# RETUNED FOR REVERSE MODE. Under the old forward-mode path this bounded a
# [chunk, d_model] evaluation on a LENGTH-1 sequence, and 128 was cheap. The
# batched-cotangent backward instead allocates chunk x S x d_model floats PER
# CAPTURED LAYER — at chunk=128, S=60 and 26 layers that is ~1.8 GB of gradient
# on top of a resident model and a retained graph, which is how a fit that
# passes every test OOMs on the first real card.
DEFAULT_CHUNK = 32

#: Ceiling on the transient allocation for ONE batched backward.
#:
#: A retuned constant is not a bound: sequence length varies across the corpus,
#: so a chunk that is safe on a 60-token prompt is not safe on a 400-token one.
#: The chunk is NARROWED to fit rather than the fit being refused — a smaller
#: chunk is slower and correct, whereas refusing would make long prompts
#: unfittable for a reason the caller cannot act on.
MAX_BACKWARD_BYTES = 2 * 1024 ** 3

# Convergence: relative Frobenius difference between two INDEPENDENT half-corpus
# estimates below this, sustained for PATIENCE consecutive prompts, stops the
# fit. See the split-half block in `fit()` for why it is not the running mean's
# own increment (MIS-E2E-080).
#
# THE THRESHOLD IS NOT TRANSFERABLE FROM THE OLD CRITERION. 1e-3 was calibrated
# against a quantity that shrinks as sigma/n; split-half agreement shrinks as
# sigma/sqrt(n), so the same number would demand roughly 1e6 prompts and no fit
# would ever converge. Verified by simulation, not reasoned about:
#
#   delta   noise 0.5   noise 1.0   noise 2.0
#   0.20         28         109         436
#   0.10        111         464        1933
#   0.05        464        1937      (>6000)
#   0.01     (>6000)     (>6000)     (>6000)
#
# 0.1 — "two independent halves of the corpus produce Jacobians agreeing to
# within 10% relative Frobenius norm" — lands in the 100-2000 prompt range the
# real fits already used (gemma 634, LFM2 1097). The residual dependence on
# noise is CORRECT here in a way it was not before: a noisier model genuinely
# needs more data to pin the same estimate to the same relative precision.
DEFAULT_CONVERGENCE_DELTA = 0.1

#: Stamped into the artifact so a reader can tell WHICH test a lens passed.
#: Artifacts fitted before MIS-E2E-080 carry no criterion and their "converged"
#: flag means the old running-mean-increment test — which measured per-prompt
#: variance, not stabilisation. Absent is therefore meaningful, and must not be
#: defaulted to this value on read.
CONVERGENCE_CRITERION = "split_half_agreement"
PATIENCE = 2

#: Serialises the process-wide SDPA patch — see `frozen_attention_and_norms`.
_FREEZE_LOCK = threading.Lock()

# How far the frozen sub-network may depart from its own linearisation before
# the fit is refused. Non-zero only to absorb floating-point error: freezing is
# meant to make the map exactly affine, so anything above this means a norm or
# an attention pattern escaped the freeze and the extracted matrix is not a lens.
# MAX_AFFINE_RESIDUAL was REMOVED (MIS-E2E-079).
#
# `CLAUDE.md` claimed "`affine_residual` refuses a fit whose freeze leaked", and
# an audit found the threshold stored and compared to nothing. Reinstating the
# comparison is the WRONG fix: freezing does not make the map affine — the MLP
# activation stays non-linear — so a global-affine gate reports a large
# departure for every real model and would refuse every genuine fit. That is
# why the check was replaced by `linearisation_residual`, a recorded diagnostic.
#
# A configured threshold that nothing reads is worse than no threshold: it
# reads like a guard, so nobody looks for the missing one. Removed, and the
# real gap — a freeze that silently applies to nothing — is closed by
# `frozen_attention_and_norms` below, which is a direct check that the patch
# landed rather than an inference from the resulting matrix.

#: Relative Frobenius distance from the identity below which a fitted layer is
#: DEGENERATE — the lens there is the logit lens, exactly.
#:
#: The last decoder layer has no blocks after it, so its sub-network is the
#: identity map and `J = I` by construction. That is correct, and it means the
#: Jacobian lens adds nothing at the top of the stack: a Diff there is empty
#: because the two lenses ARE the same lens, not because they happen to agree.
#: Read without knowing that, an empty top row looks like a finding.
IDENTITY_TOLERANCE = 1e-4


def identity_distance(matrix: torch.Tensor) -> float:
    """Relative Frobenius distance from the identity. 0.0 means J == I exactly."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return float("inf")
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    denom = float(torch.linalg.norm(eye))
    return float(torch.linalg.norm(matrix.float() - eye) / max(denom, 1e-12))


#: fp16's largest finite magnitude. The contract stores fp16 (Appendix A.1), and
#: a Jacobian that exceeds this saturates to inf on the cast.
FP16_MAX = 65504.0


#: Headroom below fp16's ceiling. A matrix scaled to exactly the maximum has
#: no room for the rounding the cast itself introduces.
FP16_TARGET_PEAK = FP16_MAX / 4.0


def _to_storage_dtype(matrix: torch.Tensor, layer: int):
    """Cast to the contract's fp16, RESCALING so the cast cannot saturate.

    FOUND ON THE FIRST REAL FIT, and it is not a marginal overflow: GPT-2's
    accumulated Jacobian at layer 6 peaks at 1.7e7, roughly 256x fp16's 65504
    ceiling. The naive cast saturated 0.3% of entries to `inf`, and that
    artifact is the worst kind — it deserialises cleanly, is exactly the right
    shape and exactly the right size, passes STRUCTURAL, NAMING and ENVELOPE,
    and every readout taken through it is garbage.

    A recorded per-layer SCALE fixes it without touching the contract's dtype or
    its size arithmetic: the tensor stays fp16 and the envelope bound is
    unchanged. The scale is stored in the artifact's `config.yaml`, so the
    matrix is reconstructible rather than merely smaller.

    Ranking is invariant to a positive scalar, so a readout is unaffected either
    way — but the ARTIFACT must be faithful, because a consumer that multiplies
    by W_U for anything other than ranking would get the wrong magnitudes.
    """
    if not torch.isfinite(matrix).all():
        raise ValueError(
            f"layer {layer}: the accumulated Jacobian is not finite before "
            "casting. The fit diverged; refusing to write it."
        )

    peak = float(matrix.abs().max())
    scale = 1.0
    if peak > FP16_TARGET_PEAK:
        scale = peak / FP16_TARGET_PEAK
        matrix = matrix / scale

    cast = matrix.to(torch.float16)
    if not torch.isfinite(cast).all():
        # Belt and braces: if rescaling somehow failed to make the cast safe,
        # the artifact must not be written. An inf here is undetectable later.
        raise ValueError(
            f"layer {layer}: the fp16 cast still saturated after rescaling "
            f"(peak {peak:.1f}). Refusing to write a non-finite lens."
        )
    return cast, scale


@dataclass
class FitProgress:
    prompts_seen: int
    last_delta: Optional[float]
    converged: bool


@dataclass
class FitResult:
    """A fitted lens plus everything needed to defend it (BR-007)."""

    jacobians: Dict[int, torch.Tensor]
    #: Per-layer factor the stored matrix was divided by so the fp16 cast could
    #: not saturate. 1.0 when no rescaling was needed. Recorded in config.yaml,
    #: because a scaled matrix with an unrecorded scale is simply wrong.
    scales: Dict[int, float]
    d_model: int
    n_layers: int
    prompts_seen: int
    converged: bool
    convergence_delta: float
    #: Which convergence test the flag above refers to (MIS-E2E-080).
    convergence_criterion: str = CONVERGENCE_CRITERION
    deltas: List[float] = field(default_factory=list)
    #: Per-layer mean and worst local-linearisation residual over the CORPUS.
    #: The mean says how local the lens usually is; the max says how bad it
    #: gets, and that is the number a reader should judge it on.
    #: MEAN and MAX of the per-layer SOURCE-POSITION SPREAD, over the corpus.
    #:
    #: MIS-E2E-081. These were published as `linearisation_residual_mean` and
    #: `linearisation_residual_max`, and they are not that. The value is
    #: `std across source positions / |J|.mean()` — how much the Jacobian's rows
    #: vary with WHERE in the sequence they were taken, which is a statement
    #: about positional stability. `linearisation_residual()` measures something
    #: else entirely (how well J predicts the map in a neighbourhood) and has no
    #: production caller.
    #:
    #: The artifact TRAVELS — to HuggingFace and into miLLM — so a consumer
    #: reading a field named after the affine approximation was getting a number
    #: about positional variation with no way to tell. Renamed rather than
    #: dual-published: keeping the old key would perpetuate exactly the
    #: mislabelling, and a missing key is a question a consumer can ask.
    position_spread_mean: Dict[int, float] = field(default_factory=dict)
    position_spread_max: Dict[int, float] = field(default_factory=dict)
    #: Mean prompt length the fit actually ran over.
    #:
    #: Meaningful again now that the whole sequence is used. Under the old
    #: single-position path this was 1 for every fit, which said nothing.
    mean_seq_len: float = 0.0
    #: Layers whose fitted J is the identity to within IDENTITY_TOLERANCE — the
    #: lens there IS the logit lens. Recorded rather than dropped silently, so a
    #: consumer can say WHY the two agree instead of reporting an empty Diff.
    degenerate_layers: List[int] = field(default_factory=list)

    def size_bytes(self, dtype_bytes: int = 2) -> int:
        return self.d_model * self.d_model * dtype_bytes * len(self.jacobians)


# --------------------------------------------------------------------------
# Freezing
# --------------------------------------------------------------------------


@contextmanager
def frozen_attention_and_norms(
    model: Any, freeze_qk: bool = False, freeze_norms: bool = False
) -> Iterator[None]:
    """Hold attention patterns and normalisation statistics fixed.

    Applied by patching the OPERATIONS rather than the modules, so it works on
    any architecture that reaches them — which is the point of BR-032. Two
    patches:

    * `scaled_dot_product_attention` recomputed with the attention weights
      detached, so gradient flows through V but not through Q/K. This is the
      "frozen Q/K" recipe variant (Appendix A.2 choice 2).
    * every normalisation module's scale computed from detached statistics, so
      the norm behaves as a fixed diagonal rescaling.

    BOTH DEFAULT TO FALSE, so the plain call is a NO-OP and gradients flow
    fully — the paper's standard recipe. An earlier version froze norms
    unconditionally and only made Q/K optional, which meant "full backward" was
    never actually available: the map was always forced affine.

    Each is a legitimate variant and the artifact records which was used, per
    layer — freezing Q/K is INAPPLICABLE on a layer that does not attend, which
    is not the same as unused.
    """
    patched: List[Callable[[], None]] = []

    # MIS-E2E-082: EVERYTHING below is inside the try.
    #
    # The lock acquisition and the SDPA patch used to sit ABOVE it. An
    # exception anywhere between them and the `yield` — a model whose norm
    # predicate matches nothing, a bad layer index, an OOM while walking the
    # modules — escaped without running the `finally`, so the process-wide
    # attention patch stayed installed for every subsequent forward pass in
    # that worker, and `_FREEZE_LOCK` was never released, so every later fit
    # blocked forever on a lock nobody held. Both failures are silent and
    # permanent, and the second one only ever presents as a hung worker.
    try:

        # PROCESS-WIDE MUTATION, SERIALISED. Patching
        # `torch.nn.functional.scaled_dot_product_attention` reaches every model in
        # this process, not only the one being fitted. That reach is the point — it
        # is what makes the freeze architecture-agnostic — but it means two
        # concurrent fits would nest their patches and restore each other's
        # originals in the wrong order, leaving attention permanently frozen for
        # everything afterwards, with no error and no way to notice.
        #
        # The task queue serialises fits today, so this guards the invariant rather
        # than a symptom already seen. It is worth closing anyway: the failure is
        # silent, permanent, and would present as "readouts went strange".
        # ONLY WHEN SOMETHING IS ACTUALLY PATCHED. With both flags off this context
        # touches no global state, and taking the lock anyway would serialise fits
        # that cannot interfere with each other.
        if freeze_qk or freeze_norms:
            _FREEZE_LOCK.acquire()
            patched.append(_FREEZE_LOCK.release)

        if freeze_qk:
            original_sdpa = torch.nn.functional.scaled_dot_product_attention

            def frozen_sdpa(query, key, value, *args, **kwargs):
                # Recover the pattern with Q/K detached, then apply it to V. The
                # pattern is a constant; V still carries gradient.
                with torch.no_grad():
                    weights = original_sdpa(
                        query,
                        key,
                        torch.eye(
                            value.shape[-2], device=value.device, dtype=value.dtype
                        ).expand(*value.shape[:-1], value.shape[-2]),
                        *args,
                        **kwargs,
                    )
                # GROUPED-QUERY ATTENTION. Under GQA there are fewer KV heads than
                # query heads, and callers may hand SDPA the un-repeated K/V with
                # `enable_gqa=True` and let it broadcast internally. The recovered
                # pattern then has one row group per QUERY head while `value` still
                # has only the KV heads, and the matmul is a shape error — which is
                # the good case. The head counts are read off the tensors rather
                # than off a config, so this covers MHA (n_rep == 1, a no-op), GQA
                # and MQA without naming an architecture (BR-032).
                n_rep = weights.shape[-3] // value.shape[-3]
                if n_rep > 1:
                    # repeat_interleave, not repeat: transformers' repeat_kv expands
                    # then reshapes, which places each KV head next to its own query
                    # group. `repeat` would tile the whole block and silently pair
                    # every query head with the WRONG KV head — same shape, wrong
                    # attention output, no error anywhere.
                    value = value.repeat_interleave(n_rep, dim=-3)
                return weights @ value

            torch.nn.functional.scaled_dot_product_attention = frozen_sdpa
            patched.append(
                lambda: setattr(
                    torch.nn.functional, "scaled_dot_product_attention", original_sdpa
                )
            )

        handles = [_freeze_norm(m) for m in _norm_modules(model)] if freeze_norms else []

        # THE FREEZE MUST HAVE ACTUALLY APPLIED (MIS-E2E-079).
        #
        # This is the gate `CLAUDE.md` promised under the name `affine_residual`,
        # implemented soundly. The affine version could not work — freezing does not
        # make the map affine, so it would refuse every genuine fit — but the hazard
        # it was aimed at is real: an incomplete freeze yields a matrix of the right
        # shape that passes STRUCTURAL, NAMING and ENVELOPE validation and reads out
        # plausible nonsense, and the artifact then records `freeze_qk: true` as
        # though it held.
        #
        # Checking that the patch LANDED is direct, cheap and certain, where
        # inferring it from the matrix is neither. `_norm_modules` is the specific
        # reason this matters: it once used a substring match, and a model whose
        # norm modules do not match the predicate yields an EMPTY list here — a
        # `freeze_norms=True` fit that froze nothing at all, silently.
        # No manual undo before these raises: they are inside the `try` now
        # (MIS-E2E-082), so the `finally` unwinds `patched`. Running both would
        # release `_FREEZE_LOCK` twice — `RuntimeError: release unlocked lock`,
        # which is what the existing tests caught the moment the try moved up.
        if freeze_qk and torch.nn.functional.scaled_dot_product_attention is not frozen_sdpa:
            raise RuntimeError(
                "freeze_qk was requested but the SDPA patch is not in place — "
                "something replaced torch.nn.functional.scaled_dot_product_attention "
                "after it was patched. Refusing to fit: the lens would record "
                "freeze_qk=true for a fit that was not frozen."
            )
        if freeze_norms and not handles:
            raise RuntimeError(
                "freeze_norms was requested but no norm module was found on this "
                "model, so nothing was frozen. Refusing to fit rather than "
                "recording freeze_norms=true for an unfrozen fit. Check that "
                "_norm_modules' predicate matches this architecture's norm layers."
            )

        yield
    finally:
        # Reverse order: the lock was acquired first and must be released
        # last, after every patch it protects has been undone.
        for undo in reversed(patched):
            undo()
        for handle in handles:
            handle()


def _norm_modules(model: Any) -> List[Any]:
    """Every normalisation module, by class name.

    A name search over the module tree, not an architecture branch: `RMSNorm`,
    `LayerNorm`, `GemmaRMSNorm` and their per-family spellings all END with
    "norm".

    ENDSWITH, not CONTAINS. A substring match also captures anything merely
    NAMED for a norm — a `NormedBlock`, a `NormalizedAttention` — and freezing a
    decoder block is not a no-op: it would replace the whole block with an
    elementwise rescaling and produce a lens with no error anywhere.
    """
    if not hasattr(model, "modules"):
        return []
    return [m for m in model.modules() if type(m).__name__.lower().endswith("norm")]


def _freeze_norm(module: Any) -> Callable[[], None]:
    """Make one norm module use detached statistics; returns the undo."""
    original_forward = module.forward

    def frozen_forward(hidden_states, *args, **kwargs):
        # Run the real norm on a detached input to obtain the scale it WOULD
        # apply, then apply that scale to the live input. The statistics become
        # constants; the linear path stays differentiable.
        with torch.no_grad():
            reference = original_forward(hidden_states.detach(), *args, **kwargs)
        # Second guard, independent of the name rule above: anything that does
        # not return a single tensor is not a norm, whatever it is called. Left
        # untouched rather than coerced — a wrong guess here is silent.
        if not isinstance(reference, torch.Tensor) or not isinstance(
            hidden_states, torch.Tensor
        ):
            return original_forward(hidden_states, *args, **kwargs)
        denom = hidden_states.detach()
        scale = torch.where(
            denom.abs() > 1e-9,
            reference / torch.where(denom.abs() > 1e-9, denom, torch.ones_like(denom)),
            torch.ones_like(denom),
        )
        return hidden_states * scale

    module.forward = frozen_forward
    return lambda: setattr(module, "forward", original_forward)


# --------------------------------------------------------------------------
# Jacobian extraction
# --------------------------------------------------------------------------


def jacobian_by_jvp(
    fn: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
) -> torch.Tensor:
    """`d fn / d point` by one forward-mode pass per input dimension.

    THE REFERENCE IMPLEMENTATION, not the production path. It makes no
    assumption about `fn` at all, which is exactly what makes it the right thing
    to check the fast path against — and exactly what makes it unusable at
    scale: d_model jvp calls per layer per prompt is millions of forward passes
    on a real model.

    NEITHER OF THESE RUNS IN PRODUCTION ANY MORE. The fitter moved to reverse
    mode — `JacobianFitter._fit_one` — because the paper's quantity is an
    expectation over source positions of the effect on all subsequent target
    positions, and a forward-mode sub-network at one position cannot express
    it. This docstring previously said "`jacobian_batched` is what actually
    runs", which stopped being true and is exactly the kind of stale claim that
    sends a reader to the wrong file.

    They are kept as a REFERENCE IMPLEMENTATION: `test_jlens_fitter` uses them
    to check the batched extraction against a JVP computed a different way, so
    the linear-algebra core stays independently verified.
    """
    d_in = point.numel()
    columns: List[torch.Tensor] = []
    for i in range(d_in):
        tangent_in = torch.zeros_like(point).reshape(-1)
        tangent_in[i] = 1.0
        _, tangent = torch.autograd.functional.jvp(
            fn, point, tangent_in.view_as(point), create_graph=False
        )
        columns.append(tangent.reshape(-1))
    return torch.stack(columns, dim=1)


def jacobian_batched(
    fn: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    chunk: int = DEFAULT_CHUNK,
) -> torch.Tensor:
    """`d fn / d point` AT THE POINT, by vectorised automatic differentiation.

    CORRECTED AFTER A HARDWARE RUN. The first version computed
    `J[:, i] = fn(e_i) - fn(0)` — secants — on the premise that freezing
    attention and normalisation makes `fn` AFFINE. It does not. Freezing removes
    the strongly input-dependent parts, but the MLP's activation (GELU on the
    model this was first run against) stays non-linear, so the residual-to-
    residual map is not affine and never was.

    A secant is not a Jacobian for a non-affine map. It answers "where does a
    unit step land", which for a curved map is a different question and gives a
    matrix that is plausible, well-shaped, and wrong.

    On the first real fit `affine_residual` measured a departure of 40.3 against
    a 1e-3 limit — the guard from review round 1 catching a premise error rather
    than a coding one, which is the only reason this was found before an
    artifact shipped.

    `vectorize=True` batches the backward passes, so this keeps the speed the
    secant version was reaching for without buying it with the wrong math.
    """
    jac = torch.autograd.functional.jacobian(
        fn, point, vectorize=True, create_graph=False
    )
    return jac.reshape(-1, point.numel())


def linearisation_residual(
    fn: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    jacobian: torch.Tensor,
    step: float = 1e-2,
) -> float:
    """How well `J` predicts `fn` in a NEIGHBOURHOOD of the fitting point.

    A DIAGNOSTIC, recorded with the fit — not a gate. This is the corrected form
    of what was `affine_residual`, and the correction matters:

    The old version compared `J h + fn(0)` against `fn(h)` — a GLOBAL affine
    prediction — on the premise that freezing makes `fn` affine. It does not
    (the MLP activation stays non-linear), so it reported a large departure for
    every real model and would have refused every genuine fit.

    A Jacobian IS a local linearisation; asking it to hold globally is asking
    the wrong question. What is worth recording is how far the linearisation
    holds LOCALLY, which is what makes a lens more or less trustworthy away from
    the exact point it was taken at. Large is informative, not disqualifying.
    """
    with torch.no_grad():
        direction = torch.randn_like(point)
        direction = direction / torch.linalg.norm(direction) * step * float(
            torch.linalg.norm(point)
        )
        predicted = fn(point).reshape(-1) + jacobian @ direction.reshape(-1)
        actual = fn(point + direction).reshape(-1)
        denom = float(torch.linalg.norm(actual.to(torch.float32)))
        if denom == 0.0:
            return 0.0
        return float(torch.linalg.norm((predicted - actual).to(torch.float32)) / denom)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def merge_shards(shards: Sequence[Dict[int, torch.Tensor]], weights: Sequence[int]) -> Dict[int, torch.Tensor]:
    """Combine per-shard means into one weighted mean.

    Fitting parallelises by splitting the CORPUS and merging, never by splitting
    the model (BRD v0.3 assumptions). Weighting by prompt count is what makes
    the merge equal to a single run over the concatenated corpus; an unweighted
    mean silently over-weights a short shard.
    """
    if not shards:
        return {}
    if len(shards) != len(weights):
        raise ValueError(f"{len(shards)} shards but {len(weights)} weights")
    total = sum(weights)
    if total <= 0:
        raise ValueError("shard weights sum to zero")

    layers = set(shards[0])
    for s in shards[1:]:
        if set(s) != layers:
            raise ValueError("shards cover different layer sets")

    merged: Dict[int, torch.Tensor] = {}
    for layer in sorted(layers):
        acc = torch.zeros_like(shards[0][layer], dtype=torch.float32)
        for shard, weight in zip(shards, weights):
            acc += shard[layer].to(torch.float32) * weight
        merged[layer] = (acc / total).to(shards[0][layer].dtype)
    return merged


def relative_change(previous: Dict[int, torch.Tensor], current: Dict[int, torch.Tensor]) -> float:
    """Relative Frobenius change across all layers.

    THE CONVERGENCE SIGNAL, and deliberately a property of `J` alone. A
    readout-quality proxy would drift toward next-token agreement, which BR-004
    forbids as a quality metric — the J-lens is meant to be worse on it.
    """
    num = 0.0
    den = 0.0
    for layer, cur in current.items():
        prev = previous.get(layer)
        if prev is None:
            return float("inf")
        num += float(torch.linalg.norm((cur - prev).to(torch.float32)) ** 2)
        den += float(torch.linalg.norm(cur.to(torch.float32)) ** 2)
    if den == 0.0:
        return 0.0
    return (num / den) ** 0.5


def _batch_kwargs(kwargs: Dict[str, Any], batch: int) -> Dict[str, Any]:
    """Reshape recorded layer kwargs to the extraction batch size.

    The reference forward ran with batch 1 and the real sequence length; the
    extraction runs with batch `n` and ONE position. Tensors whose leading
    dimension is the batch are expanded and their sequence dimension truncated
    to the final position — the position the lens is taken at.

    Anything not recognisably batch-shaped is passed through untouched rather
    than reshaped on a guess: a wrong reshape here produces a running model and
    a wrong lens.
    """
    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[0] == 1:
            sliced = value[:, -1:] if value.shape[1] > 1 else value
            out[key] = sliced.expand(batch, *sliced.shape[1:])
        elif isinstance(value, tuple) and value and all(
            isinstance(v, torch.Tensor) for v in value
        ):
            # Rotary embeddings arrive as a (cos, sin) tuple.
            out[key] = tuple(_batch_kwargs({"v": v}, batch)["v"] for v in value)
        else:
            out[key] = value
    return out


def _expand_kwargs(kwargs: Dict[str, Any], batch: int) -> Dict[str, Any]:
    """Expand recorded kwargs to `batch` WITHOUT touching the sequence axis.

    The counterpart to `_batch_kwargs`, which truncates to one position. Here
    the whole sequence is replayed, so masks, position ids and rotary tables
    must arrive at their ORIGINAL length — truncating them is exactly what
    makes the cheap path unfaithful.
    """
    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[0] == 1:
            out[key] = value.expand(batch, *value.shape[1:])
        elif isinstance(value, tuple) and value and all(
            isinstance(v, torch.Tensor) for v in value
        ):
            out[key] = tuple(_expand_kwargs({"v": v}, batch)["v"] for v in value)
        else:
            out[key] = value
    return out


class JacobianFitter:
    """Fits `J` per layer over a corpus, with convergence-based stopping."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        structure: Any,
        *,
        freeze_qk: bool = False,
        freeze_norms: bool = False,
        target_layer: str = "penultimate",
        convergence_delta: float = DEFAULT_CONVERGENCE_DELTA,
        min_prompts: int = MIN_PROMPTS,
        chunk: int = DEFAULT_CHUNK,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.structure = structure
        #: FULL BACKWARD BY DEFAULT, matching the paper. Freezing attention
        #: patterns is an ABLATION there, not the standard recipe — it slightly
        #: reduces readout quality while tending to produce directions that respond
        #: more strongly to intervention — an association the source paper
        #: reports, never a validated property of any artifact fitted here. It
        #: stays selectable and is suggested when the lens is built for
        #: intervention work rather than reading (BRD A.2 choice 2).
        self.freeze_qk = freeze_qk
        #: Norm freezing is likewise opt-in. Freezing makes the map exactly
        #: affine, which is convenient but is not what the paper computes: its
        #: J is a local linearisation and the residual is reported rather than
        #: engineered away.
        self.freeze_norms = freeze_norms
        #: Which block's output the Jacobian is taken TO (BRD A.2 choice 1).
        if target_layer not in ("final", "penultimate"):
            raise ValueError(
                f"target_layer must be 'final' or 'penultimate', got {target_layer!r}"
            )
        self.target_layer = target_layer
        self.convergence_delta = convergence_delta
        self.min_prompts = min_prompts
        self.chunk = chunk
        #: Per-layer local-linearisation residual, ACCUMULATED over the corpus.
        #:
        #: This used to hold the most recent prompt's value only — overwritten
        #: on every prompt — while being written into the artifact as though it
        #: described the fit. A hundred-prompt corpus reported the hundredth
        #: prompt's number. Mean and max are both kept: the mean says how local
        #: the lens is typically, the max says how bad it gets, and a lens is
        #: trusted or not on the second.
        self._last_residuals: Dict[int, float] = {}
        self._residual_sums: Dict[int, float] = {}
        self._residual_max: Dict[int, float] = {}
        self._residual_counts: Dict[int, int] = {}
        self._seq_lens: List[int] = []

    def fit(
        self,
        prompts: Sequence[str],
        layers: Optional[Sequence[int]] = None,
        on_progress: Optional[Callable[[FitProgress], None]] = None,
    ) -> FitResult:
        """Accumulate a mean `J` per layer until it stops moving.

        Refuses a corpus below the floor rather than warning: an under-fitted
        lens is indistinguishable from a fitted one by inspection, and the whole
        point of the validation suite is that structure can be perfect while
        content is absent.
        """
        if len(prompts) < self.min_prompts:
            raise ValueError(
                f"{len(prompts)} prompts is below the floor of {self.min_prompts} "
                "(Appendix A.2). An under-fitted lens looks exactly like a "
                "fitted one; fitting fewer is refused rather than warned about."
            )

        target_index = self._target_layer_index()
        # "EVERY LAYER" MEANS EVERY LAYER THAT CAN BE FITTED. Layers above the
        # target have a zero Jacobian to it, so including them in the default
        # would make the ordinary `layers=None` call raise.
        selected = (
            list(layers) if layers is not None else list(range(target_index + 1))
        )

        # NO LAYER PAST THE TARGET. The Jacobian runs from a source layer to the
        # TARGET block's output, so a source ABOVE the target is downstream of
        # it and its gradient is identically ZERO — the model cannot route
        # information backwards. A zero J is not a degenerate-but-honest lens
        # like the identity: every readout through it is `softmax(W_U norm(0))`,
        # a uniform distribution rendered as confident-looking tokens.
        #
        # Refused rather than dropped, because silently narrowing the layer set
        # produces an artifact with less coverage than the caller asked for and
        # nothing saying so.
        beyond = [l for l in selected if l > target_index]
        if beyond:
            raise ValueError(
                f"layers {beyond} are above the {self.target_layer} target "
                f"(block {target_index}); their Jacobian to it is zero by "
                "causality, and a zero lens reads out as uniform noise wearing "
                "a confident face. Fit up to the target, or set "
                "target_layer='final'."
            )
        # SPLIT-HALF AGREEMENT, NOT THE RUNNING MEAN'S OWN STEP SIZE.
        #
        # MIS-E2E-080. This measured `relative_change(previous, accumulated)` —
        # the increment of a running mean, which is O(sigma/n). It shrinks
        # because the DENOMINATOR GROWS, not because successive estimates of J
        # agree. The stop point was therefore n ~ sigma/delta: directly
        # proportional to per-prompt variance, and reachable by any process with
        # bounded increments, converged or not.
        #
        # Simulated by the reviewer at noise 0.5 / 1.0 / 2.0 → stop points
        # 518 / 1050 / 2030, exactly proportional — and BRACKETING the two real
        # recorded fits (gemma 634, LFM2 1097) that the docs call "paper-aligned
        # converged lenses". Those numbers are fully consistent with the
        # criterion having measured nothing but each model's per-prompt spread.
        #
        # Two INDEPENDENT accumulators over alternating prompts, compared
        # against each other, is a real stabilisation test: it asks "would a
        # different half of this corpus have produced the same lens?", which is
        # the question the word "converged" is doing evidential work for. A
        # low-variance but biased estimate no longer earns the word, and a
        # noisier model no longer merely has to run proportionally longer.
        #
        # Costs a second accumulator per layer — still O(1) in corpus size,
        # which is the property the running mean was chosen for.
        half_a: Dict[int, torch.Tensor] = {}
        half_b: Dict[int, torch.Tensor] = {}
        count_a = 0
        count_b = 0
        accumulated: Dict[int, torch.Tensor] = {}
        deltas: List[float] = []
        stable = 0
        seen = 0

        with frozen_attention_and_norms(
            self.model, freeze_qk=self.freeze_qk, freeze_norms=self.freeze_norms
        ):
            for prompt in prompts:
                per_prompt = self._fit_one(prompt, selected)
                seen += 1

                # Alternate, so the halves are interleaved rather than
                # sequential: a corpus ordered by topic would otherwise put all
                # of one subject in half A and guarantee disagreement.
                if seen % 2:
                    count_a += 1
                    target, n = half_a, count_a
                else:
                    count_b += 1
                    target, n = half_b, count_b

                for layer, mat in per_prompt.items():
                    if layer in target:
                        target[layer] += (mat - target[layer]) / n
                    else:
                        target[layer] = mat.clone()

                if seen >= self.min_prompts and count_a and count_b:
                    # The two halves must AGREE. Not "the last prompt moved the
                    # mean by little" — that is guaranteed for large n.
                    delta = relative_change(half_a, half_b)
                    deltas.append(delta)
                    stable = stable + 1 if delta < self.convergence_delta else 0
                    if on_progress:
                        on_progress(FitProgress(seen, delta, stable >= PATIENCE))
                    if stable >= PATIENCE:
                        break
                elif on_progress:
                    on_progress(FitProgress(seen, None, False))

        # The published lens is the WHOLE corpus, not either half — the split is
        # the convergence instrument, not the estimate.
        for layer in set(half_a) | set(half_b):
            a = half_a.get(layer)
            b = half_b.get(layer)
            if a is None:
                accumulated[layer] = b.clone()
            elif b is None:
                accumulated[layer] = a.clone()
            else:
                total = count_a + count_b
                accumulated[layer] = (a * count_a + b * count_b) / total

        cast_and_scale = {
            k: _to_storage_dtype(v, k) for k, v in accumulated.items()
        }
        return FitResult(
            jacobians={k: cast_and_scale[k][0] for k in cast_and_scale},
            scales={k: cast_and_scale[k][1] for k in cast_and_scale},
            d_model=int(next(iter(accumulated.values())).shape[0]) if accumulated else 0,
            n_layers=len(accumulated),
            prompts_seen=seen,
            converged=stable >= PATIENCE,
            convergence_delta=self.convergence_delta,
            deltas=deltas,
            position_spread_mean={
                l: self._residual_sums[l] / max(self._residual_counts[l], 1)
                for l in self._residual_sums
            },
            position_spread_max=dict(self._residual_max),
            mean_seq_len=(
                sum(self._seq_lens) / len(self._seq_lens) if self._seq_lens else 0.0
            ),
            degenerate_layers=sorted(
                l for l, m in accumulated.items()
                if identity_distance(m) <= IDENTITY_TOLERANCE
            ),
        )

    @property
    def device(self) -> torch.device:
        """The device the MODEL is on, taken from the model itself.

        Not a constructor argument and not inherited from the ambient default:
        a fitter told one device while the model sits on another produces
        `Expected all tensors to be on the same device` at the embedding, and
        only when a GPU is actually present. Every CPU test passes, because
        there the two agree by accident.
        """
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def _fit_one(self, prompt: str, layers: Sequence[int]) -> Dict[int, torch.Tensor]:
        """One prompt's contribution to `J`, for every requested layer at once.

        REVERSE MODE, MATCHING THE SOURCE PAPER. The quantity is

            J_l = E_t [ sum_{t' >= t} d h_target,t' / d h_l,t ]

        — an expectation over SOURCE positions `t`, of the total downstream
        effect on every subsequent target position `t'`. Three properties of
        that definition drive this implementation, and the previous forward-mode
        one satisfied none of them:

        * ALL SOURCE POSITIONS. A single backward pass yields the gradient at
          every position simultaneously, so one sweep per output dimension fills
          the whole (layer, source-position) grid. The old path took one source
          position per prompt — for a 60-token prompt that is 60x fewer samples,
          which is most of why the fit plateaued rather than converging.
        * ALL SUBSEQUENT TARGET POSITIONS. The cotangent is 1 at every target
          position, so the gradient arriving at source `t` is already the sum
          over `t' >= t`. Causality does the masking: a decoder cannot route
          information backwards, so terms with `t' < t` are identically zero and
          need no explicit mask.
        * EVERY LAYER FROM ONE SWEEP. Gradients w.r.t. all captured layers come
          out of the same backward, so cost is O(d_model) sweeps per prompt
          rather than O(d_model x n_layers) forward passes.
        """
        import torch as _torch

        encoded = self.tokenizer(prompt, return_tensors="pt")
        # MOVED TO THE MODEL'S DEVICE. The tokenizer always returns CPU tensors;
        # a model on CUDA then fails inside index_select at the embedding, and
        # that is invisible on a CPU-only test stack.
        input_ids = encoded["input_ids"].to(self.device)

        captured: Dict[int, _torch.Tensor] = {}
        target_holder: Dict[str, _torch.Tensor] = {}

        def capture(idx: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[idx] = hidden
            return hook

        def capture_target(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            target_holder["h"] = hidden

        target_index = self._target_layer_index()
        handles = [
            self._capture_module(l).register_forward_hook(capture(l)) for l in layers
        ]
        handles.append(
            self.structure.layers_module[target_index].register_forward_hook(
                capture_target
            )
        )
        try:
            # NO torch.no_grad HERE. The graph is the point.
            self.model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        target = target_holder["h"]           # [1, S, d_model]
        self._seq_lens.append(int(target.shape[1]))
        d_model = int(target.shape[-1])
        inputs = [captured[l] for l in layers if l in captured]
        present = [l for l in layers if l in captured]
        missing = [l for l in layers if l not in captured]
        if missing:
            # SILENT COVERAGE LOSS OTHERWISE. The layer would simply not appear
            # in the result, never accumulate, and the artifact would ship with
            # fewer layers than requested and nothing saying which.
            raise ValueError(
                f"layers {missing} never produced an activation during the "
                "forward pass, so nothing was captured for them."
            )
        if not inputs:
            return {}

        sums: Dict[int, _torch.Tensor] = {
            l: _torch.zeros(d_model, d_model, dtype=_torch.float32) for l in present
        }
        # HOW MUCH J VARIES BY SOURCE POSITION. `J` is a mean over positions, so
        # the honest companion number is the spread it was averaged from — not a
        # "linearisation residual", which is what this used to report and which
        # no longer describes anything: it compared the MEAN J against a SINGLE
        # position's activation and the SUMMED target, three different objects.
        #
        # For intervention work this is the number that matters: a lens with
        # large position spread transports differently depending on where in the
        # sequence you apply it.
        spread: Dict[int, _torch.Tensor] = {
            l: _torch.zeros(d_model, dtype=_torch.float32) for l in present
        }

        # NARROWED TO FIT, per prompt. Sequence length varies across the corpus,
        # so a chunk that is safe on a short prompt can OOM on a long one — the
        # bound has to be recomputed rather than chosen once at construction.
        per_dim = int(target.shape[1]) * d_model * 4 * max(len(inputs), 1)
        chunk = max(1, min(self.chunk, MAX_BACKWARD_BYTES // max(per_dim, 1)))
        if chunk < self.chunk:
            logger.debug(
                "narrowing backward chunk %d -> %d to stay under %d bytes",
                self.chunk, chunk, MAX_BACKWARD_BYTES,
            )

        for start in range(0, d_model, chunk):
            stop = min(start + chunk, d_model)
            width = stop - start
            # BATCHED COTANGENTS. cot[c] selects output dimension (start + c) at
            # EVERY target position, so one backward per output dimension
            # returns that row of J summed over t' — for all source positions
            # and all layers at once.
            # fp32 COTANGENT even on an fp16 model. The gradient is averaged
            # over hundreds of prompts; accumulating it in half precision
            # throws away resolution the averaging is trying to recover.
            cot = _torch.zeros(
                (width, *target.shape),
                dtype=torch.float32 if target.dtype == torch.float16 else target.dtype,
                device=target.device,
            )
            for c in range(width):
                cot[c, ..., start + c] = 1.0

            grads = _torch.autograd.grad(
                outputs=target,
                inputs=inputs,
                grad_outputs=cot,
                retain_graph=True,
                allow_unused=True,
                is_grads_batched=True,
            )
            for layer, g in zip(present, grads):
                if g is None:
                    # REFUSED, not skipped. `sums[layer]` starts at zeros, so
                    # skipping would leave that block of rows at zero and
                    # accumulate it as though it had been measured — a lens
                    # with a dead band that reads out as confident uniform
                    # noise, with nothing anywhere saying so.
                    raise ValueError(
                        f"layer {layer} received no gradient from the target "
                        "block. It is not on the path to the target, so its "
                        "Jacobian is undefined rather than zero."
                    )
                # g: [width, 1, S, d_model]. Rows are the mean over SOURCE
                # positions; the spread across them is kept separately.
                per_position = g[:, 0].to(_torch.float32)      # [width, S, d_model]
                sums[layer][start:stop, :] = per_position.mean(dim=1).detach()
                spread[layer][start:stop] = (
                    per_position.std(dim=1).mean(dim=-1).detach()
                    if per_position.shape[1] > 1
                    else _torch.zeros(per_position.shape[0])
                )

        out: Dict[int, _torch.Tensor] = {}
        for layer in present:
            jac = sums[layer]
            out[layer] = jac
            # RECORDED, not gated. A Jacobian is a local linearisation by
            # definition, so a non-zero residual is expected on any real model.
            # Measured against the captured point at the LAST source position,
            # which is the one the readout is most often taken at.
            scale = float(jac.abs().mean()) or 1.0
            rel_spread = float(spread[layer].mean()) / scale
            self._last_residuals[layer] = rel_spread
            self._residual_sums[layer] = self._residual_sums.get(layer, 0.0) + rel_spread
            self._residual_max[layer] = max(self._residual_max.get(layer, 0.0), rel_spread)
            self._residual_counts[layer] = self._residual_counts.get(layer, 0) + 1
        return out

    def _capture_module(self, layer: int):
        """The module whose output IS the residual stream at `layer`.

        A SEAM, on purpose. This is the decoder block itself — never a norm
        module — and it is the most expensive lesson in this repo: hooking
        `residual_norm_module` puts the norm's rescaling outside the fitted map
        and yields plausible numbers with the signal scaled away, no error
        anywhere (PADR IDL-38). `test_jlens_fitter` overrides this method to
        reproduce the wrong hook and assert the resulting lens DIFFERS, so the
        mistake cannot ship undetected.
        """
        return self.structure.layers_module[layer]

    def _target_layer_index(self) -> int:
        """Which block's output the Jacobian is taken TO.

        PENULTIMATE BY DEFAULT, per the BRD: including the last block increases
        noisy artifacts in readouts, plausibly because that block is specialised
        for calibrating next-token probabilities and carries less semantic
        content. Targeting `final` also makes the last layer's J the identity by
        construction, which is correct but adds a degenerate row to every fit.
        """
        n = int(self.structure.num_layers)
        if self.target_layer == "final":
            return n - 1
        return max(0, n - 2)

    # `_sub_network` REMOVED. It built a forward-mode sub-network at a single
    # source position and, in its cheap variant, on a length-1 sequence — which
    # gave the perturbed position full attention weight instead of its real
    # share. Both are superseded by the reverse-mode path in `_fit_one`, and
    # keeping it would leave two ways to compute J that answer different
    # questions while looking interchangeable.

