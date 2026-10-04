"""Does the SAE reconstruction still let the model do its job?

WHY THIS EXISTS. Every number reported about an SAE while it trains lives in the
SAE's own space — FVU, L0, dead count. None of them says what the reconstruction
costs the MODEL, and reconstruction error is not uniformly important: an SAE can
reach a low FVU and still wreck the next-token distribution, because the
directions that carry the output are a small share of the variance.

The standard answer is to splice the reconstruction back into the residual
stream and measure next-token cross-entropy. Per layer this module reports:

    base          — CE with the model untouched
    spliced       — CE with the SAE's reconstruction substituted at that layer
    mean-ablated  — CE with the layer output replaced by its MEAN activation
    zero-ablated  — CE with the layer output replaced by zeros (reference only)

and summarises with **loss recovered** against each ablation:

    (ablated - spliced) / (ablated - base)

MEAN ABLATION IS THE HEADLINE (remediation item 6, 2026-09-15). The previous
baseline substituted the SAE's ``b_dec`` as a RAW activation. Under
``constant_norm_rescale`` — the default — ``b_dec`` lives in the NORMALISED
space, where every token has norm sqrt(d), so it was a vector at the wrong scale
and the "ablation" measured an arbitrary perturbation. The mean activation over
the evaluation's own real tokens is in the space the model reads.

ZERO ABLATION IS A FLOOR, NOT A BASELINE. Zeroing any of layers 11-13 of
LFM2.5-1.2B gives CE exactly ln(65,536) = 11.0904: the model emits a uniform
distribution. Loss recovered against that floor is close to 1 for almost any
reconstruction, which is why it is reported for reference and never as the
headline.

Also reported: KL(base || spliced), the mean number of active latents per token,
and both FVUs over the real tokens (``ml/sae_metrics``), all accumulated by sums
so batching never changes the answer.

TWO PASSES. The mean must be known before a batch can be mean-ablated, and base
log-probabilities for a whole evaluation cannot be held in memory (65,536 x 2,048
x 4 bytes is 0.5 GB per block). Pass 1 reads each hooked layer's output once to
accumulate its mean and the SAE's reconstruction statistics; pass 2 measures CE.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..ml.model_devices import module_device
from ..ml.sae_metrics import FvuSums

logger = logging.getLogger(__name__)

#: Recorded beside every zero-ablated number, so no reader takes it as a baseline.
ZERO_ABLATION_NOTE = (
    "Zero ablation is a floor, not a baseline: on these models replacing a residual "
    "layer output with zeros makes the model emit a near-uniform distribution "
    "(CE = ln(vocab)), so loss recovered against it is near 1 for almost any SAE. "
    "Read loss_recovered_vs_mean."
)

#: A batch is a pair of (input_ids, attention_mask) tensors on the model's input device.
Batch = Tuple[torch.Tensor, Optional[torch.Tensor]]


def _sae_param(sae: torch.nn.Module) -> Optional[torch.Tensor]:
    for tensor in sae.parameters():
        return tensor
    return None


def _call_sae(sae: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(x_hat, z)`` from any SAE here, without computing its training losses."""
    try:
        accepts_return_loss = "return_loss" in inspect.signature(sae.forward).parameters
    except (TypeError, ValueError):
        accepts_return_loss = False
    out = sae(x, return_loss=False) if accepts_return_loss else sae(x)
    if isinstance(out, tuple):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


def encode_decode_at_layer(
    sae: torch.nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The SAE's reconstruction of a layer output — back on the layer's device and dtype — and its latents.

    ON A SPLIT MODEL THE LAYER AND THE SAE ARE ON DIFFERENT CARDS. The SAE was
    trained on the job's first card; the layer this hook sits on may have been
    placed on another. The activation goes to the SAE and the reconstruction
    comes back, because the next layer reads it on this layer's card.

    AND IN DIFFERENT DTYPES. The base model runs in fp16 or bf16 and the SAE's
    weights are float32; an fp16 activation into a float32 ``F.linear`` raises a
    dtype mismatch, which a caller logging "evaluation failed" turns into no
    evaluation at all. The input takes the SAE's dtype; the output takes the layer's.
    """
    param = _sae_param(sae)
    if param is None:
        x_in = x
    else:
        sae_device = module_device(sae)
        x_in = x.to(device=sae_device if sae_device is not None else x.device, dtype=param.dtype)
    x_hat, z = _call_sae(sae, x_in)
    return x_hat.to(device=x.device, dtype=x.dtype), z


def reconstruct_at_layer(sae: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The SAE's reconstruction of a layer output, on the layer's device and dtype."""
    return encode_decode_at_layer(sae, x)[0]


def mean_ablation_at_layer(mean: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The layer's mean activation in place of its output, on the layer's device and dtype.

    ``mean`` is a RAW-space activation — measured at this layer's output over the
    evaluation's real tokens — never an SAE parameter, whose space depends on how
    the SAE normalised its inputs.
    """
    return mean.to(device=x.device, dtype=x.dtype).expand_as(x)


def zero_ablation_at_layer(x: torch.Tensor) -> torch.Tensor:
    """Zeros in place of the layer output: the uniform-output floor. See ZERO_ABLATION_NOTE."""
    return torch.zeros_like(x)


def _output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


@contextlib.contextmanager
def _splice(module: torch.nn.Module, transform: Callable[[torch.Tensor], torch.Tensor]):
    """Replace a module's output for the duration of the block.

    THE HOOK MUST COME OFF. A forward hook left attached silently changes every
    subsequent forward pass — including any training that follows in the same
    process — and the symptom would be a slow, unexplained quality drift rather
    than an error. ``finally`` is what guarantees it, including when the forward raises.
    """
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (transform(output[0]),) + tuple(output[1:])
        return transform(output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def _splice_many(splices: Sequence[Tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]]):
    with contextlib.ExitStack() as stack:
        for module, transform in splices:
            stack.enter_context(_splice(module, transform))
        yield


def loss_recovered(baseline: float, spliced: float, ablated: float) -> float:
    """Fraction of the ablation's cost the reconstruction avoids.

    Returns NaN when the denominator collapses — if ablating the layer costs
    nothing, "how much of that cost did we avoid" has no answer, and returning
    1.0 there would report a perfect score for a layer that does nothing.

    NOT CLAMPED. On a layer whose ablation HELPS (an untrained model, a redundant
    layer) the denominator is negative and the score leaves [0, 1]; that is a real
    finding about the layer, and clamping would hide it behind a plausible number.
    """
    denom = ablated - baseline
    if abs(denom) < 1e-9:
        return float("nan")
    return (ablated - spliced) / denom


def prediction_mask(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Positions whose next-token prediction counts: the token AND its successor are real.

    Padding is excluded for the same reason it is excluded from training:
    including it measures how well the model predicts PAD.
    """
    if attention_mask is None:
        return torch.ones(input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.bool,
                          device=input_ids.device)
    mask = attention_mask.bool()
    return mask[:, 1:] & mask[:, :-1]


#: Upper bound on one float32 ``[positions, vocab]`` block in the cross-entropy
#: arithmetic. The KL of a chunk briefly holds five such blocks (two
#: log-softmaxes, and ``exp``, difference and product inside ``kl_div``).
CE_CHUNK_BYTES = 256 * 1024 ** 2


def positions_per_chunk(vocab_size: int, chunk_bytes: Optional[int] = None) -> int:
    """Next-token positions whose float32 log-probabilities fit in ``chunk_bytes``.

    The bound is read when called (``CE_CHUNK_BYTES`` when None), not frozen into
    a default argument, so the module constant is the one value that decides it.
    """
    budget = CE_CHUNK_BYTES if chunk_bytes is None else int(chunk_bytes)
    return max(1, budget // (4 * max(1, int(vocab_size))))


def ce_working_bytes(vocab_size: int, batch_tokens: int, logit_bytes: int = 4,
                     chunk_bytes: Optional[int] = None) -> int:
    """GPU bytes the cross-entropy pass holds beside the model, for one batch.

    The untouched model's logits are kept for the batch while every substitution
    runs, and each substitution's forward produces its own — two ``[tokens,
    vocab]`` tensors in the model's logit dtype (``logit_bytes``; 4 covers a
    model that upcasts) — plus five float32 chunks for the log-softmax and KL.
    """
    budget = CE_CHUNK_BYTES if chunk_bytes is None else int(chunk_bytes)
    logits = 2 * int(batch_tokens) * int(vocab_size) * int(logit_bytes)
    chunk = min(budget, int(batch_tokens) * int(vocab_size) * 4)
    return logits + 5 * chunk


def _prediction_chunks(valid: torch.Tensor, vocab_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """``(rows, cols)`` index pairs of the valid positions, a chunk at a time."""
    positions = valid.nonzero(as_tuple=False)
    return [(chunk[:, 0], chunk[:, 1]) for chunk in torch.split(positions, positions_per_chunk(vocab_size))]


def _nll_sum(logits: torch.Tensor, labels: torch.Tensor,
             chunks: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Σ over valid positions of −log p(label), from logits, a chunk of positions at a time.

    NEVER A WHOLE-BATCH LOG-SOFTMAX (review R1-C, 2026-09-15). The first version
    turned the batch's logits into float32 log-probabilities, held the untouched
    model's copy across every substitution, and took the KL over the full
    ``[batch, seq, vocab]`` tensor: five float32 copies of it at once, measured at
    5.0 GB for LFM2.5 (vocab 65,536) at the default 4,096 tokens a batch, and 20 GB
    for a 262,144-token vocabulary, against a 3 GB reservation.
    """
    total = torch.zeros((), dtype=torch.float64, device=logits.device)
    for rows, cols in chunks:
        log_probs = F.log_softmax(logits[rows, cols].float(), dim=-1)
        total += -log_probs.gather(-1, labels[rows, cols].unsqueeze(-1)).double().sum()
    return float(total)


def _nll_and_kl_sums(base_logits: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor,
                     chunks: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[float, float]:
    """``(Σ −log p_other(label), Σ KL(base || other))`` over valid positions, chunked."""
    nll = torch.zeros((), dtype=torch.float64, device=logits.device)
    kl = torch.zeros((), dtype=torch.float64, device=logits.device)
    for rows, cols in chunks:
        log_probs = F.log_softmax(logits[rows, cols].float(), dim=-1)
        nll += -log_probs.gather(-1, labels[rows, cols].unsqueeze(-1)).double().sum()
        base_log_probs = F.log_softmax(base_logits[rows, cols].to(log_probs.device).float(), dim=-1)
        # KL(base || other) = Σ p_base (log p_base − log p_other)
        kl += F.kl_div(log_probs, base_log_probs, log_target=True, reduction="sum").double()
        del log_probs, base_log_probs
    return float(nll), float(kl)


@dataclass
class _LayerStats:
    tokens: int = 0
    active_latents: float = 0.0
    l0_known: bool = True
    activation_sum: Optional[torch.Tensor] = field(default=None, repr=False)  # float64, CPU
    fvu: FvuSums = field(default_factory=FvuSums)
    nll_spliced: float = 0.0
    nll_mean: float = 0.0
    nll_zero: float = 0.0
    kl_spliced: float = 0.0

    def mean_activation(self) -> Optional[torch.Tensor]:
        if not self.tokens or self.activation_sum is None:
            return None
        return self.activation_sum / self.tokens


def _finite(value: Optional[float]) -> Optional[float]:
    """JSON (and Postgres JSONB) have no NaN; an undefined number is None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@torch.no_grad()
def evaluate_spliced_layers(
    model: Any,
    saes: Mapping[int, torch.nn.Module],
    layer_modules: Mapping[int, torch.nn.Module],
    batches: Callable[[], Iterable[Batch]],
    progress: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """Measure what each SAE costs the model at its layer, and all of them together.

    Args:
        model: a causal LM whose forward takes ``input_ids`` and ``attention_mask``
            and returns an object with ``.logits``.
        saes: layer index -> SAE reconstructing that layer's OUTPUT (resid_post).
        layer_modules: layer index -> the decoder-layer module whose output is spliced.
        batches: called once per pass; yields ``(input_ids, attention_mask)`` on the
            model's input device. Both passes must see the same batches.
        progress: called as ``progress(stage, batches_done)`` after every batch of
            each pass (``"means"``, then ``"cross_entropy"``) — the caller's
            heartbeat, so a long evaluation is never mistaken for a dead one.

    Returns:
        ``tokens``, ``predicted_tokens``, ``ce_base``, ``layers`` (one dict per layer)
        and ``all_layers_spliced``. Undefined numbers are None, never NaN.
    """
    layers = sorted(saes)
    if not layers:
        raise ValueError("no SAEs to evaluate")
    missing = [L for L in layers if L not in layer_modules]
    if missing:
        raise ValueError(f"no layer module for layers {missing}")

    was_training = model.training
    model.eval()
    stats = {L: _LayerStats() for L in layers}
    batches_done = 0
    try:
        # ── pass 1: each layer's mean activation over real tokens, and the SAE's
        #    reconstruction statistics on them. Nothing is spliced.
        for input_ids, attention_mask in batches():
            real = (
                attention_mask.bool() if attention_mask is not None
                else torch.ones_like(input_ids, dtype=torch.bool)
            )

            def capture(L):
                def hook(_module, _inputs, output):
                    x = _output_tensor(output)
                    xs = x[real.to(x.device)]
                    if xs.shape[0] == 0:
                        return None
                    s = stats[L]
                    x_hat, z = encode_decode_at_layer(saes[L], xs)
                    s.tokens += int(xs.shape[0])
                    summed = xs.double().sum(dim=0).cpu()
                    s.activation_sum = summed if s.activation_sum is None else s.activation_sum + summed
                    s.fvu.update(xs, x_hat)
                    if z is None:
                        s.l0_known = False
                    else:
                        s.active_latents += float((z != 0).sum())
                    return None
                return hook

            handles = [layer_modules[L].register_forward_hook(capture(L)) for L in layers]
            try:
                model(input_ids, attention_mask=attention_mask, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            batches_done += 1
            if progress is not None:
                progress("means", batches_done)

        batches_done = 0
        means = {L: stats[L].mean_activation() for L in layers}
        empty = [L for L in layers if means[L] is None]
        if empty:
            raise ValueError(f"the evaluation saw no real tokens at layers {empty}")

        # ── pass 2: cross-entropy with each substitution.
        n_pred = 0
        nll_base = 0.0
        nll_all = 0.0
        kl_all = 0.0

        def forward_logits(splices, input_ids, attention_mask):
            """The next-token logits (every position but the last), as the model produced them."""
            with _splice_many(splices):
                logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
            return logits[:, :-1, :]

        for input_ids, attention_mask in batches():
            labels = input_ids[:, 1:]
            valid = prediction_mask(input_ids, attention_mask)
            if not bool(valid.any()):
                continue
            # The untouched model's LOGITS are held for the batch, in the model's
            # own dtype — never its float32 log-probabilities.
            base_logits = forward_logits([], input_ids, attention_mask)
            labels = labels.to(base_logits.device)
            valid = valid.to(base_logits.device)
            chunks = _prediction_chunks(valid, base_logits.shape[-1])
            n_pred += int(valid.sum())
            nll_base += _nll_sum(base_logits, labels, chunks)

            for L in layers:
                module, sae, mean = layer_modules[L], saes[L], means[L]
                s = stats[L]
                logits = forward_logits([(module, lambda x, sae=sae: reconstruct_at_layer(sae, x))],
                                        input_ids, attention_mask)
                nll, kl = _nll_and_kl_sums(base_logits, logits, labels, chunks)
                s.nll_spliced += nll
                s.kl_spliced += kl
                del logits
                logits = forward_logits([(module, lambda x, mean=mean: mean_ablation_at_layer(mean, x))],
                                        input_ids, attention_mask)
                s.nll_mean += _nll_sum(logits, labels, chunks)
                del logits
                logits = forward_logits([(module, zero_ablation_at_layer)], input_ids, attention_mask)
                s.nll_zero += _nll_sum(logits, labels, chunks)
                del logits

            logits = forward_logits(
                [(layer_modules[L], lambda x, sae=saes[L]: reconstruct_at_layer(sae, x)) for L in layers],
                input_ids, attention_mask,
            )
            nll, kl = _nll_and_kl_sums(base_logits, logits, labels, chunks)
            nll_all += nll
            kl_all += kl
            del logits, base_logits
            batches_done += 1
            if progress is not None:
                progress("cross_entropy", batches_done)
    finally:
        if was_training:
            model.train()

    if n_pred == 0:
        raise ValueError("the evaluation had no predictable positions (every block was padding)")

    ce_base = nll_base / n_pred
    per_layer: List[Dict[str, Any]] = []
    for L in layers:
        s = stats[L]
        ce_spliced = s.nll_spliced / n_pred
        ce_mean = s.nll_mean / n_pred
        ce_zero = s.nll_zero / n_pred
        per_layer.append({
            "layer": L,
            "tokens": s.tokens,
            "ce_spliced": _finite(ce_spliced),
            "ce_mean_ablated": _finite(ce_mean),
            "ce_zero_ablated": _finite(ce_zero),
            "ce_delta": _finite(ce_spliced - ce_base),
            "loss_recovered_vs_mean": _finite(loss_recovered(ce_base, ce_spliced, ce_mean)),
            "loss_recovered_vs_zero": _finite(loss_recovered(ce_base, ce_spliced, ce_zero)),
            "kl": _finite(s.kl_spliced / n_pred),
            "l0": _finite(s.active_latents / s.tokens) if s.l0_known and s.tokens else None,
            "fvu_centred": _finite(s.fvu.centred()),
            "fvu_legacy": _finite(s.fvu.legacy()),
        })

    ce_all = nll_all / n_pred
    return {
        "tokens": stats[layers[0]].tokens,
        "predicted_tokens": n_pred,
        "ce_base": _finite(ce_base),
        "layers": per_layer,
        "all_layers_spliced": {
            "layers": list(layers),
            "ce": _finite(ce_all),
            "ce_delta": _finite(ce_all - ce_base),
            "kl": _finite(kl_all / n_pred),
        },
        "zero_ablation_note": ZERO_ABLATION_NOTE,
    }
