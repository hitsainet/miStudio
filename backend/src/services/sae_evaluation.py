"""Does the SAE reconstruction still let the model do its job?

WHY THIS EXISTS. Every number reported about an SAE here was measured in the
SAE's own space — FVU, L0, dead count. None of them says what the
reconstruction costs the MODEL. An SAE can hit FVU 0.02 and still destroy the
next-token distribution, because reconstruction error is not uniformly
important: the directions that matter for the output are a small part of the
variance.

The standard answer is to splice the reconstruction back into the residual
stream and measure cross-entropy. Three numbers make it interpretable:

    baseline   — CE with the model untouched
    spliced    — CE with the SAE's reconstruction substituted
    ablated    — CE with the activation replaced by the SAE's bias alone

and the summary is **loss recovered**:

    (ablated - spliced) / (ablated - baseline)

1.0 means the reconstruction costs the model nothing; 0.0 means it is worth no
more than deleting the layer's contribution entirely. This is scale-free, which
raw CE delta is not — a 0.05 nat delta means something very different on a model
whose baseline is 2.0 than on one at 8.0.

NOTE ON WIRING. The cached-activation training path never loads the base model,
so this cannot run inside that loop; it is an evaluation step that needs the
model. The on-the-fly path does have one.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _splice(module: torch.nn.Module, transform: Callable[[torch.Tensor], torch.Tensor]):
    """Replace a module's output for the duration of the block.

    THE HOOK MUST COME OFF. A forward hook left attached silently changes every
    subsequent forward pass — including the training steps that follow — and the
    symptom would be a slow, unexplained quality drift rather than an error.
    `finally` is what guarantees it, including when the forward raises.
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


def _cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor,
                   attention_mask: Optional[torch.Tensor]) -> float:
    """Next-token CE over real tokens only.

    Padding is excluded here for the same reason it is excluded from training:
    including it measures how well the model predicts PAD.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        reduction="none",
    )
    if attention_mask is not None:
        mask = attention_mask[:, 1:].reshape(-1).to(losses.dtype)
        total = mask.sum()
        if total <= 0:
            return float("nan")
        return float((losses * mask).sum() / total)
    return float(losses.mean())


def loss_recovered(baseline: float, spliced: float, ablated: float) -> float:
    """Fraction of the model's capability the reconstruction preserves.

    Returns NaN when the denominator collapses — if ablating the layer costs
    nothing, "how much of that cost did we avoid" has no answer, and returning
    1.0 there would report a perfect score for a layer that does nothing.
    """
    denom = ablated - baseline
    if abs(denom) < 1e-9:
        return float("nan")
    return (ablated - spliced) / denom


@torch.no_grad()
def spliced_ce_delta(
    model: Any,
    sae: torch.nn.Module,
    layer_module: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Measure what the SAE's reconstruction costs the model.

    `sae` must round-trip an activation: `sae(x)` returning `(x_hat, ...)`, which
    is the shape every architecture here shares.
    """
    was_training = model.training
    model.eval()
    try:
        base_logits = model(input_ids, attention_mask=attention_mask).logits
        baseline = _cross_entropy(base_logits, input_ids, attention_mask)

        def reconstruct(x: torch.Tensor) -> torch.Tensor:
            out = sae(x)
            x_hat = out[0] if isinstance(out, tuple) else out
            return x_hat.to(x.dtype)

        with _splice(layer_module, reconstruct):
            spliced_logits = model(input_ids, attention_mask=attention_mask).logits
        spliced = _cross_entropy(spliced_logits, input_ids, attention_mask)

        # The floor: what the model scores when this activation carries no
        # information beyond the SAE's own bias. Without it, a CE delta has no
        # scale and cannot be compared across layers or models.
        def ablate(x: torch.Tensor) -> torch.Tensor:
            b_dec = getattr(sae, "b_dec", None)
            if b_dec is None:
                return torch.zeros_like(x)
            return b_dec.to(x.dtype).expand_as(x)

        with _splice(layer_module, ablate):
            ablated_logits = model(input_ids, attention_mask=attention_mask).logits
        ablated = _cross_entropy(ablated_logits, input_ids, attention_mask)
    finally:
        if was_training:
            model.train()

    return {
        "ce_baseline": baseline,
        "ce_spliced": spliced,
        "ce_ablated": ablated,
        "ce_delta": spliced - baseline,
        "loss_recovered": loss_recovered(baseline, spliced, ablated),
    }
