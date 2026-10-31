"""Per-feature firing rates during training — the dense end of the distribution.

WHY THIS EXISTS. Training tracked `feature_activation_ema` and read it only at
the DEAD end (`< 0.01`). Nothing ever looked at the other tail, so an SAE where
5% of latents fire on 80% of tokens — the signature of polysemantic,
uninterpretable features — reported a perfectly healthy aggregate L0 and raised
nothing.

The existing EMA cannot answer this even if it were read. It accumulates a
BATCH-LEVEL indicator ("did this feature fire for any token in the batch") with
an unnormalised update, so its steady state is `window / batch_size`, not 1.0.
It is a staleness counter, not a frequency, and must not be read as one. This
module tracks a real rate alongside it and leaves the dead-count path untouched.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

#: A feature firing on more than this share of tokens is doing too much work to
#: be one concept. Reported, never acted on automatically.
DENSE_THRESHOLD = 0.3

#: Below this a feature is effectively silent. Kept separate from the training
#: loop's own dead-neuron rule, which uses a different quantity.
SILENT_THRESHOLD = 1e-4


def update_firing_rate(
    previous: Optional[torch.Tensor],
    z: torch.Tensor,
    momentum: float = 0.99,
) -> torch.Tensor:
    """EMA of the fraction of TOKENS on which each feature fires.

    Unlike the dead-neuron accumulator this is normalised — it converges to the
    true firing rate, so `0.3` means "30% of tokens" and can be compared across
    runs, layers and batch sizes.
    """
    rate = (z > 0).float().mean(dim=0).detach()
    if previous is None:
        return rate
    return previous * momentum + rate * (1.0 - momentum)


def density_summary(
    firing_rate: torch.Tensor,
    dense_threshold: float = DENSE_THRESHOLD,
) -> Dict[str, float]:
    """A checkable description of the firing-rate distribution.

    Percentiles rather than a raw histogram: they survive being written to a
    metrics row, and the shape question ("is a handful of features doing all the
    work?") is a tail question.
    """
    if firing_rate.numel() == 0:
        return {}

    rates = firing_rate.detach().float().flatten()
    total = rates.numel()
    quantiles = torch.tensor([0.5, 0.9, 0.99], device=rates.device, dtype=rates.dtype)
    p50, p90, p99 = torch.quantile(rates, quantiles).tolist()

    dense = int((rates > dense_threshold).sum().item())
    silent = int((rates < SILENT_THRESHOLD).sum().item())

    return {
        "density_p50": p50,
        "density_p90": p90,
        "density_p99": p99,
        "density_max": float(rates.max().item()),
        "dense_features": dense,
        "dense_fraction": dense / total,
        "silent_features": silent,
        "silent_fraction": silent / total,
        # What share of all firing is done by the top 1% of features. A healthy
        # dictionary spreads the work; a collapsing one concentrates it.
        "top1pct_share": float(
            rates.topk(max(1, total // 100)).values.sum().item()
            / max(rates.sum().item(), 1e-9)
        ),
    }


def describe(summary: Dict[str, float]) -> str:
    """One log line an operator can act on."""
    if not summary:
        return "density: no features"
    return (
        "density p50={density_p50:.4f} p90={density_p90:.4f} p99={density_p99:.4f} "
        "max={density_max:.4f} | dense(>{t:.0%})={dense_features} "
        "({dense_fraction:.1%}) | top1%share={top1pct_share:.1%}"
    ).format(t=DENSE_THRESHOLD, **summary)
