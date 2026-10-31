"""Fraction of Variance Unexplained — the ONE definition, and the legacy one beside it.

TWO FORMULAS, AND WHY BOTH ARE KEPT (SAE training remediation, item 5, 2026-09-15).

* **Centred FVU** — the standard definition, and the headline::

      FVU = Σ‖x − x̂‖² / Σ‖x − μ‖²,   μ = the PER-DIMENSION mean of x

  "What share of the variance around each dimension's own mean does the
  reconstruction fail to explain." 0 is perfect, 1 is no better than predicting
  the mean vector.

* **Legacy FVU** — what miStudio stored in ``training_metrics.fvu`` and
  ``trainings.current_fvu`` until this change::

      var(x − x̂) / var(x)   over ALL elements, with ONE global mean

  The denominator subtracts a single scalar mean from every element, so a
  dimension that sits far from zero contributes its whole offset² to the
  "variance" even when it never moves. LFM2.5-1.2B's residual stream has large,
  near-constant outlier dimensions, so the denominator is inflated and the
  reported FVU reads LOW: held-out at layer 11 of train_6247e768 the legacy
  value was 0.26 where the centred value is 0.32.

The legacy value keeps its name and meaning so no stored row silently changes
what it says; every new number is written beside it under ``fvu_centred``.

Both are computed in RAW activation space — the layer output the model reads —
never in the SAE's normalised training space, which is a per-token rescale and
would weight tokens differently from the model.

TWO SHAPES OF THE SAME ARITHMETIC. :func:`fvu_pair` computes both values on one
batch. :class:`FvuSums` accumulates the sufficient statistics over chunks so an
evaluation too large for one pass gives exactly the same answer as one pass —
the held-out evaluation and the post-run evaluation both aggregate that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

#: Added to the legacy denominator, exactly as the original JumpReLU formula did,
#: so historical and newly logged legacy values stay comparable.
LEGACY_EPS = 1e-8


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """[..., d] -> [N, d] in float32: statistics are over tokens, whatever the batch shape."""
    return x.detach().reshape(-1, x.shape[-1]).float()


@torch.no_grad()
def fvu_pair(x: torch.Tensor, x_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(legacy, centred)`` FVU of a reconstruction, as 0-d float32 tensors.

    Args:
        x: the activations the SAE was given, raw space, ``[..., d]``.
        x_hat: its reconstruction, same shape and space.
    """
    x = _flatten(x)
    residual = x - _flatten(x_hat)

    legacy = residual.var() / (x.var() + LEGACY_EPS)

    centred_total = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
    centred = residual.pow(2).sum() / centred_total.clamp_min(torch.finfo(torch.float32).tiny)
    return legacy, centred


@dataclass
class FvuSums:
    """Sufficient statistics for both FVUs, accumulated chunk by chunk.

    Float64 throughout: the centred denominator is a difference of two large
    sums (``Σx² − n‖μ‖²``), and float32 loses the small remainder on an
    activation stream with large constant offsets — the very case this exists for.
    """

    n: int = 0
    d: Optional[int] = None
    sse: float = 0.0                  # Σ‖x − x̂‖²
    sum_residual: float = 0.0         # Σ over every element of (x − x̂)
    sum_x2: float = 0.0               # Σ over every element of x²
    sum_x: Optional[torch.Tensor] = field(default=None, repr=False)  # per-dimension Σx, float64, CPU

    @torch.no_grad()
    def update(self, x: torch.Tensor, x_hat: torch.Tensor) -> "FvuSums":
        x = x.detach().reshape(-1, x.shape[-1]).double()
        residual = x - x_hat.detach().reshape(-1, x.shape[-1]).double()
        if self.d is None:
            self.d = int(x.shape[-1])
            self.sum_x = torch.zeros(self.d, dtype=torch.float64)
        elif int(x.shape[-1]) != self.d:
            raise ValueError(f"width changed between chunks: {self.d} then {x.shape[-1]}")
        self.n += int(x.shape[0])
        self.sse += float(residual.pow(2).sum())
        self.sum_residual += float(residual.sum())
        self.sum_x2 += float(x.pow(2).sum())
        self.sum_x += x.sum(dim=0).cpu()
        return self

    def centred(self) -> Optional[float]:
        """Σ‖x − x̂‖² / Σ‖x − μ‖²; None when nothing was accumulated or x never varied."""
        if not self.n or self.sum_x is None:
            return None
        total = self.sum_x2 - float(self.sum_x.pow(2).sum()) / self.n
        if total <= 0:
            return None
        return self.sse / total

    def legacy(self) -> Optional[float]:
        """var(x − x̂) / (var(x) + eps) over every element, as the legacy formula computed it."""
        if not self.n or self.sum_x is None:
            return None
        elements = self.n * self.d
        if elements < 2:
            return None
        var_residual = (self.sse - self.sum_residual ** 2 / elements) / (elements - 1)
        var_x = (self.sum_x2 - float(self.sum_x.sum()) ** 2 / elements) / (elements - 1)
        return var_residual / (var_x + LEGACY_EPS)
