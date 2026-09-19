"""
Band-report metrics (BR-002) and their null controls.

EVERY METRIC HERE IS A LAYER PROFILE, not a score. The report describes a model;
it does not rank one. That distinction is load-bearing because of BR-004: the
J-lens is DELIBERATELY worse than the logit lens on next-token agreement through
most of the network, so agreement appears here as a described profile and never
as a quality criterion. Reporting it and scoring on it are different acts, and
the one that is forbidden is scoring.

TWO METRICS CARRY NULL CONTROLS AND THE CONTROL IS PART OF THE METRIC, not a
sanity extra:

  * top-1 autocorrelation is meaningless without a POSITION-SHUFFLED NULL —
    adjacent positions share context, so a high raw autocorrelation is the
    expected result of nothing at all.
  * fraction-of-variance-explained is reported only IN EXCESS OF a size-matched
    RANDOM-DIRECTION control — any k directions explain some variance, and k
    random ones explain a surprising amount.

The control seed is recorded because the excess, not the raw figure, is the
finding, and a figure whose control cannot be reconstructed cannot be believed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


@dataclass
class LayerProfile:
    """One layer's metrics. Absent means NOT COMPUTABLE, never zero."""

    layer: int
    kurtosis: Optional[float] = None
    autocorrelation: Optional[float] = None
    autocorrelation_null: Optional[float] = None
    effective_dimensionality: Optional[float] = None
    #: STRUCTURALLY ABSENT in the J-lens band pipeline, and correctly so.
    #: `occupancy` measures how much of a SPARSITY BUDGET a decomposition uses,
    #: and this pipeline has no budget — it reads residuals through a Jacobian,
    #: it does not decompose them under a k. `None` here is the dataclass's own
    #: convention for "not computable", not a gap (MIS-E2E-088 re-check: the
    #: finding listed it alongside `excess_fve`, but its remaining references
    #: are docstrings, not dead calls).
    occupancy: Optional[float] = None
    #: Populated since MIS-E2E-088. This WAS always None because nothing called
    #: `excess_fve`, so the random-direction control `control_seed` exists to
    #: make reproducible never ran.
    excess_fve: Optional[float] = None
    # Described, never scored (BR-004).
    next_token_agreement: Optional[float] = None

    @property
    def excess_autocorrelation(self) -> Optional[float]:
        """Autocorrelation above the position-shuffled null.

        The raw figure is not a finding. Adjacent positions share context, so a
        high value is what a model with no cross-position structure at all would
        also produce.
        """
        if self.autocorrelation is None or self.autocorrelation_null is None:
            return None
        return self.autocorrelation - self.autocorrelation_null


def excess_kurtosis(values: torch.Tensor) -> float:
    """Excess kurtosis (normal = 0) of a readout distribution."""
    x = values.to(torch.float64).flatten()
    if x.numel() < 4:
        raise ValueError("excess kurtosis needs at least 4 values")
    centred = x - x.mean()
    var = (centred**2).mean()
    if var <= 0:
        return 0.0
    return float((centred**4).mean() / (var**2) - 3.0)


def top1_autocorrelation(top1_ids: Sequence[int]) -> float:
    """Fraction of adjacent positions sharing a top-1 readout."""
    if len(top1_ids) < 2:
        raise ValueError("autocorrelation needs at least 2 positions")
    matches = sum(1 for a, b in zip(top1_ids, top1_ids[1:]) if a == b)
    return matches / (len(top1_ids) - 1)


def shuffled_null_autocorrelation(
    top1_ids: Sequence[int], seed: int, trials: int = 32
) -> float:
    """The same statistic over position-shuffled sequences.

    THE COMPARISON, not a sanity check. Without it a high raw autocorrelation is
    reported as structure when it is the arithmetic consequence of a repeated
    token — shuffling destroys position structure while preserving the token
    distribution, so anything the null also achieves is not a finding.
    """
    if len(top1_ids) < 2:
        raise ValueError("autocorrelation needs at least 2 positions")
    generator = torch.Generator().manual_seed(seed)
    ids = torch.tensor(list(top1_ids))
    total = 0.0
    for _ in range(trials):
        permuted = ids[torch.randperm(ids.numel(), generator=generator)]
        total += top1_autocorrelation(permuted.tolist())
    return total / trials


def effective_dimensionality(matrix: torch.Tensor) -> float:
    """Participation ratio of the singular-value spectrum.

    A continuous count of directions carrying real variance. Not the rank: rank
    is a threshold on numerical noise and reports full for almost any real
    matrix, which is the same answer for every model.
    """
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(matrix.shape)}")
    sv = torch.linalg.svdvals(matrix.to(torch.float64))
    energy = sv**2
    total = energy.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / (energy**2).sum())


def fraction_variance_explained(
    activations: torch.Tensor, directions: torch.Tensor
) -> float:
    """FVE of `activations` by the span of `directions`."""
    if activations.ndim != 2 or directions.ndim != 2:
        raise ValueError("both arguments must be matrices")
    if activations.shape[1] != directions.shape[1]:
        raise ValueError(
            f"d_model mismatch: activations {activations.shape[1]} vs "
            f"directions {directions.shape[1]}"
        )
    x = activations.to(torch.float64)
    d = directions.to(torch.float64).T          # [d_model, k]

    # RANK, NOT COLUMN COUNT (MIS-E2E-088).
    #
    # `torch.linalg.qr(...).Q` returns as many orthonormal columns as the input
    # has, PADDING with arbitrary directions when the input is rank-deficient.
    # Those padded directions explain variance the real directions do not, and
    # the answer comes back as though they had. Reproduced: four DUPLICATE
    # directions report FVE 0.378 against a true 0.083 — a 4.5x overstatement.
    #
    # BR-002 is this product's load-bearing honesty rule: bands render only from
    # a measured report, never a constant, so that no borrowed boundary is
    # presented as measured. A report can satisfy that rule perfectly and still
    # be wrong about its content, which is what this was.
    #
    # SVD gives the rank and an orthonormal basis for the actual span in one
    # step. The tolerance is the standard `max(m, n) * eps * sigma_max`.
    u, sigma, _ = torch.linalg.svd(d, full_matrices=False)
    if sigma.numel() == 0:
        return 0.0
    tol = max(d.shape) * torch.finfo(torch.float64).eps * float(sigma[0])
    rank = int((sigma > tol).sum())
    if rank == 0:
        # Degenerate directions span nothing, so they explain nothing. Zero is
        # the correct answer; padding would have invented a subspace.
        return 0.0
    basis = u[:, :rank]

    projected = x @ basis @ basis.T
    total = float((x**2).sum())
    if total <= 0:
        return 0.0
    return float((projected**2).sum() / total)


def excess_fve(
    activations: torch.Tensor,
    directions: torch.Tensor,
    control_seed: int,
    trials: int = 8,
) -> float:
    """FVE above a SIZE-MATCHED random-direction control.

    Any k directions explain some variance and k random ones explain a
    surprising amount, so the raw figure is uninterpretable — a report claiming
    "these directions explain 60% of the variance" is not saying anything until
    it says what 60 random ones explain.

    `control_seed` is required rather than defaulted, so the control is
    reconstructible. The schema makes it required for the same reason.
    """
    k, d_model = directions.shape
    generator = torch.Generator().manual_seed(control_seed)
    observed = fraction_variance_explained(activations, directions)

    control_total = 0.0
    for _ in range(trials):
        random_directions = torch.randn(k, d_model, generator=generator, dtype=torch.float64)
        control_total += fraction_variance_explained(activations, random_directions)
    return observed - control_total / trials


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    """Linear CKA between two representations of the same points.

    Symmetric and invariant to isotropic scaling and rotation, which is what
    makes it comparable across layers whose scales differ by orders of
    magnitude.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"{a.shape[0]} rows vs {b.shape[0]} — not the same points")
    x = a.to(torch.float64) - a.to(torch.float64).mean(0, keepdim=True)
    y = b.to(torch.float64) - b.to(torch.float64).mean(0, keepdim=True)
    hsic = float((x.T @ y).pow(2).sum())
    norm_x = float((x.T @ x).pow(2).sum()) ** 0.5
    norm_y = float((y.T @ y).pow(2).sum()) ** 0.5
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return hsic / (norm_x * norm_y)


def cross_layer_cka(representations: Dict[int, torch.Tensor]) -> Dict[int, Dict[int, float]]:
    """Pairwise CKA across layers."""
    layers = sorted(representations)
    return {
        i: {j: linear_cka(representations[i], representations[j]) for j in layers}
        for i in layers
    }


def occupancy(active_counts: Sequence[int], k: int) -> float:
    """Mean fraction of the sparsity budget actually used.

    `k` is the budget, so occupancy is bounded in [0, 1]; a decomposition using
    fewer than k active tokens is saying something about the data, not failing.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not active_counts:
        raise ValueError("no decompositions to summarise")
    return sum(min(c, k) for c in active_counts) / (k * len(active_counts))


def derive_boundaries(profiles: Sequence[LayerProfile]) -> Optional[Dict[str, int]]:
    """This model's OWN sensory / workspace / motor boundaries.

    Derived from the model's measured profile and NOTHING ELSE. There is no
    fallback and no default: returning None when the profile cannot support a
    split is the honest answer, and it is what keeps the product from drawing
    bands it has not earned (BR-002).

    The published ~L38-92 figures were measured on one specific model, and
    porting them anywhere is what this function exists to make impossible —
    which is why no constant appears in it.

    The split uses excess kurtosis: the readout distribution sharpens where
    reportable content appears and sharpens again as the model commits to an
    output. The workspace begins at the first sustained rise above the
    layer-median and the motor band at the final sustained rise.
    """
    usable = [p for p in profiles if p.kurtosis is not None]
    if len(usable) < 4:
        return None

    ordered = sorted(usable, key=lambda p: p.layer)
    values = torch.tensor([p.kurtosis for p in ordered], dtype=torch.float64)
    median = float(values.median())
    if not torch.isfinite(values).all() or float(values.max()) <= median:
        return None

    # SUSTAINED, as the docstring has always claimed (MIS-E2E-088).
    #
    # This took `above[0]` — the FIRST layer over the median — so a single noisy
    # layer set `workspace_start`, including at layer 0, which yields an EMPTY
    # sensory band. "First sustained rise" was documented and first-crossing was
    # implemented.
    #
    # Two consecutive layers is the minimum that can distinguish a rise from a
    # spike, and the bar is kept there deliberately: a longer run would start
    # discarding genuinely narrow workspaces on small models, and BR-002's
    # answer to "cannot tell" is None, not a guess.
    run = [i for i, v in enumerate(values.tolist()) if v > median]
    above_set = set(run)
    sustained = [i for i in run if (i + 1) in above_set or (i - 1) in above_set]
    if not sustained:
        # Every crossing is an isolated spike: no rise to locate.
        return None

    workspace_index = sustained[0]
    # THE SAME DEFECT ON THE OTHER BOUNDARY. `peak_index` was a raw argmax over
    # every layer, so an isolated late spike could set `motor_start` exactly as
    # an isolated early one could set `workspace_start`. The docstring says
    # "final SUSTAINED rise" for both. Restricting the argmax to the sustained
    # set is what makes the two boundaries obey the same rule — found by the
    # fixture for the workspace test, not by the finding.
    peak_index = max(sustained, key=lambda i: float(values[i]))
    # The motor band starts at the last sustained rise, which is at or after the
    # peak. If the peak IS the first rise there is no three-way split to make.
    if peak_index <= workspace_index:
        return None

    return {
        "workspace_start": ordered[workspace_index].layer,
        "motor_start": ordered[peak_index].layer,
    }
