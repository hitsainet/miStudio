"""How many tokens each source contributes to a training run.

WHY THIS EXISTS. Multiple extractions were pooled by plain concatenation and
sampled uniformly, so the mixture was strictly proportional to **padded row
count**. Measured on this estate, that made intent and reality diverge sharply:

    source            sampled activations   real tokens
    OpenWebText              42.8%             72.4%
    hard-negatives           33.8%             20.8%
    Bloomberg                19.1%              1.2%
    the Pile                  4.3%              5.6%

Bloomberg was presumably selected to buy financial-domain features and
contributed 19% of the compute for 1.2% of the content, because its documents
are 17-token headlines in a 512-token window.

With padding masked out, "proportional" finally means proportional to real
text. This module is the next step: letting the operator ask for a mixture
directly, rather than inferring one from how long the documents happen to be.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


def normalise_weights(
    weights: Optional[Sequence[float]], n_parts: int
) -> Optional[List[float]]:
    """Validate and normalise a requested mixture to sum to 1.

    Returns None for "no preference", which means token-count proportional —
    the historical behaviour, and the right default.
    """
    if weights is None:
        return None
    weights = list(weights)
    if len(weights) != n_parts:
        raise ValueError(
            f"dataset_weights has {len(weights)} entries but the run has "
            f"{n_parts} sources; they must correspond one-to-one"
        )
    if any(w < 0 for w in weights):
        raise ValueError(f"dataset_weights must be non-negative, got {weights}")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("dataset_weights sum to zero; that selects no data at all")
    return [w / total for w in weights]


def allocate_tokens(
    available: Sequence[int],
    total_to_load: int,
    weights: Optional[Sequence[float]] = None,
) -> List[int]:
    """Decide how many tokens to take from each source.

    `available` is per-source trainable (non-pad) token counts.

    Without weights this is proportional to availability, which reproduces the
    previous behaviour once padding is excluded.

    With weights, a source that cannot meet its share contributes everything it
    has and the shortfall is redistributed across the sources that can — asking
    for 25% chat from a corpus that only holds 10% should not silently shrink
    the whole run to 40% of its budget.
    """
    available = [max(0, int(a)) for a in available]
    n = len(available)
    if n == 0:
        return []

    total_available = sum(available)
    total_to_load = max(0, min(int(total_to_load), total_available))
    if total_to_load == 0:
        return [0] * n

    if weights is None:
        weights = [a / total_available if total_available else 0.0 for a in available]
    else:
        weights = normalise_weights(weights, n)

    alloc = [0] * n
    remaining = total_to_load
    active = [i for i in range(n) if available[i] > 0 and weights[i] > 0]

    # Repeatedly hand out the remaining budget by weight, capping at what each
    # source actually holds, until nothing more can be placed.
    while remaining > 0 and active:
        weight_sum = sum(weights[i] for i in active)
        if weight_sum <= 0:
            break
        placed_any = False
        for i in list(active):
            share = int(remaining * (weights[i] / weight_sum))
            take = min(share, available[i] - alloc[i])
            if take > 0:
                alloc[i] += take
                placed_any = True
            if alloc[i] >= available[i]:
                active.remove(i)
        remaining = total_to_load - sum(alloc)
        if not placed_any:
            # Rounding left a remainder smaller than one share each; give it to
            # whoever still has room, largest weight first, so the budget is met.
            for i in sorted(active, key=lambda j: -weights[j]):
                room = available[i] - alloc[i]
                take = min(room, remaining)
                alloc[i] += take
                remaining -= take
                if remaining <= 0:
                    break
            break

    return alloc


def describe_mixture(labels: Sequence[str], alloc: Sequence[int]) -> str:
    """A one-line, checkable statement of what the run will actually read."""
    total = sum(alloc) or 1
    parts = [f"{l}={a:,} ({a / total * 100:.1f}%)" for l, a in zip(labels, alloc)]
    return " · ".join(parts)
