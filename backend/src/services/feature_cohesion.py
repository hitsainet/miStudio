"""Embedding cohesion: how alike are a feature's activating passages?

A deterministic alternative to asking a model "is this feature interpretable".
That question turned out to be unstable — gemma-4-12B-it flips its refuse/label
verdict on 47% of features when the same ten passages are merely reordered, and
LFM2.5-1.2B disagrees with itself 55% of the time. Cohesion is computed rather
than judged, so it has no sampling noise, no sensitivity to presentation order,
and no prompt to echo.

THE CONTRAST IS NOT OPTIONAL. Raw cohesion is dominated by the corpus, not the
feature: ten passages drawn from one news dataset resemble each other because
they are all news prose, whatever the feature does. A feature firing on "the"
would score near the top. So the reported score is

    cohesion_score = own_cohesion - baseline_cohesion

where baseline is the same statistic over random passages from the same corpus.
That subtracts the register the corpus shares with itself and leaves what is
attributable to the feature. A score near zero means "no more alike than any ten
passages"; a clearly positive score means the feature is selecting for something.

Nothing here calls a model or the network — it takes vectors and returns floats,
so it is testable without a GPU.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

Vector = Sequence[float]


def _l2_normalise(v: Vector) -> Optional[list]:
    """Unit-length copy of v, or None for a zero/degenerate vector.

    None rather than a zero vector: a zero-norm embedding has no direction, so
    every cosine against it is undefined. Silently returning zeros would make it
    look maximally DISSIMILAR to everything and drag the mean down as if that
    were a measurement.
    """
    total = 0.0
    for x in v:
        total += float(x) * float(x)
    norm = math.sqrt(total)
    if not norm or not math.isfinite(norm):
        return None
    return [float(x) / norm for x in v]


def mean_pairwise_cosine(vectors: Sequence[Vector]) -> Optional[float]:
    """Mean cosine similarity over every distinct pair, or None.

    None when fewer than two usable vectors survive normalisation — with one
    passage there are no pairs, and inventing 1.0 ("it matches itself") would
    rank a feature with a single stored example as maximally coherent.
    """
    usable = [u for u in (_l2_normalise(v) for v in vectors) if u is not None]
    if len(usable) < 2:
        return None

    total = 0.0
    pairs = 0
    for i in range(len(usable)):
        a = usable[i]
        for j in range(i + 1, len(usable)):
            b = usable[j]
            total += sum(x * y for x, y in zip(a, b))
            pairs += 1
    if not pairs:
        return None
    # Clamp: floating point can nudge a cosine a hair outside [-1, 1].
    return max(-1.0, min(1.0, total / pairs))


def cohesion_score(
    own_cohesion: Optional[float],
    baseline_cohesion: Optional[float],
) -> Optional[float]:
    """Own cohesion minus the corpus baseline.

    None propagates: an unmeasurable feature must not be reported as 0.0, which
    would be indistinguishable from a measured "exactly as alike as random".
    """
    if own_cohesion is None or baseline_cohesion is None:
        return None
    return own_cohesion - baseline_cohesion


def summarise(scores: dict) -> dict:
    """Distribution summary over {feature_id: score-or-None}."""
    vals = sorted(v for v in scores.values() if v is not None)
    n = len(vals)
    if not n:
        return {"n": 0, "unscored": len(scores)}
    return {
        "n": n,
        "unscored": len(scores) - n,
        "min": vals[0],
        "p25": vals[n // 4],
        "median": vals[n // 2],
        "p75": vals[(3 * n) // 4],
        "max": vals[-1],
        "mean": sum(vals) / n,
    }
