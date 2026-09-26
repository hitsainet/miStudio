"""Probe evaluation metrics (032 FR-9, FR-13 inputs).

AUROC with an interval, the ROC curve, recall and threshold at a target FPR, and
AUROC by input-length band — computed here as pure functions over scores and
labels so they can be tested against `sklearn` and against hand arithmetic
without a model, a database or a GPU.

REFUSAL IS A RESULT. Every function returns `None` with a stated reason rather
than a number it cannot support: below `MIN_PER_CLASS` examples of either class
an AUROC is noise that reads like a measurement, and this project has already
paid for a metric that reported a comfortable 0.5 instead of saying it could not
score (`detection_metrics.panel_score`, and the labeling judge's
`judge_unreliable`).

⚠ A DELIBERATE DEVIATION FROM THE TASK TEXT, RECORDED HERE. Task 1.6 says to
bootstrap "via `detection_metrics.bootstrap_ci`". That helper is a percentile
bootstrap of the **mean of a list of values**, and an AUROC is not such a mean —
it is a two-sample statistic. The only way to press it into service would be to
hand it per-positive placement values, which holds the negative sample FIXED and
yields an interval that is **too narrow**. The probe RUNG gates on this
interval's lower bound (FR-13: "CI lower bound above 0.5"), so a too-narrow
interval promotes probes that have not earned it — the dangerous direction. So
`auroc_ci` here does a proper **stratified two-sample** percentile bootstrap,
resampling positives and negatives independently, and `bootstrap_ci` is reused
where it IS the right tool: `mean_auroc_across_sets`, whose unit of resampling
is the evaluation set and whose statistic really is a mean.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .detection_metrics import bootstrap_ci

#: Below this many examples of EITHER class, nothing is scored. Matches the
#: figure task 1.6 fixes and the spirit of `detection_metrics`' own refusals.
MIN_PER_CLASS: int = 20

#: The FPR targets FR-9 always reports, beside whatever target a probe was
#: calibrated for.
STANDARD_FPR_TARGETS: Tuple[float, ...] = (0.01, 0.05, 0.10)

#: Bootstrap defaults. Seeded, so a reported interval is reproducible — the same
#: reason `detection_metrics.bootstrap_ci` is seeded.
DEFAULT_RESAMPLES: int = 2000
DEFAULT_ALPHA: float = 0.05
DEFAULT_SEED: int = 1337


@dataclass
class Refusal:
    """Why a metric was not computed. Never a substitute number."""

    reason: str
    n_positive: int
    n_negative: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "scored": False,
            "reason": self.reason,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


@dataclass
class RocPoint:
    #: None means "above every score": predict nothing positive. Not `inf`, which
    #: is not representable in JSON or jsonb (see `roc_points`).
    threshold: Optional[float]
    fpr: float
    tpr: float


@dataclass
class OperatingPoint:
    """A threshold, and what it actually achieves — never the target it was asked for.

    `threshold is None` means no threshold admits anything inside the budget:
    recall 0 at FPR 0. That is an honest operating point and a common one at a 1%
    budget with 20 negatives, so it must survive serialisation rather than
    arriving as `inf`.
    """

    target_fpr: float
    threshold: Optional[float]
    realised_fpr: float
    recall: float


def _split(scores: Sequence[float], labels: Sequence[int]) -> Tuple[List[float], List[float]]:
    """Positives and negatives, REFUSING any other label.

    The first version filtered for `== 1` and `== 0` and silently dropped
    everything else, so a dataset mapped to positive / negative / **excluded**
    (BR-001) could lose rows between the input and the counts with nothing
    reporting it — an AUROC over 80 rows presented as one over 100. Excluded rows
    must be dropped by the CALLER, where the exclusion is recorded.

    ⚠ VALIDATE THE RAW VALUE, NOT `int(y)`. The first version of this very guard
    compared `{int(y) for y in labels}` against `{0, 1}`, which TRUNCATES before it
    validates: `0.9` became a negative and `1.5` a positive, both silently, so a
    probability or a 1-5 rating handed in by mistake was graded as a label instead
    of refused. `int()` is applied only after every value has been proven to be
    exactly 0 or 1.
    """
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores against {len(labels)} labels")
    unexpected: List[str] = []
    seen: set = set()
    for y in labels:
        if y == 0 or y == 1:            # 0, 1, 0.0, 1.0, False, True, np.int64(1)
            continue
        shown = repr(y)
        if shown not in seen:
            seen.add(shown)
            unexpected.append(shown)
    if unexpected:
        raise ValueError(
            f"labels must be 0 or 1, found {sorted(unexpected)}. Rows a probe dataset maps to "
            f"'excluded' are dropped by the caller, where that exclusion is recorded — "
            f"dropping them here would make the reported counts disagree with the input"
        )
    positive = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    negative = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
    return positive, negative


def check_scoreable(
    scores: Sequence[float], labels: Sequence[int], *, minimum: int = MIN_PER_CLASS
) -> Optional[Refusal]:
    """The one place the class-count rule lives, so every metric refuses alike."""
    positive, negative = _split(scores, labels)
    if len(positive) < minimum or len(negative) < minimum:
        return Refusal(
            reason=(
                f"fewer than {minimum} examples of a class "
                f"({len(positive)} positive, {len(negative)} negative): an AUROC here is "
                f"noise that reads like a measurement"
            ),
            n_positive=len(positive),
            n_negative=len(negative),
        )
    return None


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """AUROC by the rank (Mann–Whitney U) identity, WITH tie correction.

    Ties get the average rank, which makes a tied pair contribute 0.5 — the
    definition sklearn uses. Counting ties as wins instead would inflate every
    AUROC on a probe whose scores saturate, which is exactly when a probe looks
    best and is most likely to be wrong.

    Returns None when either class is empty. Does NOT apply MIN_PER_CLASS: this
    is the raw statistic, and a caller that needs the refusal asks for it — the
    bootstrap resamples this function thousands of times and must not re-check.
    """
    positive, negative = _split(scores, labels)
    if not positive or not negative:
        return None

    values = positive + negative
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0          # 1-based, averaged over the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1

    rank_sum_positive = sum(ranks[: len(positive)])
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> Optional[Dict[str, float]]:
    """Stratified two-sample percentile bootstrap for AUROC.

    Positives and negatives are resampled INDEPENDENTLY, each to its own size, so
    the interval reflects variation in both samples. See the module docstring for
    why `detection_metrics.bootstrap_ci` is not used here: it bootstraps the mean
    of a value list, and pressing AUROC into that shape holds the negatives fixed
    and returns an interval that is too narrow — while the probe rung gates on the
    lower bound.
    """
    positive, negative = _split(scores, labels)
    if len(positive) < 2 or len(negative) < 2:
        return None

    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(resamples):
        pos = [positive[rng.randrange(len(positive))] for _ in range(len(positive))]
        neg = [negative[rng.randrange(len(negative))] for _ in range(len(negative))]
        value = auroc(pos + neg, [1] * len(pos) + [0] * len(neg))
        if value is not None:
            draws.append(value)
    if not draws:
        return None

    draws.sort()
    low = draws[int((alpha / 2) * len(draws))]
    high = draws[min(int((1 - alpha / 2) * len(draws)), len(draws) - 1)]
    return {"low": low, "high": high, "resamples": resamples, "alpha": alpha}


def roc_points(scores: Sequence[float], labels: Sequence[int]) -> List[RocPoint]:
    """The ROC curve, one point per distinct threshold, plus both endpoints.

    Thresholds are `score >= threshold` predicts positive. The curve starts at
    (0, 0) — a threshold above every score — so a caller plotting it does not
    have to know to add the origin.
    """
    positive, negative = _split(scores, labels)
    if not positive or not negative:
        return []

    pairs = sorted(
        ((float(s), int(y)) for s, y in zip(scores, labels)), key=lambda p: p[0], reverse=True
    )
    n_pos, n_neg = len(positive), len(negative)
    # THE ORIGIN'S THRESHOLD IS None, NOT inf. "Above every score" is a real
    # operating point (predict nothing positive), but `float("inf")` cannot be
    # serialised: Starlette renders JSON with `allow_nan=False` and jsonb rejects
    # it, so a report endpoint would 500 — and the 1%-FPR point lands here
    # whenever the top-scoring row is negative, which is common at 20 negatives.
    points = [RocPoint(threshold=None, fpr=0.0, tpr=0.0)]

    tp = fp = 0
    i = 0
    while i < len(pairs):
        threshold = pairs[i][0]
        # Consume every row at this threshold together: splitting a tie group
        # would invent operating points the threshold cannot distinguish.
        while i < len(pairs) and pairs[i][0] == threshold:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        points.append(RocPoint(threshold=threshold, fpr=fp / n_neg, tpr=tp / n_pos))

    return points


def best_point_within(points: Sequence[RocPoint], target_fpr: float) -> Optional[RocPoint]:
    """THE SELECTION RULE, as a pure function over points — deliberately extracted.

    Highest recall among the points inside the budget; ties go to the LOWER FPR,
    because spending fewer false positives for the same recall is strictly better
    and means a reported FPR never overstates what was spent.

    ⚠ WHY THIS IS ITS OWN FUNCTION. Inlined in `operating_point_at_fpr`, the
    `-p.fpr` half of the key was NOT TESTABLE: `roc_points` happens to emit points
    in increasing FPR order and `max` keeps the first maximal element, so deleting
    `-p.fpr` left the whole metrics suite green. The rule was therefore delivered by
    an accident of ordering in another function, and a later change to `roc_points`
    could have silently reversed it. Extracted and unit-tested over a SHUFFLED,
    hand-built point list, the tie-break is pinned by a test that fails when it is
    removed — this repo's recorded answer to a guard satisfied by the wrong
    occurrence.

    Returns None only when no point is inside the budget, which for a real ROC
    curve cannot happen (the origin has FPR 0).
    """
    feasible = [p for p in points if p.fpr <= target_fpr + 1e-12]
    if not feasible:
        return None
    # A UNIQUE key over the points of one ROC curve: every point consumes at least
    # one row, so no two share both tp and fp. Uniqueness — not a third tie-break —
    # is what makes the choice deterministic; the first version documented a
    # `p.threshold` tie-break as the determinism guarantee and it was unreachable.
    return max(feasible, key=lambda p: (p.tpr, -p.fpr))


def operating_point_at_fpr(
    scores: Sequence[float], labels: Sequence[int], target_fpr: float
) -> Optional[OperatingPoint]:
    """The threshold that best uses a budget of `target_fpr`, and what it achieves.

    Among thresholds whose realised FPR is **at or under** the target, take the
    highest recall. TIES GO TO THE LOWER FPR: when two thresholds reach the same
    recall, the one that spends fewer false positives is strictly better, and
    choosing it means a reported FPR never overstates what was spent. A remaining
    tie takes the HIGHER threshold, so the result is deterministic rather than
    dependent on sort order.

    There is always a feasible point — the threshold above every score has FPR 0
    — so this returns None only when a class is missing.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    points = roc_points(scores, labels)
    if not points:
        return None

    best = best_point_within(points, target_fpr)
    if best is None:
        return None
    return OperatingPoint(
        target_fpr=target_fpr,
        threshold=best.threshold,
        realised_fpr=best.fpr,
        recall=best.tpr,
    )


def recall_at_fpr(
    scores: Sequence[float], labels: Sequence[int], target_fpr: float
) -> Optional[float]:
    point = operating_point_at_fpr(scores, labels, target_fpr)
    return None if point is None else point.recall


def threshold_at_fpr(
    scores: Sequence[float], labels: Sequence[int], target_fpr: float
) -> Optional[float]:
    """The threshold to serve at `target_fpr`. None means FIRE ON NOTHING.

    ⚠ IT RAISES RATHER THAN RETURNING None FOR "COULD NOT SCORE". Those are
    opposite instructions and the first version returned None for both: a caller
    writing the natural `threshold or 0.0` turned "no threshold is inside the
    budget, so predict nothing positive" into a threshold of 0.0, which on a
    centred score fires on roughly half of all inputs — a monitor inverted from
    silent to noisy by a defaulting expression. A missing class is a caller error
    (check `check_scoreable` first); an empty firing set is a real result.
    """
    point = operating_point_at_fpr(scores, labels, target_fpr)
    if point is None:
        raise ValueError(
            "cannot give a threshold: one class is missing, so there is no ROC curve. "
            "Call check_scoreable() first — a None RETURN from this function means "
            "'no threshold is inside the budget', which is the opposite instruction"
        )
    return point.threshold


def length_bands(
    scores: Sequence[float],
    labels: Sequence[int],
    lengths: Sequence[int],
    *,
    minimum: int = MIN_PER_CLASS,
) -> List[Dict[str, object]]:
    """AUROC per quartile of token count (FR-9).

    Quartiles of the OBSERVED lengths, so the bands describe this evaluation set
    rather than an assumed distribution. A band that cannot be scored says so
    with its counts — the reason length bands exist is that performance varies
    with length, and a silently dropped band hides exactly that.

    ⚠ THE FLOOR APPLIES PER BAND, SO BANDS NEED ~160 BALANCED ROWS TO SCORE AT
    ALL. Four bands × 20 per class × 2 classes = 160, and a review found that a
    balanced 100-row set therefore produces four refusals and no numbers. That is
    deliberate rather than an oversight: the alternative is a weaker standard for
    a band than for the set it came from, which would let a 12-row band report an
    AUROC the whole-set metric would have refused. Each entry carries the
    `minimum` it applied, so a reader can see WHY a band is unscored rather than
    inferring it, and a caller that wants band numbers on a small set must pass a
    lower `minimum` explicitly and own that choice.
    """
    if not (len(scores) == len(labels) == len(lengths)):
        raise ValueError(
            f"{len(scores)} scores, {len(labels)} labels and {len(lengths)} lengths must match"
        )
    if not scores:
        return []

    ordered = sorted(int(n) for n in lengths)
    cuts = [ordered[int(q * (len(ordered) - 1))] for q in (0.25, 0.5, 0.75)]

    def band_of(n: int) -> int:
        return sum(1 for c in cuts if n > c)

    bands: List[Dict[str, object]] = []
    for index in range(4):
        rows = [
            (float(s), int(y), int(n))
            for s, y, n in zip(scores, labels, lengths)
            if band_of(int(n)) == index
        ]
        band_scores = [r[0] for r in rows]
        band_labels = [r[1] for r in rows]
        band_lengths = [r[2] for r in rows]
        entry: Dict[str, object] = {
            "band": index,
            "n": len(rows),
            "min_length": min(band_lengths) if band_lengths else None,
            "max_length": max(band_lengths) if band_lengths else None,
        }
        entry["minimum"] = minimum
        if not rows:
            # AN EMPTY BAND IS NOT AN UNDER-SAMPLED ONE. `check_scoreable` would
            # say "fewer than 20 examples of a class (0 positive, 0 negative)",
            # which reads as "this band exists and is too small to score". With
            # lengths tied at the quartile cuts — a packed corpus, where every row
            # is the same width — three of the four bands are EMPTY, and the
            # previous message reported them as three under-sampled bands. A
            # reader would go looking for more data for bands that cannot exist.
            entry.update({
                "scored": False,
                "reason": "no examples fall in this length band",
                "n_positive": 0,
                "n_negative": 0,
            })
            bands.append(entry)
            continue
        refusal = check_scoreable(band_scores, band_labels, minimum=minimum)
        if refusal is not None:
            entry.update(refusal.as_dict())
        else:
            entry.update({"scored": True, "auroc": auroc(band_scores, band_labels)})
        bands.append(entry)
    return bands


def evaluate(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    name: str = "",
    out_of_distribution: bool = False,
    lengths: Optional[Sequence[int]] = None,
    target_fpr: Optional[float] = None,
    minimum: int = MIN_PER_CLASS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    """One evaluation set's full result, or a refusal with its reason (FR-9).

    `name` and `out_of_distribution` IDENTIFY the set and are echoed into the
    result — including into a refusal, because which set could not be scored is
    exactly what a reader needs. They exist because `probe_monitor_rung`'s
    `from_evaluations` reads them: without them the two functions did not compose,
    every set graded as in-distribution, and rungs 2 and 3 were unreachable from
    real output while a test hand-built a shape this function never produced.
    """
    refusal = check_scoreable(scores, labels, minimum=minimum)
    if refusal is not None:
        refused = refusal.as_dict()
        refused.update({"name": name, "out_of_distribution": bool(out_of_distribution)})
        return refused

    positive, negative = _split(scores, labels)
    targets = list(STANDARD_FPR_TARGETS)
    if target_fpr is not None and target_fpr not in targets:
        targets.append(target_fpr)

    # A LIST, NOT A DICT KEYED BY A ROUNDED STRING. `f"{target:.2f}"` maps 0.011
    # and 0.01 to the same key, so a probe calibrated at 1.1% silently overwrote
    # the 1% point FR-9 mandates — and the surviving entry then reported
    # `target_fpr: 0.011` under the key "0.01". A list cannot collide, and each
    # entry carries its own target.
    operating: List[Dict[str, object]] = []
    for target in sorted(set(targets)):
        point = operating_point_at_fpr(scores, labels, target)
        if point is not None:
            operating.append({
                "target_fpr": point.target_fpr,
                "threshold": point.threshold,
                "realised_fpr": point.realised_fpr,
                "recall": point.recall,
            })

    return {
        "name": name,
        "out_of_distribution": bool(out_of_distribution),
        "scored": True,
        "auroc": auroc(scores, labels),
        "ci": auroc_ci(scores, labels, resamples=resamples, seed=seed),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "roc": [{"threshold": p.threshold, "fpr": p.fpr, "tpr": p.tpr} for p in roc_points(scores, labels)],
        "operating_points": operating,
        "length_bands": length_bands(scores, labels, lengths, minimum=minimum) if lengths is not None else None,
    }


def mean_auroc_across_sets(per_set: Dict[str, Optional[float]]) -> Dict[str, object]:
    """The mean AUROC over evaluation sets, with `bootstrap_ci` doing the interval.

    THIS is where `detection_metrics.bootstrap_ci` belongs: the unit of
    resampling is the evaluation SET, the statistic really is a mean of a list of
    values, and the generalization claim is about sets rather than rows — the
    same argument its own docstring makes for resampling features rather than
    items.

    Unscored sets are EXCLUDED and counted, never imputed. Imputing 0.5 would
    drag a mean toward chance and read as a measured result.
    """
    scored = {name: value for name, value in per_set.items() if value is not None}
    if not scored:
        return {
            "scored": False,
            "reason": "no evaluation set produced a score",
            "sets_scored": 0,
            "sets_total": len(per_set),
            "mean_auroc": None,
            "ci": None,
            "ci_unavailable": "no evaluation set produced a score",
        }
    values = list(scored.values())
    interval = bootstrap_ci(values)
    # `bootstrap_ci` returns None below two values, so a single scored set yields
    # a mean WITH NO INTERVAL. `scored: True` is still correct — the mean is a
    # real number — but a consumer reading `ci["low"]` after checking `scored`
    # raises, so the absence is named rather than left as a bare None to be
    # rediscovered at the call site.
    return {
        "scored": True,
        "mean_auroc": sum(values) / len(values),
        "ci": interval,
        "ci_unavailable": None if interval is not None else (
            f"an interval over evaluation sets needs at least 2 scored sets; "
            f"{len(scored)} of {len(per_set)} scored"
        ),
        "sets_scored": len(scored),
        "sets_total": len(per_set),
        "unscored": sorted(set(per_set) - set(scored)),
    }
