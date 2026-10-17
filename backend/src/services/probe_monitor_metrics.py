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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    """AUROC per quartile of token count (FR-9), and — given a `threshold` — recall at it.

    ⚠ **AUROC ALONE CANNOT SEE THE FAILURE THIS FUNCTION EXISTS TO CATCH.** AUROC is
    rank-based, so a probe whose scores all drift DOWN as inputs get longer keeps a flat
    banded AUROC while losing every verdict at a fixed bar. That is not hypothetical: a
    `mean`-pooled probe on live traffic scored 27.70 on a user's first message and 7.51 on
    turn four of the same conversation — below its 7.96 threshold — while the banded AUROC
    of the same probe moved only 0.958 → 0.949. The ranking was intact the whole time; the
    calibration was not, and nothing here measured it.

    So `threshold` adds two numbers per band:

    * `recall_at_threshold` — the share of that band's POSITIVES at or above the bar. This
      is the one that collapses when a score distribution shifts with length.
    * `mean_positive_score` — the shift itself, in the units the threshold is in, so a
      reader can see the cause and not just the symptom.

    Both are reported whenever the band holds at least one positive, INCLUDING when the
    band's AUROC was refused for being under-sampled. That is deliberate and it is a
    weaker standard than this file applies elsewhere, so each entry carries `n_positive`
    beside the recall: a recall over 5 rows cannot be over-read when its denominator is
    printed next to it, and the top length band — the one that matters most here — is
    exactly the band most likely to be under-sampled. Without `threshold`, neither field
    appears at all; absent says "not measured", which `0.0` would not.

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
        # AFTER the AUROC verdict, and independent of it: a refused band still reports
        # these, because a collapsing recall in an under-sampled top band is the finding,
        # not a detail to be suppressed along with its AUROC.
        entry.update(_fixed_threshold_band(band_scores, band_labels, threshold))
        bands.append(entry)
    return bands


def _fixed_threshold_band(
    scores: Sequence[float], labels: Sequence[int], threshold: Optional[float]
) -> Dict[str, object]:
    """`recall_at_threshold` and `mean_positive_score` for one band, or `{}`.

    Extracted rather than inlined into the band loop for the reason this repo keeps
    relearning: a decision buried in a loop can only be guarded by scraping the source,
    and a source scrape matches the comment describing the code as happily as the code.
    """
    if threshold is None:
        return {}
    positive = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    negative = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
    out: Dict[str, object] = {
        "threshold_used": float(threshold),
        "n_positive_in_band": len(positive),
        "n_negative_in_band": len(negative),
    }
    if positive:
        out["recall_at_threshold"] = sum(1 for s in positive if s >= threshold) / len(positive)
        out["mean_positive_score"] = sum(positive) / len(positive)
    if negative:
        # ⚠ THE CONTROL, AND IT IS WHAT MAKES THE RECALL READABLE. A recall that rises in
        # the long band is only better detection if the negatives stayed put; if the
        # realised FPR rose with it, the whole distribution moved and the bar is simply
        # wrong for that length. Measured on the five real evaluation sets, that is the
        # common case — on `anthropic_hh_balanced` recall rises 0.315 → 0.577 across
        # length quartiles while the realised FPR rises 0.0030 → 0.0161, five times the
        # budget it was cut to spend, and AUROC moves 0.958 → 0.947 and notices nothing.
        out["fpr_at_threshold"] = sum(1 for s in negative if s >= threshold) / len(negative)
        out["mean_negative_score"] = sum(negative) / len(negative)
    return out


def _band_threshold(
    operating: Sequence[Dict[str, object]], target_fpr: Optional[float]
) -> Optional[float]:
    """The threshold of the operating point at `target_fpr`, or `None`.

    `None` propagates honestly all the way to the band entries, which then omit the recall
    fields rather than reporting a recall against a bar nobody chose. Matched on the float
    the point CARRIES rather than on a rounded key, for the reason `operating` is a list
    and not a dict: `f"{x:.2f}"` maps 0.011 and 0.01 together.
    """
    if target_fpr is None:
        return None
    for point in operating:
        if point.get("target_fpr") == target_fpr:
            threshold = point.get("threshold")
            return float(threshold) if threshold is not None else None
    return None


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
        # ⚠ THE BANDS GET THE SET'S OWN OPERATING POINT, not the probe's calibrated
        # threshold. The probe's threshold is cut on a different corpus and may not even
        # exist yet when this runs; the question here is narrower and self-contained —
        # "at whatever bar spends `target_fpr` ON THIS SET, does recall hold as inputs get
        # longer?" — and that bar is derivable from the scores in hand.
        "length_bands": length_bands(
            scores, labels, lengths, minimum=minimum, threshold=_band_threshold(operating, target_fpr)
        ) if lengths is not None else None,
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


def threshold_transfer(
    shipped_threshold: Optional[float],
    threshold_source: Optional[str],
    evaluations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """How the SHIPPED threshold behaves on each evaluation set, beside that set's own.

    ⚠ **A PROBE'S ABSOLUTE SCORE DOES NOT TRANSFER BETWEEN DISTRIBUTIONS, AND NOTHING SAID SO.**
    A linear probe ranks well within a distribution while its scale shifts between them. Measured
    on the first shipped probe (Llama-3.1-8B, L11, mean): the five evaluation sets' own 1%-FPR
    thresholds spanned **24 points**, from -5.44 on `toolace_balanced` to 18.77 on
    `mental_health_balanced`. The shipped threshold, calibrated on chat, was 11.91 — above every
    score `toolace` produces (its maximum is 1.17, so the probe could never fire there) and below
    `mental_health`'s, where it fires on 43% of low-stakes rows.

    That probe reported a single honest number, AUROC 0.8841, and a single threshold. Read
    together they suggest one detector with one operating point. They are not wrong; they are
    incomplete in a way that only shows up per set.

    This is NOT a fix. There is no threshold that serves every distribution, and calibrating on
    the one being served — which the run now does — is the correct response. What was missing is
    that a reader could not see the spread without assembling it by hand from five ROC curves.

    `unreachable` is the sharp case: the shipped threshold exceeds every score the set produces,
    so recall there is exactly zero however good the ranking is.
    """
    per_set: List[Dict[str, Any]] = []
    own_thresholds: List[float] = []

    for evaluation in evaluations:
        metrics = evaluation.get("metrics") or {}
        if not metrics.get("scored"):
            continue
        name = metrics.get("name")
        roc = [p for p in (metrics.get("roc") or []) if p.get("threshold") is not None]
        points = {
            round(float(p["target_fpr"]), 4): p
            for p in (metrics.get("operating_points") or [])
            if p.get("target_fpr") is not None
        }
        own = points.get(0.01, {}).get("threshold")
        if own is not None:
            own_thresholds.append(float(own))

        entry: Dict[str, Any] = {
            "name": name,
            "auroc": metrics.get("auroc"),
            "own_threshold_at_1pct": own,
            "max_score": max((float(p["threshold"]) for p in roc), default=None),
        }

        if shipped_threshold is not None and roc:
            at_or_above = [p for p in roc if float(p["threshold"]) >= float(shipped_threshold)]
            if at_or_above:
                nearest = min(at_or_above, key=lambda p: float(p["threshold"]))
                entry["recall_at_shipped"] = nearest.get("tpr")
                entry["fpr_at_shipped"] = nearest.get("fpr")
                entry["unreachable"] = False
            else:
                # Every score in the set is below the shipped threshold.
                entry["recall_at_shipped"] = 0.0
                entry["fpr_at_shipped"] = 0.0
                entry["unreachable"] = True
        per_set.append(entry)

    spread = (max(own_thresholds) - min(own_thresholds)) if len(own_thresholds) > 1 else None
    unreachable = [e["name"] for e in per_set if e.get("unreachable")]

    caution: Optional[str] = None
    if unreachable:
        caution = (
            f"the shipped threshold is above every score on {', '.join(map(str, unreachable))}, "
            f"so recall there is zero however well the probe ranks"
        )
    elif spread is not None and shipped_threshold is not None and spread > abs(shipped_threshold):
        caution = (
            f"the evaluation sets' own 1% thresholds span {spread:.2f}, wider than the shipped "
            f"threshold itself — this probe's absolute score does not transfer between these "
            f"distributions, so one threshold behaves very differently on each"
        )

    return {
        "shipped_threshold": shipped_threshold,
        "threshold_source": threshold_source,
        "per_set": per_set,
        "own_threshold_spread": spread,
        "unreachable_sets": unreachable,
        "caution": caution,
    }


def validation_caveat(val_metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the probe's `val_auroc` is, said plainly, or `None` when there is none.

    ⚠ ONE VALIDATION SPLIT DOES THREE JOBS: it ranks the layers, it picks the epoch (the max over
    up to 400), and it is then reported as the probe's validation AUROC. Not leakage — the split
    is honest — but each selection makes the surviving number optimistic, and nothing
    distinguished it from a held-out estimate.

    The two appear together and invite the wrong reading: Stage 1 recorded 0.9982 in-distribution
    beside 0.8841 out of distribution, which looks like a probe that collapses off its training
    distribution. It is one upper bound beside one measurement.
    """
    if not val_metrics or val_metrics.get("val_auroc") is None:
        return None
    return {
        "val_auroc": val_metrics.get("val_auroc"),
        "selection_maximised": bool(val_metrics.get("val_auroc_is_selection_maximised", True)),
        "selected_over": ["epoch", "layer"],
        "note": val_metrics.get("val_auroc_note") or (
            "maximum over the epochs and layers this same validation split selected; "
            "optimistic by an unmeasured amount"
        ),
        "held_out_alternative": (
            "an evaluation view marked in_distribution is scored like any other set and is "
            "never touched by training"
        ),
    }

#: Length bands for a threshold that varies with input length. Four, matching `length_bands`,
#: because the measurement that motivated this showed 2 and 4 working (+0.035, +0.039 mean OOD
#: AUROC) and 8 degrading (+0.012) — fewer negatives per band, noisier quantiles.
DEFAULT_LENGTH_BANDS = 4


def length_band_decisions(
    scores: Sequence[float],
    lengths: Sequence[int],
    *,
    target_fpr: float,
    n_bands: int = DEFAULT_LENGTH_BANDS,
    global_threshold: Optional[float] = None,
) -> Optional[List[Dict[str, object]]]:
    """A threshold per ABSOLUTE token-length band, cut from negatives alone.

    ⚠ **WHY THIS EXISTS.** A probe's score drifts with input length, in a direction that
    depends on the corpus, so one constant threshold is miscalibrated at every length but the
    one it was cut at. Measured on the shipped `mean` probe across five out-of-distribution
    sets: realised FPR reaching 5.4x its budget on `anthropic_hh_balanced`'s longest quartile,
    and recall falling 0.500 -> 0.297 on `mental_health_balanced`. Correcting for length
    recovers +0.039 / +0.022 / +0.017 mean OOD AUROC at layers 11 / 16 / 21, against five null
    controls (random bands, same sizes) that all came back NEGATIVE.

    ⚠ **BOUNDARIES ARE ABSOLUTE TOKEN COUNTS, NOT QUANTILES.** They are CHOSEN as quantiles of
    the calibration corpus, because that is what puts equal evidence in each band — but they
    are EXPORTED as token counts, because a consumer sees one request at a time and cannot
    compute a quantile of anything. The last band is open-ended (`max_tokens: None`) and the
    first starts at 0, so no request can fall outside the table.

    ⚠ **A BAND THAT CANNOT AFFORD THE QUANTILE SAYS SO.** Placing a `target_fpr` quantile needs
    at least `1/target_fpr` negatives — 100 at 1%. A band below that carries
    `threshold_source: "global"` and the run's single threshold, rather than a number computed
    from four samples. That is the difference between a wide band and a fabricated one.

    Returns `None` when no band could be cut at all, which says "not attempted" rather than
    describing a constant threshold as a varying one.
    """
    if not scores or len(scores) != len(lengths) or n_bands < 1:
        return None
    order = sorted(range(len(scores)), key=lambda i: int(lengths[i]))
    ordered_lengths = [int(lengths[i]) for i in order]

    # Cut points at quantiles of the OBSERVED lengths, de-duplicated: a corpus of uniform
    # width (packed blocks, say) yields one band rather than four identical ones.
    cuts: List[int] = []
    for q in range(1, n_bands):
        value = ordered_lengths[int(q * (len(ordered_lengths) - 1) / n_bands)]
        if value not in cuts:
            cuts.append(value)

    bounds: List[Tuple[int, Optional[int]]] = []
    low = 0
    for cut in cuts:
        bounds.append((low, cut))
        low = cut + 1
    bounds.append((low, None))

    out: List[Dict[str, object]] = []
    for lo, hi in bounds:
        band = [
            float(s) for s, n in zip(scores, lengths)
            if int(n) >= lo and (hi is None or int(n) <= hi)
        ]
        entry: Dict[str, object] = {
            "min_tokens": lo,
            "max_tokens": hi,
            "n_negatives": len(band),
            "target_fpr": float(target_fpr),
        }
        affordable = len(band) >= (1.0 / target_fpr) if target_fpr > 0 else False
        if affordable:
            ranked = sorted(band, reverse=True)
            allowed = int(target_fpr * len(ranked))
            threshold = float(ranked[allowed - 1]) if allowed > 0 else None
            entry["threshold"] = threshold
            entry["threshold_source"] = "band"
            entry["realised_fpr"] = (
                sum(1 for s in band if threshold is not None and s >= threshold) / len(band)
            )
        else:
            # Honest fallback: this band gets the run's single threshold, and says so.
            entry["threshold"] = global_threshold
            entry["threshold_source"] = "global"
            entry["realised_fpr"] = (
                sum(1 for s in band if global_threshold is not None and s >= global_threshold)
                / len(band)
            ) if band else None
        out.append(entry)

    # ⚠ DROP EMPTY BANDS, THEN RE-OPEN THE LAST ONE. A corpus of uniform width yields a
    # trailing band holding no negatives at all, and a band with `n_negatives: 0` is not a
    # wide band — it is a row of the table with nothing behind it, which a reader would take
    # for evidence. Re-opening the survivor preserves the invariant that matters: every
    # possible length falls in exactly one band.
    out = [entry for entry in out if int(entry["n_negatives"]) > 0]
    if not out:
        return None
    out[-1]["max_tokens"] = None
    return out


def threshold_for_length(
    bands: Optional[Sequence[Dict[str, object]]], n_tokens: int, fallback: Optional[float]
) -> Optional[float]:
    """The threshold a request of `n_tokens` is judged against.

    The ONE place the lookup lives, so miStudio's calibration, its evaluation and any consumer
    cannot disagree about which band a length falls in. Falls back to the single threshold when
    no band table exists — which is what every probe exported before 2026-10-01 carries, and
    what a consumer that ignores the block will do anyway.
    """
    band = band_for_length(bands, n_tokens)
    if band is None:
        return fallback
    value = band.get("threshold")
    return float(value) if value is not None else fallback


def band_for_length(
    bands: Optional[Sequence[Dict[str, object]]], n_tokens: int
) -> Optional[Dict[str, object]]:
    """The band a request of `n_tokens` falls in, or `None` when there is no table or no match.

    Split out of `threshold_for_length` so a caller can say WHICH bar it applied, not just the
    number — and kept as the single place the band decision is made, so naming the band and
    applying it can never disagree. `threshold_for_length` is built on this, not beside it.
    """
    if not bands:
        return None
    for band in bands:
        lo = int(band.get("min_tokens") or 0)
        hi = band.get("max_tokens")
        if n_tokens >= lo and (hi is None or n_tokens <= int(hi)):
            return band
    return None
