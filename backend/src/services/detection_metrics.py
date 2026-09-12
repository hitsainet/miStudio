"""Pure metrics for labeling detection scoring. No I/O, no DB, no LLM.

Detection scoring asks: given ONLY a feature's label, can a judge pick out which
passages activate that feature? A good label scores well above chance; a vague
one scores at chance. That makes label quality rankable, so prompt templates can
be compared instead of read.

Everything here is a pure function so the whole refusal surface — the part that
decides when a number is NOT reportable — is unit-testable without a judge.

Three stances are load-bearing and are copied deliberately from
`manifest_service.reproduction_verdict`, which refuses to report
`within_tolerance=True` when nothing overlapped:

* Scoring nothing is not scoring. Zero scored features yields `None`, never 0.5.
* Absence is not evidence. A missing score is omitted, never imputed — imputing
  0.5 for an unparseable judge reply makes a broken judge look like a mediocre
  label, which is the most expensive lie this module could tell.
* A difference smaller than the panel can resolve is not a difference. Every
  comparison reports the minimum detectable effect alongside the delta.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

# Balanced accuracy of a coin, and of both degenerate judges ("all 1" / "all 0").
CHANCE = 0.5

MIN_FEATURES_FOR_VERDICT = 8
"""Below this a paired comparison has too little to say. NOTE: this is a floor,
not a safeguard — a bootstrap over identical deltas never straddles zero at ANY
n, so the real guard is MIN_MEANINGFUL_DELTA below."""

MIN_MEANINGFUL_DELTA = 0.02
"""The smallest balanced-accuracy difference worth calling a difference.

This is the guard that actually works. Requiring only "the interval excludes
zero" certified a uniform +0.001 across 8 features as `candidate_better`, because
resampling identical values reproduces them exactly — the interval was
[0.001, 0.001] and never contained zero. More features do not help; the interval
is degenerate by construction. An effect must clear an absolute floor as well as
a statistical one.
"""

# 2.80 = z(0.975) + z(0.80): the two-sided 5% / 80%-power constant for a paired
# t-style test. Used to report what a panel COULD have detected.
_ZERO_VARIANCE_TOL = 1e-12
"""Below this, treat the spread as zero.

`sd == 0.0` is an exact float comparison against an ACCUMULATED sum, so whether
it holds depends on the platform's rounding. Thirty identical deltas gave
exactly 0.0 on one machine and 5.6e-17 on CI, which sent the same comparison
down two different branches and produced a different verdict for identical
inputs. Any real balanced-accuracy spread is many orders of magnitude above this.
"""

_MDE_Z = 2.80


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def tpr(self) -> Optional[float]:
        """Recall on activating passages. None when there were no positives."""
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def tnr(self) -> Optional[float]:
        """Recall on non-activating passages. None when there were no negatives."""
        d = self.tn + self.fp
        return self.tn / d if d else None

    @property
    def positive_rate(self) -> Optional[float]:
        """Fraction of items the judge called activating.

        The dominant failure of a vague label is "say 1 to everything", which
        balanced accuracy correctly scores 0.5 — but the DIAGNOSIS lives here.
        It is the difference between "make the label more specific" and "make it
        more inclusive", so it is always reported.
        """
        return (self.tp + self.fp) / self.n if self.n else None

    @property
    def balanced_accuracy(self) -> Optional[float]:
        """(TPR + TNR) / 2.

        Not plain accuracy and not F1: with a fixed 50/50 mix this equals
        accuracy, but it maps BOTH degenerate judges to exactly 0.5 regardless of
        the mix, which is the property the gate depends on. Returns None if
        either class is absent — a score over one class is not balanced anything.
        """
        tpr, tnr = self.tpr, self.tnr
        if tpr is None or tnr is None:
            return None
        return (tpr + tnr) / 2.0

    @property
    def mcc(self) -> Optional[float]:
        """Matthews correlation. Secondary single number; catches skew BA smooths."""
        num = (self.tp * self.tn) - (self.fp * self.fn)
        den = math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn)
            * (self.tn + self.fp) * (self.tn + self.fn)
        )
        return num / den if den else None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            **asdict(self),
            "tpr": self.tpr,
            "tnr": self.tnr,
            "positive_rate": self.positive_rate,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.mcc,
        }


def confusion(predictions: Sequence[int], truth: Sequence[int]) -> Confusion:
    """Build a confusion matrix from aligned binary vectors.

    Raises on a length mismatch rather than zipping to the shorter one. A
    misaligned vector is worse than a missing one: it silently scrambles ground
    truth and produces a plausible-looking ~0.5 that nobody questions.
    """
    if len(predictions) != len(truth):
        raise ValueError(
            f"prediction/truth length mismatch: {len(predictions)} vs {len(truth)}; "
            f"a misaligned vector would score at chance and look like a bad label"
        )
    tp = fp = tn = fn = 0
    for p, t in zip(predictions, truth):
        if t:
            tp += 1 if p else 0
            fn += 0 if p else 1
        else:
            fp += 1 if p else 0
            tn += 0 if p else 1
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def is_degenerate(predictions: Sequence[int], truth: Sequence[int]) -> bool:
    """True when the judge answered the same thing for every item AND it could
    have done otherwise.

    `truth` is not optional in spirit. A constant answer is only evidence of a
    degenerate judge when the batch actually contained both classes: a batch of
    ten positives SHOULD be answered all-1, and a CORRECT judge does exactly
    that. Judging degeneracy from the predictions alone rejected perfect judges
    whenever items were not interleaved — which nothing guaranteed, because the
    shuffle helper existed and was called nowhere.

    `truth` is REQUIRED. It was briefly optional "for compatibility", but there
    were no other callers to be compatible with, and the default reinstated the
    exact bug the parameter exists to fix — the natural call `is_degenerate(preds)`
    would silently reject correct judges again.
    """
    if len(predictions) <= 1 or len(set(predictions)) != 1:
        return False
    return len(set(truth)) > 1


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _stdev(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> Optional[Dict[str, float]]:
    """Percentile bootstrap over FEATURES, not items.

    Resampling features is what handles the correlation batching introduces: all
    items in one prompt share a single generation, so their errors are not
    independent. Item-level resampling would yield an interval roughly 2-3x too
    narrow and manufacture significance that isn't there.

    Seeded, so a reported interval is reproducible.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(int((1 - alpha / 2) * resamples), resamples - 1)]
    return {"low": lo, "high": hi, "resamples": resamples, "alpha": alpha}


def panel_score(per_feature: Dict[str, Optional[float]]) -> Dict[str, object]:
    """Aggregate per-feature balanced accuracies into a panel result.

    Unweighted mean over SCORED features: the feature is the unit of
    generalization, and a pooled item-level mean would let a feature that
    happened to yield more usable negatives dominate the panel.

    Features whose score is None are counted as unscored and EXCLUDED — never
    imputed. If nothing scored, the result is None with a reason attached, not
    a comfortable-looking 0.5.
    """
    scored = {k: v for k, v in per_feature.items() if v is not None}
    if not scored:
        return {
            "scored": False,
            "balanced_accuracy_mean": None,
            "features_scored": 0,
            "features_total": len(per_feature),
            "ci": None,
            "reason": "no feature produced a usable score; scoring nothing is not scoring",
        }
    values = list(scored.values())
    return {
        "scored": True,
        "balanced_accuracy_mean": _mean(values),
        "features_scored": len(values),
        "features_total": len(per_feature),
        "ci": bootstrap_ci(values),
        "reason": None,
    }


def minimum_detectable_effect(deltas: Sequence[float]) -> Optional[float]:
    """The smallest paired difference this panel could have resolved.

    Reported on every comparison so a gap smaller than the panel's resolution is
    visibly not a result, rather than being read as a narrow win.
    """
    sd = _stdev(deltas)
    if sd is None or not deltas:
        return None
    if sd <= _ZERO_VARIANCE_TOL:
        # Every delta identical. The arithmetic gives 0.0, which would be
        # published as "this panel can resolve a difference of 0.000" — a claim
        # of infinite resolution from a sample that showed no variation at all.
        # There is no evidence here about resolution, so there is no number.
        return None
    return _MDE_Z * sd / math.sqrt(len(deltas))


def compare_panels(
    baseline: Dict[str, Optional[float]],
    candidate: Dict[str, Optional[float]],
    *,
    seed: int = 1337,
) -> Dict[str, object]:
    """Paired comparison of two trials over the same panel.

    Pairing is what makes ~30 features enough: both variants saw identical
    features, passages and judge, so everything except the prompt cancels.

    Only features scored on BOTH sides are compared. A verdict is issued only
    when the bootstrap interval on the mean delta excludes zero; otherwise the
    answer is `indistinguishable`, carrying the minimum effect the panel could
    have detected so the user knows what the null actually means.
    """
    overlap = [
        k for k in baseline
        if k in candidate and baseline[k] is not None and candidate[k] is not None
    ]

    # One shape for EVERY return path. The dropout counters were added to the
    # success branch only, so a consumer reading them got a KeyError precisely
    # when the comparison was refused — i.e. exactly when it most needed to know
    # how many features had been discarded.
    totals = {
        "baseline_total": len(baseline),
        "candidate_total": len(candidate),
        "dropped": len(set(baseline) | set(candidate)) - len(overlap),
    }

    if not overlap:
        return {
            "verdict": None,
            "compared": 0,
            "mean_delta": None,
            "ci": None,
            "minimum_detectable_effect": None,
            "wins": 0, "losses": 0, "ties": 0,
            **totals,
            "reason": "no overlapping scored features to compare; "
                      "comparing nothing is not comparing",
        }

    deltas = [candidate[k] - baseline[k] for k in overlap]  # type: ignore[operator]
    ci = bootstrap_ci(deltas, seed=seed)
    mde = minimum_detectable_effect(deltas)
    mean_delta = _mean(deltas) or 0.0

    if len(overlap) < MIN_FEATURES_FOR_VERDICT:
        return {
            "verdict": None,
            "compared": len(overlap),
            "mean_delta": mean_delta,
            "ci": ci,
            "minimum_detectable_effect": mde,
            "wins": sum(1 for d in deltas if d > 0),
            "losses": sum(1 for d in deltas if d < 0),
            "ties": sum(1 for d in deltas if d == 0),
            **totals,
            "reason": (
                f"only {len(overlap)} overlapping feature(s); at least "
                f"{MIN_FEATURES_FOR_VERDICT} are needed before a difference "
                f"between templates means anything"
            ),
        }

    # The effect floor is applied FIRST and unconditionally: an effect smaller
    # than anyone would act on is not a result regardless of how tight the
    # interval around it looks.
    if abs(mean_delta) < MIN_MEANINGFUL_DELTA:
        verdict = "indistinguishable"
        reason = (
            f"the mean delta ({mean_delta:+.4f}) is below the smallest difference "
            f"worth reporting ({MIN_MEANINGFUL_DELTA})"
        )
    elif mde is None:
        # Zero variance: every feature moved by exactly the same amount, so the
        # bootstrap interval is a POINT and excludes zero by construction. It
        # carries no information and must not be read as significance. The
        # effect floor above is the only evidence available, so the verdict is
        # reported WITH that caveat attached rather than with reason=None — an
        # earlier version published `candidate_better / mde: None / reason: None`,
        # which is the least informative record this function can emit.
        verdict = "candidate_better" if mean_delta > 0 else "baseline_better"
        reason = (
            f"every compared feature moved by the same amount ({mean_delta:+.4f}), "
            f"so no uncertainty interval could be estimated; the verdict rests on "
            f"the effect size alone"
        )
    elif ci["low"] > 0:
        verdict, reason = "candidate_better", None
    elif ci["high"] < 0:
        verdict, reason = "baseline_better", None
    else:
        # NOTE: the MDE is reported, never used as the test. Comparing an
        # OBSERVED effect against an a-priori minimum detectable effect is a
        # power calculation misapplied as a significance test: MDE is 2.80 SE
        # while the interval excludes zero at about 1.96 SE, so an earlier
        # version discarded every genuine effect in that 43% band — verified,
        # a +0.046 improvement with a 95% interval of [0.011, 0.082] was
        # reported "indistinguishable".
        verdict = "indistinguishable"
        reason = (
            f"the interval on the mean delta includes zero; this panel could "
            f"resolve a difference of about {mde:.3f} balanced-accuracy points"
        )

    return {
        "verdict": verdict,
        "compared": len(overlap),
        "mean_delta": _mean(deltas),
        "ci": ci,
        "minimum_detectable_effect": mde,
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
        # Dropout is NOT random: a feature disappears precisely when one variant
        # failed on it. Without these counts a template that simply gave up on
        # the hardest third of the panel looks like a winner on the easy rest.
        **totals,
        "reason": reason,
    }
