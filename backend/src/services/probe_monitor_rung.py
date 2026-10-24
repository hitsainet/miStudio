"""Compute a probe's rung from the evaluations actually run (032 FR-13, IDL-52).

The rule in one sentence: a probe climbs only on evidence that exists, and the
CI lower bound must be STRICTLY above 0.5 — an interval touching chance is
consistent with a detector that detects nothing.

WHAT THIS DELIBERATELY WILL NOT DO. It never infers a rung from an absence. No
out-of-distribution set run means rung 1 at best, not rung 2 "by default"; a
judge run that did not complete leaves rung 2, not rung 3 "pending". Every rung
names the evidence that earned it, so a caller can show the reason beside the
claim, which is what stops the wording drifting above the evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from ..schemas.evidence_ladder import (
    PROBE_RUNG_LANGUAGE,
    PROBE_RUNG_NEXT_STEP,
    ProbeRung,
)

#: The bound a CI lower limit must CLEAR. Strictly greater — see `_clears`.
CHANCE = 0.5


@dataclass(frozen=True)
class SetResult:
    """One evaluation set's outcome, as the rung computation needs it.

    `ci_low` is None when the set was not scored — too few rows per class, or it
    never ran. None is never treated as a pass, and an unscored IN-distribution
    set does not block rung 1 either: rung 1 asks for at least one set that
    passed, not for every set to have been scoreable.
    """

    name: str
    out_of_distribution: bool
    ci_low: Optional[float]
    judge_completed: bool = False


@dataclass
class RungResult:
    rung: ProbeRung
    language: str
    next_step: str
    reasons: List[str] = field(default_factory=list)
    in_distribution_passed: List[str] = field(default_factory=list)
    in_distribution_failed: List[str] = field(default_factory=list)
    out_of_distribution_passed: List[str] = field(default_factory=list)
    out_of_distribution_failed: List[str] = field(default_factory=list)
    judge_sets_completed: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "rung": int(self.rung),
            "language": self.language,
            "next_step": self.next_step,
            "reasons": list(self.reasons),
            "in_distribution_passed": list(self.in_distribution_passed),
            "in_distribution_failed": list(self.in_distribution_failed),
            "out_of_distribution_passed": list(self.out_of_distribution_passed),
            "out_of_distribution_failed": list(self.out_of_distribution_failed),
            "judge_sets_completed": list(self.judge_sets_completed),
        }


@dataclass(frozen=True)
class RungEligibility:
    """Whether ONE evaluation counts towards its probe's rung, and which question its view's labels answer.

    `label_question`: ``own`` — the view answers the question the probe was trained on; ``main`` — the
    view answers the behaviour's main question while the probe is a specialist (a sub-question probe);
    ``other`` — the view answers a sub-question that is not the probe's own (a main-question probe on a
    sub-question view, or a specialist on another specialist's view); ``None`` — the view is gone
    (purged), so nothing is known about it.
    """

    counts: bool
    label_question: Optional[str]
    note: Optional[str]


def rung_eligibility_of(
    view_role: Optional[str], view_sub_question: Optional[str], probe_sub_question: Optional[str],
) -> RungEligibility:
    """THE ONE RULE for which evaluations a rung is computed from (Feature 041 T22; ENH-006 D36/OI-1).

    ⚠ WITHOUT THIS, COLLECTING BLEND INPUTS SILENTLY UN-EXPORTS A FAMILY. `recompute_rung` counted every
    evaluation of a probe whatever its view: rung 2 needs EVERY out-of-distribution set to clear 0.5, so a
    specialist scored on a main-question view (which it does not answer) could demote itself — and a rung
    change invalidates the built definition. An evaluation counts only when

    * its view still exists (a purged view's evaluation is evidence nobody can re-read);
    * the view's role is ``eval`` (a ``blend_fit``, ``calibration`` or ``train`` view is not evidence of
      detection on unseen data); and
    * the view's labels answer the probe's OWN question: a specialist only on its own sub-question, and a
      main-question probe only on a view with no sub-question (the symmetric rule, operator D36, OI-1).

    `view_role` None means the view is gone. Pure: called by `recompute_rung` and by the evaluation-scores
    and probe-report serializers, so the rung and the label shown beside each evaluation cannot disagree.
    """
    if view_role is None:
        return RungEligibility(False, None, "its view no longer exists")
    if view_sub_question == probe_sub_question:
        label_question = "own"
    elif view_sub_question is None:
        label_question = "main"
    else:
        label_question = "other"
    if view_role != "eval":
        return RungEligibility(False, label_question, f"role {view_role}")
    if label_question == "main":
        return RungEligibility(False, label_question, "answers the main question")
    if label_question == "other":
        return RungEligibility(False, label_question, f"answers sub-question {view_sub_question}"
                               if view_sub_question is not None else "answers the main question")
    return RungEligibility(True, label_question, None)


def _clears(ci_low: Optional[float]) -> bool:
    """STRICTLY above chance. 0.5 exactly does not clear it.

    A lower bound sitting on 0.5 means the data cannot rule out a probe that is
    guessing, and this is the gate the whole ladder hangs on.
    """
    return ci_low is not None and ci_low > CHANCE


def compute(results: Sequence[SetResult]) -> RungResult:
    """The probe's rung: the highest one the evidence supports."""
    in_dist = [r for r in results if not r.out_of_distribution]
    out_dist = [r for r in results if r.out_of_distribution]

    in_passed = [r.name for r in in_dist if _clears(r.ci_low)]
    in_failed = [r.name for r in in_dist if not _clears(r.ci_low)]
    out_passed = [r.name for r in out_dist if _clears(r.ci_low)]
    out_failed = [r.name for r in out_dist if not _clears(r.ci_low)]
    judge_done = [r.name for r in out_dist if r.judge_completed]

    reasons: List[str] = []
    rung = ProbeRung.TRAINED

    # RUNG 1 REQUIRES AN IN-DISTRIBUTION PASS, as FR-13 says in as many words
    # ("on at least one *in-distribution* evaluation set"). The first version
    # granted it for an out-of-distribution pass too, reasoning that stronger
    # evidence should not be denied the weaker claim — but that made the phrase
    # "detects on held-out data" appear for a probe with no held-out set at all,
    # contradicting the spec, IDL-52 and the ProbeRung docstring written beside
    # it. Rung 2 does not require rung 1 to have been demonstrated separately
    # (a rung is the highest PASSED), so an OOD-only probe still reaches 2.
    if in_passed:
        rung = ProbeRung.HELD_OUT
        reasons.append("CI lower bound above 0.5 in-distribution on: " + ", ".join(sorted(in_passed)))
    elif out_passed:
        reasons.append(
            "no in-distribution set cleared 0.5, so rung 1 is not claimed; "
            "out-of-distribution passes: " + ", ".join(sorted(out_passed))
        )
    else:
        reasons.append("no evaluation set's CI lower bound is above 0.5")

    # ⚠ THE LADDER IS NOT MONOTONE IN THE EVIDENCE, AND FR-13 SAYS SO. Rung 2 is
    # "every out-of-distribution set run cleared 0.5, with at least one" — it does
    # not require rung 1. So an in-distribution set at 0.40 with an
    # out-of-distribution set at 0.95 IS rung 2 by the specification, while
    # `in_distribution_passed` is empty. That combination is incoherent evidence
    # (a detector that fails the data it was fitted on and succeeds elsewhere is
    # more likely to be reading a confound than a concept), and the previous
    # version left it INVISIBLE: nothing in the result named the failing set.
    # Denying rung 2 here would contradict FR-13, so the failure is REPORTED
    # instead — `in_distribution_failed` and a reason line, so the contradiction
    # travels beside the claim rather than being inferred from an empty list.
    if in_failed:
        reasons.append(
            "in-distribution sets that did NOT clear 0.5: "
            + ", ".join(sorted(in_failed))
            + " — read any higher rung with that in view"
        )

    if out_dist and not out_failed:
        rung = ProbeRung.UNSEEN_TASKS
        reasons.append(
            f"every out-of-distribution set run ({len(out_passed)}) cleared 0.5"
        )
        # Rung 3 needs the judge on EVERY out-of-distribution set. Checked per
        # RESULT, not by comparing name sets: de-duplicated names let three sets
        # sharing one name reach rung 3 on a single judge run, which a review
        # verified. `all(...)` over the results cannot be fooled by a repeated
        # name, and duplicate names are a caller's problem to notice, not a
        # promotion route.
        if out_dist and all(r.judge_completed for r in out_dist):
            rung = ProbeRung.JUDGE_COMPARED
            reasons.append("a judge run completed on every out-of-distribution set")
        elif judge_done:
            reasons.append(
                "a judge run completed on "
                f"{len(judge_done)} of {len(out_dist)} out-of-distribution sets, "
                "so rung 3 is not reached"
            )
    elif out_failed:
        reasons.append(
            "out-of-distribution sets that did not clear 0.5: " + ", ".join(sorted(out_failed))
        )
    else:
        reasons.append("no out-of-distribution set was run")

    return RungResult(
        rung=rung,
        language=PROBE_RUNG_LANGUAGE[rung],
        next_step=PROBE_RUNG_NEXT_STEP[rung],
        reasons=reasons,
        in_distribution_passed=sorted(in_passed),
        in_distribution_failed=sorted(in_failed),
        out_of_distribution_passed=sorted(out_passed),
        out_of_distribution_failed=sorted(out_failed),
        judge_sets_completed=sorted(judge_done),
    )


def from_evaluations(
    evaluations: Sequence[Mapping[str, object]],
    *,
    judge_completed_sets: Sequence[str] = (),
) -> RungResult:
    """`compute()` over the shape `probe_monitor_metrics.evaluate` ACTUALLY produces.

    That claim was false when first written: `evaluate` emitted no `name` and no
    `out_of_distribution`, so this function graded every set as in-distribution and
    rungs 2 and 3 were unreachable from real output — while the test named after
    this contract hand-built a dict `evaluate` never returned. `evaluate` now takes
    and echoes both fields, and `test_it_composes_with_real_evaluate_output` drives
    the two together instead of asserting against a fixture.

    Reads `ci.low` only when the set `scored`, so a refusal cannot be mistaken for
    a pass by a caller that forgot to check — the refusal carries no `ci` at all.

    ⚠ IT REFUSES DUPLICATE SET NAMES, AND THAT IS A PROMOTION GUARD, NOT TIDINESS.
    `judge_completed_sets` identifies sets BY NAME, so with two out-of-distribution
    sets both called `ood`, one judge run named `ood` marks BOTH complete and the
    probe reaches rung 3 on half the evidence. `compute()` was hardened against
    exactly this (it tests every result rather than comparing de-duplicated name
    sets) and the hole reopened here, at the only composition entry point, because
    the flag is derived from the name before `compute` ever sees it. An empty name
    counts: two unnamed sets are two sets that cannot be told apart by a judge run,
    by this function, or by a reader of the report.
    """
    judged = set(judge_completed_sets)
    names = [str(row.get("name", "")) for row in evaluations]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(
            f"evaluation set names must be unique, found {duplicated} more than once. "
            f"judge_completed_sets identifies sets by name, so duplicates let one judge "
            f"run mark several sets complete and promote a probe to rung 3 on evidence "
            f"that was never gathered"
        )
    results: List[SetResult] = []
    for row in evaluations:
        ci = row.get("ci") if row.get("scored") else None
        ci_low = None
        if isinstance(ci, Mapping):
            value = ci.get("low")
            ci_low = float(value) if isinstance(value, (int, float)) else None
        name = str(row.get("name", ""))
        results.append(
            SetResult(
                name=name,
                out_of_distribution=bool(row.get("out_of_distribution", False)),
                ci_low=ci_low,
                judge_completed=name in judged,
            )
        )
    return compute(results)
