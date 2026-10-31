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

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..schemas.evidence_ladder import (
    PROBE_RUNG_LANGUAGE,
    PROBE_RUNG_NEXT_STEP,
    ProbeRung,
)

#: The bound a CI lower limit must CLEAR. Strictly greater — see `_clears`.
CHANCE = 0.5

#: The view role of an everyday-text negatives set (ENH-006 D82). Ordinary prose with no positives
#: of ANY behaviour, so it is behaviour-agnostic and reusable across probes and models.
EVERYDAY_TEXT_ROLE = "everyday_text"

#: The two knobs of the everyday-text check. These are the CODE DEFAULTS; the values in force are
#: the runtime settings `everyday_text_fpr_factor` and `everyday_text_min_rows` (Settings →
#: Environment), which `recompute_rung` reads through `runtime_settings.effective()`.
DEFAULT_EVERYDAY_FPR_FACTOR = 2.0
DEFAULT_EVERYDAY_MIN_ROWS = 1000

#: The exact phrase a reader is told when no everyday-text set has been run (D82).
EVERYDAY_NOT_CHECKED = "not yet checked on everyday text — evaluate on an everyday-text set"

#: The view role of a NONSENSE-TEXT negatives set (ENH-006 D84): word salad, random characters, one token
#: repeated, garbled code, digit noise, mixed scripts — text with no meaning and nothing of any behaviour.
#: Behaviour-agnostic like `everyday_text`. Generated, seeded, by `services/nonsense_text.py`.
NONSENSE_TEXT_ROLE = "nonsense_text"

#: The roles whose evaluations feed a false-alarm check rather than the AUROC sets (negatives only).
NEGATIVES_ONLY_TEXT_ROLES = (EVERYDAY_TEXT_ROLE, NONSENSE_TEXT_ROLE)

#: Code defaults of the nonsense-text check; the values in force are the runtime settings
#: `nonsense_text_fpr_factor` and `nonsense_text_min_rows`.
DEFAULT_NONSENSE_FPR_FACTOR = 2.0
DEFAULT_NONSENSE_MIN_ROWS = 1000

#: The exact phrase a reader is told when no nonsense-text set has been run (D84).
NONSENSE_NOT_CHECKED = "not yet checked on nonsense text — evaluate on a nonsense-text set"

#: Code default of the catch-rate half of rung 2 (ENH-006 D84); in force: runtime setting `rung2_min_recall`.
DEFAULT_RUNG2_MIN_RECALL = 0.5


@dataclass(frozen=True)
class _TextKind:
    """The words one negatives-only text check speaks in (everyday text, nonsense text)."""

    role: str
    subject: str            # "everyday text"
    article_set: str        # "an everyday-text set"
    not_checked: str
    advice: str             # what the probe is doing wrong, in the failure's next step


EVERYDAY_KIND = _TextKind(
    EVERYDAY_TEXT_ROLE, "everyday text", "an everyday-text set", EVERYDAY_NOT_CHECKED,
    "the probe fires on ordinary prose; retrain with everyday text among its negatives, or cut its bar on "
    "everyday text, then evaluate again",
)
NONSENSE_KIND = _TextKind(
    NONSENSE_TEXT_ROLE, "nonsense text", "a nonsense-text set", NONSENSE_NOT_CHECKED,
    "the probe fires on text with no meaning at all; retrain with nonsense text among its negatives, then "
    "evaluate again",
)


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


@dataclass(frozen=True)
class EverydaySet:
    """One everyday-text evaluation, measured AT THE SHIPPED BAR (ENH-006 D82).

    `rows` is the number of rows scored (every one a negative), `false_alarms` how many scored at
    or above the shipped threshold (`>=`, the comparison `calibrate` and miLLM use). Both None when
    the set could not be measured — `note` then says why, and the set does not count.
    """

    name: str
    rows: Optional[int]
    false_alarms: Optional[int]
    threshold: Optional[float] = None
    note: Optional[str] = None


@dataclass
class EverydayCheck:
    """The verdict of the everyday-text false-alarm check, as rung 2 needs it.

    `status`: ``passed`` (at least one counted set, and every counted set within the allowance),
    ``not_checked`` (no set of at least `min_rows` rows was measured at the shipped bar) or
    ``too_many_false_alarms`` (a counted set exceeded it). `sets` carries every set with its
    numbers, its allowance and the plain sentence the report shows.
    """

    status: str
    reason: str
    next_step: Optional[str]
    target_fpr: Optional[float]
    factor: float
    min_rows: int
    sets: List[Dict[str, Any]] = field(default_factory=list)
    #: ENH-006 D84: the same check runs on nonsense text; `kind` is the view role it was measured on.
    kind: str = EVERYDAY_TEXT_ROLE

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def label(self) -> str:
        return "nonsense text" if self.kind == NONSENSE_TEXT_ROLE else "everyday text"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "passed": self.passed,
            "reason": self.reason,
            "next_step": self.next_step,
            "target_fpr": self.target_fpr,
            "factor": self.factor,
            "min_rows": self.min_rows,
            "allowed_rate": (self.factor * self.target_fpr) if self.target_fpr is not None else None,
            "sets": [dict(s) for s in self.sets],
        }


def _pct(rate: float) -> str:
    """0.02 -> "2%", 1.0 -> "100%", 0.0123 -> "1.23%"."""
    return f"{rate * 100:.3g}%"


def _allowed_count(rows: int, factor: float, target_fpr: float) -> int:
    """The most false alarms `rows` everyday rows may carry: floor(factor × target × rows).

    Compared as an INTEGER count, so 24 of 1,200 at 2% passes exactly rather than by the grace of
    a float rounding the right way.
    """
    return int(math.floor(factor * target_fpr * rows + 1e-9))


def everyday_check(
    sets: Sequence[EverydaySet],
    *,
    target_fpr: Optional[float],
    factor: float = DEFAULT_EVERYDAY_FPR_FACTOR,
    min_rows: int = DEFAULT_EVERYDAY_MIN_ROWS,
) -> EverydayCheck:
    """THE ONE RULE for the everyday-text half of rung 2 (ENH-006 D82, 2026-10-29).

    ⚠ WHY IT EXISTS. A probe's out-of-distribution sets were other carriers of its concept, never
    plain prose, so nothing measured whether it fires on ordinary text. Measured on LFM2.5 with run
    `pmr_bbe9a3608802` armed in miLLM: random everyday words, correct answer "absent" every time —
    the layer-0 `mean` probe fired on 280/280 and the layer-1 `attention` probe on 140/280, both at
    rung 2 ("detects on unseen tasks").

    A set COUNTS when it was measured at the SHIPPED bar and has at least `min_rows` rows. Rung 2
    needs at least one counted set, and EVERY counted set's false-alarm rate at the shipped bar to
    be at most `factor × target_fpr` (the same "every set" rule the out-of-distribution half uses).
    A smaller set is shown with its numbers and named as not counting.
    """
    return _text_check(EVERYDAY_KIND, sets, target_fpr=target_fpr, factor=factor, min_rows=min_rows)


def nonsense_check(
    sets: Sequence[EverydaySet],
    *,
    target_fpr: Optional[float],
    factor: float = DEFAULT_NONSENSE_FPR_FACTOR,
    min_rows: int = DEFAULT_NONSENSE_MIN_ROWS,
) -> EverydayCheck:
    """THE ONE RULE for the nonsense-text half of rung 2 (ENH-006 D84, 2026-10-29).

    ⚠ WHY IT EXISTS. The everyday-text check (D82) PASSED the layer-0 `mean` probe of `pmr_bbe9a3608802`
    at 0.35% on OpenWebText, while the same probe fired on 280/280 of miLLM's benchmark chats — random
    common words strung together. Prose and word salad are different failure surfaces, so rung 2 needs
    both. The rule is D82's, on a `nonsense_text` view: at least `min_rows` rows measured at the SHIPPED
    bar, and every counted set's false alarms at most `factor × target_fpr`.
    """
    return _text_check(NONSENSE_KIND, sets, target_fpr=target_fpr, factor=factor, min_rows=min_rows)


def _text_check(
    kind: _TextKind,
    sets: Sequence[EverydaySet],
    *,
    target_fpr: Optional[float],
    factor: float,
    min_rows: int,
) -> EverydayCheck:
    """The false-alarm check on a negatives-only text set, in `kind`'s words (D82 everyday, D84 nonsense)."""
    entries: List[Dict[str, Any]] = []
    counted: List[Dict[str, Any]] = []
    for s in sets:
        entry: Dict[str, Any] = {
            "name": s.name, "rows": s.rows, "false_alarms": s.false_alarms,
            "threshold": s.threshold, "counts": False, "allowed": None, "fpr": None,
            "within_allowance": None, "note": s.note, "sentence": None,
        }
        if s.rows is None or s.false_alarms is None:
            entry["sentence"] = f"On {kind.subject} ({s.name}): not measured — {s.note or 'no result'}"
            entries.append(entry)
            continue
        rows = int(s.rows)
        alarms = int(s.false_alarms)
        entry["fpr"] = (alarms / rows) if rows else None
        if target_fpr is not None and rows:
            allowed = _allowed_count(rows, factor, float(target_fpr))
            entry["allowed"] = allowed
            entry["within_allowance"] = alarms <= allowed
        if rows < int(min_rows):
            entry["note"] = (f"{rows:,} rows, fewer than the {int(min_rows):,} the check needs, "
                             f"so it does not count")
        elif target_fpr is None:
            entry["note"] = "the probe has no target false-alarm rate, so no allowance can be set"
        else:
            entry["counts"] = True
            counted.append(entry)
        if alarms == rows and rows:
            sentence = f"Fired on 100% of {kind.subject} ({rows:,} rows)"
        else:
            sentence = f"On {kind.subject}: {alarms:,} false alarm{'s' if alarms != 1 else ''} in {rows:,}"
            if entry["allowed"] is not None:
                sentence += f" (allowed up to {entry['allowed']:,})"
        if not entry["counts"] and entry["note"]:
            sentence += f" — {entry['note']}"
        entry["sentence"] = sentence
        entries.append(entry)

    allowed_rate = (factor * float(target_fpr)) if target_fpr is not None else None
    if not counted:
        return EverydayCheck("not_checked", kind.not_checked, kind.not_checked,
                             target_fpr, factor, int(min_rows), entries, kind.role)
    failing = [e for e in counted if not e["within_allowance"]]
    if failing:
        worst = max(failing, key=lambda e: e["fpr"] or 0.0)
        reason = (f"false alarms on {kind.subject} {_pct(worst['fpr'])} at the shipped bar "
                  f"(allowed ≤ {_pct(allowed_rate)}) on {worst['name']}: "
                  f"{worst['false_alarms']:,} of {worst['rows']:,} rows, allowed up to {worst['allowed']:,}")
        next_step = (f"false alarms on {kind.subject} {_pct(worst['fpr'])} at the shipped bar "
                     f"(allowed ≤ {_pct(allowed_rate)}) — {kind.advice}")
        return EverydayCheck("too_many_false_alarms", reason, next_step,
                             target_fpr, factor, int(min_rows), entries, kind.role)
    reason = f"few false alarms on {kind.subject} at the shipped bar: " + "; ".join(
        f"{e['name']} {e['false_alarms']:,} of {e['rows']:,} (allowed up to {e['allowed']:,})" for e in counted
    )
    return EverydayCheck("passed", reason, None, target_fpr, factor, int(min_rows), entries, kind.role)


def everyday_false_alarms(scores: Sequence[float], labels: Sequence[int], threshold: Optional[float]
                          ) -> Optional[Dict[str, Any]]:
    """Rows and false alarms of an everyday-text evaluation at `threshold`, or None with no bar.

    Only label-0 rows are counted (an everyday view maps nothing to positive; this keeps a stray
    excluded row from being read as a negative). `>=`, as `calibrate` places the bar.
    """
    if threshold is None:
        return None
    negatives = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
    return {
        "threshold": float(threshold),
        "rows": len(negatives),
        "false_alarms": sum(1 for s in negatives if s >= float(threshold)),
    }


@dataclass(frozen=True)
class RecallSet:
    """One counted out-of-distribution set's catch rate AT THE SHIPPED BAR (ENH-006 D84).

    `positives` is how many positive rows were scored, `caught` how many scored at or above the shipped
    threshold (`>=`, as `calibrate` and miLLM fire). Both None when the set could not be measured at the
    bar in force — `note` says why, and the set FAILS the check: an unmeasured catch rate is not a pass.
    """

    name: str
    positives: Optional[int]
    caught: Optional[int]
    threshold: Optional[float] = None
    note: Optional[str] = None


@dataclass
class CatchRateCheck:
    """The verdict of the catch-rate half of rung 2 (ENH-006 D84).

    `status`: ``passed`` (at least one out-of-distribution set, and every one catches at least
    `min_recall` of its positives at the shipped bar), ``no_sets`` (no out-of-distribution set counted),
    ``not_measured`` (a set could not be measured at the bar in force) or ``too_few_caught``.
    """

    status: str
    reason: str
    next_step: Optional[str]
    min_recall: float
    sets: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    label = "catch rate on unseen tasks"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "reason": self.reason,
            "next_step": self.next_step,
            "min_recall": self.min_recall,
            "sets": [dict(s) for s in self.sets],
        }


def _needed(positives: int, min_recall: float) -> int:
    """The fewest positives a set of `positives` must catch: ceil(min_recall × positives).

    An INTEGER comparison, so 500 of 1,000 at 50% passes exactly and 499 does not, whatever a float does.
    """
    return int(math.ceil(float(min_recall) * positives - 1e-9))


def catch_rate_check(sets: Sequence[RecallSet], *, min_recall: float = DEFAULT_RUNG2_MIN_RECALL) -> CatchRateCheck:
    """THE ONE RULE for the catch-rate half of rung 2 (ENH-006 D84, 2026-10-29).

    ⚠ WHY IT EXISTS. Rung 2 was a RANKING claim (every out-of-distribution AUROC interval above 0.5) plus,
    since D82, a false-alarm claim. Neither asks whether the probe ever FIRES on the thing. The three `max`
    probes of `pmr_bbe9a3608802` passed the everyday-text check at 0% because they fire on almost nothing:
    in-distribution AUROC ≈ 0.63, recall at their shipped bar ≤ 2.5%. The run's pick caught 10% of
    positives on unseen tasks at its bar. A monitor that misses nine in ten is not "detects on unseen tasks".

    Every set passed in is a COUNTED out-of-distribution set (the ones the AUROC half reads). Each must
    catch at least `min_recall` of its positives AT THE SHIPPED BAR — never at the set's own operating
    point, which is cut to suit the set and says nothing about what the deployed probe will do.
    """
    min_recall = float(min_recall)
    entries: List[Dict[str, Any]] = []
    for s in sets:
        entry: Dict[str, Any] = {"name": s.name, "positives": s.positives, "caught": s.caught,
                                 "threshold": s.threshold, "recall": None, "needed": None,
                                 "meets": None, "note": s.note, "sentence": None}
        if s.positives is None or s.caught is None or not s.positives:
            entry["meets"] = False
            entry["sentence"] = (f"On unseen tasks ({s.name}): catch rate not measured at the shipped bar — "
                                 f"{s.note or 'no positives were scored'}")
        else:
            positives, caught = int(s.positives), int(s.caught)
            entry["recall"] = caught / positives
            entry["needed"] = _needed(positives, min_recall)
            entry["meets"] = caught >= entry["needed"]
            entry["sentence"] = (f"On unseen tasks ({s.name}): catches {_pct(entry['recall'])} of positives at "
                                 f"the shipped bar ({caught:,} of {positives:,}; needs ≥ {_pct(min_recall)})")
        entries.append(entry)

    if not entries:
        reason = "no out-of-distribution set to measure a catch rate on"
        return CatchRateCheck("no_sets", reason, f"{reason} — evaluate on one", min_recall, entries)
    unmeasured = [e for e in entries if e["recall"] is None]
    if unmeasured:
        names = ", ".join(e["name"] for e in unmeasured)
        reason = f"catch rate on unseen tasks not measured at the shipped bar on {names}"
        return CatchRateCheck("not_measured", reason, f"{reason} — evaluate those sets again", min_recall, entries)
    failing = [e for e in entries if not e["meets"]]
    if failing:
        worst = min(failing, key=lambda e: e["recall"])
        headline = (f"catches {_pct(worst['recall'])} of positives on unseen tasks at the shipped bar "
                    f"(needs ≥ {_pct(min_recall)})")
        reason = f"{headline} on {worst['name']}: {worst['caught']:,} of {worst['positives']:,} positives"
        next_step = (f"{headline} — the probe ranks unseen tasks but rarely fires on them at its bar; retrain "
                     f"on more varied positives, then evaluate again")
        return CatchRateCheck("too_few_caught", reason, next_step, min_recall, entries)
    reason = "catches enough positives on unseen tasks at the shipped bar: " + "; ".join(
        f"{e['name']} {_pct(e['recall'])} ({e['caught']:,} of {e['positives']:,})" for e in entries
    )
    return CatchRateCheck("passed", reason, None, min_recall, entries)


def shipped_bar_recall(scores: Sequence[float], labels: Sequence[int], threshold: Optional[float]
                       ) -> Optional[Dict[str, Any]]:
    """Positives and how many fire at `threshold` (`>=`), or None with no bar. Only label-1 rows count."""
    if threshold is None:
        return None
    positives = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    return {
        "threshold": float(threshold),
        "positives": len(positives),
        "caught": sum(1 for s in positives if s >= float(threshold)),
    }


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
    #: The everyday-text check's verdict (ENH-006 D82); `None` only from a caller that predates it.
    everyday_text: Optional[Dict[str, Any]] = None
    #: ENH-006 D84: the nonsense-text check and the catch rate at the shipped bar.
    nonsense_text: Optional[Dict[str, Any]] = None
    catch_rate: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "everyday_text": dict(self.everyday_text) if self.everyday_text is not None else None,
            "nonsense_text": dict(self.nonsense_text) if self.nonsense_text is not None else None,
            "catch_rate": dict(self.catch_rate) if self.catch_rate is not None else None,
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
      detection on unseen data) — or ``everyday_text``, which counts towards the everyday-text check
      only (ENH-006 D82; `recompute_rung` routes it there, never into the AUROC sets); and
    * the view's labels answer the probe's OWN question: a specialist only on its own sub-question, and a
      main-question probe only on a view with no sub-question (the symmetric rule, operator D36, OI-1).

    `view_role` None means the view is gone. Pure: called by `recompute_rung` and by the evaluation-scores
    and probe-report serializers, so the rung and the label shown beside each evaluation cannot disagree.
    """
    if view_role is None:
        return RungEligibility(False, None, "its view no longer exists")
    if view_role in NEGATIVES_ONLY_TEXT_ROLES:
        # ENH-006 D82/D84: behaviour-agnostic, so no label question; it counts towards the rung's
        # everyday- or nonsense-text check (false alarms at the shipped bar), never as an AUROC set.
        return RungEligibility(True, None, None)
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


def stored_next_step(probe: Any) -> str:
    """The next step the ladder STORED beside this probe's rung, else the generic one for the number.

    ENH-006 D82: a probe held at rung 1 by the everyday-text check must be told "evaluate on an
    everyday-text set", not "evaluate on every out-of-distribution set" — which it already did. The
    stored step is written only by `recompute_rung` (or the D82 migration), so every surface reads
    the ladder's own words.
    """
    detail = getattr(probe, "rung_detail", None) or {}
    step = detail.get("next_step") if isinstance(detail, dict) else None
    return str(step) if step else PROBE_RUNG_NEXT_STEP[ProbeRung(int(getattr(probe, "rung", 0) or 0))]


def stored_everyday_text(probe: Any) -> Optional[Dict[str, Any]]:
    """The everyday-text verdict the last rung recompute stored, or None (never recomputed)."""
    return _stored_block(probe, "everyday_text")


def stored_nonsense_text(probe: Any) -> Optional[Dict[str, Any]]:
    """The nonsense-text verdict the last rung recompute stored (ENH-006 D84), or None."""
    return _stored_block(probe, "nonsense_text")


def stored_catch_rate(probe: Any) -> Optional[Dict[str, Any]]:
    """The catch-rate verdict the last rung recompute stored (ENH-006 D84), or None."""
    return _stored_block(probe, "catch_rate")


def _stored_block(probe: Any, key: str) -> Optional[Dict[str, Any]]:
    detail = getattr(probe, "rung_detail", None) or {}
    block = detail.get(key) if isinstance(detail, dict) else None
    return dict(block) if isinstance(block, dict) else None


def _clears(ci_low: Optional[float]) -> bool:
    """STRICTLY above chance. 0.5 exactly does not clear it.

    A lower bound sitting on 0.5 means the data cannot rule out a probe that is
    guessing, and this is the gate the whole ladder hangs on.
    """
    return ci_low is not None and ci_low > CHANCE


def compute(
    results: Sequence[SetResult],
    everyday: Optional[EverydayCheck] = None,
    *,
    nonsense: Optional[EverydayCheck] = None,
    catch_rate: Optional[CatchRateCheck] = None,
) -> RungResult:
    """The probe's rung: the highest one the evidence supports.

    Rung 2 needs FOUR things (ENH-006 D82, D84): every out-of-distribution AUROC interval above 0.5
    (with at least one set); a catch rate at the shipped bar of at least `rung2_min_recall` on every one
    of those sets (`catch_rate`, `catch_rate_check`); few false alarms on everyday text (`everyday`,
    `everyday_check`); and few false alarms on nonsense text (`nonsense`, `nonsense_check`).

    ⚠ `None` MEANS NOT CHECKED, NOT "SKIP THE CHECK": a caller that does not supply a check gets at most
    rung 1 — absence of evidence never promotes.
    """
    if everyday is None:
        everyday = everyday_check([], target_fpr=None)
    if nonsense is None:
        nonsense = nonsense_check([], target_fpr=None)
    if catch_rate is None:
        catch_rate = catch_rate_check([])
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

    # ⚠ THE EVERYDAY-TEXT GATE (ENH-006 D82). Rung 2 and 3 also need few false alarms on ordinary
    # prose at the SHIPPED bar. Applied after the AUROC rules so their reasons stay intact; the cap is
    # the highest rung below 2 the evidence supports (1 only with an in-distribution pass, as above).
    #
    # ⚠ AND THE CATCH-RATE AND NONSENSE-TEXT GATES (ENH-006 D84). Same cap, same place. The order of
    # `gates` is the order a reader is told what to do next: a probe that does not fire on the concept
    # (catch rate) has nothing to gain from a false-alarm check yet.
    gates = (catch_rate, everyday, nonsense)
    blocking = [g for g in gates if not g.passed]
    next_step = PROBE_RUNG_NEXT_STEP[rung]
    if rung >= ProbeRung.UNSEEN_TASKS and blocking:
        rung = ProbeRung.HELD_OUT if in_passed else ProbeRung.TRAINED
        for gate in blocking:
            reasons.append(f"rung 2 not reached: {gate.reason}")
        next_step = blocking[0].next_step or PROBE_RUNG_NEXT_STEP[rung]
    else:
        for gate in gates:
            if gate is catch_rate and gate.status == "no_sets":
                continue   # "no out-of-distribution set was run" is already the reason above
            reasons.append(gate.reason if gate.passed else f"{gate.label}: {gate.reason}")
        if rung == ProbeRung.HELD_OUT:
            unchecked = [g for g in (everyday, nonsense) if g.status == "not_checked"]
            failed = [g for g in gates if not g.passed and g.status not in ("not_checked", "no_sets")]
            if unchecked:
                next_step = (f"{PROBE_RUNG_NEXT_STEP[rung]}, and on "
                             + " and ".join(EVERYDAY_KIND.article_set if g is everyday else NONSENSE_KIND.article_set
                                            for g in unchecked))
            for gate in failed:
                if gate.next_step:
                    next_step = f"{next_step}; and {gate.next_step}"

    return RungResult(
        rung=rung,
        language=PROBE_RUNG_LANGUAGE[rung],
        next_step=next_step,
        everyday_text=everyday.as_dict(),
        nonsense_text=nonsense.as_dict(),
        catch_rate=catch_rate.as_dict(),
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
    everyday: Optional[EverydayCheck] = None,
    nonsense: Optional[EverydayCheck] = None,
    catch_rate: Optional[CatchRateCheck] = None,
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
    return compute(results, everyday, nonsense=nonsense, catch_rate=catch_rate)
