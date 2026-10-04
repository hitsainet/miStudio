"""032 task 1.9 — the probe ladder's boundaries and its language.

THE BOUNDARY THAT MATTERS: a CI lower bound of **exactly 0.5 does not pass**. An
interval touching chance is consistent with a detector that detects nothing, and
every rung above 0 hangs on this one comparison. It is tested at 0.5, at one
float step either side, and at the obvious values.

THE LANGUAGE IS ALSO A TEST. IDL-52 forbids "causal", "safe", "guarantee" and
"validated" in probe wording, because a detector may not borrow a circuit's
words, and miLLM mirrors these strings verbatim wherever a probe is armed. A
grep over the language dictionaries enforces it, so the rule survives a rewrite.
"""
import math
import re
from pathlib import Path

import pytest

from src.schemas.evidence_ladder import (
    PROBE_FORBIDDEN_WORDS,
    PROBE_RUNG_LANGUAGE,
    PROBE_RUNG_NEXT_STEP,
    EvidenceRung,
    ProbeRung,
    probe_rung_language,
    probe_rung_next_step,
)
from src.services.probe_monitor_rung import CHANCE, SetResult, compute, from_evaluations

REPO = Path(__file__).resolve().parents[3]


def _in(name="dev", ci_low=0.7):
    return SetResult(name=name, out_of_distribution=False, ci_low=ci_low)


def _out(name="ood", ci_low=0.7, judge=False):
    return SetResult(name=name, out_of_distribution=True, ci_low=ci_low, judge_completed=judge)


# ── the 0.5 boundary ──────────────────────────────────────────────────────────

class TestTheChanceBoundaryIsStrict:

    def test_exactly_one_half_does_NOT_pass(self):
        """The case the task names. An interval touching chance is not evidence."""
        assert compute([_in(ci_low=0.5)]).rung is ProbeRung.TRAINED

    def test_one_float_step_above_one_half_passes(self):
        assert compute([_in(ci_low=math.nextafter(0.5, 1.0))]).rung is ProbeRung.HELD_OUT

    def test_one_float_step_below_one_half_does_not(self):
        assert compute([_in(ci_low=math.nextafter(0.5, 0.0))]).rung is ProbeRung.TRAINED

    def test_below_chance_does_not_pass(self):
        assert compute([_in(ci_low=0.2)]).rung is ProbeRung.TRAINED

    def test_an_unscored_set_is_not_a_pass(self):
        assert compute([_in(ci_low=None)]).rung is ProbeRung.TRAINED

    def test_the_bound_is_the_documented_constant(self):
        assert CHANCE == 0.5


# ── rung 0 ────────────────────────────────────────────────────────────────────

class TestRungZero:

    def test_no_evaluations_at_all_is_rung_zero(self):
        result = compute([])
        assert result.rung is ProbeRung.TRAINED
        assert "no evaluation set's CI lower bound is above 0.5" in result.reasons[0]

    def test_it_names_the_next_step_rather_than_only_the_rung(self):
        assert "in-distribution" in compute([]).next_step


# ── rung 1 ────────────────────────────────────────────────────────────────────

class TestRungOne:

    def test_one_in_distribution_pass_is_enough(self):
        assert compute([_in(ci_low=0.6)]).rung is ProbeRung.HELD_OUT

    def test_a_failing_set_beside_a_passing_one_still_reaches_rung_one(self):
        result = compute([_in("a", 0.4), _in("b", 0.9)])
        assert result.rung is ProbeRung.HELD_OUT
        assert result.in_distribution_passed == ["b"]

    def test_rung_one_requires_an_IN_DISTRIBUTION_pass(self):
        """FR-13 says "on at least one *in-distribution* evaluation set", and the
        language is "detects on held-out data". The first version granted rung 1
        for an out-of-distribution pass too, so that phrase appeared for a probe
        with no held-out set at all."""
        result = compute([_out("ood", 0.9), _out("ood2", 0.4)])
        assert result.rung is ProbeRung.TRAINED
        assert any("rung 1 is not claimed" in r for r in result.reasons)

    def test_an_ood_only_probe_that_passes_everything_still_reaches_rung_two(self):
        """A rung is the highest PASSED, so rung 2 does not require rung 1 to have
        been demonstrated separately."""
        assert compute([_out(ci_low=0.8)]).rung is ProbeRung.UNSEEN_TASKS

    def test_rung_one_is_the_ceiling_when_no_ood_set_ran(self):
        result = compute([_in(ci_low=0.9), _in("second", 0.95)])
        assert result.rung is ProbeRung.HELD_OUT
        assert "no out-of-distribution set was run" in result.reasons


# ── rung 2 ────────────────────────────────────────────────────────────────────

class TestRungTwo:

    def test_every_ood_set_must_pass(self):
        assert compute([_out("a", 0.7), _out("b", 0.8)]).rung is ProbeRung.UNSEEN_TASKS

    def test_one_failing_ood_set_blocks_it(self):
        """With an in-distribution pass to stand on, a failing OOD set drops the
        probe to rung 1 — it detects on held-out data and not on unseen tasks."""
        result = compute([_in("dev", 0.8), _out("a", 0.7), _out("b", 0.5)])
        assert result.rung is ProbeRung.HELD_OUT
        assert result.out_of_distribution_failed == ["b"]

    def test_without_an_in_distribution_pass_a_failing_ood_set_leaves_rung_zero(self):
        """Rung 1 needs an in-distribution set (FR-13), so there is nothing to fall
        back to. The `reasons` are what distinguish this from a probe that was never
        evaluated at all — which is why they exist."""
        result = compute([_out("a", 0.7), _out("b", 0.5)])
        assert result.rung is ProbeRung.TRAINED
        assert result.out_of_distribution_passed == ["a"]
        assert result.out_of_distribution_failed == ["b"]

    def test_an_unscored_ood_set_blocks_it_too(self):
        """A set that could not be scored is not a set that passed."""
        result = compute([_in("dev", 0.8), _out("a", 0.9), _out("b", None)])
        assert result.rung is ProbeRung.HELD_OUT
        assert "b" in result.out_of_distribution_failed

    def test_at_least_one_ood_set_is_required(self):
        """"every set run passed" must not be vacuously true with zero sets."""
        assert compute([_in(ci_low=0.99)]).rung is ProbeRung.HELD_OUT

    def test_the_reason_says_how_many_cleared(self):
        result = compute([_out("a", 0.7), _out("b", 0.8)])
        assert any("every out-of-distribution set run (2)" in r for r in result.reasons)


# ── rung 3 ────────────────────────────────────────────────────────────────────

class TestRungThree:

    def test_a_judge_run_on_every_ood_set_reaches_it(self):
        result = compute([_out("a", 0.7, judge=True), _out("b", 0.8, judge=True)])
        assert result.rung is ProbeRung.JUDGE_COMPARED
        assert result.judge_sets_completed == ["a", "b"]

    def test_a_judge_run_on_SOME_ood_sets_does_not(self):
        result = compute([_out("a", 0.7, judge=True), _out("b", 0.8, judge=False)])
        assert result.rung is ProbeRung.UNSEEN_TASKS
        assert any("1 of 2 out-of-distribution sets" in r for r in result.reasons)

    def test_a_judge_run_cannot_lift_a_probe_that_failed_rung_two(self):
        """Rung 3 is "rung 2 PLUS a judge run", not an alternative route."""
        result = compute([_out("a", 0.4, judge=True)])
        assert result.rung is ProbeRung.TRAINED

    def test_duplicate_set_names_cannot_buy_rung_three_with_one_judge_run(self):
        """Review finding: comparing de-duplicated NAME SETS let three sets sharing
        a name reach rung 3 on a single judge run. Judge completion is a property
        of each result, not of the name collection."""
        results = [
            _out("same", 0.7, judge=True),
            _out("same", 0.8, judge=False),
            _out("same", 0.9, judge=False),
        ]
        assert compute(results).rung is ProbeRung.UNSEEN_TASKS

    def test_duplicate_names_all_judged_does_reach_rung_three(self):
        results = [_out("same", 0.7, judge=True), _out("same", 0.8, judge=True)]
        assert compute(results).rung is ProbeRung.JUDGE_COMPARED

    def test_a_judge_run_on_an_in_distribution_set_does_not_count(self):
        result = compute([
            SetResult("dev", out_of_distribution=False, ci_low=0.9, judge_completed=True),
            _out("ood", 0.9, judge=False),
        ])
        assert result.rung is ProbeRung.UNSEEN_TASKS


# ── reading the metrics module's own output ────────────────────────────────────

class TestFromEvaluations:

    def test_it_composes_with_REAL_evaluate_output(self):
        """The contract, driven end to end instead of against a fixture.

        Review finding: `evaluate` emitted no `name` and no `out_of_distribution`,
        so this reader graded every set as in-distribution and rungs 2 and 3 were
        unreachable from real output — while the test named after the contract
        hand-built a dict `evaluate` never produced. `evaluate` now echoes both,
        and this drives the two functions together.
        """
        import random

        from src.services.probe_monitor_metrics import evaluate

        def separable(seed: int, n: int = 120):
            rng = random.Random(seed)
            labels = [i % 2 for i in range(n)]
            scores = [rng.gauss(2.0 if y else 0.0, 0.5) for y in labels]
            return scores, labels

        evaluations = [
            evaluate(*separable(1), name="dev", out_of_distribution=False, resamples=200),
            evaluate(*separable(2), name="ood_a", out_of_distribution=True, resamples=200),
            evaluate(*separable(3), name="ood_b", out_of_distribution=True, resamples=200),
        ]
        assert all(e["scored"] for e in evaluations), "the fixture must be separable"

        result = from_evaluations(evaluations)
        assert result.rung is ProbeRung.UNSEEN_TASKS, result.reasons
        assert result.in_distribution_passed == ["dev"]
        assert result.out_of_distribution_passed == ["ood_a", "ood_b"]

        judged = from_evaluations(evaluations, judge_completed_sets=["ood_a", "ood_b"])
        assert judged.rung is ProbeRung.JUDGE_COMPARED

    def test_a_refusal_from_real_evaluate_output_keeps_its_identity(self):
        """A refusal must still say WHICH set could not be scored."""
        from src.services.probe_monitor_metrics import evaluate

        refused = evaluate([0.9] * 5 + [0.1] * 5, [1] * 5 + [0] * 5,
                           name="tiny", out_of_distribution=True)
        assert refused["name"] == "tiny" and refused["out_of_distribution"] is True
        assert from_evaluations([refused]).rung is ProbeRung.TRAINED

    def test_a_refusal_is_not_read_as_a_pass(self):
        """`evaluate` returns no `ci` at all when it refuses; a caller that forgot
        to check `scored` must still not get a promotion."""
        evaluations = [
            {"name": "ood", "out_of_distribution": True, "scored": False,
             "reason": "fewer than 20 examples of a class"},
        ]
        assert from_evaluations(evaluations).rung is ProbeRung.TRAINED

    def test_a_scored_false_row_carrying_a_ci_is_still_not_a_pass(self):
        """Defensive: if some future caller attaches a ci to a refusal, `scored`
        wins. The refusal is the authority on whether a number means anything."""
        evaluations = [
            {"name": "ood", "out_of_distribution": True, "scored": False, "ci": {"low": 0.99}},
        ]
        assert from_evaluations(evaluations).rung is ProbeRung.TRAINED

    def test_judge_completion_is_matched_by_set_name(self):
        evaluations = [{"name": "ood", "out_of_distribution": True, "scored": True, "ci": {"low": 0.7}}]
        assert from_evaluations(evaluations, judge_completed_sets=["ood"]).rung is ProbeRung.JUDGE_COMPARED
        assert from_evaluations(evaluations, judge_completed_sets=["other"]).rung is ProbeRung.UNSEEN_TASKS


# ── the language ──────────────────────────────────────────────────────────────

class TestProbeLanguage:

    def test_every_rung_has_language_and_a_next_step(self):
        for rung in ProbeRung:
            assert PROBE_RUNG_LANGUAGE[rung].strip()
            assert PROBE_RUNG_NEXT_STEP[rung].strip()
            assert probe_rung_language(rung) == PROBE_RUNG_LANGUAGE[rung]
            assert probe_rung_next_step(rung) == PROBE_RUNG_NEXT_STEP[rung]

    @pytest.mark.parametrize("word", PROBE_FORBIDDEN_WORDS)
    def test_no_forbidden_word_appears_in_probe_language(self, word):
        """IDL-52: a detector may not borrow a circuit's words, and miLLM mirrors
        these strings verbatim."""
        for mapping, label in ((PROBE_RUNG_LANGUAGE, "language"), (PROBE_RUNG_NEXT_STEP, "next step")):
            for rung, text in mapping.items():
                assert word not in text.lower(), (
                    f"{label} for probe rung {int(rung)} contains the forbidden word "
                    f"{word!r}: {text!r}"
                )

    def test_the_forbidden_list_is_the_four_words_idl52_names(self):
        assert set(PROBE_FORBIDDEN_WORDS) == {"causal", "safe", "guarantee", "validated"}

    def test_the_grep_would_actually_catch_a_violation(self):
        """Prove the check bites before trusting its silence: the CIRCUIT ladder
        legitimately says "causally validated", so the same rule applied there
        must fail."""
        from src.schemas.evidence_ladder import RUNG_LANGUAGE

        circuit_text = " ".join(RUNG_LANGUAGE.values()).lower()
        assert any(word in circuit_text for word in PROBE_FORBIDDEN_WORDS), (
            "the forbidden-word scan found nothing even in the circuit ladder, so it "
            "is not looking at anything"
        )

    def test_probe_language_never_claims_more_than_detection(self):
        for text in PROBE_RUNG_LANGUAGE.values():
            assert "prove" not in text.lower() and "proof" not in text.lower()


# ── the ladders stay separate, and the guard stays green ───────────────────────

class TestTheTwoLaddersAreDistinct:

    def test_probe_rung_is_not_the_circuit_rung(self):
        assert ProbeRung is not EvidenceRung
        assert PROBE_RUNG_LANGUAGE != {
            ProbeRung(int(r)): text for r, text in
            __import__("src.schemas.evidence_ladder", fromlist=["x"]).RUNG_LANGUAGE.items()
        }

    def test_both_ladders_live_in_the_one_module_so_the_guard_holds(self):
        """IDL-35's no-parallel-enums guard scans every file but this one."""
        ladder = REPO / "backend" / "src" / "schemas" / "evidence_ladder.py"
        text = ladder.read_text()
        assert "class ProbeRung(IntEnum)" in text
        assert "class EvidenceRung(IntEnum)" in text

    def test_no_other_source_file_defines_a_probe_rung_enum(self):
        """The same scan the ladder's own guard runs, aimed at ProbeRung."""
        offenders = []
        for py in (REPO / "backend" / "src").rglob("*.py"):
            if py.name == "evidence_ladder.py":
                continue
            if re.search(r"class\s+\w*ProbeRung\w*\s*\((IntEnum|Enum|str,\s*Enum)", py.read_text()):
                offenders.append(str(py))
        assert offenders == [], f"parallel probe rung enums: {offenders}"


# ── round 2: the fixes, pinned ─────────────────────────────────────────────────


class TestDuplicateSetNamesCannotBuyRungThree:
    """The composition entry point re-opened the hole `compute` was hardened against.

    `compute` tests `judge_completed` per RESULT, so a repeated name cannot promote
    a probe. But `from_evaluations` DERIVES that flag from the name — `name in
    judged` — so three out-of-distribution sets all called `ood` and one judge run
    named `ood` reached rung 3 on a third of the evidence, through the only path a
    caller actually uses.
    """

    def _row(self, name, low, ood=True):
        return {"name": name, "out_of_distribution": ood, "scored": True, "ci": {"low": low}}

    def test_three_sets_sharing_one_name_are_refused(self):
        rows = [self._row("ood", 0.8) for _ in range(3)]
        rows.append(self._row("train", 0.9, ood=False))
        with pytest.raises(ValueError, match="names must be unique"):
            from_evaluations(rows, judge_completed_sets=["ood"])

    def test_the_message_names_the_duplicate_and_the_consequence(self):
        rows = [self._row("ood", 0.8), self._row("ood", 0.8)]
        with pytest.raises(ValueError) as caught:
            from_evaluations(rows)
        text = str(caught.value)
        assert "'ood'" in text and "rung 3" in text

    def test_two_UNNAMED_sets_count_as_duplicates(self):
        """`evaluate`'s `name` defaults to "", so this is the accidental case."""
        rows = [self._row("", 0.8), self._row("", 0.8)]
        with pytest.raises(ValueError, match="names must be unique"):
            from_evaluations(rows)

    def test_ONE_unnamed_set_is_still_fine(self):
        result = from_evaluations([self._row("", 0.8, ood=False)])
        assert result.rung is ProbeRung.HELD_OUT

    def test_distinctly_named_sets_need_a_judge_run_EACH(self):
        rows = [
            self._row("train", 0.9, ood=False),
            self._row("ood_a", 0.8),
            self._row("ood_b", 0.8),
            self._row("ood_c", 0.8),
        ]
        one = from_evaluations(rows, judge_completed_sets=["ood_a"])
        assert one.rung is ProbeRung.UNSEEN_TASKS
        every = from_evaluations(rows, judge_completed_sets=["ood_a", "ood_b", "ood_c"])
        assert every.rung is ProbeRung.JUDGE_COMPARED


class TestAFailedInDistributionSetIsVisibleAtEveryRung:
    """FR-13's rung 2 does not require rung 1, so the ladder is not monotone.

    An in-distribution set at 0.40 with an out-of-distribution set at 0.95 IS rung 2
    by the specification. That is incoherent evidence — a detector failing the data
    it was fitted on and succeeding elsewhere is likelier to be reading a confound —
    and the previous result left it INVISIBLE: `in_distribution_passed` was empty and
    nothing named the set that failed. Denying rung 2 would contradict FR-13, so it
    is reported instead.
    """

    def _sets(self):
        return [
            SetResult(name="held_out", out_of_distribution=False, ci_low=0.40),
            SetResult(name="unseen", out_of_distribution=True, ci_low=0.95),
        ]

    def test_the_rung_still_follows_the_specification(self):
        assert compute(self._sets()).rung is ProbeRung.UNSEEN_TASKS

    def test_the_failing_set_is_NAMED_in_the_result(self):
        result = compute(self._sets())
        assert result.in_distribution_failed == ["held_out"]
        assert result.in_distribution_passed == []

    def test_and_in_the_reasons_a_reader_sees(self):
        reasons = " ".join(compute(self._sets()).reasons)
        assert "held_out" in reasons
        assert "did NOT clear" in reasons

    def test_it_survives_serialisation(self):
        assert compute(self._sets()).as_dict()["in_distribution_failed"] == ["held_out"]

    def test_an_UNSCORED_in_distribution_set_is_reported_too(self):
        """A set that could not be scored is not a pass, so it belongs in the list —
        rung 1 asks for at least one PASS, and an absence must not read as one."""
        result = compute([
            SetResult(name="too_small", out_of_distribution=False, ci_low=None),
            SetResult(name="unseen", out_of_distribution=True, ci_low=0.95),
        ])
        assert result.in_distribution_failed == ["too_small"]

    def test_a_clean_probe_carries_an_empty_list_and_no_such_reason(self):
        result = compute([
            SetResult(name="held_out", out_of_distribution=False, ci_low=0.7),
            SetResult(name="unseen", out_of_distribution=True, ci_low=0.8),
        ])
        assert result.in_distribution_failed == []
        assert all("did NOT clear" not in r for r in result.reasons)


class TestTheReasonsAreAuditedToo:
    """The forbidden-word rule reached the two language dicts and stopped there.

    `reasons` is the text that travels with the rung to the API and the panel — the
    sentence a reader actually sees beside the claim — and nothing looked at it.
    `test_causal_language_audit.py` now globs `probe_monitor*.py` for "causal"; this
    checks the four words against the strings the module GENERATES, which a source
    scrape cannot see.
    """

    def _every_reachable_state(self):
        """Reasons from every branch, with neutral set names so a name cannot
        contaminate the scan."""
        low, high = 0.4, 0.9
        states = [
            [],
            [SetResult(name="a", out_of_distribution=False, ci_low=high)],
            [SetResult(name="a", out_of_distribution=False, ci_low=low)],
            [SetResult(name="a", out_of_distribution=False, ci_low=None)],
            [
                SetResult(name="a", out_of_distribution=False, ci_low=high),
                SetResult(name="b", out_of_distribution=True, ci_low=high),
            ],
            [
                SetResult(name="a", out_of_distribution=False, ci_low=high),
                SetResult(name="b", out_of_distribution=True, ci_low=low),
            ],
            [
                SetResult(name="a", out_of_distribution=False, ci_low=low),
                SetResult(name="b", out_of_distribution=True, ci_low=high),
            ],
            [
                SetResult(name="a", out_of_distribution=False, ci_low=high),
                SetResult(name="b", out_of_distribution=True, ci_low=high, judge_completed=True),
            ],
            [
                SetResult(name="a", out_of_distribution=False, ci_low=high),
                SetResult(name="b", out_of_distribution=True, ci_low=high, judge_completed=True),
                SetResult(name="c", out_of_distribution=True, ci_low=high),
            ],
        ]
        return [compute(state) for state in states]

    def test_every_rung_is_actually_exercised(self):
        """Prove the corpus covers the ladder before trusting its silence."""
        reached = {result.rung for result in self._every_reachable_state()}
        assert reached == set(ProbeRung), f"unexercised rungs: {set(ProbeRung) - reached}"

    @pytest.mark.parametrize("word", PROBE_FORBIDDEN_WORDS)
    def test_no_forbidden_word_reaches_a_reason(self, word):
        for result in self._every_reachable_state():
            for reason in result.reasons:
                assert word not in reason.lower(), (
                    f"rung {int(result.rung)} reason contains {word!r}: {reason!r}"
                )

    @pytest.mark.parametrize("word", PROBE_FORBIDDEN_WORDS)
    def test_nor_the_language_or_next_step_carried_in_the_result(self, word):
        for result in self._every_reachable_state():
            assert word not in result.language.lower()
            assert word not in result.next_step.lower()

    def test_the_scan_bites(self):
        """A planted reason must fail it, or the silence above means nothing."""
        planted = compute([SetResult(name="a", out_of_distribution=False, ci_low=0.9)])
        planted.reasons.append("this probe is causally validated and safe")
        assert any(w in " ".join(planted.reasons).lower() for w in PROBE_FORBIDDEN_WORDS)
