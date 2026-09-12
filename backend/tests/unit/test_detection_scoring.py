"""Detection scoring: the metric, and every path where it must REFUSE.

The refusals matter more than the arithmetic. A scorer that returns a plausible
number when it should return nothing is worse than one that crashes: it sends a
user off rewriting prompts that were already good.

Mutation controls (each must turn a test here red):

  C4  drop the length check in parse_detection_vector (pad/truncate instead)
      -> test_a_wrong_length_reply_is_rejected_not_aligned
  C5  impute CHANCE for an unparseable batch instead of skipping
      -> test_an_unparseable_judge_scores_nothing_rather_than_chance
  C6  delete the literal-oracle gate
      -> test_a_judge_that_fails_the_literal_oracle_is_unreliable
  C7  delete the null-control (mismatched-label) gate
      -> test_a_leaking_harness_is_caught_by_the_mismatched_label_control
  C8  let score_panel score anyway when the gate failed
      -> test_a_failed_gate_yields_no_scores_at_all
  C9  make panel_score return CHANCE instead of None when nothing scored
      -> test_scoring_nothing_is_not_scoring
  C10 let compare_panels issue a verdict on an empty overlap
      -> test_comparing_nothing_is_not_comparing
  C11 seed make_rng from hash() instead of blake2b
      -> test_the_shuffle_is_reproducible_across_processes
  C12 render the prime token with its <<>> marker or activation value
      -> test_rendering_leaks_neither_marker_nor_activation
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.services.detection_metrics import (
    CHANCE,
    compare_panels,
    confusion,
    is_degenerate,
    panel_score,
)
from src.services.labeling_detection_scorer import (
    DETECTION_PROMPT_VERSION,
    MAX_DEGENERATE_RATE,
    LITERAL_ORACLE_MIN_BA,
    NULL_CONTROL_MAX_BA,
    build_detection_prompt,
    make_rng,
    panel_id_for,
    parse_detection_vector,
    render_passage,
    run_gate,
    score_feature,
    score_panel,
)


# ── judges ───────────────────────────────────────────────────────────────────

def _items(n_pos=5, n_hard=3, n_easy=4):
    """A feature's scoring items, with a REAL structure to detect.

    The concept is "locomotion"; the surface token is "running". That split is
    what makes hard negatives meaningful:

      positive       -> contains the token AND the concept
      hard_negative  -> contains the TOKEN but NOT the concept
                        ("running a server" — the word, the wrong sense)
      easy_negative  -> contains neither

    A label naming the concept is right on everything. A label naming only the
    token is right on positives and easy negatives but WRONG on every hard
    negative — which is exactly the discrimination `ba_hard` exists to measure.
    The previous fixture encoded truth in a literal marker every judge matched
    on, so no test could tell a judge that reads the label from one that ignores
    it entirely.
    """
    out = []
    for i in range(n_pos):
        out.append({"text": f"she was running across the field on foot {i}",
                    "label": 1, "kind": "positive"})
    for i in range(n_hard):
        out.append({"text": f"running a background server process {i}",
                    "label": 0, "kind": "hard_negative"})
    for i in range(n_easy):
        out.append({"text": f"the treaty was signed in vienna {i}",
                    "label": 0, "kind": "easy_negative"})
    return out


def _passages(prompt: str):
    return [l.split(". ", 1)[1] for l in prompt.split("\n")
            if l and l[0].isdigit() and ". " in l]


def _explanation(prompt: str) -> str:
    for line in prompt.split("\n"):
        if line.startswith("DESCRIPTION:"):
            return line[len("DESCRIPTION:"):].strip()
    return ""


def keyword_judge(prompt: str) -> str:
    """Answers 1 iff a content word from the EXPLANATION appears in the passage.

    This is the fixture the suite was missing. Every earlier judge computed its
    answer from the passages alone, so removing `{explanation}` from the prompt —
    which destroys the entire feature — broke no test, and the gate's two
    controls were byte-identical whichever explanation they were handed.
    """
    stop = {"text", "containing", "the", "word", "a", "an", "of", "in", "and",
            "that", "this", "with", "to", "is", "are", "or", "as", "about"}
    words = {w.strip('".,()') for w in _explanation(prompt).lower().split()}
    words -= stop
    words = {w for w in words if len(w) > 3}
    return json.dumps({"labels": [
        1 if any(w in p.lower() for w in words) else 0 for p in _passages(prompt)
    ]})


CONCEPT = "Locomotion: a person travelling on foot."
TOKEN_ONLY = 'Text containing the word "running".'


def perfect_judge(prompt: str) -> str:
    """Reads the answer straight out of the passage. Deliberately a CHEAT — it
    is used only to prove the leak detector fires, never to prove capability."""
    return json.dumps({"labels": [
        1 if "across the field" in p else 0 for p in _passages(prompt)
    ]})


def always_one_judge(prompt: str) -> str:
    return json.dumps({"labels": [1] * len(_passages(prompt))})


def always_zero_judge(prompt: str) -> str:
    return json.dumps({"labels": [0] * len(_passages(prompt))})


def unparseable_judge(prompt: str) -> str:
    return "I'm not sure I can answer that."


def coin_judge(prompt: str) -> str:
    return json.dumps({"labels": [i % 2 for i in range(len(_passages(prompt)))]})


class TestTheMetricItself:

    def test_a_perfect_judge_scores_one(self):
        r = score_feature("a real description", _items(), perfect_judge)
        assert r["balanced_accuracy"] == 1.0

    def test_both_degenerate_judges_score_exactly_chance(self):
        """'Say 1 to everything' is the dominant failure of a vague label.

        Balanced accuracy must map it to exactly 0.5 — that property is what the
        gate's thresholds depend on. Plain accuracy would not.
        """
        # batch_size pinned to the item count: otherwise a short trailing batch
        # is single-class, correctly NOT counted degenerate, and the rate becomes
        # an artefact of how the fixture happens to divide.
        items = _items()
        r = score_feature("vague", items, always_one_judge, batch_size=len(items))
        assert r["balanced_accuracy"] == CHANCE
        assert r["degenerate_rate"] == 1.0
        # And the diagnosis is available, not just the score.
        assert r["confusion"]["positive_rate"] == 1.0

    def test_hard_and_easy_negatives_are_scored_separately(self):
        r = score_feature("d", _items(), perfect_judge)
        assert r["ba_hard"] is not None and r["ba_easy"] is not None

    def test_a_feature_with_no_hard_donors_reports_absence_not_zero(self):
        """Measured on L46: only 82.3% of features share their rank-1 token with
        another feature, so ~1 in 6 has NO hard-negative donor at all.

        That must read as absence. A 0.0 would drag the panel mean down and be
        indistinguishable from a label that genuinely fails on hard negatives.
        """
        items = _items(n_pos=5, n_hard=0, n_easy=5)
        r = score_feature("d", items, perfect_judge)
        assert r["ba_hard"] is None, "no hard negatives was reported as a score"
        assert r["ba_hard"] != 0.0
        # The overall score is still usable from the easy negatives.
        assert r["balanced_accuracy"] == 1.0
        assert r["ba_easy"] == 1.0


class TestRefusals:

    def test_a_wrong_length_reply_is_rejected_not_aligned(self):
        """C4. A misaligned vector scrambles ground truth and scores ~chance.

        That reads as 'mediocre label' when the real problem is the judge, so a
        short or long reply must be discarded rather than padded or truncated.
        """
        assert parse_detection_vector('{"labels":[1,0]}', 3) is None
        assert parse_detection_vector('{"labels":[1,0,1,1]}', 3) is None
        assert parse_detection_vector('{"labels":[1,0,1]}', 3) == [1, 0, 1]

    def test_an_unparseable_judge_scores_nothing_rather_than_chance(self):
        """C5. Imputing 0.5 makes a broken judge look like a mediocre label."""
        r = score_feature("d", _items(), unparseable_judge)
        assert r["balanced_accuracy"] is None, (
            "an unparseable judge produced a score; a broken judge must not be "
            "reported as a mediocre label"
        )
        assert r["balanced_accuracy"] != CHANCE
        assert r["parse_failure_rate"] == 1.0

    def test_scoring_nothing_is_not_scoring(self):
        """C9."""
        out = panel_score({"a": None, "b": None})
        assert out["scored"] is False
        assert out["balanced_accuracy_mean"] is None
        assert out["balanced_accuracy_mean"] != CHANCE
        assert "not scoring" in out["reason"]

    def test_comparing_nothing_is_not_comparing(self):
        """C10. Disjoint panels must not produce a winner."""
        out = compare_panels({"a": 0.9}, {"b": 0.4})
        assert out["verdict"] is None
        assert out["compared"] == 0
        assert "not comparing" in out["reason"]

    def test_a_score_missing_on_one_side_is_omitted_never_zeroed(self):
        out = compare_panels({"a": 0.9, "b": None}, {"a": 0.8, "b": 0.7})
        assert out["compared"] == 1

    def test_an_inconclusive_comparison_reports_what_it_could_have_detected(self):
        base = {f"f{i}": 0.70 for i in range(30)}
        cand = {f"f{i}": 0.70 + (0.01 if i % 2 else -0.01) for i in range(30)}
        out = compare_panels(base, cand)
        assert out["verdict"] == "indistinguishable"
        assert out["minimum_detectable_effect"] is not None

    def test_a_real_improvement_is_detected(self):
        base = {f"f{i}": 0.55 for i in range(30)}
        cand = {f"f{i}": 0.85 for i in range(30)}
        out = compare_panels(base, cand)
        assert out["verdict"] == "candidate_better"
        assert out["wins"] == 30


class TestTheJudgeSanityGate:

    def _controls(self):
        return [{
            "feature_id": "f1",
            "items": _items(),
            "literal_explanation": 'Text containing the word "ACTIVATES".',
            "mismatched_explanation": "Something entirely unrelated to these passages.",
        }]

    def test_a_capable_judge_passes(self):
        gate = run_gate(self._controls(), perfect_judge)
        # A perfect judge finds the literal token AND is fooled by a wrong label
        # only insofar as the wrong label genuinely does not describe the items.
        assert gate["literal_control_ba"] == 1.0
        assert "judge_unreliable" not in gate["failures"]

    def test_a_judge_that_fails_the_literal_oracle_is_unreliable(self):
        """C6. The judge could not find a token it was told to look for.

        Reporting the resulting low numbers as label quality is the most
        expensive lie available here: it sends the user rewriting good prompts.
        """
        gate = run_gate(self._controls(), coin_judge)
        assert gate["judge_reliable"] is False
        assert "judge_unreliable" in gate["failures"]
        assert "stronger judge" in (gate["reason"] or "")

    def test_a_leaking_harness_is_caught_by_the_mismatched_label_control(self):
        """C7. A MISMATCHED explanation scoring well means the answer leaked.

        Nothing else in the module catches a rendering leak, a length artefact,
        or negatives that aren't hard — and each inflates every template equally,
        which looks like success.
        """
        gate = run_gate(self._controls(), perfect_judge)
        # perfect_judge reads the answer straight out of the passage text, which
        # is exactly what a leak looks like.
        assert gate["null_control_ba"] == 1.0
        assert "harness_leakage" in gate["failures"]

    def test_an_unparseable_judge_fails_the_gate(self):
        gate = run_gate(self._controls(), unparseable_judge)
        assert gate["passed"] is False
        assert "judge_unparseable" in gate["failures"]

    def test_a_failed_gate_yields_no_scores_at_all(self):
        """C8. Not a low score — no score."""
        gate = run_gate(self._controls(), coin_judge)
        out = score_panel(
            [{"feature_id": "f1", "explanation": "d", "items": _items()}],
            perfect_judge, gate=gate,
        )
        assert out["scored"] is False
        assert out["balanced_accuracy_mean"] is None
        assert out["per_feature"] == {}

    def test_the_gate_records_its_thresholds_and_ruler_version(self):
        gate = run_gate(self._controls(), perfect_judge)
        assert gate["prompt_version"] == DETECTION_PROMPT_VERSION
        assert gate["thresholds"]["literal_oracle_min_ba"] > CHANCE


class TestDeterminism:

    def test_panel_identity_is_order_independent_and_extraction_bound(self):
        assert panel_id_for("e1", ["b", "a"]) == panel_id_for("e1", ["a", "b"])
        assert panel_id_for("e1", ["a"]) != panel_id_for("e2", ["a"])
        # The property the id EXISTS for: a different feature set is a different
        # panel. Without this, hashing only the extraction id passes both asserts
        # above, every panel in an extraction collides, and `compare` loses the
        # only thing that lets it refuse a mismatched comparison.
        assert panel_id_for("e1", ["a", "b"]) != panel_id_for("e1", ["a", "c"])
        assert panel_id_for("e1", ["a"]) != panel_id_for("e1", ["a", "b"])

    def test_the_shuffle_is_reproducible_across_processes(self):
        """C11. Python salts str hashing per process (PYTHONHASHSEED).

        A hash()-seeded RNG differs between two Celery workers, so the run stops
        being reproducible and the comparison quietly stops being paired. This
        runs a SEPARATE interpreter to prove blake2b survives the salt.
        """
        code = (
            "import sys; sys.path.insert(0,'.');"
            "from src.services.labeling_detection_scorer import make_rng;"
            "print(make_rng('pnl_x','feat_y').random())"
        )
        outs = set()
        for salt in ("0", "1", "random"):
            # INHERIT the environment and override only the hash seed. Passing
            # a bare {PATH, PYTHONHASHSEED} dict stripped VIRTUAL_ENV,
            # PYTHONPATH and the locale, so the child could not import the
            # package at all — green locally, a hard failure in CI, for a reason
            # that had nothing to do with what the test measures.
            env = {**os.environ, "PYTHONHASHSEED": salt}
            r = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, env=env,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            assert r.returncode == 0, r.stderr[-500:]
            outs.add(r.stdout.strip().splitlines()[-1])
        assert len(outs) == 1, f"RNG differs across hash seeds: {outs}"

    def test_the_prompt_never_discloses_the_mix(self):
        """Saying 'about half activate' anchors the judge to a 50% output rate.

        That inflates balanced accuracy AND hides the all-1 failure the metric
        exists to expose.
        """
        p = build_detection_prompt("d", ["a", "b", "c", "d"])
        low = p.lower()
        for leak in ("half", "50%", "equal number", "same number", "even mix",
                     "evenly", "one in two", "balanced set", "equally"):
            assert leak not in low, f"the detection prompt discloses the mix: {leak!r}"

    def test_the_pinned_instrument_is_pinned(self):
        """The scoring prompt is the ruler. A blacklist cannot pin a template —
        it only forbids nine phrasings — so the contract that actually matters
        is asserted directly: the object output shape, the count, and the order.

        Reverting to a bare `[0,1,1,0]` array is the documented historical break
        that made eleutherai_detection unusable (it contradicts both the
        json_object response_format and the "start with {" directive).
        """
        p = build_detection_prompt("some description", ["alpha", "beta", "gamma"])
        assert '{"labels": [...]}' in p, "the reply shape is no longer an object"
        assert "some description" in p, "the explanation is not in the prompt"
        assert "exactly 3 entries" in p, "the required length is not stated"
        assert "in the order shown" in p, "order is not pinned; a reordered reply "\
                                          "would silently misalign against truth"
        assert "1. alpha" in p and "2. beta" in p and "3. gamma" in p, (
            "passages are not numbered from 1 in the given order"
        )


class TestRenderingLeaksNothing:

    def test_rendering_leaks_neither_marker_nor_activation(self):
        """C12. Either one is a total answer leak — any label would score 1.0."""
        row = {
            "prefix_tokens": ["▁the", "▁quick"],
            "prime_token": "▁fox",
            "suffix_tokens": ["▁ran"],
            "max_activation": 12.34,
        }
        out = render_passage(row)
        # Exact equality: this function ADDS nothing. Asserting only "no <<"
        # passed even with a marker-wrapping mutation in place, because a
        # scrubber downstream cleaned it up — two guards masked the mutation and
        # the control survived. The scrubber is gone; this is the invariant.
        assert out == "the quick fox ran", repr(out)
        assert "<<" not in out and ">>" not in out
        assert "12.34" not in out
        assert "activation" not in out.lower()

    def test_the_prompt_carries_no_activation_value(self):
        """The caller must not prepend '(activation: 4.12)' the way the labeling
        formatter's caller does — that is a total answer leak."""
        prompt = build_detection_prompt("d", [render_passage({
            "prefix_tokens": ["\u2581a"], "prime_token": "\u2581b",
            "suffix_tokens": ["\u2581c"], "max_activation": 9.87,
        })])
        assert "9.87" not in prompt
        assert "activation" not in prompt.lower()

    def test_positives_and_negatives_are_cropped_the_same_way(self):
        """Asymmetric truncation would let LENGTH separate the classes."""
        # Tokens must carry the sentencepiece space marker, or _clean produces
        # ONE long word and `len(out.split()) <= 25` is constant-true regardless
        # of whether truncation runs at all — which is how the whole truncation
        # block survived deletion.
        long_row = {
            "prefix_tokens": [f"\u2581w{i}" for i in range(200)],
            "prime_token": "\u2581target",
            "suffix_tokens": [f"\u2581s{i}" for i in range(200)],
        }
        out = render_passage(long_row, max_tokens=20)
        assert "target" in out, "the prime token was cropped out of its own passage"
        assert len(out.split()) <= 21, (
            f"truncation did not run: {len(out.split())} words survived a "
            f"max_tokens of 20, so a long positive could be told from a short "
            f"negative by length alone"
        )


class TestRoundOneFixes:
    """Regressions for the Round 1 review findings.

    Mutation controls:
      C17 revert is_degenerate to ignore `truth`
           -> test_a_correct_judge_is_not_called_degenerate_on_a_single_class_batch
      C18 let score_panel score when gate is None
           -> test_an_absent_gate_is_not_a_passing_gate
      C19 remove the per-feature parse/coverage floor
           -> test_a_feature_scored_from_mostly_failed_batches_is_not_scored
      C20 return 0.0 from minimum_detectable_effect when sd == 0
           -> test_zero_variance_is_not_infinite_resolution
      C21 drop MIN_FEATURES_FOR_VERDICT
           -> test_no_verdict_from_a_handful_of_features
      C22 stop calling make_rng in assemble_items
           -> test_items_are_interleaved_so_a_batch_is_never_one_class
      C23 report judge_reliable=True when the judge is unparseable
           -> test_an_unparseable_judge_is_not_reported_reliable
    """

    def test_a_correct_judge_is_not_called_degenerate_on_a_single_class_batch(self):
        """C17. The bug that would have rejected every good judge.

        With the module's own defaults (10 positives, 5+5 negatives, batch 10)
        an unshuffled first batch is ALL POSITIVES. A correct judge answers
        all-1. Judging degeneracy from predictions alone flagged that, failed the
        gate, and refused the panel. The unit tests missed it because the fixture
        returned exactly one mixed batch.
        """
        from src.services.detection_metrics import is_degenerate
        assert is_degenerate([1] * 10, [1] * 10) is False, (
            "answering all-1 to an all-positive batch was called degenerate; "
            "a correct judge would be rejected"
        )
        assert is_degenerate([1] * 10, [1] * 5 + [0] * 5) is True

    def test_items_are_interleaved_so_a_batch_is_never_one_class(self):
        """C22. make_rng existed and was called nowhere, so nothing shuffled."""
        from src.services.labeling_detection_scorer import DEFAULT_BATCH_SIZE, assemble_items
        items = assemble_items(
            [{"text": f"p{i}"} for i in range(10)],
            [{"text": f"h{i}"} for i in range(5)],
            [{"text": f"e{i}"} for i in range(5)],
            panel_id="pnl_x", feature_id="feat_y",
        )
        first = [i["label"] for i in items[:DEFAULT_BATCH_SIZE]]
        assert len(set(first)) > 1, (
            f"the first batch is a single class ({first}); a correct judge would "
            f"answer it uniformly and be scored as degenerate"
        )

    def test_no_seed_produces_a_single_class_batch(self):
        """The invariant, not one lucky seed.

        A plain shuffle made a mixed batch merely LIKELY: measured at 1.42% of
        seeds for a 10/1/1 ratio, and because the seed is a pure function of
        (panel_id, feature_id) an unlucky feature failed that way permanently.
        The interleave is stratified, so this holds by construction — the
        previous test pinned exactly one (panel_id, feature_id) pair and called
        it "never".
        """
        from src.services.labeling_detection_scorer import (
            DEFAULT_BATCH_SIZE, assemble_items,
        )
        for npos, nhard, neasy in [(10, 5, 5), (10, 1, 1), (5, 1, 1), (20, 2, 2)]:
            for seed in range(300):
                items = assemble_items(
                    [{"text": f"p{j}"} for j in range(npos)],
                    [{"text": f"h{j}"} for j in range(nhard)],
                    [{"text": f"e{j}"} for j in range(neasy)],
                    panel_id=f"pnl_{seed}", feature_id=f"feat_{seed}",
                )
                first = {i["label"] for i in items[:DEFAULT_BATCH_SIZE]}
                assert len(first) > 1, (
                    f"ratio {npos}/{nhard}/{neasy} seed {seed} produced a "
                    f"single-class first batch; that feature would score None "
                    f"forever, since the seed never changes"
                )

    def test_the_interleaving_is_reproducible(self):
        from src.services.labeling_detection_scorer import assemble_items
        def build():
            return assemble_items(
                [{"text": f"p{i}"} for i in range(6)],
                [{"text": f"h{i}"} for i in range(3)],
                [{"text": f"e{i}"} for i in range(3)],
                panel_id="pnl_x", feature_id="feat_y",
            )
        assert [i["text"] for i in build()] == [i["text"] for i in build()]

    def test_an_absent_gate_is_not_a_passing_gate(self):
        """C18. Forgetting the gate argument must not yield a scored panel."""
        out = score_panel(
            [{"feature_id": "f1", "explanation": "d", "items": _items()}],
            perfect_judge, gate=None,
        )
        assert out["scored"] is False
        assert out["balanced_accuracy_mean"] is None
        assert "ungated" in (out["reason"] or "")
        # L5: the refusal branch must carry the same keys as the success branch.
        for key in ("features_scored", "features_total", "ci"):
            assert key in out, f"refusal branch omits {key!r}; consumers KeyError"

    def test_a_feature_scored_from_mostly_failed_batches_is_not_scored(self):
        """C19. Two of three batches unparseable and the third all-1 gives
        balanced accuracy 0.5 — indistinguishable from a vague label."""
        calls = {"n": 0}

        def flaky(prompt):
            calls["n"] += 1
            if calls["n"] <= 2:
                return "sorry, no"
            return always_one_judge(prompt)

        # Interleave, so the surviving batch is MIXED. Grouped items made the
        # last batch single-class, whose balanced accuracy is None for an
        # unrelated reason — the test passed with the floor deleted.
        from src.services.labeling_detection_scorer import assemble_items
        items = assemble_items(
            [{"text": f"she was running across the field {i}"} for i in range(15)],
            [{"text": f"running a background server {i}"} for i in range(8)],
            [{"text": f"the treaty was signed {i}"} for i in range(7)],
            panel_id="pnl_c19", feature_id="feat_c19",
        )
        r = score_feature(CONCEPT, items, flaky, batch_size=10)
        assert r["balanced_accuracy"] is None, (
            "a feature scored from one surviving batch out of three produced a "
            "number; it would enter the panel mean as if it were a real measurement"
        )
        assert r["parse_failure_rate"] > 0.5

    def test_zero_variance_is_not_infinite_resolution(self):
        """C20. sd == 0 gave mde 0.0 — 'this panel resolves 0.000 points'."""
        from src.services.detection_metrics import minimum_detectable_effect
        assert minimum_detectable_effect([0.3] * 30) is None
        assert minimum_detectable_effect([0.1, 0.2, 0.3]) is not None

    def test_no_verdict_from_a_handful_of_features(self):
        """C21. A bootstrap over 2 identical deltas certifies +0.001 as a win."""
        out = compare_panels({"a": 0.5, "b": 0.5}, {"a": 0.501, "b": 0.501})
        assert out["verdict"] is None
        assert "at least" in (out["reason"] or "")

    def test_an_unparseable_judge_is_not_reported_reliable(self):
        """C23. judge_reliable is affirmative; silence is not reliability."""
        gate = run_gate([{
            "feature_id": "f1", "items": _items(),
            "literal_explanation": "x", "mismatched_explanation": "y",
        }], unparseable_judge)
        assert gate["judge_reliable"] is False
        assert gate["passed"] is False

    def test_asymmetric_dropout_is_visible_in_a_comparison(self):
        """M5. A template that gives up on the hard third must not look like a
        winner on the easy rest without the dropout being reported."""
        base = {f"f{i}": 0.6 for i in range(30)}
        cand = {f"f{i}": (0.9 if i < 20 else None) for i in range(30)}
        out = compare_panels(base, cand)
        assert out["compared"] == 20
        assert out["dropped"] == 10, "silent dropout is invisible to the caller"
        assert out["baseline_total"] == 30

    def test_the_negative_ceiling_claim_is_actually_produced(self):
        """H3. The docstring's validity argument rested on a number that was
        never computed. A promise made only in prose is not a guarantee."""
        from src.services.labeling_detection_scorer import negative_ceiling
        assert negative_ceiling([{"max_activation": 3.2}, {"max_activation": 0.67}]) == 0.67
        assert negative_ceiling([]) is None


class TestTheJudgeActuallyReadsTheLabel:
    """The gap Round 1 found: every earlier judge computed its answer from the
    passages alone, so the whole feature could be gutted without a red test.

    Mutation controls:
      C24 drop {explanation} from DETECTION_PROMPT_V1
           -> test_removing_the_explanation_destroys_the_score
      C25 swap the literal and mismatched controls in run_gate
           -> test_the_two_gate_controls_are_not_interchangeable
      C26 make _subset ignore its kind filter
           -> test_a_token_only_label_scores_worse_on_hard_negatives
      C27 count degeneracy from the null control too
           -> test_a_capable_judge_passes_the_gate
    """

    NULL_EXPL = "Culinary techniques involving braising."

    def _controls(self):
        return [{
            "feature_id": "f1", "items": _items(),
            "literal_explanation": TOKEN_ONLY,
            "mismatched_explanation": self.NULL_EXPL,
        }]

    def test_removing_the_explanation_destroys_the_score(self):
        """C24. The label is the ONLY thing being tested; without it in the
        prompt the judge has nothing to go on and cannot beat chance."""
        good = score_feature(CONCEPT, _items(), keyword_judge)["balanced_accuracy"]
        blind = score_feature("", _items(), keyword_judge)["balanced_accuracy"]
        assert good == 1.0
        assert blind is not None and blind <= CHANCE, (
            f"a judge given an EMPTY explanation scored {blind}; the fixture is "
            f"reading the answer out of the passages, so removing the label from "
            f"the prompt would break nothing"
        )

    def test_the_two_gate_controls_are_not_interchangeable(self):
        """C25. Swapping them must change the gate. If it does not, neither
        control is measuring what its name says."""
        normal = run_gate(self._controls(), keyword_judge)
        swapped = run_gate([{
            "feature_id": "f1", "items": _items(),
            "literal_explanation": self.NULL_EXPL,
            "mismatched_explanation": TOKEN_ONLY,
        }], keyword_judge)
        assert normal["literal_control_ba"] != swapped["literal_control_ba"], (
            "the gate is identical with its controls swapped; it is blind to the "
            "explanation and proves nothing about the judge"
        )

    def test_a_capable_judge_passes_the_gate(self):
        """C27. Until now NO test ever produced a passing gate, so the entire
        scoring path of score_panel was unexecuted.

        This also pins the fix for a real defect: the null control's CORRECT
        answer is all-zero (its label describes nothing present), and counting
        that as degeneracy rejected capable judges.
        """
        gate = run_gate(self._controls(), keyword_judge)
        assert gate["passed"] is True, f"a capable judge was rejected: {gate}"
        assert gate["judge_reliable"] is True
        assert gate["failures"] == []
        assert gate["literal_control_ba"] >= LITERAL_ORACLE_MIN_BA
        assert gate["null_control_ba"] <= NULL_CONTROL_MAX_BA

    def test_a_non_leaking_harness_is_not_accused_of_leaking(self):
        """The absent case. Every earlier leak test used a cheating judge, so
        `harness_leakage` was only ever asserted PRESENT — the threshold could
        have been 0.95, or unconditional, and nothing would have noticed."""
        gate = run_gate(self._controls(), keyword_judge)
        assert "harness_leakage" not in gate["failures"]

    def test_the_scoring_path_runs_end_to_end(self):
        """C27 (cont). score_panel's success branch, finally reachable."""
        gate = run_gate(self._controls(), keyword_judge)
        out = score_panel(
            [{"feature_id": "f1", "explanation": CONCEPT, "items": _items()},
             {"feature_id": "f2", "explanation": CONCEPT, "items": _items()}],
            keyword_judge, gate=gate,
        )
        assert out["scored"] is True
        assert out["balanced_accuracy_mean"] == 1.0
        assert out["features_scored"] == 2
        assert out["prompt_version"] == DETECTION_PROMPT_VERSION

    def test_a_token_only_label_scores_worse_on_hard_negatives(self):
        """C26. The module's headline claim, previously untested.

        `ba_easy - ba_hard` is meant to say how much of a label's apparent
        quality is just naming the surface token. A label that names only the
        token is right on positives and easy negatives and WRONG on every hard
        negative, which shares the token in the wrong sense.
        """
        r = score_feature(TOKEN_ONLY, _items(n_pos=5, n_hard=5, n_easy=5),
                          keyword_judge, batch_size=15)
        assert r["ba_easy"] == 1.0, "the token-only label should ace easy negatives"
        assert r["ba_hard"] < r["ba_easy"], (
            f"ba_hard ({r['ba_hard']}) is not below ba_easy ({r['ba_easy']}); the "
            f"hard/easy split is not discriminating and the gap is meaningless"
        )

    def test_the_concept_label_beats_the_token_label(self):
        """The whole point of the feature: a better label must score better."""
        items = _items(n_pos=5, n_hard=5, n_easy=5)
        good = score_feature(CONCEPT, items, keyword_judge, batch_size=15)
        weak = score_feature(TOKEN_ONLY, items, keyword_judge, batch_size=15)
        assert good["balanced_accuracy"] > weak["balanced_accuracy"], (
            "a concept label did not outscore a surface-token label; detection "
            "scoring cannot rank prompt templates"
        )


class TestTheNegativeCeilingTravels:
    """C28 — a value computed and read by nobody is the same defect as a value
    promised and never computed. The module docstring says the bound "travels
    with the result"; this is what makes that sentence true.
    """

    def test_the_bound_is_echoed_on_the_success_path(self):
        r = score_feature(CONCEPT, _items(), keyword_judge, negative_ceiling_value=0.67)
        assert r["negative_ceiling"] == 0.67

    def test_the_bound_is_echoed_on_a_refusal_too(self):
        """A refusal still needs the bound: a reader deciding whether to trust a
        skipped feature needs to know what a negative meant."""
        r = score_feature(CONCEPT, _items(), unparseable_judge, negative_ceiling_value=0.67)
        assert r["balanced_accuracy"] is None
        assert r["negative_ceiling"] == 0.67

    def test_score_panel_threads_it_through_per_feature(self):
        gate = run_gate([{
            "feature_id": "f1", "items": _items(),
            "literal_explanation": TOKEN_ONLY,
            "mismatched_explanation": "Culinary techniques involving braising.",
        }], keyword_judge)
        out = score_panel([{
            "feature_id": "f1", "explanation": CONCEPT,
            "items": _items(), "negative_ceiling": 0.42,
        }], keyword_judge, gate=gate)
        assert out["per_feature"]["f1"]["negative_ceiling"] == 0.42, (
            "the panel dropped the bound; every score would be reported without "
            "the qualifier that makes it defensible"
        )


class TestRoundTwoFixes:
    """Round 2 attacked the Round 1 fixes. These pin what it found.

    Mutation controls:
      C29 let assemble_items build a single-class list
           -> test_a_single_class_panel_is_refused_at_assembly
      C30 remove **totals from either refusal branch of compare_panels
           -> test_every_comparison_branch_returns_the_same_shape
      C31 change the shuffle so the ORDER differs
           -> test_the_shuffle_order_itself_is_pinned
    """

    def test_a_single_class_panel_is_refused_at_assembly(self):
        """C29. Balanced accuracy is undefined without both classes. Failing at
        assembly names the cause; failing at scoring looks like a bad judge."""
        from src.services.labeling_detection_scorer import (
            DetectionScoringError, assemble_items,
        )
        with pytest.raises(DetectionScoringError, match="both classes"):
            assemble_items([{"text": "p"}] * 10, [], [], panel_id="p", feature_id="f")
        with pytest.raises(DetectionScoringError, match="both classes"):
            assemble_items([], [{"text": "n"}], [], panel_id="p", feature_id="f")

    def test_every_comparison_branch_returns_the_same_shape(self):
        """C30. The dropout counters were on the success branch only, so reading
        them raised KeyError exactly when a comparison had been refused."""
        branches = {
            "no_overlap": compare_panels({"a": 0.5}, {"b": 0.5}),
            "too_few": compare_panels({"a": 0.5, "b": 0.6}, {"a": 0.7, "b": 0.8}),
            "scored": compare_panels(
                {f"f{i}": 0.5 for i in range(10)},
                {f"f{i}": 0.5 + i * 0.01 for i in range(10)}),
        }
        shapes = {name: set(d) for name, d in branches.items()}
        reference = shapes["scored"]
        for name, keys in shapes.items():
            assert keys == reference, (
                f"the {name!r} branch returns a different shape; missing "
                f"{sorted(reference - keys)}"
            )
        assert branches["no_overlap"]["dropped"] == 2

    def test_the_shuffle_order_itself_is_pinned(self):
        """C31. The cross-process test pins the SEED. The reproducibility claim
        needs the resulting ORDER, which is what two trials actually compare on.

        random.shuffle's algorithm is not a documented cross-version stability
        guarantee, so this is the canary: if a Python upgrade changes it, two
        trials recorded either side of the upgrade stop being paired and this
        test says so instead of the comparison quietly drifting.
        """
        from src.services.labeling_detection_scorer import assemble_items
        items = assemble_items(
            [{"text": f"p{i}"} for i in range(4)],
            [{"text": f"h{i}"} for i in range(2)],
            [{"text": f"e{i}"} for i in range(2)],
            panel_id="pnl_fixed", feature_id="feat_fixed",
        )
        assert [i["text"] for i in items] == ['e1', 'p1', 'h0', 'p3', 'h1', 'e0', 'p0', 'p2'], (
            "the interleave order changed; trials across this change are not paired"
        )


class TestRoundTwoDeepFixes:
    """Round 2 attacked the Round 1 fixes and found several were cosmetic.

    Mutation controls:
      C32 verdict on CI alone, without the meaningful-delta floor
           -> test_a_rounding_artefact_is_not_a_winner
      C33 restore the optional `truth` default on is_degenerate
           -> test_truth_is_required_not_optional
      C34 accept a bare {"passed": True} gate
           -> test_a_hand_built_gate_cannot_authorise_scoring
      C35 drop the ruler-version binding
           -> test_a_gate_from_a_different_ruler_is_refused
      C36 count unparsed batches in the degeneracy denominator
           -> test_unparsed_batches_are_not_evidence_of_non_degeneracy
      C37 remove MIN_ITEMS_PER_CLASS
           -> test_one_negative_item_cannot_carry_a_feature
    """

    def test_a_rounding_artefact_is_not_a_winner(self):
        """C32. THE Round 1 fix that did not work.

        MIN_FEATURES_FOR_VERDICT moved the defect from n=2 to n=8; it did not
        remove it. Resampling identical deltas reproduces them exactly, so the
        interval never straddles zero at ANY n. Worse, the zero-variance MDE fix
        replaced the visibly-wrong `0.000` with `None`, and this branch set
        reason to None too — so the published record was
        `candidate_better / mde: None / reason: None`.
        """
        out = compare_panels({f"f{i}": 0.5 for i in range(8)},
                             {f"f{i}": 0.501 for i in range(8)})
        assert out["verdict"] == "indistinguishable", (
            f"a uniform +0.001 was certified as {out['verdict']!r}"
        )
        assert out["reason"], "an inconclusive verdict must say why"

    def test_a_genuine_uniform_improvement_is_still_reported(self):
        """The other side of C32: suppressing a real result would be its own bug.
        Every feature improving by exactly 0.4 has zero variance too."""
        out = compare_panels({f"f{i}": 0.5 for i in range(8)},
                             {f"f{i}": 0.9 for i in range(8)})
        assert out["verdict"] == "candidate_better"

    def test_truth_is_required_not_optional(self):
        """C33. The 'compatibility' default had no callers to be compatible with
        and silently reinstated the bug it was added to fix."""
        from src.services.detection_metrics import is_degenerate
        with pytest.raises(TypeError):
            is_degenerate([1, 1, 1])  # type: ignore[call-arg]

    def test_a_hand_built_gate_cannot_authorise_scoring(self):
        """C34. `{"passed": True}` scored a panel with no control ever run."""
        from src.services.labeling_detection_scorer import DetectionScoringError
        with pytest.raises(DetectionScoringError, match="authorises scoring"):
            score_panel([{"feature_id": "f1", "explanation": CONCEPT,
                          "items": _items()}],
                        keyword_judge, gate={"passed": True})

    def test_a_gate_from_a_different_ruler_is_refused(self):
        """C35. The module's headline stance is that the ruler is pinned. A gate
        cached under an older prompt version must not authorise scoring under a
        newer one, or the result claims provenance it does not have."""
        from src.services.labeling_detection_scorer import DetectionScoringError
        stale = {"passed": True, "literal_control_ba": 1.0, "null_control_ba": 0.5,
                 "failures": [], "prompt_version": "detection/v0"}
        with pytest.raises(DetectionScoringError, match="not comparable"):
            score_panel([{"feature_id": "f1", "explanation": CONCEPT,
                          "items": _items()}], keyword_judge, gate=stale)

    def test_unparsed_batches_are_not_evidence_of_non_degeneracy(self):
        """C36. The same dilution the parse-rate fix removed, reproduced three
        lines above it.

        Scenario: of 10 literal-control batches, 3 are unparseable and 3 of the 7
        that parsed are degenerate. The true rate among evidence is 3/7 = 0.43,
        over the 0.30 threshold. Counting unparsed batches as non-degenerate
        reports 3/10 = 0.30, and the strict `>` then lets it through — certifying
        a judge that answered uniformly on 43% of the batches where it said
        anything at all.

        This must assert the GATE's rate, not the fixture's shape: an earlier
        version checked only that the fixture produced degenerate batches, so
        the mutation survived.
        """
        from src.services.labeling_detection_scorer import assemble_items

        state = {"n": 0}

        def flaky_degenerate(prompt):
            # 10 literal batches: #1-3 unparseable, #4-6 degenerate, rest correct.
            state["n"] += 1
            n = state["n"]
            if n <= 3:
                return "no idea"
            if n <= 6:
                return always_one_judge(prompt)
            return keyword_judge(prompt)

        # 25 positives + 25 easy = 50 items in the LITERAL control (which strips
        # hard negatives) = exactly 10 batches of 5. With 3 unparsed the two
        # denominators straddle the threshold: 3/7 = 0.43 fires, 3/10 = 0.30 does
        # not (the check is a strict `>`). Any other shape and both fire, and the
        # mutation survives — which is what happened on the first attempt.
        items = assemble_items(
            [{"text": f"she was running across the field on foot {i}"} for i in range(25)],
            [{"text": f"running a background server {i}"} for i in range(5)],
            [{"text": f"the treaty was signed {i}"} for i in range(25)],
            panel_id="pnl_c36", feature_id="feat_c36",
        )
        gate = run_gate([{
            "feature_id": "f1", "items": items,
            "literal_explanation": TOKEN_ONLY,
            "mismatched_explanation": "Culinary techniques involving braising.",
        }], flaky_degenerate, batch_size=5)

        assert gate["degenerate_rate"] is not None
        assert gate["degenerate_rate"] > MAX_DEGENERATE_RATE, (
            f"degenerate_rate {gate['degenerate_rate']:.3f} was diluted below the "
            f"{MAX_DEGENERATE_RATE} threshold by counting unparsed batches as "
            f"evidence of non-degeneracy"
        )
        assert "judge_degenerate" in gate["failures"]

    def test_one_negative_item_cannot_carry_a_feature(self):
        """C37. 3 positives + 1 negative passed a bare count floor of 4, but TNR
        from one item is quantised to {0,1} and swings balanced accuracy by 0.5 —
        at full weight in an unweighted panel mean."""
        from src.services.labeling_detection_scorer import assemble_items
        items = assemble_items(
            [{"text": "she was running across the field on foot 0"}] * 3,
            [{"text": "running a background server 0"}], [],
            panel_id="p", feature_id="f",
        )
        r = score_feature(CONCEPT, items, keyword_judge, batch_size=10)
        assert r["balanced_accuracy"] is None, (
            "a feature with a single negative item produced a score; one "
            "judgement would move it by half"
        )
        assert "at least" in (r["reason"] or "")

    def test_a_thin_control_blames_the_control_not_the_judge(self):
        """MEDIUM-1. The gate reported `judge_unparseable` while its own
        parse_failure_rate said 0.0 — a self-contradictory verdict that sent the
        operator to buy a bigger model over a negative-sampling problem."""
        thin = [{
            "feature_id": "f1",
            "items": _items(n_pos=5, n_hard=5, n_easy=0),
            "literal_explanation": TOKEN_ONLY,
            "mismatched_explanation": "Culinary techniques involving braising.",
        }]
        gate = run_gate(thin, keyword_judge)
        assert gate["passed"] is False
        assert "control_unscorable" in gate["failures"]
        assert "judge_unparseable" not in gate["failures"]
        assert gate["parse_failure_rate"] == 0.0
        assert "control itself is too thin" in (gate["reason"] or "")


class TestRoundThreeFixes:
    """Round 3 found that the Round 2 interleave fix created a worse defect.

    Mutation controls:
      C44 make the class pattern a pure function of (n_pos, n_neg) again
           -> test_the_ground_truth_pattern_is_not_guessable
      C45 accept a truthy non-bool gate['passed']
           -> test_a_truthy_non_boolean_gate_is_refused
      C46 drop judge_degenerate from the judge_reliable exclusion set
           -> test_a_degenerate_judge_is_not_reported_reliable
      C47 use the MDE as a significance test again
           -> test_a_significant_effect_is_not_discarded_by_the_mde
    """

    def test_the_ground_truth_pattern_is_not_guessable(self):
        """C44. Deterministic interleaving made every batch of every feature of
        every trial share one truth vector — at the module defaults,
        [1,0,1,0,1,0,1,0,1,0]. A judge with any alternation bias then scores 1.0
        on every label under every template, and two genuinely different
        templates are reported indistinguishable.

        Class BALANCE must be structural; class ORDER must not be.
        """
        from src.services.labeling_detection_scorer import (
            DEFAULT_BATCH_SIZE, assemble_items,
        )
        patterns = set()
        for f in range(40):
            items = assemble_items(
                [{"text": f"p{i}"} for i in range(10)],
                [{"text": f"h{i}"} for i in range(5)],
                [{"text": f"e{i}"} for i in range(5)],
                panel_id="pnl_fixed", feature_id=f"feat_{f}",
            )
            patterns.add(tuple(i["label"] for i in items[:DEFAULT_BATCH_SIZE]))
        assert len(patterns) > 10, (
            f"only {len(patterns)} distinct truth patterns across 40 features; "
            f"the ground truth is guessable and a judge with a position or "
            f"alternation bias would score perfectly on any label"
        )

    def test_class_balance_is_still_structural(self):
        """The other half. Unguessable order must not cost the guarantee."""
        from src.services.labeling_detection_scorer import (
            DEFAULT_BATCH_SIZE, assemble_items,
        )
        for npos, nhard, neasy in [(10, 5, 5), (10, 1, 1), (5, 1, 1), (20, 2, 2)]:
            for seed in range(200):
                items = assemble_items(
                    [{"text": f"p{j}"} for j in range(npos)],
                    [{"text": f"h{j}"} for j in range(nhard)],
                    [{"text": f"e{j}"} for j in range(neasy)],
                    panel_id=f"pnl_{seed}", feature_id=f"feat_{seed}",
                )
                for start in range(0, len(items), DEFAULT_BATCH_SIZE):
                    batch = items[start:start + DEFAULT_BATCH_SIZE]
                    if len(batch) < 2:
                        continue
                    assert len({i["label"] for i in batch}) > 1, (
                        f"ratio {npos}/{nhard}/{neasy} seed {seed} batch at "
                        f"{start} is single-class"
                    )

    def test_a_truthy_non_boolean_gate_is_refused(self):
        """C45. `passed` round-tripped through JSON as the string "false" is
        truthy in Python, and authorised scoring on a judge that failed every
        control — with the failure list echoed in the result."""
        from src.services.labeling_detection_scorer import DetectionScoringError
        gate = {"passed": "false", "literal_control_ba": 0.1,
                "null_control_ba": 0.99, "failures": ["judge_unreliable"],
                "prompt_version": DETECTION_PROMPT_VERSION}
        with pytest.raises(DetectionScoringError, match="not bool"):
            score_panel([], keyword_judge, gate=gate)

    def test_a_gate_claiming_success_while_listing_failures_is_refused(self):
        from src.services.labeling_detection_scorer import DetectionScoringError
        gate = {"passed": True, "literal_control_ba": 0.1, "null_control_ba": 0.99,
                "failures": ["judge_unreliable"],
                "prompt_version": DETECTION_PROMPT_VERSION}
        with pytest.raises(DetectionScoringError, match="self-contradictory"):
            score_panel([], keyword_judge, gate=gate)

    def test_a_degenerate_judge_is_not_reported_reliable(self):
        """C46. `judge_reliable` is an affirmative claim. A judge that answered
        uniformly on most batches where it said anything refused to
        discriminate, which is what a consumer reading this field means by
        unreliable."""
        # The judge must be DEGENERATE WITHOUT being UNRELIABLE. always_one_judge
        # trips both, so dropping judge_degenerate from the exclusion set changed
        # nothing and the control survived. This one answers correctly most of
        # the time — clearing the literal-oracle bar — but goes all-1 often
        # enough to exceed the degeneracy threshold.
        # Control items must be INTERLEAVED. `_items()` returns them grouped, so
        # the literal control's early batches are single-class, a constant answer
        # legitimately fits them, and the degeneracy check is inert — the first
        # attempt at this control reported degenerate_rate 0.0 for a judge that
        # was plainly answering all-1.
        from src.services.labeling_detection_scorer import assemble_items
        items = assemble_items(
            [{"text": f"she was running across the field on foot {i}"} for i in range(15)],
            [{"text": f"running a background server {i}"} for i in range(3)],
            [{"text": f"the treaty was signed {i}"} for i in range(15)],
            panel_id="pnl_c46", feature_id="feat_c46", batch_size=6)

        state = {"n": 0}

        def mostly_right_sometimes_degenerate(prompt):
            # Degenerate on the LITERAL control only. A shared counter degrades
            # both controls, dragging the oracle below its threshold so
            # judge_unreliable fires too and the mutation becomes invisible.
            if _explanation(prompt) == TOKEN_ONLY.strip():
                state["n"] += 1
                if state["n"] % 2 == 0:
                    return always_one_judge(prompt)
            return keyword_judge(prompt)

        gate = run_gate([{
            "feature_id": "f1", "items": items,
            "literal_explanation": TOKEN_ONLY,
            "mismatched_explanation": "Culinary techniques involving braising.",
        }], mostly_right_sometimes_degenerate, batch_size=6)

        assert "judge_degenerate" in gate["failures"], gate
        assert "judge_unreliable" not in gate["failures"], (
            f"the judge also failed the oracle ({gate['literal_control_ba']}), so "
            f"this fixture cannot isolate judge_degenerate"
        )
        assert gate["judge_reliable"] is False

    def test_a_significant_effect_is_not_discarded_by_the_mde(self):
        """C47. Comparing an OBSERVED effect to an a-priori minimum detectable
        effect is a power calculation misused as a significance test. MDE is
        2.80 SE while the interval excludes zero at ~1.96 SE, so everything in
        that 43% band was thrown away — a +0.046 improvement whose 95% interval
        was [0.011, 0.082] came back "indistinguishable".
        """
        import random
        # This fixture must land in the band where the two tests DISAGREE:
        # the 95% interval excludes zero (significant) while the observed effect
        # is below the a-priori MDE. Measured: mean +0.0461, mde 0.0509,
        # ci [0.0113, 0.0817]. An effect above the MDE makes the mutation a
        # no-op and the control survives — which is what happened first time.
        rng = random.Random(4)
        base = {f"f{i}": 0.5 for i in range(30)}
        cand = {f"f{i}": 0.5 + 0.05 + rng.gauss(0, 0.11) for i in range(30)}
        out = compare_panels(base, cand)
        assert out["ci"]["low"] > 0, "fixture no longer produces a significant effect"
        assert out["mean_delta"] < out["minimum_detectable_effect"], (
            "fixture no longer sits below the MDE, so the mutation would be a "
            "no-op and this control would prove nothing"
        )
        assert out["verdict"] == "candidate_better", (
            f"an effect whose 95% interval excludes zero was reported "
            f"{out['verdict']!r}"
        )

    def test_a_zero_variance_verdict_carries_its_caveat(self):
        """A point interval excludes zero by construction and carries no
        information. The verdict may still stand on effect size, but the record
        must say so — an earlier version published
        `candidate_better / mde: None / reason: None`."""
        out = compare_panels({f"f{i}": 0.5 for i in range(30)},
                             {f"f{i}": 0.9 for i in range(30)})
        assert out["verdict"] == "candidate_better"
        assert out["reason"] and "no uncertainty interval" in out["reason"]


class TestSelfAssessmentSurvivesTheParser:
    """`fit_count` and `confidence` were requested by every template, parsed off
    the wire, and then dropped — so nothing downstream could tell a confident
    label from a hedged one.

    They carry real signal: measured on gemma-4-12B-it, fit_count was 0/10 on
    incoherent features and 10/10 where all ten prime tokens were identical.

    Mutation controls:
      C63 drop fit_count/confidence from _parse_dual_label's success return
           -> test_the_parser_returns_the_self_assessment
      C64 drop them from the trial result rows
           -> test_a_trial_result_row_carries_them
    """

    def _svc(self):
        from src.services.openai_labeling_service import OpenAILabelingService
        return OpenAILabelingService(api_key="k", model="m", base_url="http://x/v1")

    def test_the_parser_returns_the_self_assessment(self):
        out = self._svc()._parse_dual_label(
            '{"category":"semantic","specific":"a_b","description":"d",'
            '"fit_count":"7/10","confidence":"high"}', "fallback")
        assert out["fit_count"] == "7/10", "fit_count was discarded by the parser"
        assert out["confidence"] == "high"

    def test_a_slash_count_is_not_mangled_into_an_identifier(self):
        """_clean_label would turn '7/10' into '710'. It must not be applied."""
        out = self._svc()._parse_dual_label(
            '{"category":"c","specific":"s","description":"","fit_count":"0/10"}', "f")
        assert out["fit_count"] == "0/10"

    def test_absent_fields_are_none_not_empty_string(self):
        """A template that does not ask for them must yield None, so 'the model
        did not say' is distinguishable from 'the model said nothing'."""
        out = self._svc()._parse_dual_label(
            '{"category":"c","specific":"s","description":"d"}', "f")
        assert out["fit_count"] is None and out["confidence"] is None

    def test_the_plaintext_fallback_recovers_them_too(self):
        """When JSON parsing fails the regex path must not silently lose them."""
        out = self._svc()._parse_dual_label(
            'category: semantic, specific: some_label, fit_count: "3/10", confidence: low',
            "fallback")
        assert out["fit_count"] == "3/10"
        assert out["confidence"] == "low"

    def test_a_trial_result_row_carries_them(self):
        """C64. The trial payload is where a comparison reads from."""
        import inspect
        from src.services import labeling_trial_service as m
        src = inspect.getsource(m)
        assert '"fit_count": label.get("fit_count")' in src, (
            "trial results drop the self-assessment; a comparison cannot tell a "
            "confident label from a hedged one"
        )
        assert '"confidence": label.get("confidence")' in src
