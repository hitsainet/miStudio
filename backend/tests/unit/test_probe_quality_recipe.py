"""What makes a probe good, and the three decisions that were making it bad.

⚠ **ALL THREE OF THESE SHIPPED A PROBE NOBODY CHOSE.**

1. `rolling_mean_max` was absent from the default rule set, so the one rule whose score
   does not depend on how many tokens were scored had never been trained here.
2. Its window was pinned at `DEFAULT_WINDOW` because the orchestrator passed none and
   `ProbeRunConfig` is `extra="forbid"` — unreachable from any caller.
3. Selection ranked on in-distribution `val_auroc`, which saturates. Every `mean` probe
   scores 0.9982 on it and the layer sweep ties to within 0.0001, while the
   out-of-distribution evaluations that *can* tell probes apart were already computed and
   committed when the selection ran, and were ignored.

The measurement that motivated all three, taken on the shipped probe against the five real
evaluation sets (6,104 labelled rows): the score distribution shifts with input length in a
direction that depends on the corpus, so one fixed threshold is miscalibrated at every
length but the one it was cut at. On `anthropic_hh_balanced`, recall at the 1% bar rises
0.315 → 0.577 across length quartiles **while the realised FPR rises 0.0030 → 0.0161** — 5x
the budget. On `mental_health_balanced` it goes the other way: recall 0.500 → 0.297. AUROC
sees none of it, moving 0.958 → 0.947 and 0.908 → 0.977 respectively.
"""

from __future__ import annotations

import pytest

from src.ml.probe_monitor_model import DEFAULT_WINDOW, RULES, WINDOWED_RULES, rule_parameters
from src.schemas.probe_monitor import ProbeRunConfig
from src.services.probe_monitor_metrics import length_bands
from src.services.probe_monitor_run import context_from_row, probe_selection_key


class _Row:
    def __init__(self, config):
        self.id = "pmr_test"
        self.config = config


# ── 1. the rule and its window are reachable ──────────────────────────────────


class TestTheRuleSetAndItsWindowAreReachable:
    def test_rolling_mean_max_is_NOT_a_default_rule(self):
        """⚠ THIS TEST ASSERTED THE OPPOSITE FOR ABOUT SIX HOURS, AND A RUN REFUTED IT.

        It was added on the argument that a windowed max "has no 1/n term" and is therefore
        length-invariant. It is not: a maximum over N windows grows with N. Measured on
        `pmr_54563518010f`, every rolling variant drifts UPWARD by 0.45-1.07 separation
        units against `mean`'s 0.23-0.46, and on `mental_health_balanced` w=64 runs from
        recall 0.071 on the shortest length quartile to 0.800 on the longest.

        Kept as an explicit test rather than silently dropped, so nobody re-adds it on the
        same reasoning. It stays TRAINABLE — this is about what a default run inherits.
        """
        assert "rolling_mean_max" not in ProbeRunConfig().rules
        assert ProbeRunConfig(rules=["rolling_mean_max"]).rules == ["rolling_mean_max"]

    def test_mean_is_a_default_rule(self):
        """Because it is the least length-sensitive rule measured, at every layer."""
        assert "mean" in ProbeRunConfig().rules

    def test_the_window_stays_settable_even_though_the_rule_is_not_a_default(self):
        # `extra="forbid"` meant an unknown field was a 422, so before this field existed
        # there was no request that could change the window.
        assert ProbeRunConfig(rolling_windows=[32]).rolling_windows == [32]

    def test_windows_are_sorted_and_de_duplicated(self):
        assert ProbeRunConfig(rolling_windows=[32, 8, 32, 16]).rolling_windows == [8, 16, 32]

    @pytest.mark.parametrize("bad", [[0], [-1], [8, 0]])
    def test_a_window_below_one_is_refused(self, bad):
        # Not a degenerate window — a crash inside the rule.
        with pytest.raises(ValueError):
            ProbeRunConfig(rolling_windows=bad)

    def test_an_empty_window_list_is_refused(self):
        with pytest.raises(ValueError):
            ProbeRunConfig(rolling_windows=[])

    def test_windowed_rules_is_derived_from_rule_parameters_not_listed(self):
        """A hand-maintained second list is how the two drift apart."""
        assert WINDOWED_RULES == frozenset(r for r in RULES if "window" in rule_parameters(r))
        assert WINDOWED_RULES == {"rolling_mean_max"}


class TestOneProbePerWindowAndNoneForTheRest:
    def test_a_windowed_rule_trains_one_probe_per_window(self):
        ctx = context_from_row(_Row({"rules": ["rolling_mean_max"], "rolling_windows": [8, 16, 32]}))
        assert ctx.rule_windows("rolling_mean_max") == [8, 16, 32]

    @pytest.mark.parametrize("rule", sorted(set(RULES) - {"rolling_mean_max"}))
    def test_every_other_rule_trains_exactly_once_and_takes_no_window(self, rule):
        """⚠ `[None]`, NOT `[DEFAULT_WINDOW]`.

        `None` means "call `train_rule` with no window", which is byte-for-byte the call
        that existed before this loop. Substituting the default would hand a window to
        rules that take none, and `rule_parameters` would then be the only thing standing
        between that and a `mean` probe exporting a `window` it never used.
        """
        ctx = context_from_row(_Row({"rules": [rule], "rolling_windows": [8, 16, 32]}))
        assert ctx.rule_windows(rule) == [None]

    def test_a_run_row_predating_the_field_reproduces_what_actually_happened(self):
        # Every probe trained before `rolling_windows` existed was trained at the module
        # default, so this fallback is a true statement about history, not a guess.
        ctx = context_from_row(_Row({"rules": ["rolling_mean_max"]}))
        assert ctx.rule_windows("rolling_mean_max") == [DEFAULT_WINDOW]


# ── 2. selection ranks on evidence that can tell probes apart ─────────────────


class TestSelectionRanksOnOutOfDistributionEvidence:
    def test_the_better_generaliser_wins_even_when_it_loses_in_distribution(self):
        """The real numbers from this estate's Stage 1 acceptance.

        `attention` scored 0.9590 in distribution and 0.6986 out of it; `mean` scored
        0.9982 and 0.8841. Here the in-distribution order is INVERTED so the test can only
        pass if the out-of-distribution term decides.
        """
        attention = probe_selection_key(0.9999, [0.6986])
        mean = probe_selection_key(0.9982, [0.8841])
        assert mean > attention

    def test_val_auroc_breaks_a_tie_and_only_a_tie(self):
        assert probe_selection_key(0.99, [0.88]) > probe_selection_key(0.97, [0.88])

    def test_a_probe_with_no_ood_evidence_loses_to_one_with_any(self):
        # Including to one that generalises badly: measured beats unmeasured.
        assert probe_selection_key(0.51, [0.40]) > probe_selection_key(0.9999, [])

    def test_missing_evidence_is_not_scored_as_chance(self):
        """0.0, not 0.5. "Never evaluated" and "evaluated at chance" are different facts."""
        assert probe_selection_key(0.0, []) == (0.0, 0.0)

    def test_sets_are_averaged_rather_than_taking_the_best(self):
        # Taking the max would let one easy set carry a probe that fails the other four.
        assert probe_selection_key(0.5, [0.9, 0.5]) == (0.7, 0.5)

    def test_a_refused_set_does_not_drag_the_mean_down(self):
        # `select_best_probe` filters refusals before they arrive here; this pins that a
        # None among the scores is dropped rather than coerced.
        assert probe_selection_key(0.5, [0.9, None]) == (0.9, 0.5)


# ── 3. the metric that can see a length-dependent score ───────────────────────


def _rows(n, *, pos_score, neg_score, length):
    scores = [pos_score] * n + [neg_score] * n
    labels = [1] * n + [0] * n
    lengths = [length] * (2 * n)
    return scores, labels, lengths


class TestLengthBandsSeeACalibrationShift:
    def _shifted(self):
        """Ranking perfect in every band; the whole distribution slides down with length.

        This is the shape AUROC cannot see and the shape that took a live monitor silent:
        positives stay above their own negatives throughout, so banded AUROC is 1.0
        everywhere, while a single fixed bar catches all of the short band and none of the
        long one.
        """
        scores, labels, lengths = [], [], []
        for band, (length, base) in enumerate([(10, 30.0), (100, 20.0), (1000, 10.0), (5000, 0.0)]):
            s, y, n = _rows(25, pos_score=base, neg_score=base - 15.0, length=length)
            scores += s
            labels += y
            lengths += n
        return scores, labels, lengths

    def test_auroc_is_perfect_in_every_band_and_says_nothing(self):
        bands = length_bands(*self._shifted(), minimum=20)
        assert [b["auroc"] for b in bands] == [1.0, 1.0, 1.0, 1.0]

    def test_recall_at_a_fixed_bar_collapses_across_the_same_bands(self):
        bands = length_bands(*self._shifted(), minimum=20, threshold=25.0)
        assert [b["recall_at_threshold"] for b in bands] == [1.0, 0.0, 0.0, 0.0]

    def test_the_mean_positive_score_shows_the_cause_not_just_the_symptom(self):
        bands = length_bands(*self._shifted(), minimum=20, threshold=25.0)
        assert [b["mean_positive_score"] for b in bands] == [30.0, 20.0, 10.0, 0.0]

    def test_the_false_positive_rate_is_reported_as_the_control(self):
        """Without it, a rising recall reads as better detection.

        On the real `anthropic_hh_balanced` it is not: recall rises 0.315 → 0.577 while the
        realised FPR rises 0.0030 → 0.0161. The distribution moved; the bar did not.
        """
        scores, labels, lengths = [], [], []
        for length, base in [(10, 0.0), (100, 10.0), (1000, 20.0), (5000, 30.0)]:
            s, y, n = _rows(25, pos_score=base, neg_score=base - 5.0, length=length)
            scores += s
            labels += y
            lengths += n
        bands = length_bands(scores, labels, lengths, minimum=20, threshold=12.0)
        assert [b["recall_at_threshold"] for b in bands] == [0.0, 0.0, 1.0, 1.0]
        # The same bands that "improved" also blew the false-positive budget.
        assert [b["fpr_at_threshold"] for b in bands] == [0.0, 0.0, 1.0, 1.0]

    def test_without_a_threshold_the_fields_are_absent_rather_than_zero(self):
        """Absent says "not measured". `0.0` would say "it caught nothing"."""
        bands = length_bands(*self._shifted(), minimum=20)
        for band in bands:
            assert "recall_at_threshold" not in band
            assert "fpr_at_threshold" not in band
            assert "threshold_used" not in band

    def test_an_under_sampled_band_still_reports_its_recall(self):
        """The top band is the one most likely to be under-sampled AND the one that matters.

        A deliberately weaker standard than this file applies to AUROC, which is why
        `n_positive_in_band` is reported beside it — a recall over 3 rows cannot be
        over-read with its denominator printed next to it.
        """
        scores, labels, lengths = _rows(25, pos_score=30.0, neg_score=0.0, length=10)
        scores += [1.0, 1.0, 1.0]
        labels += [1, 1, 0]
        lengths += [9000, 9000, 9000]
        bands = length_bands(scores, labels, lengths, minimum=20, threshold=25.0)
        top = bands[-1]
        assert top.get("scored") is False          # AUROC refused: too few per class
        assert top["n_positive_in_band"] == 2      # but the recall is reported, with its n
        assert top["recall_at_threshold"] == 0.0

    def test_a_band_with_rows_but_no_positives_reports_no_recall_rather_than_zero(self):
        """A recall of 0.0 would read as "the probe missed them all"; there was nothing to miss.

        Distinct from an EMPTY band, which never reaches this code — it carries its own
        "no examples fall in this length band" reason, because an empty band and an
        under-sampled one send a reader looking for different things.
        """
        scores = [0.0] * 20                       # band 0: all negative
        labels = [0] * 20
        lengths = [10] * 20
        for length in (20, 30, 40):               # bands 1-3: balanced
            scores += [30.0] * 10 + [0.0] * 10
            labels += [1] * 10 + [0] * 10
            lengths += [length] * 20
        bands = length_bands(scores, labels, lengths, minimum=5, threshold=25.0)
        band0 = bands[0]
        assert band0["n_positive_in_band"] == 0
        assert band0["n_negative_in_band"] == 20
        assert "recall_at_threshold" not in band0
        # The negative half is still reported — a band with only negatives is exactly where
        # a false-positive rate is most worth knowing.
        assert band0["fpr_at_threshold"] == 0.0


# ── 4. the callers, asserted on the AST ───────────────────────────────────────


class TestTheDecisionsAreActuallyWired:
    """⚠ EVERY MUTATION THAT SURVIVED IN THE 033 ARC WAS A CALLER, NOT A HELPER.

    The publish preflight, the task's digest assignment, the build record's digest and the
    round-trip measurement were each a well-tested function whose one production call was
    covered by nothing. The three functions above are pure and unit-tested; these walk the
    orchestrator's AST for the CALL, because a substring search matches the comment that
    describes the call as happily as the call.
    """

    @staticmethod
    def _tree():
        import ast
        from pathlib import Path

        import src.services.probe_monitor_run as module

        return ast.parse(Path(module.__file__).read_text())

    @staticmethod
    def _calls(tree, name):
        import ast

        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                getattr(node.func, "id", None) == name
                or getattr(node.func, "attr", None) == name
            )
        ]

    def test_the_run_selects_through_select_best_probe(self):
        assert self._calls(self._tree(), "select_best_probe"), (
            "probe_monitor_run does not call select_best_probe — the selection criterion "
            "is unreachable and the run is still ranking on val_auroc"
        )

    def test_the_old_saturated_criterion_is_gone(self):
        """The whole defect was `max(..., key=val_auroc)`; it must not survive beside the fix."""
        import ast
        from pathlib import Path

        import src.services.probe_monitor_run as module

        source = Path(module.__file__).read_text()
        assert 'key=lambda p: (p.val_metrics or {}).get("val_auroc")' not in source

    def test_the_training_loop_asks_rule_windows_which_windows_to_train(self):
        assert self._calls(self._tree(), "rule_windows"), (
            "the training loop does not call rule_windows — every rolling probe would be "
            "trained at the module default and the config field would do nothing"
        )

    def test_the_window_reaches_train_rule(self):
        """`train_rule` must be called with a `**` expansion — that is the window.

        Asserted structurally rather than by name because the call splats
        `{"window": …}` conditionally, so there is no literal `window=` keyword to find.
        """
        import ast

        calls = self._calls(self._tree(), "train_rule")
        assert calls, "train_rule is not called at all"
        splatted = [c for c in calls if any(k.arg is None for k in c.keywords)]
        assert splatted, (
            "no train_rule call expands a keyword dict, so no window is ever passed"
        )


# ── 5. the window must survive the round trip to a consumer ───────────────────


class TestARuleCannotLoseItsParameters:
    """⚠ miLLM SUBSTITUTES ITS OWN DEFAULT RATHER THAN REFUSING.

    `millm/ml/probe_head.py` carries `window=16` and `tau=1.0` defaults on `combine`, with no
    log line. A `rolling_mean_max` document that lost `params.window` is therefore served as a
    window-16 detector while carrying the name, the published metrics and the test vectors of
    a window-8 one — a plausible substitute nobody can see, which is the same shape as
    MIS-E2E-083's guessed SAE normalisation.

    miStudio cannot stop a consumer defaulting, but it can refuse to PRODUCE the document.
    """

    def test_a_rolling_probe_without_its_window_is_refused(self):
        from src.schemas.probe_definition import Aggregation

        with pytest.raises(ValueError, match="window"):
            Aggregation(rule="rolling_mean_max", params={}, streamable=True)

    def test_a_softmax_probe_without_its_tau_is_refused(self):
        from src.schemas.probe_definition import Aggregation

        with pytest.raises(ValueError, match="tau"):
            Aggregation(rule="softmax", params={}, streamable=True)

    @pytest.mark.parametrize("rule", ["mean", "max", "last", "attention"])
    def test_a_rule_that_needs_nothing_is_unaffected(self, rule):
        from src.schemas.probe_definition import Aggregation

        assert Aggregation(rule=rule, params={}, streamable=rule != "last").params == {}

    def test_the_requirement_is_derived_from_the_trainer_not_listed(self):
        """A second list is how a rule gains a parameter in one place and not the other."""
        from src.ml.probe_monitor_model import RULES, rule_parameters
        from src.schemas.probe_definition import Aggregation

        for rule in RULES:
            needed = rule_parameters(rule)
            if not needed:
                continue
            # Present -> accepted; absent -> refused. Both halves, for every rule that has one.
            Aggregation(rule=rule, params=dict(needed), streamable=rule != "last")
            with pytest.raises(ValueError):
                Aggregation(rule=rule, params={}, streamable=rule != "last")

    def test_the_window_survives_training_to_the_exported_aggregation(self):
        """The whole chain, on a real fit rather than a hand-made dict.

        `rule_parameters` -> `TrainedRule.rule_params` -> `probe_monitors.rule_params` ->
        `load_probe` -> `Aggregation.params`. The two ends are tested here; the database leg
        in the middle is a JSONB column that stores what it is given.
        """
        import numpy as np

        from src.schemas.probe_definition import Aggregation
        from src.services.probe_monitor_trainer import train_rule

        rng = np.random.default_rng(0)
        rows = [rng.normal(0, 1, (30, 8)).astype(np.float32) for _ in range(8)]
        labels = [i % 2 for i in range(8)]
        trained = train_rule(
            "rolling_mean_max", rows, labels, rows, labels, seed=1337, epochs=2, window=32
        )
        assert trained.rule_params == {"window": 32}
        exported = Aggregation(
            rule=trained.rule, params=dict(trained.rule_params), streamable=True
        )
        assert exported.params == {"window": 32}, "the window did not reach the document"


# ── 6. every long stage must tell the DATABASE it is alive ────────────────────


class TestEveryLongStageHeartbeatsToTheDatabase:
    """⚠ `training` — THE LONGEST STAGE — HAD NO DATABASE HEARTBEAT.

    `pooled_capture`, `token_capture`, `calibrating` and `evaluating` all called
    `heartbeat(...)`; `training` did not. `probe_monitor_runs.updated_at` therefore went
    stale for the entire stage, because `_persist_probe` writes to `probe_monitors` — a
    different table — so a probe landing did not refresh it either.

    Measured on run `pmr_54563518010f` (2026-10-01): the quiet timer climbed monotonically
    through 737 -> 841 -> 963 -> 1084 seconds across four probes landing, pacing at about
    3.5 min/probe on a 24-probe run. `cleanup_stuck_probe_monitor_runs` examines a row at
    90 minutes of silence and reaps it on the next pass, so that run was arithmetically
    certain to be killed mid-training while its worker burned 207% CPU.

    The socket emits were flowing the whole time — which is exactly what makes this class
    of defect invisible, and exactly what `heartbeat`'s own docstring warns about: "this
    estate has already reaped a LIVE 5.8-hour packing job that reported only over the
    WebSocket."

    Written over the STAGE LIST rather than as a single assertion about `training`, so a
    stage added later without a heartbeat fails too. A test that only pinned this fix
    would not have caught the original.
    """

    #: Stages that can run for minutes or hours and must therefore heartbeat. The two
    #: omitted — `rendering` and `layer_selection` — are CPU-seconds (a tokenizer pass and
    #: a sklearn fit over a tiny grid), and `rung` is a handful of row reads.
    LONG_STAGES = ("pooled_capture", "token_capture", "training", "calibrating", "evaluating")

    @staticmethod
    def _heartbeat_stage_names():
        """Every string literal passed as `heartbeat`'s first argument, from the AST."""
        import ast
        from pathlib import Path

        import src.services.probe_monitor_run as module

        tree = ast.parse(Path(module.__file__).read_text())
        names = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "heartbeat"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.add(node.args[0].value)
        return names

    @pytest.mark.parametrize("stage", LONG_STAGES)
    def test_the_stage_opens_a_heartbeat(self, stage):
        assert stage in self._heartbeat_stage_names(), (
            f"stage {stage!r} never calls heartbeat(), so probe_monitor_runs.updated_at "
            f"goes stale for its whole duration and the janitor reaps a live run"
        )

    def test_the_heartbeat_is_actually_invoked_not_merely_created(self):
        """A `report` callable that is built and never called is the defect with a fig leaf.

        Asserts the training heartbeat's returned callable appears as a CALL, which a
        substring search for `train_beat` would not distinguish from its assignment.
        """
        import ast
        from pathlib import Path

        import src.services.probe_monitor_run as module

        tree = ast.parse(Path(module.__file__).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "train_beat"
        ]
        assert calls, "train_beat is assigned but never called"

    def test_the_stage_list_is_covered(self):
        """Guard the guard: a renamed stage must not silently drop out of LONG_STAGES."""
        from src.services.probe_monitor_run import STAGES

        known = {name for name, _ in STAGES}
        assert set(self.LONG_STAGES) <= known, sorted(set(self.LONG_STAGES) - known)


# ── 7. a live task must survive its own stage forgetting to heartbeat ─────────


class TestLivenessDoesNotReduceToOneSignal:
    """⚠ THE HEARTBEAT AND ITS BACKSTOP WERE THE SAME SIGNAL, SO ONE GAP DEFEATED BOTH.

    On 2026-10-01 run `pmr_54563518010f` was marked `failed` at 100 minutes of row silence
    while its worker held 22.6 GB of GPU memory, burned 217% CPU, and had written 23 trained
    probes into another table during the supposed silence. The reap message read "the worker
    ... stopped reporting ... and no live task was found." Nothing had looked.

    Two mechanisms were meant to prevent it:

      1. the stage heartbeat  -> probe_monitor_runs.updated_at
      2. the janitor's `_looks_alive` -> `task_looks_alive` -> `looks_abandoned`, whose
         surviving branch is `state == "PENDING" and seconds_since_row_update > STALE`

    Both resolve to `updated_at`. `training` was the one long stage with no heartbeat, so the
    clock froze and both layers failed together. Defence in depth that shares an input is not
    defence in depth.

    ⚠ Five janitors share `task_looks_alive` — trainings, extractions, labeling, circuits and
    j-lens — so this was never a probe-monitor bug.
    """

    def test_there_is_a_liveness_signal_that_is_not_the_row_clock(self):
        from src.workers import task_heartbeat

        assert hasattr(task_heartbeat, "task_is_actively_running")

    def test_a_task_the_workers_report_active_is_alive_however_stale_the_row(self, monkeypatch):
        """The exact situation that was mis-called: ancient row, task plainly running."""
        from datetime import datetime, timedelta, timezone

        from src.workers import task_heartbeat

        monkeypatch.setattr(
            task_heartbeat, "task_is_actively_running", lambda _id: True
        )

        class Row:
            updated_at = datetime.now(timezone.utc) - timedelta(hours=5)

        assert task_heartbeat.task_looks_alive("task-1", Row(), started=True) is True

    def test_no_answer_from_the_workers_is_not_evidence_of_death(self, monkeypatch):
        """None must fall through to the old logic, never be read as False.

        A solo worker busy inside a long GPU task may not answer a broadcast at all, and
        treating silence as death is how this check would start killing healthy jobs rather
        than sparing them.
        """
        from datetime import datetime, timezone

        from src.workers import task_heartbeat

        monkeypatch.setattr(task_heartbeat, "task_is_actively_running", lambda _id: None)

        class FreshRow:
            updated_at = datetime.now(timezone.utc)

        # Falls through to the row clock, which is fresh, so still alive.
        assert task_heartbeat.task_looks_alive("task-1", FreshRow(), started=True) is True

    def test_no_answer_must_not_RESCUE_a_genuinely_dead_task(self, monkeypatch):
        """⚠ A SURVIVING MUTATION WROTE THIS TEST.

        Changing the short-circuit from `is True` to `is not False` left all 54 tests green,
        because the case above uses a FRESH row — where the correct code and the broken code
        both answer True, by different routes. The distinction only bites on a STALE row.

        `None` means "nobody answered", and a broadcast goes unanswered routinely when the
        single solo worker is busy. If that rescued every stale row, the janitors would stop
        reclaiming anything at all and the GPU would stay held by dead jobs — the opposite
        failure, and the one these janitors exist for.
        """
        from datetime import datetime, timedelta, timezone

        from src.workers import task_heartbeat

        monkeypatch.setattr(task_heartbeat, "task_is_actively_running", lambda _id: None)
        monkeypatch.setattr(task_heartbeat, "waiting_for_gpu", lambda _id: False, raising=False)

        class StaleRow:
            updated_at = datetime.now(timezone.utc) - timedelta(hours=5)

        assert (
            task_heartbeat.task_looks_alive("task-never-dispatched", StaleRow(), started=True)
            is False
        ), "an unanswered broadcast rescued a five-hour-stale row; nothing would ever be reclaimed"

    def test_the_inspect_call_never_raises_into_a_janitor(self, monkeypatch):
        """A broker hiccup must return None, not propagate — a janitor that raises on one
        row stops sweeping the rest."""
        from src.workers import task_heartbeat

        class Boom:
            def control(self):
                raise RuntimeError("broker down")

        monkeypatch.setattr(
            "src.core.celery_app.celery_app",
            type("C", (), {"control": property(lambda s: (_ for _ in ()).throw(RuntimeError()))})(),
        )
        assert task_heartbeat.task_is_actively_running("task-1") is None

    def test_no_task_id_is_not_a_liveness_claim(self):
        from src.workers import task_heartbeat

        assert task_heartbeat.task_is_actively_running(None) is None

    def test_the_check_is_consulted_before_the_row_clock(self):
        """Asserted on the AST: the call must appear, or the backstop is decorative."""
        import ast
        from pathlib import Path

        from src.workers import task_heartbeat

        tree = ast.parse(Path(task_heartbeat.__file__).read_text())
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "task_looks_alive"
        )
        calls = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "task_is_actively_running"
        ]
        assert calls, "task_looks_alive never asks the workers; it is still one signal"


# ── 8. a threshold that varies with length ────────────────────────────────────


class TestLengthBandDecisions:
    """⚠ ONE CONSTANT THRESHOLD IS MISCALIBRATED AT EVERY LENGTH BUT THE ONE IT WAS CUT AT.

    Measured on the shipped `mean` probe over five out-of-distribution sets: realised FPR
    reaching 5.4x its 1% budget on `anthropic_hh_balanced`'s longest length quartile, and
    recall falling 0.500 -> 0.297 on `mental_health_balanced`. Correcting for length recovers
    +0.039 / +0.022 / +0.017 mean OOD AUROC at L11 / L16 / L21, against five null controls —
    random bands of identical size — which all came back NEGATIVE.
    """

    @staticmethod
    def _corpus(n_per_band=400):
        """Negatives whose scores climb with length — the drift, made explicit."""
        scores, lengths = [], []
        for base, length in [(0.0, 50), (5.0, 150), (10.0, 400), (15.0, 1200)]:
            for i in range(n_per_band):
                scores.append(base + (i % 100) / 100.0)
                lengths.append(length)
        return scores, lengths

    def test_each_band_gets_its_own_threshold(self):
        from src.services.probe_monitor_metrics import length_band_decisions

        bands = length_band_decisions(*self._corpus(), target_fpr=0.05)
        assert len(bands) == 4
        thresholds = [b["threshold"] for b in bands]
        assert thresholds == sorted(thresholds), "thresholds must track the drift they correct"
        assert all(b["threshold_source"] == "band" for b in bands)

    def test_the_table_covers_every_possible_length(self):
        """No request may fall outside it — the first band starts at 0, the last never ends."""
        from src.services.probe_monitor_metrics import length_band_decisions

        bands = length_band_decisions(*self._corpus(), target_fpr=0.05)
        assert bands[0]["min_tokens"] == 0
        assert bands[-1]["max_tokens"] is None
        for a, b in zip(bands, bands[1:]):
            assert b["min_tokens"] == int(a["max_tokens"]) + 1, "a gap would drop requests"

    def test_a_band_that_cannot_afford_the_quantile_says_so(self):
        """⚠ 1% needs 100 negatives. Four samples cannot produce a 1% quantile, and a number
        computed from four samples is a fabrication wearing a threshold's clothes."""
        from src.services.probe_monitor_metrics import length_band_decisions

        scores = [1.0] * 400 + [9.0] * 4
        lengths = [50] * 400 + [5000] * 4
        bands = length_band_decisions(scores, lengths, target_fpr=0.01, global_threshold=3.0)
        thin = [b for b in bands if b["n_negatives"] < 100]
        assert thin, "expected an under-sampled band"
        for band in thin:
            assert band["threshold_source"] == "global"
            assert band["threshold"] == 3.0

    def test_a_uniform_width_corpus_yields_one_band_not_four_identical_ones(self):
        """Packed blocks are all the same width; four copies of one band is noise, not a table."""
        from src.services.probe_monitor_metrics import length_band_decisions

        bands = length_band_decisions([float(i % 50) for i in range(400)], [2048] * 400,
                                      target_fpr=0.05)
        assert len(bands) == 1
        assert bands[0]["min_tokens"] == 0 and bands[0]["max_tokens"] is None

    def test_no_data_is_not_attempted_rather_than_a_constant_dressed_as_varying(self):
        from src.services.probe_monitor_metrics import length_band_decisions

        assert length_band_decisions([], [], target_fpr=0.01) is None
        assert length_band_decisions([1.0], [1, 2], target_fpr=0.01) is None

    def test_the_lookup_picks_the_band_a_length_falls_in(self):
        from src.services.probe_monitor_metrics import length_band_decisions, threshold_for_length

        bands = length_band_decisions(*self._corpus(), target_fpr=0.05)
        for band in bands:
            lo = int(band["min_tokens"])
            assert threshold_for_length(bands, lo, fallback=None) == band["threshold"]
            if band["max_tokens"] is not None:
                assert threshold_for_length(bands, int(band["max_tokens"]), None) == band["threshold"]

    def test_an_enormous_length_lands_in_the_open_ended_band(self):
        from src.services.probe_monitor_metrics import length_band_decisions, threshold_for_length

        bands = length_band_decisions(*self._corpus(), target_fpr=0.05)
        assert threshold_for_length(bands, 10_000_000, None) == bands[-1]["threshold"]

    def test_no_table_falls_back_to_the_single_threshold(self):
        """Every probe exported before 2026-10-01 carries no table, and must keep working."""
        from src.services.probe_monitor_metrics import threshold_for_length

        assert threshold_for_length(None, 500, fallback=2.5) == 2.5
        assert threshold_for_length([], 500, fallback=2.5) == 2.5

    def test_the_calibrating_stage_actually_computes_the_band_table(self):
        """⚠ ASSERTED ON THE CALL. A table nothing fills is a column of nulls.

        Every mutation that survived in the 033 arc was a CALLER whose helper was well
        covered — the publish preflight, the task's digest assignment, the build record's
        digest, the round-trip measurement. This walks the orchestrator for the call and for
        the assignment onto the row, because a function that is imported and never invoked
        reads identically in a grep.
        """
        import ast
        from pathlib import Path

        import src.services.probe_monitor_run as module

        tree = ast.parse(Path(module.__file__).read_text())
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "length_band_decisions"
        ]
        assert calls, "the calibrating stage never calls length_band_decisions"

        assigns = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "length_bands" for t in n.targets
            )
        ]
        assert assigns, "nothing assigns probe.length_bands, so the table is never persisted"

    def test_the_scorer_returns_the_token_counts_the_bands_need(self):
        """The lengths come free from the pass already running — no extra GPU work.

        Pinned because the whole design rests on it: `ScoredRow.n_scored` has always been
        there and the calibration scorer discarded it, which made a length-varying threshold
        look like it needed another seven-minute pass per probe.
        """
        import ast
        import inspect

        import src.services.probe_monitor_run as module

        src = inspect.getsource(module._score_calibration_group)
        built = [
            n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "CalibrationScores"
        ]
        assert built, "the scorer no longer builds CalibrationScores — the scan is looking at the wrong shape"
        assert all("lengths" in {k.arg for k in n.keywords} for n in built), (
            "the scorer must carry the per-row token counts with the negatives — without them "
            "the token counts are being discarded again"
        )

    def test_the_token_counts_actually_REACH_the_band_table(self):
        """⚠ A SURVIVING MUTATION WROTE THIS TEST.

        Deleting `cal_lengths[:] = lengths` — the one line carrying token counts from the
        scoring pass to the band computation — left all 83 tests green. `probe.length_bands`
        would then be `None` on every probe forever, and the only visible symptom would be a
        column of nulls that looks exactly like "no calibration view was given".

        The call-site test above is not enough: it proves `length_band_decisions` is CALLED,
        not that it is called with anything. These two assertions pin the data flow — the
        closure fills the list, and the list is what the computation reads.
        """
        import ast
        from pathlib import Path

        import src.services.probe_monitor_run as module

        tree = ast.parse(Path(module.__file__).read_text())

        filled = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Subscript) and getattr(t.value, "id", None) == "cal_lengths"
                for t in n.targets
            )
        ]
        assert filled, (
            "nothing assigns into cal_lengths, so the band table is computed from an empty "
            "list and probe.length_bands is None on every probe"
        )

        call = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "length_band_decisions"
        )
        arg_names = [getattr(a, "id", None) for a in call.args]
        assert "cal_lengths" in arg_names, (
            f"length_band_decisions is called with {arg_names}; the captured token counts "
            f"are not among them"
        )
