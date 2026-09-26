"""A long stage writes a within-stage heartbeat (032 FR-14).

⚠ WHY. The stage helper writes once at each boundary, and the first Stage 1 acceptance
run spent OVER FORTY MINUTES inside `pooled_capture` alone on 8,000 rows x 7 layers of
Llama-3.1-8B. `cleanup_stuck_probe_monitor_runs` reaps after 90 minutes of silence, and
its own docstring asserts that "the longest single stage ... is measured in tens of
minutes rather than hours" — which a larger evaluation set falsifies. With no in-stage
write, `task_looks_alive` was the only thing between a live 3090 job and being
reclaimed, and this estate has already reaped a LIVE 5.8-hour packing job for exactly
that reason (`long-phases-need-a-db-heartbeat`).

The three capture loops all accept a `progress` callback; the run passed none of them.

MUTATION CONTROLS (each verified to fail this file):
  H1  the `progress=` argument dropped from `capture_pooled`   → the AST wiring test
  H2  the same from `capture_tokens`                           → the AST wiring test
  H3  the same from the evaluation call                        → the AST wiring test
  H4  `_evaluate_probe_on_sets` stops forwarding it            → the forwarding test
  H5  the throttle counts batches instead of seconds           → the throttle test
  H6  the write bypasses `record_progress`                     → the terminal-row test
  H7  a band given as literals instead of `heartbeat_band`      → the band test
  H8  `heartbeat_band` returns the PREVIOUS stage's value       → the band-direction test
"""
import ast
import inspect
import textwrap

import pytest

from src.services import probe_monitor_run
from src.services.probe_monitor_run import (
    HEARTBEAT_SECONDS,
    STAGE_PROGRESS,
    STAGES,
    heartbeat_band,
)


def _calls(function, name):
    tree = ast.parse(inspect.getsource(function).lstrip())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            out.append(node)
    return out


class TestEveryLongStagePassesAProgressCallback:
    """Asserted over the AST of `execute_probe_run` — the capture functions' `progress`
    parameter defaults to None, so a missing argument is silent at every level."""

    @pytest.fixture(scope="class")
    def tree(self):
        return ast.parse(inspect.getsource(probe_monitor_run.execute_probe_run).lstrip())

    def _keyword_names(self, tree, callee):
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != callee:
                continue
            yield {keyword.arg for keyword in node.keywords}

    @pytest.mark.parametrize(
        "callee", ["capture_pooled", "capture_tokens", "_evaluate_probe_on_sets"]
    )
    def test_the_call_passes_progress(self, tree, callee):
        found = list(self._keyword_names(tree, callee))
        assert found, f"{callee} is not called from execute_probe_run at all"
        for keywords in found:
            assert "progress" in keywords, (
                f"{callee} is called without progress=, so that stage writes nothing "
                f"between its boundaries and reads as a dead worker"
            )

    def test_the_callback_comes_from_the_heartbeat_factory(self, tree):
        """`progress=None` or `progress=lambda *_: None` would satisfy the test above
        and heartbeat nothing."""
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in {"capture_pooled", "capture_tokens", "_evaluate_probe_on_sets"}:
                continue
            for keyword in node.keywords:
                if keyword.arg != "progress":
                    continue
                assert isinstance(keyword.value, ast.Call), (
                    f"{node.func.id}'s progress= is {ast.dump(keyword.value)[:80]}, not a "
                    f"heartbeat(...) call"
                )
                assert isinstance(keyword.value.func, ast.Name)
                assert keyword.value.func.id == "heartbeat"

    def test_the_evaluation_helper_forwards_it_to_forward_scores(self):
        """The one indirection: `_evaluate_probe_on_sets` takes the callback and must
        hand it on, or the deepest loop still reports nothing."""
        tree = ast.parse(inspect.getsource(probe_monitor_run._evaluate_probe_on_sets).lstrip())
        forwarded = False
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "forward_scores":
                continue
            for keyword in node.keywords:
                if keyword.arg == "progress":
                    assert isinstance(keyword.value, ast.Name) and keyword.value.id == "progress"
                    forwarded = True
        assert forwarded, "forward_scores is called without the forwarded progress callback"

    def test_the_ast_walk_can_tell_a_call_from_a_mention(self):
        def decoy():
            # capture_pooled(model, examples, layers, progress=heartbeat("x", (0, 1)))
            return "progress=heartbeat"

        tree = ast.parse(inspect.getsource(decoy).lstrip())
        assert not [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "capture_pooled"
        ]


class TestTheThrottleIsOnTimeNotOnCount:
    """`% 5` over attribution batches is up to twenty minutes; `% 25` over training
    steps can be milliseconds. A batch here is seconds to minutes, so counting batches
    cannot bound the gap between writes — the cancellation module's own rule."""

    def test_the_interval_is_a_number_of_seconds(self):
        assert isinstance(HEARTBEAT_SECONDS, (int, float))
        assert 10 <= HEARTBEAT_SECONDS <= 300, HEARTBEAT_SECONDS

    def test_it_leaves_real_margin_against_the_reaper(self):
        from src.workers.cleanup_stuck_probe_monitor_runs import STUCK_THRESHOLD_MINUTES

        assert HEARTBEAT_SECONDS * 10 < STUCK_THRESHOLD_MINUTES * 60, (
            "the heartbeat interval is within an order of magnitude of the reaping "
            "window, so a handful of slow writes could still look like silence"
        )

    @staticmethod
    def _heartbeat_ast():
        """The `heartbeat` closure's own AST, isolated from the rest of the run.

        ⚠ A SUBSTRING CHECK OVER THIS BODY IS NOT ENOUGH, AND THAT IS NOT THEORETICAL:
        the first version of these tests asserted `"record_progress" in body`, and
        mutation H6 — replacing the `record_progress(...)` call with a bare
        `row.progress = ...; db.commit()` — PASSED, because the closure's own docstring
        contains the words "goes through `record_progress`". The guard was satisfied by
        the comment describing the thing it was checking for. That is the fifth time this
        repo has shipped that shape, so the walk below looks at CALLS and reads the
        docstring out of the tree first.
        """
        source = inspect.getsource(probe_monitor_run.execute_probe_run)
        start = source.index("def heartbeat(")
        body = source[start : source.index("def finish(", start)]
        # `heartbeat` is nested, so its source is indented; dedent to parse it alone.
        tree = ast.parse(textwrap.dedent(body))
        function = tree.body[0]
        assert isinstance(function, ast.FunctionDef) and function.name == "heartbeat"
        # Drop the docstring so nothing in prose can satisfy an assertion below.
        if (
            function.body
            and isinstance(function.body[0], ast.Expr)
            and isinstance(function.body[0].value, ast.Constant)
        ):
            function.body = function.body[1:]
        return function

    def _called_names(self, function):
        names = set()
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                target = node.func.value
                prefix = target.id + "." if isinstance(target, ast.Name) else ""
                names.add(prefix + node.func.attr)
        return names

    def test_the_heartbeat_writes_through_record_progress(self):
        """`record_progress` refuses to move a terminal row, which is the whole of
        cooperative cancellation: an in-flight heartbeat must not overwrite a
        cancellation the endpoint has just written."""
        called = self._called_names(self._heartbeat_ast())
        assert "record_progress" in called, (
            f"the heartbeat does not CALL record_progress (it calls {sorted(called)}); a "
            f"direct row.progress write would resurrect a cancelled run"
        )

    def test_it_does_not_write_the_row_directly(self):
        """The mutation's exact shape, refused by name."""
        function = self._heartbeat_ast()
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "progress":
                    pytest.fail(
                        "the heartbeat assigns to .progress directly instead of going "
                        "through record_progress"
                    )

    def test_the_docstring_alone_would_not_satisfy_it(self):
        """The control for the control. The closure's docstring names `record_progress`;
        if the reader did not strip it, H6 would pass again."""
        function = self._heartbeat_ast()
        rendered = ast.dump(function)
        assert "refuses to move a terminal row" not in rendered, (
            "the docstring is still in the tree, so a prose mention could satisfy the "
            "call assertions above"
        )

    def test_the_factory_reads_the_clock(self):
        called = self._called_names(self._heartbeat_ast())
        assert "time.time" in called, f"the heartbeat does not read the clock: {sorted(called)}"
        names = {
            node.id
            for node in ast.walk(self._heartbeat_ast())
            if isinstance(node, ast.Name)
        }
        assert "HEARTBEAT_SECONDS" in names, "the heartbeat does not use the interval"

    def test_the_final_report_is_never_throttled(self):
        """`done < total` in the throttle condition: the last call always writes, so a
        stage cannot finish with its progress stuck at the previous interval."""
        function = self._heartbeat_ast()
        comparisons = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name) and node.left.id == "done"
            and any(isinstance(op, ast.Lt) for op in node.ops)
            and any(isinstance(c, ast.Name) and c.id == "total" for c in node.comparators)
        ]
        assert comparisons, "the throttle has no `done < total` always-write case"


class TestTheBandIsDerivedAndPointsForwards:
    """⚠ I GOT THESE WRONG AS LITERALS. `stage()` writes its value ON ENTRY, so
    `pooled_capture` is already at 30.0 when it starts. The first version of this fix
    gave it the band (5.0, 30.0) — reading STAGES as "where this stage ends" — which
    would have sent the bar from 30% back to 5% at the first heartbeat and then crawled
    up again. Derived from STAGES now, so the two cannot disagree.
    """

    def test_a_band_starts_at_its_own_stages_value(self):
        for name, value in STAGES:
            low, _high = heartbeat_band(name)
            assert low == value, f"{name} heartbeats from {low}, but it is entered at {value}"

    def test_a_band_ends_at_the_NEXT_stages_value(self):
        names = [name for name, _ in STAGES]
        for index, name in enumerate(names[:-1]):
            _low, high = heartbeat_band(name)
            assert high == STAGE_PROGRESS[names[index + 1]], (
                f"{name} tops out at {high}, but {names[index + 1]} begins at "
                f"{STAGE_PROGRESS[names[index + 1]]} — a gap or an overlap either way"
            )

    def test_no_band_points_backwards(self):
        """The defect in one assertion."""
        for name, _ in STAGES:
            low, high = heartbeat_band(name)
            assert low <= high, f"{name}'s band {low}..{high} runs backwards"

    def test_the_last_stage_tops_out_at_100(self):
        assert heartbeat_band(STAGES[-1][0])[1] == 100.0

    def test_an_unknown_stage_is_refused(self):
        """A typo'd stage name must not silently produce (0, 100)."""
        with pytest.raises(KeyError):
            heartbeat_band("no_such_stage")

    def test_the_call_sites_use_a_derived_band(self):
        """Literals are what went wrong; asserted over the AST so a reintroduced tuple is
        caught rather than merely discouraged by a comment.

        `sub_band(...)` is accepted alongside `heartbeat_band(...)` because a stage that
        repeats work per probe needs its band DIVIDED, not restarted — the rewind found on
        `pmr_413d1e0e03cf`. `sub_band` derives from `heartbeat_band` internally, which
        `TestSubBandsPartitionTheBand` below pins, so the no-literals property still holds
        through it.
        """
        derived = {"heartbeat_band", "sub_band"}
        tree = ast.parse(inspect.getsource(probe_monitor_run.execute_probe_run).lstrip())
        seen = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "heartbeat":
                continue
            assert len(node.args) == 2, ast.dump(node)
            band = node.args[1]
            assert isinstance(band, ast.Call) and isinstance(band.func, ast.Name), (
                f"the band is {ast.dump(band)[:100]}, not a derived-band call"
            )
            assert band.func.id in derived, (
                f"the band comes from {band.func.id}(...), not one of {sorted(derived)}"
            )
            seen.add(band.func.id)
        assert seen, "no heartbeat(...) call was found at all"


class TestSubBandsPartitionTheBand:
    """A stage that repeats work per probe divides its band; it does not restart it.

    ⚠ FOUND ON THE LIVE RUN, NOT IN THE SUITE. `pmr_413d1e0e03cf` reported 99.2% → 100.0% →
    **97.7%** as its second probe's evaluation began, because every probe got its own
    heartbeat closure and each started at the band's low end.

    MUTATION CONTROLS:
      E1  the evaluating site back to the full band per probe  → the pipeline's monotonicity
      E2  `sub_band` ignores its index                         → same, with a multi-batch fixture
    """

    def test_one_probe_gets_the_whole_band(self):
        from src.services.probe_monitor_run import sub_band

        assert sub_band("evaluating", 0, 1) == heartbeat_band("evaluating")

    def test_slices_tile_the_band_without_gaps_or_overlap(self):
        from src.services.probe_monitor_run import sub_band

        low, high = heartbeat_band("evaluating")
        slices = [sub_band("evaluating", index, 4) for index in range(4)]
        assert slices[0][0] == low
        assert slices[-1][1] == pytest.approx(high)
        for earlier, later in zip(slices, slices[1:]):
            assert earlier[1] == pytest.approx(later[0]), f"{earlier} then {later}"

    def test_the_slices_ascend(self):
        from src.services.probe_monitor_run import sub_band

        lows = [sub_band("evaluating", index, 5)[0] for index in range(5)]
        assert lows == sorted(lows)
        assert len(set(lows)) == 5, "the slices are not distinct, so the bar would rewind"

    def test_it_stays_inside_the_stages_own_band(self):
        from src.services.probe_monitor_run import sub_band

        low, high = heartbeat_band("evaluating")
        for index in range(3):
            slice_low, slice_high = sub_band("evaluating", index, 3)
            assert low <= slice_low <= slice_high <= high

    def test_a_bad_index_or_count_is_refused(self):
        from src.services.probe_monitor_run import sub_band

        with pytest.raises(ValueError):
            sub_band("evaluating", 0, 0)
        with pytest.raises(ValueError):
            sub_band("evaluating", 3, 3)
        with pytest.raises(ValueError):
            sub_band("evaluating", -1, 3)

    def test_an_unknown_stage_is_still_refused_through_it(self):
        from src.services.probe_monitor_run import sub_band

        with pytest.raises(KeyError):
            sub_band("no_such_stage", 0, 1)
