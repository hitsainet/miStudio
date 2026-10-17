"""A judge run honours the caps it was given (032, found by Stage 2 acceptance).

⚠ BOTH CAPS WERE ACCEPTED AND DROPPED, and only a live run showed it. `JudgeRunCreate` declares
`max_rows_per_set` (ge=20, le=50000) and `parse_failure_limit`, validates both, and neither reached
the row, the task or the service: there was no column, the endpoint did not pass them, and
`execute_judge_run` read every row and used `judge_rows`' own default limit.

MEASURED on `pmj_da0220514811`, submitted with `max_rows_per_set: 200` over five sets: **5,288 chat
completions in 75 minutes** against an expected 1,000 — it was judging all 5,737 rows. 5.7x the
work, 77 minutes instead of about 13, against a 7B model on the node's only GPU. The operator's
parse-failure limit was silently the module default the whole time.

AND NOTHING REPORTED PROGRESS until completion, so the run was indistinguishable from a stuck one
while `cleanup_stuck_probe_monitor_runs` reaps a judge after 60 minutes of silence. The only way to
tell it was alive was to read miLLM's request log.

MUTATION CONTROLS (each verified to fail this file):
  J1  `limit=row.max_rows_per_set` dropped from the call    → the cap-is-applied test
  J2  the subsample becomes `examples[:limit]`              → the not-a-head-slice test
  J3  the subsample stops balancing                         → the balance test
  J4  the seed ignored                                      → the determinism test
  J5  `parse_failure_limit` not forwarded                    → the limit-forwarded test
  J6  the per-set `record_progress` removed                  → the progress test
  J7  `rows_judged` / `max_rows_per_set` dropped from metrics → the what-it-measured test
"""
import ast
import inspect
from types import SimpleNamespace

import pytest

from src.services import probe_monitor_judge as judge


def _examples(positives: int, negatives: int, sorted_by_class: bool = True):
    """Rows as `build_view` returns them. `sorted_by_class` mimics a corpus written one source at
    a time — the real shape, and what makes a head slice wrong."""
    rows = [SimpleNamespace(messages=[{"role": "user", "content": f"p{i}"}], label=1)
            for i in range(positives)]
    rows += [SimpleNamespace(messages=[{"role": "user", "content": f"n{i}"}], label=0)
             for i in range(negatives)]
    if not sorted_by_class:
        rows = [row for pair in zip(rows[:negatives], rows[negatives:]) for row in pair]
    return rows


class TestTheSubsampleIsNotAHeadSlice:
    """⚠ `rows[:limit]` IS THE WRONG FIX, and this estate has the receipts. These corpora are
    written a source at a time: 14 of OpenHermes' 15 labelled sources occupied a single contiguous
    row range, so an extraction that read the first 7,000 of 189,087 blocks saw 2 sources of 15. A
    judge scored on the head of a sorted file is measured on whatever is at the top and then
    compared against a probe measured on all of it."""

    def test_a_head_slice_of_a_class_sorted_set_would_be_one_class(self):
        """The premise, so the test below is not guarding a hypothetical."""
        rows = _examples(100, 100)
        assert {row.label for row in rows[:50]} == {1}, "the fixture is not class-sorted"

    def test_the_subsample_spans_both_classes(self):
        rows = _examples(100, 100)
        chosen = judge._balanced_subsample(rows, 40, seed=1)
        assert {row.label for row in chosen} == {0, 1}

    def test_it_is_balanced(self):
        rows = _examples(100, 100)
        chosen = judge._balanced_subsample(rows, 40, seed=1)
        positives = sum(1 for row in chosen if row.label == 1)
        assert abs(positives - (len(chosen) - positives)) <= 1, positives

    def test_it_returns_exactly_the_limit(self):
        rows = _examples(100, 100)
        assert len(judge._balanced_subsample(rows, 37, seed=1)) == 37

    def test_it_preserves_the_original_order(self):
        """The labels are returned alongside, so a reordering would pair each judged row with
        another row's label — an AUROC over a permutation."""
        rows = _examples(50, 50)
        chosen = judge._balanced_subsample(rows, 20, seed=1)
        contents = [row.messages[0]["content"] for row in chosen]
        original = [row.messages[0]["content"] for row in rows]
        assert contents == [c for c in original if c in set(contents)]

    def test_it_is_deterministic(self):
        rows = _examples(80, 80)
        first = judge._balanced_subsample(rows, 30, seed=7)
        second = judge._balanced_subsample(rows, 30, seed=7)
        assert [r.messages for r in first] == [r.messages for r in second]

    def test_a_different_seed_chooses_differently(self):
        """The control: if the seed were ignored, the determinism test would pass over a sampler
        that always takes the same rows for a different reason."""
        rows = _examples(200, 200)
        first = judge._balanced_subsample(rows, 40, seed=1)
        second = judge._balanced_subsample(rows, 40, seed=2)
        assert [r.messages for r in first] != [r.messages for r in second]

    def test_an_exhausted_class_fills_from_the_other(self):
        """Returning short would make the cap mean something different per set."""
        rows = _examples(5, 200)
        assert len(judge._balanced_subsample(rows, 40, seed=1)) == 40


class TestTheCapsReachTheService:
    """Asserted by AST, because both defaulted to something harmless and a missing argument was
    silent at every level — which is exactly how they shipped unwired."""

    def _calls(self, function, name):
        tree = ast.parse(inspect.getsource(function).lstrip())
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    def test_the_row_limit_is_passed_to_the_row_reader(self):
        calls = self._calls(judge.execute_judge_run, "_interactions_for")
        assert calls, "execute_judge_run does not read rows through _interactions_for"
        for call in calls:
            names = {keyword.arg for keyword in call.keywords}
            assert "limit" in names, (
                "the row reader is called without limit=, so every row of every set is judged "
                "whatever the request asked for"
            )

    def test_the_limit_comes_from_the_ROW(self):
        """A literal, or the module default, would ignore the request just as thoroughly."""
        for call in self._calls(judge.execute_judge_run, "_interactions_for"):
            limit = next(k.value for k in call.keywords if k.arg == "limit")
            assert isinstance(limit, ast.Attribute), ast.dump(limit)
            assert limit.attr == "max_rows_per_set"

    def test_the_parse_failure_limit_is_forwarded(self):
        calls = self._calls(judge.execute_judge_run, "judge_rows")
        assert calls
        for call in calls:
            names = {keyword.arg for keyword in call.keywords}
            assert "parse_failure_limit" in names, (
                "judge_rows is called without the limit, so the operator's threshold is silently "
                "the module default"
            )

    def test_the_parse_failure_limit_comes_FROM_THE_ROW(self):
        """⚠ A MUTATION SURVIVED HERE TOO. Passing `DEFAULT_PARSE_FAILURE_LIMIT` satisfies the test
        above completely while ignoring the request just as thoroughly as passing nothing —
        "present" is not "correct". The row must appear somewhere in the argument, whether directly
        or inside the `is not None` fallback."""
        for call in self._calls(judge.execute_judge_run, "judge_rows"):
            argument = next(
                keyword.value for keyword in call.keywords if keyword.arg == "parse_failure_limit"
            )
            attributes = {
                node.attr for node in ast.walk(argument) if isinstance(node, ast.Attribute)
            }
            assert "parse_failure_limit" in attributes, (
                f"the limit passed to judge_rows does not read row.parse_failure_limit "
                f"({ast.dump(argument)[:120]}); the request's value is ignored"
            )

    def test_the_reader_accepts_a_limit(self):
        assert "limit" in inspect.signature(judge._interactions_for).parameters

    def test_the_reader_SUBSAMPLES_rather_than_slicing(self):
        """⚠ A MUTATION SURVIVED HERE AND THIS IS THE FIX. `TestTheSubsampleIsNotAHeadSlice` calls
        `_balanced_subsample` directly, so replacing the CALL SITE with `examples[:limit]` left it
        green — the tests proved the helper works and said nothing about whether anything used it.
        The helper being correct is worthless if `_interactions_for` slices instead."""
        tree = ast.parse(inspect.getsource(judge._interactions_for).lstrip())
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_balanced_subsample" in called, (
            "_interactions_for does not call _balanced_subsample; if it slices instead, the judge "
            "is measured on the head of a class-sorted file"
        )

    def test_the_reader_does_not_slice_the_examples(self):
        """The other direction, so a call that does BOTH is caught too."""
        tree = ast.parse(inspect.getsource(judge._interactions_for).lstrip())
        slices = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Slice)
            and isinstance(node.value, ast.Name)
            and node.value.id == "examples"
        ]
        assert not slices, "examples is sliced, which takes the head of the file"

    def test_the_ast_walk_can_tell_a_call_from_a_mention(self):
        def decoy():
            # _interactions_for(db, view, limit=row.max_rows_per_set)
            return "limit="

        assert not self._calls(decoy, "_interactions_for")


class TestProgressIsReportedPerSet:
    def test_execute_judge_run_writes_progress_inside_its_loop(self):
        tree = ast.parse(inspect.getsource(judge.execute_judge_run).lstrip())
        function = tree.body[0]
        loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
        assert loops, "no per-set loop found"
        inside = {
            node.func.id
            for loop in loops
            for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "record_progress" in inside, (
            "progress is not written inside the per-set loop, so a run of many sets reports "
            "nothing until it finishes — and the reaper's judge window is 60 minutes"
        )

    def test_it_goes_through_record_progress_not_a_direct_write(self):
        """`record_progress` refuses to move a terminal row, so an in-flight write cannot
        overwrite an operator's cancellation."""
        source = inspect.getsource(judge.execute_judge_run)
        tree = ast.parse(source.lstrip())
        direct = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "progress"
                for target in node.targets
            )
        ]
        # The final `row.progress = 100.0` on the success path is fine — the row is about to be
        # terminal anyway. What must not exist is a direct write INSIDE the loop.
        loop_nodes = {
            id(node)
            for loop in [n for n in ast.walk(tree) if isinstance(n, ast.For)]
            for node in ast.walk(loop)
        }
        inside = [node for node in direct if id(node) in loop_nodes]
        assert not inside, "progress is assigned directly inside the loop, bypassing the guard"


class TestTheRunRecordsWhatItMeasured:
    def test_the_metrics_name_the_sample(self):
        source = inspect.getsource(judge.execute_judge_run)
        for key in ('"max_rows_per_set"', '"rows_judged"', '"parse_failure_limit"'):
            assert key in source, (
                f"the judge's metrics omit {key}; a comparison that does not state its sample "
                f"cannot be read against the probe's numbers"
            )

    def test_the_schema_and_the_row_agree_on_the_field_names(self):
        """The two were declared and never joined up; pin the join."""
        from src.models.probe_monitor import ProbeMonitorJudgeRun
        from src.schemas.probe_monitor import JudgeRunCreate

        columns = set(ProbeMonitorJudgeRun.__table__.columns.keys())
        for field in ("max_rows_per_set", "parse_failure_limit"):
            assert field in JudgeRunCreate.model_fields, field
            assert field in columns, (
                f"{field} is a request field with no column, so it cannot survive the request — "
                f"which is how both of these were dropped"
            )

    def test_the_endpoint_persists_them(self):
        from src.api.v1.endpoints import probe_monitors

        source = inspect.getsource(probe_monitors.submit_judge_run)
        for field in ("max_rows_per_set", "parse_failure_limit"):
            assert f"{field}=request.{field}" in source, (
                f"the endpoint builds the row without {field}, so the request's value is dropped"
            )
