"""A cancelled probe run and a cancelled judge run actually STOP.

⚠ THE FILENAME MATTERS. `test_cancel_registry_completeness` discovers Shape-A tests by
globbing `test_*cancel*.py`; a cancellation test in a differently-named file satisfies
nothing, and that guard then reports — correctly — that nobody has demonstrated the
scope stops work. Asserting `status == "cancelled"` would also be worthless: the
endpoint wrote that. Something has to prove the WORK stops.

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M152  `stage()` drops `raise_if_cancelled`        → the run test fails
  M153  `judge_rows` stops polling its checker      → the judge test fails
  M154  the checker is called bare (bool discarded) → both tests fail
"""
import os
import uuid

import numpy as np
import pytest
import torch

from tests.unit.test_probe_monitor_pipeline import (  # noqa: F401 - shared fixtures
    D_MODEL,
    arrow_dataset,
    monkeypatch_module,
    pipeline_db,
    prepared,
    tiny_lm,
    tiny_tokenizer,
)


class TestCancellationStopsIt:
    def test_a_requested_cancel_aborts_at_the_NEXT_stage_boundary(
        self, pipeline_db, prepared, tiny_lm, tiny_tokenizer, tmp_path_factory,
        monkeypatch_module,
    ):
        """Cooperative, not a terminating revoke: every worker here is `--pool=solo`,
        which has no pool child to signal and never reads the control queue while busy."""
        from src.core import config as config_module
        from src.core.cancellation import OperatorCancelled, clear_cancel_request, request_cancel
        from src.models.probe_monitor import ProbeMonitorRun
        from src.services import probe_monitor_run as run_module

        artifacts = tmp_path_factory.mktemp("probe_cancel")
        monkeypatch_module.setattr(config_module.settings, "data_dir", artifacts, raising=False)

        run = ProbeMonitorRun(
            model_id=prepared["model"].id,
            train_dataset_id=prepared["train_view"].id,
            eval_dataset_ids=[],
            config={"layers": [1], "rules": ["mean"], "max_length": 64,
                    "val_fraction": 0.25, "seed": 1337, "target_fpr": 0.1},
            status="running",
            environment={},
        )
        pipeline_db.add(run)
        pipeline_db.commit()

        # `db=` MATTERS HERE: without it `request_cancel` opens its own session against
        # the configured database, and this test's rows live in a scratch one — so the
        # flag would be written where the checker never looks, and the test would report
        # "cancellation does not work" for a reason that is purely about wiring.
        request_cancel("probe_monitor_run", run.id, db=pipeline_db)
        pipeline_db.commit()
        try:
            with pytest.raises(OperatorCancelled):
                run_module.execute_probe_run(
                    pipeline_db,
                    run.id,
                    model_loader=lambda row: (tiny_lm, tiny_tokenizer, "LlamaForCausalLM"),
                )
        finally:
            clear_cancel_request("probe_monitor_run", run.id, db=pipeline_db)

    def test_the_guard_is_at_every_stage_boundary(self):
        """AST: `stage()` must CALL the guard. A boundary without it means a cancelled
        run keeps its card until the next one that has it."""
        import ast
        import inspect
        import textwrap

        from src.services import probe_monitor_run as run_module

        source = textwrap.dedent(inspect.getsource(run_module.execute_probe_run))
        tree = ast.parse(source)
        stage_fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "stage"
        )
        called = {
            node.func.id
            for node in ast.walk(stage_fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        # The CALL must be `raise_if_cancelled`, not the bare checker: `__call__` returns
        # a bool and raises nothing, so `check_cancelled()` alone is a checkpoint that
        # cannot stop anything — the same defect as the `guard_allows` mix-up below.
        attrs = {
            node.func.attr
            for node in ast.walk(stage_fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "raise_if_cancelled" in attrs, (
            "stage() does not RAISE on cancellation; a discarded bool stops nothing"
        )
        # ⚠ AND IT MUST NOT BE `guard_allows`. The first version called that — a pure
        # predicate about status WRITES that never reads the cancel flag — and discarded
        # its bool, so every stage boundary had a checkpoint that could not fire.
        assert "guard_allows" not in called, (
            "guard_allows is a status-write predicate, not a cancellation checkpoint"
        )


class TestACancelledJudgeRunStops:
    """The scope existed with NOTHING reading its flag until the completeness guard
    demanded it — a lifecycle that could be started and never stopped."""

    def test_the_loop_polls_its_checker_per_row(self):
        from src.services.probe_monitor_judge import judge_rows

        class _Reply:
            def __init__(self, content):
                self.choices = [
                    type("C", (), {"message": type("M", (), {"content": content})()})()
                ]

        class _Client:
            def __init__(self):
                self.calls = 0
                self.chat = type("Chat", (), {"completions": self})()

            def create(self, **kwargs):
                self.calls += 1
                return _Reply('{"rating": 5}')

        from src.core.cancellation import OperatorCancelled

        class _Checker:
            def __init__(self, stop_after):
                self.polls = 0
                self._stop_after = stop_after

            def raise_if_cancelled(self, detail=""):
                self.polls += 1
                if self.polls > self._stop_after:
                    raise OperatorCancelled("probe_monitor_judge", "pmj_x", "cancelled", detail)

        client = _Client()
        checker = _Checker(stop_after=3)
        interactions = [[{"role": "user", "content": f"row {i}"}] for i in range(50)]
        with pytest.raises(OperatorCancelled):
            judge_rows(client, "qwen", interactions, cancel_check=checker)
        assert client.calls == 3, (
            f"the judge made {client.calls} calls after the cancel; it polls per row, so "
            f"it must stop at the next row boundary"
        )

    def test_without_a_checker_it_still_runs(self):
        """The parameter is optional so a caller that has no scope (a test, a one-off)
        is not forced to invent one."""
        from src.services.probe_monitor_judge import judge_rows

        class _Reply:
            def __init__(self, content):
                self.choices = [
                    type("C", (), {"message": type("M", (), {"content": content})()})()
                ]

        class _Client:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": self})()

            def create(self, **kwargs):
                return _Reply('{"rating": 5}')

        outcome = judge_rows(_Client(), "qwen", [[{"role": "user", "content": "x"}]])
        assert outcome.ratings == [5]

    def test_the_judge_service_CREATES_a_checker_for_its_scope(self):
        """AST: the run must build a checker naming `probe_monitor_judge`, or the route's
        flag is written where nothing reads it."""
        import ast
        import inspect
        import textwrap

        from src.services import probe_monitor_judge as module

        source = textwrap.dedent(inspect.getsource(module.execute_judge_run))
        tree = ast.parse(source)
        names = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "probe_monitor_judge"
        ]
        assert names, "execute_judge_run never names its cancel scope"
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "cancel_checker" in called
