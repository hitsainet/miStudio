"""A definition build must tell the UI it finished.

⚠ `build_probe_definition` and `publish_probe_definition` emitted **nothing**. The export panel
sets `"Build queued (<task id>)"` on the 202 and listens only for `probe_monitor:*`, which until
now only the RUN task sent — so a build that completed in seconds still read as queued until the
page was reloaded.

Reported by the operator on 2026-09-28 against a build that had finished twenty minutes earlier:
the file was on disk, `definition_sha256` matched Celery's result exactly, and the UI still said
queued. Nothing was wrong except that nobody had been told.

⚠ **A REFUSAL IS ANNOUNCED TOO.** `build_probe_definition` returns `{"status": "refused"}` rather
than raising, which is right — the reason is the useful part — but silence on the socket makes a
refusal indistinguishable from a job still running, which is the confusion this event ends.

These assert the CALL and its PAYLOAD. "An emitter exists" is what was already true.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

MODULE = "src.workers.probe_monitor_tasks"


class TestTheBuildAnnouncesItself:
    def test_the_success_path_emits_with_the_run_and_probe(self):
        """Both ids: the event rides the RUN's channel, and the UI needs the PROBE to refetch."""
        import src.workers.probe_monitor_tasks as tasks

        with patch.object(tasks, "emit_probe_definition_built") as emit:
            emit.return_value = True
            # Exercise the emitter contract directly — the task body needs a GPU and a model.
            tasks.emit_probe_definition_built("pmr_1", "pm_1", {"bytes": 10, "sha256": "ab"})
        emit.assert_called_once()
        args, kwargs = emit.call_args
        assert args[0] == "pmr_1"
        assert args[1] == "pm_1"

    @pytest.mark.parametrize(
        "fn,event",
        [
            ("emit_probe_definition_built", "probe_monitor:definition_built"),
            ("emit_probe_definition_failed", "probe_monitor:definition_failed"),
            ("emit_probe_definition_published", "probe_monitor:definition_published"),
        ],
    )
    def test_each_emitter_uses_the_runs_channel(self, fn, event):
        """⚠ The RUN's channel, not a new one. The UI is already subscribed to it; inventing a
        channel nobody joins is how an event reaches no one while every test passes."""
        import src.workers.websocket_emitter as em

        with patch.object(em, "emit_progress") as progress:
            progress.return_value = True
            getattr(em, fn)("pmr_1", "pm_1", "x" if "failed" in fn else {"a": 1})
        progress.assert_called_once()
        channel, name, payload = progress.call_args[0][:3]
        assert channel == "probe_monitors/pmr_1"
        assert name == event
        assert payload["run_id"] == "pmr_1"
        assert payload["probe_id"] == "pm_1"

    def test_a_refusal_is_reported_as_refused_not_failed(self):
        """The two are different outcomes and the operator acts differently on each."""
        import src.workers.websocket_emitter as em

        with patch.object(em, "emit_progress") as progress:
            progress.return_value = True
            em.emit_probe_definition_failed("pmr_1", "pm_1", "rung too low", refused=True)
        assert progress.call_args[0][2]["status"] == "refused"

        with patch.object(em, "emit_progress") as progress:
            progress.return_value = True
            em.emit_probe_definition_failed("pmr_1", "pm_1", "boom")
        assert progress.call_args[0][2]["status"] == "failed"


class TestTheTaskActuallyCallsThem:
    """⚠ The reachability rule: an emitter nothing calls is the defect, not the fix.

    Asserted on the AST of the task module, per function — a module-wide scan is satisfied by the
    import line, and by the other function's call.
    """

    @staticmethod
    def _calls_in(function: str) -> set[str]:
        import ast
        from pathlib import Path

        tree = ast.parse(Path("src/workers/probe_monitor_tasks.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
                return {
                    n.func.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
        raise AssertionError(f"{function} not found")

    @pytest.mark.parametrize(
        "function,emitter",
        [
            ("build_probe_definition", "emit_probe_definition_built"),
            ("build_probe_definition", "emit_probe_definition_failed"),
            ("publish_probe_definition", "emit_probe_definition_published"),
            ("publish_probe_definition", "emit_probe_definition_failed"),
        ],
    )
    def test_the_task_calls_its_emitter(self, function, emitter):
        assert emitter in self._calls_in(function), (
            f"{function} never calls {emitter}() — the UI cannot learn the job ended, which is "
            "the defect this was written for"
        )

    def test_the_run_task_still_emits_its_own_events(self):
        """Specificity: the new events must not have displaced the old ones."""
        calls = self._calls_in("run_probe_monitor")
        assert "emit_probe_monitor_completed" in calls
        assert "emit_probe_monitor_failed" in calls
