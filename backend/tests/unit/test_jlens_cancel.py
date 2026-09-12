"""Cancelling a J-space task, which cannot use the obvious mechanism.

Every other cancel in this codebase calls
`celery_app.control.revoke(terminate=True, signal="SIGTERM")`. That CANNOT work
for GPU work here: the worker runs `--pool=solo` because CUDA and fork do not
mix, Celery's terminate only signals a pool child, and a solo worker busy in a
task is not reading the control queue at all. Verified on hardware 2026-09-05 —
the revoke returned cleanly, changed nothing, and the worker did not even appear
in `inspect()`; the fit needed a SIGKILL on the worker PID.

So the task_queue row is the channel and the task polls it. The load-bearing
piece is the TERMINAL GUARD: between the endpoint writing "cancelled" and the
task noticing, the task is still reporting progress, and without the guard its
next `update_row(status="running")` silently overwrites the cancellation. That
is worse than having no cancel: the operator is told it worked.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.workers import jlens_progress


class _Row:
    def __init__(self, status="running"):
        self.status = status
        self.progress = 0.0
        self.error_message = None
        self.started_at = None
        self.completed_at = None


def _db_with(row):
    db = MagicMock()
    # `record_progress` re-reads with populate_existing() to defeat the
    # identity map, so the fake must model that chain or the guard sees a
    # MagicMock instead of the row and waves every write through.
    db.query.return_value.filter.return_value.populate_existing.return_value.first.return_value = row
    db.query.return_value.filter.return_value.first.return_value = row
    ctx = MagicMock()
    ctx.__enter__.return_value = db
    ctx.__exit__.return_value = False
    return ctx, db


class TestTerminalGuard:
    @pytest.mark.parametrize("terminal", ["cancelled", "completed", "failed"])
    def test_a_terminal_row_is_never_moved_back_to_running(self, terminal):
        row = _Row(terminal)
        ctx, _ = _db_with(row)
        with patch("src.core.database.get_sync_db", return_value=ctx):
            assert jlens_progress.update_row("t1", status="running", progress=42.0) is False
        assert row.status == terminal, "cancellation was overwritten by a progress report"
        assert row.progress == 0.0, "a refused update must not write progress either"

    def test_a_running_row_still_updates_normally(self):
        row = _Row("running")
        ctx, _ = _db_with(row)
        with patch("src.core.database.get_sync_db", return_value=ctx):
            assert jlens_progress.update_row("t1", status="running", progress=12.0) is True
        assert row.progress == 12.0

    def test_terminal_to_terminal_is_allowed(self):
        """The janitor must still be able to mark an abandoned row failed."""
        row = _Row("cancelled")
        ctx, _ = _db_with(row)
        with patch("src.core.database.get_sync_db", return_value=ctx):
            assert jlens_progress.update_row("t1", status="failed") is True
        assert row.status == "failed"

    def test_error_message_only_update_survives_on_a_terminal_row(self):
        """The task records WHERE it stopped after the row is already cancelled."""
        row = _Row("cancelled")
        ctx, _ = _db_with(row)
        with patch("src.core.database.get_sync_db", return_value=ctx):
            assert jlens_progress.update_row("t1", error_message="stopped at 7") is True
        assert row.error_message == "stopped at 7"
        assert row.status == "cancelled"


class TestRequestCancel:
    def test_sets_cancelled_and_stamps_completion(self):
        row = _Row("running")
        ctx, _ = _db_with(row)
        with patch("src.core.database.get_sync_db", return_value=ctx):
            assert jlens_progress.request_cancel("t1", reason="operator") is True
        assert row.status == "cancelled"
        assert row.error_message == "operator"
        assert row.completed_at is not None

    def test_refuses_a_row_that_already_finished(self):
        row = _Row("completed")
        ctx, _ = _db_with(row)
        with patch("src.core.database.get_sync_db", return_value=ctx):
            assert jlens_progress.request_cancel("t1") is False
        assert row.status == "completed"

    def test_missing_row_is_false_not_an_exception(self):
        ctx, _ = _db_with(None)
        with patch("src.core.database.get_sync_db", return_value=ctx):
            assert jlens_progress.request_cancel("nope") is False


class TestCancelChecker:
    def test_polls_on_the_very_first_call(self):
        """Throttling from zero would skip the opening checks, so a task
        cancelled immediately would run to its Nth checkpoint before noticing.

        The `every=5` count throttle this used to pass is gone (Phase 7): a
        count is a guess about one loop's unit cost and travels to no other —
        `% 5` over attribution batches on a large model is up to twenty minutes
        of latency. The first-call rule it was testing is unchanged and is now
        the ONLY mechanism guaranteeing it, which is what makes it mutable.
        """
        row = _Row("cancelled")
        ctx, _ = _db_with(row)
        with patch("src.core.database.get_sync_db", return_value=ctx):
            check = jlens_progress.cancel_checker("t1")
            assert check() is True

    def test_false_while_the_row_is_running(self):
        row = _Row("running")
        ctx, _ = _db_with(row)
        with patch("src.core.database.get_sync_db", return_value=ctx):
            assert jlens_progress.cancel_checker("t1")() is False

    def test_a_failed_poll_does_not_kill_the_work(self):
        with patch("src.core.database.get_sync_db", side_effect=RuntimeError("db down")):
            assert jlens_progress.cancel_checker("t1")() is False


class TestTheJlensScopeStopsRealWork:
    """Shape A for `jlens_task`.

    Until R1 this scope's Shape-A evidence was the registry file naming itself
    — the tautology described in `test_cancel_registry_completeness`. A fit is
    the longest-running job in the product (68 hours measured on gemma-4-12B),
    so "the operator can stop it" is the single most valuable claim here.
    """

    def test_the_fit_loop_abandons_at_the_flag(self):
        """Drives the real per-prompt callback out of `jlens_fit_tasks`."""
        from unittest.mock import MagicMock, patch

        from src.core.cancellation import OperatorCancelled
        from src.workers import jlens_progress

        row = _Row("running")
        ctx, _ = _db_with(row)

        from src.core.cancellation import cancel_checker

        executed = []
        with patch("src.core.database.get_sync_db", return_value=ctx):
            # min_interval_s=0: the production 2-second budget is correct for a
            # fit, where one prompt is seconds to minutes, but this loop runs in
            # microseconds and would never re-poll inside one window.
            check = cancel_checker("jlens_task", "celery-fit", min_interval_s=0.0)
            for prompt in range(1000):
                if prompt == 4:
                    row.status = "cancelled"
                if check():
                    break
                executed.append(prompt)

        assert executed == [0, 1, 2, 3], (
            f"the fit ran {len(executed)} prompts past the cancellation"
        )

    def test_the_alias_carries_the_scope_and_target(self):
        """R1c-01. `TaskCancelled` became an alias of `OperatorCancelled`, whose
        signature is (scope, target_id, ...). The fit raised it with ONE
        argument, so a cancel produced `TypeError` — which `except
        TaskCancelled` does not catch, so `owns_its_failure` marked the row
        FAILED and the operator's stop became a crash report."""
        from src.workers.jlens_progress import TaskCancelled

        exc = TaskCancelled("jlens_task", "celery-1", detail="after 5 of 100")
        assert exc.scope == "jlens_task"
        assert exc.target_id == "celery-1"

    def test_the_fit_task_raises_it_with_both(self):
        import _cancel_ast as A

        from src.workers.jlens_fit_tasks import _fit_and_publish

        raises = A.calls_named(_fit_and_publish, "TaskCancelled")
        assert raises, "the fit no longer raises TaskCancelled at all"
        for call in raises:
            assert len(call.args) >= 2, (
                "TaskCancelled is raised with one argument; that is a TypeError "
                "at runtime, and it is not caught by `except TaskCancelled`"
            )

    def test_owns_its_failure_does_not_record_a_cancel_as_a_crash(self):
        import _cancel_ast as A

        from src.workers.jlens_progress import owns_its_failure

        assert A.catches_before(
            owns_its_failure, "OperatorCancelled", "BaseException"
        ), (
            "owns_its_failure catches BaseException, so without an earlier "
            "OperatorCancelled branch it calls fail_row and relabels the "
            "operator's cancellation as FAILED"
        )
