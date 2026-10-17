"""A janitor that could not clear the one failure it was written for.

`cap_cda1e1da6a0a` was OOM-killed at 45.6%. It then sat in "running" for hours
while this task ran every ten minutes returning `{'cleaned': 0}` — and because
`assert_no_active_gpu_run` counts any non-terminal row, every new capture was
refused with a 409. A permanent lockout, from the janitor written to prevent
exactly that ("Without this, an OOM-killed or pod-restarted capture leaves its
row in 'running' forever" — its own module docstring).

The cause: Celery reports PENDING for any task id it holds no result for, which
is what a killed worker leaves behind and is indistinguishable from a task that
has not started. The old rule counted PENDING as alive, so a dead task looked
alive forever.

`looks_abandoned` had already solved this for the task-queue surface, with a
docstring describing the same symptom on a J-lens fit. It was never carried
across.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.workers import cleanup_stuck_circuit_runs as janitor


def _row(minutes_stale: int, task_id="t-1"):
    return SimpleNamespace(
        celery_task_id=task_id,
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_stale),
    )


def _celery(state, info=None):
    return patch.object(
        janitor.celery_app, "AsyncResult",
        return_value=SimpleNamespace(state=state, info=info))


class TestAPendingTaskOnAStaleRow:
    def test_a_RUNNING_row_stuck_at_PENDING_is_reclaimed(self):
        """The production case, exactly."""
        with _celery("PENDING"):
            assert janitor._is_abandoned(_row(112), "running") is True

    def test_an_ESTIMATING_row_is_treated_the_same(self):
        with _celery("PENDING"):
            assert janitor._is_abandoned(_row(112), "estimating") is True

    def test_a_PENDING_row_is_NOT_condemned_by_row_age(self):
        """The trap in the other direction, and it is worse than the bug.

        "pending" is what a run looks like while QUEUED behind a 45-minute fit.
        Nothing has reported progress, so the row's clock says nothing about the
        task — condemning on it would clear the queue every time one job ran
        long. `looks_abandoned`'s own docstring warns about this.
        """
        with _celery("PENDING"):
            assert janitor._is_abandoned(_row(112), "pending") is False

    def test_a_FRESH_running_row_is_left_alone(self):
        """Under the staleness window, a PENDING task is just starting."""
        with _celery("PENDING"):
            assert janitor._is_abandoned(_row(1), "running") is False


class TestLiveTasksAreNeverKilled:
    @pytest.mark.parametrize("state", ["STARTED", "RETRY", "RECEIVED"])
    def test_a_task_celery_calls_live_is_not_reclaimed(self, state):
        with _celery(state):
            assert janitor._is_abandoned(_row(999), "running") is False

    def test_a_broker_hiccup_does_NOT_reclaim(self):
        """Fail safe: an unreachable broker must never look like a dead task."""
        with patch.object(janitor.celery_app, "AsyncResult",
                          side_effect=RuntimeError("broker down")):
            assert janitor._is_abandoned(_row(999), "running") is False

    def test_a_row_with_no_task_id_at_all_is_reclaimed(self):
        """Past the staleness filter with no task ever recorded — nothing can
        be waiting on it."""
        with _celery("PENDING"):
            assert janitor._is_abandoned(_row(112, task_id=None), "running") is True


class TestEverySTUCKLifecycleUsesTheSameRule:
    """Calibration, faithfulness and recording all hold the single-GPU guard.

    Fixing the capture path alone would leave four siblings able to lock out
    every circuit task in exactly the same way — the "fixed one representative,
    never generalized" pattern this repo keeps hitting.
    """

    def test_no_lifecycle_still_uses_the_old_pending_is_alive_rule(self):
        import inspect

        src = inspect.getsource(janitor)
        assert "_task_is_active" not in src.replace(
            "The rule this replaced", ""), (
            "a lifecycle still treats PENDING as alive; it can wedge the GPU "
            "guard the same way the capture did"
        )

    def test_the_sub_lifecycle_view_carries_its_OWN_task_id(self):
        run = SimpleNamespace(attribution_task_id="attr-9",
                              updated_at=datetime.now(timezone.utc))
        view = janitor._SubLifecycleView(run.attribution_task_id, run.updated_at)

        assert view.celery_task_id == "attr-9", (
            "the view reported the row's primary task instead of the "
            "lifecycle's own — they die independently"
        )
