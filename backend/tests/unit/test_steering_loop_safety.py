"""Regression tests for the 2026-07 event-loop freeze (steering endpoints).

The API froze for 37 minutes because sync result-backend I/O ran on the
asyncio event loop. These tests pin the fix: blocking reads run in a thread
with a hard timeout, a wedged backend yields 503 (not a frozen loop), and the
solo-pool steering worker self-terminates after generation tasks.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.api.v1.endpoints.steering import (
    RESULT_BACKEND_TIMEOUT_SECONDS,
    _read_celery_result,
    get_steering_task_result,
)
from src.workers.steering_tasks import (
    _SELF_EXIT_TASKS,
    _schedule_self_exit,
    _steering_task_postrun,
)


class TestResultEndpointLoopSafety:
    @pytest.mark.asyncio
    async def test_wedged_result_backend_returns_503_not_freeze(self):
        """A result-backend read that never returns must 503 within the
        timeout while the event loop stays responsive."""
        release = threading.Event()

        def wedged_read(task_id):
            release.wait(timeout=60)  # simulates an unbounded kombu retry loop
            return "PENDING", None, False, False, None

        loop_alive = {"ticks": 0}

        async def heartbeat():
            for _ in range(10):
                await asyncio.sleep(0.05)
                loop_alive["ticks"] += 1

        with patch(
            "src.api.v1.endpoints.steering._read_celery_result", side_effect=wedged_read
        ), patch(
            "src.api.v1.endpoints.steering.RESULT_BACKEND_TIMEOUT_SECONDS", 0.5
        ):
            from fastapi import HTTPException

            hb = asyncio.create_task(heartbeat())
            start = time.monotonic()
            with pytest.raises(HTTPException) as exc_info:
                await get_steering_task_result("wedged-task")
            elapsed = time.monotonic() - start
            await hb

        release.set()
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["code"] == "RESULT_BACKEND_TIMEOUT"
        assert elapsed < 5, "timeout must be enforced promptly"
        # The loop kept running while the blocking read was stuck in its thread
        assert loop_alive["ticks"] >= 5

    @pytest.mark.asyncio
    async def test_successful_result_still_served(self):
        def quick_read(task_id):
            return "SUCCESS", {"percent": 100}, True, False, {"sweep_id": "s1"}

        with patch(
            "src.api.v1.endpoints.steering._read_celery_result", side_effect=quick_read
        ):
            response = await get_steering_task_result("ok-task")
        assert response.status.status == "success"
        assert response.result == {"sweep_id": "s1"}

    def test_read_celery_result_is_plain_sync(self):
        """The helper must stay sync (it is executed via asyncio.to_thread)."""
        assert not asyncio.iscoroutinefunction(_read_celery_result)

    def test_default_timeout_is_bounded(self):
        assert 0 < RESULT_BACKEND_TIMEOUT_SECONDS <= 30


class TestSteeringWorkerSelfExit:
    def test_generation_tasks_registered_for_self_exit(self):
        assert _SELF_EXIT_TASKS == {"steering.compare", "steering.sweep", "steering.combined"}

    def test_cleanup_task_not_registered(self):
        assert "steering.cleanup" not in _SELF_EXIT_TASKS

    def test_postrun_ignores_other_workers(self):
        """Without our pidfile, the handler must do nothing (API/celery-worker
        processes also receive task_postrun)."""
        sender = MagicMock()
        sender.name = "steering.sweep"
        with patch(
            "src.workers.steering_tasks._pidfile_is_ours", return_value=False
        ), patch("src.workers.steering_tasks._schedule_self_exit") as scheduled:
            _steering_task_postrun(sender=sender)
            scheduled.assert_not_called()

    def test_postrun_triggers_exit_in_steering_worker(self):
        sender = MagicMock()
        sender.name = "steering.compare"
        with patch(
            "src.workers.steering_tasks._pidfile_is_ours", return_value=True
        ), patch("src.workers.steering_tasks._schedule_self_exit") as scheduled:
            _steering_task_postrun(sender=sender)
            scheduled.assert_called_once()

    def test_postrun_ignores_non_generation_tasks(self):
        sender = MagicMock()
        sender.name = "steering.cleanup"
        with patch(
            "src.workers.steering_tasks._pidfile_is_ours", return_value=True
        ), patch("src.workers.steering_tasks._schedule_self_exit") as scheduled:
            _steering_task_postrun(sender=sender)
            scheduled.assert_not_called()

    def test_schedule_self_exit_sends_sigterm_after_delay(self):
        with patch("src.workers.steering_tasks.__name__", create=True), patch(
            "os.kill"
        ) as mock_kill:
            _schedule_self_exit(delay_seconds=0.05)
            time.sleep(0.5)
            assert mock_kill.called
            args = mock_kill.call_args[0]
            import os as _os
            import signal as _signal
            assert args[0] == _os.getpid()
            assert args[1] == _signal.SIGTERM
