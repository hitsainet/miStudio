"""
Pins for the steering worker crash-race fixes (2026-07-18).

Three defects, three guards:
1. Mid-task SIGTERM raised SystemExit inside a running task and crashed the
   solo pool ("cannot unpack non-iterable ExceptionInfo") -> the handler now
   DEFERS when a steering task is executing.
2. _ensure_steering_worker_running SIGKILLed a mid-generation worker on every
   submit, stranding its acks_late message for the 12h visibility timeout ->
   a fresh busy marker owned by the live worker blocks the kill.
3. Nothing respawned a worker for stranded queue messages -> the internal
   reconcile endpoint spawns iff queue non-empty and no worker alive.
"""

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workers import steering_worker_state as worker_state


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_state, "_marker_path", lambda: tmp_path / "busy.json")
    worker_state.busy_task_id = None
    worker_state.shutdown_deferred = False
    yield
    worker_state.busy_task_id = None
    worker_state.shutdown_deferred = False


# ── 1. Cooperative SIGTERM ──────────────────────────────────────────────────

class TestCooperativeSignalHandler:
    def test_mid_task_sigterm_defers_instead_of_raising(self):
        import signal as signal_mod

        from src.services.steering_service import _signal_handler

        worker_state.busy_task_id = "task-123"
        # Must NOT raise SystemExit mid-task — that was the crash.
        _signal_handler(signal_mod.SIGTERM, None)
        assert worker_state.shutdown_deferred is True

    def test_idle_sigterm_still_exits(self):
        import signal as signal_mod

        from src.services.steering_service import _signal_handler

        worker_state.busy_task_id = None
        with patch("src.services.steering_service._emergency_gpu_cleanup"), \
             patch("src.services.steering_service.signal.signal"):
            with pytest.raises(SystemExit):
                _signal_handler(signal_mod.SIGTERM, None)

    def test_postrun_completes_deferred_shutdown_for_any_steering_task(self):
        """A deferred shutdown must exit even after a non-generation task
        (e.g. steering.cleanup), or the worker would linger consuming."""
        from src.workers.steering_tasks import _steering_task_postrun

        worker_state.shutdown_deferred = True
        sender = MagicMock()
        sender.name = "steering.cleanup"
        with patch("src.workers.steering_tasks._pidfile_is_ours", return_value=True), \
             patch("src.workers.steering_tasks._schedule_self_exit") as exit_mock:
            _steering_task_postrun(sender=sender)
        exit_mock.assert_called_once()

    def test_prerun_postrun_marker_lifecycle(self):
        from src.workers.steering_tasks import (
            _steering_task_postrun,
            _steering_task_prerun,
        )

        sender = MagicMock()
        sender.name = "steering.combined"
        with patch("src.workers.steering_tasks._pidfile_is_ours", return_value=True), \
             patch("src.workers.steering_tasks._schedule_self_exit"):
            _steering_task_prerun(sender=sender, task_id="t1")
            assert worker_state.busy_task_id == "t1"
            assert worker_state.read_busy_marker()["task_id"] == "t1"
            _steering_task_postrun(sender=sender)
            assert worker_state.busy_task_id is None
            assert worker_state.read_busy_marker() is None


# ── 2. Busy-aware ensure ────────────────────────────────────────────────────

class TestBusyAwareEnsure:
    @pytest.mark.asyncio
    async def test_fresh_marker_blocks_the_kill(self):
        from src.api.v1.endpoints.steering import _ensure_steering_worker_running

        worker_state.write_busy_marker("t-live")
        marker = json.loads(worker_state._marker_path().read_text())
        live_pid = marker["pid"]

        with patch(
            "src.api.v1.endpoints.steering._is_steering_worker_running",
            return_value=(True, live_pid),
        ), patch("src.api.v1.endpoints.steering.os.kill") as kill_mock:
            ok, pid = await _ensure_steering_worker_running()
        assert (ok, pid) == (True, live_pid)
        kill_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_marker_does_not_block(self):
        from src.api.v1.endpoints.steering import _ensure_steering_worker_running

        stale = {"pid": 4242, "task_id": "t-old",
                 "ts": time.time() - worker_state.BUSY_MARKER_STALE_SECONDS - 10}
        worker_state._marker_path().write_text(json.dumps(stale))

        calls = iter([(True, 4242)])
        with patch(
            "src.api.v1.endpoints.steering._is_steering_worker_running",
            side_effect=lambda: next(calls, (True, 5555)),
        ), patch("src.api.v1.endpoints.steering.os.kill") as kill_mock, \
             patch("src.api.v1.endpoints.steering.subprocess.run"), \
             patch("src.api.v1.endpoints.steering.subprocess.Popen") as popen_mock, \
             patch("src.api.v1.endpoints.steering.open", create=True, new=MagicMock()), \
             patch("src.api.v1.endpoints.steering.asyncio.sleep", new=AsyncMock()):
            ok, pid = await _ensure_steering_worker_running()
        kill_mock.assert_called()  # stale marker => hung worker => killed
        popen_mock.assert_called()

    @pytest.mark.asyncio
    async def test_dead_workers_leftover_marker_does_not_shield_successor(self):
        from src.api.v1.endpoints.steering import _ensure_steering_worker_running

        # Fresh marker, but written by a pid that is NOT the live worker.
        worker_state._marker_path().write_text(
            json.dumps({"pid": 99999, "task_id": "t-dead", "ts": time.time()})
        )
        calls = iter([(True, 1234)])
        with patch(
            "src.api.v1.endpoints.steering._is_steering_worker_running",
            side_effect=lambda: next(calls, (True, 5555)),
        ), patch("src.api.v1.endpoints.steering.os.kill") as kill_mock, \
             patch("src.api.v1.endpoints.steering.subprocess.run"), \
             patch("src.api.v1.endpoints.steering.subprocess.Popen"), \
             patch("src.api.v1.endpoints.steering.open", create=True, new=MagicMock()), \
             patch("src.api.v1.endpoints.steering.asyncio.sleep", new=AsyncMock()):
            await _ensure_steering_worker_running()
        kill_mock.assert_called()


# ── 3. Reconcile endpoint ───────────────────────────────────────────────────

class TestReconcileEndpoint:
    def _client(self):
        from fastapi.testclient import TestClient

        from src.core.config import settings
        from src.main import app

        return TestClient(app, raise_server_exceptions=False), settings

    def test_requires_internal_token(self):
        client, _ = self._client()
        response = client.post("/api/internal/steering/reconcile-worker")
        assert response.status_code == 403

    def test_spawns_only_when_stranded(self):
        client, settings = self._client()
        headers = {"X-Internal-Token": settings.internal_api_secret}

        with patch("src.api.v1.endpoints.steering._steering_queue_depth",
                   new=AsyncMock(return_value=2)), \
             patch("src.api.v1.endpoints.steering._is_steering_worker_running",
                   return_value=(False, None)), \
             patch("src.api.v1.endpoints.steering._ensure_steering_worker_running",
                   new=AsyncMock(return_value=(True, 777))) as ensure_mock:
            response = client.post(
                "/api/internal/steering/reconcile-worker", headers=headers)
        assert response.status_code == 200
        assert response.json()["action"] == "spawned"
        ensure_mock.assert_awaited_once()

    def test_no_spawn_when_queue_empty_or_worker_alive(self):
        client, settings = self._client()
        headers = {"X-Internal-Token": settings.internal_api_secret}

        with patch("src.api.v1.endpoints.steering._steering_queue_depth",
                   new=AsyncMock(return_value=0)), \
             patch("src.api.v1.endpoints.steering._ensure_steering_worker_running",
                   new=AsyncMock()) as ensure_mock:
            response = client.post(
                "/api/internal/steering/reconcile-worker", headers=headers)
        assert response.json()["action"] == "none"
        ensure_mock.assert_not_awaited()

        with patch("src.api.v1.endpoints.steering._steering_queue_depth",
                   new=AsyncMock(return_value=3)), \
             patch("src.api.v1.endpoints.steering._is_steering_worker_running",
                   return_value=(True, 42)), \
             patch("src.api.v1.endpoints.steering._ensure_steering_worker_running",
                   new=AsyncMock()) as ensure_mock:
            response = client.post(
                "/api/internal/steering/reconcile-worker", headers=headers)
        assert response.json()["action"] == "none"
        ensure_mock.assert_not_awaited()
