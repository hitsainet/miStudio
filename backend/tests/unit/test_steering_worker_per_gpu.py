"""One steering worker per GPU, with its own queue, PID file and log (multi-GPU Phase 3).

In per-card mode a steering request goes to one card's worker — the card it
names, or the idle card with the most free memory — which the API spawns with
``MISTUDIO_WORKER_GPU_UUID`` and finds and kills by PID, never by a command-line
pattern. The worker side reads its card from its environment for its PID file
and busy marker, so two cards' workers cannot clear each other's marker.

MUTATION CONTROLS (2026-09-14; each alone, restored byte-identically, sha256 checked,
`git diff` clean) — all red:
  T1 a spawned card worker's PID is not recorded         -> recorded under its card; no-pattern registration
  T2 the worker is not told its card                     -> told its card and consumes its queue
  T3 killing one card's workers kills every card's       -> one card's workers are killed and the others are not
  T4 no queue option for a card                          -> all nine per-card submit-endpoint cases
  T5 a card worker writes the single worker's marker     -> each card worker has its own busy marker
  T6 a card worker reads the single worker's PID file    -> a card worker knows its own PID file
  T7 the internal reconcile ignores per-card mode        -> the internal endpoint reconciles per card
"""

import ast
import asyncio
import fnmatch
import os
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.api.v1 import steering_workers as W
from src.core.config import settings
from src.services import gpu_placement
from src.services import steering_worker_card as P
from src.services.gpu_placement import GpuCard
from src.services.gpu_worker_queues import STEERING_QUEUE_PATTERN
from src.workers.gpu_supervisor import WORKER_GPU_ENV

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(0, TI_UUID, "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_900)
RTX = GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, 21_500)
CARDS = [TI, RTX]


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "_run_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _forget_spawned():
    W.SPAWNED_PIDS_BY_CARD.clear()
    W.SPAWN_START_TICKS.clear()
    yield
    W.SPAWNED_PIDS_BY_CARD.clear()
    W.SPAWN_START_TICKS.clear()


def _pretend_spawned(monkeypatch, *pids):
    """Fake PIDs recorded at spawn, each still the process spawned under it.

    Since review round 2 a recorded PID is killed only while /proc shows the process it
    was recorded for (test_gpu_phase3_review_r2_workers.py); these tests are about WHICH
    card's workers are killed, so their fake PIDs are given a matching identity.
    """
    real = W.process_start_ticks
    monkeypatch.setattr(W, "process_start_ticks", lambda pid: 7 if pid in pids else real(pid))
    for pid in pids:
        W.SPAWN_START_TICKS[pid] = 7


class TestEveryCardHasItsOwnWorker:
    def test_names_differ_per_card_and_from_the_single_worker(self, run_dir):
        for name in ("queue", "hostname", "pid_file", "log_file", "busy_marker"):
            fn = getattr(P, name)
            assert len({fn(TI_UUID), fn(RTX_UUID), fn(None)}) == 3, name

    def test_the_single_worker_keeps_its_names(self, run_dir):
        assert (P.queue(None), P.hostname(None)) == ("steering", "steering@%h")
        assert (P.pid_file(None).name, P.log_file(None).name, P.busy_marker(None).name) == (
            "mistudio-celery-steering.pid", "celery-steering.log", "steering-worker-busy.json")

    def test_a_card_worker_drains_its_own_queue_then_the_legacy_one(self):
        assert P.consumed_queues(TI_UUID).split(",") == [P.queue(TI_UUID), "steering"]

    def test_a_cards_queue_is_the_one_the_rescue_sweeps(self):
        assert fnmatch.fnmatchcase(P.queue(RTX_UUID), STEERING_QUEUE_PATTERN)


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid


class TestSpawning:
    def test_the_worker_is_told_its_card_and_consumes_its_queue(self, run_dir, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        seen = []

        def popen(argv, **kwargs):
            seen.append((argv, kwargs))
            return FakeProcess(4242)

        assert W.spawn_card_worker(RTX_UUID, popen=popen) == 4242
        (argv, kwargs), = seen
        assert argv[argv.index("-Q") + 1] == P.consumed_queues(RTX_UUID)
        assert f"--hostname={P.hostname(RTX_UUID)}" in argv
        assert f"--pidfile={P.pid_file(RTX_UUID)}" in argv
        assert kwargs["env"][WORKER_GPU_ENV] == RTX_UUID
        assert "CUDA_VISIBLE_DEVICES" not in kwargs["env"], "a card worker must still see every card"
        assert kwargs["stdout"].name == str(P.log_file(RTX_UUID))

    def test_the_pid_is_recorded_under_its_card(self, run_dir):
        W.spawn_card_worker(TI_UUID, popen=lambda argv, **kw: FakeProcess(11))
        W.spawn_card_worker(RTX_UUID, popen=lambda argv, **kw: FakeProcess(22))
        assert W.SPAWNED_PIDS_BY_CARD == {TI_UUID: {11}, RTX_UUID: {22}}


class TestKillingByRecordedPid:
    def test_one_cards_workers_are_killed_and_the_others_are_not(self, monkeypatch):
        killed = []
        monkeypatch.setattr(W.os, "kill", lambda pid, sig: killed.append((pid, sig)))
        W.SPAWNED_PIDS_BY_CARD.update({TI_UUID: {11}, RTX_UUID: {22}})
        _pretend_spawned(monkeypatch, 11, 22)

        assert W.kill_card_workers(TI_UUID) == 1
        assert killed == [(11, signal.SIGKILL)]
        assert W.SPAWNED_PIDS_BY_CARD == {RTX_UUID: {22}}

    def test_every_card_when_none_is_named(self, monkeypatch):
        killed = []
        monkeypatch.setattr(W.os, "kill", lambda pid, sig: killed.append(pid))
        W.SPAWNED_PIDS_BY_CARD.update({TI_UUID: {11}, RTX_UUID: {22}})
        _pretend_spawned(monkeypatch, 11, 22)
        assert W.kill_card_workers() == 2 and sorted(killed) == [11, 22]

    def test_no_worker_is_found_or_killed_by_a_command_line_pattern(self):
        tree = ast.parse(Path(W.__file__).read_text())
        literals = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not {"pgrep", "pkill"} & literals
        popens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "popen"]
        assert len(popens) == 1
        registrations = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add"
                         and "SPAWNED_PIDS_BY_CARD" in ast.dump(n)]
        assert len(registrations) == len(popens), "a spawned worker whose pid is not recorded can never be reaped"


class TestFindingAWorker:
    def test_by_its_cards_pid_file(self, run_dir):
        P.pid_file(TI_UUID).write_text(str(os.getpid()))
        assert W.card_worker_pid(TI_UUID) == os.getpid()
        assert W.card_worker_pid(RTX_UUID) is None

    def test_a_dead_pid_is_not_a_worker(self, run_dir):
        P.pid_file(TI_UUID).write_text("2147480000")
        assert W.card_worker_pid(TI_UUID) is None


class TestChoosingTheCard:
    def test_a_named_card(self):
        assert W.choose_steering_card(TI_UUID.lower(), CARDS, {}) == TI_UUID

    def test_auto_takes_the_most_free_idle_card(self):
        assert W.choose_steering_card("auto", CARDS, {}) == RTX_UUID
        assert W.choose_steering_card("auto", CARDS, {RTX_UUID: "training:t:1"}) == TI_UUID

    def test_every_card_busy_still_names_one(self):
        assert W.choose_steering_card("auto", CARDS, {RTX_UUID: "a", TI_UUID: "b"}) == RTX_UUID

    def test_no_card(self):
        assert W.choose_steering_card("auto", [], {}) is None

    def test_single_mode_names_none(self, monkeypatch):
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        assert asyncio.run(W.steering_card_for("auto")) is None


class TestEnsuringACardsWorker:
    def test_a_worker_mid_generation_is_left_alone(self, run_dir, monkeypatch):
        from src.workers import steering_worker_state

        P.pid_file(RTX_UUID).write_text(str(os.getpid()))
        P.busy_marker(RTX_UUID).write_text(f'{{"pid": {os.getpid()}, "task_id": "t", "ts": 9e12}}')
        monkeypatch.setattr(steering_worker_state.time, "time", lambda: 9e12)
        spawned = []
        monkeypatch.setattr(W, "spawn_card_worker", lambda card: spawned.append(card))

        assert asyncio.run(W.ensure_card_worker(RTX_UUID)) == (True, os.getpid())
        assert spawned == []

    def test_an_idle_worker_is_replaced_by_a_fresh_one(self, run_dir, monkeypatch):
        """REAL processes: since review round 2 a PID file names a worker only when /proc says
        its process started before the file was written, so fake PIDs name no worker at all."""
        import subprocess

        real_kill = os.kill
        old_worker = subprocess.Popen(["/bin/sleep", "120"])
        signals = []

        def fake_kill(pid, sig):
            if sig == 0:
                return real_kill(pid, sig)
            signals.append((pid, sig))
            if sig == signal.SIGKILL:
                P.pid_file(RTX_UUID).unlink(missing_ok=True)

        try:
            monkeypatch.setattr(W.os, "kill", fake_kill)
            monkeypatch.setattr(W.asyncio, "sleep", AsyncMock())
            P.pid_file(RTX_UUID).write_text(str(old_worker.pid))

            def spawn(card):
                P.pid_file(card).write_text(str(os.getpid()))   # the fresh worker: a process that exists

            monkeypatch.setattr(W, "spawn_card_worker", spawn)
            assert asyncio.run(W.ensure_card_worker(RTX_UUID)) == (True, os.getpid())
            assert (old_worker.pid, signal.SIGKILL) in signals
        finally:
            real_kill(old_worker.pid, signal.SIGKILL)
            old_worker.wait(timeout=10)

    def test_with_no_card_the_single_workers_ensure_runs(self):
        single = AsyncMock(return_value=(True, 5))
        assert asyncio.run(W.ensure_steering_worker(None, single)) == (True, 5)
        single.assert_awaited_once_with()


class TestTheSubmitEndpointsUseTheCardsWorker:
    @pytest.mark.parametrize("requested, card", [("1", RTX_UUID), ("auto", RTX_UUID), (TI_UUID, TI_UUID)])
    @pytest.mark.parametrize("kind", ["compare", "sweep", "combined"])
    def test_per_card_mode(self, monkeypatch, kind, requested, card):
        from src.api.v1.endpoints import steering as endpoints
        from tests.unit.test_steering_gpu_placement import ENDPOINTS, _endpoint_environment, _endpoint_request

        monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
        monkeypatch.setattr(gpu_placement, "list_cards", lambda: list(CARDS))
        monkeypatch.setattr(W, "list_cards", lambda: list(CARDS))
        monkeypatch.setattr(W, "_live_leases", lambda: {})
        ensure = AsyncMock(return_value=(True, 99))
        monkeypatch.setattr(W, "ensure_card_worker", ensure)
        handler_name, task_name = ENDPOINTS[kind]
        with _endpoint_environment(task_name) as env:
            asyncio.run(getattr(endpoints, handler_name)(
                _endpoint_request(kind, requested), env.http_request, db=AsyncMock()))

        ensure.assert_awaited_once_with(card)
        assert env.ensure.await_count == 0, "the single worker was started in per-card mode"
        assert env.task.apply_async.call_count == 1
        assert env.task.apply_async.call_args.kwargs["queue"] == P.queue(card)

    @pytest.mark.parametrize("kind", ["compare", "sweep", "combined"])
    def test_single_mode_is_unchanged(self, monkeypatch, kind):
        from src.api.v1.endpoints import steering as endpoints
        from tests.unit.test_steering_gpu_placement import ENDPOINTS, _endpoint_environment, _endpoint_request

        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        monkeypatch.setattr(gpu_placement, "list_cards", lambda: list(CARDS))
        handler_name, task_name = ENDPOINTS[kind]
        with _endpoint_environment(task_name) as env:
            asyncio.run(getattr(endpoints, handler_name)(
                _endpoint_request(kind, "auto"), env.http_request, db=AsyncMock()))

        assert env.ensure.await_count == 1
        assert "queue" not in env.task.apply_async.call_args.kwargs


class TestTheWorkerSide:
    def test_a_card_worker_knows_its_own_pid_file(self, run_dir, monkeypatch):
        from src.workers import steering_tasks

        monkeypatch.setenv(WORKER_GPU_ENV, TI_UUID)
        P.pid_file(TI_UUID).write_text(str(os.getpid()))
        assert steering_tasks._pidfile_is_ours() is True
        monkeypatch.setenv(WORKER_GPU_ENV, RTX_UUID)
        assert steering_tasks._pidfile_is_ours() is False

    def test_each_card_worker_has_its_own_busy_marker(self, run_dir, monkeypatch):
        from src.workers import steering_worker_state as state

        monkeypatch.setenv(WORKER_GPU_ENV, TI_UUID)
        state.write_busy_marker("task-ti")
        monkeypatch.delenv(WORKER_GPU_ENV)  # the API process reading it

        assert state.read_busy_marker(TI_UUID)["task_id"] == "task-ti"
        assert state.read_busy_marker(RTX_UUID) is None
        assert state.read_busy_marker() is None, "a card worker wrote the single worker's marker"


class TestExitingSteeringMode:
    def test_every_cards_worker_is_killed_by_a_tracked_pid(self, run_dir, monkeypatch):
        """One recorded at spawn (the 3080 Ti's), one found only in its PID file (the 3090's)."""
        from src.api.v1.endpoints import steering as endpoints

        signals = []
        monkeypatch.setattr(W.os, "kill", lambda pid, sig: signals.append((pid, sig)))
        monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
        monkeypatch.setattr(W, "list_cards", lambda: list(CARDS))
        monkeypatch.setattr(endpoints, "_is_steering_worker_running", lambda: (False, None))
        monkeypatch.setattr(endpoints, "_get_gpu_memory_mb", lambda: None)
        monkeypatch.setattr(endpoints.asyncio, "sleep", AsyncMock())
        W.SPAWNED_PIDS_BY_CARD[TI_UUID] = {11}
        _pretend_spawned(monkeypatch, 11)
        P.pid_file(RTX_UUID).write_text(str(os.getpid()))

        result = asyncio.run(endpoints.exit_steering_mode())

        assert (11, signal.SIGKILL) in signals and (os.getpid(), signal.SIGKILL) in signals
        assert result["success"] is True and not result.get("already_inactive")
        assert not P.pid_file(RTX_UUID).exists()

    def test_single_mode_reads_no_card(self, monkeypatch):
        from src.api.v1.endpoints import steering as endpoints

        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        monkeypatch.setattr(W, "list_cards", lambda: (_ for _ in ()).throw(AssertionError("read the cards")))
        monkeypatch.setattr(endpoints, "_is_steering_worker_running", lambda: (False, None))
        assert asyncio.run(endpoints.exit_steering_mode())["already_inactive"] is True


class TestReconcile:
    def _run(self, *, depths, running=(), rescue=None):
        ensure = AsyncMock(return_value=(True, 7))
        rescued = []

        async def depth(name):
            return depths.get(name, 0)

        def rescue_fn(live):
            rescued.append(live)
            return rescue or {}

        with patch.object(W, "card_worker_pid", lambda uuid: 1 if uuid in running else None), \
             patch.object(W, "_live_leases", lambda: {}):
            result = asyncio.run(W.reconcile_card_workers(
                depth=depth, ensure=ensure, rescue=rescue_fn, inventory=lambda: list(CARDS)))
        return result, ensure, rescued

    def test_a_cards_queue_with_work_and_no_worker_spawns_that_cards_worker(self):
        result, ensure, rescued = self._run(depths={P.queue(TI_UUID): 2})
        ensure.assert_awaited_once_with(TI_UUID)
        assert result["action"] == "spawned"
        assert rescued == [[TI_UUID, RTX_UUID]]

    def test_a_running_worker_is_not_respawned(self):
        result, ensure, _ = self._run(depths={P.queue(TI_UUID): 2}, running={TI_UUID})
        ensure.assert_not_awaited()
        assert result["action"] == "none"

    def test_legacy_work_with_no_worker_at_all_gets_one(self):
        _, ensure, _ = self._run(depths={"steering": 1})
        ensure.assert_awaited_once_with(RTX_UUID)

    def test_the_internal_endpoint_reconciles_per_card_in_per_card_mode(self, monkeypatch):
        from src import main

        monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
        reconcile = AsyncMock(return_value={"status": "ok", "action": "none"})
        monkeypatch.setattr(W, "reconcile_card_workers", reconcile)
        assert asyncio.run(main.reconcile_steering_worker(x_internal_token=settings.internal_api_secret)) == {
            "status": "ok", "action": "none"}
        reconcile.assert_awaited_once_with()


class TestTheModeEndpointsKnowEveryCard:
    """Phase 3 review round 1, item 2. `/steering/mode` and `/steering/enter-mode` knew only
    the single worker: in per-card mode the status said steering was off while card
    workers ran, and enter-mode started the SINGLE worker — no card of its own,
    consuming only the legacy queue every request bypasses. Exit-mode could only stop
    every card at once.

    MUTATION CONTROLS (2026-09-14; each alone, restored byte-identically, sha256
    checked, `git diff` clean) — all red:
      ST1 get_steering_mode_status skips the per-card branch
            -> test_status_lists_every_cards_worker_and_is_active_while_any_runs
      ST2 enter_steering_mode skips the per-card branch (starts the single worker)
            -> test_enter_mode_starts_the_auto_cards_worker_not_the_single_one
      ST3 exit-mode for one card stops every card instead
            -> test_exit_mode_for_one_card_stops_only_that_cards_worker
      ST4 _is_steering_worker_running looks the worker up with `pgrep -f steering@` again
            -> test_the_single_worker_is_never_found_by_a_command_line_pattern
      ST5 pid_alive treats a zombie as alive
            -> test_a_zombie_named_in_a_pid_file_is_not_a_worker
      ST6 enter_card_workers restarts a card whose worker already runs
            -> test_enter_mode_all_starts_each_card_without_a_worker_and_leaves_a_running_one
    """

    @pytest.fixture
    def per_card(self, run_dir, monkeypatch):
        from src.api.v1.endpoints import steering as endpoints

        monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
        monkeypatch.setattr(gpu_placement, "list_cards", lambda: list(CARDS))
        monkeypatch.setattr(W, "list_cards", lambda: list(CARDS))
        monkeypatch.setattr(W, "_live_leases", lambda: {})
        monkeypatch.setattr(endpoints, "PID_FILE", str(run_dir / "single.pid"))
        monkeypatch.setattr(endpoints, "_get_gpu_memory_mb", lambda: None)
        monkeypatch.setattr(endpoints, "_spawn_steering_worker",
                            lambda: pytest.fail("the single worker was started in per-card mode"))
        return endpoints

    def test_status_lists_every_cards_worker_and_is_active_while_any_runs(self, per_card):
        import json
        import time

        P.pid_file(TI_UUID).write_text(str(os.getpid()))
        P.busy_marker(TI_UUID).write_text(json.dumps({"pid": os.getpid(), "task_id": "gen-1", "ts": time.time()}))

        status = asyncio.run(per_card.get_steering_mode_status())

        assert status["active"] is True and status["worker_pid"] == os.getpid()
        assert [(w["card"], w["worker_pid"], w["busy"], w["task_id"]) for w in status["workers"]] == [
            (TI_UUID, os.getpid(), True, "gen-1"), (RTX_UUID, None, False, None)]

    def test_status_is_off_when_no_card_has_a_worker(self, per_card):
        status = asyncio.run(per_card.get_steering_mode_status())
        assert status["active"] is False and status["worker_pid"] is None
        assert [w["worker_pid"] for w in status["workers"]] == [None, None]

    def test_a_zombie_named_in_a_pid_file_is_not_a_worker(self, run_dir):
        import subprocess
        import time

        exited = subprocess.Popen(["/bin/true"])
        deadline = time.monotonic() + 10
        while Path(f"/proc/{exited.pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
            assert time.monotonic() < deadline, "the child never exited"
            time.sleep(0.01)
        try:
            P.pid_file(TI_UUID).write_text(str(exited.pid))
            assert W.card_worker_pid(TI_UUID) is None, "an exited worker read as running"
        finally:
            exited.wait()

    def test_enter_mode_starts_the_auto_cards_worker_not_the_single_one(self, per_card, monkeypatch):
        ensure = AsyncMock(return_value=(True, 4242))
        monkeypatch.setattr(W, "ensure_card_worker", ensure)

        result = asyncio.run(per_card.enter_steering_mode())

        ensure.assert_awaited_once_with(RTX_UUID)
        assert (result["success"], result["worker_pid"], result["already_active"]) == (True, 4242, False)
        assert [w["card"] for w in result["workers"]] == [RTX_UUID]

    def test_enter_mode_all_starts_each_card_without_a_worker_and_leaves_a_running_one(self, per_card, monkeypatch):
        P.pid_file(TI_UUID).write_text(str(os.getpid()))
        ensure = AsyncMock(return_value=(True, 4242))
        monkeypatch.setattr(W, "ensure_card_worker", ensure)

        result = asyncio.run(per_card.enter_steering_mode(gpu="all"))

        ensure.assert_awaited_once_with(RTX_UUID)
        assert [(w["card"], w["already_active"], w["worker_pid"]) for w in result["workers"]] == [
            (TI_UUID, True, os.getpid()), (RTX_UUID, False, 4242)]
        assert result["success"] is True and result["already_active"] is False

    def test_enter_mode_for_a_card_by_index(self, per_card, monkeypatch):
        ensure = AsyncMock(return_value=(True, 5))
        monkeypatch.setattr(W, "ensure_card_worker", ensure)
        asyncio.run(per_card.enter_steering_mode(gpu="0"))
        ensure.assert_awaited_once_with(TI_UUID)

    def test_enter_mode_refuses_a_card_this_node_does_not_have(self, per_card, monkeypatch):
        from fastapi import HTTPException

        ensure = AsyncMock(return_value=(True, 5))
        monkeypatch.setattr(W, "ensure_card_worker", ensure)
        with pytest.raises(HTTPException) as refused:
            asyncio.run(per_card.enter_steering_mode(gpu="GPU-deadbeef-0000-0000-0000-000000000000"))
        assert refused.value.status_code == 400
        ensure.assert_not_awaited()

    def test_exit_mode_for_one_card_stops_only_that_cards_worker(self, per_card, monkeypatch):
        killed = []
        monkeypatch.setattr(W.os, "kill", lambda pid, sig: sig and killed.append((pid, sig)))
        monkeypatch.setattr(per_card.asyncio, "sleep", AsyncMock())
        single = AsyncMock(return_value=0)
        monkeypatch.setattr(per_card, "_kill_orphan_steering_workers", single)
        W.SPAWNED_PIDS_BY_CARD.update({TI_UUID: {11}, RTX_UUID: {22}})
        P.pid_file(TI_UUID).write_text("11")
        P.pid_file(RTX_UUID).write_text(str(os.getpid()))

        result = asyncio.run(per_card.exit_steering_mode(gpu=RTX_UUID))

        assert killed == [(os.getpid(), signal.SIGKILL)]
        assert W.SPAWNED_PIDS_BY_CARD == {TI_UUID: {11}} and P.pid_file(TI_UUID).exists()
        assert not P.pid_file(RTX_UUID).exists()
        single.assert_not_awaited()
        assert result["success"] is True and result["already_inactive"] is False

    def test_the_gpu_query_parameter_reaches_the_mode_routes(self, per_card, monkeypatch):
        from fastapi.testclient import TestClient

        from src.main import app

        entered, exited = [], []
        monkeypatch.setattr(per_card, "_enter_card_steering_mode",
                            AsyncMock(side_effect=lambda gpu: entered.append(gpu) or {"success": True}))
        monkeypatch.setattr(per_card, "_exit_card_steering_mode",
                            AsyncMock(side_effect=lambda card: exited.append(card) or {"success": True}))
        client = TestClient(app, raise_server_exceptions=False)

        assert client.post("/api/v1/steering/enter-mode?gpu=all").status_code == 200
        assert client.post("/api/v1/steering/exit-mode?gpu=1").status_code == 200
        assert (entered, exited) == (["all"], [RTX_UUID])

    def test_the_single_worker_is_never_found_by_a_command_line_pattern(self, tmp_path, monkeypatch):
        import subprocess

        from src.api.v1.endpoints import steering as endpoints

        tree = ast.parse(Path(endpoints.__file__).read_text())
        literals = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not {"pgrep", "pkill"} & literals
        monkeypatch.setattr(endpoints, "PID_FILE", str(tmp_path / "absent.pid"))
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("looked the worker up in the process table"))
        assert endpoints._is_steering_worker_running() == (False, None)
