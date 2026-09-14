"""Multi-GPU Phase 3, review round 2 — the supervisor's drain and the steering workers' PIDs.

The supervisor tests run the REAL ``supervise``/``_drain`` on a fake clock and a fake
Popen through the real signal handler (the fakes are the round-1 module's). The PID
tests run against REAL child processes and the real ``/proc``: a PID's identity is a
kernel fact, and a fake would agree with whatever the code assumes.

PROOF (2026-09-14): gpu_supervisor.py, steering_workers.py and endpoints/steering.py
swapped back to ccd91cc6 (scratchpad p3-r2/spec_baseline.json, B-R2-workers): both
drain tests and both stale-PID-file tests that expect a refusal red, the live-worker
control green. The recorded-spawn tests read SPAWN_START_TICKS, which did not exist,
and are proven by C-R2-6b/6c/6d/6f.

MUTATION CONTROLS (2026-09-14; scratchpad p3-r2/mutate.py and spec_r2.json; each
alone, restored byte-identically with sha256 checked, `git status` clean) — all red:
  C-R2-5a the drain's deadline starts when the loop notices -> a stop noticed late
  C-R2-5b the stop's time is not recorded                   -> the same
  C-R2-5c each stuck worker killed then waited for in turn  -> every stuck worker killed first
  C-R2-6a a PID file names any live process                 -> the stale-file tests (card, single)
  C-R2-6b a recorded PID is killed whatever its start time  -> the card spawn; the orphan sweep; unrecorded
  C-R2-6c a card worker's spawn records no identity         -> the card spawn
  C-R2-6d the single worker's spawn records no identity     -> the orphan sweep; test_privilege_operations
  C-R2-6e the single worker found by liveness alone         -> exit-mode leaves a process that reused the PID
  C-R2-6f the orphan sweep kills by number                  -> the orphan sweep

ROUND 1 CONTROLS RE-RUN on the lines this round changed (spec_r1_rerun.json) — all red:
R6 (SIGQUIT), R7 (no drain wait), R11 (a worker started during the stop), ST3, ST4
(a pattern lookup behind the new file check), ST5 (a zombie), T3 (one card's kill).
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from unittest.mock import AsyncMock

import pytest

from src.api.v1 import steering_workers as W
from src.core.config import settings
from src.services import steering_worker_card as P
from src.workers import gpu_supervisor as S
from tests.unit.test_gpu_supervisor import RTX, TI, Clock, FakeWorker, Launcher, restore_signals, run  # noqa: F401

TI_UUID = TI.uuid
RTX_UUID = RTX.uuid


# ── the drain ────────────────────────────────────────────────────────────────


class StuckWorker(FakeWorker):
    """A worker in uninterruptible sleep (a hung driver call): SIGKILL does not reap it at once."""

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.clock.t += timeout or 0.0
        raise subprocess.TimeoutExpired("celery", timeout)


class TestTheDrainKeepsInsideTheGracePeriod:
    """``terminationGracePeriodSeconds: 30`` against ``DRAIN_TIMEOUT_S`` = 25 s."""

    def test_a_stop_noticed_late_is_still_timed_from_the_signal(self, restore_signals):
        """A maintenance tick blocked on Redis or the database when the stop arrives: every
        worker gets SIGTERM at once (the handler), but the drain's clock started only when the
        loop came back — so a stuck worker was killed ~20 s after the kubelet had already."""
        clock = Clock()
        plans = {RTX_UUID: lambda index, now: FakeWorker(clock, 0, drains_for=1_000.0)}
        launcher = Launcher(clock, plans)
        sent = []

        def slow_tick():
            if clock() >= 5.0 and not sent:
                sent.append(clock())
                os.kill(os.getpid(), signal.SIGTERM)
                clock.t += 20.0     # the rest of the tick: blocked on the broker

        run(launcher, clock, S.worker_specs([RTX], python="py"), on_tick=slow_tick)

        (worker,) = launcher.workers[RTX_UUID]
        stopped_at = worker.signals[0][1]
        assert stopped_at == sent[0], "the worker was not signalled when the stop arrived"
        assert worker.killed_at is not None
        assert worker.killed_at - stopped_at <= S.DRAIN_TIMEOUT_S + 1, (
            f"killed {worker.killed_at - stopped_at:.0f} s after the stop, past the pod's grace period"
        )

    def test_every_stuck_worker_is_killed_before_any_is_waited_for(self, restore_signals):
        clock = Clock()
        plans = {uuid: (lambda index, now: StuckWorker(clock, 0)) for uuid in (TI_UUID, RTX_UUID)}
        launcher = Launcher(clock, plans)
        sent = []

        def tick():
            if clock() >= 5.0 and not sent:
                sent.append(clock())
                os.kill(os.getpid(), signal.SIGTERM)

        assert run(launcher, clock, S.worker_specs([TI, RTX], python="py"), on_tick=tick) == 0

        (ti,), (rtx,) = launcher.workers[TI_UUID], launcher.workers[RTX_UUID]
        assert ti.killed_at is not None and rtx.killed_at is not None
        assert ti.killed_at == rtx.killed_at, (
            f"the second worker was killed {abs(rtx.killed_at - ti.killed_at):.0f} s after the first"
        )
        assert rtx.killed_at - sent[0] <= S.DRAIN_TIMEOUT_S + 1
        assert clock() - sent[0] <= S.DRAIN_TIMEOUT_S + S.KILL_WAIT_S + 2, "the stop outlived the grace period"


# ── which process a steering worker's PID names ─────────────────────────────


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "_run_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _forget_spawned():
    W.SPAWNED_PIDS_BY_CARD.clear()
    W.SPAWN_START_TICKS.clear() if hasattr(W, "SPAWN_START_TICKS") else None
    yield
    W.SPAWNED_PIDS_BY_CARD.clear()
    W.SPAWN_START_TICKS.clear() if hasattr(W, "SPAWN_START_TICKS") else None


#: The kernel's kill, captured before any test replaces ``os.kill``: a fixture's teardown can run
#: while a test's recording ``os.kill`` is still installed, and ``Popen.kill`` would then signal
#: nothing and ``wait()`` for the whole life of the child.
_REAL_KILL = os.kill


@pytest.fixture
def child():
    """A real, live process of our own, which the test may name in a PID file or a spawn record."""
    process = subprocess.Popen(["/bin/sleep", "120"])
    yield process
    try:
        _REAL_KILL(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


@pytest.fixture
def kills(monkeypatch):
    """SIGKILLs recorded instead of sent. Signal 0 (the liveness probe) still reaches the kernel."""
    sent = []
    real_kill = os.kill

    def kill(pid, sig):
        if sig == 0:
            return real_kill(pid, sig)
        sent.append((pid, sig))

    monkeypatch.setattr(os, "kill", kill)
    return sent


def _backdate(path, seconds):
    then = time.time() - seconds
    os.utime(path, (then, then))


class TestAStalePidFileNamesNoWorker:
    """A SIGKILLed steering worker leaves its PID file (Celery removes it only on a clean
    exit). Once the kernel hands the number to another process, the file names THAT process:
    the mode endpoint reports it as a live worker, and exit-mode and the ensure before every
    steering request SIGKILL it. ``pid_alive`` proves only that SOME process has the number."""

    def test_a_card_worker_file_naming_a_process_that_started_later_is_not_a_worker(self, run_dir, child, kills):
        P.pid_file(TI_UUID).write_text(str(child.pid))
        _backdate(P.pid_file(TI_UUID), 3_600)   # written an hour before this process began

        assert W.card_worker_pid(TI_UUID) is None, "a reused PID read as the card's steering worker"
        W.kill_card_worker(TI_UUID)
        assert (child.pid, signal.SIGKILL) not in kills, "exit-mode SIGKILLed a process that reused the PID"

    def test_a_file_written_by_its_live_worker_still_names_it(self, run_dir, child, kills):
        P.pid_file(TI_UUID).write_text(str(child.pid))

        assert W.card_worker_pid(TI_UUID) == child.pid
        assert W.kill_card_worker(TI_UUID) == 1
        assert kills == [(child.pid, signal.SIGKILL)]

    def test_exit_mode_leaves_a_process_that_reused_the_single_workers_pid(self, tmp_path, child, kills, monkeypatch):
        from src.api.v1.endpoints import steering as endpoints

        pid_file = tmp_path / "single.pid"
        pid_file.write_text(str(child.pid))
        _backdate(pid_file, 3_600)
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        monkeypatch.setattr(endpoints, "PID_FILE", str(pid_file))
        monkeypatch.setattr(endpoints, "_get_gpu_memory_mb", lambda: None)
        monkeypatch.setattr(endpoints.asyncio, "sleep", AsyncMock())
        endpoints._SPAWNED_WORKER_PIDS.clear()

        result = asyncio.run(endpoints.exit_steering_mode())

        assert (child.pid, signal.SIGKILL) not in kills, "exit-mode SIGKILLed a process that reused the PID"
        assert result["already_inactive"] is True


class TestARecordedSpawnIsKilledOnlyWhileItIsTheSameProcess:
    """The PIDs recorded at spawn are never pruned: a worker exits after every generation
    (``--max-tasks-per-child=1``), and exit-mode or the orphan sweep later SIGKILLs every
    number ever recorded. A number the kernel has handed on names an unrelated process."""

    def test_a_card_spawn(self, run_dir, child, kills):
        assert W.spawn_card_worker(TI_UUID, popen=lambda argv, **kw: child) == child.pid
        assert W.SPAWN_START_TICKS[child.pid] == W.process_start_ticks(child.pid), "the spawn recorded no identity"

        W.SPAWN_START_TICKS[child.pid] -= 1     # the number now names a later process
        assert W.kill_card_workers(TI_UUID) == 0
        assert kills == [], "a recorded PID that now names another process was SIGKILLed"

        W.spawn_card_worker(TI_UUID, popen=lambda argv, **kw: child)
        assert W.kill_card_workers(TI_UUID) == 1
        assert kills == [(child.pid, signal.SIGKILL)]

    def test_the_single_workers_orphan_sweep(self, tmp_path, child, kills, monkeypatch):
        from src.api.v1.endpoints import steering as endpoints

        monkeypatch.setattr(endpoints, "STEERING_LOG", str(tmp_path / "steering.log"))
        monkeypatch.setattr(endpoints.subprocess, "Popen", lambda *a, **k: child)
        endpoints._SPAWNED_WORKER_PIDS.clear()
        endpoints._spawn_steering_worker()
        assert W.SPAWN_START_TICKS[child.pid] == W.process_start_ticks(child.pid), "the spawn recorded no identity"

        W.SPAWN_START_TICKS[child.pid] -= 1
        assert asyncio.run(endpoints._kill_orphan_steering_workers()) == 0
        assert kills == []

        endpoints._spawn_steering_worker()
        assert asyncio.run(endpoints._kill_orphan_steering_workers()) == 1
        assert kills == [(child.pid, signal.SIGKILL)]
        assert not endpoints._SPAWNED_WORKER_PIDS

    def test_an_unrecorded_pid_is_never_killed(self, run_dir, child, kills):
        W.SPAWNED_PIDS_BY_CARD[TI_UUID] = {child.pid}
        assert W.kill_card_workers(TI_UUID) == 0
        assert kills == []
