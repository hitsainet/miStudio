"""One solo worker per GPU, from the live inventory (multi-GPU Phase 3, D7).

The supervisor's argv construction is pure and its supervision loop takes a fake
Popen and a fake clock, so both are tested without a GPU, a broker or a sleep.
The node: RTX 3080 Ti at index 0 beside the RTX 3090 at index 1; a third card
must need no code change.

Stops go through the REAL signal handler (the test sends itself SIGTERM from the
tick callback, at a fake-clock time it chooses); handlers are restored after
each test. Zombie reaping runs against real child processes. The down-card leases
use real Postgres (``tests/unit/gpu_lease_db.py``): a fake session reproduces no
primary key.

MUTATION CONTROLS (2026-09-14; each alone on src/workers/gpu_supervisor.py,
restored byte-identically) — all five went red:
  S1 a card worker does not consume gpu.auto   -> own queue and the shared one
  S2 a worker is not told its card             -> told its card, env reaches the child
  S5 hostnames collide                          -> hostnames are unique
  (S3/S4 pinned "a crash stops the set", the behaviour review round 1 replaced.)

Phase 3 wiring (2026-09-14; same procedure, sha256 and `git diff` checked) — both red:
  Q3 main() skips the start-up maintenance tick  -> main rescues before it starts the workers
  Q4 supervise() never calls on_tick             -> the tick runs while the workers run; a failing tick is retried

REVIEW ROUND 1 (2026-09-14, supervisor/workers/deployment). Each alone, restored
byte-identically (sha256 checked), `git diff` clean afterwards — all red:
  R1 exited() stops and returns instead of scheduling a restart
        -> test_a_crashed_worker_is_restarted_alone_and_the_others_are_not_touched
  R2 backoff_for ignores the exit count (constant base delay)
        -> test_the_backoff_doubles_with_each_exit_in_the_window
  R3 the crash-loop comparison `>` becomes `> max + 1` (one restart too many)
        -> test_a_crash_looping_worker_is_given_up_and_the_others_keep_running
  R4 the window pruning loop removed
        -> test_exits_outside_the_window_do_not_count
  R5 the every-worker-crash-looping return removed
        -> test_when_every_worker_crash_loops_the_supervisor_exits_non_zero (the fake
           clock's cap turns the endless loop into a failure, not a hang)
  R6 terminate_once sends SIGQUIT instead of SIGTERM
        -> test_a_stop_is_warm_every_worker_gets_one_sigterm_and_no_sigquit
  R7 the drain deadline is `clock()` (no wait)
        -> test_a_busy_worker_is_waited_for_until_the_drain_timeout
  R8 reap_orphans no longer skips a known PID
        -> test_a_workers_exit_status_is_left_for_popen
  R9 CardAvailability.update does not release a card whose worker runs again
        -> test_a_card_is_leased_while_its_worker_is_down_and_released_when_it_runs
  R10 main() stops passing on_status=availability.update
        -> test_main_rescues_resets_markers_and_wires_the_status_before_starting
  R11 a restart during backoff ignores the stop (starts the worker anyway)
        -> test_a_stop_during_a_backoff_starts_nothing
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from src.services import gpu_leases
from src.services.gpu_claim import AUTO_QUEUE, RunHere, SendTo, decide_claim, queue_for
from src.services.gpu_placement import GpuCard
from src.workers import gpu_supervisor as S

TI = GpuCard(0, "GPU-f47ba814-49a2-603f-3595-275284140251", "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_900)
RTX = GpuCard(1, "GPU-247aa582-0d1b-e161-8156-983ed1fefc57", "NVIDIA GeForce RTX 3090", 24_576, 21_500)
NEW = GpuCard(2, "GPU-11111111-2222-3333-4444-555555555555", "NVIDIA GeForce RTX 4090", 24_564, 24_000)


def _arg(spec, flag):
    argv = list(spec.argv)
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return next(a.split("=", 1)[1] for a in argv if a.startswith(flag + "="))


class TestWorkerSpecs:
    def test_one_worker_per_card_on_its_own_queue_and_the_shared_one(self):
        specs = S.worker_specs([TI, RTX], python="py")
        assert [_arg(s, "-Q").split(",") for s in specs] == [
            [queue_for(TI.uuid), AUTO_QUEUE],
            [queue_for(RTX.uuid), AUTO_QUEUE],
        ]

    def test_each_worker_is_told_its_card(self):
        specs = S.worker_specs([TI, RTX], python="py")
        assert [s.env[S.WORKER_GPU_ENV] for s in specs] == [TI.uuid, RTX.uuid]
        assert [s.card for s in specs] == [TI.uuid, RTX.uuid]

    def test_every_worker_is_solo_and_single_slot(self):
        for spec in S.worker_specs([TI, RTX, NEW], python="py"):
            assert _arg(spec, "-c") == "1"
            assert "--pool=solo" in spec.argv

    def test_hostnames_are_unique(self):
        names = [_arg(s, "--hostname") for s in S.worker_specs([TI, RTX, NEW], python="py")]
        assert len(set(names)) == 3

    def test_a_third_card_needs_no_code_change(self):
        assert len(S.worker_specs([TI, RTX, NEW], python="py")) == 3

    def test_no_card_means_one_auto_worker(self):
        (spec,) = S.worker_specs([], python="py")
        assert _arg(spec, "-Q") == AUTO_QUEUE
        assert S.WORKER_GPU_ENV not in spec.env
        assert spec.card is None


# ── fakes ────────────────────────────────────────────────────────────────────


class Clock:
    """Fake monotonic time. `sleep` advances it; a supervisor that never returns hits the cap."""

    CAP = 5_000.0

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        if self.t > self.CAP:
            raise AssertionError(f"the supervisor was still looping at t={self.t:.0f} s")


class FakeWorker:
    """A worker process on the fake clock: exits at `exit_at`, or `drains_for` after SIGTERM."""

    def __init__(self, clock, pid, *, exit_at=None, code=0, drains_for=0.0):
        self.clock, self.pid = clock, pid
        self.exit_at, self.code, self.drains_for = exit_at, code, drains_for
        self.signals = []
        self.term_at = None
        self.killed_at = None

    def poll(self):
        now = self.clock()
        if self.killed_at is not None:
            return -signal.SIGKILL
        if self.term_at is not None and now >= self.term_at + self.drains_for:
            return 0
        if self.exit_at is not None and now >= self.exit_at:
            return self.code
        return None

    def send_signal(self, signum):
        self.signals.append((signum, self.clock()))
        if signum == signal.SIGTERM and self.term_at is None:
            self.term_at = self.clock()

    def kill(self):
        self.killed_at = self.clock()

    def wait(self, timeout=None):
        return self.poll()


class Launcher:
    """Popen for the supervisor: `plans[card]` makes each successive start's FakeWorker."""

    def __init__(self, clock, plans):
        self.clock, self.plans = clock, plans
        self.started = []   # (card, t, env card)
        self.workers = {}   # card -> [FakeWorker]

    def __call__(self, argv, env):
        card = env.get(S.WORKER_GPU_ENV)
        index = len(self.workers.setdefault(card, []))
        worker = self.plans[card](index, self.clock())
        worker.pid = 1000 + len(self.started)
        self.workers[card].append(worker)
        self.started.append((card, self.clock(), env.get(S.WORKER_GPU_ENV)))
        return worker

    def starts(self, card):
        return [t for c, t, _ in self.started if c == card]


@pytest.fixture
def restore_signals():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def stop_at(clock, when, also=None):
    """A tick that sends this process SIGTERM once the fake clock reaches `when`."""
    sent = []

    def tick():
        if clock() >= when and not sent:
            sent.append(clock())
            os.kill(os.getpid(), signal.SIGTERM)
        if also is not None:
            also()

    return tick


def run(launcher, clock, specs, **kwargs):
    kwargs.setdefault("policy", S.RestartPolicy())
    return S.supervise(specs, popen=launcher, poll_interval=1.0, install_signal_handlers=True,
                       clock=clock, sleep=clock.sleep, **kwargs)


def forever(clock):
    return lambda index, now: FakeWorker(clock, 0)


# ── crash isolation ──────────────────────────────────────────────────────────


class TestCrashIsolation:
    def test_a_crashed_worker_is_restarted_alone_and_the_others_are_not_touched(self, restore_signals):
        clock = Clock()
        plans = {
            # The 3080 Ti's worker is OOM-killed at t=3; its replacement runs.
            TI.uuid: lambda index, now: FakeWorker(clock, 0, exit_at=3.0 if index == 0 else None, code=-9),
            RTX.uuid: forever(clock),
        }
        launcher = Launcher(clock, plans)
        specs = S.worker_specs([TI, RTX], python="py")

        code = run(launcher, clock, specs, on_tick=stop_at(clock, 40.0))

        assert code == 0
        assert launcher.starts(RTX.uuid) == [0.0], "the 3090's worker was restarted because the 3080 Ti's crashed"
        (rtx,) = launcher.workers[RTX.uuid]
        assert [sig for sig, _ in rtx.signals] == [signal.SIGTERM]
        assert rtx.signals[0][1] >= 40.0, "the 3090's worker was signalled before the stop"
        assert rtx.killed_at is None
        ti_starts = launcher.starts(TI.uuid)
        assert len(ti_starts) == 2 and ti_starts[1] >= 3.0 + S.RestartPolicy().base_backoff_s
        assert [env for c, _, env in launcher.started if c == TI.uuid] == [TI.uuid, TI.uuid]

    def test_the_backoff_doubles_with_each_exit_in_the_window(self, restore_signals):
        clock = Clock()
        policy = S.RestartPolicy(base_backoff_s=5.0, max_backoff_s=40.0, window_s=10_000.0, max_restarts_in_window=10)
        plans = {RTX.uuid: lambda index, now: FakeWorker(clock, 0, exit_at=now + 1.0, code=1)}
        launcher = Launcher(clock, plans)

        run(launcher, clock, S.worker_specs([RTX], python="py"), policy=policy,
            on_tick=stop_at(clock, 200.0))

        starts = launcher.starts(RTX.uuid)
        gaps = [round(b - a) for a, b in zip(starts, starts[1:])]
        # Each gap is the 1 s the worker ran (its exit is seen on that poll) plus the backoff.
        assert [g - 1 for g in gaps[:5]] == [5, 10, 20, 40, 40]

    def test_a_crash_looping_worker_is_given_up_and_the_others_keep_running(self, restore_signals):
        clock = Clock()
        policy = S.RestartPolicy(base_backoff_s=1.0, max_backoff_s=1.0, window_s=600.0, max_restarts_in_window=2)
        plans = {
            TI.uuid: lambda index, now: FakeWorker(clock, 0, exit_at=now + 1.0, code=1),
            RTX.uuid: forever(clock),
        }
        launcher = Launcher(clock, plans)
        statuses = []

        code = run(launcher, clock, S.worker_specs([TI, RTX], python="py"), policy=policy,
                   on_status=statuses.append, on_tick=stop_at(clock, 100.0))

        assert code == 0
        assert len(launcher.starts(TI.uuid)) == 3, "two restarts are allowed, and no more"
        final = {s["card"]: s for s in statuses[-1]}
        assert final[TI.uuid]["state"] == S.CRASH_LOOPING
        assert final[TI.uuid]["restarts"] == 2 and final[TI.uuid]["last_exit_code"] == 1
        assert final[RTX.uuid]["state"] == S.RUNNING
        assert launcher.starts(RTX.uuid) == [0.0]

    def test_exits_outside_the_window_do_not_count(self, restore_signals):
        clock = Clock()
        policy = S.RestartPolicy(base_backoff_s=1.0, max_backoff_s=64.0, window_s=50.0, max_restarts_in_window=1)
        plans = {RTX.uuid: lambda index, now: FakeWorker(clock, 0, exit_at=now + 100.0, code=1)}
        launcher = Launcher(clock, plans)

        run(launcher, clock, S.worker_specs([RTX], python="py"), policy=policy, on_tick=stop_at(clock, 700.0))

        starts = launcher.starts(RTX.uuid)
        assert len(starts) >= 6, "a worker whose exits are 100 s apart was given up on a 50 s window"
        assert {round(b - a) for a, b in zip(starts, starts[1:])} == {101}, "the backoff kept growing"

    def test_when_every_worker_crash_loops_the_supervisor_exits_non_zero(self, restore_signals):
        clock = Clock()
        policy = S.RestartPolicy(base_backoff_s=1.0, max_backoff_s=1.0, window_s=600.0, max_restarts_in_window=1)
        plans = {uuid: (lambda index, now: FakeWorker(clock, 0, exit_at=now + 1.0, code=1)) for uuid in (TI.uuid, RTX.uuid)}
        launcher = Launcher(clock, plans)

        assert run(launcher, clock, S.worker_specs([TI, RTX], python="py"), policy=policy) == S.EXIT_ALL_CRASH_LOOPING

    def test_a_stop_during_a_backoff_starts_nothing(self, restore_signals):
        clock = Clock()
        plans = {RTX.uuid: lambda index, now: FakeWorker(clock, 0, exit_at=1.0, code=1)}
        launcher = Launcher(clock, plans)

        assert run(launcher, clock, S.worker_specs([RTX], python="py"), on_tick=stop_at(clock, 3.0)) == 0
        assert launcher.starts(RTX.uuid) == [0.0]

    def test_the_card_env_reaches_the_child_on_top_of_the_parent_env(self, monkeypatch, restore_signals):
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        clock = Clock()
        seen = {}

        def popen(argv, env):
            seen.update(env)
            return FakeWorker(clock, 1)

        S.supervise(S.worker_specs([RTX], python="py"), popen=popen, poll_interval=1.0, clock=clock,
                    sleep=clock.sleep, on_tick=stop_at(clock, 2.0))

        assert seen[S.WORKER_GPU_ENV] == RTX.uuid
        assert seen["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


# ── stopping ─────────────────────────────────────────────────────────────────


class TestDrain:
    def test_a_stop_is_warm_every_worker_gets_one_sigterm_and_no_sigquit(self, restore_signals):
        clock = Clock()
        plans = {uuid: (lambda index, now: FakeWorker(clock, 0, drains_for=2.0)) for uuid in (TI.uuid, RTX.uuid)}
        launcher = Launcher(clock, plans)

        assert run(launcher, clock, S.worker_specs([TI, RTX], python="py"), on_tick=stop_at(clock, 5.0)) == 0

        for workers in launcher.workers.values():
            (worker,) = workers
            assert [sig for sig, _ in worker.signals] == [signal.SIGTERM]
            assert worker.killed_at is None

    def test_a_busy_worker_is_waited_for_until_the_drain_timeout(self, restore_signals):
        clock = Clock()
        plans = {RTX.uuid: lambda index, now: FakeWorker(clock, 0, drains_for=S.DRAIN_TIMEOUT_S - 5)}
        launcher = Launcher(clock, plans)

        run(launcher, clock, S.worker_specs([RTX], python="py"), on_tick=stop_at(clock, 5.0))

        (worker,) = launcher.workers[RTX.uuid]
        assert worker.killed_at is None, "a task that finishes inside the drain was killed"

    def test_a_worker_started_as_the_stop_arrives_is_still_stopped_warmly(self, restore_signals):
        """The signal lands while a restart's Popen runs: the handler sees no running
        worker to signal, so the drain must signal the new one itself."""
        clock = Clock()

        def plan(index, now):
            if index == 1:
                os.kill(os.getpid(), signal.SIGTERM)   # arrives mid-start
                return FakeWorker(clock, 0, drains_for=2.0)
            return FakeWorker(clock, 0, exit_at=1.0, code=1)

        launcher = Launcher(clock, {RTX.uuid: plan})
        assert run(launcher, clock, S.worker_specs([RTX], python="py")) == 0

        restarted = launcher.workers[RTX.uuid][1]
        assert [sig for sig, _ in restarted.signals] == [signal.SIGTERM]
        assert restarted.killed_at is None, "a worker started during the stop was never asked to stop"

    def test_a_worker_outliving_the_drain_is_killed(self, restore_signals):
        clock = Clock()
        plans = {RTX.uuid: lambda index, now: FakeWorker(clock, 0, drains_for=1_000.0)}
        launcher = Launcher(clock, plans)

        run(launcher, clock, S.worker_specs([RTX], python="py"), on_tick=stop_at(clock, 5.0))

        (worker,) = launcher.workers[RTX.uuid]
        stopped_at = worker.signals[0][1]
        assert worker.killed_at is not None
        assert S.DRAIN_TIMEOUT_S <= worker.killed_at - stopped_at <= S.DRAIN_TIMEOUT_S + 1


# ── PID 1 ────────────────────────────────────────────────────────────────────


def _wait_until_zombie(pid, timeout=10.0):
    deadline = time.monotonic() + timeout
    stat = Path(f"/proc/{pid}/stat")
    while time.monotonic() < deadline:
        fields = stat.read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return
        time.sleep(0.01)
    raise AssertionError(f"process {pid} never exited")


class TestReapingOrphans:
    def test_an_adopted_zombie_is_reaped(self):
        pid = os.posix_spawn("/bin/true", ["/bin/true"], dict(os.environ))
        _wait_until_zombie(pid)

        assert S.reap_orphans([]) == [pid]
        assert not Path(f"/proc/{pid}").exists()

    def test_a_workers_exit_status_is_left_for_popen(self):
        worker = subprocess.Popen(["/bin/sh", "-c", "exit 7"])
        _wait_until_zombie(worker.pid)

        assert S.reap_orphans([worker.pid]) == []
        assert worker.poll() == 7, "the supervisor stole a worker's exit status"

    def test_supervise_reaps_with_the_running_workers_pids(self, restore_signals):
        clock = Clock()
        launcher = Launcher(clock, {TI.uuid: forever(clock), RTX.uuid: forever(clock)})
        seen = []

        run(launcher, clock, S.worker_specs([TI, RTX], python="py"), reap=lambda pids: seen.append(sorted(pids)) or [],
            on_tick=stop_at(clock, 3.0))

        assert seen and seen[0] == sorted(w[0].pid for w in launcher.workers.values())


# ── maintenance and status callbacks ─────────────────────────────────────────


class TestCallbacksWhileTheWorkersRun:
    def test_the_tick_runs_on_every_poll(self, restore_signals):
        clock = Clock()
        ticks = []
        launcher = Launcher(clock, {RTX.uuid: forever(clock)})
        run(launcher, clock, S.worker_specs([RTX], python="py"), on_tick=stop_at(clock, 3.0, also=lambda: ticks.append(clock())))
        assert ticks[:4] == [0.0, 1.0, 2.0, 3.0]

    def test_a_failing_callback_never_stops_the_workers(self, restore_signals):
        clock = Clock()
        attempts = []

        def broken(*_):
            attempts.append(clock())
            raise ConnectionError("redis down")

        launcher = Launcher(clock, {RTX.uuid: forever(clock)})
        code = run(launcher, clock, S.worker_specs([RTX], python="py"), on_status=broken,
                   on_tick=stop_at(clock, 3.0, also=broken))
        assert code == 0
        assert len(attempts) >= 8, "a failing callback must be retried on every poll"
        assert launcher.starts(RTX.uuid) == [0.0]

    def test_main_rescues_resets_markers_and_wires_the_status_before_starting(self, monkeypatch):
        from src.core import database
        from src.services import gpu_placement, gpu_worker_queues

        order = []

        def tick():
            order.append("tick")

        monkeypatch.setattr(gpu_placement, "list_cards", lambda: [TI, RTX])
        monkeypatch.setattr(gpu_worker_queues, "maintenance_tick", lambda live: tick)
        monkeypatch.setattr(database, "get_sync_db", lambda: pytest.fail("the supervisor opened a session at start"))
        monkeypatch.setattr(S.CardAvailability, "reset", lambda self, cards: order.append(("reset", list(cards))))

        def supervise(specs, on_tick=None, on_status=None, reap=None):
            order.append(("supervise", [s.card for s in specs], on_tick, getattr(on_status, "__func__", None)))
            return 0

        monkeypatch.setattr(S, "supervise", supervise)

        assert S.main() == 0
        assert order == [
            "tick",
            ("reset", [TI.uuid, RTX.uuid]),
            ("supervise", [TI.uuid, RTX.uuid], tick, S.CardAvailability.update),
        ]


# ── a card with no worker is leased as unavailable ──────────────────────────


@pytest.fixture
def lease_db():
    from tests.unit.gpu_lease_db import clear, lease_engine, session_factory

    engine = lease_engine("mistudio_test_supervisor_leases")
    clear(engine)
    yield session_factory(engine)
    engine.dispose()


def _live(session):
    with session() as db:
        return gpu_leases.live_leases(db)


def _status(card, state):
    return {"card": card, "state": state}


class TestCardAvailability:
    def test_a_card_is_leased_while_its_worker_is_down_and_released_when_it_runs(self, lease_db):
        clock = Clock()
        other = "training:t-1:aaaaaaaa"
        with lease_db() as db:
            assert gpu_leases.acquire(db, [RTX.uuid], other, task_id="t-1")
        availability = S.CardAvailability(lease_db, clock=clock)

        availability.update([_status(TI.uuid, S.RESTARTING), _status(RTX.uuid, S.RUNNING)])
        assert _live(lease_db) == {TI.uuid: S.CardAvailability.holder_for(TI.uuid), RTX.uuid: other}

        clock.t = 30.0
        availability.update([_status(TI.uuid, S.RUNNING), _status(RTX.uuid, S.RUNNING)])
        assert _live(lease_db) == {RTX.uuid: other}, "the card stayed unavailable after its worker came back"

    def test_the_marker_is_what_keeps_an_auto_job_off_the_dead_cards_queue(self, lease_db):
        """The 3080 Ti's worker is down: its card is idle and, with no model, the most free.
        The 3090's worker must not hand an Auto job to a queue nothing consumes."""
        idle_ti = GpuCard(0, TI.uuid, TI.name, 12_288, 12_000)
        busy_rtx = GpuCard(1, RTX.uuid, RTX.name, 24_576, 10_000)
        decide = lambda leases: decide_claim(
            worker_uuid=RTX.uuid, requested="auto", required_mb=4_000, allow_shard=False,
            cards=[idle_ti, busy_rtx], leases=leases, holder="h")

        assert isinstance(decide(_live(lease_db)), SendTo), "precondition: without the marker the job goes to the dead card"
        S.CardAvailability(lease_db, clock=Clock()).update([_status(TI.uuid, S.CRASH_LOOPING)])
        assert isinstance(decide(_live(lease_db)), RunHere)

    def test_a_restarted_supervisor_clears_the_markers_a_previous_one_left(self, lease_db):
        S.CardAvailability(lease_db, clock=Clock()).update([_status(TI.uuid, S.CRASH_LOOPING)])
        assert TI.uuid in _live(lease_db)

        S.CardAvailability(lease_db, clock=Clock()).reset([TI.uuid, RTX.uuid])
        assert _live(lease_db) == {}

    def test_a_card_its_crashed_job_still_holds_is_taken_once_that_lease_goes(self, lease_db):
        clock = Clock()
        crashed_job = "extraction:e-1:bbbbbbbb"
        with lease_db() as db:
            assert gpu_leases.acquire(db, [TI.uuid], crashed_job, task_id="e-1")
        availability = S.CardAvailability(lease_db, clock=clock, retry_every_s=10.0)

        availability.update([_status(TI.uuid, S.RESTARTING)])
        assert _live(lease_db) == {TI.uuid: crashed_job}
        with lease_db() as db:
            gpu_leases.release(db, crashed_job)
        clock.t = 11.0
        availability.update([_status(TI.uuid, S.RESTARTING)])
        assert _live(lease_db) == {TI.uuid: S.CardAvailability.holder_for(TI.uuid)}

    def test_the_marker_is_renewed_while_the_worker_stays_down(self, lease_db, monkeypatch):
        clock = Clock()
        availability = S.CardAvailability(lease_db, clock=clock, renew_every_s=60.0)
        calls = []
        real = gpu_leases.acquire
        monkeypatch.setattr(gpu_leases, "acquire", lambda *a, **k: calls.append(clock()) or real(*a, **k))

        for t in (0.0, 1.0, 30.0, 61.0, 62.0):
            clock.t = t
            availability.update([_status(TI.uuid, S.CRASH_LOOPING)])

        assert calls == [0.0, 61.0], "renewed on every poll, or not at all"
