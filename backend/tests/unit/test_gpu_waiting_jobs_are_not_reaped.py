"""A job waiting for a GPU is not reaped as dead (multi-GPU Phase 3, review round 1, R1-3).

A job handed off at PLACEMENT re-runs what its body did before placing, every time
it is republished — and some of that makes the row look started:
``run_circuit_calibration`` and ``run_circuit_record`` write status "running", and
``compute_readout``/``compute_probe`` mark their task_queue row running. Writing the
same value again changes nothing, so the row's clock freezes at the first attempt.
The circuit janitor (60 min on a started row whose Celery result is PENDING) and
``cleanup_orphaned_tasks`` (10 min) then marked a job that was merely waiting for a
card as failed. ``acquire_jlens_artifact`` waits IN PLACE for up to 30 minutes with no
heartbeat, and was reaped the same way.

The rule now: a job is WAITING — and a janitor spares it — when its task id has a
fresh waiting mark (written at every hand-off and every in-place poll, cleared once
it leases), a parked re-dispatch, or a message on a GPU queue. A dead job has none of
these: its mark expires, and nothing of it is queued or parked, so it is reaped as
before. Training was checked too: its pre-placement INITIALIZING never reads as
started, so it was not reaped (pinned below as a guard, not a fix).

Redis is faked in memory with the calls used; kombu's message layout (``headers.id``
on the queue list) was checked against the local Redis on a private DB number.

PROOF: on c78f0f21's janitors `test_a_parked_calibration_is_not_abandoned` and
`test_a_parked_readout_is_left_running` were red (the parked job was reaped), while
both dead-job tests passed.

MUTATION CONTROLS (2026-09-14, fix 03920f91; scratchpad p3-r1-claim/mutate.py, each
alone, restored byte-identically with sha256 checked, `git diff` clean) — all 12 red:
  C-R1-3a task_looks_alive ignores waiting          -> a started row whose job waits is alive
  C-R1-3b the circuit janitor ignores waiting       -> a parked calibration is not abandoned
  C-R1-3c cleanup_orphaned_tasks ignores waiting    -> parked readout; acquire waiting in place
  C-R1-3d requeue does not mark                     -> every hand-off marks the job (both delays)
  C-R1-3e an in-place poll does not mark            -> marked at every poll, cleared once it leases
  C-R1-3f a successful lease does not clear         -> marked at every poll, cleared once it leases
  C-R1-3g parked entries not consulted              -> a parked job is waiting; parked calibration
  C-R1-3h GPU queues not consulted                  -> queued on gpu.auto / a card / a priority queue; parked readout
  C-R1-3i an unreadable Redis condemns              -> Redis down spares rather than condemns
  C-R1-3j waiting_for_gpu not gated on per-card     -> single mode never asks Redis
  C-R1-3k the mark is written without its expiry    -> a mark counts until it expires
  C-R1-3l the queued message's id read from `task`  -> queued on a GPU queue (3); parked readout
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from celery import Task

from src.core.celery_app import celery_app
from src.core.config import settings
from src.services import gpu_job_claim as C
from src.services import gpu_leases
from src.services import gpu_worker_queues as Q
from src.services.gpu_claim import AUTO_QUEUE, queue_for
from src.services.gpu_placement import GpuCard
from src.workers import gpu_job as G
from src.workers.task_heartbeat import beat, task_looks_alive
from tests.unit.gpu_lease_db import clear, lease_engine, session_factory

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(0, TI_UUID, "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_900)
RTX = GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, 21_500)
CARDS = [TI, RTX]
OTHER = "training:someone-else:00000000"


class FakeRedis:
    """The Redis calls the waiting check makes, with Redis's semantics, on a settable clock."""

    def __init__(self):
        self.now = 1_000.0
        self.strings = {}  # key -> (value, expires_at or None)
        self.zsets = {}
        self.lists = {}

    @staticmethod
    def _k(key):
        return key.decode() if isinstance(key, bytes) else key

    @staticmethod
    def _b(value):
        return value.encode() if isinstance(value, str) else value

    def set(self, key, value, ex=None):
        self.strings[self._k(key)] = (self._b(value), None if ex is None else self.now + ex)

    def exists(self, key):
        entry = self.strings.get(self._k(key))
        return int(entry is not None and (entry[1] is None or entry[1] > self.now))

    def delete(self, *keys):
        return sum(1 for key in keys if self.strings.pop(self._k(key), None) is not None)

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update({self._b(m): float(s) for m, s in mapping.items()})

    def zrange(self, key, start, end):
        members = [m for _, m in sorted((s, m) for m, s in self.zsets.get(key, {}).items())]
        return members[start:] if end == -1 else members[start:end + 1]

    def lpush(self, key, *values):
        for value in values:
            self.lists.setdefault(key, []).insert(0, self._b(value))

    def lrange(self, key, start, end):
        items = self.lists.get(self._k(key), [])
        return items[start:] if end == -1 else items[start:end + 1]

    def scan_iter(self, match):
        keys = [k for k in self.lists if self.lists[k]] + list(self.zsets) + list(self.strings)
        return [k.encode() for k in keys if fnmatch.fnmatchcase(k, match)]

    def type(self, key):
        key = self._k(key)
        if self.lists.get(key):
            return b"list"
        if key in self.zsets:
            return b"zset"
        return b"string" if key in self.strings else b"none"


class BrokenRedis(FakeRedis):
    def exists(self, key):
        raise ConnectionError("redis is down")


def queue_message(client, queue, task_id):
    """A message as kombu's Redis transport stores it (checked against the local Redis)."""
    client.lpush(queue, json.dumps({
        "body": "e30=", "content-encoding": "utf-8", "content-type": "application/json",
        "headers": {"id": task_id, "task": "src.workers.x.task", "lang": "py"}, "properties": {},
    }))


@pytest.fixture
def redis_fake(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(Q, "redis_client", lambda: client)
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    return client


# ── the waiting check ────────────────────────────────────────────────────────


class TestIsAJobWaiting:
    def test_a_parked_job_is_waiting(self):
        client = FakeRedis()
        Q.park_job("src.workers.x.task", ["run-1"], {}, {"queue": AUTO_QUEUE, "task_id": "tid-parked"},
                   due_at=2_000.0, client=client)
        assert Q.is_waiting_for_gpu("tid-parked", client=client) is True
        assert Q.is_waiting_for_gpu("tid-other", client=client) is False

    @pytest.mark.parametrize("queue", [AUTO_QUEUE, queue_for(RTX_UUID), AUTO_QUEUE + "\x06\x163"])
    def test_a_job_queued_on_a_gpu_queue_is_waiting(self, queue):
        client = FakeRedis()
        queue_message(client, queue, "tid-queued")
        assert Q.is_waiting_for_gpu("tid-queued", client=client) is True

    def test_a_message_on_another_queue_does_not_count(self):
        client = FakeRedis()
        queue_message(client, "training", "tid-legacy")
        assert Q.is_waiting_for_gpu("tid-legacy", client=client) is False

    def test_a_mark_counts_until_it_expires(self):
        client = FakeRedis()
        Q.mark_waiting("tid-marked", client=client)
        assert Q.is_waiting_for_gpu("tid-marked", client=client) is True
        client.now += Q.WAITING_MARK_TTL_S + 1
        assert Q.is_waiting_for_gpu("tid-marked", client=client) is False

    def test_a_cleared_mark_does_not_count(self):
        client = FakeRedis()
        Q.mark_waiting("tid-placed", client=client)
        Q.clear_waiting("tid-placed", client=client)
        assert Q.is_waiting_for_gpu("tid-placed", client=client) is False

    def test_redis_down_spares_rather_than_condemns(self):
        assert Q.is_waiting_for_gpu("tid-x", client=BrokenRedis()) is True

    def test_marking_never_raises(self):
        class Refusing(FakeRedis):
            def set(self, *args, **kwargs):
                raise ConnectionError("redis is down")

        Q.mark_waiting("tid-x", client=Refusing())
        Q.clear_waiting("tid-x", client=BrokenRedis())

    def test_single_mode_never_asks_redis(self, monkeypatch):
        monkeypatch.setattr(settings, "gpu_worker_mode", "single")
        monkeypatch.setattr(Q, "redis_client", lambda: pytest.fail("single mode reached Redis"))
        assert G.waiting_for_gpu("tid-x") is False

    def test_per_card_mode_asks_the_broker(self, redis_fake):
        Q.mark_waiting("tid-x", client=redis_fake)
        assert G.waiting_for_gpu("tid-x") is True
        assert G.waiting_for_gpu(None) is False


# ── the marks are written where a job waits ─────────────────────────────────


class FakeTask(Task):
    request = None
    name = "tests.waiting_gpu_task"

    def __init__(self, request):
        self.request = request
        self.published = []

    def apply_async(self, args=None, kwargs=None, **options):
        self.published.append((args, kwargs, options))
        return SimpleNamespace(id=options.get("task_id"))


def request(**overrides):
    base = dict(id="tid-1", args=["job-1"], kwargs={"gpu_request": "auto"}, timelimit=(None, None),
                gpu_hops=None, headers=None, argsrepr=None, kwargsrepr=None)
    base.update(overrides)
    return SimpleNamespace(**base)


class TestTheMarksAreWritten:
    @pytest.mark.parametrize("delay_s", [0.0, C.HANDOFF_WAIT_S])
    def test_every_hand_off_marks_the_job_waiting(self, monkeypatch, delay_s):
        marked, parked = [], []
        monkeypatch.setattr(G, "mark_waiting", lambda task_id: marked.append(task_id))
        G.requeue(FakeTask(request(id="tid-h")), request(id="tid-h"),
                  C.JobHandoff(AUTO_QUEUE, "busy", delay_s=delay_s),
                  park=lambda *a, **k: parked.append(a))
        assert marked == ["tid-h"]

    @pytest.fixture(scope="class")
    def engine(self):
        eng = lease_engine("mistudio_test_gpu_waiting_jobs")
        yield eng
        eng.dispose()

    def test_a_job_waiting_in_place_is_marked_at_every_poll_and_cleared_once_it_leases(self, engine, monkeypatch):
        clear(engine)
        db = session_factory(engine)
        with db() as s:
            assert gpu_leases.acquire(s, [RTX_UUID], OTHER, task_id="other")
        events = []
        monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
        monkeypatch.setattr(Q, "mark_waiting", lambda task_id: events.append(("mark", task_id)))
        monkeypatch.setattr(Q, "clear_waiting", lambda task_id: events.append(("clear", task_id)))
        polls = {"n": 0}

        def sleep(seconds):
            polls["n"] += 1
            if polls["n"] == 2:
                with db() as s:
                    gpu_leases.release(s, OTHER)

        claim = C.ClaimContext(
            holder=C.make_holder("jlens_acquire", "tid-acq"), task_id="tid-acq", worker_uuid=None,
            handoff=False, session=db, inventory=lambda: list(CARDS), available_mb=lambda: 1e6, sleep=sleep,
        )
        try:
            assert claim.claim(RTX_UUID, required_mb=4_000) == (RTX,)
        finally:
            claim.close()
        assert events == [("mark", "tid-acq"), ("mark", "tid-acq"), ("clear", "tid-acq")]


# ── the janitors spare a waiting job and still reap a dead one ─────────────


def _stale(minutes):
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


class TestTheCircuitJanitor:
    """`run_circuit_calibration` / `run_circuit_record` wrote "running" before placing."""

    @pytest.fixture
    def pending(self, monkeypatch):
        monkeypatch.setattr(celery_app, "AsyncResult", lambda task_id, app=None: SimpleNamespace(state="PENDING", info=None))

    def test_a_parked_calibration_is_not_abandoned(self, redis_fake, pending):
        from src.workers.cleanup_stuck_circuit_runs import _SubLifecycleView, _is_abandoned

        Q.park_job("src.workers.circuit_calibration_tasks.run_circuit_calibration", ["crc-1", {}], {},
                   {"queue": AUTO_QUEUE, "task_id": "tid-cal"}, due_at=2_000.0, client=redis_fake)
        view = _SubLifecycleView("tid-cal", _stale(90))
        assert _is_abandoned(view, "running") is False

    def test_a_calibration_whose_job_is_gone_is_still_abandoned(self, redis_fake, pending):
        from src.workers.cleanup_stuck_circuit_runs import _SubLifecycleView, _is_abandoned

        view = _SubLifecycleView("tid-dead", _stale(90))
        assert _is_abandoned(view, "running") is True


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return list(self._rows)


class TestTheOrphanedTaskJanitor:
    """`compute_readout`/`compute_probe` marked their task_queue row running before placing;
    `acquire_jlens_artifact` waits in place with its last heartbeat going stale."""

    @pytest.fixture
    def sweep(self, monkeypatch):
        import celery.result

        from src.core import database

        rows = []
        committed = []
        session = SimpleNamespace(query=lambda model: _Query(rows), commit=lambda: committed.append(True))

        @contextlib.contextmanager
        def get_sync_db():
            yield session

        stale_progress = SimpleNamespace(state="PROGRESS", info=beat({"stage": "loading_model"}))
        stale_progress.info["heartbeat"] -= 3_600
        monkeypatch.setattr(database, "get_sync_db", get_sync_db)
        monkeypatch.setattr(celery.result, "AsyncResult", lambda task_id, app=None: stale_progress)
        monkeypatch.setattr(G, "release_reaped_leases", lambda task_id, **kwargs: 0)

        def run(*new_rows):
            rows[:] = new_rows
            from src.workers.cleanup_orphaned_tasks import cleanup_orphaned_tasks_task

            return cleanup_orphaned_tasks_task.apply().get()

        return run

    @staticmethod
    def row(task_id):
        return SimpleNamespace(task_id=task_id, status="running", updated_at=_stale(60), error_message=None)

    def test_a_parked_readout_is_left_running(self, redis_fake, sweep):
        queue_message(redis_fake, AUTO_QUEUE, "tid-readout")
        parked = self.row("tid-readout")
        assert sweep(parked) == {"closed": 0}
        assert parked.status == "running"

    def test_an_acquire_waiting_in_place_is_left_running(self, redis_fake, sweep):
        Q.mark_waiting("tid-acquire", client=redis_fake)
        waiting = self.row("tid-acquire")
        assert sweep(waiting) == {"closed": 0}
        assert waiting.status == "running"

    def test_a_dead_job_is_still_closed(self, redis_fake, sweep):
        dead = self.row("tid-dead")
        assert sweep(dead) == {"closed": 1}
        assert dead.status == "failed"


class TestTheSharedLivenessRule:
    """`task_looks_alive` — used by the training, activation, SAE extraction and labeling janitors."""

    @pytest.fixture
    def pending(self, monkeypatch):
        monkeypatch.setattr(celery_app, "AsyncResult", lambda task_id, app=None: SimpleNamespace(state="PENDING", info=None))

    def test_a_started_row_whose_job_waits_is_alive(self, redis_fake, pending):
        Q.mark_waiting("tid-wait", client=redis_fake)
        row = SimpleNamespace(updated_at=_stale(90))
        assert task_looks_alive("tid-wait", row, started=True) is True

    def test_a_started_row_whose_job_is_gone_is_not(self, redis_fake, pending):
        row = SimpleNamespace(updated_at=_stale(90))
        assert task_looks_alive("tid-gone", row, started=True) is False

    def test_a_parked_training_was_never_reaped_it_reads_as_not_started(self, redis_fake, pending):
        """Guard, not a fix: train_sae's pre-placement INITIALIZING is judged not started."""
        row = SimpleNamespace(updated_at=_stale(90))
        assert task_looks_alive("tid-training", row, started=False) is True
