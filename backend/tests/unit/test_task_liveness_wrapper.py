"""False_Reap fix #1 — liveness is stamped by the task WRAPPER, not by stage code.

`0xcc/tasks/DEBT_TASKS|False_Reap_Of_Live_Jobs_2026-10-01.md`: a live probe run was
marked failed because its `training` stage never heartbeat, the row clock froze, and the
janitor's backstop read that same clock. Fix #1 makes liveness independent of every stage:
Celery's task_prerun/task_postrun start and stop a background stamper for EVERY task.

The wiring tests go through Celery's LIVE signal registry (`task_prerun.send`), not a
direct call of the handler, so deleting the `connect_signals()` line in
`core/celery_app.py` turns them red. The janitor tests reproduce the incident's shape:
an ancient row, Celery answering PENDING with no info, and no worker answering inspect().
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import src.core.celery_app  # noqa: F401 - importing the app is what wires the signals
from src.workers import task_heartbeat, task_liveness


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.sets = 0

    def set(self, key, value, ex=None):
        self.sets += 1
        self.store[key] = (value, ex)

    def delete(self, key):
        self.store.pop(key, None)

    def exists(self, key):
        return 1 if key in self.store else 0


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(task_liveness, "_client", lambda: fake)
    yield fake
    # Never leave a stamper thread behind for another test.
    for task_id in list(task_liveness._ACTIVE):
        task_liveness.on_task_postrun(task_id=task_id)


class _Task:
    """A task as a WORKER sees it: not eager. Hashable, as Celery's signal sender must be."""

    def __init__(self, eager: bool = False):
        self.name = "src.workers.probe_monitor_tasks.train"
        self.request = SimpleNamespace(is_eager=eager)


def _worker_task():
    return _Task()


KEY = task_liveness.LIVENESS_PREFIX


class TestTheWrapperIsWiredIntoEveryTask:
    def test_task_prerun_through_the_live_registry_stamps_the_task(self, fake_redis):
        from celery.signals import task_prerun

        task = _worker_task()
        task_prerun.send(sender=task, task_id="pmr-live-1", task=task, args=(), kwargs={})

        assert KEY + "pmr-live-1" in fake_redis.store, (
            "no liveness stamp after task_prerun — the stamper is not wired to Celery's "
            "signals, so a stage that forgets to heartbeat can get a live job reaped again"
        )
        value, ttl = fake_redis.store[KEY + "pmr-live-1"]
        assert ttl == task_liveness.LIVENESS_TTL_S
        assert "pmr-live-1" in task_liveness._ACTIVE

    def test_task_postrun_through_the_live_registry_stops_and_clears(self, fake_redis):
        from celery.signals import task_postrun, task_prerun

        task = _worker_task()
        task_prerun.send(sender=task, task_id="pmr-live-2", task=task, args=(), kwargs={})
        stamper = task_liveness._ACTIVE["pmr-live-2"]
        task_postrun.send(sender=task, task_id="pmr-live-2", task=task, args=(), kwargs={},
                          retval=None, state="SUCCESS")

        assert KEY + "pmr-live-2" not in fake_redis.store, (
            "a finished task kept its stamp — a janitor would spare it for the TTL"
        )
        assert "pmr-live-2" not in task_liveness._ACTIVE
        assert not stamper._thread.is_alive()

    def test_exactly_one_receiver_each_so_no_double_stamper(self):
        from celery.signals import task_postrun, task_prerun

        task_liveness.connect_signals()  # a second call must be a no-op
        for signal, uid in ((task_prerun, "mistudio-task-liveness-prerun"),
                            (task_postrun, "mistudio-task-liveness-postrun")):
            keys = [key for key, _ref in signal.receivers if key[0] == uid]
            assert len(keys) == 1, f"{uid}: {len(keys)} receivers"

    def test_an_eager_call_is_not_stamped(self, fake_redis):
        from celery.signals import task_prerun

        task = _Task(eager=True)
        task_prerun.send(sender=task, task_id="eager-1", task=task, args=(), kwargs={})
        assert fake_redis.store == {}


class TestTheStamperRunsForTheWholeTask:
    def test_it_restamps_on_its_interval_until_stopped(self):
        fake = FakeRedis()
        stamper = task_liveness.LivenessStamper("long-1", interval_s=0.05, client=fake).start()
        try:
            time.sleep(0.4)
            assert fake.sets >= 4, f"only {fake.sets} stamps in 0.4 s at a 0.05 s interval"
        finally:
            stamper.stop()
        assert "long-1" not in {k[len(KEY):] for k in fake.store}
        assert not stamper._thread.is_alive()

    def test_the_first_stamp_is_synchronous(self):
        fake = FakeRedis()
        stamper = task_liveness.LivenessStamper("sync-1", interval_s=3600, client=fake)
        stamper.start()
        try:
            assert KEY + "sync-1" in fake.store
        finally:
            stamper.stop()

    def test_the_ttl_outlives_several_intervals_and_expires_well_before_the_janitor(self):
        assert task_liveness.LIVENESS_TTL_S >= 2 * task_liveness.LIVENESS_INTERVAL_S
        assert task_liveness.LIVENESS_TTL_S < task_heartbeat.STALE_AFTER_SECONDS

    def test_a_broker_failure_never_raises_into_the_task(self):
        class Broken:
            def set(self, *a, **k):
                raise ConnectionError("broker down")

            def delete(self, *a, **k):
                raise ConnectionError("broker down")

            def exists(self, *a, **k):
                raise ConnectionError("broker down")

        assert task_liveness.stamp("x", client=Broken()) is False
        task_liveness.clear("x", client=Broken())
        assert task_liveness.has_live_stamp("x", client=Broken()) is None


class _DeadLookingCelery:
    """The incident's Celery picture: PENDING, no info, nobody answers inspect()."""

    state = "PENDING"
    info = None


@pytest.fixture
def incident(monkeypatch):
    monkeypatch.setattr(task_heartbeat, "task_is_actively_running", lambda _id: None)
    import src.workers.gpu_job as gpu_job

    monkeypatch.setattr(gpu_job, "waiting_for_gpu", lambda _id: False)
    monkeypatch.setattr(src.core.celery_app.celery_app, "AsyncResult",
                        lambda *a, **k: _DeadLookingCelery())
    return SimpleNamespace(updated_at=datetime.now(timezone.utc) - timedelta(minutes=100))


class TestTheJanitorsReadTheStamp:
    def test_the_2026_10_01_shape_is_SPARED_when_the_wrapper_says_alive(self, fake_redis, incident):
        task_liveness.stamp("pmr_54563518010f-task", client=fake_redis)
        assert task_heartbeat.task_looks_alive(
            "pmr_54563518010f-task", incident, started=True
        ) is True

    def test_without_a_stamp_the_same_row_is_still_reaped(self, fake_redis, incident):
        """The stamp must not rescue a genuinely dead job: no stamp → old rules."""
        assert task_heartbeat.task_looks_alive("dead-task", incident, started=True) is False

    def test_an_unreadable_broker_falls_through_rather_than_rescuing(self, monkeypatch, incident):
        monkeypatch.setattr(task_liveness, "has_live_stamp", lambda *_a, **_k: None)
        assert task_heartbeat.task_looks_alive("dead-task", incident, started=True) is False

    def test_the_circuit_run_janitor_spares_a_stamped_run(self, fake_redis, incident, monkeypatch):
        from src.workers import cleanup_stuck_circuit_runs as janitor

        monkeypatch.setattr(janitor, "progress_stalled_seconds", lambda *a, **k: None)
        monkeypatch.setattr(janitor.celery_app, "AsyncResult", lambda *a, **k: _DeadLookingCelery())
        run = SimpleNamespace(id="cap-1", celery_task_id="cap-task", progress=None,
                              status="running", updated_at=incident.updated_at)
        assert janitor._is_abandoned(run, "running") is True, (
            "precondition: unstamped, this row is abandoned"
        )
        task_liveness.stamp("cap-task", client=fake_redis)
        assert janitor._is_abandoned(run, "running") is False


class TestTheOrphanedTaskJanitorReadsTheStamp:
    """`cleanup_orphaned_tasks` (task_queue rows: J-space fits, readouts, probes...)."""

    @pytest.fixture
    def sweep(self, monkeypatch):
        import contextlib

        import celery.result

        from src.core import database
        from src.workers import gpu_job
        from src.workers.task_heartbeat import beat

        rows = []

        class _Q:
            def filter(self, *a, **k):
                return self

            def all(self):
                return list(rows)

        session = SimpleNamespace(query=lambda model: _Q(), commit=lambda: None)

        @contextlib.contextmanager
        def get_sync_db():
            yield session

        stale = SimpleNamespace(state="PROGRESS", info=beat({"stage": "training"}))
        stale.info["heartbeat"] -= 3_600
        monkeypatch.setattr(database, "get_sync_db", get_sync_db)
        monkeypatch.setattr(celery.result, "AsyncResult", lambda task_id, app=None: stale)
        monkeypatch.setattr(gpu_job, "release_reaped_leases", lambda task_id, **kw: 0)
        monkeypatch.setattr(gpu_job, "waiting_for_gpu", lambda task_id: False)

        def run(*new_rows):
            rows[:] = new_rows
            from src.workers.cleanup_orphaned_tasks import cleanup_orphaned_tasks_task

            return cleanup_orphaned_tasks_task.apply().get()

        return run

    @staticmethod
    def _row(task_id):
        return SimpleNamespace(
            task_id=task_id, status="running",
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=60), error_message=None,
        )

    def test_a_stamped_task_is_left_running(self, fake_redis, sweep):
        task_liveness.stamp("tid-live", client=fake_redis)
        live = self._row("tid-live")
        assert sweep(live) == {"closed": 0}
        assert live.status == "running"

    def test_an_unstamped_dead_task_is_still_closed(self, fake_redis, sweep):
        dead = self._row("tid-dead")
        assert sweep(dead) == {"closed": 1}
        assert dead.status == "failed"


class TestTheMonitorDoesNotLabelALiveJobOrphaned:
    """Active Operations' `orphaned` label (task_queue `_celery_view`) reads the stamp too."""

    @pytest.fixture
    def stale_celery(self, monkeypatch):
        import celery.result

        from src.workers.task_heartbeat import beat

        stale = SimpleNamespace(state="PROGRESS", info=beat({"stage": "training"}))
        stale.info["heartbeat"] -= 3_600
        monkeypatch.setattr(celery.result, "AsyncResult", lambda task_id, app=None: stale)

    def _row(self, task_id):
        return SimpleNamespace(task_id=task_id, status="running",
                               updated_at=datetime.now(timezone.utc) - timedelta(minutes=60))

    def test_stamped_is_not_orphaned(self, fake_redis, stale_celery):
        from src.api.v1.endpoints.task_queue import _celery_view

        task_liveness.stamp("tid-ui", client=fake_redis)
        orphaned, _live = _celery_view(self._row("tid-ui"))
        assert orphaned is False

    def test_unstamped_still_is(self, fake_redis, stale_celery):
        from src.api.v1.endpoints.task_queue import _celery_view

        orphaned, _live = _celery_view(self._row("tid-gone"))
        assert orphaned is True
