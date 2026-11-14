"""Workers and beat keep their old-name uses inside the 24-hour window (042 FR-042.9; FTID §3.4; CF-5 M7;
TPD-29; FTASKS 3.17).

Signals are SENT through Celery's live signal objects after importing the Celery app, so the wiring line
in `core/celery_app.py` is what connects the receivers (the `test_environment_refresh_points.py` pattern).
Time is a fake clock (`legacy_names.reset(clock=, mono=)`); the database sink is replaced by a collector.

Controls: K-3.38 (`connect_signals()` deleted), K-3.39 (the heartbeat thread never started), K-3.40 (the
refresh throttle set to 0).
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.core import legacy_names
from src.workers import legacy_names_signals as sig

T0 = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)


class Sender:
    """Celery hashes a signal's sender (a SimpleNamespace is not hashable)."""

    def __init__(self, hostname=None):
        self.hostname = hostname


class Clocks:
    def __init__(self):
        self.now = T0
        self.mono = 0.0

    def advance(self, seconds: float):
        self.now += timedelta(seconds=seconds)
        self.mono += seconds


@pytest.fixture
def world(monkeypatch):
    from src.services import legacy_names_store

    for r in legacy_names.ENV:
        monkeypatch.delenv(r.new, raising=False)
        monkeypatch.delenv(r.old, raising=False)
    clocks = Clocks()
    legacy_names.reset(clock=lambda: clocks.now, mono=lambda: clocks.mono)
    reports: list[dict] = []
    lock = threading.Lock()

    def collect(report):
        with lock:
            reports.append(report)

    monkeypatch.setattr(legacy_names_store, "db_sink", collect)
    sig.stop_heartbeat()
    yield SimpleNamespace(clocks=clocks, reports=reports, lock=lock)
    sig.stop_heartbeat()
    legacy_names.reset()


def _receivers(signal, uid):
    import src.core.celery_app  # noqa: F401  the import is what wires it

    return [ref for key, ref in signal.receivers if key[0] == uid]


def test_prerun_receiver_in_live_signal_registry():
    from celery.signals import beat_init, task_prerun, worker_ready

    for signal, uid in ((task_prerun, sig.DISPATCH_PRERUN), (worker_ready, sig.DISPATCH_WORKER_READY),
                        (beat_init, sig.DISPATCH_BEAT_INIT)):
        assert len(_receivers(signal, uid)) == 1, f"{uid}: not exactly one receiver"


def test_celery_app_import_wires_it():
    from celery.signals import task_prerun

    (ref,) = _receivers(task_prerun, sig.DISPATCH_PRERUN)
    target = ref() if callable(ref) and not hasattr(ref, "__code__") else ref
    assert getattr(target, "__name__", None) == "on_task_prerun", target
    assert getattr(target, "__module__", None) == sig.__name__


def test_25h_old_use_back_in_window_after_one_task(world, monkeypatch):
    from celery.signals import task_prerun

    _receivers(task_prerun, sig.DISPATCH_PRERUN)
    # The row is retired since ENH-009 P7; the refresh mechanism is pinned on it put back into transition.
    row = next(r for r in legacy_names.ENV if r.old == "MILLM_API_URL")
    live = legacy_names.EnvName(new=row.new, old=row.old, status="transition")
    monkeypatch.setitem(legacy_names._ENV_BY_NEW, row.new, live)  # noqa: SLF001
    monkeypatch.setitem(legacy_names._ENV_BY_OLD, row.old, live)  # noqa: SLF001
    monkeypatch.setattr(legacy_names, "ENV", tuple(live if r is row else r for r in legacy_names.ENV))
    monkeypatch.setenv("MILLM_API_URL", "http://old")
    legacy_names.set_sink(world.reports.append, "worker:celery@test")
    assert legacy_names.env("MISERVER_API_URL") == "http://old"
    first = world.reports[-1]["uses"][0]
    assert first["last_seen"] == "2026-11-08T12:00:00Z"
    world.clocks.advance(25 * 3600)
    n = len(world.reports)
    task_prerun.send(sender=None, task_id="t-1", task=None)
    assert len(world.reports) == n + 1, "one task_prerun must report once"
    use = next(u for u in world.reports[-1]["uses"] if u["old"] == "MILLM_API_URL")
    assert use["last_seen"] == "2026-11-09T13:00:00Z", "the task re-stamped the use still in force"
    assert world.reports[-1]["reporter"] == "worker:celery@test"


def _wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_idle_heartbeat_refreshes(world, monkeypatch):
    from celery.signals import worker_ready

    _receivers(worker_ready, sig.DISPATCH_WORKER_READY)
    monkeypatch.setattr(sig, "HEARTBEAT_S", 0.05)
    # each refresh sees a minute pass, so the throttle never hides a beat
    real_refresh = legacy_names.refresh

    def stepped():
        world.clocks.advance(61)
        return real_refresh()

    monkeypatch.setattr(legacy_names, "refresh", stepped)
    worker_ready.send(sender=Sender("celery@idle"))
    assert legacy_names.reporter() == "worker:celery@idle"
    assert _wait_for(lambda: len(world.reports) >= 3), f"{len(world.reports)} heartbeat report(s)"
    with world.lock:
        assert all(r["reporter"] == "worker:celery@idle" and r["uses"] == [] for r in world.reports)


def test_refresh_writes_at_most_once_a_minute(world):
    legacy_names.set_sink(world.reports.append, "worker:celery@test")
    ran = [legacy_names.refresh() for _ in range(5)]
    assert ran == [True, False, False, False, False] and len(world.reports) == 1
    world.clocks.advance(59)
    assert legacy_names.refresh() is False and len(world.reports) == 1
    world.clocks.advance(1)
    assert legacy_names.refresh() is True and len(world.reports) == 2


def test_beat_refreshes_on_heartbeat(world, monkeypatch):
    from celery.signals import beat_init

    _receivers(beat_init, sig.DISPATCH_BEAT_INIT)
    monkeypatch.setattr(sig, "HEARTBEAT_S", 0.05)
    real_refresh = legacy_names.refresh

    def stepped():
        world.clocks.advance(61)
        return real_refresh()

    monkeypatch.setattr(legacy_names, "refresh", stepped)
    beat_init.send(sender=Sender())
    assert legacy_names.reporter() == "beat"
    assert _wait_for(lambda: len(world.reports) >= 2)
    with world.lock:
        assert {r["reporter"] for r in world.reports} == {"beat"}
