"""The blend janitor (Feature 041 T31; FTASKS 8.4): `worker_lost` and `time_limit`, never a requeue."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core import runtime_settings
from src.workers import cleanup_stuck_blend_reports as janitor


@pytest.fixture(autouse=True)
def fresh():
    runtime_settings.reset()
    yield
    runtime_settings.reset()


@pytest.fixture
def db(async_engine):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", "")))
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def report(db, status, *, heartbeat=None, created=None):
    from src.models.probe_blend import ProbeBlendReport

    row = ProbeBlendReport(behaviour_key="k-x", behaviour_version=1, model_id="m", status=status, request={},
                           resolved={}, created_by="operator", heartbeat_at=heartbeat,
                           created_at=created or datetime.now(timezone.utc))
    db.add(row)
    db.commit()
    return row


def test_stale_heartbeat_fails_worker_lost(db):
    now = datetime.now(timezone.utc)
    stale = report(db, "running", heartbeat=now - timedelta(seconds=janitor.STALE_HEARTBEAT_S + 5))
    fresh_row = report(db, "running", heartbeat=now - timedelta(seconds=5))
    reaped = janitor.sweep(db, now)
    db.refresh(stale), db.refresh(fresh_row)
    assert (stale.status, stale.failure_code) == ("failed", "worker_lost")
    assert fresh_row.status == "running" and reaped["worker_lost"] == 1


def test_queued_past_time_limit_fails(db):
    runtime_settings.apply({"blend_evaluation_time_limit_s": 120})
    now = datetime.now(timezone.utc)
    old = report(db, "queued", created=now - timedelta(seconds=121))
    young = report(db, "queued", created=now - timedelta(seconds=60))
    janitor.sweep(db, now)
    db.refresh(old), db.refresh(young)
    assert (old.status, old.failure_code) == ("failed", "time_limit") and young.status == "queued"
    assert old.failure_message and "rerun" in old.failure_message


def test_never_requeues(db, monkeypatch):
    from src.workers import probe_blend_tasks

    sent = []
    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay", lambda *a: sent.append(a))
    now = datetime.now(timezone.utc)
    row = report(db, "running", heartbeat=now - timedelta(hours=2))
    janitor.sweep(db, now)
    db.refresh(row)
    assert row.status == "failed" and sent == []


def test_the_beat_entry_is_wired():
    from src.core.celery_app import celery_app

    entry = celery_app.conf.beat_schedule["cleanup-stuck-blend-reports"]
    assert entry["task"] == "cleanup_stuck_blend_reports" and entry["schedule"] == 600.0
    # By NAME: once anything in the worker process has routed through `celery_app.amqp.router` (another test
    # file on the same xdist worker), Celery has expanded this option into a kombu `Queue` in place (Review R1:
    # the full run went red here while the file alone was green).
    queue = entry["options"]["queue"]
    assert getattr(queue, "name", queue) == "low_priority" and entry["options"]["expires"] < 600
    assert celery_app.conf.task_routes["cleanup_stuck_blend_reports"] == {"queue": "low_priority"}
    assert "cleanup_stuck_blend_reports" in celery_app.tasks
