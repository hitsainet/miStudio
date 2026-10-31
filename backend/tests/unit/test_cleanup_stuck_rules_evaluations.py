"""The rules-evaluation reaper and cache pruning (Feature 040 S3; FR-37; FTASKS 21.3, 21.6).

* a `running` row silent past `rules_evaluation_stale_after_seconds` (the RUNTIME setting) becomes `failed`,
  `worker_lost`, with a rerun `next`; a fresh one is untouched; nothing is requeued;
* the verdict cache is pruned by age and by size, least recently used first — never an entry a pending or
  running evaluation is reading (`in_use_by`).
The beat entry itself is pinned by `test_rules_evaluation_task.py::test_the_reaper_is_on_the_beat`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.core import runtime_settings
from src.workers import cleanup_stuck_rules_evaluations as reaper
from tests.support.next_calls import assert_next_calls_are_live


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


def evaluation(db, status, heartbeat=None):
    from src.models.rules_evaluation import ProbeRulesEvaluation

    row = ProbeRulesEvaluation(identity_sha256=uuid.uuid4().hex * 2, probe_id="pm_x", view_ids=["pmd_x"],
                               behaviour_key="k-x", behaviour_version=1, engine_digest="e" * 64, calibration=False,
                               status=status, inputs={}, created_by="operator", heartbeat_at=heartbeat,
                               started_at=heartbeat)
    db.add(row)
    db.commit()
    return row


def test_a_silent_running_row_fails_worker_lost(db):
    runtime_settings.apply({"rules_evaluation_stale_after_seconds": 120})
    now = datetime.now(timezone.utc)
    stale = evaluation(db, "running", now - timedelta(seconds=121))
    live = evaluation(db, "running", now - timedelta(seconds=60))
    pending = evaluation(db, "pending")
    out = reaper.sweep(db, now)
    db.refresh(stale), db.refresh(live), db.refresh(pending)
    assert (stale.status, stale.error["code"]) == ("failed", "worker_lost")
    assert_next_calls_are_live(stale.error["next"])
    assert live.status == "running" and pending.status == "pending" and out["worker_lost"] == 1


def _cache(db, tmp_path, view, used_days_ago, size, in_use_by=None):
    from src.models.rules_evaluation import RuleVerdictCache

    path = tmp_path / "rule_verdicts" / view / uuid.uuid4().hex
    path.mkdir(parents=True)
    (path / "rule_match.npy").write_bytes(b"x")
    now = datetime.now(timezone.utc)
    row = RuleVerdictCache(view_id=view, rule_set_digest="s" * 64, engine_digest=uuid.uuid4().hex * 2,
                           reading_digest="r" * 64, n_rows=1, bytes=size, path=str(path),
                           created_at=now, last_used_at=now - timedelta(days=used_days_ago), in_use_by=in_use_by)
    db.add(row)
    db.commit()
    return row, path


def test_pruning_by_age_and_size_spares_entries_in_use(db, tmp_path):
    runtime_settings.apply({"rule_verdict_cache_max_age_days": 30, "rule_verdict_cache_max_bytes": 250})
    running = evaluation(db, "running", datetime.now(timezone.utc))
    old, old_path = _cache(db, tmp_path, "pmd_a", 40, 100)
    old_in_use, in_use_path = _cache(db, tmp_path, "pmd_b", 40, 100, in_use_by=running.id)
    lru, lru_path = _cache(db, tmp_path, "pmd_c", 5, 100)
    recent, recent_path = _cache(db, tmp_path, "pmd_d", 1, 100)
    out = reaper.sweep(db)
    assert out["cache_pruned"] == {"age": 1, "size": 1}
    assert not old_path.exists() and not lru_path.exists()
    assert in_use_path.exists() and recent_path.exists()
