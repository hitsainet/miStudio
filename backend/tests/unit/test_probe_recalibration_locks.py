"""C2 and C3 on the REAL database: two re-cuts committed at once, and a commit over a decision the
caller never read (Feature 039 FTASKS 4.3).

⚠ A STUB SESSION CANNOT PROVE A ROW LOCK. `with_for_update` on a `SimpleNamespace` is whatever the
stub says it is; only two real sessions in two threads, released together by a barrier, show that
the second commit WAITS for the first and then sees the decision it moved. Without the lock both
read the same history, both append "revision 2", and one revision number names two bars.
"""
from __future__ import annotations

import logging
import threading

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.services import millm_read
from src.services import probe_recalibration as R
from tests.unit.test_probe_decision import make_rows, sync_db  # noqa: F401


@pytest.fixture
def session_factory(async_engine):
    url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", "")
    )
    engine = create_engine(url, pool_size=4)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def _no_millm(monkeypatch):
    monkeypatch.setattr(millm_read, "imported_copies",
                        lambda pid: millm_read.NotConfigured("not connected"))


def test_concurrent_commits_one_wins(sync_db, session_factory, tmp_path, monkeypatch, caplog):
    """Two sessions propose against the SAME decision, then commit at once: one commits revision
    2; the other waits on the row lock, finds the decision moved, and is refused `decision_changed`
    — no duplicate revision."""
    rows = make_rows(sync_db, tmp_path)
    probe_id = rows.probe.id
    barrier = threading.Barrier(2)
    real_lock = R._lock_row

    def lock_after_both_proposed(db, probe):
        barrier.wait(timeout=20)
        return real_lock(db, probe)

    monkeypatch.setattr(R, "_lock_row", lock_after_both_proposed)
    outcomes: dict = {}

    def worker(name, target):
        session = session_factory()
        try:
            outcomes[name] = R.recalibrate_probe(session, probe_id, target_fpr=target, commit=True)
        except R.RecalibrationRefused as refused:
            outcomes[name] = refused
        finally:
            session.close()

    with caplog.at_level(logging.INFO, logger="src.services.probe_recalibration"):
        threads = [threading.Thread(target=worker, args=("a", 0.02)),
                   threading.Thread(target=worker, args=("b", 0.005))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    refused = [o for o in outcomes.values() if isinstance(o, R.RecalibrationRefused)]
    won = [o for o in outcomes.values() if isinstance(o, dict)]
    assert len(won) == 1 and len(refused) == 1, outcomes
    assert refused[0].code == "decision_changed"
    sync_db.expire_all()
    history = rows.probe.calibration_history
    assert [entry["revision"] for entry in history] == [1, 2], (
        "both commits appended a revision: one revision number now names two bars"
    )
    assert refused[0].extra["current"] == rows.probe.decision_digest == (
        won[0]["current"]["decision_digest"]
    )
    assert any(r.getMessage().startswith(f"decision_changed probe={probe_id}")
               for r in caplog.records)


def test_expect_digest_refuses(sync_db, tmp_path):
    """C3: a commit naming a decision other than the one in force is refused; naming the one in
    force commits."""
    rows = make_rows(sync_db, tmp_path)
    preview = R.recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.02, commit=False)
    held = preview["current"]["decision_digest"]
    R.recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.005, commit=True)
    with pytest.raises(R.RecalibrationRefused) as refused:
        R.recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.02, commit=True,
                            expect_decision_digest=held)
    assert refused.value.code == "decision_changed"
    assert refused.value.extra["expected"] == held
    sync_db.expire_all()
    out = R.recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.02, commit=True,
                              expect_decision_digest=rows.probe.decision_digest)
    assert out["committed"] is True


def test_a_not_formable_decision_moved_under_a_held_proposal_is_refused(sync_db, tmp_path):
    """Review R1 (Slice 2): a probe whose decision is NOT formable has no digest, so a proposal held
    while another re-cut commits "started from None" and the locked row is "None" too — the
    digest comparison saw no move and the second commit overwrote the first's bar with no refusal.
    The revision under the lock is what moved; the held proposal is refused and nothing is written."""
    rows = make_rows(sync_db, tmp_path, hf_repo_id=None)
    held = R.recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.02, commit=False)
    assert held["current"]["decision_digest"] is None, "the fixture must be NOT formable"
    moved = R.recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.005, commit=True)
    assert moved["committed"] is True and moved["current"]["decision_digest"] is None
    sync_db.expire_all()
    with pytest.raises(R.RecalibrationRefused) as refused:
        R.apply(sync_db, rows.probe, held)
    assert refused.value.code == "decision_changed"
    sync_db.expire_all()
    assert [entry["revision"] for entry in rows.probe.calibration_history] == [1, 2]
    assert rows.probe.target_fpr == 0.005, "the refused proposal overwrote the committed bar"
