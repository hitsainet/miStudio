"""The extraction reaper must not kill work that is waiting its turn.

THE INCIDENT (2026-08-28). A batch of three SAE feature extractions started at
08:08. At 11:14 the janitor failed the third with:

    Extraction job stuck - no progress for more than 185 minutes.
    This may indicate a crashed worker or system issue.

Nothing had crashed. From the live database:

    pos 1/3  completed   progress 1.000  task set   ran 2h49m (169 min)
    pos 2/3  extracting  progress 0.128  task set   row updated <1s ago
    pos 3/3  failed      progress 0.000  task NULL  reaped at 186 min

Only the first member of a batch is dispatched; `_start_next_batch_job` starts
each successor after the previous one's NLP completes. So member 3 waits ~5.6
hours behind two ~2.8-hour predecessors, against BATCH_QUEUED_THRESHOLD_MINUTES
= 180. The third job of any three-job batch was structurally guaranteed to die.

`task_looks_alive` gave it no cover either: it returns False immediately when
`celery_task_id is None`, so the flat grace period was the only defence.

This file did not exist before. The janitor's thresholds, message text and
column writes were entirely unpinned.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.models.extraction_job import ExtractionStatus
from src.workers import cleanup_stuck_extractions as mod

NOW = datetime(2026, 8, 28, 11, 14, 0, tzinfo=timezone.utc)


def job(**kw):
    base = dict(
        id="extr_x",
        status=ExtractionStatus.QUEUED.value,
        celery_task_id=None,
        batch_id=None,
        batch_position=None,
        batch_total=None,
        progress=0.0,
        updated_at=NOW - timedelta(minutes=186),
        error_message=None,
        completed_at=None,
        training_id=None,
        external_sae_id=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Query:
    def __init__(self, db):
        self._db = db

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._db.candidates


class _DB:
    def __init__(self, candidates):
        self.candidates = candidates
        self.commits = 0

    def query(self, *a, **k):
        return _Query(self)

    def commit(self):
        self.commits += 1


def run(db, *, live_sibling=False, batch_idle_minutes=None, stalled=None,
        task_alive=False, restart_ok=False):
    """Drive the sweep with each external signal pinned.

    `restart_ok` defaults to FALSE deliberately. With it on, the chain-restart
    path spares every queued member and the batch checks above it are never
    exercised — the first version of these tests passed against a janitor that
    still judged queued members by row age. Only the restart test turns it on.
    """
    last_activity = (
        NOW - timedelta(minutes=batch_idle_minutes)
        if batch_idle_minutes is not None
        else None
    )

    class _Ctx:
        def __enter__(self_inner):
            return db

        def __exit__(self_inner, *a):
            return False

    with patch.object(mod.cleanup_stuck_extractions_task, "get_db", lambda: _Ctx()), \
         patch.object(mod, "batch_has_live_sibling", lambda *a, **k: live_sibling), \
         patch.object(mod, "batch_last_activity", lambda *a, **k: last_activity), \
         patch.object(mod, "progress_stalled_seconds", lambda *a, **k: stalled), \
         patch.object(mod, "_try_restart_batch", lambda *a, **k: restart_ok), \
         patch.object(mod, "emit_extraction_job_progress", lambda **k: True), \
         patch("src.workers.task_heartbeat.task_looks_alive", lambda *a, **k: task_alive), \
         patch.object(mod, "datetime", _FrozenDatetime):
        return mod.cleanup_stuck_extractions_task.run()


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class TestTheIncident:
    """The exact configuration that was reaped, reproduced."""

    def _member_three(self):
        return job(
            id="extr_20260828_080835_sae_sae_5bad_003",
            status=ExtractionStatus.QUEUED.value,
            celery_task_id=None,
            batch_id="batch_20260828_080834_3514f7a7",
            batch_position=3,
            batch_total=3,
            progress=0.0,
            updated_at=NOW - timedelta(minutes=186),
        )

    def test_a_member_waiting_behind_a_live_sibling_is_spared(self):
        """Member 2 was mid-run with its row updated seconds earlier."""
        db = _DB([self._member_three()])
        # Batch silent for longer than the threshold AND restart unavailable:
        # the live sibling must be the only thing standing between this job and
        # deletion, or the test proves nothing.
        result = run(db, live_sibling=True, batch_idle_minutes=400, restart_ok=False)

        assert result == {"cleaned": 0}
        assert db.candidates[0].status == ExtractionStatus.QUEUED.value
        assert db.candidates[0].error_message is None

    def test_it_is_spared_even_at_186_minutes_of_its_own_age(self):
        """Row age is the signal that condemned it. It must no longer decide."""
        db = _DB([self._member_three()])
        # No live sibling, no restart: only "the batch moved 1 minute ago" can
        # spare it, despite its own row being 186 minutes old.
        run(db, live_sibling=False, batch_idle_minutes=1, restart_ok=False)

        assert db.candidates[0].status == ExtractionStatus.QUEUED.value

    def test_a_silent_batch_restarts_the_chain_before_condemning(self):
        """The queued WORK is fine; nothing dispatched it."""
        db = _DB([self._member_three()])
        result = run(db, live_sibling=False, batch_idle_minutes=400, restart_ok=True)

        assert result == {"cleaned": 0}
        assert db.candidates[0].status == ExtractionStatus.QUEUED.value

    def test_a_dead_chain_that_cannot_restart_is_finally_reclaimed(self):
        db = _DB([self._member_three()])
        result = run(db, live_sibling=False, batch_idle_minutes=400, restart_ok=False)

        assert result == {"cleaned": 1}
        assert db.candidates[0].status == ExtractionStatus.FAILED.value

    def test_the_message_does_not_blame_a_crashed_worker(self):
        """It said "may indicate a crashed worker" to a user whose batch was
        working. For a queued member that is simply false."""
        db = _DB([self._member_three()])
        run(db, live_sibling=False, batch_idle_minutes=400, restart_ok=False)

        msg = db.candidates[0].error_message
        assert "Batch stopped advancing" in msg
        assert "position 3 of 3" in msg
        assert "crashed worker" not in msg


class TestProgressDecidesForRunningJobs:
    def _running(self, progress=0.128):
        return job(
            id="extr_running",
            status=ExtractionStatus.EXTRACTING.value,
            celery_task_id="task-1",
            progress=progress,
            updated_at=NOW - timedelta(minutes=90),
        )

    def test_a_job_that_advanced_recently_is_spared(self):
        """Slow is not dead. A 10,000-sample extraction at 1 sample/s runs for
        nearly three hours."""
        db = _DB([self._running()])
        result = run(db, stalled=120.0)

        assert result == {"cleaned": 0}
        assert db.candidates[0].status == ExtractionStatus.EXTRACTING.value

    def test_a_job_whose_progress_has_not_moved_is_reclaimed(self):
        db = _DB([self._running()])
        result = run(db, stalled=7200.0)

        assert result == {"cleaned": 1}
        assert db.candidates[0].status == ExtractionStatus.FAILED.value

    def test_no_evidence_falls_back_to_the_previous_behaviour(self):
        """None means Redis is unreachable or this is a first sighting. It must
        not read as 'stalled', and must not read as 'alive' either — the clock
        decides, exactly as before."""
        db = _DB([self._running()])
        result = run(db, stalled=None)

        assert result == {"cleaned": 1}

    def test_the_message_names_where_it_stalled(self):
        db = _DB([self._running(progress=0.128)])
        run(db, stalled=7200.0)

        msg = db.candidates[0].error_message
        assert "12.8%" in msg
        assert "Batch stopped advancing" not in msg


class TestUnchangedSafeguards:
    def test_a_live_celery_task_is_still_spared(self):
        db = _DB([job(
            id="extr_live",
            status=ExtractionStatus.EXTRACTING.value,
            celery_task_id="task-1",
            updated_at=NOW - timedelta(minutes=600),
        )])
        assert run(db, task_alive=True, stalled=99999.0) == {"cleaned": 0}

    def test_a_reclaimed_job_records_when_it_ended(self):
        db = _DB([job(
            id="extr_dead",
            status=ExtractionStatus.EXTRACTING.value,
            celery_task_id="task-1",
            updated_at=NOW - timedelta(minutes=600),
        )])
        run(db, stalled=99999.0)
        assert db.candidates[0].completed_at is not None
