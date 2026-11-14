"""An abandoned NLP pass must stop claiming to be running.

OBSERVED 2026-07-26: a pod roll killed an NLP pass at 16,217/32,759 features.
Minutes later the row still read `nlp_status='processing'` — and nothing in the
system would ever have corrected it, because every `cleanup_stuck_*` task
watches `ExtractionJob.status`, not `nlp_status`.

Same failure shape as the queue starvation found the same day: not a crash,
just going quietly dark.

MUTATION CONTROLS:
  * remove the beat_schedule entry            -> reachability test fails
  * drop the exact-name route                 -> routing test fails
  * clear nlp_processed_count on cleanup      -> resume test fails
  * mark ExtractionJob.status failed too      -> extraction-intact test fails
  * consult celery_task_id to decide liveness -> wrong-task test fails
"""

import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.workers import cleanup_stuck_nlp as mod
from src.workers.cleanup_stuck_nlp import (
    IN_FLIGHT_NLP_STATUSES,
    NLP_PENDING_THRESHOLD_MINUTES,
    NLP_STALE_THRESHOLD_MINUTES,
    _threshold_for,
)


def make_job(**kw):
    now = datetime.now(timezone.utc)
    defaults = dict(
        id="extr_1",
        status="completed",
        nlp_status="processing",
        nlp_processed_count=16217,
        nlp_progress=0.495,
        nlp_error_message=None,
        updated_at=now - timedelta(minutes=90),
        celery_task_id="extraction-task-id",
        training_id=None,
        external_sae_id="sae_1",
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class FakeQuery:
    def __init__(self, rows): self._rows = rows
    def filter(self, *a, **k): return self
    def all(self): return self._rows


class FakeDB:
    def __init__(self, rows): self._rows = rows; self.commits = 0; self.rollbacks = 0
    def query(self, *a): return FakeQuery(self._rows)
    def commit(self): self.commits += 1
    def rollback(self): self.rollbacks += 1


class Ctx:
    """Stand-in for DatabaseTask.get_db()'s context manager."""

    def __init__(self, db): self._db = db
    def __enter__(self): return self._db
    def __exit__(self, *a): return False


def run_task(rows, monkeypatch, emit=None):
    """Invoke the real task body against a fake session.

    The task is bind=True, so `self` is already bound on the registered task —
    patching `get_db` on it is what injects the fake session.
    """
    db = FakeDB(rows)
    emitted = []

    monkeypatch.setattr(
        mod.cleanup_stuck_nlp_task, "get_db", lambda: Ctx(db), raising=False
    )
    monkeypatch.setattr(
        "src.workers.nlp_analysis_tasks.emit_nlp_analysis_progress",
        emit or (lambda **kw: emitted.append(kw)),
    )
    result = mod.cleanup_stuck_nlp_task.run()
    return db, result, emitted


class TestItCleansWhatIsActuallyStuck:
    def test_a_long_silent_processing_pass_is_failed(self, monkeypatch):
        job = make_job(updated_at=datetime.now(timezone.utc) - timedelta(minutes=90))
        _, result, _ = run_task([job], monkeypatch)

        assert result["cleaned"] == 1
        assert job.nlp_status == "failed"
        assert "16,217" in job.nlp_error_message

    def test_a_recently_active_pass_is_left_alone(self, monkeypatch):
        # The loop commits every ~1.4s; two minutes of silence is normal.
        job = make_job(updated_at=datetime.now(timezone.utc) - timedelta(minutes=2))
        _, result, _ = run_task([job], monkeypatch)

        assert result["cleaned"] == 0
        assert job.nlp_status == "processing"

    def test_pending_gets_a_shorter_grace_period(self, monkeypatch):
        """'pending' means the task was never picked up, not that it stalled."""
        assert NLP_PENDING_THRESHOLD_MINUTES < NLP_STALE_THRESHOLD_MINUTES
        assert _threshold_for("pending") == NLP_PENDING_THRESHOLD_MINUTES
        assert _threshold_for("processing") == NLP_STALE_THRESHOLD_MINUTES

        age = NLP_PENDING_THRESHOLD_MINUTES + 1
        job = make_job(
            nlp_status="pending",
            nlp_processed_count=0,
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=age),
        )
        _, result, _ = run_task([job], monkeypatch)
        assert result["cleaned"] == 1

    def test_a_pending_row_inside_its_grace_period_survives(self, monkeypatch):
        age = NLP_PENDING_THRESHOLD_MINUTES - 1
        job = make_job(
            nlp_status="pending",
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=age),
        )
        _, result, _ = run_task([job], monkeypatch)
        assert result["cleaned"] == 0

    def test_a_row_without_a_timestamp_is_skipped_not_guessed(self, monkeypatch):
        job = make_job(updated_at=None)
        _, result, _ = run_task([job], monkeypatch)
        assert result["cleaned"] == 0
        assert job.nlp_status == "processing"

    def test_naive_timestamps_do_not_crash(self, monkeypatch):
        job = make_job(updated_at=datetime.now() - timedelta(minutes=90))
        _, result, _ = run_task([job], monkeypatch)
        assert result["cleaned"] == 1

    def test_only_in_flight_statuses_are_candidates(self):
        assert set(IN_FLIGHT_NLP_STATUSES) == {"pending", "processing"}
        for terminal in ("completed", "failed", "cancelled"):
            assert terminal not in IN_FLIGHT_NLP_STATUSES


class TestItDoesNotDestroyRecoverableWork:
    def test_analysed_features_are_kept_for_resume(self, monkeypatch):
        """Resuming with force_reprocess=false continues from this count."""
        job = make_job(nlp_processed_count=16217, nlp_progress=0.495)
        run_task([job], monkeypatch)

        assert job.nlp_processed_count == 16217
        assert job.nlp_progress == 0.495

    def test_the_extraction_itself_is_untouched(self, monkeypatch):
        """NLP is post-processing; the feature set is good."""
        job = make_job(status="completed")
        run_task([job], monkeypatch)

        assert job.status == "completed"

    def test_the_error_message_says_progress_is_kept(self, monkeypatch):
        job = make_job()
        run_task([job], monkeypatch)
        assert "resuming continues from there" in job.nlp_error_message


class TestItTellsTheUI:
    def test_a_failure_event_is_emitted(self, monkeypatch):
        job = make_job()
        _, _, emitted = run_task([job], monkeypatch)

        assert len(emitted) == 1
        assert emitted[0]["event"] == "failed"
        assert emitted[0]["extraction_job_id"] == "extr_1"
        assert emitted[0]["data"]["features_analyzed"] == 16217

    def test_an_emit_failure_does_not_abort_the_sweep(self, monkeypatch):
        def boom(**kw):
            raise RuntimeError("ws down")

        _, result, _ = run_task(
            [make_job(id="a"), make_job(id="b")], monkeypatch, emit=boom
        )
        assert result["cleaned"] == 2


class TestItAsksTheRightQuestion:
    def test_liveness_is_not_decided_from_celery_task_id(self):
        """`celery_task_id` is the EXTRACTION task, not the NLP task.

        There is no nlp_celery_task_id column, so consulting it would answer a
        different question — confidently and wrongly.
        """
        src = inspect.getsource(mod.cleanup_stuck_nlp_task)
        assert "celery_task_id" not in src, (
            "cleanup consults celery_task_id, which belongs to the extraction "
            "task; a finished extraction would make a dead NLP pass look alive"
        )

    def test_the_module_explains_why(self):
        assert "nlp_celery_task_id" in (mod.__doc__ or ""), (
            "the reason there is no task-liveness check is undocumented, so the "
            "next reader will 'fix' it by adding one"
        )
