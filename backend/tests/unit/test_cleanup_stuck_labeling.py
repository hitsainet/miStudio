"""The bulk-labeling janitor: `labeling_jobs` was the only long-running
lifecycle in this system without one.

Its absence is what made the 409 lock a trap. A job orphaned by a worker restart
sits QUEUED forever and 409s every future labeling run on that extraction,
naming a job id only a manual DELETE can clear.

Mutation controls:
  C49 remove the task_looks_alive check      -> test_a_live_task_is_spared
  C50 remove the progress gate               -> test_an_advancing_job_is_spared
  C51 treat a None progress marker as stalled-> test_absence_of_evidence_is_not_evidence
  C52 assign FAILED before reading the status-> test_the_message_names_the_status_it_was_stuck_in
  C53 skip failing the trial run             -> test_a_reaped_trial_releases_its_panel
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from src.models.labeling_job import LabelingJob, LabelingMode, LabelingStatus
from src.models.labeling_trial_run import LabelingTrialRun
from src.workers.cleanup_stuck_labeling import (
    _STUCK_THRESHOLD_MINUTES,
    cleanup_stuck_labeling_task,
)


def _job(**kw):
    j = Mock(spec=LabelingJob)
    j.id = kw.get("id", "label_1")
    j.status = kw.get("status", LabelingStatus.LABELING.value)
    j.updated_at = kw.get(
        "updated_at", datetime.now(timezone.utc) - timedelta(minutes=120))
    j.celery_task_id = kw.get("celery_task_id")
    j.features_labeled = kw.get("features_labeled", 5)
    j.trial_run_id = kw.get("trial_run_id")
    j.error_message = None
    return j


class _Ctx:
    def __init__(self, db): self._db = db
    def __enter__(self): return self._db
    def __exit__(self, *a): return False


def _run(jobs, monkeypatch, *, alive=False, stalled=None, trial=None):
    """Drive the real task with a fake session.

    `get_db` is patched on the task INSTANCE (the Celery PromiseProxy), matching
    tests/unit/test_cleanup_stuck_nlp.py — patching the class does not work
    because the decorator yields a proxy, not the class.
    """
    db = Mock()
    q = Mock()
    q.filter.return_value = q
    q.all.return_value = jobs
    q.first.return_value = trial
    db.query.return_value = q
    db.commit = Mock()
    db.rollback = Mock()

    monkeypatch.setattr(
        cleanup_stuck_labeling_task, "get_db", lambda: _Ctx(db), raising=False)
    monkeypatch.setattr(
        "src.workers.cleanup_stuck_labeling.task_looks_alive",
        lambda *a, **k: alive)
    monkeypatch.setattr(
        "src.workers.cleanup_stuck_labeling.progress_stalled_seconds",
        lambda *a, **k: stalled)
    return cleanup_stuck_labeling_task.run(), db


class TestTheJanitorReaps:
    def test_a_genuinely_stuck_job_is_failed_with_a_reason(self, monkeypatch):
        job = _job()
        result, _ = _run([job], monkeypatch, stalled=None)
        assert result["cleaned"] == 1
        assert job.status == LabelingStatus.FAILED.value
        assert job.error_message and "no progress" in job.error_message

    def test_the_message_names_the_status_it_was_stuck_in(self, monkeypatch):
        """C52. Assigning FAILED first made every message read 'stuck in FAILED',
        discarding the only field that said what it was actually stuck in."""
        job = _job(status=LabelingStatus.QUEUED.value)
        _run([job], monkeypatch, stalled=None)
        assert "stuck in queued" in job.error_message.lower(), job.error_message
        assert "stuck in failed" not in job.error_message.lower()


class TestTheJanitorSpares:
    def test_a_live_task_is_spared(self, monkeypatch):
        """C49. A quiet row is not evidence of death while the task is running."""
        job = _job(celery_task_id="celery-123")
        result, _ = _run([job], monkeypatch, alive=True)
        assert result["cleaned"] == 0
        assert job.status == LabelingStatus.LABELING.value

    def test_an_advancing_job_is_spared(self, monkeypatch):
        """C50. Bulk labeling commits features_labeled every batch; a moving
        counter means the row's age is a lie."""
        job = _job()
        result, _ = _run([job], monkeypatch, stalled=30.0)
        assert result["cleaned"] == 0
        assert job.status == LabelingStatus.LABELING.value

    def test_absence_of_evidence_is_not_evidence(self, monkeypatch):
        """C51. `None` means the marker said nothing — no Redis, first sighting.
        Reading that as 'stalled' is how a healthy job gets killed; reading it as
        'advancing' is how a dead one survives. It must fall through to the clock.
        """
        job = _job()
        result, _ = _run([job], monkeypatch, stalled=None)
        assert result["cleaned"] == 1, (
            "a None progress marker prevented the clock-based reap; absence of "
            "evidence was treated as evidence of progress"
        )

    def test_the_threshold_is_longer_than_the_enhanced_sibling(self):
        """Bulk labeling runs for hours. The 10-minute figure next door is
        calibrated for a per-feature job that finishes in seconds, and using it
        here would reap healthy runs constantly."""
        from src.workers.cleanup_stuck_enhanced_labeling import (
            _STUCK_THRESHOLD_MINUTES as ENHANCED,
        )
        assert _STUCK_THRESHOLD_MINUTES > ENHANCED
        assert _STUCK_THRESHOLD_MINUTES >= 45


class TestTrialRunsAreReleased:
    def test_a_reaped_trial_releases_its_panel(self, monkeypatch):
        """C53. A trial's RESULT row must fail alongside its job, or the
        in-flight check keeps the panel locked against every future trial."""
        run = Mock(spec=LabelingTrialRun)
        run.id, run.status, run.error = "ltr_1", "running", None
        job = _job(trial_run_id="ltr_1")
        _run([job], monkeypatch, stalled=None, trial=run)
        assert run.status == "failed"
        assert run.error and "reaped" in run.error


class TestItIsWiredIn:
    def test_it_is_registered_and_routed_and_scheduled(self):
        """Declaring a janitor is not running one."""
        from src.core.celery_app import celery_app
        import src.workers.cleanup_stuck_labeling  # noqa: F401

        assert "cleanup_stuck_labeling" in celery_app.tasks
        route = celery_app.amqp.router.route({}, "cleanup_stuck_labeling")
        queue = route.get("queue")
        assert str(getattr(queue, "name", queue)) == "low_priority", (
            f"routed to {queue!r}; a short task name with no explicit route "
            f"lands silently on the default queue"
        )
        assert "cleanup-stuck-labeling" in celery_app.conf.beat_schedule
