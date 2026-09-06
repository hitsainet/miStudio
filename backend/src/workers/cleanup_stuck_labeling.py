"""Periodic reaper for bulk labeling jobs that stopped without reporting.

`labeling_jobs` was the only long-running lifecycle in this system with no
janitor. Enhanced labeling has had one since MIS-E2E-092; bulk labeling did not,
and its absence is what turned the 409 lock into a trap: a job orphaned by a
worker restart sits QUEUED forever and 409s **every future labeling run on that
extraction**, naming a job id that only a manual DELETE can clear.

Threshold is 45 minutes, not the enhanced sibling's 10. Bulk labeling
legitimately runs for hours over tens of thousands of features; the 10-minute
figure is calibrated for a per-feature job that finishes in seconds.

Two additive guards, both from the shipped janitor pattern:

* `task_looks_alive` — the Celery task is genuinely running, so a quiet row is
  not evidence of death.
* `progress_stalled_seconds` — the work is ADVANCING. `None` means no evidence
  and must never be read as "stalled"; reaping on absence is how a healthy job
  gets killed.
"""

import logging
from datetime import datetime, timezone, timedelta

from src.core.celery_app import celery_app
from src.models.labeling_job import LabelingJob, LabelingStatus
from src.models.labeling_trial_run import LabelingTrialRun
from src.workers.base_task import DatabaseTask
from src.workers.job_progress import progress_stalled_seconds
from src.workers.task_heartbeat import task_looks_alive

logger = logging.getLogger(__name__)

_STUCK_THRESHOLD_MINUTES = 45


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_labeling",
)
def cleanup_stuck_labeling_task(self):
    """Mark bulk labeling jobs stuck in QUEUED/LABELING as FAILED."""
    logger.info("Running stuck labeling cleanup task")

    with self.get_db() as db:
        try:
            threshold = datetime.now(timezone.utc) - timedelta(
                minutes=_STUCK_THRESHOLD_MINUTES)

            stuck = db.query(LabelingJob).filter(
                LabelingJob.status.in_([
                    LabelingStatus.QUEUED.value,
                    LabelingStatus.LABELING.value,
                ]),
                LabelingJob.updated_at < threshold,
            ).all()

            cleaned = 0
            for job in stuck:
                if job.celery_task_id and task_looks_alive(
                    job.celery_task_id, job,
                    started=str(getattr(job, "status", "")).lower() == "labeling",
                ):
                    logger.info(
                        "Labeling job %s has an active Celery task %s, skipping",
                        job.id, job.celery_task_id)
                    continue

                # IS THE WORK ADVANCING? A labeling run commits features_labeled
                # every batch, so a moving counter means the row's age is a lie.
                stalled = progress_stalled_seconds(
                    "labeling_job", job.id, getattr(job, "features_labeled", None))
                if stalled is not None and stalled < _STUCK_THRESHOLD_MINUTES * 60:
                    logger.info(
                        "%s advanced %.0fs ago; sparing it despite a stale row",
                        job.id, stalled)
                    continue

                # Capture BEFORE mutating: the message interpolates the status,
                # and assigning FAILED first makes every one of them read
                # "stuck in FAILED", discarding the only field that says what it
                # was actually stuck in.
                stuck_in = job.status
                stuck_minutes = int(
                    (datetime.now(timezone.utc) - job.updated_at).total_seconds() / 60)

                logger.warning(
                    "Marking stuck labeling job %s as FAILED (was %s, stuck %d min, "
                    "task_id: %s)", job.id, stuck_in, stuck_minutes,
                    job.celery_task_id or "None")

                job.status = LabelingStatus.FAILED.value
                job.error_message = (
                    f"Labeling job stuck in {stuck_in} for {stuck_minutes} minutes "
                    f"with no progress - the worker was restarted or the task was lost"
                )
                job.updated_at = datetime.now(timezone.utc)

                # A trial's RESULT row must be failed alongside its job, or the
                # panel stays locked against future trials by the in-flight check.
                if job.trial_run_id:
                    run = db.query(LabelingTrialRun).filter(
                        LabelingTrialRun.id == job.trial_run_id).first()
                    if run and run.status in ("queued", "running"):
                        run.status = "failed"
                        run.error = f"labeling job {job.id} was reaped as stuck"
                cleaned += 1

            if cleaned:
                db.commit()
            logger.info("Stuck labeling cleanup: %d job(s) marked FAILED", cleaned)
            return {"cleaned": cleaned, "scanned": len(stuck)}
        except Exception as exc:
            db.rollback()
            logger.error("Stuck labeling cleanup failed: %s", exc, exc_info=True)
            raise
