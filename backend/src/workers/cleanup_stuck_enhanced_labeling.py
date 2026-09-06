"""
Periodic task to clean up stuck enhanced labeling jobs.

Runs every 5 minutes and marks jobs as FAILED if they've been stuck in
QUEUED or RUNNING status for more than 10 minutes without any update.
Enhanced labeling jobs are short-lived (seconds to a few minutes), so
a 10-minute threshold is conservative enough to avoid false positives.
"""

import logging
from datetime import datetime, timezone, timedelta

from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask
from src.models.enhanced_labeling_job import EnhancedLabelingJob, EnhancedLabelingStatus
from src.workers.websocket_emitter import emit_enhanced_labeling_failed
from src.workers.job_progress import progress_stalled_seconds

logger = logging.getLogger(__name__)

_STUCK_THRESHOLD_MINUTES = 10


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_enhanced_labeling",
)
def cleanup_stuck_enhanced_labeling_task(self):
    """Mark enhanced labeling jobs stuck in QUEUED/RUNNING as FAILED."""
    logger.info("Running stuck enhanced labeling cleanup task")

    with self.get_db() as db:
        try:
            threshold = datetime.now(timezone.utc) - timedelta(minutes=_STUCK_THRESHOLD_MINUTES)

            stuck_jobs = db.query(EnhancedLabelingJob).filter(
                EnhancedLabelingJob.status.in_([
                    EnhancedLabelingStatus.QUEUED.value,
                    EnhancedLabelingStatus.RUNNING.value,
                ]),
                EnhancedLabelingJob.updated_at < threshold,
            ).all()

            cleaned = 0
            for job in stuck_jobs:
                task_is_active = False
                if job.celery_task_id:
                    # MIS-E2E-092. This janitor's own error text — "the worker
                    # was restarted or the task was lost" — names precisely the
                    # state Celery reports as PENDING and this check read as
                    # healthy, so it could never fire for the case it describes.
                    from src.workers.task_heartbeat import task_looks_alive

                    task_is_active = task_looks_alive(
                        job.celery_task_id,
                        job,
                        started=str(getattr(job, "status", "")).lower()
                        in ("processing", "running", "in_progress"),
                    )
                    if task_is_active:
                        logger.info(
                            "Enhanced labeling job %s has an active Celery task %s, skipping",
                            job.id, job.celery_task_id,
                        )

                if not task_is_active:
                    stuck_minutes = int(
                        (datetime.now(timezone.utc) - job.updated_at).total_seconds() / 60
                    )
                    # IS THE WORK ADVANCING? Additive to the clock — reap only
                    # when stale AND the counter has not moved. None means "no
                    # evidence" and must never read as "stalled".
                    _stalled = progress_stalled_seconds("labeling", job.id, getattr(job, "examples_completed", None))
                    if _stalled is not None and _stalled < _STUCK_THRESHOLD_MINUTES * 60:
                        logger.info(
                            "%s advanced %.0fs ago; sparing it despite a stale row",
                            job.id, _stalled,
                        )
                        continue

                    logger.warning(
                        "Marking stuck enhanced labeling job %s as FAILED "
                        "(status: %s, stuck for %d min, task_id: %s)",
                        job.id, job.status, stuck_minutes, job.celery_task_id or "None",
                    )
                    # MIS-E2E-096: capture BEFORE mutating. The message below
                    # interpolates the status, and assigning FAILED first made
                    # every one of them read "Job stuck in FAILED", discarding
                    # the only field that says what it was actually stuck in —
                    # which is the whole diagnostic value of the message.
                    stuck_in = job.status
                    job.status = EnhancedLabelingStatus.FAILED.value
                    job.phase = None
                    job.error_message = (
                        f"Job stuck in {stuck_in} for {stuck_minutes} minutes with no progress. "
                        "This may indicate the Celery worker was restarted or the task was lost. "
                        "Please try again."
                    )
                    job.completed_at = datetime.now(timezone.utc)
                    job.updated_at = datetime.now(timezone.utc)
                    db.commit()
                    cleaned += 1

                    try:
                        emit_enhanced_labeling_failed(job.id, job.error_message)
                    except Exception as e:
                        logger.warning("Failed to emit WebSocket event for %s: %s", job.id, e)

            if cleaned:
                logger.info("Cleaned up %d stuck enhanced labeling job(s)", cleaned)
            else:
                logger.info("No stuck enhanced labeling jobs found")

            return {"cleaned": cleaned}

        except Exception as e:
            logger.error("Error in stuck enhanced labeling cleanup: %s", e, exc_info=True)
            raise
