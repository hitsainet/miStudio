"""
Periodic task to clean up stuck activation extraction jobs.

This task runs every 10 minutes and marks activation extractions as FAILED if they've been
stuck in QUEUED, LOADING, or EXTRACTING status for too long without updates.

Threshold: 1 hour without progress.
"""

import logging
from datetime import datetime, timezone, timedelta
from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask
from src.models.activation_extraction import ActivationExtraction, ExtractionStatus
from src.workers.websocket_emitter import emit_extraction_failed
from src.workers.job_progress import progress_stalled_seconds

logger = logging.getLogger(__name__)

STUCK_THRESHOLD_MINUTES = 60  # 1 hour


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_activations"
)
def cleanup_stuck_activations_task(self):
    """
    Clean up activation extraction jobs that have been stuck for more than 1 hour.

    An activation extraction is considered stuck if:
    - Status is QUEUED, LOADING, EXTRACTING, or SAVING
    - No update in the last hour
    - Either has no celery_task_id or task is not running
    """
    logger.info("Running stuck activation extraction cleanup task")

    with self.get_db() as db:
        try:
            stuck_threshold = datetime.now(timezone.utc) - timedelta(minutes=STUCK_THRESHOLD_MINUTES)

            stuck_extractions = db.query(ActivationExtraction).filter(
                ActivationExtraction.status.in_([
                    ExtractionStatus.QUEUED,
                    ExtractionStatus.LOADING,
                    ExtractionStatus.EXTRACTING,
                    ExtractionStatus.SAVING,
                ]),
                ActivationExtraction.updated_at < stuck_threshold
            ).all()

            cleaned_count = 0
            for extraction in stuck_extractions:
                # Check if Celery task is actually running
                task_is_running = False

                if extraction.celery_task_id:

                    # MIS-E2E-092: see task_looks_alive — a dead worker and a
                    # queued task are both PENDING, so the old state check could
                    # never be false for a row carrying a task id.
                    from src.workers.task_heartbeat import task_looks_alive

                    task_is_running = task_looks_alive(
                        extraction.celery_task_id,
                        extraction,
                        started=str(getattr(extraction, "status", "")).lower()
                        in ("extracting", "running", "processing"),
                    )
                    if task_is_running:
                        logger.info(
                            f"Activation extraction {extraction.id} has an active Celery "
                            f"task {extraction.celery_task_id}, skipping cleanup"
                        )


                # IS THE WORK ADVANCING? A quiet row and a stalled job are not
                # the same thing. Additive to the clock: reap only when stale
                # AND the counter has not moved. None means "no evidence" —
                # first sighting, or Redis unreachable — and must never read as
                # "stalled". See workers/job_progress.py for the incident that
                # motivated this.
                _stalled = progress_stalled_seconds(
                    "activation", extraction.id, getattr(extraction, "samples_processed", None)
                )
                if _stalled is not None and _stalled < STUCK_THRESHOLD_MINUTES * 60:
                    logger.info(
                        "%s advanced %.0fs ago; sparing it despite a stale row",
                        extraction.id, _stalled,
                    )
                    continue
                if not task_is_running:
                    age_minutes = (datetime.now(timezone.utc) - extraction.updated_at).total_seconds() / 60
                    logger.warning(
                        f"Marking stuck activation extraction {extraction.id} as FAILED "
                        f"(status: {extraction.status}, age: {age_minutes:.0f}min, "
                        f"task_id: {extraction.celery_task_id or 'None'})"
                    )

                    extraction.status = ExtractionStatus.FAILED
                    extraction.error_message = (
                        f"Extraction job stuck - no progress for more than {int(age_minutes)} minutes. "
                        "This may indicate a crashed worker or system issue."
                    )
                    extraction.error_type = "TIMEOUT"
                    extraction.completed_at = datetime.now(timezone.utc)
                    extraction.updated_at = datetime.now(timezone.utc)

                    db.commit()
                    cleaned_count += 1

                    # Emit WebSocket event to notify frontend
                    try:
                        emit_extraction_failed(
                            model_id=extraction.model_id,
                            extraction_id=extraction.id,
                            error_message=extraction.error_message,
                            error_type="TIMEOUT",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit WebSocket event for {extraction.id}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stuck activation extraction(s)")
            else:
                logger.info("No stuck activation extractions found")

            return {"cleaned": cleaned_count}

        except Exception as e:
            logger.error(f"Error in stuck activation extraction cleanup: {e}", exc_info=True)
            raise
