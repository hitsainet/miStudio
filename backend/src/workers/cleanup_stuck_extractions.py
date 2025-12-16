"""
Periodic task to clean up stuck extraction jobs.

This task runs every 10 minutes and marks extraction jobs as FAILED if they've been
stuck in QUEUED or EXTRACTING status for too long without updates.
"""

import logging
from datetime import datetime, timezone, timedelta
from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.workers.websocket_emitter import emit_extraction_job_progress

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_extractions"
)
def cleanup_stuck_extractions_task(self):
    """
    Clean up extraction jobs that have been stuck for more than 10 minutes.

    An extraction is considered stuck if:
    - Status is QUEUED or EXTRACTING
    - No update in the last 10 minutes
    - Either has no celery_task_id or task is not running
    """
    logger.info("Running stuck extraction cleanup task")

    with self.get_db() as db:
        try:
            # Find potentially stuck extractions
            stuck_threshold = datetime.now(timezone.utc) - timedelta(minutes=10)

            stuck_extractions = db.query(ExtractionJob).filter(
                ExtractionJob.status.in_([
                    ExtractionStatus.QUEUED.value,
                    ExtractionStatus.EXTRACTING.value
                ]),
                ExtractionJob.updated_at < stuck_threshold
            ).all()

            cleaned_count = 0
            for extraction in stuck_extractions:
                # Check if Celery task is actually running
                task_is_running = False

                if extraction.celery_task_id:
                    from src.core.celery_app import get_task_status
                    task_status = get_task_status(extraction.celery_task_id)

                    if task_status['state'] in ['PENDING', 'STARTED', 'RETRY']:
                        task_is_running = True
                        logger.info(
                            f"Extraction {extraction.id} has active Celery task "
                            f"{extraction.celery_task_id} ({task_status['state']}), skipping cleanup"
                        )

                if not task_is_running:
                    # Mark as failed
                    logger.warning(
                        f"Marking stuck extraction {extraction.id} as FAILED "
                        f"(status: {extraction.status}, last update: {extraction.updated_at}, "
                        f"task_id: {extraction.celery_task_id or 'None'})"
                    )

                    extraction.status = ExtractionStatus.FAILED.value
                    extraction.error_message = (
                        "Extraction job stuck - no progress for more than 10 minutes. "
                        "This may indicate a crashed worker or system issue."
                    )
                    extraction.completed_at = datetime.now(timezone.utc)
                    extraction.updated_at = datetime.now(timezone.utc)

                    db.commit()
                    cleaned_count += 1

                    # Emit WebSocket event to notify frontend
                    try:
                        emit_extraction_job_progress(
                            extraction_id=extraction.id,
                            training_id=extraction.training_id,
                            sae_id=extraction.sae_id,
                            status="failed",
                            message=extraction.error_message,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit WebSocket event for {extraction.id}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stuck extraction(s)")
            else:
                logger.info("No stuck extractions found")

            return {"cleaned": cleaned_count}

        except Exception as e:
            logger.error(f"Error in stuck extraction cleanup: {e}", exc_info=True)
            raise
