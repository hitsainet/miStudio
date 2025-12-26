"""
Periodic task to clean up stuck activation extraction jobs.

This task runs every 10 minutes and marks activation extractions as FAILED if they've been
stuck in QUEUED, LOADING, or EXTRACTING status for too long without updates.
"""

import logging
from datetime import datetime, timezone, timedelta
from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask
from src.models.activation_extraction import ActivationExtraction, ExtractionStatus
from src.workers.websocket_emitter import emit_extraction_failed

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_activations"
)
def cleanup_stuck_activations_task(self):
    """
    Clean up activation extraction jobs that have been stuck for more than 10 minutes.

    An activation extraction is considered stuck if:
    - Status is QUEUED, LOADING, or EXTRACTING
    - No update in the last 10 minutes
    - Either has no celery_task_id or task is not running
    """
    logger.info("Running stuck activation extraction cleanup task")

    with self.get_db() as db:
        try:
            # Find potentially stuck activation extractions
            stuck_threshold = datetime.now(timezone.utc) - timedelta(minutes=10)

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
                    from src.core.celery_app import get_task_status
                    task_status = get_task_status(extraction.celery_task_id)

                    if task_status['state'] in ['PENDING', 'STARTED', 'RETRY']:
                        task_is_running = True
                        logger.info(
                            f"Activation extraction {extraction.id} has active Celery task "
                            f"{extraction.celery_task_id} ({task_status['state']}), skipping cleanup"
                        )

                if not task_is_running:
                    # Mark as failed
                    logger.warning(
                        f"Marking stuck activation extraction {extraction.id} as FAILED "
                        f"(status: {extraction.status}, last update: {extraction.updated_at}, "
                        f"task_id: {extraction.celery_task_id or 'None'})"
                    )

                    extraction.status = ExtractionStatus.FAILED
                    extraction.error_message = (
                        "Extraction job stuck - no progress for more than 10 minutes. "
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
