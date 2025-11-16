"""
Periodic task to clean up stuck training jobs.

This task runs every 10 minutes and marks training jobs as FAILED if they've been
stuck in QUEUED or TRAINING status for too long without updates.

Training jobs can take hours, so we use a conservative 30-minute threshold.
"""

import logging
from datetime import datetime, timezone, timedelta
from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask
from src.models.training import Training, TrainingStatus
from src.workers.websocket_emitter import emit_training_progress

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_trainings"
)
def cleanup_stuck_trainings_task(self):
    """
    Clean up training jobs that have been stuck for more than 30 minutes.

    A training is considered stuck if:
    - Status is QUEUED or TRAINING
    - No update in the last 30 minutes (conservative threshold)
    - Either has no celery_task_id or task is not running

    Note: 30 minutes is conservative since training can take hours.
    We only clean up jobs that are truly stuck (no Celery task or inactive task).
    """
    logger.info("Running stuck training cleanup task")

    with self.get_db() as db:
        try:
            # Find potentially stuck trainings (30 minute threshold)
            stuck_threshold = datetime.now(timezone.utc) - timedelta(minutes=30)

            stuck_trainings = db.query(Training).filter(
                Training.status.in_([
                    TrainingStatus.QUEUED.value,
                    TrainingStatus.TRAINING.value
                ]),
                Training.updated_at < stuck_threshold
            ).all()

            cleaned_count = 0
            for training in stuck_trainings:
                # Check if Celery task is actually running
                task_is_running = False

                if training.celery_task_id:
                    from src.core.celery_app import get_task_status
                    task_status = get_task_status(training.celery_task_id)

                    # Consider task running if in active states
                    if task_status['state'] in ['PENDING', 'STARTED', 'RETRY']:
                        task_is_running = True
                        logger.info(
                            f"Training {training.id} has active Celery task "
                            f"{training.celery_task_id} ({task_status['state']}), skipping cleanup"
                        )

                if not task_is_running:
                    # Calculate how long it's been stuck
                    time_stuck = datetime.now(timezone.utc) - training.updated_at
                    
                    # Mark as failed
                    logger.warning(
                        f"Marking stuck training {training.id} as FAILED "
                        f"(status: {training.status}, stuck for: {time_stuck}, "
                        f"task_id: {training.celery_task_id or 'None'})"
                    )

                    training.status = TrainingStatus.FAILED.value
                    training.error_message = (
                        f"Training job stuck - no progress for {int(time_stuck.total_seconds() / 60)} minutes. "
                        "This may indicate a crashed worker or system issue. "
                        "Please check logs and try again."
                    )
                    training.completed_at = datetime.now(timezone.utc)
                    training.updated_at = datetime.now(timezone.utc)

                    db.commit()
                    cleaned_count += 1

                    # Emit WebSocket event to notify frontend
                    try:
                        emit_training_progress(
                            training_id=training.id,
                            event="failed",
                            data={
                                "training_id": training.id,
                                "status": "failed",
                                "error": training.error_message,
                                "progress": 0.0
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit WebSocket event for {training.id}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stuck training job(s)")
            else:
                logger.info("No stuck training jobs found")

            return {"cleaned": cleaned_count}

        except Exception as e:
            logger.error(f"Error in stuck training cleanup: {e}", exc_info=True)
            raise
