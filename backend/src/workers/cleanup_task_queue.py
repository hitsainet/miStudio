"""
Periodic task to clean up old task_queue entries.

This task runs hourly and:
- Deletes completed entries older than 7 days
- Deletes stale queued/running entries older than 7 days (ghost entries whose
  worker never reported back, e.g. from a crashed worker before the
  success-path completion fix)

Without this, the task_queue table grows without bound.
"""

import logging
from datetime import datetime, timezone, timedelta

from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask
from src.models.task_queue import TaskQueue

logger = logging.getLogger(__name__)

COMPLETED_RETENTION_DAYS = 7
STALE_ACTIVE_RETENTION_DAYS = 7


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_task_queue"
)
def cleanup_task_queue_task(self):
    """
    Delete old completed and stale active task_queue entries.

    Failed entries are kept indefinitely — they back the Failed Operations
    retry UI and are removed explicitly by the user.
    """
    logger.info("Running task_queue cleanup task")

    with self.get_db() as db:
        try:
            now = datetime.now(timezone.utc)
            completed_cutoff = now - timedelta(days=COMPLETED_RETENTION_DAYS)
            stale_cutoff = now - timedelta(days=STALE_ACTIVE_RETENTION_DAYS)

            completed_deleted = (
                db.query(TaskQueue)
                .filter(
                    TaskQueue.status == "completed",
                    TaskQueue.completed_at < completed_cutoff,
                )
                .delete(synchronize_session=False)
            )

            stale_deleted = (
                db.query(TaskQueue)
                .filter(
                    TaskQueue.status.in_(["queued", "running"]),
                    TaskQueue.created_at < stale_cutoff,
                )
                .delete(synchronize_session=False)
            )

            db.commit()

            if completed_deleted or stale_deleted:
                logger.info(
                    f"task_queue cleanup: deleted {completed_deleted} completed, "
                    f"{stale_deleted} stale active entries"
                )

            return {
                "completed_deleted": completed_deleted,
                "stale_deleted": stale_deleted,
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error in task_queue cleanup: {e}", exc_info=True)
            raise
