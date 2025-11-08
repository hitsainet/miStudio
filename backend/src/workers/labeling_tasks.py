"""
Celery tasks for semantic labeling of SAE features.

These tasks run asynchronously to label features extracted from SAE models
without blocking the API. Labeling is independent from extraction, allowing
re-labeling without re-extraction.
"""

import logging
from typing import Dict, Any

from src.core.celery_app import celery_app
from src.services.labeling_service import LabelingService
from src.workers.base_task import DatabaseTask

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="label_features",
    max_retries=0,  # No automatic retries
    autoretry_for=(),  # Explicit no auto-retry (empty tuple)
)
def label_features_task(
    self,
    labeling_job_id: str
) -> Dict[str, Any]:
    """
    Celery task for labeling features from an extraction job.

    This task:
    1. Fetches the labeling job record
    2. Updates labeling status to 'labeling'
    3. Calls LabelingService.label_features_for_extraction() for core logic
    4. Handles errors and updates status accordingly

    Args:
        labeling_job_id: ID of the labeling job to execute

    Returns:
        Dict with labeling statistics
    """
    logger.info(f"Starting labeling task for job {labeling_job_id}")

    with self.get_db() as db:
        try:
            # Pre-flight check: Verify labeling hasn't already completed
            from src.models.labeling_job import LabelingJob, LabelingStatus
            from datetime import datetime, timezone, timedelta

            labeling_job = db.query(LabelingJob).filter(
                LabelingJob.id == labeling_job_id
            ).first()

            if labeling_job:
                if labeling_job.status == LabelingStatus.COMPLETED.value:
                    logger.info(
                        f"Labeling {labeling_job.id} already completed at "
                        f"{labeling_job.completed_at}, skipping re-execution"
                    )
                    return labeling_job.statistics or {}

                if labeling_job.status == LabelingStatus.LABELING.value:
                    # Check if it's been running for too long (> 2 hours = likely stuck)
                    if labeling_job.updated_at:
                        time_since_update = datetime.now(timezone.utc) - labeling_job.updated_at
                        if time_since_update > timedelta(hours=2):
                            logger.warning(
                                f"Labeling {labeling_job.id} appears stuck "
                                f"(no update for {time_since_update}), allowing restart"
                            )
                        else:
                            logger.info(
                                f"Labeling {labeling_job.id} is already in progress "
                                f"(last update: {time_since_update} ago), skipping"
                            )
                            return {}

            labeling_service = LabelingService(db)

            # Core labeling logic is delegated to service
            statistics = labeling_service.label_features_for_extraction(labeling_job_id)

            logger.info(f"Labeling completed for job {labeling_job_id}")
            logger.info(f"Statistics: {statistics}")

            return statistics

        except Exception as e:
            logger.error(
                f"Labeling task failed for job {labeling_job_id}: {e}",
                exc_info=True
            )
            # Service already handles status update on error
            raise
