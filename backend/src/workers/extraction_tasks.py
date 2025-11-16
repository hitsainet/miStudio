"""
Celery tasks for feature extraction from trained SAE models.

These tasks run asynchronously to extract and analyze interpretable features
from Sparse Autoencoders without blocking the API.
"""

import logging
from typing import Dict, Any

from src.core.celery_app import celery_app
from src.services.extraction_service import ExtractionService
from src.workers.base_task import DatabaseTask

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.extraction_tasks.extract_features",
    max_retries=0,  # No automatic retries
    autoretry_for=(),  # Explicit no auto-retry (empty tuple)
)
def extract_features_task(
    self,
    training_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Celery task for extracting features from a trained SAE model.

    This task:
    1. Fetches the training and extraction job records
    2. Updates extraction status to 'extracting'
    3. Calls ExtractionService.extract_features_for_training() for core logic
    4. Handles errors and updates status accordingly

    Args:
        training_id: ID of the training to extract features from
        config: Extraction configuration (evaluation_samples, top_k_examples)

    Returns:
        Dict with extraction statistics
    """
    logger.info(f"Starting feature extraction task for training {training_id}")
    logger.info(f"Config: {config}")

    with self.get_db() as db:
        try:
            # Pre-flight check: Verify extraction hasn't already completed
            from src.models.extraction_job import ExtractionJob, ExtractionStatus
            from sqlalchemy import desc
            from datetime import datetime, timezone, timedelta

            extraction_job = db.query(ExtractionJob).filter(
                ExtractionJob.training_id == training_id
            ).order_by(desc(ExtractionJob.created_at)).first()

            if extraction_job:
                if extraction_job.status == ExtractionStatus.COMPLETED.value:
                    logger.info(
                        f"Extraction {extraction_job.id} already completed at "
                        f"{extraction_job.completed_at}, skipping re-execution"
                    )
                    return extraction_job.statistics or {}

                if extraction_job.status == ExtractionStatus.EXTRACTING.value:
                    # Check if it's been running for too long (> 3 hours = likely stuck)
                    if extraction_job.updated_at:
                        time_since_update = datetime.now(timezone.utc) - extraction_job.updated_at
                        if time_since_update > timedelta(hours=3):
                            logger.warning(
                                f"Extraction {extraction_job.id} appears stuck "
                                f"(no update for {time_since_update}), allowing restart"
                            )
                        else:
                            logger.info(
                                f"Extraction {extraction_job.id} is already in progress "
                                f"(last update: {time_since_update} ago), skipping"
                            )
                            return {}

            extraction_service = ExtractionService(db)

            # Core extraction logic is delegated to service
            statistics = extraction_service.extract_features_for_training(training_id, config)

            logger.info(f"Feature extraction completed for training {training_id}")
            logger.info(f"Statistics: {statistics}")

            return statistics

        except Exception as e:
            logger.error(
                f"Feature extraction task failed for training {training_id}: {e}",
                exc_info=True
            )
            # Service already handles status update and WebSocket events on error
            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="delete_extraction",
    max_retries=0,
)
def delete_extraction_task(self, extraction_id: str) -> Dict[str, Any]:
    """
    Background task for deleting large extractions.

    Large extractions (>10k features) take too long for synchronous deletion
    due to CASCADE deleting hundreds of thousands of feature_activations.

    Args:
        extraction_id: ID of the extraction job to delete

    Returns:
        Dict with deletion statistics
    """
    logger.info(f"Starting background deletion for extraction {extraction_id}")

    with self.get_db() as db:
        try:
            from src.models.extraction_job import ExtractionJob, ExtractionStatus
            from src.models.feature import Feature
            from datetime import datetime, timezone, timedelta

            # Verify extraction exists
            extraction_job = db.query(ExtractionJob).filter(
                ExtractionJob.id == extraction_id
            ).first()

            if not extraction_job:
                raise ValueError(f"Extraction job {extraction_id} not found")

            # Cannot delete active extraction (unless stuck for > 5 minutes)
            if extraction_job.status in [ExtractionStatus.QUEUED, ExtractionStatus.EXTRACTING]:
                time_since_update = datetime.now(timezone.utc) - extraction_job.updated_at

                if time_since_update < timedelta(minutes=5):
                    raise ValueError(
                        f"Cannot delete active extraction job. Please wait or cancel it first."
                    )

            # Count features before deletion
            feature_count = db.query(Feature).filter(
                Feature.extraction_job_id == extraction_id
            ).count()

            logger.info(f"Deleting {feature_count} features for extraction {extraction_id}")

            # Delete features (CASCADE will automatically delete feature_activations)
            db.query(Feature).filter(
                Feature.extraction_job_id == extraction_id
            ).delete(synchronize_session=False)

            # Delete extraction job
            db.query(ExtractionJob).filter(
                ExtractionJob.id == extraction_id
            ).delete(synchronize_session=False)

            # Commit transaction
            db.commit()

            logger.info(f"Successfully deleted extraction {extraction_id} with {feature_count} features")

            return {
                "extraction_id": extraction_id,
                "feature_count": feature_count,
                "status": "deleted"
            }

        except Exception as e:
            logger.error(
                f"Background deletion failed for extraction {extraction_id}: {e}",
                exc_info=True
            )
            db.rollback()
            raise
