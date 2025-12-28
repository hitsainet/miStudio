"""
Celery tasks for feature extraction from trained SAE models.

These tasks run asynchronously to extract and analyze interpretable features
from Sparse Autoencoders without blocking the API.
"""

import logging
from typing import Dict, Any

from src.core.celery_app import celery_app
from src.core.config import settings
from src.services.extraction_service import ExtractionService
from src.workers.base_task import DatabaseTask
from src.workers.websocket_emitter import emit_extraction_deleted

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
    name="src.workers.extraction_tasks.extract_features_from_sae",
    max_retries=0,
    autoretry_for=(),
)
def extract_features_from_sae_task(
    self,
    sae_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Celery task for extracting features from an external SAE.

    This task:
    1. Loads the external SAE from local_path
    2. Loads the associated model and dataset
    3. Extracts features through the SAE
    4. Stores results in database

    Args:
        sae_id: ID of the external SAE
        config: Extraction configuration (dataset_id, evaluation_samples, top_k_examples)

    Returns:
        Dict with extraction statistics
    """
    logger.info(f"Starting feature extraction task for external SAE {sae_id}")
    logger.info(f"Config: {config}")

    with self.get_db() as db:
        try:
            from src.models.extraction_job import ExtractionJob, ExtractionStatus
            from src.models.external_sae import ExternalSAE
            from sqlalchemy import desc

            # Get extraction job for this SAE
            extraction_job = db.query(ExtractionJob).filter(
                ExtractionJob.external_sae_id == sae_id
            ).order_by(desc(ExtractionJob.created_at)).first()

            if not extraction_job:
                raise ValueError(f"No extraction job found for SAE {sae_id}")

            # Emit starting progress
            from src.workers.websocket_emitter import emit_sae_extraction_progress
            emit_sae_extraction_progress(
                sae_id=sae_id,
                extraction_id=extraction_job.id,
                progress=0.0,
                status="starting",
                message="Starting feature extraction..."
            )

            # Idempotency check
            if extraction_job.status == ExtractionStatus.COMPLETED.value:
                logger.warning(f"Extraction {extraction_job.id} already completed")
                return extraction_job.statistics or {}

            if extraction_job.status == ExtractionStatus.FAILED.value:
                logger.warning(f"Extraction {extraction_job.id} previously failed")
                return {}

            # Get external SAE record
            external_sae = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
            if not external_sae:
                raise ValueError(f"External SAE {sae_id} not found")

            # Validate SAE local path exists
            from pathlib import Path
            if not external_sae.local_path:
                extraction_job.status = ExtractionStatus.FAILED.value
                extraction_job.error_message = f"External SAE {sae_id} has no local path"
                db.commit()
                raise ValueError(f"External SAE {sae_id} has no local path")

            sae_path = settings.resolve_data_path(external_sae.local_path)
            if not sae_path.exists():
                extraction_job.status = ExtractionStatus.FAILED.value
                extraction_job.error_message = f"SAE local path does not exist: {external_sae.local_path}"
                db.commit()
                raise ValueError(f"SAE local path does not exist: {external_sae.local_path}")

            logger.info(f"SAE path validated: {sae_path}")

            # Emit extracting progress before starting
            emit_sae_extraction_progress(
                sae_id=sae_id,
                extraction_id=extraction_job.id,
                progress=5.0,
                status="extracting",
                message="Loading SAE and starting feature extraction..."
            )

            # Delegate to service
            extraction_service = ExtractionService(db)
            statistics = extraction_service.extract_features_for_sae(sae_id, config)

            logger.info(f"Feature extraction completed for SAE {sae_id}")
            logger.info(f"Statistics: {statistics}")

            # Emit completion progress
            emit_sae_extraction_progress(
                sae_id=sae_id,
                extraction_id=extraction_job.id,
                progress=100.0,
                status="completed",
                message="Feature extraction completed successfully",
                features_extracted=statistics.get("total_features"),
                total_features=statistics.get("total_features")
            )

            return statistics

        except Exception as e:
            logger.error(
                f"Feature extraction task failed for SAE {sae_id}: {e}",
                exc_info=True
            )
            # Emit failure progress
            try:
                emit_sae_extraction_progress(
                    sae_id=sae_id,
                    extraction_id=extraction_job.id if extraction_job else "unknown",
                    progress=0.0,
                    status="failed",
                    message=f"Extraction failed: {str(e)}"
                )
            except Exception:
                pass  # Best effort emission on failure
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

            # Emit WebSocket event to notify frontend
            emit_extraction_deleted(extraction_id, feature_count)

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
