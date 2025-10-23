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


@celery_app.task(bind=True, base=DatabaseTask, name="extract_features")
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
