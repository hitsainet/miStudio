"""
Service for managing activation extraction database records.

This service handles CRUD operations for activation extraction tracking.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from sqlalchemy.orm import Session

from ..models.activation_extraction import ActivationExtraction, ExtractionStatus
from ..core.config import settings

logger = logging.getLogger(__name__)


class ExtractionDatabaseService:
    """Service for managing activation extraction database records."""

    @staticmethod
    def create_extraction(
        db: Session,
        extraction_id: str,
        model_id: str,
        dataset_id: str,
        layer_indices: List[int],
        hook_types: List[str],
        max_samples: int,
        batch_size: int,
        micro_batch_size: Optional[int] = None,
        celery_task_id: Optional[str] = None,
    ) -> ActivationExtraction:
        """
        Create a new extraction record in the database.

        Args:
            db: Database session
            extraction_id: Unique extraction ID
            model_id: Model ID
            dataset_id: Dataset ID
            layer_indices: List of layer indices
            hook_types: List of hook types
            max_samples: Maximum samples to process
            batch_size: Batch size for processing
            micro_batch_size: GPU micro-batch size for memory efficiency
            celery_task_id: Optional Celery task ID

        Returns:
            Created ActivationExtraction record
        """
        output_path = str(settings.data_dir / "activations" / extraction_id)
        metadata_path = str(Path(output_path) / "metadata.json")

        extraction = ActivationExtraction(
            id=extraction_id,
            model_id=model_id,
            dataset_id=dataset_id,
            celery_task_id=celery_task_id,
            layer_indices=layer_indices,
            hook_types=hook_types,
            max_samples=max_samples,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            status=ExtractionStatus.QUEUED,
            progress=0.0,
            samples_processed=0,
            output_path=output_path,
            metadata_path=metadata_path,
            statistics={},
            saved_files=[],
        )

        db.add(extraction)
        db.commit()
        db.refresh(extraction)

        logger.info(f"Created extraction record: {extraction_id}")
        return extraction

    @staticmethod
    def update_progress(
        db: Session,
        extraction_id: str,
        progress: float,
        status: ExtractionStatus,
        samples_processed: int,
        message: Optional[str] = None,
    ) -> Optional[ActivationExtraction]:
        """
        Update extraction progress in database.

        Args:
            db: Database session
            extraction_id: Extraction ID
            progress: Progress percentage (0-100)
            status: Current status
            samples_processed: Number of samples processed
            message: Optional status message

        Returns:
            Updated ActivationExtraction record or None if not found
        """
        extraction = db.query(ActivationExtraction).filter_by(id=extraction_id).first()

        if not extraction:
            logger.warning(f"Extraction {extraction_id} not found for progress update")
            return None

        extraction.progress = progress
        extraction.status = status
        extraction.samples_processed = samples_processed

        db.commit()
        db.refresh(extraction)

        return extraction

    @staticmethod
    def update_statistics(
        db: Session,
        extraction_id: str,
        statistics: Dict[str, Any],
        saved_files: Optional[List[str]] = None,
    ) -> Optional[ActivationExtraction]:
        """
        Update extraction statistics in database.

        Args:
            db: Database session
            extraction_id: Extraction ID
            statistics: Statistics dictionary
            saved_files: Optional list of saved file names

        Returns:
            Updated ActivationExtraction record or None if not found
        """
        extraction = db.query(ActivationExtraction).filter_by(id=extraction_id).first()

        if not extraction:
            logger.warning(f"Extraction {extraction_id} not found for statistics update")
            return None

        extraction.statistics = statistics
        if saved_files is not None:
            extraction.saved_files = saved_files

        db.commit()
        db.refresh(extraction)

        return extraction

    @staticmethod
    def mark_completed(
        db: Session,
        extraction_id: str,
        statistics: Dict[str, Any],
        saved_files: List[str],
    ) -> Optional[ActivationExtraction]:
        """
        Mark extraction as completed.

        Args:
            db: Database session
            extraction_id: Extraction ID
            statistics: Final statistics
            saved_files: List of saved file names

        Returns:
            Updated ActivationExtraction record or None if not found
        """
        extraction = db.query(ActivationExtraction).filter_by(id=extraction_id).first()

        if not extraction:
            logger.warning(f"Extraction {extraction_id} not found to mark completed")
            return None

        extraction.status = ExtractionStatus.COMPLETED
        extraction.progress = 100.0
        extraction.statistics = statistics
        extraction.saved_files = saved_files
        extraction.completed_at = datetime.now()

        db.commit()
        db.refresh(extraction)

        logger.info(f"Marked extraction {extraction_id} as completed")
        return extraction

    @staticmethod
    def mark_failed(
        db: Session,
        extraction_id: str,
        error_message: str,
    ) -> Optional[ActivationExtraction]:
        """
        Mark extraction as failed.

        Args:
            db: Database session
            extraction_id: Extraction ID
            error_message: Error message

        Returns:
            Updated ActivationExtraction record or None if not found
        """
        extraction = db.query(ActivationExtraction).filter_by(id=extraction_id).first()

        if not extraction:
            logger.warning(f"Extraction {extraction_id} not found to mark failed")
            return None

        extraction.status = ExtractionStatus.FAILED
        extraction.error_message = error_message
        extraction.completed_at = datetime.now()

        db.commit()
        db.refresh(extraction)

        logger.info(f"Marked extraction {extraction_id} as failed: {error_message}")
        return extraction

    @staticmethod
    def get_extraction(db: Session, extraction_id: str) -> Optional[ActivationExtraction]:
        """
        Get extraction by ID.

        Args:
            db: Database session
            extraction_id: Extraction ID

        Returns:
            ActivationExtraction record or None if not found
        """
        return db.query(ActivationExtraction).filter_by(id=extraction_id).first()

    @staticmethod
    def get_active_extraction_for_model(db: Session, model_id: str) -> Optional[ActivationExtraction]:
        """
        Get active extraction for a model.

        Args:
            db: Database session
            model_id: Model ID

        Returns:
            Active ActivationExtraction record or None
        """
        return (
            db.query(ActivationExtraction)
            .filter(
                ActivationExtraction.model_id == model_id,
                ActivationExtraction.status.in_([
                    ExtractionStatus.QUEUED,
                    ExtractionStatus.LOADING,
                    ExtractionStatus.EXTRACTING,
                    ExtractionStatus.SAVING,
                ])
            )
            .order_by(ActivationExtraction.created_at.desc())
            .first()
        )

    @staticmethod
    def list_extractions_for_model(
        db: Session,
        model_id: str,
        limit: int = 50,
    ) -> List[ActivationExtraction]:
        """
        List all extractions for a model.

        Args:
            db: Database session
            model_id: Model ID
            limit: Maximum number of records to return

        Returns:
            List of ActivationExtraction records
        """
        return (
            db.query(ActivationExtraction)
            .filter_by(model_id=model_id)
            .order_by(ActivationExtraction.created_at.desc())
            .limit(limit)
            .all()
        )
