"""
Feature extraction service for SAE trained models.

This service manages the extraction of interpretable features from trained
Sparse Autoencoders, including activation analysis, feature labeling, and
statistics calculation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc
import torch
import numpy as np

from src.models.training import Training, TrainingStatus
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature, LabelSource
from src.models.feature_activation import FeatureActivation
from src.core.database import get_db
from src.core.websocket import manager as ws_manager
from src.utils.auto_labeling import auto_label_feature
from src.core.celery_app import celery_app


logger = logging.getLogger(__name__)


class ExtractionService:
    """
    Service for extracting interpretable features from trained SAE models.

    Manages the feature extraction workflow:
    1. Start extraction job and validate training
    2. Extract activations from dataset samples
    3. Analyze SAE neuron activations
    4. Calculate statistics and auto-label features
    5. Store results and emit WebSocket events
    """

    def __init__(self, db: Session):
        """Initialize extraction service."""
        self.db = db

    def start_extraction(
        self,
        training_id: str,
        config: Dict[str, Any]
    ) -> ExtractionJob:
        """
        Start a feature extraction job for a completed training.

        Args:
            training_id: ID of the training to extract features from
            config: Extraction configuration (evaluation_samples, top_k_examples)

        Returns:
            ExtractionJob: Created extraction job record

        Raises:
            ValueError: If training not found, not completed, or active extraction exists
        """
        # Validate training exists and is completed
        training = self.db.query(Training).filter(Training.id == training_id).first()
        if not training:
            raise ValueError(f"Training {training_id} not found")

        if training.status != TrainingStatus.COMPLETED.value:
            raise ValueError(f"Training {training_id} must be completed before extraction")

        if not training.final_checkpoint_id:
            raise ValueError(f"Training {training_id} has no final checkpoint")

        # Check for active extraction on this training
        active_extraction = (
            self.db.query(ExtractionJob)
            .filter(
                ExtractionJob.training_id == training_id,
                ExtractionJob.status.in_([
                    ExtractionStatus.QUEUED.value,
                    ExtractionStatus.EXTRACTING.value
                ])
            )
            .first()
        )

        if active_extraction:
            raise ValueError(
                f"Training {training_id} already has an active extraction job: {active_extraction.id}"
            )

        # Create extraction job record
        extraction_job = ExtractionJob(
            id=f"extr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{training_id[:8]}",
            training_id=training_id,
            status=ExtractionStatus.QUEUED.value,
            config=config,
            progress=0.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        self.db.add(extraction_job)
        self.db.commit()
        self.db.refresh(extraction_job)

        logger.info(
            f"Created extraction job {extraction_job.id} for training {training_id}. "
            f"Config: {config}"
        )

        # Enqueue Celery task for async extraction
        from src.workers.extraction_tasks import extract_features_task
        extract_features_task.delay(training_id, config)

        return extraction_job

    def get_extraction_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of the most recent extraction job for a training.

        Args:
            training_id: ID of the training

        Returns:
            Dict with extraction status, progress, config, statistics, or None if no extraction
        """
        # Get most recent extraction job for this training
        extraction_job = (
            self.db.query(ExtractionJob)
            .filter(ExtractionJob.training_id == training_id)
            .order_by(desc(ExtractionJob.created_at))
            .first()
        )

        if not extraction_job:
            return None

        # Calculate features_extracted and total_features if completed
        features_extracted = None
        total_features = None

        if extraction_job.status == ExtractionStatus.COMPLETED.value:
            features_extracted = self.db.query(Feature).filter(
                Feature.extraction_job_id == extraction_job.id
            ).count()
            total_features = features_extracted
        elif extraction_job.status == ExtractionStatus.EXTRACTING.value:
            # Estimate based on progress (actual count would be in real-time update)
            training = self.db.query(Training).filter(Training.id == training_id).first()
            if training and extraction_job.progress:
                total_features = training.config.get("dict_size", 16384)
                features_extracted = int(total_features * extraction_job.progress)

        return {
            "id": extraction_job.id,
            "training_id": extraction_job.training_id,
            "status": extraction_job.status,
            "progress": extraction_job.progress,
            "features_extracted": features_extracted,
            "total_features": total_features,
            "error_message": extraction_job.error_message,
            "config": extraction_job.config,
            "statistics": extraction_job.statistics,
            "created_at": extraction_job.created_at,
            "updated_at": extraction_job.updated_at,
            "completed_at": extraction_job.completed_at
        }

    def update_extraction_status(
        self,
        extraction_id: str,
        status: str,
        progress: Optional[float] = None,
        statistics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update extraction job status and progress.

        Args:
            extraction_id: ID of the extraction job
            status: New status (queued, extracting, completed, failed, cancelled)
            progress: Progress percentage (0.0-1.0)
            statistics: Extraction statistics (on completion)
            error_message: Error message (if failed)
        """
        extraction_job = self.db.query(ExtractionJob).filter(
            ExtractionJob.id == extraction_id
        ).first()

        if not extraction_job:
            logger.error(f"Extraction job {extraction_id} not found")
            return

        extraction_job.status = status
        extraction_job.updated_at = datetime.now(timezone.utc)

        if progress is not None:
            extraction_job.progress = progress

        if statistics is not None:
            extraction_job.statistics = statistics

        if error_message is not None:
            extraction_job.error_message = error_message

        if status == ExtractionStatus.COMPLETED.value:
            extraction_job.completed_at = datetime.now(timezone.utc)

        self.db.commit()

        logger.debug(
            f"Updated extraction {extraction_id}: status={status}, progress={progress}"
        )

    def extract_features_for_training(
        self,
        training_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Core feature extraction logic (called by Celery task).

        This method:
        1. Loads SAE checkpoint and dataset
        2. Extracts activations for evaluation samples
        3. Analyzes SAE neuron activations
        4. Auto-labels features and calculates statistics
        5. Stores results in database

        Args:
            training_id: ID of the training
            config: Extraction configuration

        Returns:
            Dict with extraction statistics

        Raises:
            Exception: If extraction fails at any step
        """
        # Get extraction job for this training
        extraction_job = (
            self.db.query(ExtractionJob)
            .filter(ExtractionJob.training_id == training_id)
            .order_by(desc(ExtractionJob.created_at))
            .first()
        )

        if not extraction_job:
            raise ValueError(f"No extraction job found for training {training_id}")

        try:
            # Update status to extracting
            self.update_extraction_status(
                extraction_job.id,
                ExtractionStatus.EXTRACTING.value,
                progress=0.0
            )

            # Load training and checkpoint
            training = self.db.query(Training).filter(Training.id == training_id).first()
            if not training or not training.final_checkpoint_id:
                raise ValueError(f"Training {training_id} not found or has no final checkpoint")

            # TODO: Phase 4 Tasks 4.5-4.20 will be implemented next
            # This placeholder shows the extraction workflow structure

            logger.info(f"Starting feature extraction for training {training_id}")

            # Placeholder statistics
            statistics = {
                "total_features": 0,
                "interpretable_count": 0,
                "avg_activation_frequency": 0.0,
                "avg_interpretability": 0.0
            }

            # Mark as completed
            self.update_extraction_status(
                extraction_job.id,
                ExtractionStatus.COMPLETED.value,
                progress=1.0,
                statistics=statistics
            )

            # Emit WebSocket completion event
            ws_manager.emit(
                room=f"training:{training_id}",
                event="extraction:completed",
                data={
                    "extraction_id": extraction_job.id,
                    "training_id": training_id,
                    "statistics": statistics
                }
            )

            return statistics

        except Exception as e:
            logger.error(f"Feature extraction failed for training {training_id}: {e}", exc_info=True)

            # Update status to failed
            self.update_extraction_status(
                extraction_job.id,
                ExtractionStatus.FAILED.value,
                error_message=str(e)
            )

            # Emit WebSocket failure event
            ws_manager.emit(
                room=f"training:{training_id}",
                event="extraction:failed",
                data={
                    "extraction_id": extraction_job.id,
                    "training_id": training_id,
                    "error": str(e)
                }
            )

            raise

    def calculate_interpretability_score(
        self,
        top_examples: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate interpretability score for a feature based on activation patterns.

        Combines consistency (similarity across examples) and sparsity (selectivity):
        - Consistency: measure similarity of activation patterns across top examples
        - Sparsity: ideal 10-30% of tokens activated, penalize extremes
        - Final score: (consistency * 0.7) + (sparsity_score * 0.3)

        Args:
            top_examples: List of max-activating examples with tokens and activations

        Returns:
            Float score between 0.0 and 1.0 (higher = more interpretable)
        """
        if not top_examples or len(top_examples) < 2:
            return 0.0

        # TODO: Phase 5 Tasks 5.2-5.6 will implement full calculation
        # Placeholder returns 0.5 for now
        return 0.5


def get_extraction_service(db: Session) -> ExtractionService:
    """Dependency injection helper for ExtractionService."""
    return ExtractionService(db)
