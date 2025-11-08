"""
Feature labeling service for semantic labeling of SAE features.

This service manages semantic labeling of features extracted from SAE models.
Labeling is independent from extraction, allowing re-labeling without re-extraction.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import desc, select
from collections import defaultdict
import asyncio

from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.labeling_job import LabelingJob, LabelingStatus, LabelingMethod
from src.models.feature import Feature, LabelSource
from src.models.feature_activation import FeatureActivation
from src.core.config import settings
from src.services.local_labeling_service import LocalLabelingService
from src.services.openai_labeling_service import OpenAILabelingService

logger = logging.getLogger(__name__)


class LabelingService:
    """
    Service for semantic labeling of SAE features.

    Manages the feature labeling workflow:
    1. Create labeling job for an extraction
    2. Fetch features and their activations
    3. Aggregate token statistics for each feature
    4. Generate semantic labels using OpenAI or local LLM
    5. Update feature names and track labeling job
    6. Emit WebSocket progress events
    """

    def __init__(self, db: Union[AsyncSession, Session]):
        """Initialize labeling service with either async or sync session."""
        self.db = db
        self.is_async = isinstance(db, AsyncSession)

    async def start_labeling(
        self,
        extraction_job_id: str,
        config: Dict[str, Any]
    ) -> LabelingJob:
        """
        Start a feature labeling job for a completed extraction.

        Args:
            extraction_job_id: ID of the extraction to label features from
            config: Labeling configuration (labeling_method, openai_model, etc.)

        Returns:
            LabelingJob: Created labeling job record

        Raises:
            ValueError: If extraction not found, not completed, or active labeling exists
        """
        from sqlalchemy import func

        # Validate extraction exists and is completed
        result = await self.db.execute(
            select(ExtractionJob).where(ExtractionJob.id == extraction_job_id)
        )
        extraction_job = result.scalar_one_or_none()

        if not extraction_job:
            raise ValueError(f"Extraction job {extraction_job_id} not found")

        if extraction_job.status != ExtractionStatus.COMPLETED.value:
            raise ValueError(
                f"Extraction {extraction_job_id} must be completed before labeling "
                f"(current status: {extraction_job.status})"
            )

        # Check for active labeling on this extraction
        result = await self.db.execute(
            select(LabelingJob).where(
                LabelingJob.extraction_job_id == extraction_job_id,
                LabelingJob.status.in_([
                    LabelingStatus.QUEUED.value,
                    LabelingStatus.LABELING.value
                ])
            )
        )
        active_labeling = result.scalar_one_or_none()

        if active_labeling:
            raise ValueError(
                f"Extraction {extraction_job_id} already has an active labeling job: "
                f"{active_labeling.id}"
            )

        # Count features to label
        count_result = await self.db.execute(
            select(func.count()).select_from(Feature).where(
                Feature.extraction_job_id == extraction_job_id
            )
        )
        total_features = count_result.scalar_one()

        if total_features == 0:
            raise ValueError(f"Extraction {extraction_job_id} has no features to label")

        # Create labeling job ID: label_{extraction_id}_{timestamp}
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = f"label_{extraction_job_id}_{timestamp}"

        # Create labeling job record
        labeling_job = LabelingJob(
            id=job_id,
            extraction_job_id=extraction_job_id,
            labeling_method=config.get("labeling_method", "openai"),
            openai_model=config.get("openai_model"),
            openai_api_key=config.get("openai_api_key"),
            local_model=config.get("local_model"),
            status=LabelingStatus.QUEUED.value,
            progress=0.0,
            features_labeled=0,
            total_features=total_features
        )

        self.db.add(labeling_job)
        await self.db.commit()
        await self.db.refresh(labeling_job)

        logger.info(
            f"Created labeling job {job_id} for extraction {extraction_job_id} "
            f"with {total_features} features using method: {labeling_job.labeling_method}"
        )

        return labeling_job

    def label_features_for_extraction(
        self,
        labeling_job_id: str
    ) -> Dict[str, Any]:
        """
        Execute semantic labeling for features from an extraction job.

        This is the core labeling logic that:
        1. Fetches features and their activations
        2. Aggregates token statistics for each feature
        3. Generates semantic labels using specified method
        4. Updates feature names and tracks progress
        5. Calculates statistics and marks job complete

        Args:
            labeling_job_id: ID of the labeling job to execute

        Returns:
            Dict with labeling statistics

        Raises:
            ValueError: If labeling job not found or extraction invalid
        """
        # Fetch labeling job
        labeling_job = self.db.query(LabelingJob).filter(
            LabelingJob.id == labeling_job_id
        ).first()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        # Update status to labeling
        labeling_job.status = LabelingStatus.LABELING.value
        labeling_job.updated_at = datetime.now(timezone.utc)
        self.db.commit()

        start_time = datetime.now(timezone.utc)

        try:
            # Fetch extraction job
            extraction_job = self.db.query(ExtractionJob).filter(
                ExtractionJob.id == labeling_job.extraction_job_id
            ).first()

            if not extraction_job:
                raise ValueError(f"Extraction job {labeling_job.extraction_job_id} not found")

            # Fetch all features for this extraction
            features = self.db.query(Feature).filter(
                Feature.extraction_job_id == labeling_job.extraction_job_id
            ).order_by(Feature.neuron_index).all()

            if not features:
                raise ValueError(f"No features found for extraction {labeling_job.extraction_job_id}")

            logger.info(f"Labeling {len(features)} features for extraction {labeling_job.extraction_job_id}")

            # Aggregate token statistics for each feature
            features_token_stats = []
            neuron_indices = []

            for feature in features:
                # Fetch activations for this feature
                activations = self.db.query(FeatureActivation).filter(
                    FeatureActivation.feature_id == feature.id
                ).all()

                # Aggregate token statistics
                token_stats = defaultdict(lambda: {
                    "count": 0,
                    "total_activation": 0.0,
                    "max_activation": 0.0
                })

                for activation in activations:
                    token = activation.token_str
                    act = activation.activation_value
                    token_stats[token]["count"] += 1
                    token_stats[token]["total_activation"] += act
                    token_stats[token]["max_activation"] = max(
                        token_stats[token]["max_activation"], act
                    )

                features_token_stats.append(dict(token_stats))
                neuron_indices.append(feature.neuron_index)

            logger.info(f"Aggregation complete for {len(features)} features")

            # Initialize appropriate labeling service
            labeling_method = labeling_job.labeling_method
            labels = []

            try:
                if labeling_method == LabelingMethod.LOCAL.value:
                    local_model = labeling_job.local_model or "meta-llama/Llama-3.2-1B"
                    logger.info(f"Initializing local labeling service with model: {local_model}")
                    labeling_service = LocalLabelingService(model_name=local_model)

                    # Generate labels (load/unload handled internally)
                    labels = labeling_service.batch_generate_labels(
                        features_token_stats=features_token_stats,
                        neuron_indices=neuron_indices,
                        progress_callback=None  # Could add WebSocket progress here
                    )
                    label_source_value = LabelSource.LLM.value

                elif labeling_method == LabelingMethod.OPENAI.value:
                    # Get API key from labeling job, fallback to settings if invalid/missing
                    openai_api_key = labeling_job.openai_api_key

                    # Validate API key - if None, empty, or looks corrupted, use settings
                    if not openai_api_key or len(openai_api_key) > 200 or any(ord(c) > 127 for c in openai_api_key):
                        logger.warning(f"Invalid/missing OpenAI API key in labeling job, using settings.openai_api_key")
                        openai_api_key = getattr(settings, 'openai_api_key', None)

                    if not openai_api_key:
                        raise ValueError("OpenAI API key not provided and not found in settings")

                    openai_model = labeling_job.openai_model or "gpt-4o-mini"
                    logger.info(f"Initializing OpenAI labeling service with model: {openai_model}")
                    labeling_service = OpenAILabelingService(
                        api_key=openai_api_key,
                        model=openai_model
                    )

                    # Generate labels asynchronously
                    labels = asyncio.run(labeling_service.batch_generate_labels(
                        features_token_stats=features_token_stats,
                        neuron_indices=neuron_indices,
                        progress_callback=None,
                        batch_size=10
                    ))
                    label_source_value = LabelSource.LLM.value

                else:
                    raise ValueError(f"Unsupported labeling method: {labeling_method}")

                # Update feature names, label_source, and labeling_job_id
                labeled_at = datetime.now(timezone.utc)
                for feature, label in zip(features, labels):
                    feature.name = label
                    feature.label_source = label_source_value
                    feature.labeling_job_id = labeling_job.id
                    feature.labeled_at = labeled_at
                    feature.updated_at = labeled_at

                self.db.commit()
                logger.info(f"Successfully labeled {len(labels)} features using {labeling_method}")

                # Calculate statistics
                end_time = datetime.now(timezone.utc)
                duration_seconds = (end_time - start_time).total_seconds()

                successfully_labeled = len([l for l in labels if l and not l.startswith("feature_")])
                failed_labels = len(labels) - successfully_labeled
                avg_label_length = sum(len(l) for l in labels) / len(labels) if labels else 0

                statistics = {
                    "total_features": len(features),
                    "successfully_labeled": successfully_labeled,
                    "failed_labels": failed_labels,
                    "avg_label_length": round(avg_label_length, 2),
                    "labeling_duration_seconds": round(duration_seconds, 2),
                    "labeling_method": labeling_method
                }

                # Mark labeling job as completed
                labeling_job.status = LabelingStatus.COMPLETED.value
                labeling_job.progress = 1.0
                labeling_job.features_labeled = len(labels)
                labeling_job.completed_at = end_time
                labeling_job.updated_at = end_time
                labeling_job.statistics = statistics
                self.db.commit()

                logger.info(f"Labeling job {labeling_job_id} completed successfully")

                return statistics

            except Exception as e:
                logger.error(f"Batch labeling failed: {e}", exc_info=True)
                raise

        except Exception as e:
            logger.error(f"Feature labeling failed for job {labeling_job_id}: {e}", exc_info=True)

            # Mark labeling job as failed
            labeling_job.status = LabelingStatus.FAILED.value
            labeling_job.error_message = str(e)
            labeling_job.updated_at = datetime.now(timezone.utc)
            self.db.commit()

            raise

    async def get_labeling_job(self, labeling_job_id: str) -> Optional[LabelingJob]:
        """
        Get a labeling job by ID.

        Args:
            labeling_job_id: ID of the labeling job

        Returns:
            LabelingJob or None if not found
        """
        result = await self.db.execute(
            select(LabelingJob).where(LabelingJob.id == labeling_job_id)
        )
        return result.scalar_one_or_none()

    async def list_labeling_jobs(
        self,
        extraction_job_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[List[LabelingJob], int]:
        """
        List labeling jobs with optional filtering.

        Args:
            extraction_job_id: Optional filter by extraction job ID
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            Tuple of (list of labeling jobs, total count)
        """
        from sqlalchemy import func

        # Build query
        query = select(LabelingJob).order_by(desc(LabelingJob.created_at))

        if extraction_job_id:
            query = query.where(LabelingJob.extraction_job_id == extraction_job_id)

        # Get total count
        count_query = select(func.count()).select_from(LabelingJob)
        if extraction_job_id:
            count_query = count_query.where(LabelingJob.extraction_job_id == extraction_job_id)

        count_result = await self.db.execute(count_query)
        total = count_result.scalar_one()

        # Get paginated results
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        jobs = result.scalars().all()

        return list(jobs), total

    async def cancel_labeling_job(self, labeling_job_id: str) -> bool:
        """
        Cancel a labeling job.

        Args:
            labeling_job_id: ID of the labeling job to cancel

        Returns:
            True if cancelled successfully

        Raises:
            ValueError: If job not found or not in cancellable state
        """
        result = await self.db.execute(
            select(LabelingJob).where(LabelingJob.id == labeling_job_id)
        )
        labeling_job = result.scalar_one_or_none()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        if labeling_job.status not in [LabelingStatus.QUEUED.value, LabelingStatus.LABELING.value]:
            raise ValueError(
                f"Cannot cancel labeling job {labeling_job_id} with status {labeling_job.status}"
            )

        labeling_job.status = LabelingStatus.CANCELLED.value
        labeling_job.updated_at = datetime.now(timezone.utc)
        await self.db.commit()

        logger.info(f"Cancelled labeling job {labeling_job_id}")
        return True

    async def delete_labeling_job(self, labeling_job_id: str) -> bool:
        """
        Delete a labeling job.

        This does NOT delete the features or their labels, only the labeling job record.
        Feature labels will remain intact.

        Args:
            labeling_job_id: ID of the labeling job to delete

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If job not found or in active state
        """
        from sqlalchemy import update

        result = await self.db.execute(
            select(LabelingJob).where(LabelingJob.id == labeling_job_id)
        )
        labeling_job = result.scalar_one_or_none()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        if labeling_job.status in [LabelingStatus.QUEUED.value, LabelingStatus.LABELING.value]:
            raise ValueError(
                f"Cannot delete active labeling job {labeling_job_id}. "
                f"Cancel it first or wait for completion."
            )

        # Clear labeling_job_id reference from features
        await self.db.execute(
            update(Feature).where(
                Feature.labeling_job_id == labeling_job_id
            ).values(labeling_job_id=None)
        )

        # Delete labeling job
        await self.db.delete(labeling_job)
        await self.db.commit()

        logger.info(f"Deleted labeling job {labeling_job_id}")
        return True
