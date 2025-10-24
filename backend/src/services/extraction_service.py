"""
Feature extraction service for SAE trained models.

This service manages the extraction of interpretable features from trained
Sparse Autoencoders, including activation analysis, feature labeling, and
statistics calculation.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import desc, select
from collections import defaultdict
import torch
import numpy as np
from datasets import load_from_disk

from src.models.training import Training, TrainingStatus
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature, LabelSource
from src.models.feature_activation import FeatureActivation
from src.models.checkpoint import Checkpoint
from src.models.dataset import Dataset
from src.core.database import get_db
from src.workers.websocket_emitter import emit_training_progress
from src.core.config import settings
from src.utils.auto_labeling import auto_label_feature
from src.services.checkpoint_service import CheckpointService
from src.ml.sparse_autoencoder import SparseAutoencoder
from src.ml.model_loader import load_model_from_hf
from src.ml.forward_hooks import HookManager, HookType
from src.models.model import Model as ModelRecord, QuantizationFormat


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

    def __init__(self, db: Union[AsyncSession, Session]):
        """Initialize extraction service with either async or sync session."""
        self.db = db

    async def start_extraction(
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
        result = await self.db.execute(
            select(Training).where(Training.id == training_id)
        )
        training = result.scalar_one_or_none()
        if not training:
            raise ValueError(f"Training {training_id} not found")

        if training.status != TrainingStatus.COMPLETED.value:
            raise ValueError(f"Training {training_id} must be completed before extraction")

        # Check if training has at least one checkpoint
        checkpoint_result = await self.db.execute(
            select(Checkpoint)
            .where(Checkpoint.training_id == training_id)
            .order_by(Checkpoint.step.desc())
            .limit(1)
        )
        latest_checkpoint = checkpoint_result.scalar_one_or_none()
        if not latest_checkpoint:
            raise ValueError(f"Training {training_id} has no checkpoints")

        # Check for active extraction on this training
        result = await self.db.execute(
            select(ExtractionJob).where(
                ExtractionJob.training_id == training_id,
                ExtractionJob.status.in_([
                    ExtractionStatus.QUEUED,
                    ExtractionStatus.EXTRACTING
                ])
            )
        )
        active_extraction = result.scalar_one_or_none()

        if active_extraction:
            raise ValueError(
                f"Training {training_id} already has an active extraction job: {active_extraction.id}"
            )

        # Create extraction job record
        extraction_job = ExtractionJob(
            id=f"extr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{training_id[:8]}",
            training_id=training_id,
            status=ExtractionStatus.QUEUED,
            config=config,
            progress=0.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        self.db.add(extraction_job)
        await self.db.commit()
        await self.db.refresh(extraction_job)

        logger.info(
            f"Created extraction job {extraction_job.id} for training {training_id}. "
            f"Config: {config}"
        )

        # Enqueue Celery task for async extraction
        from src.workers.extraction_tasks import extract_features_task
        extract_features_task.delay(training_id, config)

        return extraction_job

    async def get_extraction_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of the most recent extraction job for a training.

        Args:
            training_id: ID of the training

        Returns:
            Dict with extraction status, progress, config, statistics, or None if no extraction
        """
        # Get most recent extraction job for this training
        result = await self.db.execute(
            select(ExtractionJob)
            .where(ExtractionJob.training_id == training_id)
            .order_by(desc(ExtractionJob.created_at))
        )
        extraction_job = result.scalar_one_or_none()

        if not extraction_job:
            return None

        # Calculate features_extracted and total_features if completed
        features_extracted = None
        total_features = None

        if extraction_job.status == ExtractionStatus.COMPLETED.value:
            from sqlalchemy import func
            result = await self.db.execute(
                select(func.count()).select_from(Feature).where(
                    Feature.extraction_job_id == extraction_job.id
                )
            )
            features_extracted = result.scalar_one()
            total_features = features_extracted
        elif extraction_job.status == ExtractionStatus.EXTRACTING.value:
            # Estimate based on progress (actual count would be in real-time update)
            result = await self.db.execute(
                select(Training).where(Training.id == training_id)
            )
            training = result.scalar_one_or_none()
            if training and extraction_job.progress:
                total_features = training.hyperparameters.get("dict_size", 16384)
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

    async def list_extractions(
        self,
        status_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        List all extraction jobs with optional filtering.

        Args:
            status_filter: Optional list of statuses to filter by
            limit: Maximum number of results to return
            offset: Number of results to skip

        Returns:
            Tuple of (list of extraction job dicts, total count)
        """
        from sqlalchemy import func

        # Build query
        query = select(ExtractionJob).order_by(desc(ExtractionJob.created_at))

        if status_filter:
            query = query.where(ExtractionJob.status.in_(status_filter))

        # Get total count
        count_query = select(func.count()).select_from(ExtractionJob)
        if status_filter:
            count_query = count_query.where(ExtractionJob.status.in_(status_filter))

        count_result = await self.db.execute(count_query)
        total = count_result.scalar_one()

        # Get paginated results
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        extraction_jobs = result.scalars().all()

        # Build response list
        extractions_list = []
        for extraction_job in extraction_jobs:
            # Calculate features_extracted and total_features
            features_extracted = None
            total_features = None

            if extraction_job.status == ExtractionStatus.COMPLETED.value:
                result = await self.db.execute(
                    select(func.count()).select_from(Feature).where(
                        Feature.extraction_job_id == extraction_job.id
                    )
                )
                features_extracted = result.scalar_one()
                total_features = features_extracted
            elif extraction_job.status == ExtractionStatus.EXTRACTING.value:
                # Estimate based on progress
                result = await self.db.execute(
                    select(Training).where(Training.id == extraction_job.training_id)
                )
                training = result.scalar_one_or_none()
                if training and extraction_job.progress:
                    total_features = training.hyperparameters.get("dict_size", 16384)
                    features_extracted = int(total_features * extraction_job.progress)

            extractions_list.append({
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
            })

        return extractions_list, total

    async def cancel_extraction(self, training_id: str) -> None:
        """
        Cancel an active extraction job for a training.

        Args:
            training_id: ID of the training

        Raises:
            ValueError: If no active extraction job found
        """
        # Get active extraction job
        result = await self.db.execute(
            select(ExtractionJob).where(
                ExtractionJob.training_id == training_id,
                ExtractionJob.status.in_([
                    ExtractionStatus.QUEUED,
                    ExtractionStatus.EXTRACTING
                ])
            )
        )
        extraction_job = result.scalar_one_or_none()

        if not extraction_job:
            raise ValueError(f"No active extraction job found for training {training_id}")

        # Revoke Celery task if it exists
        if extraction_job.task_id:
            from src.core.celery_app import celery_app
            celery_app.control.revoke(extraction_job.task_id, terminate=True)
            logger.info(f"Revoked Celery task {extraction_job.task_id}")

        # Update status to failed with cancellation message
        extraction_job.status = ExtractionStatus.FAILED
        extraction_job.error_message = "Extraction cancelled by user"
        extraction_job.completed_at = datetime.now(timezone.utc)

        await self.db.commit()
        logger.info(f"Cancelled extraction {extraction_job.id} for training {training_id}")

        # Emit WebSocket event
        emit_training_progress(
            training_id=training_id,
            event="extraction:failed",
            data={
                "extraction_id": extraction_job.id,
                "training_id": training_id,
                "error": "Extraction cancelled by user"
            }
        )

    async def delete_extraction(self, extraction_id: str) -> None:
        """
        Delete an extraction job and all associated features.

        Args:
            extraction_id: ID of the extraction job

        Raises:
            ValueError: If extraction not found or is still active
        """
        # Get extraction job
        result = await self.db.execute(
            select(ExtractionJob).where(ExtractionJob.id == extraction_id)
        )
        extraction_job = result.scalar_one_or_none()

        if not extraction_job:
            raise ValueError(f"Extraction job {extraction_id} not found")

        # Cannot delete active extraction
        if extraction_job.status in [ExtractionStatus.QUEUED, ExtractionStatus.EXTRACTING]:
            raise ValueError(
                f"Cannot delete active extraction job. Please cancel it first."
            )

        # Get feature IDs first
        from sqlalchemy import delete as sql_delete
        feature_ids_result = await self.db.execute(
            select(Feature.id).where(Feature.extraction_job_id == extraction_id)
        )
        feature_ids = [row[0] for row in feature_ids_result.fetchall()]

        # Delete feature activations first
        if feature_ids:
            await self.db.execute(
                sql_delete(FeatureActivation).where(
                    FeatureActivation.feature_id.in_(feature_ids)
                )
            )

        # Delete features
        await self.db.execute(
            sql_delete(Feature).where(Feature.extraction_job_id == extraction_id)
        )

        # Delete extraction job
        await self.db.execute(
            sql_delete(ExtractionJob).where(ExtractionJob.id == extraction_id)
        )

        await self.db.commit()
        logger.info(f"Deleted extraction job {extraction_id} and associated features")

    async def update_extraction_status(
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
        result = await self.db.execute(
            select(ExtractionJob).where(ExtractionJob.id == extraction_id)
        )
        extraction_job = result.scalar_one_or_none()

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

        await self.db.commit()

        logger.debug(
            f"Updated extraction {extraction_id}: status={status}, progress={progress}"
        )

    def update_extraction_status_sync(
        self,
        extraction_id: str,
        status: str,
        progress: Optional[float] = None,
        statistics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Synchronous version of update_extraction_status for use in Celery tasks.

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
        Core feature extraction logic (called by Celery task with sync Session).

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
        # Get extraction job for this training (sync query)
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
            self.update_extraction_status_sync(
                extraction_job.id,
                ExtractionStatus.EXTRACTING.value,
                progress=0.0
            )

            # Load training and checkpoint (sync query)
            training = self.db.query(Training).filter(Training.id == training_id).first()
            if not training:
                raise ValueError(f"Training {training_id} not found")

            # Check if training has at least one checkpoint
            checkpoint = self.db.query(Checkpoint).filter(
                Checkpoint.training_id == training_id
            ).order_by(Checkpoint.step.desc()).first()

            if not checkpoint:
                raise ValueError(f"Training {training_id} has no checkpoints")

            logger.info(f"Starting feature extraction for training {training_id}")
            logger.info(f"Config: {config}")

            # Get configuration parameters
            evaluation_samples = config.get("evaluation_samples", 10000)
            top_k_examples = config.get("top_k_examples", 100)
            latent_dim = training.hyperparameters.get("latent_dim", 16384)
            batch_size = 32  # Process 32 samples at a time for GPU efficiency

            # Task 4.5: Load SAE checkpoint
            logger.info(f"Using latest checkpoint at step {checkpoint.step}")

            logger.info(f"Loading SAE checkpoint from {checkpoint.storage_path}")

            # Initialize SAE model
            sae = SparseAutoencoder(
                hidden_dim=training.hyperparameters["hidden_dim"],
                latent_dim=latent_dim,
                l1_alpha=training.hyperparameters.get("l1_alpha", 0.001)
            )

            # Load checkpoint weights
            device = "cuda" if torch.cuda.is_available() else "cpu"
            CheckpointService.load_checkpoint(
                storage_path=checkpoint.storage_path,
                model=sae,
                device=device
            )
            sae.to(device)
            sae.eval()  # Set to evaluation mode

            logger.info(f"SAE loaded on device: {device}")

            # Task 4.6: Load dataset samples
            dataset_record = self.db.query(Dataset).filter(
                Dataset.id == training.dataset_id
            ).first()
            if not dataset_record:
                raise ValueError(f"Dataset {training.dataset_id} not found")

            logger.info(f"Loading dataset from {dataset_record.tokenized_path}")
            dataset = load_from_disk(dataset_record.tokenized_path)

            # Limit to evaluation_samples
            if len(dataset) > evaluation_samples:
                dataset = dataset.select(range(evaluation_samples))

            logger.info(f"Dataset loaded: {len(dataset)} samples")

            # Task 4.7: Get base model record for tokenizer and activation extraction
            model_record = self.db.query(ModelRecord).filter(
                ModelRecord.id == training.model_id
            ).first()
            if not model_record:
                raise ValueError(f"Model {training.model_id} not found")

            logger.info(f"Loading base model: {model_record.repo_id}")

            # Load base model for activation extraction
            base_model, tokenizer, model_config, metadata = load_model_from_hf(
                repo_id=model_record.repo_id,
                quant_format=QuantizationFormat(model_record.quantization),
                cache_dir=Path(model_record.file_path).parent if model_record.file_path else None,
                device_map=device
            )
            base_model.eval()

            # Data structures for accumulating feature activations
            # feature_activations[neuron_idx] = list of (sample_idx, max_activation, tokens, activations)
            feature_activations = defaultdict(list)
            feature_activation_counts = np.zeros(latent_dim)  # Count activations > threshold per feature

            # Process dataset in batches with real model activations
            logger.info(f"Extracting features from {len(dataset)} samples...")

            # Get extraction configuration
            layer_indices = config.get("layer_indices", [0])
            hook_types = config.get("hook_types", ["residual"])
            architecture = model_record.architecture

            # Use HookManager to extract real activations from base model
            with HookManager(base_model) as hook_manager:
                # Register hooks on the specified layers
                hook_type_enums = [HookType(ht) for ht in hook_types]
                hook_manager.register_hooks(layer_indices, hook_type_enums, architecture)

                # Get text column from dataset metadata or use default
                text_column = (dataset_record.extra_metadata or {}).get("text_column", "text")

                # Process samples in batches
                with torch.no_grad():
                    for batch_start in range(0, len(dataset), batch_size):
                        batch_end = min(batch_start + batch_size, len(dataset))
                        batch = dataset[batch_start:batch_end]

                        # Extract input_ids from batch
                        batch_input_ids = []
                        batch_texts = []

                        if isinstance(batch, dict) and text_column in batch:
                            texts = batch[text_column]
                            if not isinstance(texts, list):
                                texts = [texts]

                            # Tokenize texts
                            for text in texts:
                                encoded = tokenizer(
                                    text,
                                    max_length=config.get("max_length", 512),
                                    truncation=True,
                                    return_tensors="pt"
                                )
                                batch_input_ids.append(encoded["input_ids"][0].tolist())
                                batch_texts.append(text)

                        # Skip empty batches
                        if not batch_input_ids:
                            continue

                        # Pad sequences to same length
                        max_length = max(len(ids) for ids in batch_input_ids)
                        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                        padded_input_ids = []
                        attention_masks = []

                        for input_ids in batch_input_ids:
                            padding_length = max_length - len(input_ids)
                            padded_ids = input_ids + [pad_token_id] * padding_length
                            mask = [1] * len(input_ids) + [0] * padding_length

                            padded_input_ids.append(padded_ids)
                            attention_masks.append(mask)

                        # Convert to tensors
                        input_ids_tensor = torch.tensor(padded_input_ids, device=device)
                        attention_mask_tensor = torch.tensor(attention_masks, device=device)

                        # Run model forward pass to capture activations
                        hook_manager.reset()
                        _ = base_model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)

                        # Get captured activations from hooks
                        # hook_manager.activations is a dict: {layer_name: tensor}
                        # We take the first (and typically only) layer's activations
                        layer_name = list(hook_manager.activations.keys())[0]
                        base_model_activations = hook_manager.activations[layer_name]  # Shape: (batch_size, seq_len, hidden_dim)

                        # Task 4.8: Pass through SAE encoder to get feature activations
                        # Process each sample in the batch
                        for batch_idx in range(len(batch_input_ids)):
                            global_sample_idx = batch_start + batch_idx

                            # Get activations for this sample
                            sample_activations = base_model_activations[batch_idx]  # Shape: (seq_len, hidden_dim)

                            # Get actual sequence length (before padding)
                            actual_length = len(batch_input_ids[batch_idx])
                            sample_activations = sample_activations[:actual_length]  # Remove padding

                            # Pass through SAE encoder
                            sae_features = sae.encode(sample_activations)  # Shape: (seq_len, latent_dim)

                            # Get token strings for this sample
                            token_strings = tokenizer.convert_ids_to_tokens(batch_input_ids[batch_idx])

                            # Process each SAE neuron (feature)
                            for neuron_idx in range(latent_dim):
                                neuron_activations = sae_features[:, neuron_idx].cpu().numpy()  # Shape: (seq_len,)
                                max_activation = float(neuron_activations.max())

                                # Task 4.9: Count activations above threshold (0.01)
                                if max_activation > 0.01:
                                    feature_activation_counts[neuron_idx] += 1

                                # Task 4.11: Store top-K examples per feature
                                if max_activation > 0:  # Only store if feature activated
                                    feature_activations[neuron_idx].append({
                                        "sample_index": global_sample_idx,
                                        "max_activation": max_activation,
                                        "tokens": token_strings,
                                        "activations": neuron_activations.tolist()
                                    })

                        # Task 4.15-4.16: Update progress every 5%
                        progress = batch_end / len(dataset)
                        if int(progress * 20) > int((batch_start / len(dataset)) * 20):  # Every 5%
                            self.update_extraction_status_sync(
                                extraction_job.id,
                                ExtractionStatus.EXTRACTING.value,
                                progress=progress
                            )

                            # Emit WebSocket progress event
                            emit_training_progress(
                                training_id=training_id,
                                event="extraction:progress",
                                data={
                                    "extraction_id": extraction_job.id,
                                    "training_id": training_id,
                                    "progress": progress,
                                    "features_extracted": int(latent_dim * progress),
                                    "total_features": latent_dim
                                }
                            )

                        # Clear GPU cache between batches
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # Clean up base model from GPU memory
            logger.info("Cleaning up base model from GPU memory")
            base_model.cpu()
            del base_model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Activation extraction complete. Creating feature records...")

            # Task 4.9: Calculate activation_frequency per feature
            activation_frequencies = feature_activation_counts / len(dataset)

            # Process features and store in database
            interpretable_count = 0
            total_interpretability = 0.0
            total_activation_freq = 0.0

            for neuron_idx in range(latent_dim):
                # Task 4.11: Sort and select top-K examples
                examples = feature_activations[neuron_idx]
                examples.sort(key=lambda x: x["max_activation"], reverse=True)
                top_examples = examples[:top_k_examples]

                if not top_examples:
                    continue  # Skip features with no activations

                # Task 4.10: Calculate interpretability score
                interpretability_score = self.calculate_interpretability_score(top_examples)

                # Task 4.12: Auto-generate label
                feature_name = auto_label_feature(top_examples, neuron_idx)

                # Task 4.13: Create feature record
                feature = Feature(
                    id=f"feat_{training_id[:8]}_{neuron_idx:05d}",
                    training_id=training_id,
                    extraction_job_id=extraction_job.id,
                    neuron_index=neuron_idx,
                    name=feature_name,
                    description=None,
                    label_source=LabelSource.AUTO.value,
                    activation_frequency=float(activation_frequencies[neuron_idx]),
                    interpretability_score=interpretability_score,
                    max_activation=float(top_examples[0]["max_activation"]),
                    mean_activation=float(np.mean([ex["max_activation"] for ex in top_examples])),
                    is_favorite=False,
                    notes=None,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )

                self.db.add(feature)

                # Task 4.14: Store top-K examples in feature_activations table
                for example in top_examples:
                    activation_record = FeatureActivation(
                        feature_id=feature.id,
                        sample_index=example["sample_index"],
                        max_activation=example["max_activation"],
                        tokens=example["tokens"],
                        activations=example["activations"]
                    )
                    self.db.add(activation_record)

                # Accumulate statistics
                if interpretability_score > 0.5:
                    interpretable_count += 1
                total_interpretability += interpretability_score
                total_activation_freq += activation_frequencies[neuron_idx]

            # Commit all features and activations
            self.db.commit()

            logger.info(f"Created {latent_dim} feature records")

            # Task 4.17: Calculate final statistics
            statistics = {
                "total_features": latent_dim,
                "interpretable_count": interpretable_count,
                "avg_activation_frequency": float(total_activation_freq / latent_dim),
                "avg_interpretability": float(total_interpretability / latent_dim)
            }

            logger.info(f"Extraction statistics: {statistics}")

            # Mark as completed
            self.update_extraction_status_sync(
                extraction_job.id,
                ExtractionStatus.COMPLETED.value,
                progress=1.0,
                statistics=statistics
            )

            # Emit WebSocket completion event
            emit_training_progress(
                training_id=training_id,
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
            self.update_extraction_status_sync(
                extraction_job.id,
                ExtractionStatus.FAILED.value,
                error_message=str(e)
            )

            # Emit WebSocket failure event
            emit_training_progress(
                training_id=training_id,
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

        # Use up to top 10 examples for consistency calculation
        examples_for_consistency = top_examples[:min(10, len(top_examples))]

        # Task 5.2: Calculate consistency - similarity of activation patterns
        # For each example, normalize activations and calculate pairwise similarity
        normalized_patterns = []
        for example in examples_for_consistency:
            activations = np.array(example["activations"])
            max_act = activations.max()

            if max_act > 0:
                # Normalize to 0-1 range
                normalized = activations / max_act
                # Binarize: tokens with activation > 0.3 are "active"
                binary_pattern = (normalized > 0.3).astype(float)
                normalized_patterns.append(binary_pattern)

        if len(normalized_patterns) < 2:
            consistency = 0.0
        else:
            # Calculate pairwise cosine similarity
            similarities = []
            for i in range(len(normalized_patterns)):
                for j in range(i + 1, len(normalized_patterns)):
                    pattern_i = normalized_patterns[i]
                    pattern_j = normalized_patterns[j]

                    # Pad shorter pattern to match length
                    max_len = max(len(pattern_i), len(pattern_j))
                    if len(pattern_i) < max_len:
                        pattern_i = np.pad(pattern_i, (0, max_len - len(pattern_i)))
                    if len(pattern_j) < max_len:
                        pattern_j = np.pad(pattern_j, (0, max_len - len(pattern_j)))

                    # Cosine similarity
                    dot_product = np.dot(pattern_i, pattern_j)
                    norm_i = np.linalg.norm(pattern_i)
                    norm_j = np.linalg.norm(pattern_j)

                    if norm_i > 0 and norm_j > 0:
                        similarity = dot_product / (norm_i * norm_j)
                        similarities.append(similarity)

            # Average pairwise similarity
            consistency = float(np.mean(similarities)) if similarities else 0.0

        # Task 5.3-5.4: Calculate sparsity - ideal 10-30% of tokens activated
        sparsity_values = []
        for example in examples_for_consistency:
            activations = np.array(example["activations"])
            # Count tokens with activation > 0.01
            active_count = np.sum(activations > 0.01)
            total_count = len(activations)

            if total_count > 0:
                sparsity_fraction = active_count / total_count
                sparsity_values.append(sparsity_fraction)

        if not sparsity_values:
            sparsity_score = 0.0
        else:
            avg_sparsity = np.mean(sparsity_values)

            # Ideal sparsity: 10-30% (0.1-0.3)
            if 0.1 <= avg_sparsity <= 0.3:
                # Perfect score in ideal range
                sparsity_score = 1.0
            elif avg_sparsity < 0.1:
                # Too sparse - penalize linearly
                # 0% -> 0.0, 10% -> 1.0
                sparsity_score = avg_sparsity / 0.1
            else:
                # Too dense - penalize
                # 30% -> 1.0, 100% -> 0.0
                sparsity_score = max(0.0, (1.0 - avg_sparsity) / 0.7)

        # Task 5.5: Combine scores - consistency weighted 70%, sparsity 30%
        interpretability = (consistency * 0.7) + (sparsity_score * 0.3)

        # Task 5.6: Clamp to 0.0-1.0 range
        interpretability = max(0.0, min(1.0, interpretability))

        return interpretability


def get_extraction_service(db: Session) -> ExtractionService:
    """Dependency injection helper for ExtractionService."""
    return ExtractionService(db)
