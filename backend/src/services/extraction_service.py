"""
Feature extraction service for SAE trained models.

This service manages the extraction of interpretable features from trained
Sparse Autoencoders, including activation analysis, feature labeling, and
statistics calculation.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import desc, select
from collections import defaultdict
import heapq
import torch
import numpy as np
from datasets import load_from_disk

from src.models.training import Training, TrainingStatus
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature, LabelSource
from src.models.feature_activation import FeatureActivation
from src.models.checkpoint import Checkpoint
from src.models.dataset import Dataset
from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from src.models.external_sae import ExternalSAE, SAEStatus
from src.core.database import get_db
from src.workers.websocket_emitter import emit_training_progress, emit_extraction_job_progress, emit_extraction_deleted
from src.core.config import settings
from src.services.resource_config import ResourceConfig
from src.utils.auto_labeling import auto_label_feature
from src.services.checkpoint_service import CheckpointService
# Labeling services removed - labeling is now a separate independent process
# from src.services.local_labeling_service import LocalLabelingService
# from src.services.openai_labeling_service import OpenAILabelingService
from src.ml.sparse_autoencoder import SparseAutoencoder, create_sae
from src.ml.community_format import load_sae_auto_detect
from src.ml.model_loader import load_model_from_hf
from src.ml.forward_hooks import HookManager, HookType
from src.models.model import Model as ModelRecord, QuantizationFormat


logger = logging.getLogger(__name__)


def cleanup_gpu_memory(
    models_to_cleanup: Optional[List[Any]] = None,
    context: str = "unknown"
) -> None:
    """
    Aggressively clean up GPU memory.

    This function ensures GPU memory is released even if normal cleanup fails.
    Should be called in finally blocks to guarantee cleanup on success or failure.

    Args:
        models_to_cleanup: Optional list of model objects to delete (will be moved to CPU first)
        context: Description of calling context for logging
    """
    if not torch.cuda.is_available():
        return

    try:
        initial_memory = torch.cuda.memory_allocated(0) / (1024**3)
        logger.info(f"[GPU Cleanup - {context}] Starting cleanup. Memory before: {initial_memory:.2f}GB")

        # Step 1: Move models to CPU and delete them
        if models_to_cleanup:
            for i, model in enumerate(models_to_cleanup):
                if model is not None:
                    try:
                        # Try to move to CPU first (releases GPU tensors)
                        if hasattr(model, 'cpu'):
                            model.cpu()
                        # Delete all parameters to release GPU memory
                        if hasattr(model, 'parameters'):
                            for param in model.parameters():
                                param.data = torch.empty(0)
                                if param.grad is not None:
                                    param.grad = None
                        # Delete all buffers
                        if hasattr(model, 'buffers'):
                            for buffer in model.buffers():
                                buffer.data = torch.empty(0)
                    except Exception as e:
                        logger.warning(f"[GPU Cleanup - {context}] Error cleaning model {i}: {e}")

        # Step 2: Force Python garbage collection (multiple rounds for circular refs)
        for _ in range(3):
            gc.collect()

        # Step 3: Clean up CUDA IPC memory (shared memory between processes)
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()

        # Step 4: Synchronize CUDA to ensure all operations complete
        torch.cuda.synchronize()

        # Step 5: Empty CUDA cache
        torch.cuda.empty_cache()

        # Step 6: Another round of garbage collection after cache clear
        gc.collect()

        # Step 7: Reset memory stats for cleaner tracking
        torch.cuda.reset_peak_memory_stats()

        final_memory = torch.cuda.memory_allocated(0) / (1024**3)
        freed = initial_memory - final_memory
        logger.info(
            f"[GPU Cleanup - {context}] Cleanup complete. "
            f"Memory after: {final_memory:.2f}GB, Freed: {freed:.2f}GB"
        )

        # Log warning if significant memory still allocated
        if final_memory > 0.5:  # More than 500MB still allocated
            logger.warning(
                f"[GPU Cleanup - {context}] Warning: {final_memory:.2f}GB still allocated after cleanup. "
                f"This may indicate a memory leak or uncollected references."
            )

    except Exception as e:
        logger.error(f"[GPU Cleanup - {context}] Error during cleanup: {e}")
        # Still try basic cleanup even if above failed
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_token_with_marker(tokenizer, token_id: int) -> str:
    """
    Get token string preserving BPE markers (Ġ, ▁, ##).

    Uses convert_ids_to_tokens() which preserves markers, with fallback
    handling for byte-level tokens like <0x0A>.

    This is important for proper word reconstruction in the frontend.
    The markers indicate word boundaries:
    - Ġ (U+0120): GPT-2/LLaMA word start marker
    - ▁ (U+2581): SentencePiece/T5 word start marker
    - ##: BERT continuation marker

    Args:
        tokenizer: HuggingFace tokenizer instance
        token_id: Token ID to convert

    Returns:
        Token string with BPE markers preserved
    """
    raw_token = tokenizer.convert_ids_to_tokens(token_id)

    # Handle None or empty
    if not raw_token:
        return ""

    # Handle byte tokens like <0x0A> (newline), <0x20> (space)
    # These are used by some tokenizers for control characters
    if isinstance(raw_token, str) and raw_token.startswith('<0x') and raw_token.endswith('>'):
        try:
            byte_val = int(raw_token[3:-1], 16)
            # Convert printable ASCII to actual char, keep others as-is
            if 32 <= byte_val < 127:
                return chr(byte_val)
            # Keep special bytes (newline, tab, etc.) as marker for visibility
            return raw_token
        except ValueError:
            return raw_token

    return raw_token


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

    async def _check_active_extraction(
        self,
        training_id: Optional[str] = None,
        sae_id: Optional[str] = None
    ) -> None:
        """
        Check if there's an active extraction for the given source.

        This method prevents duplicate extractions by verifying no extraction
        job is currently queued or running for the same source.

        Args:
            training_id: Training ID to check (mutually exclusive with sae_id)
            sae_id: External SAE ID to check (mutually exclusive with training_id)

        Raises:
            ValueError: If must specify either training_id or sae_id
            ValueError: If active extraction exists for the source
        """
        if not training_id and not sae_id:
            raise ValueError("Must specify either training_id or sae_id")
        if training_id and sae_id:
            raise ValueError("Cannot specify both training_id and sae_id")

        # Build query for active extraction
        query = select(ExtractionJob).where(
            ExtractionJob.status.in_([
                ExtractionStatus.QUEUED,
                ExtractionStatus.EXTRACTING
            ])
        )

        if training_id:
            query = query.where(ExtractionJob.training_id == training_id)
            source_type = "training"
            source_id = training_id
        else:
            query = query.where(ExtractionJob.external_sae_id == sae_id)
            source_type = "SAE"
            source_id = sae_id

        query = query.order_by(desc(ExtractionJob.created_at)).limit(1)
        result = await self.db.execute(query)
        active_extraction = result.scalar_one_or_none()

        if active_extraction:
            from src.core.celery_app import get_task_status
            from datetime import timedelta

            if active_extraction.celery_task_id:
                task_status = get_task_status(active_extraction.celery_task_id)

                # Check if task is genuinely active
                if task_status['state'] in ['PENDING', 'STARTED', 'RETRY']:
                    raise ValueError(
                        f"{source_type} {source_id} already has an active extraction job: "
                        f"{active_extraction.id} (Celery task: {active_extraction.celery_task_id}, "
                        f"state: {task_status['state']})"
                    )
                elif task_status['state'] in ['SUCCESS', 'FAILURE', 'REVOKED']:
                    # Task finished but DB not updated - check staleness
                    time_since_update = datetime.now(timezone.utc) - active_extraction.updated_at
                    if time_since_update < timedelta(minutes=5):
                        # Recent activity, task may still be committing results
                        raise ValueError(
                            f"{source_type} {source_id} has a recently completed extraction task "
                            f"that may still be finalizing: {active_extraction.id}"
                        )
                    else:
                        # Stale - allow new extraction but log warning
                        logger.warning(
                            f"Found stale extraction {active_extraction.id} with finished "
                            f"Celery task (state: {task_status['state']}), allowing new extraction"
                        )
                else:
                    # Unknown state
                    logger.warning(
                        f"Extraction {active_extraction.id} has Celery task in unknown state: "
                        f"{task_status['state']}, allowing new extraction"
                    )
            else:
                # Legacy job without task_id - check staleness by timestamp
                time_since_update = datetime.now(timezone.utc) - active_extraction.updated_at
                if time_since_update < timedelta(hours=3):
                    raise ValueError(
                        f"{source_type} {source_id} already has an active extraction job: "
                        f"{active_extraction.id} (last updated {time_since_update} ago)"
                    )
                else:
                    logger.warning(
                        f"Found stale extraction {active_extraction.id} (no task_id, "
                        f"last update {time_since_update} ago), allowing new extraction"
                    )

    async def start_extraction_for_sae(
        self,
        sae_id: str,
        config: Dict[str, Any]
    ) -> ExtractionJob:
        """
        Start a feature extraction job for an external SAE.

        Args:
            sae_id: ID of the external SAE to extract features from
            config: Extraction configuration (evaluation_samples, top_k_examples, dataset_id)

        Returns:
            ExtractionJob: Created extraction job record

        Raises:
            ValueError: If SAE not found, not ready, or active extraction exists
        """
        # Validate SAE exists and is ready
        result = await self.db.execute(
            select(ExternalSAE).where(ExternalSAE.id == sae_id)
        )
        external_sae = result.scalar_one_or_none()
        if not external_sae:
            raise ValueError(f"External SAE {sae_id} not found")

        if external_sae.status != SAEStatus.READY.value:
            raise ValueError(f"External SAE {sae_id} must be ready before extraction (status: {external_sae.status})")

        if not external_sae.local_path:
            raise ValueError(f"External SAE {sae_id} has no local path")

        # Validate dataset_id is provided
        dataset_id = config.get("dataset_id")
        if not dataset_id:
            raise ValueError("dataset_id is required for external SAE extraction")

        # Validate dataset exists and is ready
        dataset_result = await self.db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = dataset_result.scalar_one_or_none()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        if dataset.status != 'ready':
            raise ValueError(f"Dataset {dataset_id} is not ready (status: {dataset.status})")

        # Check for active extraction on this SAE (uses shared method)
        await self._check_active_extraction(sae_id=sae_id)

        # Create extraction job record with external_sae_id (no training_id)
        extraction_job = ExtractionJob(
            id=f"extr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_sae_{sae_id[:8]}",
            external_sae_id=sae_id,
            training_id=None,  # No training - this is an external SAE
            status=ExtractionStatus.QUEUED,
            config=config,
            filter_special=config.get('filter_special', True),
            filter_single_char=config.get('filter_single_char', True),
            filter_punctuation=config.get('filter_punctuation', True),
            filter_numbers=config.get('filter_numbers', True),
            filter_fragments=config.get('filter_fragments', True),
            filter_stop_words=config.get('filter_stop_words', False),
            context_prefix_tokens=config.get('context_prefix_tokens', 25),
            context_suffix_tokens=config.get('context_suffix_tokens', 25),
            progress=0.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        self.db.add(extraction_job)
        await self.db.commit()
        await self.db.refresh(extraction_job)

        logger.info(
            f"Created extraction job {extraction_job.id} for external SAE {sae_id}. "
            f"Config: {config}"
        )

        # Enqueue Celery task for async extraction
        from src.workers.extraction_tasks import extract_features_from_sae_task

        soft_time_limit = config.get("soft_time_limit", 144000)
        time_limit = config.get("time_limit", 172800)

        task_result = extract_features_from_sae_task.apply_async(
            args=(sae_id, config),
            soft_time_limit=soft_time_limit,
            time_limit=time_limit
        )

        extraction_job.celery_task_id = task_result.id
        await self.db.commit()
        await self.db.refresh(extraction_job)

        logger.info(f"Queued SAE extraction task {task_result.id} for job {extraction_job.id}")

        return extraction_job

    async def start_batch_extraction_for_saes(
        self,
        sae_ids: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Start feature extraction jobs for multiple SAEs in a single batch.

        Creates extraction jobs for all specified SAEs using the same dataset
        and configuration. Jobs are queued and processed sequentially.

        Args:
            sae_ids: List of external SAE IDs to extract features from
            config: Extraction configuration (dataset_id, evaluation_samples, top_k_examples, etc.)

        Returns:
            Dict containing:
            - batch_id: Unique identifier for this batch
            - created_jobs: List of created job info (sae_id, sae_name, job_id, position, status)
            - skipped_saes: List of skipped SAE info (sae_id, reason)
            - total_requested: Total SAEs requested
            - total_created: Number of jobs created
            - total_skipped: Number of SAEs skipped
        """
        import uuid
        from src.workers.extraction_tasks import extract_features_from_sae_task

        # Generate unique batch ID
        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        created_jobs: List[Dict[str, Any]] = []
        skipped_saes: List[Dict[str, Any]] = []
        total_saes = len(sae_ids)

        logger.info(f"Starting batch extraction {batch_id} for {total_saes} SAEs")

        # Process each SAE
        for position, sae_id in enumerate(sae_ids, start=1):
            try:
                # Validate SAE exists and is ready
                result = await self.db.execute(
                    select(ExternalSAE).where(ExternalSAE.id == sae_id)
                )
                external_sae = result.scalar_one_or_none()

                if not external_sae:
                    skipped_saes.append({
                        "sae_id": sae_id,
                        "reason": f"SAE not found"
                    })
                    continue

                if external_sae.status != SAEStatus.READY.value:
                    skipped_saes.append({
                        "sae_id": sae_id,
                        "reason": f"SAE not ready (status: {external_sae.status})"
                    })
                    continue

                if not external_sae.local_path:
                    skipped_saes.append({
                        "sae_id": sae_id,
                        "reason": "SAE has no local path"
                    })
                    continue

                # Check for active extraction (catch ValueError instead of raising)
                try:
                    await self._check_active_extraction(sae_id=sae_id)
                except ValueError as e:
                    skipped_saes.append({
                        "sae_id": sae_id,
                        "reason": str(e)
                    })
                    continue

                # Create extraction job with batch metadata
                extraction_job = ExtractionJob(
                    id=f"extr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_sae_{sae_id[:8]}_{position:03d}",
                    external_sae_id=sae_id,
                    training_id=None,
                    status=ExtractionStatus.QUEUED,
                    config=config,
                    filter_special=config.get('filter_special', True),
                    filter_single_char=config.get('filter_single_char', True),
                    filter_punctuation=config.get('filter_punctuation', True),
                    filter_numbers=config.get('filter_numbers', True),
                    filter_fragments=config.get('filter_fragments', True),
                    filter_stop_words=config.get('filter_stop_words', False),
                    context_prefix_tokens=config.get('context_prefix_tokens', 25),
                    context_suffix_tokens=config.get('context_suffix_tokens', 25),
                    progress=0.0,
                    # Batch metadata
                    batch_id=batch_id,
                    batch_position=position,
                    batch_total=total_saes,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )

                self.db.add(extraction_job)
                await self.db.commit()
                await self.db.refresh(extraction_job)

                # Queue Celery task
                soft_time_limit = config.get("soft_time_limit", 144000)
                time_limit = config.get("time_limit", 172800)

                task_result = extract_features_from_sae_task.apply_async(
                    args=(sae_id, config),
                    soft_time_limit=soft_time_limit,
                    time_limit=time_limit
                )

                extraction_job.celery_task_id = task_result.id
                await self.db.commit()

                created_jobs.append({
                    "sae_id": sae_id,
                    "sae_name": external_sae.name,
                    "job_id": extraction_job.id,
                    "position": position,
                    "status": "queued"
                })

                logger.info(
                    f"Batch {batch_id}: Created job {extraction_job.id} for SAE {sae_id} "
                    f"(position {position}/{total_saes})"
                )

            except Exception as e:
                logger.error(f"Batch {batch_id}: Error creating job for SAE {sae_id}: {e}")
                skipped_saes.append({
                    "sae_id": sae_id,
                    "reason": f"Error creating job: {str(e)}"
                })

        logger.info(
            f"Batch {batch_id} complete: {len(created_jobs)} jobs created, "
            f"{len(skipped_saes)} SAEs skipped"
        )

        return {
            "batch_id": batch_id,
            "created_jobs": created_jobs,
            "skipped_saes": skipped_saes,
            "total_requested": total_saes,
            "total_created": len(created_jobs),
            "total_skipped": len(skipped_saes)
        }

    async def get_extraction_status_for_sae(self, sae_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of the most recent extraction job for an external SAE.

        Args:
            sae_id: ID of the external SAE

        Returns:
            Dict with extraction status or None if no extraction
        """
        result = await self.db.execute(
            select(ExtractionJob)
            .where(ExtractionJob.external_sae_id == sae_id)
            .order_by(desc(ExtractionJob.created_at))
            .limit(1)
        )
        extraction_job = result.scalar_one_or_none()

        if not extraction_job:
            return None

        # Get SAE info
        sae_result = await self.db.execute(
            select(ExternalSAE).where(ExternalSAE.id == sae_id)
        )
        external_sae = sae_result.scalar_one_or_none()

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
            if external_sae and extraction_job.progress:
                total_features = external_sae.n_features or 16384
                features_extracted = int(total_features * extraction_job.progress)

        return {
            "id": extraction_job.id,
            "external_sae_id": extraction_job.external_sae_id,
            "training_id": None,
            "source_type": "external_sae",
            "sae_name": external_sae.name if external_sae else None,
            "model_name": external_sae.model_name if external_sae else None,
            "status": extraction_job.status,
            "progress": extraction_job.progress,
            "features_extracted": features_extracted,
            "total_features": total_features,
            "error_message": extraction_job.error_message,
            "config": extraction_job.config,
            "statistics": extraction_job.statistics,
            "created_at": extraction_job.created_at,
            "updated_at": extraction_job.updated_at,
            "completed_at": extraction_job.completed_at,
            "filter_special": extraction_job.filter_special,
            "filter_single_char": extraction_job.filter_single_char,
            "filter_punctuation": extraction_job.filter_punctuation,
            "filter_numbers": extraction_job.filter_numbers,
            "filter_fragments": extraction_job.filter_fragments,
            "filter_stop_words": extraction_job.filter_stop_words,
            "context_prefix_tokens": extraction_job.context_prefix_tokens,
            "context_suffix_tokens": extraction_job.context_suffix_tokens
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
            # Determine source type
            source_type = "external_sae" if extraction_job.external_sae_id else "training"
            model_name = None
            dataset_name = None
            sae_name = None

            if extraction_job.training_id:
                # Training-based extraction
                result = await self.db.execute(
                    select(Training).where(Training.id == extraction_job.training_id)
                )
                training = result.scalar_one_or_none()

                if training and training.model_id:
                    model_result = await self.db.execute(
                        select(ModelRecord).where(ModelRecord.id == training.model_id)
                    )
                    model = model_result.scalar_one_or_none()
                    if model:
                        model_name = model.name

                if training and training.dataset_id:
                    dataset_result = await self.db.execute(
                        select(Dataset).where(Dataset.id == training.dataset_id)
                    )
                    dataset = dataset_result.scalar_one_or_none()
                    if dataset:
                        dataset_name = dataset.name

            elif extraction_job.external_sae_id:
                # External SAE-based extraction
                sae_result = await self.db.execute(
                    select(ExternalSAE).where(ExternalSAE.id == extraction_job.external_sae_id)
                )
                external_sae = sae_result.scalar_one_or_none()

                if external_sae:
                    sae_name = external_sae.name
                    model_name = external_sae.model_name

                    # Try to get dataset name from config
                    dataset_id = extraction_job.config.get("dataset_id")
                    if dataset_id:
                        dataset_result = await self.db.execute(
                            select(Dataset).where(Dataset.id == dataset_id)
                        )
                        dataset = dataset_result.scalar_one_or_none()
                        if dataset:
                            dataset_name = dataset.name

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
                if extraction_job.training_id:
                    training = training if 'training' in dir() else None
                    if training and extraction_job.progress:
                        total_features = training.hyperparameters.get("latent_dim", 16384)
                        features_extracted = int(total_features * extraction_job.progress)
                elif extraction_job.external_sae_id:
                    external_sae = external_sae if 'external_sae' in dir() else None
                    if external_sae and extraction_job.progress:
                        total_features = external_sae.n_features or 16384
                        features_extracted = int(total_features * extraction_job.progress)

            extractions_list.append({
                "id": extraction_job.id,
                "training_id": extraction_job.training_id,
                "external_sae_id": extraction_job.external_sae_id,
                "source_type": source_type,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "sae_name": sae_name,
                "layer_index": extraction_job.layer_index,  # Layer index for multi-layer trainings
                "hook_type": extraction_job.hook_type,  # Hook type for multi-hook trainings
                "status": extraction_job.status,
                "progress": extraction_job.progress,
                "features_extracted": features_extracted,
                "total_features": total_features,
                "error_message": extraction_job.error_message,
                "config": extraction_job.config,
                "statistics": extraction_job.statistics,
                "created_at": extraction_job.created_at,
                "updated_at": extraction_job.updated_at,
                "completed_at": extraction_job.completed_at,
                # Token filtering configuration (matches labeling filter structure)
                "filter_special": extraction_job.filter_special,
                "filter_single_char": extraction_job.filter_single_char,
                "filter_punctuation": extraction_job.filter_punctuation,
                "filter_numbers": extraction_job.filter_numbers,
                "filter_fragments": extraction_job.filter_fragments,
                "filter_stop_words": extraction_job.filter_stop_words,
                # Context window configuration
                "context_prefix_tokens": extraction_job.context_prefix_tokens,
                "context_suffix_tokens": extraction_job.context_suffix_tokens,
                # NLP Processing status
                "nlp_status": extraction_job.nlp_status,
                "nlp_progress": extraction_job.nlp_progress,
                "nlp_processed_count": extraction_job.nlp_processed_count,
                "nlp_error_message": extraction_job.nlp_error_message
            })

        return extractions_list, total

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

        # Cannot delete active extraction (unless it's been stuck for > 5 minutes)
        if extraction_job.status in [ExtractionStatus.QUEUED, ExtractionStatus.EXTRACTING]:
            # Check if extraction has been stuck (not updated in over 5 minutes)
            from datetime import datetime, timezone, timedelta
            time_since_update = datetime.now(timezone.utc) - extraction_job.updated_at

            # Allow deletion if stuck for > 5 minutes (likely crashed/stuck)
            if time_since_update < timedelta(minutes=5):
                raise ValueError(
                    f"Cannot delete active extraction job. Please wait or cancel it first."
                )

        # Count features before deletion for WebSocket emit
        from sqlalchemy import delete as sql_delete, func

        feature_count_result = await self.db.execute(
            select(func.count(Feature.id)).where(Feature.extraction_job_id == extraction_id)
        )
        feature_count = feature_count_result.scalar() or 0

        # Delete features (CASCADE will automatically delete feature_activations)
        # No need to manually delete activations - foreign key has ON DELETE CASCADE
        await self.db.execute(
            sql_delete(Feature).where(Feature.extraction_job_id == extraction_id)
        )

        # Delete extraction job
        await self.db.execute(
            sql_delete(ExtractionJob).where(ExtractionJob.id == extraction_id)
        )

        await self.db.commit()
        logger.info(f"Deleted extraction job {extraction_id} and {feature_count} associated features")

        # Emit WebSocket notification for frontend
        emit_extraction_deleted(extraction_id, feature_count)

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
        features_extracted: Optional[int] = None,
        statistics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Synchronous version of update_extraction_status for use in Celery tasks.

        Args:
            extraction_id: ID of the extraction job
            status: New status (queued, extracting, completed, failed, cancelled)
            progress: Progress percentage (0.0-1.0)
            features_extracted: Number of features extracted so far
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

        if features_extracted is not None:
            extraction_job.features_extracted = features_extracted

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

        # Emit WebSocket progress update
        try:
            from ..workers.websocket_emitter import emit_progress

            # Prepare event data
            event_data = {
                "extraction_id": extraction_id,
                "status": status,
                "progress": progress if progress is not None else extraction_job.progress,
                "features_extracted": extraction_job.features_extracted,
                "total_features": extraction_job.total_features,
            }

            # Emit appropriate event based on status
            if status == ExtractionStatus.COMPLETED.value:
                # Check if auto_nlp is enabled BEFORE emitting completion event
                # This allows frontend to know NLP will start and subscribe to updates
                auto_nlp = extraction_job.config.get("auto_nlp", True) if extraction_job.config else True
                if auto_nlp:
                    # Set nlp_status to 'pending' so frontend knows NLP is about to start
                    extraction_job.nlp_status = "pending"
                    extraction_job.nlp_progress = 0.0
                    self.db.commit()  # Sync version - no await
                    event_data["nlp_status"] = "pending"
                    event_data["nlp_progress"] = 0.0

                emit_progress(
                    channel=f"extraction/{extraction_id}",
                    event="extraction:completed",
                    data=event_data
                )

                # Queue NLP analysis task if auto_nlp is enabled
                if auto_nlp:
                    try:
                        from ..workers.nlp_analysis_tasks import analyze_features_nlp_task
                        analyze_features_nlp_task.delay(
                            extraction_job_id=extraction_id,
                            feature_ids=None,  # Analyze all features
                            batch_size=100
                        )
                        logger.info(f"Queued NLP analysis task for extraction {extraction_id} (auto_nlp=True)")
                    except Exception as nlp_err:
                        logger.warning(f"Failed to queue NLP analysis for extraction {extraction_id}: {nlp_err}")
                else:
                    logger.info(f"Skipping NLP analysis for extraction {extraction_id} (auto_nlp=False)")

            elif status == ExtractionStatus.FAILED.value:
                event_data["error_message"] = error_message
                emit_progress(
                    channel=f"extraction/{extraction_id}",
                    event="extraction:failed",
                    data=event_data
                )
            else:
                emit_progress(
                    channel=f"extraction/{extraction_id}",
                    event="extraction:progress",
                    data=event_data
                )
        except Exception as e:
            logger.warning(f"Failed to emit WebSocket update for extraction {extraction_id}: {e}")

    def extract_features_for_sae(
        self,
        sae_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Core feature extraction logic for external SAEs (called by Celery task with sync Session).

        This method loads an external SAE and extracts features using a specified dataset.

        Args:
            sae_id: ID of the external SAE
            config: Extraction configuration including dataset_id

        Returns:
            Dict with extraction statistics

        Raises:
            Exception: If extraction fails at any step
        """
        from src.workers.websocket_emitter import emit_progress

        # Get extraction job for this SAE (sync query)
        extraction_job = (
            self.db.query(ExtractionJob)
            .filter(ExtractionJob.external_sae_id == sae_id)
            .order_by(desc(ExtractionJob.created_at))
            .first()
        )

        if not extraction_job:
            raise ValueError(f"No extraction job found for SAE {sae_id}")

        # Idempotency check
        if extraction_job.status == ExtractionStatus.COMPLETED.value:
            logger.warning(f"Extraction {extraction_job.id} already completed")
            return extraction_job.statistics or {}

        if extraction_job.status == ExtractionStatus.FAILED.value:
            logger.warning(f"Extraction {extraction_job.id} previously failed")
            return {}

        # Initialize model references for cleanup in finally block
        # These MUST be defined before try block to ensure finally can access them
        base_model = None
        sae = None
        tokenizer = None
        incremental_heap = None

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                initial_memory = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"GPU memory before extraction: {initial_memory:.2f}GB allocated")

            # Update status to extracting
            self.update_extraction_status_sync(
                extraction_job.id,
                ExtractionStatus.EXTRACTING.value,
                progress=0.0
            )

            # Load external SAE record
            external_sae = self.db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
            if not external_sae:
                raise ValueError(f"External SAE {sae_id} not found")

            if not external_sae.local_path:
                raise ValueError(f"External SAE {sae_id} has no local path")

            logger.info(f"Starting feature extraction for external SAE {sae_id}")
            logger.info(f"SAE: {external_sae.name} (layer {external_sae.layer})")
            logger.info(f"Config: {config}")

            # Get configuration parameters
            evaluation_samples = config.get("evaluation_samples", 10000)
            start_sample = config.get("start_sample", 0)
            top_k_examples = config.get("top_k_examples", 100)
            dataset_id = config.get("dataset_id")

            if not dataset_id:
                raise ValueError("dataset_id is required for external SAE extraction")

            # Load SAE using auto-detect (supports community, gemma_scope, mistudio formats)
            resolved_sae_path = settings.resolve_data_path(external_sae.local_path)
            logger.info(f"Loading SAE from {resolved_sae_path}")
            sae_state_dict, sae_config, format_type = load_sae_auto_detect(
                resolved_sae_path,
                device=device
            )
            logger.info(f"SAE format detected: {format_type}")

            # Determine SAE dimensions
            latent_dim = external_sae.n_features or sae_config.d_sae if sae_config else 16384
            hidden_dim = external_sae.d_model or sae_config.d_in if sae_config else 2304

            # Create SAE model with appropriate architecture
            architecture_type = external_sae.architecture or "standard"
            if sae_config and sae_config.architecture:
                architecture_type = sae_config.architecture

            logger.info(f"Creating {architecture_type} SAE: hidden_dim={hidden_dim}, latent_dim={latent_dim}")

            # Create SAE using factory
            sae = create_sae(
                architecture_type=architecture_type,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                l1_alpha=sae_config.l1_coefficient if sae_config else 0.001,
                initial_threshold=sae_config.threshold if (sae_config and hasattr(sae_config, 'threshold')) else None,
            )

            # Load weights into SAE
            # Convert community format keys to miStudio format if needed
            weight_keys = list(sae_state_dict.keys())
            if any(k.startswith("model.") for k in weight_keys):
                # Already has model. prefix
                sae.load_state_dict({k.replace("model.", ""): v for k, v in sae_state_dict.items()})
            else:
                sae.load_state_dict(sae_state_dict)

            sae.to(device)
            sae.eval()
            logger.info(f"SAE loaded on device: {device}")

            # Find model record
            model_record = None
            if external_sae.model_id:
                model_record = self.db.query(ModelRecord).filter(
                    ModelRecord.id == external_sae.model_id
                ).first()

            if not model_record and external_sae.model_name:
                # Try to find by name
                model_record = self.db.query(ModelRecord).filter(
                    ModelRecord.name.ilike(f"%{external_sae.model_name}%")
                ).first()

            if not model_record:
                raise ValueError(
                    f"No model found for SAE. model_id={external_sae.model_id}, "
                    f"model_name={external_sae.model_name}"
                )

            # Load dataset and tokenization
            dataset_record = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset_record:
                raise ValueError(f"Dataset {dataset_id} not found")

            # Find tokenization for this dataset + model
            tokenization = self.db.query(DatasetTokenization).filter(
                DatasetTokenization.dataset_id == dataset_id,
                DatasetTokenization.model_id == model_record.id
            ).first()

            if not tokenization or tokenization.status != TokenizationStatus.READY:
                raise ValueError(
                    f"No ready tokenization found for dataset {dataset_id} with model {model_record.id}. "
                    f"Please tokenize the dataset first."
                )

            # Resolve Docker-style /data/ paths for native mode
            resolved_dataset_path = settings.resolve_data_path(tokenization.tokenized_path)
            logger.info(f"Loading dataset from {resolved_dataset_path}")
            dataset = load_from_disk(str(resolved_dataset_path))

            # Select sample range
            dataset_size = len(dataset)
            end_sample = min(start_sample + evaluation_samples, dataset_size)
            if start_sample > 0 or end_sample < dataset_size:
                dataset = dataset.select(range(start_sample, end_sample))
                logger.info(f"Selected samples [{start_sample}:{end_sample}]")

            logger.info(f"Dataset loaded: {len(dataset)} samples")

            # Load base model
            logger.info(f"Loading base model: {model_record.repo_id}")
            resolved_model_path = settings.resolve_data_path(model_record.file_path) if model_record.file_path else None
            model_is_downloaded = resolved_model_path and resolved_model_path.exists()
            base_model, tokenizer, model_config, metadata = load_model_from_hf(
                repo_id=model_record.repo_id,
                quant_format=QuantizationFormat(model_record.quantization),
                cache_dir=resolved_model_path,
                device_map=device,
                local_files_only=model_is_downloaded,
            )
            base_model.eval()

            # Get resource configuration
            from src.services.resource_config import ResourceConfig
            recommended_settings = ResourceConfig.get_optimal_settings(
                training_config={"hidden_dim": hidden_dim, "latent_dim": latent_dim},
                extraction_config=config,
            )
            batch_size = config.get("batch_size") or recommended_settings["batch_size"]
            db_commit_batch = config.get("db_commit_batch") or recommended_settings["db_commit_batch"]

            # Import vectorized extraction utilities
            from src.services.extraction_vectorized import (
                IncrementalTopKHeap,
                batch_process_features,
                get_vectorization_config
            )

            # Get layer index for hooks
            layer_index = external_sae.layer or 0
            hook_types = ["residual"]
            architecture = model_record.architecture

            # Context window configuration
            context_prefix_tokens = extraction_job.context_prefix_tokens
            context_suffix_tokens = extraction_job.context_suffix_tokens
            min_activation_frequency = config.get("min_activation_frequency", 0.001)

            # Initialize data structures
            incremental_heap = IncrementalTopKHeap(
                num_features=latent_dim,
                top_k=top_k_examples
            )
            feature_activation_counts = np.zeros(latent_dim)

            # Process dataset with hooks
            with HookManager(base_model) as hook_manager:
                hook_type_enums = [HookType(ht) for ht in hook_types]
                hook_manager.register_hooks([layer_index], hook_type_enums, architecture)

                text_column = (dataset_record.extra_metadata or {}).get("text_column", "text")

                with torch.no_grad():
                    for batch_start in range(0, len(dataset), batch_size):
                        batch_end = min(batch_start + batch_size, len(dataset))
                        batch = dataset[batch_start:batch_end]

                        # Process batch (simplified version of training extraction)
                        batch_input_ids = []
                        batch_texts = []

                        if isinstance(batch, dict) and "input_ids" in batch:
                            input_ids = batch["input_ids"]
                            if not isinstance(input_ids, list):
                                input_ids = [input_ids]
                            for ids in input_ids:
                                if isinstance(ids, list):
                                    batch_input_ids.append(ids)
                                else:
                                    batch_input_ids.append(ids.tolist() if hasattr(ids, 'tolist') else list(ids))
                                try:
                                    # Use decode() for proper Unicode handling across all tokenizer types
                                    decoded_text = tokenizer.decode(batch_input_ids[-1], skip_special_tokens=False)
                                    batch_texts.append(decoded_text)
                                except:
                                    batch_texts.append(f"[Tokens: {len(batch_input_ids[-1])}]")

                        if not batch_input_ids:
                            continue

                        # Pad sequences
                        max_length = max(len(ids) for ids in batch_input_ids)
                        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

                        padded_input_ids = []
                        attention_masks = []
                        for input_ids in batch_input_ids:
                            padding_length = max_length - len(input_ids)
                            padded_ids = input_ids + [pad_token_id] * padding_length
                            mask = [1] * len(input_ids) + [0] * padding_length
                            padded_input_ids.append(padded_ids)
                            attention_masks.append(mask)

                        input_ids_tensor = torch.tensor(padded_input_ids, device=device, dtype=torch.long)
                        attention_mask_tensor = torch.tensor(attention_masks, device=device, dtype=torch.long)

                        # Run model forward pass
                        hook_manager.clear_activations()
                        _ = base_model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)

                        if not hook_manager.activations:
                            raise ValueError("Hook did not capture activations")

                        layer_name = list(hook_manager.activations.keys())[0]
                        base_model_activations = hook_manager.activations[layer_name][0]

                        # Process through SAE encoder
                        batch_sae_features = []
                        batch_token_strings = []

                        for batch_idx in range(len(batch_input_ids)):
                            sample_activations = base_model_activations[batch_idx]
                            actual_length = len(batch_input_ids[batch_idx])
                            sample_activations = sample_activations[:actual_length]

                            sae_features = sae.encode(
                                sample_activations.to(device=device, dtype=torch.float32)
                            ).detach()
                            batch_sae_features.append(sae_features)

                            # Get token strings preserving BPE markers for word reconstruction
                            # Uses get_token_with_marker() which preserves Ġ/▁/## markers while handling byte tokens
                            token_strings = [get_token_with_marker(tokenizer, tid) for tid in batch_input_ids[batch_idx]]
                            batch_token_strings.append(token_strings)

                        if batch_sae_features:
                            max_seq_len = max(f.shape[0] for f in batch_sae_features)
                            padded_features = []
                            for features in batch_sae_features:
                                if features.shape[0] < max_seq_len:
                                    padding = torch.zeros(
                                        (max_seq_len - features.shape[0], features.shape[1]),
                                        device=features.device,
                                        dtype=features.dtype
                                    )
                                    features = torch.cat([features, padding], dim=0)
                                padded_features.append(features)

                            batch_sae_features_tensor = torch.stack(padded_features, dim=0)

                            vectorization_batch_size = get_vectorization_config(
                                config=config,
                                available_vram_gb=torch.cuda.mem_get_info()[0] / (1024**3) if torch.cuda.is_available() else None,
                                latent_dim=latent_dim,
                                seq_len=max_seq_len
                            )

                            feature_indices, max_activations, examples = batch_process_features(
                                batch_sae_features=batch_sae_features_tensor,
                                token_strings_batch=batch_token_strings,
                                sample_indices=list(range(batch_start, batch_end)),
                                vectorization_batch_size=vectorization_batch_size,
                                top_k=5,
                                filter_special=extraction_job.filter_special,
                                filter_single_char=extraction_job.filter_single_char,
                                filter_punctuation=extraction_job.filter_punctuation,
                                filter_numbers=extraction_job.filter_numbers,
                                filter_fragments=extraction_job.filter_fragments,
                                filter_stop_words=extraction_job.filter_stop_words,
                                context_prefix_tokens=context_prefix_tokens,
                                context_suffix_tokens=context_suffix_tokens
                            )

                            for feat_idx in feature_indices:
                                feature_activation_counts[feat_idx] += 1

                            incremental_heap.add_batch(feature_indices, max_activations, examples)

                            del batch_sae_features_tensor
                            del padded_features
                            del batch_sae_features

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        # Update progress
                        progress = batch_end / len(dataset)
                        if int(progress * 20) > int((batch_start / len(dataset)) * 20):
                            self.update_extraction_status_sync(
                                extraction_job.id,
                                ExtractionStatus.EXTRACTING.value,
                                progress=progress
                            )
                            emit_progress(
                                channel=f"extraction/{extraction_job.id}",
                                event="extraction:progress",
                                data={
                                    "extraction_id": extraction_job.id,
                                    "sae_id": sae_id,
                                    "progress": progress,
                                    "features_extracted": int(latent_dim * progress),
                                    "total_features": latent_dim
                                }
                            )

            # Cleanup base model
            base_model.cpu()
            base_model = None  # Clear reference instead of del to avoid UnboundLocalError in finally
            tokenizer = None  # Clear reference instead of del to avoid UnboundLocalError in finally
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            logger.info("Building final feature records...")

            # Get final heaps
            final_heaps = incremental_heap.get_heaps()
            feature_activations = defaultdict(list)
            heap_counter = 0

            for feat_idx, examples in final_heaps.items():
                for activation, example_dict in examples:
                    feature_activations[feat_idx].append((activation, heap_counter, example_dict))
                    heap_counter += 1

            # Calculate activation frequencies
            activation_frequencies = feature_activation_counts / len(dataset)

            # Process features
            interpretable_count = 0
            total_interpretability = 0.0
            total_activation_freq = 0.0
            features_processed = 0
            dead_neurons_filtered = 0

            extraction_timestamp = extraction_job.id[5:20]

            for neuron_idx in range(latent_dim):
                heap_items = feature_activations[neuron_idx]
                if not heap_items:
                    dead_neurons_filtered += 1
                    continue

                neuron_activation_freq = activation_frequencies[neuron_idx]
                if neuron_activation_freq < min_activation_frequency:
                    dead_neurons_filtered += 1
                    continue

                top_examples = [example for (activation, counter, example) in
                               sorted(heap_items, key=lambda x: x[0], reverse=True)]

                interpretability_score = self.calculate_interpretability_score(top_examples)

                feature_name = f"feature_{neuron_idx:05d}"

                # Create feature with external_sae_id (no training_id)
                feature = Feature(
                    id=f"feat_sae_{extraction_timestamp}_{neuron_idx:05d}",
                    external_sae_id=sae_id,
                    training_id=None,  # External SAE - no training
                    extraction_job_id=extraction_job.id,
                    neuron_index=neuron_idx,
                    name=feature_name,
                    description=None,
                    label_source=LabelSource.AUTO.value,
                    activation_frequency=float(activation_frequencies[neuron_idx]),
                    interpretability_score=float(interpretability_score),
                    max_activation=float(top_examples[0]["max_activation"]),
                    mean_activation=float(np.mean([ex["max_activation"] for ex in top_examples])),
                    is_favorite=False,
                    notes=None,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )

                self.db.add(feature)

                for example in top_examples:
                    if "prefix_tokens" in example and "prime_token" in example:
                        activation_record = FeatureActivation(
                            feature_id=feature.id,
                            sample_index=example["sample_index"],
                            max_activation=example["max_activation"],
                            tokens=example["tokens"],
                            activations=example["activations"],
                            prefix_tokens=example["prefix_tokens"],
                            prime_token=example["prime_token"],
                            suffix_tokens=example["suffix_tokens"],
                            prime_activation_index=example["prime_activation_index"]
                        )
                    else:
                        activation_record = FeatureActivation(
                            feature_id=feature.id,
                            sample_index=example["sample_index"],
                            max_activation=example["max_activation"],
                            tokens=example["tokens"],
                            activations=example["activations"]
                        )
                    self.db.add(activation_record)

                if interpretability_score > 0.5:
                    interpretable_count += 1
                total_interpretability += interpretability_score
                total_activation_freq += activation_frequencies[neuron_idx]
                features_processed += 1

                if features_processed % db_commit_batch == 0:
                    self.db.commit()
                    logger.info(f"Committed batch: {features_processed}/{latent_dim} features")
                    self.update_extraction_status_sync(
                        extraction_job.id,
                        ExtractionStatus.EXTRACTING.value,
                        features_extracted=features_processed
                    )

            self.db.commit()

            live_neurons = latent_dim - dead_neurons_filtered
            logger.info(f"Created {features_processed} features ({dead_neurons_filtered} dead neurons filtered)")

            # Final statistics
            statistics = {
                "total_neurons": latent_dim,
                "live_neurons": live_neurons,
                "dead_neurons_filtered": dead_neurons_filtered,
                "total_features": features_processed,
                "interpretable_count": interpretable_count,
                "avg_activation_frequency": float(total_activation_freq / features_processed) if features_processed > 0 else 0.0,
                "avg_interpretability": float(total_interpretability / features_processed) if features_processed > 0 else 0.0,
                "min_activation_frequency_threshold": min_activation_frequency,
                "sae_format": format_type,
                "sae_architecture": architecture_type
            }

            # Cleanup
            if torch.cuda.is_available():
                sae.cpu()
                del sae
                del incremental_heap
                gc.collect()
                torch.cuda.empty_cache()

            # Mark completed
            self.update_extraction_status_sync(
                extraction_job.id,
                ExtractionStatus.COMPLETED.value,
                progress=1.0,
                features_extracted=features_processed,
                statistics=statistics
            )

            emit_progress(
                channel=f"extraction/{extraction_job.id}",
                event="extraction:completed",
                data={
                    "extraction_id": extraction_job.id,
                    "sae_id": sae_id,
                    "statistics": statistics
                }
            )

            return statistics

        except Exception as e:
            error_str = str(e)

            # Check if this is a CUDA OOM error and provide helpful diagnostics
            is_oom_error = "CUDA out of memory" in error_str or "OutOfMemoryError" in error_str

            if is_oom_error:
                # Build diagnostic information for OOM errors
                local_vars = locals()
                oom_diagnostics = self._build_oom_diagnostics(
                    error_str=error_str,
                    batch_size=local_vars.get('batch_size'),
                    tokenization=local_vars.get('tokenization'),
                    model_record=local_vars.get('model_record'),
                    model_config=local_vars.get('model_config'),
                )
                error_str = oom_diagnostics
                logger.error(f"Feature extraction OOM for SAE {sae_id}:\n{oom_diagnostics}")
            else:
                logger.error(f"Feature extraction failed for SAE {sae_id}: {e}", exc_info=True)

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            self.update_extraction_status_sync(
                extraction_job.id,
                ExtractionStatus.FAILED.value,
                error_message=error_str
            )

            emit_progress(
                channel=f"extraction/{extraction_job.id}",
                event="extraction:failed",
                data={
                    "extraction_id": extraction_job.id,
                    "sae_id": sae_id,
                    "error": error_str
                }
            )

            raise

        finally:
            # CRITICAL: Always clean up GPU memory regardless of success or failure
            # This prevents VRAM leaks that accumulate across extraction jobs
            logger.info(f"[extract_features_for_sae] Entering finally block for cleanup")
            models_to_cleanup = []
            if base_model is not None:
                models_to_cleanup.append(base_model)
            if sae is not None:
                models_to_cleanup.append(sae)

            cleanup_gpu_memory(
                models_to_cleanup=models_to_cleanup if models_to_cleanup else None,
                context=f"extract_features_for_sae({sae_id})"
            )

            # Also clean up large data structures
            if incremental_heap is not None:
                del incremental_heap
            if tokenizer is not None:
                tokenizer = None  # Clear reference instead of del to avoid UnboundLocalError in finally

            # Final garbage collection
            gc.collect()

    def _build_oom_diagnostics(
        self,
        error_str: str,
        batch_size: Optional[int] = None,
        tokenization: Optional[Any] = None,
        model_record: Optional[Any] = None,
        model_config: Optional[Any] = None,
    ) -> str:
        """
        Build a helpful diagnostic message for CUDA OOM errors.

        This method provides users with:
        - Current configuration causing the OOM
        - Memory estimate breakdown
        - Specific recommendations for fixing the issue

        Args:
            error_str: Original error message
            batch_size: Current batch size
            tokenization: DatasetTokenization record (has max_length)
            model_record: Model database record
            model_config: HuggingFace model config

        Returns:
            Formatted diagnostic string for display in UI
        """
        lines = ["CUDA Out of Memory Error"]
        lines.append("=" * 50)
        lines.append("")

        # Current Configuration Section
        lines.append("CURRENT CONFIGURATION:")
        if batch_size:
            lines.append(f"  • Batch size: {batch_size}")
        if tokenization and hasattr(tokenization, 'max_length'):
            lines.append(f"  • Sequence length: {tokenization.max_length} tokens")
        if model_record:
            lines.append(f"  • Model: {model_record.name or model_record.repo_id}")
            lines.append(f"  • Quantization: {model_record.quantization}")
        lines.append("")

        # Memory Estimate Section
        seq_len = tokenization.max_length if tokenization and hasattr(tokenization, 'max_length') else None
        num_heads = getattr(model_config, 'num_attention_heads', None) if model_config else None
        num_layers = getattr(model_config, 'num_hidden_layers', None) if model_config else None

        if seq_len and batch_size and num_heads and num_layers:
            # Attention memory: batch * heads * seq_len * seq_len * 4 bytes (FP32) per layer
            attention_per_layer_gb = (batch_size * num_heads * seq_len * seq_len * 4) / (1024**3)
            total_attention_gb = attention_per_layer_gb * num_layers
            lines.append("MEMORY ESTIMATE (attention intermediates only):")
            lines.append(f"  • Attention memory per layer: {attention_per_layer_gb:.2f} GB")
            lines.append(f"  • Total attention memory ({num_layers} layers): {total_attention_gb:.2f} GB")
            lines.append(f"  • Formula: batch({batch_size}) × heads({num_heads}) × seq²({seq_len}²) × 4 bytes")
            lines.append("")

        # Recommendations Section
        lines.append("RECOMMENDATIONS:")
        if seq_len and seq_len >= 1024:
            recommended_seq = 512 if seq_len >= 2048 else seq_len // 2
            lines.append(f"  1. Re-tokenize dataset with shorter max_length (try {recommended_seq})")
            lines.append(f"     Attention memory scales with sequence_length² (quadratic)")
        if batch_size and batch_size > 4:
            recommended_batch = max(1, batch_size // 2)
            lines.append(f"  2. Reduce batch_size to {recommended_batch}")
            lines.append(f"     (Attention memory scales linearly with batch_size)")
        if model_record and model_record.quantization == 'FP32':
            lines.append("  3. Re-download model with FP16 quantization")
            lines.append("     (Reduces model memory by ~50%)")
        lines.append("")

        # Quick Reference
        lines.append("MEMORY SCALING QUICK REFERENCE:")
        lines.append("  • seq_len 2048→1024: ~4× less attention memory")
        lines.append("  • seq_len 1024→512:  ~4× less attention memory")
        lines.append("  • batch_size 8→4:    ~2× less attention memory")
        lines.append("")

        # Original error (truncated)
        lines.append("ORIGINAL ERROR:")
        # Truncate long error messages
        short_error = error_str[:200] + "..." if len(error_str) > 200 else error_str
        lines.append(f"  {short_error}")

        return "\n".join(lines)

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
