"""
Celery tasks for SAE training operations.

This module contains Celery tasks for training Sparse Autoencoders,
including the main training loop, metric logging, and checkpoint management.
"""

import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from celery import Task
from datasets import load_from_disk

from .base_task import DatabaseTask
from ..ml.sparse_autoencoder import create_sae, project_decoder_gradients, JumpReLUSAE
from ..models.training import Training, TrainingStatus
from ..models.dataset import Dataset
from ..models.model import Model
from ..models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from ..services.training_service import TrainingService
from ..services.checkpoint_service import CheckpointService
from ..core.config import settings
from ..utils.resource_estimation import (
    estimate_training_memory,
    estimate_multilayer_training_memory,
    estimate_oom_reduced_batch_size,
)
from ..services.training_validator import TrainingValidator
from ..ml.model_loader import load_model_from_hf, QuantizationFormat
from ..ml.forward_hooks import HookManager, HookType

logger = logging.getLogger(__name__)


def get_celery_app():
    """Import celery app lazily to avoid circular imports."""
    from ..core.celery_app import celery_app
    return celery_app


class TrainingTask(DatabaseTask):
    """Base class for training tasks with additional utilities."""

    def update_training_progress(
        self,
        training_id: str,
        step: int,
        total_steps: int,
        loss: float,
        l0_sparsity: Optional[float] = None,
        dead_neurons: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ):
        """
        Update training progress in database.

        Args:
            training_id: Training job ID
            step: Current training step
            total_steps: Total training steps
            loss: Current loss
            l0_sparsity: Current L0 sparsity
            dead_neurons: Current dead neuron count
            learning_rate: Current learning rate
        """
        with self.get_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            if training:
                progress = (step / total_steps) * 100.0
                training.progress = progress
                training.current_step = step
                training.current_loss = loss
                training.current_l0_sparsity = l0_sparsity
                training.current_dead_neurons = dead_neurons
                training.current_learning_rate = learning_rate
                training.status = TrainingStatus.RUNNING.value
                db.commit()

    def log_metric(
        self,
        training_id: str,
        step: int,
        loss: float,
        l0_sparsity: Optional[float] = None,
        l1_sparsity: Optional[float] = None,
        dead_neurons: Optional[int] = None,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None,
        gpu_memory_used_mb: Optional[float] = None,
        samples_per_second: Optional[float] = None,
        layer_idx: Optional[int] = None,
        fvu: Optional[float] = None,
    ):
        """
        Log training metric to database.

        Args:
            training_id: Training job ID
            step: Training step
            loss: Total loss
            l0_sparsity: L0 sparsity
            l1_sparsity: L1 sparsity penalty
            dead_neurons: Dead neuron count
            learning_rate: Learning rate
            grad_norm: Gradient norm
            gpu_memory_used_mb: GPU memory usage
            samples_per_second: Training throughput
            layer_idx: Layer index (None for aggregated metrics)
            fvu: Fraction of Variance Unexplained (var_residuals / var_original)
        """
        with self.get_db() as db:
            from ..models.training_metric import TrainingMetric

            metric = TrainingMetric(
                training_id=training_id,
                step=step,
                loss=loss,
                l0_sparsity=l0_sparsity,
                l1_sparsity=l1_sparsity,
                dead_neurons=dead_neurons,
                learning_rate=learning_rate,
                grad_norm=grad_norm,
                gpu_memory_used_mb=gpu_memory_used_mb,
                samples_per_second=samples_per_second,
                layer_idx=layer_idx,
                fvu=fvu,
            )
            db.add(metric)
            db.commit()

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """
        Hook called after task returns (success, failure, or revocation).

        CRITICAL: This ensures GPU memory cleanup even when task is cancelled/revoked.
        Without this, cancelled training jobs leave models in GPU memory.

        Args:
            status: Task state ('SUCCESS', 'FAILURE', 'REVOKED', etc.)
            retval: Return value (or exception if failed)
            task_id: Task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
            einfo: Exception info (if failed)
        """
        logger.info(f"Task {self.name}[{task_id}] after_return: status={status}")

        # Force GPU cleanup on task exit (especially important for REVOKED tasks)
        try:
            import gc
            import torch

            logger.info("Forcing GPU memory cleanup after task return...")

            # Force garbage collection to clean up any lingering references
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                # Log memory after cleanup
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                    reserved = torch.cuda.memory_reserved() / (1024**2)    # MB
                    logger.info(f"GPU memory after cleanup: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

            logger.info("GPU memory cleanup completed in after_return")

        except Exception as cleanup_error:
            logger.warning(f"Error during after_return GPU cleanup: {cleanup_error}")

        # Call parent's after_return
        super().after_return(status, retval, task_id, args, kwargs, einfo)


@get_celery_app().task(
    base=TrainingTask,
    bind=True,
    name="train_sae",
    acks_late=False,  # Acknowledge task when it STARTS (not completes) to prevent re-execution
    task_reject_on_worker_lost=True,  # Reject (don't requeue) if worker crashes
)
def train_sae_task(
    self,
    training_id: str,
) -> Dict[str, Any]:
    """
    Main SAE training task.

    This task performs the full training loop for a Sparse Autoencoder,
    including:
    - Model initialization
    - Data loading (from extraction or dataset)
    - Training loop with gradient updates
    - Metric logging
    - Checkpoint saving
    - Error handling and recovery

    Args:
        training_id: Training job ID

    Returns:
        Dictionary with training results

    Note:
        This task uses acks_late=False to prevent automatic re-execution
        after worker restarts. Combined with the idempotency check at the
        start of the task, this ensures completed trainings are never
        accidentally restarted.
    """
    logger.info(f"Starting SAE training task for training_id={training_id}")

    # Get training record
    with self.get_db() as db:
        training = db.query(Training).filter_by(id=training_id).first()
        if not training:
            raise ValueError(f"Training not found: {training_id}")

        # IDEMPOTENCY CHECK: Skip if training is already completed
        # This prevents re-execution when tasks are requeued due to worker restarts
        if training.status == TrainingStatus.COMPLETED.value:
            logger.warning(
                f"Training {training_id} is already completed at step {training.current_step}. "
                f"Skipping task execution to prevent duplicate work."
            )
            return {
                "status": "already_completed",
                "steps": training.current_step,
                "final_loss": training.current_loss,
                "message": f"Training was already completed at step {training.current_step}",
            }

        # Update status to initializing
        training.status = TrainingStatus.INITIALIZING.value
        db.commit()

        # Extract hyperparameters
        hp = training.hyperparameters
        logger.info(f"Hyperparameters: {hp}")

        # Extract training layers (default to [0] for backward compatibility)
        training_layers = hp.get('training_layers', [0])
        if not isinstance(training_layers, list):
            training_layers = [training_layers]  # Convert single int to list
        logger.info(f"Training layers: {training_layers}")

    try:
        # Memory budget validation
        logger.info("Validating memory budget...")
        batch_size = hp['batch_size']
        num_layers = len(training_layers)

        if num_layers == 1:
            # Single-layer training
            memory_estimate = estimate_training_memory(
                hidden_dim=hp['hidden_dim'],
                latent_dim=hp['latent_dim'],
                batch_size=batch_size,
            )
        else:
            # Multi-layer training
            memory_estimate = estimate_multilayer_training_memory(
                hidden_dim=hp['hidden_dim'],
                latent_dim=hp['latent_dim'],
                batch_size=batch_size,
                num_layers=num_layers,
            )

        available_gpu_gb = memory_estimate.get('available_gpu_gb', 6.0)
        logger.info(f"Estimated memory usage: {memory_estimate['total_gb']:.2f} GB (Available: {available_gpu_gb:.2f} GB)")
        if num_layers > 1:
            logger.info(f"Per-layer memory: {memory_estimate['per_layer_gb']:.2f} GB")
            logger.info(f"Max layers in available memory: {memory_estimate['max_layers_in_6gb']}")

        if not memory_estimate['fits_in_6gb']:
            error_msg = (
                f"Training requires {memory_estimate['total_gb']:.2f} GB but only {available_gpu_gb:.2f} GB available. "
                f"{memory_estimate.get('recommendation', 'Reduce batch_size or latent_dim.')}"
            )
            logger.error(error_msg)
            with self.get_db() as db:
                training = db.query(Training).filter_by(id=training_id).first()
                training.status = TrainingStatus.FAILED.value
                training.error_message = error_msg
                db.commit()
            raise RuntimeError(error_msg)

        # Validate sparsity configuration
        logger.info("Validating sparsity configuration...")
        warnings, errors = TrainingValidator.validate_sparsity_config(hp)

        # Log errors and fail if critical issues found
        if errors:
            error_msg = "Sparsity configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            with self.get_db() as db:
                training = db.query(Training).filter_by(id=training_id).first()
                training.status = TrainingStatus.FAILED.value
                training.error_message = error_msg
                db.commit()
            raise ValueError(error_msg)

        # Log warnings (non-blocking)
        if warnings:
            logger.warning("Sparsity configuration warnings:")
            for warning in warnings:
                logger.warning(f"  {warning}")

            # Calculate recommended l1_alpha for reference
            recommended_l1_alpha = TrainingValidator.calculate_recommended_l1_alpha(hp['latent_dim'])
            logger.info(f"Recommended l1_alpha for latent_dim {hp['latent_dim']}: {recommended_l1_alpha:.6f}")

        # Initialize models, optimizers, and schedulers (one per layer)
        logger.info(f"Initializing SAE models for {num_layers} layer(s)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        models = {}
        optimizers = {}
        schedulers = {}

        for layer_idx in training_layers:
            # Create SAE for this layer
            architecture_type = hp.get('architecture_type', 'standard')
            model = create_sae(
                architecture_type=architecture_type,
                hidden_dim=hp['hidden_dim'],
                latent_dim=hp['latent_dim'],
                l1_alpha=hp['l1_alpha'],
                ghost_gradient_penalty=hp.get('ghost_gradient_penalty', 0.0),
                normalize_activations=hp.get('normalize_activations', 'constant_norm_rescale'),
                top_k_sparsity=hp.get('top_k_sparsity', None),
                # JumpReLU-specific parameters
                initial_threshold=hp.get('initial_threshold', 0.001),
                bandwidth=hp.get('bandwidth', 0.001),
                sparsity_coeff=hp.get('sparsity_coeff'),
                normalize_decoder=hp.get('normalize_decoder', True),
            ).to(device)
            models[layer_idx] = model

            # Initialize optimizer for this layer
            # JumpReLU uses Adam with betas=(0.0, 0.999) per Gemma Scope paper
            if architecture_type == 'jumprelu':
                adam_betas = (0.0, 0.999)
            else:
                adam_betas = (0.9, 0.999)  # Default Adam betas

            optimizer = optim.Adam(
                model.parameters(),
                lr=hp['learning_rate'],
                weight_decay=hp.get('weight_decay', 0.0),
                betas=adam_betas,
            )
            optimizers[layer_idx] = optimizer

            # Learning rate scheduler (linear warmup + constant)
            warmup_steps = hp.get('warmup_steps', 0)

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return 1.0

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            schedulers[layer_idx] = scheduler

            logger.info(f"  Layer {layer_idx}: SAE model initialized")

        # Initialize gradient scalers for mixed precision training (one per layer)
        scalers = {}
        if torch.cuda.is_available():
            for layer_idx in training_layers:
                scalers[layer_idx] = GradScaler()
            logger.info("Mixed precision training (FP16) enabled with GradScaler")
        else:
            logger.info("CPU training detected, mixed precision disabled")

        # Create checkpoint directory
        checkpoint_dir = settings.data_dir / "trainings" / training_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with self.get_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            training.checkpoint_dir = str(checkpoint_dir)
            training.status = TrainingStatus.RUNNING.value
            db.commit()

        logger.info("Model initialized successfully")

        # Training loop configuration
        total_steps = hp['total_steps']
        batch_size = hp['batch_size']
        checkpoint_interval = hp.get('checkpoint_interval', 1000)
        log_interval = hp.get('log_interval', 100)

        # Gradient accumulation settings
        effective_batch_size = batch_size
        grad_accum_steps = 1
        if batch_size < 64:
            # Use gradient accumulation to maintain effective batch size of 64
            grad_accum_steps = max(1, 64 // batch_size)
            effective_batch_size = batch_size * grad_accum_steps
            logger.info(f"Using gradient accumulation: {grad_accum_steps} steps for effective batch size {effective_batch_size}")

        # OOM retry tracking
        oom_retry_count = 0
        max_oom_retries = 3

        # Throughput monitoring
        import time
        step_start_time = time.time()
        steps_per_min_target = 100  # Minimum acceptable throughput

        # Check if using cached activations or need to extract on-the-fly
        use_cached_activations = training.extraction_id is not None
        cached_activations = {}
        dataset = None
        base_model = None
        tokenizer = None
        architecture = None
        hook_types = None

        if use_cached_activations:
            # Load cached activations from extraction
            logger.info(f"Using cached activations from extraction: {training.extraction_id}")
            from ..models.activation_extraction import ActivationExtraction

            with self.get_db() as db:
                extraction = db.query(ActivationExtraction).filter(
                    ActivationExtraction.id == training.extraction_id
                ).first()
                if not extraction:
                    raise ValueError(f"Extraction {training.extraction_id} not found")

                if extraction.status != "completed":
                    raise ValueError(f"Extraction {training.extraction_id} is not completed (status: {extraction.status})")

            extraction_path = settings.resolve_data_path(extraction.output_path)
            logger.info(f"Loading activations from: {extraction_path}")

            # Load metadata
            metadata_path = extraction_path / "metadata.json"
            with open(metadata_path, 'r') as f:
                extraction_metadata = json.load(f)

            logger.info(f"Extraction metadata: {extraction_metadata['num_samples_processed']} samples")

            # Load activation files for each training layer
            for layer_idx in training_layers:
                activation_file = extraction_path / f"layer_{layer_idx}_residual.npy"
                if not activation_file.exists():
                    raise ValueError(
                        f"Activation file not found for layer {layer_idx}: {activation_file}. "
                        f"Available layers in extraction: {extraction_metadata['layer_indices']}"
                    )

                logger.info(f"Loading layer {layer_idx} activations from {activation_file}")
                # Use memory-mapped loading for large files to avoid RAM exhaustion
                # Shape: (num_samples, seq_len, hidden_dim)
                layer_acts_mmap = np.load(activation_file, mmap_mode='r')
                logger.info(f"  Memory-mapped shape: {layer_acts_mmap.shape}, dtype: {layer_acts_mmap.dtype}")

                # Average over sequence dimension in chunks to save RAM
                # (num_samples, seq_len, hidden_dim) -> (num_samples, hidden_dim)
                num_samples, seq_len, hidden_dim = layer_acts_mmap.shape
                chunk_size = 1000  # Process 1000 samples at a time
                layer_acts_mean = np.zeros((num_samples, hidden_dim), dtype=np.float32)

                logger.info(f"  Averaging over sequence dimension in chunks of {chunk_size}...")
                for start_idx in range(0, num_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_samples)
                    chunk = layer_acts_mmap[start_idx:end_idx]  # Load chunk into RAM
                    layer_acts_mean[start_idx:end_idx] = chunk.mean(axis=1).astype(np.float32)

                # Convert to torch tensor and move to GPU
                layer_acts_tensor = torch.from_numpy(layer_acts_mean).to(device)
                cached_activations[layer_idx] = layer_acts_tensor
                logger.info(f"  Loaded shape: {layer_acts_mmap.shape} -> averaged to {layer_acts_tensor.shape} on GPU")

            num_samples = cached_activations[training_layers[0]].shape[0]
            logger.info(f"Cached activations ready: {num_samples} samples across {len(training_layers)} layers (all on GPU)")

        else:
            # Load dataset and base model for on-the-fly activation extraction
            logger.info("Loading dataset and base model for activation extraction...")
            with self.get_db() as db:
                dataset_record = db.query(Dataset).filter(
                    Dataset.id == training.dataset_id
                ).first()
                if not dataset_record:
                    raise ValueError(f"Dataset {training.dataset_id} not found")

                model_record = db.query(Model).filter(
                    Model.id == training.model_id
                ).first()
                if not model_record:
                    raise ValueError(f"Model {training.model_id} not found")

                # Query the tokenization for this dataset + model combination
                tokenization = db.query(DatasetTokenization).filter(
                    DatasetTokenization.dataset_id == training.dataset_id,
                    DatasetTokenization.model_id == training.model_id
                ).first()
                if not tokenization:
                    raise ValueError(
                        f"No tokenization found for dataset {training.dataset_id} with model {training.model_id}. "
                        f"Please tokenize the dataset with this model first."
                    )
                if tokenization.status != TokenizationStatus.READY:
                    raise ValueError(
                        f"Tokenization for dataset {training.dataset_id} with model {training.model_id} "
                        f"is not ready (status: {tokenization.status}). Please wait for tokenization to complete."
                    )

            # Resolve relative path to absolute using data_dir setting
            resolved_tokenized_path = str(settings.resolve_data_path(tokenization.tokenized_path))
            logger.info(f"Loading dataset from {resolved_tokenized_path}")
            dataset = load_from_disk(resolved_tokenized_path)

            logger.info(f"Loading base model: {model_record.repo_id}")
            # Use local_files_only=True when model is already downloaded to avoid
            # HuggingFace API calls that require authentication for gated models
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

            # Validate tokenizer/model vocabulary compatibility
            dataset_tokenizer_name = tokenization.tokenizer_repo_id
            dataset_vocab_size = tokenization.vocab_size

            model_vocab_size = model_config.vocab_size if hasattr(model_config, "vocab_size") else tokenizer.vocab_size

            if dataset_vocab_size and model_vocab_size:
                vocab_size_diff = abs(dataset_vocab_size - model_vocab_size)
                vocab_size_ratio = vocab_size_diff / model_vocab_size

                if vocab_size_ratio > 0.1:  # More than 10% difference
                    error_msg = (
                        f"Tokenizer/model vocabulary mismatch:\n"
                        f"  Dataset tokenizer: {dataset_tokenizer_name or 'unknown'} (vocab_size: {dataset_vocab_size})\n"
                        f"  Model: {model_record.repo_id} (vocab_size: {model_vocab_size})\n"
                        f"  Please re-tokenize the dataset using the model's tokenizer."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                elif vocab_size_diff > 100:
                    logger.warning(
                        f"Minor vocabulary size difference: "
                        f"dataset={dataset_vocab_size}, model={model_vocab_size}"
                    )

            logger.info(
                f"Vocabulary check: dataset_tokenizer={dataset_tokenizer_name or 'unknown'}, "
                f"model_vocab_size={model_vocab_size}"
            )

            # Extract actual hidden dimension from model config
            # Override user-provided hidden_dim to match the actual model
            actual_hidden_dim = getattr(model_config, 'hidden_size', None)
            if actual_hidden_dim is None:
                # Try alternative attribute names
                actual_hidden_dim = getattr(model_config, 'd_model', None)

            if actual_hidden_dim is not None:
                if hp['hidden_dim'] != actual_hidden_dim:
                    logger.warning(
                        f"User-provided hidden_dim ({hp['hidden_dim']}) does not match "
                        f"model's actual hidden dimension ({actual_hidden_dim}). "
                        f"Using model's actual dimension."
                    )
                    hp['hidden_dim'] = actual_hidden_dim
                logger.info(f"Model hidden dimension: {actual_hidden_dim}")
            else:
                logger.warning(
                    f"Could not auto-detect model's hidden dimension. "
                    f"Using user-provided value: {hp['hidden_dim']}"
                )

            architecture = model_record.architecture
            hook_types = [HookType.RESIDUAL]  # Default to residual stream

            num_samples = len(dataset)
            logger.info(f"Dataset: {num_samples} samples, Model: {model_record.repo_id}")
        logger.info(f"Starting training loop: {total_steps} steps, batch_size={batch_size}")

        # Training loop
        best_loss = float('inf')

        for step in range(total_steps):
            # Check for pause/stop signals
            with self.get_db() as db:
                training = db.query(Training).filter_by(id=training_id).first()
                if training.status == TrainingStatus.PAUSED.value:
                    logger.info(f"Training paused at step {step}")
                    return {"status": "paused", "step": step}
                elif training.status == TrainingStatus.CANCELLED.value:
                    logger.info(f"Training cancelled at step {step}")
                    return {"status": "cancelled", "step": step}

            try:
                # ==================================================================
                # SANITY CHECK: First step validates activation extraction
                # ==================================================================
                if step == 1:
                    logger.info("=" * 70)
                    logger.info("STEP 1 VALIDATION: Checking activation extraction...")
                    logger.info("=" * 70)

                # Get activations for this training step
                layer_activations = {}

                if use_cached_activations:
                    # Sample from cached activations (already on GPU)
                    batch_indices = torch.randint(0, num_samples, (batch_size,), device=device)

                    for layer_idx in training_layers:
                        # Get cached activations: shape (num_samples, hidden_dim) already on GPU
                        cached = cached_activations[layer_idx]
                        # Sample batch directly on GPU - super fast!
                        layer_activations[layer_idx] = cached[batch_indices]  # Shape: (batch_size, hidden_dim)

                        # VALIDATION: Check activation statistics on first step
                        if step == 1:
                            act_mean = cached.mean().item()
                            act_std = cached.std().item()
                            act_min = cached.min().item()
                            act_max = cached.max().item()
                            logger.info(f"Layer {layer_idx} cached activations sampled successfully:")
                            logger.info(f"  Cached shape on GPU: {cached.shape}")
                            logger.info(f"  Mean: {act_mean:.4f}, Std: {act_std:.4f}")
                            logger.info(f"  Range: [{act_min:.4f}, {act_max:.4f}]")

                            # Sanity check
                            if act_std < 0.01 or act_std > 100:
                                logger.error(f"SUSPICIOUS: Layer {layer_idx} std={act_std:.4f} is unusual!")
                            if abs(act_mean) > 50:
                                logger.error(f"SUSPICIOUS: Layer {layer_idx} mean={act_mean:.4f} is unusual!")

                else:
                    # Extract activations on-the-fly from base model
                    # Sample random batch from dataset
                    batch_indices = torch.randint(0, num_samples, (batch_size,)).tolist()
                    batch = dataset.select(batch_indices)

                    # Get input_ids from batch
                    batch_input_ids = []
                    if "input_ids" in batch.column_names:
                        for ids in batch["input_ids"]:
                            if isinstance(ids, list):
                                batch_input_ids.append(ids)
                            else:
                                batch_input_ids.append(ids.tolist() if hasattr(ids, 'tolist') else list(ids))

                    # Pad sequences to same length
                    max_length = min(max(len(ids) for ids in batch_input_ids), 512)
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                    padded_input_ids = []
                    attention_masks = []

                    for input_ids in batch_input_ids:
                        # Truncate if too long
                        if len(input_ids) > max_length:
                            input_ids = input_ids[:max_length]

                        padding_length = max_length - len(input_ids)
                        padded_ids = input_ids + [pad_token_id] * padding_length
                        mask = [1] * len(input_ids) + [0] * padding_length

                        padded_input_ids.append(padded_ids)
                        attention_masks.append(mask)

                    # Convert to tensors
                    input_ids_tensor = torch.tensor(padded_input_ids, device=device)
                    attention_mask_tensor = torch.tensor(attention_masks, device=device)

                    # Extract activations using HookManager
                    with HookManager(base_model) as hook_manager:
                        hook_manager.register_hooks(training_layers, hook_types, architecture)

                        with torch.no_grad():
                            _ = base_model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)

                        # Get captured activations for each layer
                        for layer_idx in training_layers:
                            # Find the activation for this layer
                            # Key format is "layer_{idx}_{hook_type}" (e.g., "layer_9_residual")
                            layer_key = None
                            for key in hook_manager.activations.keys():
                                if f"layer_{layer_idx}_" in key:
                                    layer_key = key
                                    break

                            if layer_key and hook_manager.activations[layer_key]:
                                acts = hook_manager.activations[layer_key][0]  # Shape: (batch_size, seq_len, hidden_dim)
                                # Average over sequence dimension to get (batch_size, hidden_dim)
                                # Move to correct device (activations are on CPU from hook)
                                layer_activations[layer_idx] = acts.mean(dim=1).detach().to(device)

                                # VALIDATION: Check activation statistics on first step
                                if step == 1:
                                    act_mean = acts.mean().item()
                                    act_std = acts.std().item()
                                    act_min = acts.min().item()
                                    act_max = acts.max().item()
                                    logger.info(f"Layer {layer_idx} activations captured successfully:")
                                    logger.info(f"  Shape: {acts.shape}")
                                    logger.info(f"  Mean: {act_mean:.4f}, Std: {act_std:.4f}")
                                    logger.info(f"  Range: [{act_min:.4f}, {act_max:.4f}]")

                                    # Sanity check: Real activations should have reasonable statistics
                                    if act_std < 0.01 or act_std > 100:
                                        logger.error(f"SUSPICIOUS: Layer {layer_idx} std={act_std:.4f} is unusual!")
                                    if abs(act_mean) > 50:
                                        logger.error(f"SUSPICIOUS: Layer {layer_idx} mean={act_mean:.4f} is unusual!")
                            else:
                                # CRITICAL ERROR: No activations captured means hooks failed
                                logger.error(f"FATAL: No activations captured for layer {layer_idx}")
                                logger.error(f"Available keys: {list(hook_manager.activations.keys())}")
                                logger.error(f"Expected key pattern: layer_{layer_idx}_*")
                                raise RuntimeError(
                                    f"Failed to capture activations for layer {layer_idx}. "
                                    f"Hook registration failed. Available keys: {list(hook_manager.activations.keys())}"
                                )

                # Train all layers
                layer_losses = {}
                layer_sparsities = {}
                layer_dead_neurons = {}
                layer_fvu = {}

                for layer_idx in training_layers:
                    x = layer_activations[layer_idx]
                    model = models[layer_idx]
                    optimizer = optimizers[layer_idx]
                    scheduler = schedulers[layer_idx]
                    scaler = scalers.get(layer_idx)  # None if CPU training

                    # Forward pass
                    if step % grad_accum_steps == 0:
                        optimizer.zero_grad()

                    # Forward pass with mixed precision (FP16) if GPU available
                    if scaler is not None:
                        with autocast():
                            # Handle different architecture types
                            architecture_type = hp.get('architecture_type', 'standard')
                            if architecture_type == 'transcoder':
                                x_reconstructed, z, losses = model(x, x, return_loss=True)
                            else:
                                x_reconstructed, z, losses = model(x, return_loss=True)

                            loss = losses['loss']
                            if grad_accum_steps > 1:
                                loss = loss / grad_accum_steps

                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                    else:
                        # CPU training - no mixed precision
                        architecture_type = hp.get('architecture_type', 'standard')
                        if architecture_type == 'transcoder':
                            x_reconstructed, z, losses = model(x, x, return_loss=True)
                        else:
                            x_reconstructed, z, losses = model(x, return_loss=True)

                        loss = losses['loss']
                        if grad_accum_steps > 1:
                            loss = loss / grad_accum_steps
                        loss.backward()

                    # Optimizer step (only every grad_accum_steps)
                    if (step + 1) % grad_accum_steps == 0:
                        if scaler is not None:
                            # Mixed precision: unscale gradients before clipping
                            scaler.unscale_(optimizer)

                        # Gradient clipping
                        grad_clip_norm = hp.get('grad_clip_norm')
                        if grad_clip_norm:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                        # JumpReLU: Project decoder gradients orthogonal to decoder columns
                        # This prevents the decoder from learning to increase norms
                        if isinstance(model, JumpReLUSAE):
                            project_decoder_gradients(model)

                        # Optimizer step with scaler
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        # JumpReLU: Normalize decoder columns to unit norm after each step
                        if isinstance(model, JumpReLUSAE):
                            model.normalize_decoder()

                        scheduler.step()

                    # Store layer metrics
                    layer_losses[layer_idx] = loss.item() * grad_accum_steps  # Undo accumulation scaling
                    layer_sparsities[layer_idx] = (z != 0).float().mean().item()
                    layer_dead_neurons[layer_idx] = (z == 0).all(dim=0).sum().item()
                    # Store FVU if available (JumpReLU SAE computes this)
                    # Convert tensor to float for database storage
                    fvu_val = losses.get('fvu', None)
                    if fvu_val is not None:
                        layer_fvu[layer_idx] = fvu_val.item() if hasattr(fvu_val, 'item') else float(fvu_val)
                    else:
                        layer_fvu[layer_idx] = None

                # Calculate aggregated metrics across all layers
                avg_loss = sum(layer_losses.values()) / len(layer_losses)
                avg_sparsity = sum(layer_sparsities.values()) / len(layer_sparsities)
                avg_dead_neurons = sum(layer_dead_neurons.values()) / len(layer_dead_neurons)
                # Calculate avg FVU only if any layer has FVU (JumpReLU)
                fvu_values = [v for v in layer_fvu.values() if v is not None]
                avg_fvu = float(sum(fvu_values) / len(fvu_values)) if fvu_values else None

                # Clear GPU cache after every step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Reset OOM retry count on successful step
                oom_retry_count = 0

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM error handling
                    oom_retry_count += 1
                    logger.warning(f"OOM error at step {step} (retry {oom_retry_count}/{max_oom_retries})")

                    if oom_retry_count >= max_oom_retries:
                        error_msg = f"Training failed after {max_oom_retries} OOM errors. Batch size too large."
                        logger.error(error_msg)
                        with self.get_db() as db:
                            training = db.query(Training).filter_by(id=training_id).first()
                            training.status = TrainingStatus.FAILED.value
                            training.error_message = error_msg
                            db.commit()
                        raise RuntimeError(error_msg)

                    # Reduce batch size and retry
                    old_batch_size = batch_size
                    batch_size = estimate_oom_reduced_batch_size(batch_size)
                    logger.info(f"Reducing batch_size from {old_batch_size} to {batch_size}")

                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Update hyperparameters
                    hp['batch_size'] = batch_size
                    with self.get_db() as db:
                        training = db.query(Training).filter_by(id=training_id).first()
                        training.hyperparameters['batch_size'] = batch_size
                        db.commit()

                    # Skip to next iteration with new batch size
                    continue
                else:
                    # Re-raise other runtime errors
                    raise

            # Get aggregated metrics
            current_lr = schedulers[training_layers[0]].get_last_lr()[0]  # Use first layer's LR

            # Log metrics periodically
            if step % log_interval == 0:
                # THROUGHPUT MONITORING: Check if training is proceeding at acceptable speed
                if step >= 100:  # Check after 100 steps
                    elapsed_time = time.time() - step_start_time
                    actual_steps_per_min = (step / elapsed_time) * 60
                    if actual_steps_per_min < steps_per_min_target:
                        logger.error("=" * 70)
                        logger.error(f"PERFORMANCE ALERT: Training is too slow!")
                        logger.error(f"  Current: {actual_steps_per_min:.1f} steps/min")
                        logger.error(f"  Target:  {steps_per_min_target} steps/min minimum")
                        logger.error(f"  This is {steps_per_min_target/actual_steps_per_min:.1f}x slower than expected!")
                        logger.error("=" * 70)

                # GPU memory monitoring
                gpu_memory_mb = None
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
                    gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
                    gpu_memory_mb = gpu_memory_allocated

                    # Calculate throughput
                    if step >= 100:
                        elapsed_time = time.time() - step_start_time
                        actual_steps_per_min = (step / elapsed_time) * 60
                        logger.info(
                            f"Step {step}: avg_loss={avg_loss:.4f}, avg_sparsity={avg_sparsity:.4f}, "
                            f"throughput={actual_steps_per_min:.1f} steps/min, "
                            f"GPU memory: allocated={gpu_memory_allocated:.2f}MB"
                        )
                    else:
                        logger.info(
                            f"Step {step}: avg_loss={avg_loss:.4f}, avg_sparsity={avg_sparsity:.4f}, "
                            f"GPU memory: allocated={gpu_memory_allocated:.2f}MB"
                        )

                # Log aggregated metrics (layer_idx=None)
                self.log_metric(
                    training_id=training_id,
                    step=step,
                    loss=avg_loss,
                    l0_sparsity=avg_sparsity,
                    dead_neurons=int(avg_dead_neurons),
                    learning_rate=current_lr,
                    gpu_memory_used_mb=gpu_memory_mb,
                    layer_idx=None,  # Aggregated across all layers
                    fvu=avg_fvu,  # FVU metric (for JumpReLU SAE)
                )

                # Log per-layer metrics
                for layer_idx in training_layers:
                    self.log_metric(
                        training_id=training_id,
                        step=step,
                        loss=layer_losses[layer_idx],
                        l0_sparsity=layer_sparsities[layer_idx],
                        dead_neurons=int(layer_dead_neurons[layer_idx]),
                        learning_rate=current_lr,
                        layer_idx=layer_idx,
                        fvu=layer_fvu.get(layer_idx),  # Per-layer FVU
                    )

                # Update progress with aggregated metrics
                self.update_training_progress(
                    training_id=training_id,
                    step=step,
                    total_steps=total_steps,
                    loss=avg_loss,
                    l0_sparsity=avg_sparsity,
                    dead_neurons=int(avg_dead_neurons),
                    learning_rate=current_lr,
                )

                # Check training quality (with race-to-zero detection)
                quality_warnings = TrainingValidator.check_training_quality(
                    step=step,
                    l0_sparsity=avg_sparsity,
                    dead_neurons=int(avg_dead_neurons),
                    latent_dim=hp['latent_dim'],
                    target_l0=hp.get('target_l0', 0.05),
                    warmup_steps=hp.get('warmup_steps', 0),
                    training_id=training_id
                )
                if quality_warnings:
                    for warning in quality_warnings:
                        logger.warning(warning)

                # Dead neuron resampling (if enabled)
                if hp.get('resample_dead_neurons', False):
                    dead_neuron_threshold = hp.get('dead_neuron_threshold', 10000)
                    resample_interval = hp.get('resample_interval', 5000)

                    # Perform resampling at specified intervals after warmup
                    if step > 0 and step % resample_interval == 0 and step >= hp.get('warmup_steps', 0):
                        for layer_idx in training_layers:
                            model = models[layer_idx]

                            # Get current batch activations to identify dead neurons
                            x = layer_activations[layer_idx]
                            with torch.no_grad():
                                z = model.encode(x)
                                # Identify dead neurons (never activated in current batch)
                                dead_mask = (z == 0).all(dim=0)  # [latent_dim]
                                num_dead = dead_mask.sum().item()

                                if num_dead > 0:
                                    logger.info(f"Layer {layer_idx}: Resampling {num_dead} dead neurons at step {step}")

                                    # Resample dead neurons by reinitializing to high-loss examples
                                    # Strategy: Set encoder weights to point toward high-loss directions
                                    with torch.no_grad():
                                        # Get reconstruction loss per sample
                                        x_reconstructed, _, losses_dict = model(x, return_loss=True)
                                        reconstruction_errors = (x - x_reconstructed).pow(2).sum(dim=-1)  # [batch]

                                        # Find samples with highest reconstruction error
                                        topk_indices = torch.topk(reconstruction_errors, k=min(num_dead, x.size(0))).indices

                                        # Resample dead neurons
                                        dead_indices = torch.where(dead_mask)[0]
                                        for i, dead_idx in enumerate(dead_indices[:len(topk_indices)]):
                                            # Reinitialize encoder weights for this dead neuron
                                            # Point it toward a high-loss input example
                                            sample_idx = topk_indices[i]
                                            model.encoder.weight[dead_idx] = x[sample_idx] * 0.1  # Small scale
                                            model.encoder.bias[dead_idx] = 0.0

                                            # Reinitialize decoder weights
                                            # IMPORTANT: Handle JumpReLUSAE separately - its decoder property is a wrapper
                                            # that returns a TRANSPOSED copy, so assignments don't update the actual W_dec
                                            if not model.tied_weights:
                                                if hasattr(model, 'W_dec') and model.W_dec is not None:
                                                    # JumpReLUSAE: W_dec shape is [d_model, d_sae]
                                                    model.W_dec.data[:, dead_idx] = torch.randn_like(model.W_dec.data[:, dead_idx]) * 0.01
                                                elif hasattr(model, 'decoder') and hasattr(model.decoder, 'weight'):
                                                    # Standard SAE: decoder.weight shape is [hidden_dim, latent_dim]
                                                    model.decoder.weight[:, dead_idx] = torch.randn_like(model.decoder.weight[:, dead_idx]) * 0.01

                                        # Re-normalize decoder columns after resampling (critical for JumpReLUSAE)
                                        if hasattr(model, 'normalize_decoder') and callable(model.normalize_decoder):
                                            model.normalize_decoder()

                                        logger.info(f"  Resampled {min(num_dead, len(topk_indices))} neurons using high-loss examples")

                # Emit training:progress WebSocket event
                from ..workers.websocket_emitter import emit_training_progress
                emit_training_progress(
                    training_id=training_id,
                    event="training:progress",
                    data={
                        "training_id": training_id,
                        "current_step": step,
                        "total_steps": total_steps,
                        "progress": (step / total_steps) * 100.0,
                        "loss": avg_loss,
                        "l0_sparsity": avg_sparsity,
                        "dead_neurons": int(avg_dead_neurons),
                        "learning_rate": current_lr,
                        "num_layers": num_layers,
                        "training_layers": training_layers,
                    }
                )

            # Save checkpoint periodically
            if step % checkpoint_interval == 0 and step > 0:
                logger.info(f"Saving checkpoint at step {step}...")

                # Save multi-layer checkpoint
                checkpoint_paths = CheckpointService.save_multilayer_checkpoint(
                    models=models,
                    optimizers=optimizers,
                    step=step,
                    base_storage_path=str(checkpoint_dir),
                    training_layers=training_layers,
                    extra_metadata={
                        'avg_loss': avg_loss,
                        'avg_sparsity': avg_sparsity,
                        'layer_losses': {str(k): v for k, v in layer_losses.items()},
                    }
                )

                # Use first layer's checkpoint path for database record
                checkpoint_path = checkpoint_paths[training_layers[0]]

                # Create checkpoint record (using aggregated metrics)
                with self.get_db() as db:
                    from ..models.checkpoint import Checkpoint
                    from uuid import uuid4

                    is_best = avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss

                        # Unmark previous best checkpoints
                        prev_best = db.query(Checkpoint).filter_by(
                            training_id=training_id,
                            is_best=True
                        ).all()
                        for ckpt in prev_best:
                            ckpt.is_best = False

                    checkpoint_id = f"ckpt_{uuid4().hex[:8]}"
                    checkpoint = Checkpoint(
                        id=checkpoint_id,
                        training_id=training_id,
                        step=step,
                        loss=avg_loss,
                        l0_sparsity=avg_sparsity,
                        storage_path=checkpoint_path,
                        is_best=is_best,
                        extra_metadata={
                            'num_layers': num_layers,
                            'training_layers': training_layers,
                            'layer_losses': {str(k): v for k, v in layer_losses.items()},
                        },
                    )
                    db.add(checkpoint)
                    db.commit()

                    logger.info(f"Checkpoint saved: {checkpoint_id} (is_best={is_best})")

                    # Emit checkpoint:created WebSocket event
                    from ..workers.websocket_emitter import emit_checkpoint_created
                    emit_checkpoint_created(
                        training_id=training_id,
                        checkpoint_id=checkpoint_id,
                        step=step,
                        loss=avg_loss,
                        is_best=is_best,
                        storage_path=checkpoint_path,
                    )

                logger.info(f"Saved checkpoint at step {step} (best={is_best})")

        # Training completed
        logger.info(f"Training completed: {total_steps} steps")

        # Save final checkpoint in Community Standard format for interoperability
        logger.info("Saving final checkpoint in Community Standard format...")
        community_output_dir = settings.data_dir / "trainings" / training_id / "community_format"

        # Get model name from database
        with self.get_db() as db:
            training_record = db.query(Training).filter_by(id=training_id).first()
            model_record_for_name = db.query(Model).filter_by(id=training_record.model_id).first()
            model_name = model_record_for_name.repo_id if model_record_for_name else "unknown"

        # Save in Community Standard format
        CheckpointService.save_multilayer_community_checkpoint(
            models=models,
            base_output_dir=str(community_output_dir),
            model_name=model_name,
            training_layers=training_layers,
            hyperparams=hp,
            training_id=training_id,
            checkpoint_step=total_steps,
            tied_weights=hp.get('tied_weights', False),
        )
        logger.info(f"Saved Community Standard checkpoint to {community_output_dir}")

        # Cleanup: Unload base model and SAE models from GPU
        logger.info("Cleaning up GPU memory...")
        del base_model
        del tokenizer
        for layer_idx in training_layers:
            del models[layer_idx]
            del optimizers[layer_idx]
            del schedulers[layer_idx]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GPU memory cleanup completed")

        with self.get_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            training.status = TrainingStatus.COMPLETED.value
            training.progress = 100.0
            from datetime import datetime, UTC
            training.completed_at = datetime.now(UTC)
            db.commit()

        # Emit training:completed WebSocket event
        from ..workers.websocket_emitter import emit_training_progress
        emit_training_progress(
            training_id=training_id,
            event="training:completed",
            data={
                "training_id": training_id,
                "status": "completed",
                "final_loss": avg_loss,
            }
        )

        return {
            "status": "completed",
            "steps": total_steps,
            "final_loss": avg_loss,
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())

        # Cleanup: Unload models from GPU
        try:
            logger.info("Cleaning up GPU memory after failure...")
            if 'base_model' in locals():
                del base_model
            if 'tokenizer' in locals():
                del tokenizer
            if 'models' in locals():
                for layer_idx in training_layers:
                    if layer_idx in models:
                        del models[layer_idx]
                    if layer_idx in optimizers:
                        del optimizers[layer_idx]
                    if layer_idx in schedulers:
                        del schedulers[layer_idx]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("GPU memory cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")

        # Mark training as failed
        with self.get_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            if training:
                training.status = TrainingStatus.FAILED.value
                training.error_message = str(e)
                training.error_traceback = traceback.format_exc()
                from datetime import datetime, UTC
                training.completed_at = datetime.now(UTC)
                db.commit()

        # Emit training:failed WebSocket event
        from ..workers.websocket_emitter import emit_training_progress
        emit_training_progress(
            training_id=training_id,
            event="training:failed",
            data={
                "training_id": training_id,
                "error_message": str(e),
            }
        )

        raise


@get_celery_app().task(name="resume_training")
def resume_training_task(training_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Resume a paused training job.

    Args:
        training_id: Training job ID
        checkpoint_id: Optional checkpoint ID to resume from

    Returns:
        Dictionary with resume result
    """
    # TODO: Implement resume logic
    # This would load the latest (or specified) checkpoint and continue training
    logger.info(f"Resume training not yet implemented: {training_id}")
    return {"status": "not_implemented"}


@get_celery_app().task(name="src.workers.training_tasks.delete_training_files")
def delete_training_files(training_id: str, training_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Delete training files from disk after database deletion.

    This task runs in the background to clean up training files without
    blocking the API response. Emits WebSocket progress updates.

    Args:
        training_id: Training job ID
        training_dir: Path to training directory to delete

    Returns:
        Dictionary with deletion results
    """
    import shutil
    from pathlib import Path
    from .websocket_emitter import emit_deletion_progress

    logger.info(f"Starting file cleanup for training: {training_id}")
    deleted_files = []
    errors = []

    # Emit in_progress status first
    emit_deletion_progress(training_id, "files", "in_progress", "Deleting training files...")

    try:
        # Delete training directory
        if training_dir and Path(training_dir).exists():
            try:
                shutil.rmtree(training_dir)
                deleted_files.append(training_dir)
                logger.info(f"Deleted training directory: {training_dir}")
                # Emit success
                emit_deletion_progress(training_id, "files", "completed", "Deleted training files")
            except Exception as e:
                error_msg = f"Failed to delete training directory {training_dir}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Emit error
                emit_deletion_progress(training_id, "files", "completed", f"Error deleting files: {str(e)}")
        elif training_dir:
            logger.warning(f"Training directory does not exist: {training_dir}")
            # Still emit completion since there's nothing to delete
            emit_deletion_progress(training_id, "files", "completed", "No files to delete")
        else:
            error_msg = "No training directory path provided"
            logger.error(error_msg)
            errors.append(error_msg)
            emit_deletion_progress(training_id, "files", "completed", "No directory path provided")

        result = {
            "training_id": training_id,
            "deleted_files": deleted_files,
            "errors": errors,
        }

        if deleted_files:
            logger.info(f"Successfully deleted {len(deleted_files)} paths for training {training_id}")
        if errors:
            logger.error(f"Encountered {len(errors)} errors during cleanup for training {training_id}")

        return result

    except Exception as e:
        error_msg = f"Failed to delete files for training {training_id}: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        emit_deletion_progress(training_id, "files", "completed", f"Error: {str(e)}")

        return {
            "training_id": training_id,
            "deleted_files": deleted_files,
            "errors": errors,
        }
