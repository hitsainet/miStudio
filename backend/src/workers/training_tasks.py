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
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from celery import Task
from datasets import load_from_disk, concatenate_datasets

from .base_task import DatabaseTask
from ..ml.sparse_autoencoder import create_sae, project_decoder_gradients, JumpReLUSAE, TopKSAE
from ..models.training import Training, TrainingStatus
from ..models.dataset import Dataset
from ..models.model import Model
from ..models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from ..services.training_service import TrainingService
from ..services.checkpoint_service import CheckpointService
from ..core.cancellation import record_progress
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
        fvu: Optional[float] = None,
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
        # THE GUARDED WRITE NOW LIVES IN `core.cancellation`. This method had
        # independently rediscovered the same rule as `jlens_progress.update_row`
        # — a concurrent PAUSED/CANCELLED set by the API must not be clobbered
        # back to RUNNING — and the `training` scope is where that four-status
        # terminal set is now written down once. Training keeps its own
        # dict-return cancellation convention (Feature 21: stopping must run
        # finalize, not unwind); it adopts the registry and the guard only.
        #
        # STRONGER THAN BEFORE IN ONE RESPECT: the whole write is refused on a
        # terminal row, not just the status. Previously a paused run went on
        # accruing steps and losses for up to `status_check_interval` steps
        # after the operator paused it, so the row it was resumed from disagreed
        # with the checkpoint it was resumed at.
        with self.get_db() as db:
            fields = {
                "current_step": step,
                "current_loss": loss,
                "current_l0_sparsity": l0_sparsity,
                "current_dead_neurons": dead_neurons,
                "current_learning_rate": learning_rate,
            }
            # Only overwrite when a value was reported. Architectures that do
            # not compute FVU pass None, and writing that would erase a good
            # reading from a multi-SAE run where one layer reports it.
            if fvu is not None:
                fields["current_fvu"] = fvu
            record_progress(
                "training",
                training_id,
                status=TrainingStatus.RUNNING.value,
                progress=(step / total_steps) * 100.0,
                db=db,
                **fields,
            )

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
    start_step: int = 0,
    checkpoint_id: Optional[str] = None,
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
        start_step: Step to start/resume from (0 = fresh start)
        checkpoint_id: Checkpoint ID to resume from (None = fresh start)

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

        # Get hook types to train on (default to residual for backward compatibility)
        # Supports both old 'hook_type' (string) and new 'hook_types' (list) format
        hook_types_config = hp.get('hook_types', hp.get('hook_type', ['residual']))
        if isinstance(hook_types_config, str):
            hook_types_config = [hook_types_config]
        # Ensure we have at least residual
        if not hook_types_config:
            hook_types_config = ['residual']
        logger.info(f"Training hook types: {hook_types_config}")

        # Create all (layer, hook_type) combinations
        layer_hook_combinations = [
            (layer_idx, hook_type)
            for layer_idx in training_layers
            for hook_type in hook_types_config
        ]
        num_sae_models = len(layer_hook_combinations)
        logger.info(f"Will train {num_sae_models} SAE(s): {len(training_layers)} layers × {len(hook_types_config)} hook types")

        # CRITICAL: Detect actual hidden_dim from cached activations BEFORE creating SAEs
        # This must happen before memory estimation and SAE initialization
        if training.extraction_id is not None:
            logger.info(f"Detecting hidden_dim from cached extraction: {training.extraction_id}")
            from ..models.activation_extraction import ActivationExtraction

            extraction = db.query(ActivationExtraction).filter(
                ActivationExtraction.id == training.extraction_id
            ).first()
            if extraction and extraction.output_path:
                extraction_path = settings.resolve_data_path(extraction.output_path)
                # Find any activation file to peek at its shape
                first_layer = training_layers[0]
                first_hook = hook_types_config[0]
                sample_file = extraction_path / f"layer_{first_layer}_{first_hook}.npy"
                if sample_file.exists():
                    # Peek at file shape without loading full data
                    sample_acts = np.load(sample_file, mmap_mode='r')
                    actual_hidden_dim = sample_acts.shape[2]  # (samples, seq_len, hidden_dim)
                    if hp['hidden_dim'] != actual_hidden_dim:
                        logger.warning(
                            f"HIDDEN_DIM MISMATCH DETECTED: User-provided hidden_dim ({hp['hidden_dim']}) "
                            f"does not match extraction's actual hidden dimension ({actual_hidden_dim}). "
                            f"Overriding to use extraction's actual dimension."
                        )
                        hp['hidden_dim'] = actual_hidden_dim
                    else:
                        logger.info(f"Hidden dim verified: {actual_hidden_dim}")
                    del sample_acts  # Release mmap

    try:
        # Memory budget validation
        logger.info("Validating memory budget...")
        batch_size = hp['batch_size']
        num_layers = len(training_layers)
        num_hook_types = len(hook_types_config)

        # Total number of SAE models = layers × hook_types
        if num_sae_models == 1:
            # Single SAE training
            memory_estimate = estimate_training_memory(
                hidden_dim=hp['hidden_dim'],
                latent_dim=hp['latent_dim'],
                batch_size=batch_size,
            )
        else:
            # Multi-SAE training (multiple layers and/or hook types)
            memory_estimate = estimate_multilayer_training_memory(
                hidden_dim=hp['hidden_dim'],
                latent_dim=hp['latent_dim'],
                batch_size=batch_size,
                num_layers=num_sae_models,  # Total number of SAEs
            )

        available_gpu_gb = memory_estimate.get('available_gpu_gb', 6.0)
        logger.info(f"Estimated memory usage: {memory_estimate['total_gb']:.2f} GB (Available: {available_gpu_gb:.2f} GB)")
        if num_sae_models > 1:
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

        # Initialize models, optimizers, and schedulers (one per layer/hook_type combination)
        logger.info(f"Initializing {num_sae_models} SAE model(s)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        models = {}  # Key: (layer_idx, hook_type)
        optimizers = {}
        schedulers = {}

        # Load framework defaults for this architecture type
        from ..core.framework_defaults import get_framework_defaults
        architecture_type = hp.get('architecture_type', 'standard')
        # Backward compat: map 'standard' to 'standard_saelens'
        if architecture_type == 'standard':
            architecture_type = 'standard_saelens'
        fw = get_framework_defaults(architecture_type)
        logger.info(f"Framework: {fw['display_name']} ({fw['paper']}), sparsity_type={fw['sparsity_type']}")

        for layer_idx, hook_type in layer_hook_combinations:
            # Create SAE for this layer/hook_type combination
            l1_alpha = hp.get('l1_alpha') or fw.get('default_l1_alpha', 5e-4)
            model = create_sae(
                architecture_type=architecture_type,
                hidden_dim=hp['hidden_dim'],
                latent_dim=hp['latent_dim'],
                l1_alpha=l1_alpha,
                ghost_gradient_penalty=hp.get('ghost_gradient_penalty', 0.0),
                normalize_activations=hp.get('normalize_activations', fw['normalize_activations']),
                top_k_sparsity=hp.get('top_k_sparsity', None),
                # TopK-specific parameters
                top_k=hp.get('top_k'),
                aux_k=hp.get('aux_k'),
                aux_loss_alpha=hp.get('aux_loss_alpha'),
                # JumpReLU-specific parameters
                initial_threshold=hp.get('initial_threshold', 0.5),
                bandwidth=hp.get('bandwidth', 0.01),
                sparsity_coeff=hp.get('sparsity_coeff'),
                normalize_decoder=hp.get('normalize_decoder', fw['normalize_decoder']),
            ).to(device)
            models[(layer_idx, hook_type)] = model

            # Initialize optimizer using framework defaults
            adam_betas = fw['optimizer_betas']
            adam_eps = hp.get('adam_epsilon') or fw['adam_epsilon']

            optimizer = optim.Adam(
                model.parameters(),
                lr=hp['learning_rate'],
                weight_decay=hp.get('weight_decay', fw['weight_decay']),
                betas=adam_betas,
                eps=adam_eps,
            )
            optimizers[(layer_idx, hook_type)] = optimizer

            # Learning rate scheduler (linear warmup + constant)
            warmup_steps = hp.get('warmup_steps', 0)

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return 1.0

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            schedulers[(layer_idx, hook_type)] = scheduler

            # Log model details
            if isinstance(model, TopKSAE):
                logger.info(f"  Layer {layer_idx}/{hook_type}: TopKSAE — K={model.k}, aux_k={model.aux_k}, alpha={model.aux_loss_alpha}")
            elif isinstance(model, JumpReLUSAE):
                logger.info(f"  Layer {layer_idx}/{hook_type}: JumpReLUSAE — sparsity_coeff={model.sparsity_coeff}")
            else:
                logger.info(f"  Layer {layer_idx}/{hook_type}: {architecture_type} — l1_alpha={l1_alpha}")

        # Initialize gradient scalers for mixed precision training (one per layer/hook_type)
        scalers = {}
        if torch.cuda.is_available():
            for layer_idx, hook_type in layer_hook_combinations:
                scalers[(layer_idx, hook_type)] = GradScaler()
            logger.info(f"Mixed precision training (FP16) enabled with {len(scalers)} GradScaler(s)")
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

        # Load checkpoint weights if resuming
        if start_step > 0 and checkpoint_id:
            logger.info(f"Resuming from checkpoint {checkpoint_id} at step {start_step}")
            with self.get_db() as db:
                from ..models.checkpoint import Checkpoint
                ckpt = db.query(Checkpoint).filter_by(id=checkpoint_id).first()
                if not ckpt:
                    raise ValueError(f"Checkpoint not found: {checkpoint_id}")
                ckpt_path = Path(settings.resolve_data_path(ckpt.storage_path))

            # ckpt_path is e.g. .../checkpoints/checkpoint_{step}/layer_{idx}_{hook}/checkpoint.safetensors
            # so ckpt_path.parent.parent is the per-step directory already.
            checkpoint_step_dir = ckpt_path.parent.parent
            for (layer_idx, hook_type), model in models.items():
                # Try new naming (layer_{idx}_{hook_type}) then legacy (layer_{idx})
                layer_ckpt = checkpoint_step_dir / f"layer_{layer_idx}_{hook_type}" / "checkpoint.safetensors"
                if not layer_ckpt.exists():
                    layer_ckpt = checkpoint_step_dir / f"layer_{layer_idx}" / "checkpoint.safetensors"
                if not layer_ckpt.exists():
                    layer_ckpt = ckpt_path  # Final fallback: the specific layer from the DB record
                if layer_ckpt.exists():
                    CheckpointService.load_checkpoint(str(layer_ckpt), model=model, device=str(device))
                    logger.info(f"Loaded checkpoint for layer {layer_idx}/{hook_type} from {layer_ckpt}")
                else:
                    logger.warning(f"Checkpoint file not found for layer {layer_idx}/{hook_type}: {layer_ckpt}")

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
        # Support multi-extraction: extraction_ids (list) takes precedence over extraction_id (singular)
        extraction_ids = training.extraction_ids if training.extraction_ids else (
            [training.extraction_id] if training.extraction_id else None
        )
        use_cached_activations = extraction_ids is not None and len(extraction_ids) > 0
        cached_activations = {}
        use_gpu_activations = True  # Default; may be set to False for cached activations on large models
        dataset = None
        base_model = None
        tokenizer = None
        architecture = None
        hook_types = None

        if use_cached_activations:
            from ..models.activation_extraction import ActivationExtraction

            logger.info(f"Using cached activations from {len(extraction_ids)} extraction(s): {extraction_ids}")

            # Load and validate all extractions
            extractions = []
            with self.get_db() as db:
                for ext_id in extraction_ids:
                    extraction = db.query(ActivationExtraction).filter(
                        ActivationExtraction.id == ext_id
                    ).first()
                    if not extraction:
                        raise ValueError(f"Extraction {ext_id} not found")
                    if extraction.status != "completed":
                        raise ValueError(f"Extraction {ext_id} is not completed (status: {extraction.status})")
                    extractions.append(extraction)

            # Load activations from each extraction and concatenate
            all_activation_parts = {}  # {(layer, hook): [tensor1, tensor2, ...]}
            actual_hidden_dim = None

            for ext_idx, extraction in enumerate(extractions):
                extraction_path = settings.resolve_data_path(extraction.output_path)
                logger.info(f"Loading extraction {ext_idx + 1}/{len(extractions)}: {extraction.id} (dataset: {extraction.dataset_id})")

                # Load metadata
                metadata_path = extraction_path / "metadata.json"
                with open(metadata_path, 'r') as f:
                    extraction_metadata = json.load(f)

                logger.info(f"  Metadata: {extraction_metadata['num_samples_processed']} samples")

                # Validate layer coverage
                extraction_layers = set(extraction_metadata.get('layer_indices', []))
                requested_layers = set(hp.get('training_layers', []))
                missing_layers = requested_layers - extraction_layers
                if missing_layers:
                    raise ValueError(
                        f"Extraction {extraction.id} is missing layers {sorted(missing_layers)}. "
                        f"Available: {sorted(extraction_layers)}. Requested: {sorted(requested_layers)}."
                    )

                # Validate hook types
                available_hook_types = extraction_metadata.get('hook_types', ['residual'])
                for ht in hook_types_config:
                    if ht not in available_hook_types:
                        raise ValueError(
                            f"Extraction {extraction.id}: hook_type '{ht}' not available. "
                            f"Available: {available_hook_types}."
                        )

                # Load activation files for each (layer, hook_type) combination
                for layer_idx, hook_type in layer_hook_combinations:
                    activation_file = extraction_path / f"layer_{layer_idx}_{hook_type}.npy"
                    if not activation_file.exists():
                        available_files = list(extraction_path.glob(f"layer_{layer_idx}_*.npy"))
                        available_types = [f.stem.split('_')[-1] for f in available_files]
                        raise ValueError(
                            f"Activation file not found: {activation_file}. "
                            f"Available for layer {layer_idx}: {available_types}"
                        )

                    logger.info(f"  Loading layer {layer_idx}/{hook_type} from {activation_file}")
                    layer_acts_mmap = np.load(activation_file, mmap_mode='r')

                    num_samples_in_file, seq_len, hidden_dim = layer_acts_mmap.shape
                    if actual_hidden_dim is None:
                        actual_hidden_dim = hidden_dim
                    elif hidden_dim != actual_hidden_dim:
                        raise ValueError(
                            f"Extraction {extraction.id} has hidden_dim={hidden_dim} but expected {actual_hidden_dim}"
                        )

                    total_tokens = num_samples_in_file * seq_len
                    key = (layer_idx, hook_type)
                    if key not in all_activation_parts:
                        all_activation_parts[key] = []
                    all_activation_parts[key].append({
                        'file': activation_file,
                        'num_samples': num_samples_in_file,
                        'seq_len': seq_len,
                        'total_tokens': total_tokens,
                    })
                    logger.info(f"    Shape: {layer_acts_mmap.shape} = {total_tokens:,} token activations")

            # Determine GPU memory budget for cached activations
            # SAE models are already on GPU. Use actual free memory for accurate budgeting.
            num_layer_hooks = len(layer_hook_combinations)
            latent_dim = hp['latent_dim']

            # Get ACTUAL free GPU memory (accounts for CUDA context, driver, loaded SAE models)
            try:
                gpu_free, gpu_total = torch.cuda.mem_get_info(device)
            except Exception:
                gpu_free = 16 * 1024**3
                gpu_total = 24 * 1024**3

            # SAE models are already loaded. Pending allocations (happen lazily during training):
            # - Adam optimizer: 2 momentum buffers per param × float32
            # - Gradients: 1 buffer per param × float32
            # Total pending: 3 × param_bytes per SAE
            sae_params_per_model = 2 * actual_hidden_dim * latent_dim  # encoder + decoder weights
            pending_sae_bytes = num_layer_hooks * sae_params_per_model * 4 * 3  # 3 buffers × float32

            # Training overhead: batch forward/backward intermediates + fragmentation margin
            batch_intermediate_bytes = (
                batch_size * (actual_hidden_dim + latent_dim) * 4  # fwd tensors per SAE
                * num_layer_hooks * 3  # all SAEs × (fwd + bwd + margin)
            )
            training_overhead = max(1 * 1024**3, batch_intermediate_bytes)  # at least 1 GB

            gpu_mem_for_activations = gpu_free - pending_sae_bytes - training_overhead

            logger.info(
                f"GPU memory budget: {gpu_total / 1024**3:.1f} GB total, "
                f"{gpu_free / 1024**3:.2f} GB free (after SAE models + CUDA context), "
                f"{pending_sae_bytes / 1024**3:.2f} GB pending (optimizer+gradients), "
                f"{training_overhead / 1024**3:.2f} GB training overhead, "
                f"{max(0, gpu_mem_for_activations) / 1024**3:.2f} GB for activations"
            )

            bytes_per_token = actual_hidden_dim * 4  # float32

            # Count total available tokens across all extractions (use first layer as reference)
            first_key = layer_hook_combinations[0]
            total_available_tokens = sum(p['total_tokens'] for p in all_activation_parts[first_key])

            # Decide activation storage: GPU (fast indexing) vs CPU (memory-safe, batch streaming)
            # GPU path: all activations on GPU, instant batch sampling
            # CPU path: activations in pinned CPU memory, batch transferred per step (~5-10ms overhead)
            min_useful_tokens = min(total_available_tokens, max(50_000, batch_size * 20))
            max_gpu_tokens_per_layer = max(0, int(gpu_mem_for_activations * 0.9 / bytes_per_token / num_layer_hooks))

            if max_gpu_tokens_per_layer >= min_useful_tokens:
                # GPU path: load (possibly subsampled) activations to GPU
                use_gpu_activations = True
                tokens_to_load = min(total_available_tokens, max_gpu_tokens_per_layer)
                logger.info(
                    f"Activation storage: GPU — {tokens_to_load:,} / {total_available_tokens:,} "
                    f"tokens per layer ({tokens_to_load / total_available_tokens * 100:.1f}%)"
                )
            else:
                # CPU path: keep all activations on CPU, stream batches to GPU per step
                use_gpu_activations = False
                tokens_to_load = total_available_tokens
                logger.info(
                    f"Activation storage: CPU (streaming) — GPU can hold {max_gpu_tokens_per_layer:,} "
                    f"tokens/layer but need {min_useful_tokens:,} minimum. "
                    f"Loading all {tokens_to_load:,} tokens to CPU with batch streaming."
                )

            # Generate a shared random token selection (same tokens for all layers)
            # This ensures all layers see the same token positions for consistency
            if tokens_to_load < total_available_tokens:
                rng = np.random.RandomState(42)
                selected_indices = np.sort(rng.choice(total_available_tokens, size=tokens_to_load, replace=False))
                logger.info(f"Subsampled {tokens_to_load:,} random token positions (seed=42)")
            else:
                selected_indices = None  # Load all tokens

            # Load and flatten activations: (N, seq_len, d) -> (N*seq_len, d) per-token
            # SAE training requires individual token activations, NOT sequence averages.
            # Averaging over the sequence dimension destroys per-token variance and causes
            # degenerate training where b_dec alone explains the data (L0->0, all features die).
            for key in layer_hook_combinations:
                parts_info = all_activation_parts[key]

                # Build a flat array of all token activations from all extractions
                all_flat_parts = []
                cumulative_offset = 0
                for part_info in parts_info:
                    mmap = np.load(part_info['file'], mmap_mode='r')
                    n, s, d = mmap.shape

                    if selected_indices is not None:
                        # Determine which indices fall within this file's range
                        file_end = cumulative_offset + part_info['total_tokens']
                        local_mask = (selected_indices >= cumulative_offset) & (selected_indices < file_end)
                        local_indices = selected_indices[local_mask] - cumulative_offset
                        cumulative_offset = file_end

                        if len(local_indices) == 0:
                            continue

                        # Convert flat indices to (sample, token) pairs
                        sample_indices = local_indices // s
                        token_indices = local_indices % s

                        # Load sample-by-sample in sorted order for sequential mmap access
                        unique_samples = np.unique(sample_indices)
                        chunk_size = 200  # samples per mmap read
                        chunk_parts = []
                        for ci in range(0, len(unique_samples), chunk_size):
                            ci_end = min(ci + chunk_size, len(unique_samples))
                            sample_batch = unique_samples[ci:ci_end]
                            s_min, s_max = sample_batch[0], sample_batch[-1]
                            # Read contiguous range from mmap (fast sequential I/O)
                            block = mmap[s_min:s_max + 1]  # (range, seq_len, d)
                            # Extract the specific (sample, token) pairs from this block
                            for s_idx in sample_batch:
                                token_mask = sample_indices == s_idx
                                toks = token_indices[token_mask]
                                chunk_parts.append(block[s_idx - s_min, toks].astype(np.float32))
                        if chunk_parts:
                            all_flat_parts.append(np.concatenate(chunk_parts, axis=0))
                    else:
                        # Load all tokens — flatten (N, seq_len, d) to (N*seq_len, d) in chunks
                        cumulative_offset += part_info['total_tokens']
                        chunk_size = 500  # samples at a time
                        chunk_parts = []
                        for ci in range(0, n, chunk_size):
                            ci_end = min(ci + chunk_size, n)
                            chunk = mmap[ci:ci_end].reshape(-1, d).astype(np.float32)
                            chunk_parts.append(chunk)
                        all_flat_parts.append(np.concatenate(chunk_parts, axis=0))

                if all_flat_parts:
                    combined = np.concatenate(all_flat_parts, axis=0) if len(all_flat_parts) > 1 else all_flat_parts[0]
                    if use_gpu_activations:
                        cached_activations[key] = torch.from_numpy(combined).to(device)
                        storage_label = "GPU"
                    else:
                        # CPU path: pinned memory for faster GPU transfers during batch streaming
                        tensor = torch.from_numpy(np.ascontiguousarray(combined))
                        try:
                            tensor = tensor.pin_memory()
                            storage_label = "CPU (pinned)"
                        except RuntimeError:
                            storage_label = "CPU"
                        cached_activations[key] = tensor
                        del combined  # free numpy copy
                else:
                    raise ValueError(f"No activations loaded for {key}")

                logger.info(f"  {key}: {cached_activations[key].shape} on {storage_label} ({cached_activations[key].nbytes / 1024**3:.2f} GB)")

            # Get sample count and hidden dimension
            first_key = layer_hook_combinations[0]
            num_samples = cached_activations[first_key].shape[0]

            # Override hidden_dim with actual dimension from extraction
            if hp['hidden_dim'] != actual_hidden_dim:
                logger.warning(
                    f"User-provided hidden_dim ({hp['hidden_dim']}) != extraction ({actual_hidden_dim}). Using extraction's."
                )
                hp['hidden_dim'] = actual_hidden_dim

            logger.info(
                f"Cached activations ready: {num_samples:,} token activations from "
                f"{len(extractions)} extraction(s), hidden_dim={actual_hidden_dim}"
            )

        else:
            # Load dataset(s) and base model for on-the-fly activation extraction
            # Supports multiple datasets via training.dataset_ids
            dataset_ids = training.dataset_ids if training.dataset_ids else [training.dataset_id]
            logger.info(f"Loading {len(dataset_ids)} dataset(s) and base model for activation extraction...")

            datasets_to_concat = []
            first_tokenization = None

            with self.get_db() as db:
                model_record = db.query(Model).filter(
                    Model.id == training.model_id
                ).first()
                if not model_record:
                    raise ValueError(f"Model {training.model_id} not found")

                # Load each dataset and its tokenization
                for ds_id in dataset_ids:
                    dataset_record = db.query(Dataset).filter(
                        Dataset.id == ds_id
                    ).first()
                    if not dataset_record:
                        raise ValueError(f"Dataset {ds_id} not found")

                    # Query the tokenization for this dataset + model combination
                    tokenization = db.query(DatasetTokenization).filter(
                        DatasetTokenization.dataset_id == ds_id,
                        DatasetTokenization.model_id == training.model_id
                    ).first()
                    if not tokenization:
                        raise ValueError(
                            f"No tokenization found for dataset {ds_id} with model {training.model_id}. "
                            f"Please tokenize the dataset with this model first."
                        )
                    if tokenization.status != TokenizationStatus.READY:
                        raise ValueError(
                            f"Tokenization for dataset {ds_id} with model {training.model_id} "
                            f"is not ready (status: {tokenization.status}). Please wait for tokenization to complete."
                        )

                    # Store first tokenization for vocab validation
                    if first_tokenization is None:
                        first_tokenization = tokenization

                    # Load the tokenized dataset
                    resolved_tokenized_path = str(settings.resolve_data_path(tokenization.tokenized_path))
                    logger.info(f"Loading dataset {ds_id} from {resolved_tokenized_path}")
                    ds = load_from_disk(resolved_tokenized_path)
                    datasets_to_concat.append(ds)
                    logger.info(f"  - {ds_id}: {len(ds)} samples")

            # Concatenate datasets if multiple
            if len(datasets_to_concat) == 1:
                dataset = datasets_to_concat[0]
            else:
                logger.info(f"Concatenating {len(datasets_to_concat)} datasets...")
                dataset = concatenate_datasets(datasets_to_concat)
                logger.info(f"Combined dataset: {len(dataset)} total samples")

            # Use first tokenization for vocab validation (all should match since same model)
            tokenization = first_tokenization

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

            # Determine hook types for on-the-fly extraction
            # Use hook_types from hyperparameters (already validated at the top of the function)
            hook_type_map = {
                'residual': HookType.RESIDUAL,
                'mlp': HookType.MLP,
                'attention': HookType.ATTENTION,
            }
            hook_types = [hook_type_map.get(ht, HookType.RESIDUAL) for ht in hook_types_config]
            logger.info(f"Using hook_types {hook_types_config} for on-the-fly activation extraction")

            num_samples = len(dataset)
            logger.info(f"Dataset: {num_samples} samples, Model: {model_record.repo_id}")
        logger.info(f"Starting training loop: {total_steps} steps, batch_size={batch_size}")

        # Initialize decoder bias (b_dec) to the mean of NORMALIZED data.
        # This is critical for the centering formulation: encode(normalize(x) - b_dec).
        # Without this, b_dec starts at 0 and the encoder must learn the data mean
        # during training, causing massive feature death in early steps.
        # SAELens initializes b_dec to the geometric median; we use the mean
        # (faster, equivalent for high-dimensional data).
        #
        # IMPORTANT: Since normalization (constant_norm_rescale / anthropic_rescale)
        # is applied BEFORE centering in the forward pass, b_dec must be initialized
        # from the NORMALIZED activations, not the raw activations.
        if use_cached_activations:
            logger.info("Initializing decoder bias (b_dec) to normalized data mean for proper centering...")
            for sae_key, model in models.items():
                layer_idx, hook_type = sae_key
                raw_acts = cached_activations[sae_key]

                # Compute mean in the NORMALIZED space (matching forward pass order)
                # The model normalizes BEFORE centering, so b_dec must be in normalized space
                if hasattr(model, 'normalize'):
                    # Use the model's own normalize method for consistency
                    # Process in chunks to avoid memory spikes (works for both GPU and CPU activations)
                    chunk_size = 4096
                    n_samples = raw_acts.shape[0]
                    running_sum = torch.zeros(raw_acts.shape[1], device=device)
                    for i in range(0, n_samples, chunk_size):
                        chunk = raw_acts[i:i+chunk_size]
                        if chunk.device != device:
                            chunk = chunk.to(device)
                        normed_chunk, _ = model.normalize(chunk)
                        running_sum += normed_chunk.sum(dim=0)
                    data_mean = running_sum / n_samples
                    logger.info(f"  L{layer_idx}/{hook_type}: computed mean from normalized activations")
                else:
                    # Fallback: use raw mean if model has no normalize method
                    data_mean = raw_acts.mean(dim=0)
                    if data_mean.device != device:
                        data_mean = data_mean.to(device)
                    logger.info(f"  L{layer_idx}/{hook_type}: no normalize method, using raw mean")

                with torch.no_grad():
                    if hasattr(model, 'b_pre'):
                        # TopKSAE: b_pre is the centering bias
                        model.b_pre.data = data_mean
                        logger.info(f"  L{layer_idx}/{hook_type}: b_pre initialized (norm={data_mean.norm().item():.4f})")
                    elif hasattr(model, 'b_dec'):
                        # JumpReLUSAE: b_dec is the centering bias
                        model.b_dec.data = data_mean
                        logger.info(f"  L{layer_idx}/{hook_type}: b_dec initialized (norm={data_mean.norm().item():.4f})")
                    elif hasattr(model, 'decoder_bias'):
                        # SparseAutoencoder / SkipAutoencoder: decoder_bias is b_dec
                        model.decoder_bias.data = data_mean
                        logger.info(f"  L{layer_idx}/{hook_type}: decoder_bias initialized (norm={data_mean.norm().item():.4f})")

        # Training loop
        best_loss = float('inf')

        # Sparsity warmup: store base sparsity coefficients before training
        # TopK has structural sparsity (no penalty to warm up)
        sparsity_type = fw['sparsity_type']
        sparsity_warmup_steps = hp.get('sparsity_warmup_steps', fw.get('sparsity_warmup_steps', 0))
        if sparsity_type == 'topk':
            sparsity_warmup_steps = 0  # TopK: no sparsity penalty to warm up
        base_sparsity_coeffs = {}
        for sae_key, model in models.items():
            if sparsity_type == 'l0':
                base_sparsity_coeffs[sae_key] = model.sparsity_coeff
            elif sparsity_type == 'l1':
                base_sparsity_coeffs[sae_key] = model.l1_alpha
            # topk: no sparsity coefficient to store
        if sparsity_warmup_steps > 0:
            logger.info(f"Sparsity warmup enabled ({sparsity_type}): ramping from 0 to full over {sparsity_warmup_steps} steps")

        # Dead neuron tracking: exponential moving average of per-feature activation frequency
        # This is more reliable than per-batch detection which can miss infrequent features
        feature_activation_ema = {}  # Key: sae_key, Value: tensor [latent_dim]
        ema_window_tokens = 50000  # Approximate token window for EMA decay

        # Data-driven threshold calibration for JumpReLU
        # Sets thresholds from actual pre-activation distribution so features start active
        if architecture_type == 'jumprelu' and use_cached_activations:
            target_l0_frac = hp.get('target_l0', 0.05) or 0.05
            cal_size = min(4096, num_samples)
            logger.info(f"JumpReLU threshold calibration: sampling {cal_size} activations (target L0: {target_l0_frac*100:.1f}%)")
            for sae_key, model in models.items():
                layer_idx, hook_type = sae_key
                cal_indices = torch.randint(0, num_samples, (cal_size,))
                cal_batch = cached_activations[sae_key][cal_indices]
                if cal_batch.device != device:
                    cal_batch = cal_batch.to(device)
                thresholds = model.calibrate_thresholds(cal_batch, target_l0_frac)
                actual_l0 = ((cal_batch @ model.W_enc.T + model.b_enc) > thresholds.unsqueeze(0)).float().mean().item()
                logger.info(
                    f"  L{layer_idx}/{hook_type}: threshold mean={thresholds.mean().item():.4f}, "
                    f"range=[{thresholds.min().item():.4f}, {thresholds.max().item():.4f}], "
                    f"calibrated L0={actual_l0:.4f}"
                )
            # Update base_sparsity_coeffs since model params may have changed
            for sae_key, model in models.items():
                base_sparsity_coeffs[sae_key] = model.sparsity_coeff

        # ======================================================================
        # PRE-TRAINING DIAGNOSTIC: Run one forward pass to check loss components
        # ======================================================================
        if use_cached_activations:
            logger.info("=" * 70)
            logger.info("PRE-TRAINING DIAGNOSTIC: Initial loss decomposition")
            logger.info("=" * 70)
            with torch.no_grad():
                diag_key = layer_hook_combinations[0]
                diag_model = models[diag_key]
                diag_batch = cached_activations[diag_key][:min(batch_size, 256)]
                if diag_batch.device != device:
                    diag_batch = diag_batch.to(device)
                _, diag_z, diag_losses = diag_model(diag_batch, return_loss=True)
                diag_recon = diag_losses.get('loss_reconstruction')
                diag_l0_sparsity = diag_losses.get('l0_sparsity')
                diag_total = diag_losses.get('loss')
                logger.info(f"  Layer {diag_key[0]}/{diag_key[1]}:")
                logger.info(f"    Total loss:          {diag_total.item():.6f}")
                if diag_recon is not None:
                    logger.info(f"    Reconstruction loss: {diag_recon.item():.6f}")
                if sparsity_type == 'l0':
                    diag_loss_l0 = diag_losses.get('loss_l0')
                    coeff_val = getattr(diag_model, 'sparsity_coeff', 0.0)
                    if diag_loss_l0 is not None:
                        logger.info(f"    L0 loss (weighted):  {diag_loss_l0.item():.6f}")
                    logger.info(f"    Sparsity coeff:      {coeff_val}")
                elif sparsity_type == 'l1':
                    diag_l1_raw = diag_losses.get('l1_penalty')
                    l1_alpha_val = getattr(diag_model, 'l1_alpha', 0.0)
                    if diag_l1_raw is not None:
                        logger.info(f"    L1 penalty (raw):    {diag_l1_raw.item():.4f}")
                        logger.info(f"    L1 alpha (current):  {l1_alpha_val}")
                        logger.info(f"    L1 loss (weighted):  {l1_alpha_val * diag_l1_raw.item():.6f}")
                if diag_l0_sparsity is not None:
                    logger.info(f"    L0 sparsity:         {diag_l0_sparsity.item():.4f} ({diag_l0_sparsity.item()*100:.1f}% features active)")
                logger.info(f"    Active features:     {(diag_z != 0).any(dim=0).sum().item()}/{hp['latent_dim']}")
                logger.info(f"    Batch z stats:       mean={diag_z[diag_z>0].mean().item():.4f}, max={diag_z.max().item():.4f}")
                if sparsity_warmup_steps > 0:
                    if sparsity_type == 'l0':
                        coeff_val = getattr(diag_model, 'sparsity_coeff', 0.0)
                        logger.info(f"    Sparsity warmup:     L0 coeff will ramp from 0 to {coeff_val} over {sparsity_warmup_steps} steps")
                    elif sparsity_type == 'l1':
                        l1_alpha_val = getattr(diag_model, 'l1_alpha', 0.0)
                        logger.info(f"    Sparsity warmup:     L1 alpha will ramp from 0 to {l1_alpha_val} over {sparsity_warmup_steps} steps")
            logger.info("=" * 70)

        # Pause/cancel checks hit the database, so throttle them: a per-step
        # query adds up to total_steps round-trips on the GPU hot path.
        status_check_interval = min(25, max(1, log_interval))

        for step in range(start_step, total_steps):
            # Check for pause/stop signals (throttled)
            if step % status_check_interval == 0:
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
                    # Sample from cached activations (GPU or CPU depending on memory budget)
                    batch_indices = torch.randint(0, num_samples, (batch_size,))

                    for layer_idx, hook_type in layer_hook_combinations:
                        cached = cached_activations[(layer_idx, hook_type)]
                        batch = cached[batch_indices]
                        # Transfer to GPU if activations are on CPU (streaming mode)
                        if batch.device != device:
                            batch = batch.to(device, non_blocking=True)
                        layer_activations[(layer_idx, hook_type)] = batch

                        # VALIDATION: Check activation statistics on first step
                        if step == 1:
                            act_mean = cached.mean().item()
                            act_std = cached.std().item()
                            act_min = cached.min().item()
                            act_max = cached.max().item()
                            logger.info(f"Layer {layer_idx}/{hook_type} cached activations sampled successfully:")
                            logger.info(f"  Cached shape on {cached.device}: {cached.shape}")
                            logger.info(f"  Mean: {act_mean:.4f}, Std: {act_std:.4f}")
                            logger.info(f"  Range: [{act_min:.4f}, {act_max:.4f}]")

                            # Sanity check
                            if act_std < 0.01 or act_std > 100:
                                logger.error(f"SUSPICIOUS: Layer {layer_idx}/{hook_type} std={act_std:.4f} is unusual!")
                            if abs(act_mean) > 50:
                                logger.error(f"SUSPICIOUS: Layer {layer_idx}/{hook_type} mean={act_mean:.4f} is unusual!")

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

                        # Get captured activations for each (layer, hook_type) combination
                        for layer_idx, hook_type in layer_hook_combinations:
                            # Key format is "layer_{idx}_{hook_type}" (e.g., "layer_9_residual")
                            layer_key = f"layer_{layer_idx}_{hook_type}"

                            if layer_key in hook_manager.activations and hook_manager.activations[layer_key]:
                                acts = hook_manager.activations[layer_key][0]  # Shape: (batch_size, seq_len, hidden_dim)
                                # Flatten to per-token activations: (batch_size, seq_len, hidden_dim) → (batch_size*seq_len, hidden_dim)
                                # Then randomly subsample to batch_size tokens.
                                # This preserves per-token variance instead of averaging over the sequence.
                                acts_flat = acts.detach().reshape(-1, acts.shape[-1])  # (batch*seq, hidden_dim)
                                if acts_flat.shape[0] > batch_size:
                                    token_indices = torch.randperm(acts_flat.shape[0], device=acts_flat.device)[:batch_size]
                                    acts_flat = acts_flat[token_indices]
                                layer_activations[(layer_idx, hook_type)] = acts_flat.to(device)

                                # VALIDATION: Check activation statistics on first step
                                if step == 1:
                                    act_mean = acts_flat.mean().item()
                                    act_std = acts_flat.std().item()
                                    act_min = acts_flat.min().item()
                                    act_max = acts_flat.max().item()
                                    logger.info(f"Layer {layer_idx}/{hook_type} activations captured successfully:")
                                    logger.info(f"  Shape: {acts.shape} → flattened to {acts_flat.shape}")
                                    logger.info(f"  Mean: {act_mean:.4f}, Std: {act_std:.4f}")
                                    logger.info(f"  Range: [{act_min:.4f}, {act_max:.4f}]")

                                    # Sanity check: Real activations should have reasonable statistics
                                    if act_std < 0.01 or act_std > 100:
                                        logger.error(f"SUSPICIOUS: Layer {layer_idx}/{hook_type} std={act_std:.4f} is unusual!")
                                    if abs(act_mean) > 50:
                                        logger.error(f"SUSPICIOUS: Layer {layer_idx}/{hook_type} mean={act_mean:.4f} is unusual!")
                            else:
                                # CRITICAL ERROR: No activations captured means hooks failed
                                logger.error(f"FATAL: No activations captured for layer {layer_idx}/{hook_type}")
                                logger.error(f"Available keys: {list(hook_manager.activations.keys())}")
                                logger.error(f"Expected key: {layer_key}")
                                raise RuntimeError(
                                    f"Failed to capture activations for layer {layer_idx}/{hook_type}. "
                                    f"Hook registration failed. Available keys: {list(hook_manager.activations.keys())}"
                                )

                # Data-driven threshold calibration for on-the-fly mode (step 0 only)
                if step == 0 and architecture_type == 'jumprelu' and not use_cached_activations:
                    target_l0_frac = hp.get('target_l0', 0.05) or 0.05
                    logger.info(f"JumpReLU threshold calibration from first batch (target L0: {target_l0_frac*100:.1f}%)")
                    for sae_key, model in models.items():
                        x_cal = layer_activations[sae_key]
                        thresholds = model.calibrate_thresholds(x_cal, target_l0_frac)
                        logger.info(
                            f"  {sae_key}: threshold mean={thresholds.mean().item():.4f}, "
                            f"range=[{thresholds.min().item():.4f}, {thresholds.max().item():.4f}]"
                        )
                    for sae_key, model in models.items():
                        base_sparsity_coeffs[sae_key] = model.sparsity_coeff

                # Train all SAEs (one per layer/hook_type combination)
                layer_losses = {}  # Key: (layer_idx, hook_type)
                layer_recon_losses = {}  # Reconstruction loss component
                layer_l1_losses = {}  # Sparsity loss: L1*alpha for L1 types, loss_l0 for JumpReLU
                layer_sparsities = {}
                layer_dead_neurons = {}
                layer_fvu = {}

                # Apply sparsity warmup: linearly scale L1/L0 penalty from 0 to full
                # TopK: no warmup (structural sparsity, sparsity_warmup_steps forced to 0)
                if sparsity_warmup_steps > 0:
                    sparsity_scale = min(1.0, step / sparsity_warmup_steps)
                    for sae_key, model_ref in models.items():
                        base_coeff = base_sparsity_coeffs.get(sae_key)
                        if base_coeff is not None:
                            if sparsity_type == 'l0':
                                model_ref.sparsity_coeff = base_coeff * sparsity_scale
                            elif sparsity_type == 'l1':
                                model_ref.l1_alpha = base_coeff * sparsity_scale

                for layer_idx, hook_type in layer_hook_combinations:
                    sae_key = (layer_idx, hook_type)
                    x = layer_activations[sae_key]
                    model = models[sae_key]
                    optimizer = optimizers[sae_key]
                    scheduler = schedulers[sae_key]
                    scaler = scalers.get(sae_key)  # None if CPU training

                    # Forward pass
                    if step % grad_accum_steps == 0:
                        optimizer.zero_grad()

                    # Forward pass with mixed precision (FP16) if GPU available
                    is_transcoder = (architecture_type == 'transcoder')
                    if scaler is not None:
                        with autocast():
                            if is_transcoder:
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
                        if is_transcoder:
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
                            # Mixed precision: unscale gradients before clipping.
                            # CRITICAL: Once unscale_() is called, update() MUST follow
                            # before the next unscale_() call, even if an error occurs
                            # (e.g. OOM during step/clip). Use try/finally to guarantee this.
                            scaler.unscale_(optimizer)
                            try:
                                # Gradient clipping
                                grad_clip_norm = hp.get('grad_clip_norm')
                                if grad_clip_norm:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                                # JumpReLU: Project decoder gradients orthogonal to decoder columns
                                if isinstance(model, JumpReLUSAE):
                                    project_decoder_gradients(model)

                                scaler.step(optimizer)
                            finally:
                                # Always call update() to reset scaler state to READY.
                                # Without this, an OOM between unscale_() and update()
                                # leaves the scaler in UNSCALED state, causing
                                # "unscale_() has already been called" on the next step.
                                scaler.update()
                        else:
                            # CPU training - no mixed precision
                            grad_clip_norm = hp.get('grad_clip_norm')
                            if grad_clip_norm:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                            if isinstance(model, JumpReLUSAE):
                                project_decoder_gradients(model)

                            optimizer.step()

                        # Normalize decoder columns to unit norm after each step.
                        # Critical for L1-based SAEs: without this, a few features develop
                        # large decoder norms and dominate reconstruction, making other
                        # features expendable under L1 penalty → cascade feature death.
                        if hp.get('normalize_decoder', True):
                            if isinstance(model, JumpReLUSAE):
                                model.normalize_decoder()
                            elif isinstance(model, TopKSAE):
                                # TopK: normalize decoder weight columns
                                with torch.no_grad():
                                    model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0, p=2)
                            elif hasattr(model, 'decoder') and model.decoder is not None:
                                # SparseAutoencoder / SkipAutoencoder: decoder.weight is [hidden_dim, latent_dim]
                                with torch.no_grad():
                                    model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0, p=2)

                        scheduler.step()

                    # Store SAE metrics (keyed by (layer_idx, hook_type) tuple)
                    layer_losses[sae_key] = loss.item() * grad_accum_steps  # Undo accumulation scaling
                    layer_sparsities[sae_key] = (z != 0).float().mean().item()

                    # Extract loss components for detailed logging
                    recon_loss_val = losses.get('loss_reconstruction')
                    layer_recon_losses[sae_key] = recon_loss_val.item() if recon_loss_val is not None and hasattr(recon_loss_val, 'item') else None

                    # Extract sparsity loss: L0 for JumpReLU, L1 for standard/skip/transcoder
                    loss_l0_val = losses.get('loss_l0')
                    l1_penalty_val = losses.get('l1_penalty')
                    if loss_l0_val is not None and hasattr(loss_l0_val, 'item'):
                        # JumpReLU: loss_l0 is already the weighted penalty (sparsity_coeff * l0_fraction)
                        layer_l1_losses[sae_key] = loss_l0_val.item()
                    elif l1_penalty_val is not None and hasattr(l1_penalty_val, 'item'):
                        # L1-based SAEs: weighted L1 loss = l1_alpha * raw_penalty
                        current_l1_alpha = getattr(model, 'l1_alpha', 0.0)
                        layer_l1_losses[sae_key] = current_l1_alpha * l1_penalty_val.item()
                    else:
                        layer_l1_losses[sae_key] = None

                    # Update EMA-based dead neuron tracking
                    with torch.no_grad():
                        fired = (z > 0).any(dim=0).float()  # [latent_dim] — 1 if any sample activated this feature
                        if sae_key not in feature_activation_ema:
                            feature_activation_ema[sae_key] = torch.zeros(hp['latent_dim'], device=z.device)
                        ema_decay = max(0.0, 1.0 - batch_size / ema_window_tokens)
                        feature_activation_ema[sae_key] = feature_activation_ema[sae_key] * ema_decay + fired
                        # Dead neurons: EMA below threshold means feature essentially never fires
                        layer_dead_neurons[sae_key] = (feature_activation_ema[sae_key] < 0.01).sum().item()

                    # Store FVU if available (JumpReLU SAE computes this)
                    # Convert tensor to float for database storage
                    fvu_val = losses.get('fvu', None)
                    if fvu_val is not None:
                        layer_fvu[sae_key] = fvu_val.item() if hasattr(fvu_val, 'item') else float(fvu_val)
                    else:
                        layer_fvu[sae_key] = None

                # Calculate aggregated metrics across all layers
                avg_loss = sum(layer_losses.values()) / len(layer_losses)
                avg_sparsity = sum(layer_sparsities.values()) / len(layer_sparsities)
                avg_dead_neurons = sum(layer_dead_neurons.values()) / len(layer_dead_neurons)
                # Calculate avg FVU only if any layer has FVU (JumpReLU)
                fvu_values = [v for v in layer_fvu.values() if v is not None]
                avg_fvu = float(sum(fvu_values) / len(fvu_values)) if fvu_values else None
                # Calculate avg reconstruction and L1 losses
                recon_values = [v for v in layer_recon_losses.values() if v is not None]
                avg_recon_loss = float(sum(recon_values) / len(recon_values)) if recon_values else None
                l1_values = [v for v in layer_l1_losses.values() if v is not None]
                avg_l1_loss = float(sum(l1_values) / len(l1_values)) if l1_values else None

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

                    # Clear GPU memory and reset scaler states
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # Recreate all GradScalers to ensure clean state.
                        # An OOM can interrupt the inner loop at any point,
                        # potentially leaving a scaler in a dirty state.
                        for sae_key in scalers:
                            scalers[sae_key] = GradScaler()

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

            # Get aggregated metrics (use first SAE's learning rate as representative)
            first_sae_key = layer_hook_combinations[0]
            current_lr = schedulers[first_sae_key].get_last_lr()[0]

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

                    # Build loss decomposition string
                    loss_parts = f"loss={avg_loss:.6f}"
                    if avg_recon_loss is not None:
                        loss_parts += f" (recon={avg_recon_loss:.6f}"
                        if avg_l1_loss is not None:
                            sparsity_label = "L0" if sparsity_type == "l0" else "L1"
                            loss_parts += f", {sparsity_label}={avg_l1_loss:.6f}"
                        loss_parts += ")"

                    # Calculate throughput
                    if step >= 100:
                        elapsed_time = time.time() - step_start_time
                        actual_steps_per_min = (step / elapsed_time) * 60
                        logger.info(
                            f"Step {step}: {loss_parts}, L0={avg_sparsity:.4f}, "
                            f"throughput={actual_steps_per_min:.1f} steps/min, "
                            f"GPU mem={gpu_memory_allocated:.0f}MB"
                        )
                    else:
                        logger.info(
                            f"Step {step}: {loss_parts}, L0={avg_sparsity:.4f}, "
                            f"GPU mem={gpu_memory_allocated:.0f}MB"
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

                # Log per-SAE metrics (one per layer/hook_type combination)
                for layer_idx, hook_type in layer_hook_combinations:
                    sae_key = (layer_idx, hook_type)
                    self.log_metric(
                        training_id=training_id,
                        step=step,
                        loss=layer_losses[sae_key],
                        l0_sparsity=layer_sparsities[sae_key],
                        dead_neurons=int(layer_dead_neurons[sae_key]),
                        learning_rate=current_lr,
                        layer_idx=layer_idx,  # For backward compat, log layer_idx only
                        fvu=layer_fvu.get(sae_key),  # Per-SAE FVU
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
                    fvu=avg_fvu,
                )

                # Check training quality (with race-to-zero detection)
                quality_warnings = TrainingValidator.check_training_quality(
                    step=step,
                    l0_sparsity=avg_sparsity,
                    dead_neurons=int(avg_dead_neurons),
                    latent_dim=hp['latent_dim'],
                    target_l0=hp.get('target_l0') or 0.05,
                    warmup_steps=hp.get('warmup_steps') or 0,
                    training_id=training_id,
                    sparsity_warmup_steps=sparsity_warmup_steps,
                )
                if quality_warnings:
                    for warning in quality_warnings:
                        logger.warning(warning)

                # Dead neuron resampling (if enabled)
                # Skip for TopK — it uses aux loss for dead feature recovery instead
                if hp.get('resample_dead_neurons', False) and sparsity_type != 'topk':
                    dead_neuron_threshold = hp.get('dead_neuron_threshold', 10000)
                    resample_interval = hp.get('resample_interval', 5000)

                    # Perform resampling at specified intervals after warmup
                    # Wait until after both LR warmup and sparsity warmup are done
                    effective_warmup = max(hp.get('warmup_steps', 0), sparsity_warmup_steps)
                    if step > 0 and step % resample_interval == 0 and step >= effective_warmup:
                        for layer_idx, hook_type in layer_hook_combinations:
                            sae_key = (layer_idx, hook_type)
                            model = models[sae_key]

                            # Use EMA-based dead neuron detection (more reliable than per-batch)
                            x = layer_activations[sae_key]
                            with torch.no_grad():
                                if sae_key in feature_activation_ema:
                                    dead_mask = feature_activation_ema[sae_key] < 0.01
                                else:
                                    # Fallback: per-batch detection.
                                    # MIS-E2E-083 sibling sweep: `x` is a raw
                                    # layer activation and `encode` does not
                                    # normalize, so dead-neuron counts on this
                                    # fallback were measured off-distribution.
                                    from ..ml.sparse_autoencoder import (
                                        encode_with_training_normalization,
                                    )

                                    z = encode_with_training_normalization(model, x)
                                    dead_mask = (z == 0).all(dim=0)
                                num_dead = dead_mask.sum().item()

                                if num_dead > 0:
                                    logger.info(f"Layer {layer_idx}/{hook_type}: Resampling {num_dead} dead neurons at step {step}")

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

                # Compute current l1_alpha for reporting (after warmup scaling)
                current_l1_alpha = None
                if sparsity_type == 'l1':
                    first_model = models[layer_hook_combinations[0]]
                    current_l1_alpha = getattr(first_model, 'l1_alpha', None)

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
                        "reconstruction_loss": avg_recon_loss,
                        "l1_loss": avg_l1_loss,
                        "l1_alpha": current_l1_alpha,
                        "l0_sparsity": avg_sparsity,
                        # FVU was computed at line ~1404 and persisted to
                        # training_metrics, but never emitted — so the live UI
                        # could not show the one metric that actually indicates
                        # convergence.
                        "fvu": avg_fvu,
                        "dead_neurons": int(avg_dead_neurons),
                        "learning_rate": current_lr,
                        "num_layers": num_layers,
                        "num_hook_types": num_hook_types,
                        "num_sae_models": num_sae_models,
                        "training_layers": training_layers,
                        "hook_types": hook_types_config,
                    }
                )

            # Save checkpoint periodically
            if step % checkpoint_interval == 0 and step > 0:
                logger.info(f"Saving checkpoint at step {step}...")

                # Save multi-layer/multi-hook checkpoint
                checkpoint_paths = CheckpointService.save_multilayer_checkpoint(
                    models=models,
                    optimizers=optimizers,
                    step=step,
                    base_storage_path=str(checkpoint_dir),
                    layer_hook_combinations=layer_hook_combinations,
                    extra_metadata={
                        'avg_loss': avg_loss,
                        'avg_sparsity': avg_sparsity,
                        'hook_types': hook_types_config,
                        'layer_losses': {str(k): v for k, v in layer_losses.items()},
                    }
                )

                # Create checkpoint record for EACH SAE (multi-layer/multi-hook support)
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

                    # Create a checkpoint record for each (layer, hook_type) combination
                    for layer_idx, hook_type in layer_hook_combinations:
                        sae_key = (layer_idx, hook_type)
                        checkpoint_path = checkpoint_paths[sae_key]
                        sae_loss = layer_losses.get(sae_key, avg_loss)
                        sae_sparsity = layer_sparsities.get(sae_key, avg_sparsity)

                        checkpoint_id = f"ckpt_{uuid4().hex[:8]}"
                        checkpoint = Checkpoint(
                            id=checkpoint_id,
                            training_id=training_id,
                            step=step,
                            loss=sae_loss,
                            l0_sparsity=sae_sparsity,
                            storage_path=checkpoint_path,
                            is_best=is_best,
                            extra_metadata={
                                'layer_idx': layer_idx,
                                'hook_type': hook_type,
                                'num_sae_models': num_sae_models,
                                'training_layers': training_layers,
                                'hook_types': hook_types_config,
                                'avg_loss': avg_loss,
                                'avg_sparsity': avg_sparsity,
                            },
                        )
                        db.add(checkpoint)

                        logger.info(f"Checkpoint saved: {checkpoint_id} layer={layer_idx}/{hook_type} (is_best={is_best})")

                    db.commit()

                    # Emit checkpoint:created WebSocket event (use first SAE for backward compat)
                    first_sae_key = layer_hook_combinations[0]
                    from ..workers.websocket_emitter import emit_checkpoint_created
                    emit_checkpoint_created(
                        training_id=training_id,
                        checkpoint_id=checkpoint_id,  # Last created checkpoint ID
                        step=step,
                        loss=avg_loss,
                        is_best=is_best,
                        storage_path=checkpoint_paths[first_sae_key],
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
            layer_hook_combinations=layer_hook_combinations,
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
        for layer_idx, hook_type in layer_hook_combinations:
            sae_key = (layer_idx, hook_type)
            del models[sae_key]
            del optimizers[sae_key]
            del schedulers[sae_key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GPU memory cleanup completed")

        from datetime import datetime, UTC
        completed_at = datetime.now(UTC)
        with self.get_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            training.status = TrainingStatus.COMPLETED.value
            training.progress = 100.0
            training.completed_at = completed_at
            db.commit()

        # Emit training:completed WebSocket event
        from ..workers.websocket_emitter import emit_training_progress
        _emit_ok = emit_training_progress(
            training_id=training_id,
            event="training:completed",
            data={
                "training_id": training_id,
                "status": "completed",
                "final_loss": avg_loss,
                "completed_at": completed_at.isoformat(),
            },
            retries=2,  # Terminal event: frontend stays "running" if this is lost
        )
        if not _emit_ok:
            logger.warning(f"[Training {training_id}] WebSocket emit for 'training:completed' failed — frontend may not update")

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
            if 'models' in locals() and 'layer_hook_combinations' in locals():
                for layer_idx, hook_type in layer_hook_combinations:
                    sae_key = (layer_idx, hook_type)
                    if sae_key in models:
                        del models[sae_key]
                    if sae_key in optimizers:
                        del optimizers[sae_key]
                    if sae_key in schedulers:
                        del schedulers[sae_key]
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
        _emit_ok = emit_training_progress(
            training_id=training_id,
            event="training:failed",
            data={
                "training_id": training_id,
                "error_message": str(e),
            },
            retries=2,  # Terminal event: frontend stays "running" if this is lost
        )
        if not _emit_ok:
            logger.warning(f"[Training {training_id}] WebSocket emit for 'training:failed' failed — frontend may not update")

        raise


@get_celery_app().task(name="resume_training")
def resume_training_task(training_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Resume a paused training job from its latest (or specified) checkpoint.

    Queries the database for the checkpoint to resume from, then dispatches
    train_sae_task with start_step set to the checkpoint step.

    Args:
        training_id: Training job ID
        checkpoint_id: Checkpoint ID to resume from (None = use latest)

    Returns:
        Dictionary with resume result
    """
    from ..models.checkpoint import Checkpoint
    from ..core.database import get_sync_db

    logger.info(f"Resuming training {training_id} from checkpoint={checkpoint_id or 'latest'}")

    with get_sync_db() as db:
        training = db.query(Training).filter_by(id=training_id).first()
        if not training:
            raise ValueError(f"Training not found: {training_id}")

        if checkpoint_id:
            ckpt = db.query(Checkpoint).filter_by(id=checkpoint_id).first()
        else:
            # Use the best checkpoint, falling back to the latest by step
            ckpt = (
                db.query(Checkpoint)
                .filter_by(training_id=training_id, is_best=True)
                .order_by(Checkpoint.step.desc())
                .first()
            ) or (
                db.query(Checkpoint)
                .filter_by(training_id=training_id)
                .order_by(Checkpoint.step.desc())
                .first()
            )

        if not ckpt:
            raise ValueError(
                f"No checkpoint found for training {training_id}. "
                "Cannot resume without a saved checkpoint."
            )

        start_step = ckpt.step
        resolved_checkpoint_id = ckpt.id

    logger.info(f"Resuming training {training_id} from step {start_step} (checkpoint {resolved_checkpoint_id})")
    train_sae_task.delay(
        training_id=training_id,
        start_step=start_step,
        checkpoint_id=resolved_checkpoint_id,
    )
    return {"status": "queued", "start_step": start_step, "checkpoint_id": resolved_checkpoint_id}


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
    from ..core.config import settings

    logger.info(f"Starting file cleanup for training: {training_id}")
    deleted_files = []
    errors = []

    # Emit in_progress status first
    emit_deletion_progress(training_id, "files", "in_progress", "Deleting training files...")

    try:
        # Resolve Docker-style /data/ paths for native mode compatibility
        resolved_dir = None
        if training_dir:
            try:
                resolved_dir = str(settings.resolve_deletable_path(training_dir))
            except ValueError as e:
                errors.append(f"Refusing to delete training_dir {training_dir!r}: {e}")
                logger.error(errors[-1])

        # Delete training directory
        if resolved_dir and Path(resolved_dir).exists():
            try:
                shutil.rmtree(resolved_dir)
                deleted_files.append(resolved_dir)
                logger.info(f"Deleted training directory: {resolved_dir}")
                # Emit success
                emit_deletion_progress(training_id, "files", "completed", "Deleted training files")
            except Exception as e:
                error_msg = f"Failed to delete training directory {resolved_dir}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Emit error
                emit_deletion_progress(training_id, "files", "completed", f"Error deleting files: {str(e)}")
        elif training_dir:
            logger.warning(f"Training directory does not exist: {training_dir} (resolved: {resolved_dir})")
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
