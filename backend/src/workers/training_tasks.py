"""
Celery tasks for SAE training operations.

This module contains Celery tasks for training Sparse Autoencoders,
including the main training loop, metric logging, and checkpoint management.
"""

import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.optim as optim
from celery import Task

from .base_task import DatabaseTask
from ..ml.sparse_autoencoder import create_sae
from ..models.training import Training, TrainingStatus
from ..services.training_service import TrainingService
from ..services.checkpoint_service import CheckpointService
from ..core.config import settings
from ..utils.resource_estimation import estimate_training_memory, estimate_oom_reduced_batch_size

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
            )
            db.add(metric)
            db.commit()


@get_celery_app().task(base=TrainingTask, bind=True, name="train_sae")
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
    """
    logger.info(f"Starting SAE training task for training_id={training_id}")

    # Get training record
    with self.get_db() as db:
        training = db.query(Training).filter_by(id=training_id).first()
        if not training:
            raise ValueError(f"Training not found: {training_id}")

        # Update status to initializing
        training.status = TrainingStatus.INITIALIZING.value
        db.commit()

        # Extract hyperparameters
        hp = training.hyperparameters
        logger.info(f"Hyperparameters: {hp}")

    try:
        # Memory budget validation
        logger.info("Validating memory budget...")
        batch_size = hp['batch_size']
        memory_estimate = estimate_training_memory(
            hidden_dim=hp['hidden_dim'],
            latent_dim=hp['latent_dim'],
            batch_size=batch_size,
        )
        logger.info(f"Estimated memory usage: {memory_estimate['total_gb']:.2f} GB")

        if not memory_estimate['fits_in_6gb']:
            error_msg = (
                f"Training requires {memory_estimate['total_gb']:.2f} GB but only 6 GB available. "
                f"Reduce batch_size (current: {batch_size}) or latent_dim."
            )
            logger.error(error_msg)
            with self.get_db() as db:
                training = db.query(Training).filter_by(id=training_id).first()
                training.status = TrainingStatus.FAILED.value
                training.error_message = error_msg
                db.commit()
            raise RuntimeError(error_msg)

        # Initialize model
        logger.info("Initializing SAE model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model = create_sae(
            architecture_type=hp.get('architecture_type', 'standard'),
            hidden_dim=hp['hidden_dim'],
            latent_dim=hp['latent_dim'],
            l1_alpha=hp['l1_alpha'],
            ghost_gradient_penalty=hp.get('ghost_gradient_penalty', 0.0),
        ).to(device)

        # Initialize optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=hp['learning_rate'],
            weight_decay=hp.get('weight_decay', 0.0),
        )

        # Learning rate scheduler (linear warmup + constant)
        warmup_steps = hp.get('warmup_steps', 0)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
                # TODO: Load real data batch
                # For now, use dummy data for testing
                x = torch.randn(batch_size, hp['hidden_dim']).to(device)

                # Forward pass
                if step % grad_accum_steps == 0:
                    optimizer.zero_grad()

                x_reconstructed, z, losses = model(x, return_loss=True)

                # Backward pass with gradient accumulation
                loss = losses['loss']
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps
                loss.backward()

                # Optimizer step (only every grad_accum_steps)
                if (step + 1) % grad_accum_steps == 0:
                    # Gradient clipping
                    grad_clip_norm = hp.get('grad_clip_norm')
                    if grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                    # Optimizer step
                    optimizer.step()
                    scheduler.step()

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

            # Get metrics
            loss_value = loss.item()
            l0_sparsity = losses['l0_sparsity'].item()
            l1_penalty = losses['l1_penalty'].item()
            current_lr = scheduler.get_last_lr()[0]

            # Count dead neurons
            with torch.no_grad():
                dead_mask = model.get_dead_neurons(z, threshold=1e-6)
                dead_neurons = dead_mask.sum().item()

            # Log metrics periodically
            if step % log_interval == 0:
                # GPU memory monitoring
                gpu_memory_mb = None
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
                    gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
                    gpu_memory_mb = gpu_memory_allocated
                    logger.info(
                        f"GPU memory: allocated={gpu_memory_allocated:.2f}MB, "
                        f"reserved={gpu_memory_reserved:.2f}MB"
                    )

                self.log_metric(
                    training_id=training_id,
                    step=step,
                    loss=loss_value,
                    l0_sparsity=l0_sparsity,
                    l1_sparsity=l1_penalty,
                    dead_neurons=int(dead_neurons),
                    learning_rate=current_lr,
                    gpu_memory_used_mb=gpu_memory_mb,
                )

                # Update progress
                self.update_training_progress(
                    training_id=training_id,
                    step=step,
                    total_steps=total_steps,
                    loss=loss_value,
                    l0_sparsity=l0_sparsity,
                    dead_neurons=int(dead_neurons),
                    learning_rate=current_lr,
                )

                # Emit training:progress WebSocket event
                from ..workers.websocket_emitter import emit_training_progress
                emit_training_progress(
                    training_id=training_id,
                    event="progress",
                    data={
                        "training_id": training_id,
                        "current_step": step,
                        "total_steps": total_steps,
                        "progress": (step / total_steps) * 100.0,
                        "loss": loss_value,
                        "l0_sparsity": l0_sparsity,
                        "dead_neurons": int(dead_neurons),
                        "learning_rate": current_lr,
                    }
                )

                logger.info(
                    f"Step {step}/{total_steps}: "
                    f"loss={loss_value:.4f}, l0={l0_sparsity:.4f}, "
                    f"dead={dead_neurons}, lr={current_lr:.6f}"
                )

            # Save checkpoint periodically
            if step % checkpoint_interval == 0 and step > 0:
                checkpoint_path = f"{checkpoint_dir}/step_{step}.safetensors"

                # Save checkpoint file
                CheckpointService.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    storage_path=checkpoint_path,
                    extra_metadata={
                        'loss': loss_value,
                        'l0_sparsity': l0_sparsity,
                    }
                )

                # Create checkpoint record
                with self.get_db() as db:
                    from ..models.checkpoint import Checkpoint
                    from uuid import uuid4

                    is_best = loss_value < best_loss
                    if is_best:
                        best_loss = loss_value

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
                        loss=loss_value,
                        l0_sparsity=l0_sparsity,
                        storage_path=checkpoint_path,
                        is_best=is_best,
                    )
                    db.add(checkpoint)
                    db.commit()

                    # Emit checkpoint:created WebSocket event
                    from ..workers.websocket_emitter import emit_checkpoint_created
                    emit_checkpoint_created(
                        training_id=training_id,
                        checkpoint_id=checkpoint_id,
                        step=step,
                        loss=loss_value,
                        is_best=is_best,
                        storage_path=checkpoint_path,
                    )

                logger.info(f"Saved checkpoint at step {step} (best={is_best})")

        # Training completed
        logger.info(f"Training completed: {total_steps} steps")

        with self.get_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            training.status = TrainingStatus.COMPLETED.value
            training.progress = 100.0
            from datetime import datetime, UTC
            training.completed_at = datetime.now(UTC)
            db.commit()

        return {
            "status": "completed",
            "steps": total_steps,
            "final_loss": loss_value,
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())

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
