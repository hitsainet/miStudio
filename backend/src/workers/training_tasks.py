"""
Celery tasks for SAE training operations.

This module contains Celery tasks for training Sparse Autoencoders,
including the main training loop, metric logging, and checkpoint management.
"""

import logging
import traceback
import zlib
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
from .gpu_job import gpu_job, training_can_split, training_request
from ..services.gpu_dispatch import gpu_delay
from ..ml.sparse_autoencoder import create_sae, project_decoder_gradients, JumpReLUSAE, TopKSAE
from ..models.training import Training, TrainingStatus
from ..models.dataset import Dataset
from ..models.model import Model
from ..models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from ..services.training_service import TrainingService
from ..services import (
    activation_buffer,
    activation_mask,
    activation_plan,
    dataset_mixture,
    feature_density,
    holdout_evaluation,
    model_activation_source,
)
from .model_tasks import select_tokenization_for_model
from ..services.checkpoint_service import (
    CheckpointService,
    build_training_state,
    load_training_state,
    restore_rng_state,
    restore_training_state,
    save_training_state,
    select_resume_checkpoint,
    training_state_path,
)
from ..services.checkpoint_disk import (
    checkpoint_step_bytes,
    is_out_of_space,
    remove_partial_step,
    room_for_one_step,
    sae_footprint,
    verdict_from_sync_session,
)
from ..services.dead_latent_resampling import DeadLatentTracker, resample_dead_latents, resample_due
from ..services.lr_schedule import build_lr_scheduler
from ..schemas.training import TrainingHyperparameters
from ..core.cancellation import GPU_LEASE_LOST, guard_allows, record_progress
from ..core.config import settings
from ..services.gpu_job_claim import lease_lost_reason
from ..utils.resource_estimation import (
    estimate_training_memory,
    estimate_multilayer_training_memory,
    estimate_oom_reduced_batch_size,
)
from ..services.training_validator import TrainingValidator
from ..services.gpu_placement import AUTO, place_job
from ..services.base_model_budget import base_model_mb, budget_beside_sae
from ..ml.model_devices import input_device
from ..ml.model_loader import load_model_from_hf, QuantizationFormat
from ..ml.forward_hooks import HookManager, HookType

logger = logging.getLogger(__name__)


def get_celery_app():
    """Import celery app lazily to avoid circular imports."""
    from ..core.celery_app import celery_app
    return celery_app


#: Batches smaller than this accumulate gradients over several steps.
MIN_EFFECTIVE_BATCH = 64


def steps_per_minute(steps_done: int, elapsed_seconds: float) -> Optional[float]:
    """Throughput over the steps THIS PROCESS ran, or None when there is nothing to report.

    `step` is the run's absolute step number while the clock starts when the process
    does, so `step / elapsed` was nonsense on every resumed run (review R2C-8): a
    resume at step 120,000 reported millions of steps per minute in its first
    seconds, and since the slow-training alert fires BELOW a threshold, it could
    never fire again for the rest of the run — the one run most likely to be slow.

    None rather than a number when no step has finished or no time has passed: the
    caller's alert divides by this to report how many times slower than target it
    is, and 0 steps/min is a ZeroDivisionError, not a performance report.
    """
    if steps_done <= 0 or elapsed_seconds <= 0:
        return None
    return (steps_done / elapsed_seconds) * 60.0


def grad_accum_steps_for(batch_size: int) -> int:
    """Training steps per optimizer step: enough to reach ``MIN_EFFECTIVE_BATCH``.

    One definition, read by the step loop and by the LR scheduler, which counts
    optimizer steps and must convert them to the training steps the schedule is
    configured in.
    """
    batch_size = int(batch_size)
    return max(1, MIN_EFFECTIVE_BATCH // batch_size) if batch_size < MIN_EFFECTIVE_BATCH else 1


def _fresh_scaler_at(scale: Any) -> "GradScaler":
    """A clean GradScaler carrying forward an already-calibrated loss scale.

    An OOM rebuilds the scalers to clear any half-applied state, and rebuilding them
    with `GradScaler()` restarted the scale at its 65536 default (review A9). That
    scale is calibrated over thousands of steps; restarting it costs a run of
    overflowing, skipped updates every time a batch is reduced. A non-finite or
    non-positive scale is not carried forward — there is nothing to preserve.
    """
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return GradScaler()
    if scale > 0 and scale not in (float("inf"), float("-inf")) and scale == scale:
        return GradScaler(init_scale=scale)
    return GradScaler()


def resolve_grad_accum_steps(saved_grad_accum_steps: Optional[int], batch_size: int) -> int:
    """The accumulation window this run trains with: the saved one, else derived.

    THE WINDOW BELONGS TO THE RUN, NOT TO THE CURRENT BATCH SIZE (review A9). An OOM
    halves `batch_size` and persists it to the row, so a resumed process that derived
    the window afresh got a DIFFERENT one from the process it is continuing — which
    changes two things at once. The effective batch changes (batch x window), and so
    does the step -> optimizer-step conversion the LR scheduler is built with, while
    the scheduler's restored position was recorded under the old conversion. The run
    then follows a learning-rate curve neither half of it was configured for.

    Keeping the saved window means an OOM's smaller batch shows up honestly as a
    smaller effective batch, logged where it happens, instead of being papered over
    by a wider window on the next resume.

    A pure function, so the choice is unit-testable without a training run.
    """
    if saved_grad_accum_steps:
        return int(saved_grad_accum_steps)
    return grad_accum_steps_for(batch_size)


def draw_cached_batch(activation_stream, cached_activations, keys, batch_size, num_samples, device):
    """One training batch per (layer, hook) from cached activations.

    With a rolling buffer: the next tokens, without replacement, from a buffer
    that cycles the whole pool. With a fixed pool that fits in memory: a uniform
    draw with replacement, as before.

    A function, not an inline branch, so the choice can be driven by a test. An
    AST check cannot see a branch that is never taken: making the condition
    `if False:` left a `next_batch` call in the source and the suite green, while
    training silently went back to sampling one fixed subset.
    """
    if activation_stream is not None:
        batches = activation_stream.next_batch(batch_size)
    else:
        indices = torch.randint(0, num_samples, (batch_size,))
        batches = {key: cached_activations[key][indices] for key in keys}
    out = {}
    for key in keys:
        batch = batches[key]
        # Transfer to the training device when the pool lives on the CPU.
        if batch.device != device:
            batch = batch.to(device, non_blocking=True)
        out[key] = batch
    return out


def stop_signal_for(
    row,
    step: int,
    lease_lost: Optional[str] = None,
    *,
    terminal_is_a_stop: bool = False,
) -> Optional[Dict[str, Any]]:
    """The training loop's decision at a status check: None to keep going, else the task's result.

    A DELETED ROW IS A STOP. The row is created before dispatch, so a missing
    row means someone deleted it — typically Stop, then Delete before the next
    check. Reading `.status` off that None used to crash the worker with an
    AttributeError, and the failed task's tensors stayed allocated (18 GB on the
    3090, 2026-09-14) until the worker ran its next task.

    A LOST GPU LEASE IS A STOP TOO (multi-GPU Phase 3): `lease_lost` is
    `gpu_job_claim.lease_lost_reason()`. The operator's own stop wins; a lease
    loss is the job's failure, which the loop records with the reason.

    `terminal_is_a_stop` IS FOR THE LOOP ONLY, and the asymmetry is the point.
    Two callers decide through this function and they need OPPOSITE answers for
    COMPLETED:

        check_stop (the loop)       COMPLETED -> someone finalized underneath
                                                 me, so STOP
        post_run_stop_reason (eval) COMPLETED -> the normal state, KEEP GOING

    because R3-A (decided 2026-09-15) marks the row COMPLETED in the commit
    AFTER the export and BEFORE the post-run evaluation, so a Stop during the
    evaluation cancels only the evaluation and leaves the run completed with its
    final weights. Making every terminal status a stop unconditionally breaks
    that — it aborts each evaluation at its first forward.

    THE LOOP USED TO RECOGNISE CANCELLED ALONE, and on 2026-09-16 that cost a
    run. Stop & Finalize writes CANCELLED, then
    `finalize_training_from_checkpoint_task` writes COMPLETED asynchronously; on
    train_c29aa474 those two landed 3 SECONDS apart, inside the loop's ~4 s poll
    window. The finalize overwrote the cancellation before the loop ever read
    it, the loop then saw a status it did not recognise, and it trained on for
    2.5 hours past a run the database called finished, to FVU > 5, stopping only
    when the row was set to CANCELLED by hand. `revoke_task(terminate=True)` is
    no backstop: this pool is `--pool=solo` and revoke signals a pool child that
    does not exist. Nor was Stop, which `TrainingService.stop_training` refuses
    on an already-terminal row.

    `start_refusal` below has always consulted the registry through
    `guard_allows`, so the same row that could not START a task could not STOP
    one. The loop now derives its answer from the scope's `terminal_values` too;
    a second hardcoded list was the defect. PAUSED and CANCELLED stay explicit
    and unconditional — both callers must honour them — and the terminal check
    stays ABOVE the lease check so the operator's stop still wins.
    """
    if row is None:
        return {"status": "cancelled", "step": step, "reason": "deleted"}
    status = getattr(row.status, "value", row.status)
    if status == TrainingStatus.PAUSED.value:
        return {"status": "paused", "step": step}
    if status == TrainingStatus.CANCELLED.value:
        return {"status": "cancelled", "step": step}
    if terminal_is_a_stop and not guard_allows(
        "training", status, TrainingStatus.RUNNING.value
    ):
        return {"status": str(status), "step": step}
    if lease_lost:
        return {"status": "failed", "step": step, "reason": GPU_LEASE_LOST, "detail": lease_lost}
    return None


def start_refusal(row, before: str) -> Optional[Dict[str, Any]]:
    """What a training task returns INSTEAD of running, when its row says it must not run.

    A deleted row, or one already terminal: CANCELLED, FAILED, PAUSED (and COMPLETED) —
    the training scope's terminal values (core/cancellation.py). Decided by
    ``guard_allows`` against the live status the task is about to write, the rule
    ``record_progress`` applies, so a task never writes INITIALIZING or RUNNING over an
    operator's Stop or Pause (review round 2, R2D-5). Cancellation here is cooperative:
    a revoke does nothing to a solo-pool worker, and revokes are not persisted, so a
    queued task that starts after a worker restart would otherwise train to completion.
    ``None`` means the task may run.
    """
    if row is None:
        return {"status": "cancelled", "step": 0, "reason": "deleted"}
    status = getattr(row.status, "value", row.status)
    if guard_allows("training", status, TrainingStatus.INITIALIZING.value):
        return None
    return {"status": str(status), "step": int(row.current_step or 0), "reason": f"stopped before {before}"}


#: Why a run stopped when the checkpoint volume ran out (debt R2D-7 / R3D-15).
DISK_FULL = "insufficient_disk"


def pause_for_disk(training_id: str, *, step: int, message: str) -> Dict[str, Any]:
    """PAUSE this run because its checkpoints will not fit — never FAIL it.

    THIS IS THE WHOLE POINT OF THE DISK GUARDS. A full disk used to raise inside
    the loop, and the task's outer handler marks the run FAILED. Resuming accepts
    PAUSED **or** FAILED, so that was survivable only by luck: the very disk that
    failed the run is the one that may have truncated the checkpoint its resume
    would then read. Pausing stops the run at a checkpoint it is known to have
    written completely, puts the reason on the row, and leaves Resume as the
    obvious next action once space exists.

    Module-level, not a closure in the task, so the reachability guard can assert
    the CALL by walking the AST and so the decision is testable without running a
    training.
    """
    logger.error("Training %s paused at step %s: %s", training_id, step, message)
    record_progress(
        "training",
        training_id,
        status=TrainingStatus.PAUSED.value,
        error_message=message,
    )
    return {"status": "paused", "step": int(step), "reason": DISK_FULL, "detail": message}


def discard_metrics_after_step(db, training_id: str, step: int) -> int:
    """Delete this training's metric rows logged AFTER ``step``. Returns how many went.

    A resume continues at ``step + 1`` and re-runs every step up to where the run
    stopped, logging each again. ``training_metrics`` is unique on (training_id,
    step, layer_idx), so the per-layer insert of the first re-run log step raised
    IntegrityError and the resumed training failed — on the ordinary pause, which
    lands wherever the operator presses it, several log steps past the newest
    checkpoint (review round 1, R1-A). The aggregated rows have a NULL layer, which
    Postgres never treats as a duplicate, so they would have been doubled instead.

    HOOK-AGNOSTIC BY DESIGN (review R1-A, A5). The key now includes the hook type,
    and this deletes every row after ``step`` whatever its layer or hook: a resumed
    run re-logs every SAE of those steps, so every SAE's rows must go.

    The rows discarded describe steps the resumed run replaces; the checkpoint's own
    step ran before the checkpoint was written and stays.
    """
    from ..models.training_metric import TrainingMetric

    removed = (
        db.query(TrainingMetric)
        .filter(TrainingMetric.training_id == training_id, TrainingMetric.step > int(step))
        .delete(synchronize_session=False)
    )
    db.commit()
    return int(removed or 0)


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
        fvu_centred: Optional[float] = None,
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
            # The headline, beside the legacy value above — never in its place:
            # `current_fvu` keeps its global-mean meaning on every row.
            if fvu_centred is not None:
                fields["current_fvu_centred"] = fvu_centred
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
        loss_reconstructed: Optional[float] = None,
        loss_zero: Optional[float] = None,
        l0_mean: Optional[float] = None,
        l0_sparsity: Optional[float] = None,
        l1_sparsity: Optional[float] = None,
        dead_neurons: Optional[int] = None,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None,
        gpu_memory_used_mb: Optional[float] = None,
        samples_per_second: Optional[float] = None,
        layer_idx: Optional[int] = None,
        fvu: Optional[float] = None,
        fvu_centred: Optional[float] = None,
        hook_type: Optional[str] = None,
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
            hook_type: The SAE's hook type on per-SAE and held-out rows; None on
                aggregated rows. Part of the table's unique key (review R1-A, A5):
                without it, two hook types on one layer write the same key.
        """
        with self.get_db() as db:
            from ..models.training_metric import TrainingMetric

            metric = TrainingMetric(
                training_id=training_id,
                step=step,
                # These columns existed and were NEVER written — log_metric
                # accepted some of them and no caller passed any. `loss_zero` had
                # a comment elsewhere claiming it was persisted.
                loss_reconstructed=loss_reconstructed,
                loss_zero=loss_zero,
                l0_mean=l0_mean,
                loss=loss,
                l0_sparsity=l0_sparsity,
                l1_sparsity=l1_sparsity,
                dead_neurons=dead_neurons,
                learning_rate=learning_rate,
                grad_norm=grad_norm,
                gpu_memory_used_mb=gpu_memory_used_mb,
                samples_per_second=samples_per_second,
                layer_idx=layer_idx,
                hook_type=hook_type,
                fvu=fvu,
                fvu_centred=fvu_centred,
            )
            db.add(metric)
            db.commit()

    def _close_activation_stream(self) -> None:
        """Stop the rolling buffer's read threads, whichever way the task ended.

        A training returns from a dozen places (stop, delete, OOM, success). The
        buffer's prefetch thread holds the buffer — ~15 GB of GPU storage and as
        much pinned RAM on a 3-layer run — so a missed close would keep both
        alive in a worker that runs the next job. This hook runs after every one.
        """
        stream = getattr(self, "_activation_stream", None)
        self._activation_stream = None
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must reach the GPU release below
                logger.warning(f"Error closing the activation buffer: {exc}")

    def release_before_evaluation(self) -> None:
        """Close the rolling buffer or the on-the-fly source IN PLACE before the post-run evaluation.

        Review R1-D, R1D-4/R1D-5. `close()` frees the buffer's storage (and the on-the-fly
        capture's reference to the model, which `base_model` still holds for the
        evaluation). Unlike `release_job_memory`, it does not DETACH the source:
        `after_return` does that, so what the source recorded (quotas, rows read) stays
        readable to the end of the task. The step loop's locals are cleared by the caller.
        """
        stream = getattr(self, "_activation_stream", None)
        if stream is not None:
            stream.close()

    def release_job_memory(self) -> None:
        """For the GPU claim (multi-GPU Phase 3): free this training's buffer BEFORE its lease goes.

        ``@gpu_job`` calls this, then collects garbage and empties the leased
        cards' caches, while the cards are still leased — so no job is placed on
        a card this training's buffer still fills. ``after_return`` runs later
        and repeats the release harmlessly (single mode relies on it alone).
        """
        self._close_activation_stream()

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

        self._close_activation_stream()

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
                # PINNED HOST MEMORY TOO, and only now. `empty_cache` above releases
                # device memory; a freed pinned block stays locked in PyTorch's host
                # cache under the worker's next job. The buffer's close() releases
                # its own, but the fixed pool (cpu_all) pins every layer with no
                # buffer at all, and a block held by garbage is freed only by the
                # collection above — so the release belongs after it.
                activation_buffer._empty_pinned_host_cache()

                # Log memory after cleanup — for EVERY card. This hook does not
                # know which card the job ran on, and the bare calls read only
                # the current device.
                for index in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(index) / (1024**2)  # MB
                    reserved = torch.cuda.memory_reserved(index) / (1024**2)    # MB
                    logger.info(
                        f"GPU {index} memory after cleanup: "
                        f"{allocated:.1f}MB allocated, {reserved:.1f}MB reserved"
                    )

            logger.info("GPU memory cleanup completed in after_return")

        except Exception as cleanup_error:
            logger.warning(f"Error during after_return GPU cleanup: {cleanup_error}")

        # Call parent's after_return
        super().after_return(status, retval, task_id, args, kwargs, einfo)


def extraction_ids_of(training) -> Optional[list]:
    """The extractions a training reads cached activations from; None when it extracts on the fly.

    ONE DEFINITION, because two decisions hang on it: what the step loop reads,
    and whether the job may be placed across GPUs. Only the on-the-fly path loads
    a base model; a cached run holds SAEs, optimizer and rolling buffer on one
    card, and a copy of this test that disagreed would place such a run split.
    """
    if training.extraction_ids:
        return list(training.extraction_ids)
    return [training.extraction_id] if training.extraction_id else None


#: Rows per on-the-fly forward before the token budget below shortens it.
ON_THE_FLY_MICRO_BATCH_ROWS = 8
#: Tokens per on-the-fly forward, scaled down for long windows by the
#: extraction's own rule (``micro_batch_size_for_length``): 2 rows at 2,048.
ON_THE_FLY_FORWARD_TOKENS = 16_384


#: Latent-sized float32 tensors one training step holds per batch token (forward,
#: what autograd saves, backward), measured on CPU allocations at 16,384 latents and
#: a 2,048 residual (review R1-B F5): JumpReLU ~11 — 608 MiB at batch 1,024 and
#: 2,816 MiB at 4,096 — and the standard SAE ~3 (344 / 1,024 MiB), TopK ~3.3
#: (1,090 MiB at 4,096). At a smaller shape (8,192 latents, 256 wide, batch 1,024)
#: TopK held ~5 latent-sized tensors once the optimizer's weight-sized temporaries
#: are counted, so the default covers both measured shapes. Rounded up; an
#: architecture not listed takes the default.
STEP_LATENT_TENSORS = {"jumprelu": 12}
DEFAULT_STEP_LATENT_TENSORS = 5
#: Residual-width float32 tensors per batch token in one step (measured ~4).
STEP_WIDTH_TENSORS = 5


def training_step_bytes(
    *, batch_size: int, hidden_dim: int, latent_dim: int, architecture_type: Optional[str] = None
) -> int:
    """Peak bytes one SAE's training step allocates beyond its weights, gradients and Adam moments.

    Latent- and width-sized intermediates per batch token, plus one weight-sized
    temporary for the optimizer update. Keys step one after another, so this is
    per step, not per key.
    """
    latent_tensors = STEP_LATENT_TENSORS.get(str(architecture_type or ""), DEFAULT_STEP_LATENT_TENSORS)
    per_token = latent_tensors * int(latent_dim) + STEP_WIDTH_TENSORS * int(hidden_dim)
    return int(batch_size) * 4 * per_token + 2 * int(hidden_dim) * int(latent_dim) * 4


def sae_buffer_budget(
    gpu_free_bytes: int,
    *,
    num_keys: int,
    hidden_dim: int,
    latent_dim: int,
    batch_size: int,
    reserved_bytes: int = 0,
    architecture_type: Optional[str] = None,
) -> Dict[str, int]:
    """GPU bytes left for a training buffer once the SAEs' pending state is set aside.

    Pending: the Adam moments and gradients of every SAE (3 x float32 x its
    encoder and decoder). Overhead: one step's forward/backward intermediates,
    at least 1 GiB, and never less than :func:`training_step_bytes` for the
    architecture — the older per-key formula alone reserved 1 GiB for a
    single-layer JumpReLU step measured at ~2.8 GiB (review R1-B F5).
    ``reserved_bytes`` is whatever else must stay free beside the buffer — one
    held-out evaluation chunk on both paths, and one model forward on the fly —
    and comes out of the buffer one for one.
    """
    pending = int(num_keys) * 2 * int(hidden_dim) * int(latent_dim) * 4 * 3
    overhead = max(
        1024**3,
        int(batch_size) * (int(hidden_dim) + int(latent_dim)) * 4 * int(num_keys) * 3,
        training_step_bytes(
            batch_size=batch_size, hidden_dim=hidden_dim, latent_dim=latent_dim,
            architecture_type=architecture_type,
        ),
    )
    return {
        "pending_sae_bytes": pending,
        "training_overhead": overhead,
        "reserved_bytes": int(reserved_bytes),
        "available": int(gpu_free_bytes) - pending - overhead - int(reserved_bytes),
    }


def _log_storage_choice(choice, plan) -> None:
    """Say what a resume did with the checkpoint's activation storage plan. Silent on a fresh run."""
    if choice.outcome == activation_plan.REPLANNED:
        logger.warning(activation_plan.describe_replan(choice, plan))
    elif choice.outcome == activation_plan.NOT_RECORDED:
        logger.warning(
            "This checkpoint records no activation storage plan (it predates saved plans), so the "
            "buffer is planned from this process's memory: %s with %s tokens per layer, quotas %s. "
            "If that differs from the interrupted run's plan, its data position cannot be restored "
            "and the resume is refused.",
            plan["mode"], f"{plan['buffer_tokens']:,}", plan["quotas"],
        )
    elif choice.outcome == activation_plan.REUSED:
        logger.info(
            "Resume reuses the checkpoint's activation storage plan: %s with %s tokens per layer, "
            "quotas %s (this process could hold %s on the GPU and %s in RAM; a fresh plan would be %s)",
            plan["mode"], f"{plan['buffer_tokens']:,}", plan["quotas"],
            f"{choice.gpu_capacity_tokens:,}", f"{choice.ram_capacity_tokens:,}", choice.fresh,
        )


def hidden_size_of(architecture_config) -> Optional[int]:
    """The residual width a model row's recorded config names, or None."""
    if not isinstance(architecture_config, dict):
        return None
    for config in (architecture_config, architecture_config.get("text_config")):
        if isinstance(config, dict):
            for name in ("hidden_size", "d_model"):
                value = config.get(name)
                if isinstance(value, int) and value > 0:
                    return value
    return None


class _StopForward(Exception):
    """Raised once the deepest hooked layer has run: nothing above it is needed."""


def _raise_stop_forward(module, inputs, output):
    raise _StopForward()


class LayerCapture:
    """The on-the-fly path's reader: one no-grad forward, each (layer, hook) output on the CPU.

    Given to ``ModelActivationSource`` as its capture. The source ENTERS it around
    a refill, so hooks are registered once per refill (HookManager logs every
    registration) and removed before anything else — the spliced-CE evaluation —
    runs the model.

    THE FORWARD STOPS AFTER THE DEEPEST TRAINED LAYER. The layers above it and the
    LM head, whose logits (rows x seq x vocab) are the largest tensor of a whole
    forward, are never computed.

    INPUT IDS GO TO THE EMBEDDING'S CARD (``input_device``), which on a split model
    need not be ``device``, the SAEs' card.
    """

    def __init__(self, base_model, layers, hook_types, architecture, keys) -> None:
        self.base_model = base_model
        self.layers = list(layers)
        self.hook_types = list(hook_types)
        self.architecture = architecture
        self.keys = [tuple(key) for key in keys]
        self.model_input_device = input_device(base_model)
        self._manager = None
        self._stop_handle = None

    def __enter__(self) -> "LayerCapture":
        manager = HookManager(self.base_model)
        manager.register_hooks(self.layers, self.hook_types, self.architecture)
        self._manager = manager
        layers_module = manager.structure.layers_module
        present = [layer for layer in self.layers if layer < len(layers_module)]
        if present:
            # Registered AFTER HookManager's hooks, so a hook on the same module
            # (the residual stream) has captured before this one stops the forward.
            self._stop_handle = layers_module[max(present)].register_forward_hook(_raise_stop_forward)
        return self

    def __exit__(self, *exc_info) -> bool:
        if self._stop_handle is not None:
            self._stop_handle.remove()
            self._stop_handle = None
        if self._manager is not None:
            self._manager.remove_hooks()
            self._manager = None
        return False

    def __call__(self, padded_input_ids, attention_masks) -> Dict[tuple, torch.Tensor]:
        if self._manager is None:
            with self:
                return self(padded_input_ids, attention_masks)
        manager = self._manager
        manager.activations.clear()
        input_ids_tensor = torch.tensor(padded_input_ids, device=self.model_input_device)
        attention_mask_tensor = torch.tensor(attention_masks, device=self.model_input_device)
        try:
            with torch.no_grad():
                self.base_model(
                    input_ids=input_ids_tensor, attention_mask=attention_mask_tensor, use_cache=False,
                )
        except _StopForward:
            pass
        captured = {}
        for layer_idx, hook_type in self.keys:
            layer_key = f"layer_{layer_idx}_{hook_type}"
            if not manager.activations.get(layer_key):
                raise RuntimeError(
                    f"Failed to capture activations for layer {layer_idx}/{hook_type}. "
                    f"Hook registration failed. Available keys: {list(manager.activations.keys())}"
                )
            captured[(layer_idx, hook_type)] = manager.activations[layer_key][0]
        manager.activations.clear()
        return captured


def gpu_memory_across(devices) -> tuple:
    """``(allocated, reserved)`` in MB summed over every GPU in ``devices``.

    A split model's layers sit on each of its cards; reading one card reported a
    fraction of the job and hid growth on the others.
    """
    cards = [torch.device(d) for d in devices]
    cards = [d for d in cards if d.type == "cuda"]
    allocated = sum(torch.cuda.memory_allocated(d) for d in cards) / (1024 ** 2)
    reserved = sum(torch.cuda.memory_reserved(d) for d in cards) / (1024 ** 2)
    return allocated, reserved


def post_run_stop_reason(get_db, training_id, step) -> Optional[str]:
    """Why the post-run evaluation must stop now, or None (review R1-D, R1D-7).

    A Stop, a Pause, a deleted row or a lost GPU lease, read from the training row
    between the evaluation's batches with the loop's own rule (`stop_signal_for`).
    Module-level, not a closure in `train_sae_task`: the loop's status check stays
    the task's ONE call to `stop_signal_for`, which `test_gpu_claim_review_r1` pins.
    """
    with get_db() as db:
        row = db.query(Training).filter_by(id=training_id).first()
        signal = stop_signal_for(row, step, lease_lost=lease_lost_reason())
    if signal is None:
        return None
    return signal.get("detail") or signal.get("reason") or f"the training was {signal['status']}"


def completion_after_export(row) -> Dict[str, Any]:
    """What a training row becomes once the loop's full-length export is on disk (review R3-A, R2D-1, R2D-10).

    THE RUN IS COMPLETED THE MOMENT ITS EXPORT IS SAVED. The post-run evaluation used to
    run first, for minutes, on a row still RUNNING: a Stop there left CANCELLED beside a
    full-length export (import locked), Stop & Finalize replaced that export with the
    newest periodic checkpoint's, and a Pause left PAUSED, so a resume retrained up to a
    checkpoint interval. Decided (2026-09-15): once the full-length export exists, a Stop
    cancels only the evaluation and the run stays COMPLETED.

    - A live row (pending, initializing, running): completed, and evaluated.
    - CANCELLED or PAUSED, a Stop or Pause that landed after the loop's last status check
      while the last steps and the export ran, or COMPLETED by a finalize that raced them:
      completed from THIS export, and not evaluated, because the operator asked the job to stop.
    - FAILED: left as it is, and not evaluated. A failure record is not overwritten.
    - Deleted: nothing to write.

    Returns ``{"complete": bool, "evaluate": bool, "stopped_as": Optional[str]}``.
    """
    if row is None:
        return {"complete": False, "evaluate": False, "stopped_as": "deleted"}
    status = getattr(row.status, "value", row.status)
    if status == TrainingStatus.FAILED.value:
        return {"complete": False, "evaluate": False, "stopped_as": status}
    if status in (TrainingStatus.CANCELLED.value, TrainingStatus.PAUSED.value, TrainingStatus.COMPLETED.value):
        return {"complete": True, "evaluate": False, "stopped_as": status}
    return {"complete": True, "evaluate": True, "stopped_as": None}


def _not_evaluated_because(stopped_as: str) -> str:
    what = {
        TrainingStatus.CANCELLED.value: "was stopped",
        TrainingStatus.PAUSED.value: "was paused",
        TrainingStatus.COMPLETED.value: "was finalized from a checkpoint",
    }.get(stopped_as, f"became {stopped_as}")
    return (
        f"not run: the training {what} while its last steps ran. Its full-length export is saved, "
        "so the run is completed; run the evaluation again from the training to measure it"
    )


def run_post_run_evaluation(
    task,
    *,
    training_id,
    hp,
    models,
    placement,
    base_model,
    extractions,
    sae_mb,
    eval_sources=None,
    should_stop=None,
):
    """The post-run evaluation of this training's SAEs. Never raises (remediation item 6).

    WHAT THE SAEs COST THE MODEL, on blocks the training never read: base,
    spliced, mean-ablated and zero-ablated CE per layer, loss recovered, KL, L0,
    centred FVU and the all-layers-spliced CE, recorded in
    ``trainings.evaluation`` — not overloaded onto ``training_metrics`` columns.
    See ``services/training_evaluation.py``.

    BOTH PATHS. A cached-activation run (``extractions`` given) reads blocks at or
    above each extraction's ``max_samples``, and loads the base model here with
    the training's placement. An on-the-fly run already holds its model and passes
    ``eval_sources``: one ``EvalSource`` per dataset, whose ``candidate_rows`` are
    that dataset's held-out rows — the only rows the run guarantees it never
    trained on. With no held-out split there are none, and it records "skipped"
    with the reason.

    Runs AFTER the community export, while the job still holds its GPU lease, so
    a failure costs nothing but the evaluation.
    """
    from ..services.training_evaluation import (
        base_model_loader,
        run_evaluation,
        sources_from_extractions,
    )

    model_fields = None
    with task.get_db() as db:
        row = db.query(Training).filter_by(id=training_id).first()
        model_row = db.query(Model).filter_by(id=row.model_id).first() if row is not None else None
        if model_row is not None:
            model_fields = {
                "repo_id": model_row.repo_id,
                "quantization": model_row.quantization,
                "file_path": model_row.file_path,
            }

    return run_evaluation(
        get_db=task.get_db,
        training_id=training_id,
        hp=hp,
        saes=models,
        sources=lambda: (
            sources_from_extractions(extractions, settings.resolve_data_path, hp.get('dataset_weights'))
            if extractions else list(eval_sources or [])
        ),
        load_base_model=base_model_loader(
            model_fields=model_fields,
            placement=placement,
            sae_mb=sae_mb,
            loader=load_model_from_hf,
            resolve_path=settings.resolve_data_path,
            keep=base_model,
        ),
        trigger="post_run",
        extra={"task_id": getattr(getattr(task, "request", None), "id", None)},
        should_stop=should_stop,
    )


@get_celery_app().task(
    base=TrainingTask,
    bind=True,
    name="train_sae",
    acks_late=False,  # Acknowledge task when it STARTS (not completes) to prevent re-execution
    task_reject_on_worker_lost=True,  # Reject (don't requeue) if worker crashes
)
@gpu_job("training", request_from=training_request, can_split=training_can_split)
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

    # A buffer a previous training left on this task object — it ran outside
    # Celery, so after_return never closed it — still holds its GPU storage, its
    # pinned copy and its threads. Close it before anything measures free memory,
    # and before this training's buffer overwrites the only reference to it.
    self._close_activation_stream()

    # Get training record
    with self.get_db() as db:
        training = db.query(Training).filter_by(id=training_id).first()

        # IDEMPOTENCY CHECK: Skip if training is already completed
        # This prevents re-execution when tasks are requeued due to worker restarts
        if training is not None and training.status == TrainingStatus.COMPLETED.value:
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

        # A STOP ON A QUEUED TRAINING STANDS (review round 2, R2D-5). The Stop endpoint
        # marks any live row CANCELLED, and its revoke is inert on a solo pool and
        # forgotten by a restarted worker; this line then wrote INITIALIZING over the
        # CANCELLED row and the run trained to completion. Refused here, before
        # placement: no card is claimed and nothing is loaded.
        refused = start_refusal(training, "its task started")
        if refused is not None:
            logger.warning(
                f"Training {training_id} is {refused['status']} ({refused['reason']}); its task "
                "returns without training"
            )
            return refused

        # Update status to initializing, and NAME THE TASK THAT RUNS IT (review round 2,
        # R2D-3): a resumed or handed-off execution is the one the stuck-job janitor and
        # the GPU lease release must judge, not the task that paused.
        training.status = TrainingStatus.INITIALIZING.value
        executing_task_id = getattr(getattr(self, "request", None), "id", None)
        if executing_task_id:
            training.celery_task_id = str(executing_task_id)
        db.commit()

        # Extract hyperparameters
        hp = training.hyperparameters
        logger.info(f"Hyperparameters: {hp}")

        # What the job asked for, read from the ROW so a resume (which only
        # re-dispatches the id) asks for the same card. NULL is a row older than
        # GPU selection, which means auto.
        gpu_request = training.gpu_request or AUTO

        # REPRODUCIBILITY. `torch.manual_seed` was never called anywhere in the
        # training path, so weight init, threshold calibration and every batch
        # draw ran off the global unseeded RNG inside a Celery worker: two runs
        # of the same config were not comparable, and "run 2-3 seeds" could not
        # be attributed to anything. The one seed that did exist was a hardcoded
        # RandomState(42) for token subsampling, which made two supposedly
        # independent runs read the IDENTICAL token subset — the opposite
        # problem, and arguably worse.
        training_seed = hp.get('seed')
        if training_seed is None:
            # Derived from the training id so an existing run stays stable and a
            # new one differs, without silently pinning everything to one value.
            training_seed = zlib.crc32(training_id.encode("utf-8")) % (2**31 - 1)
            logger.info("No seed configured; derived %s from the training id", training_seed)
        training_seed = int(training_seed)
        torch.manual_seed(training_seed)
        np.random.seed(training_seed)
        logger.info("Training seed: %s", training_seed)

        # PERSIST IT, or the schema's promise that a multi-seed comparison is
        # "attributable" is false. A derived seed that is only logged is
        # recoverable solely by knowing to re-run crc32 on the training id.
        if hp.get('seed') != training_seed:
            hp['seed'] = training_seed
            try:
                with self.get_db() as db:
                    row = db.query(Training).filter(Training.id == training_id).first()
                    if row is not None:
                        merged = dict(row.hyperparameters or {})
                        merged['seed'] = training_seed
                        row.hyperparameters = merged
                        db.commit()
            except Exception as exc:  # noqa: BLE001 - bookkeeping must not kill a run
                logger.warning("Could not persist seed %s: %s", training_seed, exc)

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
        elif extraction_ids_of(training) is None:
            # ON THE FLY, THE WIDTH COMES FROM THE MODEL, and it must be known
            # BEFORE the SAEs are built below. It used to be corrected only after
            # the base model loaded — by then every SAE had the requested width,
            # so a form that sent the wrong one failed its first step with a
            # shape error. The model row records its config at download.
            model_row = db.query(Model).filter_by(id=training.model_id).first()
            recorded = hidden_size_of(getattr(model_row, "architecture_config", None))
            if recorded is not None and hp['hidden_dim'] != recorded:
                logger.warning(
                    "HIDDEN_DIM MISMATCH: requested %s, the model records %s; using the model's",
                    hp['hidden_dim'], recorded,
                )
                hp['hidden_dim'] = recorded

    try:
        # WILL ITS CHECKPOINTS FIT? (debt R2D-7 / R3D-15) Asked again here and not
        # only at create: a run can sit in the queue for hours behind other jobs, and
        # the volume it was measured against is shared with every one of them — other
        # trainings, extractions, model downloads. Refusing now costs a second;
        # discovering it at step 40,000 costs the run. A resume re-enters here too,
        # so a run paused for disk cannot be resumed back onto a still-full volume.
        with self.get_db() as db:
            disk = verdict_from_sync_session(
                db, hp, training_id=training_id, start_step=start_step
            )
        if not disk.fits:
            return pause_for_disk(training_id, step=start_step, message=disk.message())
        logger.info(disk.message())

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
                gpu=gpu_request,
            )
        else:
            # Multi-SAE training (multiple layers and/or hook types)
            memory_estimate = estimate_multilayer_training_memory(
                hidden_dim=hp['hidden_dim'],
                latent_dim=hp['latent_dim'],
                batch_size=batch_size,
                num_layers=num_sae_models,  # Total number of SAEs
                gpu=gpu_request,
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

        # WHICH CARD. Chosen against live free memory now, not at submit: a
        # queued job starts after whatever ran before it. A named card without
        # room raises GpuPlacementError, which the handler below records as the
        # job's error — never a silent move to another card. place_job also
        # makes the card CURRENT, so bare CUDA calls inside libraries land on it.
        #
        # A SPLIT ONLY FOR A JOB THAT LOADS A BASE MODEL. Training on cached
        # activations holds its SAEs, optimizer and rolling buffer on one card and
        # loads nothing that could be split. The on-the-fly path loads the model,
        # so it is sized WITH the model — Auto cannot choose a split without a
        # size — and may be split when no one card holds the model and the SAEs.
        loads_base_model = extraction_ids_of(training) is None
        base_mb = None
        if loads_base_model:
            with self.get_db() as db:
                model_row = db.query(Model).filter_by(id=training.model_id).first()
                base_mb = base_model_mb(
                    getattr(model_row, "params_count", None),
                    getattr(model_row, "quantization", None),
                    # A Q4 row's count is the packed count; its description is not.
                    getattr(model_row, "architecture_config", None),
                )
        placement = place_job(
            gpu_request,
            required_mb=memory_estimate['total_mb'] + (base_mb or 0),
            allow_shard=loads_base_model,
        )
        device = placement.device
        # Recorded BEFORE anything is loaded, so a job that dies loading still
        # says which card(s) it was on.
        with self.get_db() as db:
            placed_row = db.query(Training).filter_by(id=training_id).first()
            if placed_row is None:
                # Deleted while queued: there is nothing left to train for.
                logger.info(f"Training {training_id} was deleted before it started; not running it")
                return {"status": "cancelled", "step": 0, "reason": "deleted"}
            for column, value in placement.gpu_columns().items():
                setattr(placed_row, column, value)
            db.commit()
        logger.info(
            f"Training {training_id} placed on {placement.describe()} as {device} "
            f"(requested {gpu_request})"
        )

        # THE RESUME'S SAVED STATE IS READ BEFORE THE SCHEDULERS ARE BUILT (review A9).
        # A scheduler counts OPTIMIZER steps and is constructed with the accumulation
        # window so it can convert them to training steps, so the window has to be
        # known here — not after the schedulers already exist. Read once and reused by
        # the RESUME block below: training_state.pt is hundreds of megabytes and
        # reading it twice is real I/O on the GPU node.
        checkpoint_step = None
        checkpoint_step_dir = None
        saved_state = None
        if checkpoint_id:
            with self.get_db() as db:
                from ..models.checkpoint import Checkpoint
                ckpt = db.query(Checkpoint).filter_by(id=checkpoint_id).first()
                if not ckpt:
                    raise ValueError(f"Checkpoint not found: {checkpoint_id}")
                checkpoint_step = int(ckpt.step)
                ckpt_path = Path(settings.resolve_data_path(ckpt.storage_path))
            # ckpt_path is e.g. .../checkpoints/checkpoint_{step}/layer_{idx}_{hook}/checkpoint.safetensors
            # so ckpt_path.parent.parent is the per-step directory already.
            checkpoint_step_dir = ckpt_path.parent.parent
            saved_state = load_training_state(checkpoint_step_dir)

        #: The accumulation window for THIS RUN: the one the interrupted process used
        #: when resuming, else derived from the batch size (review A9).
        run_grad_accum_steps = resolve_grad_accum_steps(
            None if saved_state is None else saved_state.get("grad_accum_steps"),
            hp['batch_size'],
        )

        # Initialize models, optimizers, and schedulers (one per layer/hook_type combination)
        logger.info(f"Initializing {num_sae_models} SAE model(s)...")

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
                # JumpReLU ONLY. Passed for every architecture since 81381997, and
                # create_sae does not filter it out for the others, so every
                # Standard, Anthropic, Skip and Transcoder training raised
                # TypeError in SparseAutoencoder.__init__ before step 0.
                **({'ste_bandwidth': hp.get('ste_bandwidth', 0.5)} if architecture_type == 'jumprelu' else {}),
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

            # Learning rate: linear warmup, constant, and an opt-in linear decay
            # to 0 over the final `lr_decay_steps` (tracker item 7), in TRAINING
            # steps. The scheduler counts optimizer steps, so it is told the
            # accumulation factor; the inline lambda this replaced read its own
            # count as training steps and warmed up that many times too slowly.
            scheduler = build_lr_scheduler(
                optimizer,
                total_steps=hp['total_steps'],
                warmup_steps=hp.get('warmup_steps') or 0,
                decay_steps=hp.get('lr_decay_steps') or 0,
                grad_accum_steps=run_grad_accum_steps,
            )
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
            # A STOP OR PAUSE DURING SET-UP STANDS (review round 2, with R2D-5). Loading
            # a base model takes minutes, and this RUNNING write overwrote a Stop or a
            # Pause landing in that window, so the loop's status check then saw RUNNING.
            refused = start_refusal(training, "its first step")
            if refused is not None:
                if training is None:
                    logger.info(f"Training {training_id} was deleted before its first step; stopping")
                else:
                    logger.warning(
                        f"Training {training_id} became {refused['status']} during set-up; "
                        "stopping before its first step"
                    )
                return refused
            training.checkpoint_dir = str(checkpoint_dir)
            training.status = TrainingStatus.RUNNING.value
            db.commit()

        # Dead-latent statistics resampling reads (services/dead_latent_resampling.py).
        # Built on the CPU: each moves to its activations' device at the first step,
        # so set-up allocates nothing on the GPU before placement is proven.
        dead_latent_trackers = {
            key: DeadLatentTracker(hp['latent_dim']) for key in layer_hook_combinations
        }

        # RESUME (tracker item 4). A checkpoint id means: continue AFTER its step.
        #
        # A checkpoint is written after its step's update, so the run continues
        # at step + 1. This loop used to start AT the checkpoint's step and apply
        # that step twice, from an optimizer with zero moments and a warmup
        # restarted from 0, then re-initialise b_dec and re-calibrate thresholds
        # over the weights it had just loaded. The step's training_state.pt is
        # restored into the optimizers, schedulers, scalers and trackers here;
        # the RNG and the activation source are restored just before the first
        # step, after everything in set-up that draws random numbers.
        resumed_from_checkpoint = False
        resume_state = None
        if checkpoint_id:
            if start_step not in (0, checkpoint_step, checkpoint_step + 1):
                logger.warning(
                    f"Resume asked to start at step {start_step}, but checkpoint {checkpoint_id} "
                    f"is step {checkpoint_step}; continuing after the checkpoint"
                )
            start_step = checkpoint_step + 1
            logger.info(
                f"Resuming from checkpoint {checkpoint_id} (step {checkpoint_step}); "
                f"the next step is {start_step}"
            )

            for (layer_idx, hook_type), model in models.items():
                # Try new naming (layer_{idx}_{hook_type}) then legacy (layer_{idx})
                layer_ckpt = checkpoint_step_dir / f"layer_{layer_idx}_{hook_type}" / "checkpoint.safetensors"
                if not layer_ckpt.exists():
                    layer_ckpt = checkpoint_step_dir / f"layer_{layer_idx}" / "checkpoint.safetensors"
                if not layer_ckpt.exists():
                    # A warning used to go here, and the layer trained on from its
                    # fresh random initialisation as though it had been resumed.
                    raise ValueError(
                        f"Checkpoint step {checkpoint_step} has no weights for layer "
                        f"{layer_idx}/{hook_type} in {checkpoint_step_dir}; refusing a partial resume"
                    )
                CheckpointService.load_checkpoint(str(layer_ckpt), model=model, device=str(device))
                logger.info(f"Loaded checkpoint for layer {layer_idx}/{hook_type} from {layer_ckpt}")
            resumed_from_checkpoint = True

            if saved_state is None:
                logger.warning(
                    f"LEGACY CHECKPOINT: {checkpoint_step_dir} holds SAE weights only. Its optimizer "
                    "state was never saved, so Adam restarts from zero moments and the learning-rate "
                    "warmup restarts from step 0; the loss scale, dead-latent statistics, RNG and "
                    "data position restart too. The weights, b_dec and thresholds are kept as saved."
                )
            else:
                if int(saved_state["step"]) != checkpoint_step:
                    raise ValueError(
                        f"{checkpoint_step_dir} holds training state for step {saved_state['step']}, "
                        f"not {checkpoint_step}"
                    )
                resume_state = restore_training_state(
                    saved_state,
                    sae_keys=layer_hook_combinations,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    scalers=scalers,
                    dead_latent_trackers=dead_latent_trackers,
                    models=models,
                )
                logger.info(
                    f"Restored optimizer, scheduler, loss scale and dead-latent state from "
                    f"{checkpoint_step_dir}"
                )

        logger.info("Model initialized successfully")

        # Training loop configuration
        total_steps = hp['total_steps']
        batch_size = hp['batch_size']
        checkpoint_interval = hp.get('checkpoint_interval', 1000)
        log_interval = hp.get('log_interval', 100)

        #: What ONE checkpoint step costs, measured from the REAL SAE through the
        #: real writers (services/checkpoint_disk). The mid-accumulation figure —
        #: the larger of the two save shapes, because such a step also saves
        #: gradients — so the check before each save is made against the worst
        #: case that save might turn out to be.
        #: MERGE NOTE (WS-B + WS-C): this reads `hp` and `num_sae_models` only, never
        #: the accumulation window, so WS-C replacing the assignment below does not
        #: move the forecast.
        per_checkpoint_bytes = checkpoint_step_bytes(
            sae_footprint(hp), num_sae_models, include_grads=True
        )

        # Gradient accumulation settings: the same factor the LR scheduler was built
        # with, and on a resume the one the interrupted process used (review A9).
        grad_accum_steps = run_grad_accum_steps
        effective_batch_size = batch_size * grad_accum_steps
        if grad_accum_steps != grad_accum_steps_for(batch_size):
            logger.warning(
                "This run's accumulation window is %d, not the %d that batch size %d alone "
                "implies: the window was fixed when the run started and an OOM has reduced "
                "the batch since. Effective batch %d.",
                grad_accum_steps, grad_accum_steps_for(batch_size), batch_size,
                effective_batch_size,
            )
        if grad_accum_steps > 1:
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
        extraction_ids = extraction_ids_of(training)
        use_cached_activations = extraction_ids is not None and len(extraction_ids) > 0
        cached_activations = {}
        use_gpu_activations = True  # Default; may be set to False for cached activations on large models
        #: Set when the cached pool is larger than a buffer: batches then come
        #: from `activation_stream.next_batch`, which cycles the whole pool.
        activation_stream = None
        activation_stream_mode = False
        #: The storage plan the source is built from, saved in every checkpoint so a
        #: resume builds the same source instead of re-planning from whatever memory is
        #: free then (review round 1, R1D-1/R1D-2; services/activation_plan.py).
        activation_storage_plan = None
        storage_choice = None
        #: This run's resumes, as every later checkpoint records them.
        resume_history = list(resume_state["resume_history"]) if resume_state is not None else []
        dataset = None
        base_model = None
        tokenizer = None
        architecture = None
        hook_types = None

        #: Held-out activations per (layer, hook), on the CPU and evaluated in
        #: chunks at every log step. Assigned HERE, for both paths: the on-the-fly
        #: branch never assigned it, so its first log step read an unbound local
        #: and every on-the-fly training FAILED at step 0.
        holdout_activations = {}
        holdout_fraction = float(hp.get('holdout_fraction') or 0.0)
        holdout_eval_tokens = int(
            hp.get('holdout_eval_tokens') or holdout_evaluation.DEFAULT_HOLDOUT_EVAL_TOKENS
        )
        holdout_eval_chunk_tokens = int(
            hp.get('holdout_eval_chunk_tokens') or holdout_evaluation.DEFAULT_HOLDOUT_EVAL_CHUNK_TOKENS
        )
        #: What one evaluation chunk may allocate on the training card; both
        #: buffer budgets leave it free.
        holdout_eval_bytes = (
            holdout_evaluation.holdout_eval_peak_bytes(
                holdout_eval_chunk_tokens, hp['hidden_dim'], hp['latent_dim']
            ) if holdout_fraction > 0 else 0
        )

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
            #: extraction id -> flat indices of real (non-pad) token positions,
            #: or None when the mask could not be recovered. One entry per
            #: extraction; every layer of that extraction shares it.
            valid_masks_by_extraction = {}
            #: Token positions reserved for out-of-sample evaluation, by
            #: extraction. Empty unless `holdout_fraction` is configured.
            holdout_by_extraction = {}
            #: Extractions whose held-out split ran with no recorded mask, so their
            #: held-out documents include padding positions (review R1-D L7).
            holdout_unmasked_extractions = []

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

                    # PAD POSITIONS ARE NOT TRAINING DATA.
                    #
                    # Tokenization right-pads to `max_length`, and extraction saves
                    # the activation tensor whole — so without this the residual
                    # stream at PAD positions is sampled as if it were text. Measured
                    # on this estate: OpenWebText is 87.7% real tokens, Bloomberg 3.3%,
                    # and a four-corpus run is ~48% padding.
                    #
                    # Recovered once per extraction and shared by every layer of it;
                    # the mask is a property of the tokenization, not of the layer.
                    if extraction.id not in valid_masks_by_extraction:
                        mask, mask_source = activation_mask.load_valid_mask(
                            extraction_path, num_samples_in_file, seq_len
                        )
                        if mask is None:
                            # NEVER SILENT. Training on padding is the defect; doing
                            # it unknowingly is what made it survive this long.
                            logger.warning(
                                "PADDING NOT MASKED for extraction %s: %s. Every "
                                "padded position will be sampled as if it were a real "
                                "token. Re-extract to record the mask.",
                                extraction.id, mask_source,
                            )
                            valid_masks_by_extraction[extraction.id] = None
                            if holdout_fraction > 0:
                                # THE HELD-OUT SPLIT DOES NOT NEED THE MASK. It ran only
                                # in the masked branch below, so an extraction with no
                                # recoverable mask dropped `holdout_fraction` without a
                                # word: every document trained and no held-out row was
                                # ever written (review R1-B F4). Whole documents are held
                                # out over every position instead — padding included,
                                # exactly as training reads this extraction.
                                flat, held = activation_mask.split_documents(
                                    np.arange(num_samples_in_file * seq_len, dtype=np.int64),
                                    seq_len, holdout_fraction, training_seed,
                                )
                                holdout_by_extraction[extraction.id] = held
                                holdout_unmasked_extractions.append(extraction.id)
                                valid_masks_by_extraction[extraction.id] = flat
                                logger.info(
                                    "  Held out %s positions (%.0f%% of documents, padding "
                                    "included) from %s",
                                    f"{held.size:,}", holdout_fraction * 100, extraction.id,
                                )
                        else:
                            flat = activation_mask.valid_flat_indices(mask)
                            total_positions = mask.size
                            logger.info(
                                "  Real-token mask for %s from %s: %s/%s positions "
                                "(%.1f%% real)",
                                extraction.id, mask_source, f"{flat.size:,}",
                                f"{total_positions:,}",
                                flat.size / max(total_positions, 1) * 100,
                            )

                            # HOLD OUT WHOLE DOCUMENTS, not tokens. Adjacent
                            # positions share a prefix and a topic, so a
                            # token-level split leaks near-duplicates and the
                            # held-out number flatters the SAE for reasons
                            # unrelated to generalisation. Every number this
                            # project reports today is in-sample.
                            if holdout_fraction > 0:
                                flat, held = activation_mask.split_documents(
                                    flat, seq_len, holdout_fraction, training_seed
                                )
                                holdout_by_extraction[extraction.id] = held
                                logger.info(
                                    "  Held out %s tokens (%.0f%% of documents) from %s",
                                    f"{held.size:,}", holdout_fraction * 100,
                                    extraction.id,
                                )
                            valid_masks_by_extraction[extraction.id] = flat

                    valid_flat = valid_masks_by_extraction[extraction.id]
                    all_positions = num_samples_in_file * seq_len
                    total_tokens = all_positions if valid_flat is None else int(valid_flat.size)

                    key = (layer_idx, hook_type)
                    if key not in all_activation_parts:
                        all_activation_parts[key] = []
                    all_activation_parts[key].append({
                        'file': activation_file,
                        'num_samples': num_samples_in_file,
                        'seq_len': seq_len,
                        'total_tokens': total_tokens,
                        'valid_flat': valid_flat,
                        'holdout_flat': holdout_by_extraction.get(extraction.id),
                    })
                    logger.info(
                        f"    Shape: {layer_acts_mmap.shape} = {total_tokens:,} "
                        f"trainable token activations"
                    )

            if holdout_unmasked_extractions:
                # VISIBLE, NOT SILENT (review R1-D L7). The split ran, but these
                # extractions record no mask, so their held-out documents include
                # padding positions: not the evaluation a masked extraction gives.
                logger.warning(
                    "Held-out split WITHOUT a recorded mask for %s: whole documents are held out, "
                    "but their padding positions cannot be told apart, so the held-out evaluation "
                    "includes padding and is not comparable with a masked extraction's. Recorded in "
                    "the training's hyperparameters as holdout_unmasked_extractions.",
                    ", ".join(holdout_unmasked_extractions),
                )
                with self.get_db() as db:
                    row = db.query(Training).filter_by(id=training_id).first()
                    if row is not None:
                        # A NEW DICT: `hyperparameters` is a plain JSONB column, so an
                        # in-place edit is invisible to the session and never saved.
                        row.hyperparameters = {
                            **(row.hyperparameters or {}),
                            'holdout_unmasked_extractions': list(holdout_unmasked_extractions),
                        }
                        db.commit()

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
            # A resumed run has already restored the optimizer state the budget counts as pending.
            gpu_free = activation_plan.free_bytes_for_planning(
                gpu_free, resuming=resume_state is not None, optimizers=optimizers, models=models, device=device,
            )

            # SAE models are already loaded. Pending allocations (happen lazily during training):
            # - Adam optimizer: 2 momentum buffers per param × float32
            # - Gradients: 1 buffer per param × float32
            # Total pending: 3 × param_bytes per SAE
            # The held-out evaluation runs beside the buffer, one chunk at a time;
            # its tokens stay on the CPU. Sized with the extraction's real width.
            if holdout_fraction > 0:
                holdout_eval_bytes = holdout_evaluation.holdout_eval_peak_bytes(
                    holdout_eval_chunk_tokens, actual_hidden_dim, latent_dim
                )

            # Adam moments + gradients of every SAE, one step's intermediates
            # (at least 1 GB), and the evaluation chunk, out of free memory.
            budget = sae_buffer_budget(
                gpu_free,
                num_keys=num_layer_hooks,
                hidden_dim=actual_hidden_dim,
                latent_dim=latent_dim,
                batch_size=batch_size,
                reserved_bytes=holdout_eval_bytes,
                architecture_type=architecture_type,
            )
            pending_sae_bytes = budget["pending_sae_bytes"]
            training_overhead = budget["training_overhead"]
            gpu_mem_for_activations = budget["available"]

            logger.info(
                f"GPU memory budget: {gpu_total / 1024**3:.1f} GB total, "
                f"{gpu_free / 1024**3:.2f} GB free (after SAE models + CUDA context), "
                f"{pending_sae_bytes / 1024**3:.2f} GB pending (optimizer+gradients), "
                f"{training_overhead / 1024**3:.2f} GB training overhead, "
                f"{holdout_eval_bytes / 1024**3:.2f} GB held-out evaluation chunk, "
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

            # THE WHOLE POOL IS USED, NOT ONE SAMPLE OF IT.
            #
            # This used to load `min(total, GPU capacity)` tokens ONCE and draw
            # every batch from them with replacement for the whole run — ~2.4M
            # tokens per layer for LFM2.5-1.2B on a 24 GB card, so a 20M-token
            # extraction was ~90% unread and the rest was seen ~43 times over
            # the default 102M samples. When the pool did not fit on the GPU the
            # CPU path loaded EVERYTHING into pinned RAM as float32 instead.
            # Now a pool larger than the buffer is CYCLED through it (see
            # services/activation_buffer.py); a pool that fits loads once, as before.
            from ..services.activation_service import _available_memory_bytes

            max_ram_tokens_per_layer = max(
                0, int((_available_memory_bytes() or 0) * 0.5 / bytes_per_token / num_layer_hooks)
            )
            fresh_storage = activation_buffer.plan_activation_storage(
                total_available_tokens,
                max_gpu_tokens_per_layer,
                min_useful_tokens,
                max_ram_tokens_per_layer,
            )
            # A RESUME BUILDS THE RUN'S OWN BUFFER when its saved plan still fits.
            storage_choice = activation_plan.choose_storage_plan(
                saved_plan=resume_state["activation_plan"] if resume_state is not None else None,
                resuming=resume_state is not None,
                path="cached",
                source_tokens=[p['total_tokens'] for p in all_activation_parts[first_key]],
                fresh=fresh_storage,
                gpu_capacity_tokens=max_gpu_tokens_per_layer,
                ram_capacity_tokens=max_ram_tokens_per_layer,
            )
            storage_mode, tokens_to_load = storage_choice.mode, storage_choice.buffer_tokens
            use_gpu_activations = storage_mode.startswith("gpu")
            activation_stream_mode = storage_mode.endswith("rolling")
            # Guard the ratio: an all-pad recovered mask makes the total 0
            # and this line the first thing to fail, with a ZeroDivisionError
            # instead of the real problem.
            pct = (
                tokens_to_load / total_available_tokens * 100
                if total_available_tokens else 0.0
            )
            logger.info(
                f"Activation storage: {storage_mode} — buffer {tokens_to_load:,} of "
                f"{total_available_tokens:,} trainable tokens per layer ({pct:.1f}%); "
                f"GPU could hold {max_gpu_tokens_per_layer:,}, RAM {max_ram_tokens_per_layer:,}"
                + (". The pool is CYCLED through the buffer, so every token is used."
                   if activation_stream_mode else ".")
            )

            if total_available_tokens == 0:
                raise ValueError(
                    "No trainable tokens: every position in every selected "
                    "extraction is padding, or the recovered masks are empty. "
                    "Check the tokenization's attention_mask column."
                )

            # Generate a shared token selection (same tokens for all layers) so
            # every SAE sees the same positions.
            #
            # THE MIXTURE IS DECIDED HERE. Previously this drew uniformly over the
            # pooled index space, which made each source's share proportional to
            # its PADDED ROW COUNT — Bloomberg supplied 19.1% of sampled
            # activations for 1.2% of the real text. With padding excluded the
            # default is now proportional to real tokens, and `dataset_weights`
            # lets the operator state the mixture outright.
            requested_weights = hp.get('dataset_weights')
            part_counts = [p['total_tokens'] for p in all_activation_parts[first_key]]

            plan_quotas = None
            # A rolling buffer always takes quotas: a re-planned resume can cycle a
            # pool that would fit whole (activation_plan.choose_storage_plan).
            if requested_weights is not None or tokens_to_load < total_available_tokens or activation_stream_mode:
                allocation = (
                    storage_choice.quotas if storage_choice.quotas is not None
                    else dataset_mixture.allocate_tokens(part_counts, tokens_to_load, requested_weights)
                )
                plan_quotas = [int(a) for a in allocation]
                logger.info(
                    "Mixture: %s",
                    dataset_mixture.describe_mixture(
                        [e.id for e in extractions], allocation
                    ),
                )

                # SAY SO WHEN THE REQUEST COULD NOT BE HONOURED.
                #
                # `allocate_tokens` caps the budget at what exists, so when
                # everything fits — the CPU-streaming path always, and the GPU
                # path whenever the activations are small enough — a requested
                # 90/10 quietly becomes availability-proportional. The log line
                # above reports the REALISED split truthfully, which is easy to
                # read as confirmation that the request was applied.
                if requested_weights is not None:
                    requested = dataset_mixture.normalise_weights(
                        requested_weights, len(part_counts)
                    )
                    realised_total = sum(allocation) or 1
                    realised = [a / realised_total for a in allocation]
                    drift = max(
                        abs(r - w) for r, w in zip(realised, requested)
                    ) if requested else 0.0
                    if drift > 0.02:
                        logger.warning(
                            "dataset_weights could NOT be honoured: asked for %s, "
                            "got %s. Every source fits in the budget, so the whole "
                            "pool is used and the mixture follows availability. "
                            "Reduce the token budget to make weights binding.",
                            [round(w, 3) for w in requested],
                            [round(r, 3) for r in realised],
                        )
                if activation_stream_mode:
                    # No fixed subsample: `allocation` becomes the per-refill
                    # quota of each source, and the buffer cycles through all of it.
                    selected_indices = None
                    logger.info(
                        "Rolling buffer quotas per refill: %s of %s trainable tokens (seed=%s)",
                        [f"{a:,}" for a in allocation], f"{total_available_tokens:,}",
                        training_seed,
                    )
                else:
                    # Seeded so a run is reproducible and two seeds are genuinely
                    # different. This used to be a hardcoded RandomState(42), so
                    # every "independent" run read the identical token subset.
                    rng = np.random.RandomState(training_seed)
                    chunks = []
                    offset = 0
                    for count, take in zip(part_counts, allocation):
                        if take >= count:
                            local = np.arange(count, dtype=np.int64)
                        else:
                            local = np.sort(rng.choice(count, size=take, replace=False))
                        chunks.append(local + offset)
                        offset += count
                    selected_indices = (
                        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
                    )
                    logger.info(
                        "Selected %s / %s trainable token positions (seed=%s)",
                        f"{selected_indices.size:,}", f"{total_available_tokens:,}",
                        training_seed,
                    )
            else:
                selected_indices = None  # Load all tokens

            activation_storage_plan = activation_plan.build_plan(
                path="cached",
                mode=storage_mode,
                buffer_tokens=tokens_to_load,
                quotas=plan_quotas,
                source_tokens=part_counts,
                gpu_capacity_tokens=max_gpu_tokens_per_layer,
                ram_capacity_tokens=max_ram_tokens_per_layer,
                min_useful_tokens=min_useful_tokens,
            )
            _log_storage_choice(storage_choice, activation_storage_plan)

            # Load and flatten activations: (N, seq_len, d) -> (N*seq_len, d) per-token
            # SAE training requires individual token activations, NOT sequence averages.
            # Averaging over the sequence dimension destroys per-token variance and causes
            # degenerate training where b_dec alone explains the data (L0->0, all features die).
            # THE HELD-OUT SAMPLE, DRAWN ACROSS EVERY EXTRACTION. It used to take
            # each extraction's held-out positions lowest index first until a
            # 100,000-token cap, so a multi-extraction run evaluated the first
            # rows of the FIRST extraction and usually nothing else. Now each
            # extraction's share follows dataset_weights (equal when unset), and
            # whole held-out rows are taken in a seeded order. Every layer reads
            # the same positions.
            holdout_positions = [np.empty(0, dtype=np.int64) for _ in extractions]
            if holdout_by_extraction:
                held_counts = [
                    int(holdout_by_extraction[e.id].size) if e.id in holdout_by_extraction else 0
                    for e in extractions
                ]
                held_quotas = holdout_evaluation.holdout_quotas(
                    held_counts, holdout_eval_tokens, requested_weights
                )
                for idx, extraction in enumerate(extractions):
                    if extraction.id in holdout_by_extraction:
                        holdout_positions[idx] = holdout_evaluation.select_holdout_positions(
                            holdout_by_extraction[extraction.id],
                            all_activation_parts[first_key][idx]['seq_len'],
                            held_quotas[idx],
                            seed=training_seed,
                            source_index=idx,
                        )
                logger.info(
                    "Held-out evaluation sample (%s tokens per layer, chunks of %s): %s",
                    f"{sum(p.size for p in holdout_positions):,}", f"{holdout_eval_chunk_tokens:,}",
                    dataset_mixture.describe_mixture(
                        [e.id for e in extractions], [int(p.size) for p in holdout_positions]
                    ),
                )

            def _gather_holdout(mmap, part_idx, held_parts):
                """THE HELD-OUT HALF. Without this the split only DELETES
                training data and produces no out-of-sample number at all — a
                knob that costs you 10% of your corpus and returns nothing.
                Shared by the fixed-pool and rolling-buffer paths."""
                positions = holdout_positions[part_idx]
                if positions.size:
                    held_parts.append(activation_mask.gather_tokens(mmap, positions))

            for key in layer_hook_combinations:
                parts_info = all_activation_parts[key]

                # Build a flat array of all token activations from all extractions
                all_flat_parts = []
                held_parts = []
                cumulative_offset = 0
                for part_idx, part_info in enumerate(parts_info):
                    mmap = np.load(part_info['file'], mmap_mode='r')
                    n, s, d = mmap.shape

                    # `total_tokens` counts TRAINABLE positions, so both the
                    # subsample index space and the "load everything" case are
                    # already expressed in real tokens when a mask was recovered.
                    valid_flat = part_info.get('valid_flat')

                    if activation_stream_mode:
                        # Training tokens come from the rolling buffer, built
                        # below; only the held-out evaluation set is read here.
                        _gather_holdout(mmap, part_idx, held_parts)
                        continue

                    if selected_indices is not None:
                        # Determine which indices fall within this file's range
                        file_end = cumulative_offset + part_info['total_tokens']
                        local_mask = (selected_indices >= cumulative_offset) & (selected_indices < file_end)
                        local_indices = selected_indices[local_mask] - cumulative_offset
                        cumulative_offset = file_end

                        if len(local_indices) == 0:
                            continue
                    else:
                        cumulative_offset += part_info['total_tokens']
                        local_indices = None

                    # Resolve to positions in this file's flat (N*seq_len) space.
                    # With a mask, `local_indices` index the VALID subset, so they
                    # must be mapped through it — using them directly would silently
                    # read the wrong tokens rather than fail. Kept in one tested
                    # helper for exactly that reason.
                    flat_indices = activation_mask.resolve_flat_indices(
                        valid_flat, local_indices
                    )

                    if flat_indices is None:
                        # No mask and no subsample: the whole tensor is trainable,
                        # so a bulk reshape beats a gather.
                        chunk_size = 500  # samples at a time
                        chunk_parts = []
                        for ci in range(0, n, chunk_size):
                            ci_end = min(ci + chunk_size, n)
                            chunk = mmap[ci:ci_end].reshape(-1, d).astype(np.float32)
                            chunk_parts.append(chunk)
                        all_flat_parts.append(np.concatenate(chunk_parts, axis=0))
                    elif flat_indices.size:
                        all_flat_parts.append(
                            activation_mask.gather_tokens(mmap, flat_indices)
                        )

                    _gather_holdout(mmap, part_idx, held_parts)

                if all_flat_parts:
                    combined = np.concatenate(all_flat_parts, axis=0) if len(all_flat_parts) > 1 else all_flat_parts[0]
                    if use_gpu_activations:
                        cached_activations[key] = torch.from_numpy(combined).to(device)
                        storage_label = "GPU"
                    else:
                        # CPU path: pinned memory for faster GPU transfers during
                        # batch streaming — but only within the storage plan's RAM
                        # allowance. PyTorch pins a power-of-two block, up to twice
                        # the pool, locked and unswappable, for every layer.
                        tensor, pinned = activation_buffer.pin_within_budget(
                            torch.from_numpy(np.ascontiguousarray(combined)),
                            max_ram_tokens_per_layer * bytes_per_token,
                            f"activation pool {key}",
                        )
                        storage_label = "CPU (pinned)" if pinned else "CPU"
                        cached_activations[key] = tensor
                        del combined  # free numpy copy
                elif not activation_stream_mode:
                    raise ValueError(f"No activations loaded for {key}")

                if held_parts:
                    held = np.concatenate(held_parts, axis=0) if len(held_parts) > 1 else held_parts[0]
                    # ON THE CPU. Moving it to the card put up to 100,000 x d x
                    # layers beside the buffer, outside every budget; evaluation
                    # copies one chunk at a time instead.
                    holdout_activations[key] = torch.from_numpy(np.ascontiguousarray(held))
                    logger.info(
                        "  %s: %s held-out tokens reserved for out-of-sample eval (CPU)",
                        key, f"{holdout_activations[key].shape[0]:,}",
                    )

                if not activation_stream_mode:
                    logger.info(f"  {key}: {cached_activations[key].shape} on {storage_label} ({cached_activations[key].nbytes / 1024**3:.2f} GB)")

            # Get sample count and hidden dimension
            first_key = layer_hook_combinations[0]

            if activation_stream_mode:
                # One source per extraction. The positions handed over are the
                # TRAINABLE ones — padding and held-out documents already removed
                # — and `allocation` is each source's per-refill quota, so
                # dataset_weights govern every buffer, not just the first.
                buffer_sources = []
                for idx, extraction in enumerate(extractions):
                    ref = all_activation_parts[first_key][idx]
                    buffer_sources.append(activation_buffer.BufferSource(
                        label=extraction.id,
                        files={k: all_activation_parts[k][idx]['file'] for k in layer_hook_combinations},
                        num_rows=ref['num_samples'],
                        seq_len=ref['seq_len'],
                        valid_flat=ref['valid_flat'],
                    ))
                activation_stream = activation_buffer.RollingActivationBuffer(
                    buffer_sources,
                    layer_hook_combinations,
                    allocation,
                    seed=training_seed,
                    storage_device=device if use_gpu_activations else torch.device('cpu'),
                    train_device=device,
                    pin_memory=not use_gpu_activations,
                    # The storage plan's RAM allowance bounds the prepared copy of
                    # the next buffer; past it, refills stay synchronous.
                    host_ram_tokens=max_ram_tokens_per_layer,
                )
                # Its threads must stop however the task ends: after_return closes it.
                self._activation_stream = activation_stream
                # The same dict object, updated in place on every refill, so the
                # mean init, threshold calibration and diagnostics below read the
                # current buffer exactly as they read the fixed pool before.
                cached_activations = activation_stream.tensors

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
            # ON THE FLY: the base model reads the corpus while training runs. Each
            # tokenized dataset is a source of whole rows for a rolling buffer
            # (services/model_activation_source.py) — the cached path's contract
            # with the model as the reader.
            dataset_ids = training.dataset_ids if training.dataset_ids else [training.dataset_id]
            logger.info(f"Loading {len(dataset_ids)} dataset(s) and base model for activation extraction...")

            # THE MIXTURE IS POSITIONAL OVER dataset_ids ON THIS PATH, as it is
            # over extraction_ids on the cached one. It was ignored here with a
            # warning: datasets were concatenated and rows drawn uniformly, so a
            # corpus's share followed its row count. A length mismatch is refused
            # now, before the model is loaded.
            requested_weights = hp.get('dataset_weights')
            if requested_weights is not None:
                dataset_mixture.normalise_weights(requested_weights, len(dataset_ids))

            #: (dataset id, tokenized dataset, tokenization row), in dataset_ids order.
            token_datasets = []
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

                    # THE SECOND SELECTION SITE, and it had the same defect.
                    #
                    # An unordered `.first()` on (dataset, model) when the
                    # uniqueness constraint is (dataset, model, max_length) —
                    # several rows per pair are EXPECTED. Which window a run
                    # trained on was decided by Postgres row order. Fixing only
                    # the extraction path would be the "fixed one representative"
                    # pattern this repo keeps recording; worse, giving the
                    # directory name a max_length made the two rows point at
                    # genuinely different data, so this got MORE dangerous.
                    candidates = db.query(DatasetTokenization).filter(
                        DatasetTokenization.dataset_id == ds_id,
                        DatasetTokenization.model_id == training.model_id,
                    ).all()
                    tokenization = select_tokenization_for_model(
                        candidates, training.model_id, ds_id
                    ) if candidates else None
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
                    token_datasets.append((ds_id, ds, tokenization))
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
                # The placed card, whole — or, for a model no one card holds
                # beside the SAEs, accelerate's split over the placement's cards
                # with the SAEs' share kept free on theirs.
                device_map=placement.device_map,
                max_memory=budget_beside_sae(placement, memory_estimate['total_mb']),
                local_files_only=model_is_downloaded,
            )
            base_model.eval()
            # Where every batch's input ids go: the embedding's card, which on a
            # split need not be `device`, the SAEs' card.
            model_input_device = input_device(base_model)

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
                    # The SAEs above were built at hp['hidden_dim']. Overriding the
                    # number here (as this once did) left them the wrong width and
                    # failed the first step with a shape error; the width is now
                    # taken from the model row before they are built, so reaching
                    # this means the row records no width. Say so plainly.
                    raise ValueError(
                        f"The SAEs were built for hidden_dim={hp['hidden_dim']} but "
                        f"{model_record.repo_id} has hidden size {actual_hidden_dim}. Set "
                        f"hidden_dim to {actual_hidden_dim}."
                    )
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

            from ..services.activation_service import (
                _available_memory_bytes,
                micro_batch_size_for_length,
            )

            # HELD-OUT BLOCKS: whole rows, chosen by the cached path's rule
            # (activation_mask.choose_held_documents) over each tokenization's rows.
            token_sources = []
            held_rows_by_source = []
            # THE POST-RUN EVALUATION'S UNSEEN ROWS on this path are exactly the
            # held-out rows: training never reads them, and nothing else the run
            # does can promise that (remediation items 3 and 6).
            from ..services.training_evaluation import EvalSource

            onthefly_eval_sources = []
            requested_eval_weights = hp.get('dataset_weights')
            for ds_index, (ds_id, ds, ds_tokenization) in enumerate(token_datasets):
                train_rows, held_rows = activation_mask.split_rows(
                    len(ds), holdout_fraction, training_seed
                )
                width = getattr(ds_tokenization, "max_length", None)
                if not width:
                    width = len(model_activation_source.read_row(ds, 0)[0]) if len(ds) else 1
                token_sources.append(model_activation_source.TokenRowSource(
                    label=ds_id, dataset=ds, rows=train_rows, max_row_tokens=int(width),
                ))
                held_rows_by_source.append(held_rows)
                if held_rows.size:
                    onthefly_eval_sources.append(EvalSource(
                        label=str(ds_id),
                        # The path the dataset was loaded from above.
                        dataset_path=str(settings.resolve_data_path(ds_tokenization.tokenized_path)),
                        candidate_rows=tuple(int(row) for row in held_rows),
                        weight=float(requested_eval_weights[ds_index]) if requested_eval_weights else 1.0,
                    ))
                if holdout_fraction > 0:
                    logger.info(
                        "  Held out %s of %s rows (%.0f%%) of %s",
                        f"{held_rows.size:,}", f"{len(ds):,}", holdout_fraction * 100, ds_id,
                    )

            capture = LayerCapture(
                base_model, training_layers, hook_types, architecture, layer_hook_combinations
            )
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_token_id = int(pad_token_id) if pad_token_id is not None else 0
            max_row_tokens = max(src.max_row_tokens for src in token_sources)
            # The extraction's token-budget rule, so a 2,048-token window forwards
            # fewer rows at once than a 512-token one.
            micro_batch_rows = micro_batch_size_for_length(
                ON_THE_FLY_MICRO_BATCH_ROWS, max_row_tokens, ON_THE_FLY_FORWARD_TOKENS
            )

            # THE BUFFER'S SIZE: free memory on the SAEs' card AFTER the model and
            # the SAEs, less their pending optimizer state, a training step, one
            # forward and one held-out evaluation chunk.
            num_layer_hooks = len(layer_hook_combinations)
            hidden = hp['hidden_dim']
            bytes_per_token = hidden * 4
            gpu_tokens = 0
            if device.type == "cuda":
                try:
                    gpu_free, _gpu_total = torch.cuda.mem_get_info(device)
                except Exception:  # noqa: BLE001 - no reading, no GPU buffer
                    gpu_free = 0
                # A resumed run has already restored the optimizer state the budget counts as pending.
                gpu_free = activation_plan.free_bytes_for_planning(
                    gpu_free, resuming=resume_state is not None, optimizers=optimizers, models=models, device=device,
                )
                forward_reserve = max(1024**3, micro_batch_rows * max_row_tokens * bytes_per_token * 16)
                budget = sae_buffer_budget(
                    gpu_free,
                    num_keys=num_layer_hooks,
                    hidden_dim=hidden,
                    latent_dim=hp['latent_dim'],
                    batch_size=batch_size,
                    reserved_bytes=forward_reserve + holdout_eval_bytes,
                    architecture_type=architecture_type,
                )
                gpu_tokens = max(0, int(budget["available"] * 0.9 / bytes_per_token / num_layer_hooks))
            ram_tokens = max(
                0, int((_available_memory_bytes() or 0) * 0.5 / bytes_per_token / num_layer_hooks)
            )
            saved_onthefly_plan = resume_state["activation_plan"] if resume_state is not None else None
            if saved_onthefly_plan is not None and saved_onthefly_plan.get("path") == "on_the_fly":
                # NOT ESTIMATED AGAIN on resume: the quotas were allocated over the run's own
                # sample-based estimates, and an estimator changed between pause and resume
                # would otherwise move them (review round 1, reviewer B).
                available_tokens = [int(n) for n in saved_onthefly_plan["source_tokens"]]
            else:
                available_tokens = [
                    model_activation_source.estimate_real_tokens(src, training_seed, idx)
                    for idx, src in enumerate(token_sources)
                ]
            total_available_tokens = sum(available_tokens)
            if total_available_tokens == 0:
                raise ValueError(
                    "No trainable tokens: every sampled row of every selected dataset is "
                    "padding. Check the tokenization's attention_mask column."
                )
            onthefly_min_useful = min(total_available_tokens, max(50_000, batch_size * 20))
            fresh_storage = activation_buffer.plan_activation_storage(
                total_available_tokens,
                gpu_tokens,
                onthefly_min_useful,
                ram_tokens,
            )
            # A RESUME BUILDS THE RUN'S OWN BUFFER when its saved plan still fits.
            storage_choice = activation_plan.choose_storage_plan(
                saved_plan=resume_state["activation_plan"] if resume_state is not None else None,
                resuming=resume_state is not None,
                path="on_the_fly",
                source_tokens=available_tokens,
                fresh=fresh_storage,
                gpu_capacity_tokens=gpu_tokens,
                ram_capacity_tokens=ram_tokens,
            )
            storage_mode, buffer_tokens = storage_choice.mode, storage_choice.buffer_tokens
            use_gpu_activations = storage_mode.startswith("gpu")
            # Per-refill quotas of REAL tokens, by dataset_weights when set and by
            # (estimated) real tokens otherwise — the cached path's allocator.
            allocation = (
                storage_choice.quotas if storage_choice.quotas is not None
                else dataset_mixture.allocate_tokens(available_tokens, buffer_tokens, requested_weights)
            )
            activation_storage_plan = activation_plan.build_plan(
                path="on_the_fly",
                mode=storage_mode,
                buffer_tokens=buffer_tokens,
                quotas=allocation,
                source_tokens=available_tokens,
                gpu_capacity_tokens=gpu_tokens,
                ram_capacity_tokens=ram_tokens,
                min_useful_tokens=onthefly_min_useful,
            )
            _log_storage_choice(storage_choice, activation_storage_plan)
            # A resume that restores the data position replays the saved buffer in
            # load_state_dict, so the source below is built without a prefill (review
            # R1-B). The quotas above come from the checkpoint's own storage plan when it
            # still fits (R1D-2), so the replay loads under the interrupted run's quotas.
            saved_source = resume_state.get("activation_source") if resume_state is not None else None
            if not (isinstance(saved_source, dict)
                    and saved_source.get("kind") == model_activation_source.STATE_KIND):
                saved_source = None
            logger.info(
                "On-the-fly buffer: %s (%s tokens per layer; GPU could hold %s, RAM %s); "
                "estimated real tokens %s; mixture per refill: %s; %d rows per forward",
                storage_mode, f"{buffer_tokens:,}", f"{gpu_tokens:,}", f"{ram_tokens:,}",
                [f"{a:,}" for a in available_tokens],
                dataset_mixture.describe_mixture([src.label for src in token_sources], allocation),
                micro_batch_rows,
            )
            if requested_weights is not None:
                requested = dataset_mixture.normalise_weights(requested_weights, len(allocation))
                realised_total = sum(allocation) or 1
                drift = max(abs(a / realised_total - w) for a, w in zip(allocation, requested))
                if drift > 0.02:
                    logger.warning(
                        "dataset_weights could NOT be honoured on the fly: asked for %s, got %s. "
                        "A dataset's quota is capped at its estimated real tokens: either every "
                        "dataset fits in the buffer, or a small one is read whole on every refill.",
                        [round(w, 3) for w in requested],
                        [round(a / realised_total, 3) for a in allocation],
                    )

            if holdout_fraction > 0:
                # Each dataset's share is capped by rows x max_length: an upper bound on
                # its real held-out tokens, so the total is never under-counted. A
                # padding-heavy dataset can fall short of that share; the collection
                # then hands the shortfall to the datasets with held-out rows left, by
                # weight (review R1-B F3). Before, the shortfall was not evaluated.
                holdout_activations = model_activation_source.collect_holdout_activations(
                    [
                        (src.label, src.dataset, held)
                        for src, held in zip(token_sources, held_rows_by_source)
                    ],
                    layer_hook_combinations,
                    holdout_evaluation.holdout_quotas(
                        [int(held.size) * src.max_row_tokens for src, held in zip(token_sources, held_rows_by_source)],
                        holdout_eval_tokens,
                        requested_weights,
                    ),
                    capture=capture,
                    seed=training_seed,
                    micro_batch_rows=micro_batch_rows,
                    pad_token_id=pad_token_id,
                    weights=requested_weights,
                )

            model_stream = model_activation_source.ModelActivationSource(
                token_sources,
                layer_hook_combinations,
                allocation,
                capture=capture,
                hidden_dim=hidden,
                seed=training_seed,
                storage_device=device if use_gpu_activations else torch.device('cpu'),
                train_device=device,
                micro_batch_rows=micro_batch_rows,
                pad_token_id=pad_token_id,
                # Not on a resume that restores the position: load_state_dict replays
                # the saved buffer, and a prefill would be a full refill thrown away.
                prefill=saved_source is None,
            )
            # Released however the task ends: after_return closes it.
            self._activation_stream = model_stream
            activation_stream = model_stream
            activation_stream_mode = True
            cached_activations = model_stream.tensors
            num_samples = model_stream.size
            if saved_source is None:
                logger.info(
                    f"On-the-fly activations ready: {num_samples:,} tokens in the first buffer "
                    f"from {len(token_sources)} dataset(s), Model: {model_record.repo_id}"
                )
            else:
                logger.info(
                    f"On-the-fly source ready for {len(token_sources)} dataset(s), Model: "
                    f"{model_record.repo_id}; its buffer is replayed from the checkpoint before the first step"
                )
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
        # BOTH PATHS: `cached_activations` is the extraction pool, the rolling
        # buffer, or the on-the-fly source's first buffer. The on-the-fly path used
        # to skip this, so its SAEs started with b_dec = 0.
        # NOT ON RESUME: the loaded weights carry the b_dec the run trained. Re-deriving
        # it from the first buffer overwrote that with a fresh mean on every resume.
        if cached_activations and not resumed_from_checkpoint:
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
        if resume_state is not None:
            best_loss = resume_state["best_loss"]

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
        feature_firing_rate = {}  # Key: sae_key, Value: per-feature firing RATE
        ema_window_tokens = 50000  # Approximate token window for EMA decay
        if resume_state is not None:
            feature_activation_ema = {
                key: value.to(device) for key, value in resume_state["activation_ema"].items()
            }
            feature_firing_rate = {
                key: value.to(device) for key, value in resume_state["firing_rate"].items()
            }

        # Data-driven threshold calibration for JumpReLU
        # Sets thresholds from actual pre-activation distribution so features start active,
        # on both paths (the on-the-fly source's first buffer included).
        # NOT ON RESUME: the loaded thresholds are the trained ones, and calibrating
        # them against the first buffer threw the training away on every resume.
        if architecture_type == 'jumprelu' and cached_activations and not resumed_from_checkpoint:
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
                # Measured the way calibrate_thresholds set them: normalized and
                # centered. The raw `cal_batch @ W_enc.T` this used logged an L0
                # for an input the encoder never sees.
                with torch.no_grad():
                    _, cal_z = model.encode(model.normalize(cal_batch)[0], return_pre_activations=True)
                actual_l0 = (cal_z > thresholds.unsqueeze(0)).float().mean().item()
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
        if cached_activations:
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

        # THE DATA STREAM'S SOURCE: the rolling buffer, or — for a pool that fits in
        # memory — a sampler with its OWN generator, so a resample or a calibration
        # drawing random numbers cannot move the batches. Either way it is the one
        # object whose state_dict() a checkpoint saves (tracker item 4 contract).
        if use_cached_activations and activation_stream is None:
            activation_stream = activation_buffer.FixedPoolSampler(
                cached_activations, layer_hook_combinations, seed=training_seed
            )

        # LAST BEFORE THE FIRST STEP: the data position and the RNG. Everything in
        # set-up that draws random numbers has run by now, so the step after the
        # checkpoint draws exactly what an uninterrupted run would have drawn.
        if resume_state is not None:
            source_state = resume_state["activation_source"]
            if source_state is not None and activation_stream is not None:
                restored = activation_plan.restore_source_position(
                    activation_stream, source_state, storage_choice
                )
                if restored["bit_identical"]:
                    logger.info("Restored the activation source's position from the checkpoint")
                else:
                    logger.warning(
                        "RESUME IS NOT BIT-IDENTICAL: the activation source continues the checkpoint's "
                        "pass under the re-planned buffer; %s unserved tokens of the interrupted buffer "
                        "are skipped until their source's next pass%s",
                        f"{restored['tokens_skipped']:,}",
                        " (the checkpoint's fixed pool had no pass to continue; the rolling buffer starts "
                        "its first)" if restored["source_restarted"] else "",
                    )
                # RECORDED in every later checkpoint (training_state.pt and the rows).
                resume_history = resume_history + [activation_plan.resume_report(
                    checkpoint_step=checkpoint_step,
                    checkpoint_id=checkpoint_id,
                    choice=storage_choice,
                    plan=activation_storage_plan,
                    restored=restored,
                )]
            elif source_state is not None:
                logger.warning(
                    "The checkpoint recorded a data position, but this run has no activation "
                    "source to restore it into; its batches will not continue the interrupted run's"
                )
            restore_rng_state(resume_state["rng"])
        elif resumed_from_checkpoint:
            # A LEGACY weights-only checkpoint (review round 2, R2-A): only the weights
            # continued, and every later checkpoint must say so.
            resume_history = resume_history + [activation_plan.weights_only_resume_report(
                checkpoint_step=checkpoint_step,
                checkpoint_id=checkpoint_id,
                plan=activation_storage_plan,
            )]

        # THE HOST MEMORY THIS TRAINING RESERVED IS NOW ALLOCATED (multi-GPU
        # Phase 3, review round 1, R1-6): the rolling buffer and the pinned pool
        # exist by here, so MemAvailable already excludes them, and another job's
        # host RAM guard must stop subtracting this training's reserve on top.
        # AFTER THE RESTORE (review round 2, R2-A): a resumed on-the-fly source is
        # built empty (review R1-B) and filled by the restore above. Marked before it,
        # other jobs' guards stopped counting this job's reserve for the whole replay.
        from ..services.gpu_job_claim import host_reserve_allocated

        host_reserve_allocated()

        # The steps after the checkpoint run again, and log again. Their old rows
        # go first, or the table's unique key refuses the first re-logged step.
        if resumed_from_checkpoint:
            with self.get_db() as db:
                discarded = discard_metrics_after_step(db, training_id, start_step - 1)
            if discarded:
                logger.info(
                    f"Discarded {discarded} metric rows logged after checkpoint step {start_step - 1}; "
                    "the resumed run logs those steps again"
                )

        # Pause/cancel checks hit the database, so throttle them: a per-step
        # query adds up to total_steps round-trips on the GPU hot path.
        status_check_interval = min(25, max(1, log_interval))

        def stop_now(signal, at_step):
            logger.info(
                f"Training {signal['status']} at step {at_step}"
                + (" (its row was deleted)" if signal.get("reason") == "deleted" else "")
            )
            if signal.get("reason") == GPU_LEASE_LOST:
                # The job's own failure, not the operator's stop: record why.
                record_progress(
                    "training", training_id, status=TrainingStatus.FAILED.value,
                    error_message=(
                        f"Stopped at step {at_step}: {signal['detail']}. Resume it from its "
                        "last checkpoint."
                    ),
                )
            return signal

        def check_stop(at_step):
            """The loop's ONE status check: every caller passes the lease state.

            `terminal_is_a_stop=True` is the LOOP's opt-in: a row that has gone
            terminal under a running loop — a Stop & Finalize racing its own
            finalize to COMPLETED — must stop it. The post-run evaluation asks
            the same function WITHOUT this, because R3-A leaves the row COMPLETED
            while the evaluation runs (see `stop_signal_for`).
            """
            with self.get_db() as db:
                row = db.query(Training).filter_by(id=training_id).first()
                return stop_signal_for(
                    row, at_step, lease_lost=lease_lost_reason(), terminal_is_a_stop=True
                )

        # A STOP IS CHECKED AT THE END OF THE STEP BEFORE IT (R1-D L8), so a pause can
        # checkpoint the step it stops after. The check here covers the steps that one
        # did not: the first step of this process, and a step after an OOM retry.
        checked_step = None
        # A PAUSE CAUGHT BY THE CHECK ABOVE AFTER AN OOM RETRY (review R3-A). An OOM
        # `continue`s past the end-of-step check that would have caught it, so the
        # top-of-step check did, and returned with no checkpoint: the steps trained since
        # the newest one were repeated on resume, and a run paused before its first
        # periodic checkpoint could not be resumed at all (R2A-11). Instead, once this
        # process has trained a step, the Pause waits one step and pauses at that step's
        # end-of-step check, with the checkpoint every pause writes.
        pause_pending = False
        trained_a_step = False
        # Which part of the step an OOM came from (R1-D L5): the draw (a refill) or
        # the SAE step.
        oom_phase = "step"

        for step in range(start_step, total_steps):
            # Check for pause/stop signals (throttled)
            if step % status_check_interval == 0 and checked_step != step:
                signal = check_stop(step)
                if signal is not None:
                    if signal["status"] == "paused" and trained_a_step:
                        pause_pending = True
                    else:
                        return stop_now(signal, step)

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

                if activation_stream is not None or cached_activations:
                    # Every path draws from a buffer: the rolling buffer when a
                    # cached pool exceeds memory, the on-the-fly source when the
                    # model is the reader (every token used, none repeated within a
                    # pass), and the fixed pool otherwise.
                    oom_phase = "draw"
                    layer_activations = draw_cached_batch(
                        activation_stream, cached_activations,
                        layer_hook_combinations, batch_size, num_samples, device,
                    )
                    oom_phase = "step"

                    for layer_idx, hook_type in layer_hook_combinations:
                        cached = cached_activations[(layer_idx, hook_type)]

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

                # Train all SAEs (one per layer/hook_type combination)
                layer_losses = {}  # Key: (layer_idx, hook_type)
                layer_recon_losses = {}  # Reconstruction loss component
                layer_zero_losses = {}   # Bias-only baseline, previously discarded
                layer_l0_means = {}      # Active features per token, previously discarded
                layer_l1_losses = {}  # Sparsity loss: L1*alpha for L1 types, loss_l0 for JumpReLU
                layer_sparsities = {}
                layer_dead_neurons = {}
                layer_fvu = {}
                layer_fvu_centred = {}
                # THE PRE-CLIP TOTAL NORM, which `clip_grad_norm_` RETURNS and both
                # call sites used to discard — so `training_metrics.grad_norm` was
                # declared by the ORM, documented in data-model.md, accepted by
                # `log_metric`, and never written by anything. Reset with the other
                # per-step dicts: only an update step clips, so a step that does not
                # clip logs no norm rather than repeating the last one.
                layer_grad_norms = {}

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

                # THE RATE THIS STEP IS APPLIED WITH (review R2-C, L3), read before any
                # scheduler steps: the update that closes this step's accumulation
                # window uses it. It was read after `scheduler.step()`, so every row
                # carried the next window's rate (step 0 logged warmup x 1/warmup while
                # its update ran at x 0).
                step_lr = schedulers[layer_hook_combinations[0]].get_last_lr()[0]

                # THE STEP IS TWO PASSES OVER THE SAEs, NOT ONE (review R2C-5).
                #
                # Forward-and-backward for every SAE first, then the update for every
                # SAE. Interleaved, an OOM raised while SAE 3 ran its forward left SAEs
                # 1 and 2 having ALREADY taken their optimizer and scheduler steps for a
                # step the handler then abandoned (`continue` moves to the next step, it
                # does not retry this one) — so they ran permanently one optimizer step
                # and one LR position ahead of the rest, on a batch the others never saw.
                # Nothing detected it and nothing could undo it.
                #
                # Forward-and-backward is where the large transient allocations are, so
                # this is where an OOM almost always comes from; split this way it lands
                # before ANY SAE has updated, and the step is abandoned cleanly for all
                # of them. The per-SAE metrics stay in the first pass: every one of them
                # is computed from this step's forward (`loss`, `z`, `losses`), none
                # reads the weights, so the numbers are unchanged by the split.
                for layer_idx, hook_type in layer_hook_combinations:
                    sae_key = (layer_idx, hook_type)
                    x = layer_activations[sae_key]
                    model = models[sae_key]
                    optimizer = optimizers[sae_key]
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

                    # Store SAE metrics (keyed by (layer_idx, hook_type) tuple)
                    layer_losses[sae_key] = loss.item() * grad_accum_steps  # Undo accumulation scaling
                    layer_sparsities[sae_key] = (z != 0).float().mean().item()

                    # Extract loss components for detailed logging
                    recon_loss_val = losses.get('loss_reconstruction')
                    layer_recon_losses[sae_key] = recon_loss_val.item() if recon_loss_val is not None and hasattr(recon_loss_val, 'item') else None

                    # `loss_zero` (bias-only baseline) and `l0_mean` (active
                    # features PER TOKEN) were computed every step and discarded.
                    # l0_mean is the interpretable sparsity number — "47 features"
                    # — while the stored l0_sparsity is a fraction of d_sae.
                    zero_val = losses.get('loss_zero')
                    layer_zero_losses[sae_key] = (
                        zero_val.item() if hasattr(zero_val, 'item') else zero_val
                    )
                    l0_mean_val = losses.get('l0_mean')
                    layer_l0_means[sae_key] = (
                        l0_mean_val.item() if hasattr(l0_mean_val, 'item') else l0_mean_val
                    )

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
                        # A REAL FIRING RATE, kept ALONGSIDE the dead-neuron
                        # accumulator rather than replacing it. That accumulator
                        # is a batch-level indicator with an unnormalised update,
                        # so its steady state is window/batch_size — it answers
                        # "how long since this fired", not "how often". Reading it
                        # as a frequency would be wrong, and the dense tail was
                        # never read at all.
                        feature_firing_rate[sae_key] = feature_density.update_firing_rate(
                            feature_firing_rate.get(sae_key), z
                        )
                        # What resampling judges death by: consecutive steps since
                        # each latent last fired (`dead_neuron_threshold`'s unit).
                        dead_latent_trackers[sae_key].update(z)

                        fired = (z > 0).any(dim=0).float()  # [latent_dim] — 1 if any sample activated this feature
                        if sae_key not in feature_activation_ema:
                            feature_activation_ema[sae_key] = torch.zeros(hp['latent_dim'], device=z.device)
                        ema_decay = max(0.0, 1.0 - batch_size / ema_window_tokens)
                        feature_activation_ema[sae_key] = feature_activation_ema[sae_key] * ema_decay + fired
                        # Dead neurons: EMA below threshold means feature essentially never fires
                        layer_dead_neurons[sae_key] = (feature_activation_ema[sae_key] < 0.01).sum().item()

                    # Both FVUs — every architecture reports them (remediation
                    # item 5). `fvu` is the legacy global-mean value and keeps its
                    # column; `fvu_centred` is the headline, stored beside it.
                    fvu_val = losses.get('fvu', None)
                    if fvu_val is not None:
                        layer_fvu[sae_key] = fvu_val.item() if hasattr(fvu_val, 'item') else float(fvu_val)
                    else:
                        layer_fvu[sae_key] = None
                    fvu_centred_val = losses.get('fvu_centred', None)
                    layer_fvu_centred[sae_key] = (
                        None if fvu_centred_val is None
                        else fvu_centred_val.item() if hasattr(fvu_centred_val, 'item')
                        else float(fvu_centred_val)
                    )

                # PASS TWO: the update, for every SAE, reached only once no SAE can
                # still raise from its forward or backward (review R2C-5). Either they
                # all take this step or none of them does, so they cannot drift apart
                # in optimizer steps or in LR-schedule position.
                if (step + 1) % grad_accum_steps == 0:
                    for layer_idx, hook_type in layer_hook_combinations:
                        sae_key = (layer_idx, hook_type)
                        model = models[sae_key]
                        optimizer = optimizers[sae_key]
                        scheduler = schedulers[sae_key]
                        scaler = scalers.get(sae_key)  # None if CPU training

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
                                    # KEPT AS A TENSOR. Converting here would sync the
                                    # device every update step for a number read only
                                    # at `log_interval`; the log site converts it.
                                    layer_grad_norms[sae_key] = torch.nn.utils.clip_grad_norm_(
                                        model.parameters(), grad_clip_norm
                                    )

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
                                layer_grad_norms[sae_key] = torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), grad_clip_norm
                                )

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

                        # RELEASED ONCE APPLIED (review R2-C, R2C-3). Gradients were zeroed
                        # only at a window's first step, after the draw. An OOM before that
                        # point skipped the step, the next step is mid-window, and its
                        # backward accumulated onto this gradient, so the update closing
                        # that window applied it a second time.
                        optimizer.zero_grad()

                # Calculate aggregated metrics across all layers
                avg_loss = sum(layer_losses.values()) / len(layer_losses)
                avg_sparsity = sum(layer_sparsities.values()) / len(layer_sparsities)
                avg_dead_neurons = sum(layer_dead_neurons.values()) / len(layer_dead_neurons)
                # Calculate avg FVU only if any layer has FVU (JumpReLU)
                fvu_values = [v for v in layer_fvu.values() if v is not None]
                avg_fvu = float(sum(fvu_values) / len(fvu_values)) if fvu_values else None
                fvu_centred_values = [v for v in layer_fvu_centred.values() if v is not None]
                avg_fvu_centred = (
                    float(sum(fvu_centred_values) / len(fvu_centred_values))
                    if fvu_centred_values else None
                )
                # Calculate avg reconstruction and L1 losses
                recon_values = [v for v in layer_recon_losses.values() if v is not None]
                avg_recon_loss = float(sum(recon_values) / len(recon_values)) if recon_values else None
                l1_values = [v for v in layer_l1_losses.values() if v is not None]
                avg_l1_loss = float(sum(l1_values) / len(l1_values)) if l1_values else None
                zero_values = [v for v in layer_zero_losses.values() if v is not None]
                avg_zero_loss = float(sum(zero_values) / len(zero_values)) if zero_values else None
                l0_mean_values = [v for v in layer_l0_means.values() if v is not None]
                avg_l0_mean = float(sum(l0_mean_values) / len(l0_mean_values)) if l0_mean_values else None

                # Clear GPU cache after every step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Reset OOM retry count on successful step
                oom_retry_count = 0
                trained_a_step = True

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM error handling
                    oom_retry_count += 1
                    logger.warning(f"OOM error at step {step} (retry {oom_retry_count}/{max_oom_retries})")

                    if oom_retry_count >= max_oom_retries:
                        if oom_phase == "draw":
                            error_msg = (
                                f"Training failed after {max_oom_retries} OOM errors reading the "
                                "activation buffer; the batch size was not the cause."
                            )
                        else:
                            error_msg = f"Training failed after {max_oom_retries} OOM errors. Batch size too large."
                        logger.error(error_msg)
                        with self.get_db() as db:
                            training = db.query(Training).filter_by(id=training_id).first()
                            if training is not None:
                                training.status = TrainingStatus.FAILED.value
                                training.error_message = error_msg
                                db.commit()
                        raise RuntimeError(error_msg)

                    # A REFILL OOM IS NOT THE BATCH'S (R1-D L5). It came from reading the
                    # next buffer, which the buffer retries on the same rows; halving and
                    # persisting batch_size here changed the optimisation batch for the
                    # rest of the run and every resume over a transient failure.
                    if oom_phase == "draw":
                        oom_phase = "step"
                        logger.warning(
                            f"OOM reading the activation buffer at step {step}; retrying with the "
                            f"same batch size ({batch_size})"
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    # Reduce batch size and retry
                    old_batch_size = batch_size
                    batch_size = estimate_oom_reduced_batch_size(batch_size)
                    logger.info(f"Reducing batch_size from {old_batch_size} to {batch_size}")
                    # THE ACCUMULATION WINDOW DOES NOT WIDEN TO COMPENSATE (review A9).
                    # Widening it would change the training-step -> optimizer-step
                    # conversion the LR schedulers were built with, in the middle of the
                    # run, so the rest of the run would follow a curve it was not
                    # configured for. The smaller batch is taken honestly instead, said
                    # out loud here, and the window is recorded in every later checkpoint
                    # so a resume continues with THIS conversion rather than deriving a
                    # different one from the reduced batch size.
                    logger.warning(
                        "Effective batch size is now %d (%d x %d accumulation steps), down from %d; "
                        "the rest of this run trains with it",
                        batch_size * grad_accum_steps, batch_size, grad_accum_steps,
                        old_batch_size * grad_accum_steps,
                    )

                    # Clear GPU memory and reset scaler states
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # Recreate all GradScalers to ensure clean state.
                        # An OOM can interrupt the inner loop at any point,
                        # potentially leaving a scaler in a dirty state.
                        # AT THE SCALE EACH HAD REACHED, not the default (review A9).
                        for sae_key in scalers:
                            scalers[sae_key] = _fresh_scaler_at(scalers[sae_key].get_scale())

                    # EVERY SAE DROPS THE ABANDONED WINDOW'S GRADIENTS (review R2C-5).
                    # The SAEs are stepped in a loop, so an OOM part-way through it left
                    # the ones already backwarded holding gradients for a batch this step
                    # never applied, while the rest held none — and the update closing the
                    # window then applied that asymmetry, moving some SAEs on a batch the
                    # others never saw. Cleared for all of them, the window restarts from
                    # the same place for every SAE.
                    for sae_key in layer_hook_combinations:
                        optimizers[sae_key].zero_grad(set_to_none=True)

                    # Update hyperparameters
                    hp['batch_size'] = batch_size
                    with self.get_db() as db:
                        training = db.query(Training).filter_by(id=training_id).first()
                        if training is not None:
                            # A NEW DICT, not an in-place edit: `hyperparameters` is a
                            # plain JSONB column, so mutating it is invisible to the
                            # session and the reduced batch size was never saved.
                            training.hyperparameters = {
                                **(training.hyperparameters or {}), 'batch_size': batch_size
                            }
                            db.commit()

                    # Skip to next iteration with new batch size
                    continue
                else:
                    # Re-raise other runtime errors
                    raise

            # The rate this step's update applies (read before the schedulers stepped;
            # every SAE's scheduler follows the same curve).
            current_lr = step_lr

            # Log metrics periodically
            if step % log_interval == 0:
                # THROUGHPUT MONITORING: Check if training is proceeding at acceptable
                # speed. Measured over the steps THIS PROCESS ran, not the run's
                # absolute step number, so a resumed run reports its real rate (R2C-8).
                steps_this_process = step - start_step + 1
                actual_steps_per_min = steps_per_minute(
                    steps_this_process, time.time() - step_start_time
                )
                if steps_this_process >= 100 and actual_steps_per_min is not None:
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
                    # Every card the job holds, not only the SAEs' card.
                    gpu_memory_allocated, gpu_memory_reserved = gpu_memory_across(placement.all_devices)
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
                    if steps_this_process >= 100 and actual_steps_per_min is not None:
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
                    loss_reconstructed=avg_recon_loss,
                    loss_zero=avg_zero_loss,
                    l0_mean=avg_l0_mean,
                    l1_sparsity=avg_l1_loss,
                    l0_sparsity=avg_sparsity,
                    dead_neurons=int(avg_dead_neurons),
                    learning_rate=current_lr,
                    gpu_memory_used_mb=gpu_memory_mb,
                    layer_idx=None,  # Aggregated across all layers
                    hook_type=None,  # ...and every hook type
                    fvu=avg_fvu,  # legacy global-mean FVU
                    fvu_centred=avg_fvu_centred,  # the headline FVU
                )

                # OUT-OF-SAMPLE NUMBERS. Every figure this project reports
                # about an SAE is measured on data it trained on; batches are
                # drawn with replacement from one pool. These tokens are from
                # DOCUMENTS the model never saw.
                if holdout_activations:
                    for sae_key, held in holdout_activations.items():
                        model_ref = models.get(sae_key)
                        if model_ref is None:
                            continue
                        try:
                            # IN CHUNKS. This was one forward over every held-out
                            # token: ~6 GiB per latent-sized tensor at 100,000 tokens
                            # and 16,384 latents, an OOM this handler then swallowed.
                            # Eval mode and no_grad are the evaluator's, restored
                            # however it ends; every architecture reports both FVUs.
                            held_out = holdout_evaluation.evaluate_holdout(
                                model_ref, held, device, holdout_eval_chunk_tokens,
                                transcoder=(architecture_type == 'transcoder'),
                            )
                            logger.info(
                                "  L%s/%s HELD-OUT (%s tokens): fvu=%s fvu_centred=%s l0=%s",
                                sae_key[0], sae_key[1], f"{held_out['n_tokens']:,}",
                                f"{held_out['fvu']:.4f}" if held_out['fvu'] is not None else "n/a",
                                f"{held_out['fvu_centred']:.4f}" if held_out['fvu_centred'] is not None else "n/a",
                                f"{held_out['l0_mean']:.1f}" if held_out['l0_mean'] is not None else "n/a",
                            )
                            if held_out['fvu'] is not None:
                                # `fvu` keeps its stored meaning (the global-mean
                                # ratio); `fvu_centred` is the per-dimension-centred
                                # FVU, in its own column (remediation item 5).
                                self.log_metric(
                                    training_id=training_id,
                                    step=step,
                                    # This read `loss_total`, a key no SAE returns,
                                    # so every held-out row stored loss=0.0.
                                    loss=held_out['loss_reconstruction'] or 0.0,
                                    loss_reconstructed=held_out['loss_reconstruction'],
                                    loss_zero=held_out['loss_zero'],
                                    l1_sparsity=held_out['loss_l0'],
                                    l0_sparsity=held_out['l0_sparsity'],
                                    fvu=held_out['fvu'],
                                    fvu_centred=held_out['fvu_centred'],
                                    l0_mean=held_out['l0_mean'],
                                    # layer_idx is the only discriminator on
                                    # this table; a negative value marks the
                                    # row as held-out rather than in-sample,
                                    # so the two are never averaged together.
                                    layer_idx=-1 - int(sae_key[0]),
                                    # ...and the hook beside it, so the encoding
                                    # stays unambiguous with two hooks on a layer
                                    # (review R1-A, A5).
                                    hook_type=sae_key[1],
                                )
                        except Exception as exc:  # noqa: BLE001 - eval must not kill a run
                            logger.warning("Held-out evaluation failed for %s: %s", sae_key, exc)

                # THE DENSE TAIL. An SAE where a handful of latents fire on
                # most tokens scores a healthy aggregate L0 and is not
                # interpretable; nothing surfaced that before.
                for sae_key, rates in feature_firing_rate.items():
                    summary = feature_density.density_summary(rates)
                    if summary:
                        logger.info(
                            "  L%s/%s %s", sae_key[0], sae_key[1],
                            feature_density.describe(summary),
                        )

                # Log per-SAE metrics (one per layer/hook_type combination)
                for layer_idx, hook_type in layer_hook_combinations:
                    sae_key = (layer_idx, hook_type)
                    self.log_metric(
                        training_id=training_id,
                        step=step,
                        loss=layer_losses[sae_key],
                        loss_reconstructed=layer_recon_losses.get(sae_key),
                        loss_zero=layer_zero_losses.get(sae_key),
                        l0_mean=layer_l0_means.get(sae_key),
                        l1_sparsity=layer_l1_losses.get(sae_key),
                        l0_sparsity=layer_sparsities[sae_key],
                        dead_neurons=int(layer_dead_neurons[sae_key]),
                        learning_rate=current_lr,
                        # The norm BEFORE clipping, so a row shows whether the clip
                        # bit: a value above `grad_clip_norm` was clipped to it.
                        # None when this step did not clip (no `grad_clip_norm`, or
                        # an accumulation step that took no optimizer step).
                        grad_norm=(
                            float(layer_grad_norms[sae_key])
                            if sae_key in layer_grad_norms else None
                        ),
                        layer_idx=layer_idx,
                        # The SAE's hook: two hook types on one layer are two rows
                        # (review R1-A, A5). Before this, the second hook's row hit
                        # the unique key and the training failed at step 0.
                        hook_type=hook_type,
                        fvu=layer_fvu.get(sae_key),  # Per-SAE legacy FVU
                        fvu_centred=layer_fvu_centred.get(sae_key),  # Per-SAE headline FVU
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
                    fvu_centred=avg_fvu_centred,
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

                # Dead-latent resampling runs at its own gate after the step (see
                # "DEAD-LATENT RESAMPLING" above), not here.

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
                        "fvu_centred": avg_fvu_centred,
                        "dead_neurons": int(avg_dead_neurons),
                        "learning_rate": current_lr,
                        "num_layers": num_layers,
                        "num_hook_types": num_hook_types,
                        "num_sae_models": num_sae_models,
                        "training_layers": training_layers,
                        "hook_types": hook_types_config,
                    }
                )

            # DEAD-LATENT RESAMPLING (tracker item 1). Its own gate: inside the
            # logging block it ran only at common multiples of log_interval and
            # resample_interval. A latent is dead after `dead_neuron_threshold`
            # consecutive steps without firing. TopK is never resampled — its
            # auxiliary loss revives dead latents — and every other architecture
            # uses the repaired routine (services/dead_latent_resampling.py).
            #
            # AFTER THE LOG BLOCK, BEFORE THE CHECKPOINT (review R2-C, L1). Before the
            # log block, the held-out row of a step that was both a resample and a log
            # step scored the freshly re-initialised latents under that step's number.
            # Now it scores the model the step trained, the next log shows the
            # resample, and a checkpoint at this step still holds the resampled state
            # that a resume continues from.
            hp_fields = TrainingHyperparameters.model_fields
            if (
                hp.get('resample_dead_neurons', hp_fields['resample_dead_neurons'].default)
                and sparsity_type != 'topk'
                and resample_due(
                    step,
                    interval=hp.get('resample_interval') or hp_fields['resample_interval'].default,
                    warmup_steps=hp.get('warmup_steps') or 0,
                    sparsity_warmup_steps=sparsity_warmup_steps,
                    total_steps=total_steps,
                    lr_decay_steps=hp.get('lr_decay_steps') or 0,
                )
            ):
                dead_threshold = hp.get('dead_neuron_threshold') or hp_fields['dead_neuron_threshold'].default
                for sae_key in layer_hook_combinations:
                    tracker = dead_latent_trackers[sae_key]
                    dead = tracker.dead_mask(dead_threshold)
                    if not bool(dead.any()):
                        continue
                    result = resample_dead_latents(
                        models[sae_key], optimizers[sae_key], layer_activations[sae_key], dead
                    )
                    tracker.mark_revived(result.latents)
                    # WHICH latents, not just how many. `mark_revived` above zeroes
                    # the same counter `update` zeroes for every latent that fired
                    # this step, so after this line a checkpoint cannot tell a
                    # revived latent from one that simply fired. Recorded here, a
                    # later checkpoint's `firing_rate` can be read for exactly these
                    # latents — the drift measurement the acceptance checklist asks
                    # for, which was impossible before.
                    tracker.record_resample(step, result.latents)
                    if result.skipped_reason:
                        logger.info(f"L{sae_key[0]}/{sae_key[1]}: not resampled at step {step}: {result.skipped_reason}")
                    else:
                        logger.info(
                            f"L{sae_key[0]}/{sae_key[1]}: resampled {result.count} of {result.dead_before} dead "
                            f"latents at step {step} (none fired for {dead_threshold} steps); encoder rows "
                            f"at norm {result.encoder_norm:.4f}"
                        )

            # THE STOP CHECK FOR THE NEXT STEP, HERE (R1-D L8). Pausing used to save
            # nothing, so a pause lost every step since the newest periodic checkpoint.
            # Deciding it now lets a pause write the same checkpoint a periodic save
            # writes, at the same point in the step, so the resume continues exactly.
            stop_signal = None
            next_step = step + 1
            if next_step < total_steps and (next_step % status_check_interval == 0 or pause_pending):
                stop_signal = check_stop(next_step)
                checked_step = next_step
                pause_pending = False
            periodic_checkpoint = step % checkpoint_interval == 0 and step > 0
            pause_checkpoint = (
                stop_signal is not None
                and stop_signal["status"] == "paused"
                and not periodic_checkpoint
            )

            # Save checkpoint periodically, and when pausing
            if periodic_checkpoint or pause_checkpoint:
                # ROOM FOR THIS SAVE, ASKED IMMEDIATELY BEFORE IT. Free space is
                # shared, so the create-time and task-start answers are both already
                # stale by now: another training, an extraction or a model download
                # can consume the volume between two of this run's saves. Refusing to
                # BEGIN a save that cannot finish is what keeps the last good
                # checkpoint intact — a half-written one is worse than none.
                has_room, free_now, needed_now = room_for_one_step(
                    per_step_bytes=per_checkpoint_bytes, checkpoint_dir=checkpoint_dir,
                )
                if not has_room:
                    return pause_for_disk(
                        training_id, step=step,
                        message=(
                            f"the checkpoint volume has {free_now / 1024**3:.1f} GiB free and "
                            f"this step needs {needed_now / 1024**3:.1f} GiB including its "
                            f"reserve; paused at step {step} with its last complete checkpoint "
                            f"intact. Free space and resume."
                        ),
                    )
                logger.info(f"Saving checkpoint at step {step}...")

                try:
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

                    # EVERYTHING ELSE A RESUME NEEDS (tracker item 4), beside the weights
                    # and BEFORE the rows: a row names a checkpoint a resume may choose,
                    # so its state must already be on disk. is_best is decided first so
                    # the state carries the best loss this step was judged against.
                    # Gradients are saved only when the step ends mid-accumulation.
                    is_best = avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss
                    state_bytes = save_training_state(
                        Path(checkpoint_dir) / f"checkpoint_{step}",
                        build_training_state(
                            step=step,
                            sae_keys=layer_hook_combinations,
                            optimizers=optimizers,
                            schedulers=schedulers,
                            scalers=scalers,
                            dead_latent_trackers=dead_latent_trackers,
                            activation_ema=feature_activation_ema,
                            firing_rate=feature_firing_rate,
                            best_loss=best_loss,
                            activation_source=activation_stream,
                            models=models,
                            include_grads=(step + 1) % grad_accum_steps != 0,
                            activation_plan=activation_storage_plan,
                            resume_history=resume_history,
                            # MERGE NOTE (WS-B + WS-C): the two lanes edited adjacent
                            # lines here and git could not tell they were compatible.
                            # WS-B wrapped this save in the ENOSPC arm below; WS-C added
                            # this argument, so the state records the window the run
                            # actually used (A9). Keeping only one side would have been
                            # silent either way — no pause on a full disk, or a resumed
                            # run rebuilding accumulation from the wrong window.
                            grad_accum_steps=grad_accum_steps,
                        ),
                    )
                except OSError as exc:
                    # THE VOLUME FILLED MID-SAVE. Anything else is a real failure and
                    # must go on propagating to the task's handler untouched.
                    if not is_out_of_space(exc):
                        raise
                    # The half-written step directory HAS TO GO. It is visible to two
                    # consumers that read the filesystem rather than the rows:
                    # `training_finalize_service.list_checkpoint_steps` scans for
                    # `checkpoint_<n>` directories, so Stop & Finalize would happily
                    # rebuild this run's exported SAEs from a torn step. Removing it
                    # also returns the bytes, which is the difference between a pause
                    # that can be resumed and one that cannot write its own checkpoint.
                    freed = remove_partial_step(Path(checkpoint_dir) / f"checkpoint_{step}")
                    return pause_for_disk(
                        training_id, step=step,
                        message=(
                            f"the checkpoint volume filled while writing step {step} ({exc}); "
                            f"the partial checkpoint was removed "
                            f"({freed / 1024**3:.1f} GiB reclaimed) and the run is paused at its "
                            f"last complete checkpoint. Free space and resume."
                        ),
                    )
                logger.info(f"Saved training state for step {step} ({state_bytes / 1024**2:.1f} MB)")

                # Create checkpoint record for EACH SAE (multi-layer/multi-hook support)
                with self.get_db() as db:
                    from ..models.checkpoint import Checkpoint
                    from uuid import uuid4

                    if is_best:
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
                                # Visible through the checkpoints API: whether the weights
                                # went through a resume that was not bit-identical.
                                **({'resume_history': resume_history} if resume_history else {}),
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

            if stop_signal is not None:
                return stop_now(stop_signal, next_step)

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

        # COMPLETED IN THE COMMIT AFTER THE EXPORT, BEFORE THE EVALUATION (review R3-A,
        # R2D-1 and R2D-10; the rule is `completion_after_export`). The evaluation used to
        # run first on a RUNNING row, so a Stop, Stop & Finalize or Pause pressed during
        # it forfeited or replaced the final weights. Now the Stop and Pause endpoints
        # refuse the row, and a Stop reaches the evaluation through its record instead.
        # The post-run record is written in the same commit: `pending`, which a Stop
        # pressed before the evaluation's first write lands on, or `cancelled` when a
        # Stop or Pause landed while the last steps and the export ran.
        from datetime import datetime, UTC
        from ..services.training_evaluation import announce_evaluation, post_run_placeholder

        completed_at = datetime.now(UTC)
        executing_task_id = getattr(getattr(self, "request", None), "id", None)
        evaluation_enabled = bool(hp.get('evaluate_ce_delta', True))
        placeholder = None
        with self.get_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            completion = completion_after_export(training)
            stopped_as = completion["stopped_as"]
            if completion["complete"]:
                if evaluation_enabled:
                    placeholder = post_run_placeholder(
                        task_id=executing_task_id,
                        not_run_reason=None if completion["evaluate"] else _not_evaluated_because(stopped_as),
                    )
                    training.evaluation = placeholder
                training.status = TrainingStatus.COMPLETED.value
                training.progress = 100.0
                training.completed_at = completed_at
                # This export holds the final weights, whatever a raced finalize recorded.
                training.finalized_from_step = None
                db.commit()
        if not completion["complete"]:
            if stopped_as == "deleted":
                logger.warning(f"Training {training_id} finished after its row was deleted; nothing to mark completed")
                return {"status": "cancelled", "step": total_steps, "reason": "deleted"}
            logger.warning(
                f"Training {training_id} is {stopped_as} although it wrote its full-length export; "
                "left as it is and not evaluated"
            )
            return {"status": stopped_as, "steps": total_steps, "final_loss": avg_loss,
                    "reason": f"the training was {stopped_as} when its export was saved"}
        if stopped_as is not None:
            logger.warning(
                f"Training {training_id} {_not_evaluated_because(stopped_as)[len('not run: the training '):]}"
            )
        if placeholder is not None and placeholder["status"] != "pending":
            announce_evaluation(training_id, placeholder)

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

        # WHAT THE SAEs COST THE MODEL (remediation item 6).
        #
        # Every other number here — FVU, L0, dead count — lives in the SAE's own
        # space. None says what the reconstruction costs the MODEL: an SAE can
        # reach a low FVU and still wreck the next-token distribution.
        #
        # Run AFTER the export is saved, deliberately, while this job still holds
        # its GPU lease. A cached-activation run loads the base model here, so
        # first free what training held that the evaluation does not need — the
        # rolling buffer, the in-memory pool, optimizer and scheduler state.
        #
        # ONE RELEASE, ON BOTH PATHS (review R1-D, R1D-4 and R1D-5). The first
        # version released only on the cached path, and only the dict entries: the
        # step loop's locals (`cached`, `raw_acts`, `chunk`, `diag_batch`, the
        # batch `x`, the last `optimizer`) still referenced the pool or buffer and
        # the last SAE's Adam moments, so ~10 GB stayed allocated on a 3-layer 16K
        # run while the base model loaded. The on-the-fly path released nothing,
        # although its buffer was sized to fill what the model left. The model
        # itself stays on the on-the-fly path: the evaluation runs it.
        # The source closes IN PLACE (see `release_before_evaluation`); these locals go too.
        self.release_before_evaluation()
        activation_stream = model_stream = None
        cached_activations = {}
        holdout_activations = {}
        optimizers.clear()
        schedulers.clear()
        scalers.clear()
        cached = diag_batch = diag_z = diag_losses = raw_acts = chunk = normed_chunk = None
        cal_batch = cal_z = layer_activations = x = z = x_reconstructed = losses = None
        optimizer = scheduler = scaler = held = model_ref = None
        import gc as _gc

        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # A disabled evaluation still records itself as skipped; a stopped one was recorded above.
        if completion["evaluate"] or not evaluation_enabled:
            try:
                run_post_run_evaluation(
                    self,
                    training_id=training_id,
                    hp=hp,
                    models=models,
                    placement=placement,
                    base_model=base_model,
                    extractions=extractions if use_cached_activations else None,
                    sae_mb=memory_estimate['total_mb'],
                    eval_sources=None if use_cached_activations else onthefly_eval_sources,
                    should_stop=lambda: post_run_stop_reason(self.get_db, training_id, total_steps),
                )
            except Exception as exc:  # noqa: BLE001 - belt and braces: it records its own failures
                logger.warning("Post-run evaluation failed (training is safe): %s", exc)

        # Cleanup: Unload base model and SAE models from GPU
        logger.info("Cleaning up GPU memory...")
        del base_model
        del tokenizer
        for layer_idx, hook_type in layer_hook_combinations:
            sae_key = (layer_idx, hook_type)
            models.pop(sae_key, None)
            optimizers.pop(sae_key, None)
            schedulers.pop(sae_key, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GPU memory cleanup completed")

        # THE RUN STAYS COMPLETED (review R3-A, replacing R1-D R1D-6's "the Stop stands").
        # The Stop and Pause endpoints refuse a COMPLETED row, so CANCELLED or PAUSED can
        # reach it now only from a request that read the row before the commit above and
        # wrote after it. That request was a Stop or Pause of a run whose export is final:
        # it cancels the evaluation (which reads the row between batches) and nothing more.
        # A FAILED here is the janitor's or a lost lease's, and is left as it is.
        with self.get_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            if training is not None and training.status in (
                TrainingStatus.CANCELLED.value, TrainingStatus.PAUSED.value,
            ):
                logger.warning(
                    f"Training {training_id} was marked {training.status} after it completed with its "
                    "full-length export; it stays completed (only its evaluation was stopped)"
                )
                training.status = TrainingStatus.COMPLETED.value
                training.progress = 100.0
                training.completed_at = completed_at
                db.commit()

        result = {
            "status": "completed",
            "steps": total_steps,
            "final_loss": avg_loss,
        }
        if not completion["evaluate"]:
            result["reason"] = f"the post-run evaluation was not run: the training was {stopped_as} as it finished"
        return result

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
    Resume a paused training job from its newest complete (or a specified) checkpoint.

    NEWEST, NOT BEST. This used to prefer the ``is_best`` checkpoint, which
    discarded every step trained after the lowest-loss save and re-trained them
    from older weights. It now takes the newest step whose rows cover every SAE of
    the run with their files on disk (``select_resume_checkpoint``), and the run
    continues at the step AFTER it — a checkpoint is written after its step ran.

    Args:
        training_id: Training job ID
        checkpoint_id: Checkpoint ID to resume from (None = newest complete)

    Returns:
        Dictionary with resume result
    """
    from ..models.checkpoint import Checkpoint
    from ..core.database import get_sync_db

    logger.info(f"Resuming training {training_id} from checkpoint={checkpoint_id or 'newest complete'}")

    with get_sync_db() as db:
        training = db.query(Training).filter_by(id=training_id).first()
        if not training:
            raise ValueError(f"Training not found: {training_id}")
        # The resumed run waits on the queue its row's card request names.
        gpu_request = training.gpu_request

        if checkpoint_id:
            ckpt = db.query(Checkpoint).filter_by(id=checkpoint_id).first()
            if not ckpt:
                raise ValueError(f"Checkpoint not found: {checkpoint_id}")
            checkpoint_step, resolved_checkpoint_id = int(ckpt.step), ckpt.id
        else:
            hp = training.hyperparameters or {}
            layers = hp.get('training_layers', [0])
            layers = layers if isinstance(layers, list) else [layers]
            hooks = hp.get('hook_types', hp.get('hook_type', ['residual']))
            hooks = ([hooks] if isinstance(hooks, str) else list(hooks)) or ['residual']
            expected = [(int(layer), str(hook)) for layer in layers for hook in hooks]
            checkpoint_rows = db.query(Checkpoint).filter_by(training_id=training_id).all()
            chosen = select_resume_checkpoint(
                checkpoint_rows,
                expected,
                exists=lambda path: settings.resolve_data_path(path).exists(),
                # A step's weights are not enough to CONTINUE it (review R2A-9): the
                # newest step that also has its training_state.pt is preferred, so a
                # resume does not silently restart Adam, the warmup, the loss scale,
                # the RNG and the data position to save re-training a few steps.
                has_state=lambda path: training_state_path(
                    settings.resolve_data_path(path).parent.parent
                ).is_file(),
            )
            if chosen is not None:
                checkpoint_step, rows_by_key = chosen
                resolved_checkpoint_id = rows_by_key[expected[0]].id
            elif not checkpoint_rows and not (training.current_step or 0):
                # PAUSED BEFORE IT TRAINED A STEP (review round 2, with R2D-5): stopped
                # while queued or during set-up, which the task now honours. There is
                # nothing to resume from and nothing to lose: it starts at step 0.
                checkpoint_step, resolved_checkpoint_id = None, None
            else:
                # THE ROW MUST NOT KEEP CLAIMING IT IS RUNNING (review R2A-7).
                # `TrainingService.resume_training` sets RUNNING and clears the previous
                # failure's message BEFORE dispatching this task, so a refusal here left
                # a row reading RUNNING with no worker behind it and no error to explain
                # why — until `cleanup_stuck_trainings` reaped it half an hour later as
                # "no progress for N minutes ... a crashed worker or system issue",
                # which describes something that never happened. The status and the
                # message now agree, and they say what actually stopped the resume.
                from datetime import datetime, timezone

                refusal = (
                    f"No complete checkpoint found for training {training_id}. "
                    "Cannot resume without a saved checkpoint that has every layer. "
                    "Finalize it from a checkpoint, or retry it from step 0."
                )
                training.status = TrainingStatus.FAILED.value
                training.error_message = refusal
                training.completed_at = datetime.now(timezone.utc)
                db.commit()
                raise ValueError(refusal)

    if resolved_checkpoint_id is None and checkpoint_step is None:
        start_step = 0
        logger.warning(
            f"Training {training_id} has no checkpoint and never trained a step; it starts from step 0"
        )
    else:
        start_step = checkpoint_step + 1
        logger.info(
            f"Resuming training {training_id} after checkpoint step {checkpoint_step} "
            f"(checkpoint {resolved_checkpoint_id}); the next step is {start_step}"
        )
    dispatched = gpu_delay(train_sae_task, gpu_request)(
        training_id=training_id,
        start_step=start_step,
        checkpoint_id=resolved_checkpoint_id,
    )
    # THE ROW NAMES THE TASK THAT RUNS IT (review round 2, R2D-3). It kept the id of the
    # task that PAUSED, so the stuck-job janitor judged a finished task and
    # `release_reaped_leases` targeted the wrong holder. The resumed task also records
    # its own id when it starts; this covers the time it waits in the queue.
    resumed_task_id = getattr(dispatched, "id", None)
    if resumed_task_id:
        with get_sync_db() as db:
            row = db.query(Training).filter_by(id=training_id).first()
            if row is not None:
                row.celery_task_id = str(resumed_task_id)
                db.commit()
    return {
        "status": "queued",
        "start_step": start_step,
        "checkpoint_step": checkpoint_step,
        "checkpoint_id": resolved_checkpoint_id,
        "task_id": resumed_task_id,
    }


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
