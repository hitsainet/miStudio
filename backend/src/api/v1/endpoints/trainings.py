"""
Training API endpoints.

This module contains all FastAPI routes for SAE training management operations.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....models.training import TrainingStatus
from ....schemas.training import (
    TrainingCreate,
    TrainingUpdate,
    TrainingResponse,
    TrainingListResponse,
    TrainingMetricsListResponse,
    CheckpointListResponse,
    TrainingControlRequest,
    TrainingControlResponse,
)
from ....services.training_service import TrainingService
from ....services.checkpoint_service import CheckpointService
from ....workers.training_tasks import train_sae_task, resume_training_task
from ....core.celery_app import revoke_task
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trainings", tags=["trainings"])


@router.post("", response_model=TrainingResponse, status_code=201)
async def create_training(
    training: TrainingCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new SAE training job.

    Args:
        training: Training creation data
        db: Database session

    Returns:
        Created training job

    Raises:
        HTTPException: If training creation fails
    """
    try:
        db_training = await TrainingService.create_training(db, training)

        # MIS-E2E-104: generate the id BEFORE dispatching, and persist it before
        # the task can exist. The old order was `delay()` then write the id
        # returned — so between those two statements a GPU training was running
        # with nothing in the database recording which Celery task it was. A
        # failure in that window (or a worker restart) left an orphan the
        # janitors could not revoke, because revocation needs the id.
        #
        # `apply_async(task_id=...)` accepts a caller-supplied id, so the id
        # exists before anything is queued.
        task_id = str(uuid.uuid4())
        await TrainingService.start_training(db, db_training.id, task_id)
        train_sae_task.apply_async(args=[db_training.id], task_id=task_id)

        # Refresh to get updated record
        db_training = await TrainingService.get_training(db, db_training.id)

        return db_training
    except ValueError as e:
        # Validation failures (missing model/extraction, bad config) → 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create training")
        raise HTTPException(
            status_code=500,
            detail="Failed to create training"
        )


@router.get("", response_model=TrainingListResponse)
async def list_trainings(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    status: Optional[TrainingStatus] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db)
):
    """
    List training jobs with filtering and pagination.

    Args:
        page: Page number (1-indexed)
        limit: Items per page
        model_id: Filter by model ID
        dataset_id: Filter by dataset ID
        status: Filter by training status
        db: Database session

    Returns:
        Paginated list of training jobs with metadata
    """
    skip = (page - 1) * limit

    trainings, total = await TrainingService.list_trainings(
        db=db,
        model_id=model_id,
        dataset_id=dataset_id,
        status=status,
        skip=skip,
        limit=limit,
    )

    # Get status counts (independent of status filter)
    status_counts = await TrainingService.get_status_counts(
        db=db,
        model_id=model_id,
        dataset_id=dataset_id,
    )

    return {
        "data": trainings,
        "pagination": {
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit,
        },
        "status_counts": status_counts,
    }


@router.get("/{training_id}", response_model=TrainingResponse)
async def get_training(
    training_id: str = Path(..., description="Training job ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific training job by ID.

    Args:
        training_id: Training job ID
        db: Database session

    Returns:
        Training job details

    Raises:
        HTTPException: If training not found
    """
    db_training = await TrainingService.get_training(db, training_id)
    if not db_training:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    return db_training


# PATCH /api/trainings/{id} IS DELETED (MIS-E2E-106).
#
# `TrainingUpdate` is entirely lifecycle and worker-owned metric fields, and the
# `trainings` table has no user-editable column, so there was nothing left after
# removing the unsafe ones. The route had no caller — the frontend issues no
# PATCH, no MCP tool wraps it, no test exercised it.
#
# What it allowed: `{"status": "completed"}` on a running job unlocked SAE
# import from a partial checkpoint (`sae_manager_service` gates solely on
# `status != COMPLETED`) with no `finalized_from_step` marker, made the job
# uncancellable (`cancel_training` returns None for terminal statuses, so it
# silently no-ops while the worker keeps the GPU), and let the same request set
# `progress: 100` and `current_loss: 0.01` so the record agreed.
#
# `TrainingService.update_training` remains — the workers use it, and it now
# enforces its own allow-list.


@router.delete("/{training_id}", status_code=204)
async def delete_training(
    training_id: str = Path(..., description="Training job ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a training job and queue background file cleanup.

    This endpoint:
    1. Deletes the database record
    2. Queues a background Celery task to delete training files

    Args:
        training_id: Training job ID
        db: Database session

    Raises:
        HTTPException: If training not found
    """
    deletion_info = await TrainingService.delete_training(db, training_id)
    if not deletion_info:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    # Queue background file cleanup task
    training_dir = deletion_info.get("training_dir")
    if training_dir:
        from ....workers.training_tasks import delete_training_files
        delete_training_files.delay(
            training_id=training_id,
            training_dir=training_dir
        )
        logger.info(f"Queued file cleanup for training {training_id}: {training_dir}")


@router.post("/{training_id}/control", response_model=TrainingControlResponse)
async def control_training(
    control_request: TrainingControlRequest,
    training_id: str = Path(..., description="Training job ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Control a training job (pause/resume/stop).

    Args:
        training_id: Training job ID
        control_request: Control action to perform
        db: Database session

    Returns:
        Control response with new status

    Raises:
        HTTPException: If training not found or action fails
    """
    action = control_request.action

    try:
        if action == "pause":
            db_training = await TrainingService.pause_training(db, training_id)
            message = "Training paused"
        elif action == "resume":
            db_training = await TrainingService.resume_training(db, training_id)
            if db_training:
                resume_training_task.delay(training_id)
            message = "Training resumed"
        elif action == "stop":
            db_training = await TrainingService.stop_training(db, training_id)
            if db_training and db_training.celery_task_id:
                revoke_task(db_training.celery_task_id, terminate=True)
            message = "Training stopped"
        elif action == "stop_and_finalize":
            # Stop, then rebuild the SAEs from the newest checkpoint and write
            # community_format. Without the finalize step a stopped run leaves
            # usable checkpoints that nothing downstream can read.
            db_training = await TrainingService.stop_training(db, training_id)
            if db_training and db_training.celery_task_id:
                revoke_task(db_training.celery_task_id, terminate=True)
            message = "Training stopped"
            if db_training:
                from ....services.training_finalize_service import list_checkpoint_steps
                from ....workers.training_finalize_tasks import (
                    finalize_training_from_checkpoint_task,
                )

                # Only claim a finalize if there is actually a checkpoint to
                # finalize from — otherwise the user is told their SAE was saved
                # when nothing was written.
                if list_checkpoint_steps(training_id):
                    finalize_training_from_checkpoint_task.delay(training_id, None)
                    message = "Training stopped; finalizing from latest checkpoint"
                else:
                    message = (
                        "Training stopped, but it has no checkpoints to finalize from"
                    )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")

        if not db_training:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot {action} training: invalid state or not found"
            )

        return {
            "success": True,
            "training_id": training_id,
            "action": action,
            "status": TrainingStatus(db_training.status),
            "message": message,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to {action} training {training_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to {action} training"
        )


@router.get("/{training_id}/metrics", response_model=TrainingMetricsListResponse)
async def get_training_metrics(
    training_id: str = Path(..., description="Training job ID"),
    start_step: Optional[int] = Query(None, description="Start step (inclusive)"),
    end_step: Optional[int] = Query(None, description="End step (inclusive)"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum metrics to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get training metrics for a job.

    Args:
        training_id: Training job ID
        start_step: Start step (inclusive)
        end_step: End step (inclusive)
        limit: Maximum number of metrics
        db: Database session

    Returns:
        List of training metrics

    Raises:
        HTTPException: If training not found
    """
    # Verify training exists
    db_training = await TrainingService.get_training(db, training_id)
    if not db_training:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    metrics = await TrainingService.get_metrics(
        db=db,
        training_id=training_id,
        start_step=start_step,
        end_step=end_step,
        limit=limit,
    )

    return {"data": metrics}


@router.get("/{training_id}/checkpoints", response_model=CheckpointListResponse)
async def list_checkpoints(
    training_id: str = Path(..., description="Training job ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db)
):
    """
    List checkpoints for a training job.

    Args:
        training_id: Training job ID
        page: Page number
        limit: Items per page
        db: Database session

    Returns:
        Paginated list of checkpoints

    Raises:
        HTTPException: If training not found
    """
    # Verify training exists
    db_training = await TrainingService.get_training(db, training_id)
    if not db_training:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    skip = (page - 1) * limit

    checkpoints, total = await CheckpointService.list_checkpoints(
        db=db,
        training_id=training_id,
        skip=skip,
        limit=limit,
    )

    return {
        "data": checkpoints,
        "pagination": {
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit,
        }
    }


@router.get("/{training_id}/checkpoints/best", response_model=CheckpointListResponse)
async def get_best_checkpoint(
    training_id: str = Path(..., description="Training job ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the best checkpoint for a training job.

    Args:
        training_id: Training job ID
        db: Database session

    Returns:
        Best checkpoint

    Raises:
        HTTPException: If training not found or no checkpoints exist
    """
    checkpoint = await CheckpointService.get_best_checkpoint(db, training_id)
    if not checkpoint:
        raise HTTPException(
            status_code=404,
            detail=f"No best checkpoint found for training: {training_id}"
        )

    return {"data": [checkpoint]}


# ROUTE ORDER MATTERS: the literal "/checkpoints/prune*" paths must be declared
# BEFORE the parameterised "/checkpoints/{checkpoint_id}" route below. FastAPI
# matches in declaration order, so a parameterised route declared first would
# capture "prune-preview" as a checkpoint id.


@router.get("/checkpoints/prune-preview-all")
async def preview_checkpoint_prune_all(db: AsyncSession = Depends(get_db)):
    """Report what a sweep across EVERY training would delete. Read-only.

    The scheduled sweep's own summary reports `bytes_freed` only for what it
    actually deleted, so a dry-run pass reports 0 and cannot answer "how much
    would this reclaim?". Sizing the plan is the whole point of a preview, so
    it is computed here without touching anything.
    """
    from ....models.training import Training
    from ....services.checkpoint_retention import build_plan, load_policy

    def _preview(sync_db):
        policy = load_policy(sync_db)
        trainings = sync_db.query(Training).all()

        per_training = []
        total_checkpoints = 0
        total_bytes = 0
        skipped = {}

        for training in trainings:
            plan = build_plan(sync_db, training, policy)
            if plan.skipped_reason:
                skipped[plan.skipped_reason] = skipped.get(plan.skipped_reason, 0) + 1
                continue
            if not plan.checkpoint_ids:
                continue
            total_checkpoints += len(plan.checkpoint_ids)
            total_bytes += plan.estimated_bytes
            per_training.append({
                "training_id": plan.training_id,
                "delete_count": len(plan.checkpoint_ids),
                "keep_steps": plan.kept_steps,
                "estimated_bytes": plan.estimated_bytes,
            })

        per_training.sort(key=lambda r: r["estimated_bytes"], reverse=True)
        return {
            "policy": {
                "enabled": policy.enabled,
                "dry_run": policy.dry_run,
                "keep_last": policy.keep_last,
                "keep_best": policy.keep_best,
                "min_age_hours": policy.min_age_hours,
            },
            "trainings_scanned": len(trainings),
            "trainings_affected": len(per_training),
            "total_checkpoints": total_checkpoints,
            "estimated_bytes": total_bytes,
            "skipped": skipped,
            "per_training": per_training,
        }

    return {"data": await db.run_sync(lambda sync_db: _preview(sync_db))}


@router.post("/checkpoints/prune-all", status_code=202)
async def prune_all_checkpoints_now():
    """Sweep every training's checkpoints now.

    An explicit operator action, so it runs even when the DAILY sweep is
    disabled — the same rule the per-training route follows. Every safety guard
    still applies (never the best checkpoint, never the newest `keep_last`,
    never a training that could still resume, never anything under the minimum
    age), and `checkpoint_prune_dry_run` is still honoured.

    Exists because the sweep was previously reachable only from the scheduler,
    so reclaiming space meant previewing and pruning one training at a time.
    """
    from ....workers.prune_checkpoints import prune_checkpoints_task

    task = prune_checkpoints_task.delay(force=True)
    return {"data": {"task_id": task.id, "status": "queued", "scope": "all_trainings"}}


@router.get("/{training_id}/checkpoints/prune-preview")
async def preview_checkpoint_prune(
    training_id: str = Path(..., description="Training job ID"),
    db: AsyncSession = Depends(get_db),
):
    """Report which checkpoints the retention policy would delete.

    Strictly read-only, so it is safe to call at any time — a running training
    simply reports back as skipped rather than erroring.
    """
    from sqlalchemy import select

    from ....models.app_setting import AppSetting
    from ....models.checkpoint import Checkpoint
    from ....services.checkpoint_retention import (
        SETTING_KEYS,
        plan_from_checkpoints,
        policy_from_values,
    )

    db_training = await TrainingService.get_training(db, training_id)
    if not db_training:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    # load_policy() needs a sync session; read the same rows asynchronously and
    # hand them to the SHARED builder so this endpoint and the worker can never
    # disagree about what the policy is.
    setting_rows = await db.execute(
        select(AppSetting).where(AppSetting.key.in_(SETTING_KEYS))
    )
    policy = policy_from_values({r.key: r.value for r in setting_rows.scalars().all()})

    ckpt_rows = await db.execute(
        select(Checkpoint).where(Checkpoint.training_id == training_id)
    )
    checkpoints = list(ckpt_rows.scalars().all())

    plan = plan_from_checkpoints(
        training_id=training_id,
        training_status=db_training.status,
        checkpoints=checkpoints,
        policy=policy,
    )

    return {
        "data": {
            "training_id": training_id,
            "policy": {
                "enabled": policy.enabled,
                "dry_run": policy.dry_run,
                "keep_last": policy.keep_last,
                "keep_best": policy.keep_best,
                "min_age_hours": policy.min_age_hours,
            },
            "prunable_steps": plan.prunable_steps,
            "kept_steps": plan.kept_steps,
            "checkpoint_count": len(plan.checkpoint_ids),
            "estimated_bytes": plan.estimated_bytes,
            "skipped_reason": plan.skipped_reason,
        }
    }


@router.post("/{training_id}/checkpoints/prune", status_code=202)
async def prune_checkpoints_now(
    training_id: str = Path(..., description="Training job ID"),
    db: AsyncSession = Depends(get_db),
):
    """Prune this training's checkpoints now, bypassing the scheduler's enabled flag.

    This is an explicit operator action, so it runs even when the periodic
    pruner is disabled. Every SAFETY guard still applies (never the best
    checkpoint, never the newest steps, never an active training), and
    ``checkpoint_prune_dry_run`` is still honoured.
    """
    db_training = await TrainingService.get_training(db, training_id)
    if not db_training:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    from ....workers.prune_checkpoints import prune_single_training_task

    task = prune_single_training_task.delay(training_id)
    return {
        "data": {"training_id": training_id, "task_id": task.id, "status": "queued"}
    }


@router.delete("/{training_id}/checkpoints/{checkpoint_id}", status_code=204)
async def delete_checkpoint(
    training_id: str = Path(..., description="Training job ID"),
    checkpoint_id: str = Path(..., description="Checkpoint ID"),
    allow_best: bool = Query(
        False, description="Permit deleting a checkpoint flagged as best"
    ),
    db: AsyncSession = Depends(get_db),
):
    """Delete a single checkpoint and its file.

    The frontend has called this route for some time while it did not exist
    (silently 404ing), so implementing it also fixes the existing Checkpoint
    Management UI.
    """
    db_training = await TrainingService.get_training(db, training_id)
    if not db_training:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    checkpoint = await CheckpointService.get_checkpoint(db, checkpoint_id)
    if not checkpoint:
        raise HTTPException(
            status_code=404, detail=f"Checkpoint not found: {checkpoint_id}"
        )
    if checkpoint.training_id != training_id:
        # Never allow a checkpoint to be deleted via an unrelated training's URL.
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint {checkpoint_id} does not belong to training {training_id}",
        )

    from ....services.checkpoint_service import CheckpointFileDeleteError

    try:
        await CheckpointService.delete_checkpoint(
            db, checkpoint_id, delete_file=True, allow_best=allow_best
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except CheckpointFileDeleteError as e:
        # The row was deliberately kept so the file is not stranded — report the
        # failure rather than 204ing over a delete that did not happen.
        #
        # MIS-E2E-110: the message is NOT returned. It embeds the checkpoint's
        # `storage_path` and the raw OSError text, and this route is
        # unauthenticated — so a 500 handed a caller a real filesystem path and
        # an errno. Every other `detail=str(e)` in this API is a domain
        # exception on a 4xx, where the message IS the explanation the user
        # needs; this was the only one on a 500. The operator still gets the
        # path, in the log.
        logger.error(
            "Checkpoint %s: file delete failed, row kept to avoid stranding it: %s",
            checkpoint_id, e, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=("The checkpoint file could not be deleted, so the record was "
                    "kept rather than stranding the file. See the server log."),
        )

    return None


@router.post("/{training_id}/finalize", status_code=202)
async def finalize_training(
    training_id: str = Path(..., description="Training job ID"),
    checkpoint_step: Optional[int] = Query(
        None, description="Checkpoint step to finalize from; defaults to the newest"
    ),
    allow_failed: bool = Query(
        False, description="Permit finalizing a training whose run FAILED"
    ),
    force: bool = Query(
        False,
        description="Overwrite an existing export on an already-COMPLETED training",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Produce the Community Standard export for a stopped training.

    Stopping a run skips the training loop's finalize block, leaving usable
    checkpoints that no downstream consumer can read. This rebuilds the SAEs
    from a checkpoint and writes ``community_format/`` — which is what unlocks
    "Import to SAEs".
    """
    db_training = await TrainingService.get_training(db, training_id)
    if not db_training:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    # Reuse the retention module's definition so "active" has ONE meaning.
    # PAUSED belongs here: a paused run's Celery task is still alive and
    # resumable, and finalizing it would set status=COMPLETED underneath a job
    # that can later resume and overwrite it.
    from ....services.checkpoint_retention import ACTIVE_TRAINING_STATUSES

    if db_training.status in ACTIVE_TRAINING_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot finalize a {db_training.status} training; stop it first",
        )

    # A FAILED run crashed — finalizing would flip it to COMPLETED while its
    # error_message/traceback stay populated, and the SAE would then import as a
    # clean run. Require the caller to say explicitly that they want the SAE
    # from a crashed run.
    # A COMPLETED run already has the community_format written from its FINAL
    # weights. Re-finalizing rebuilds from an older checkpoint, overwriting those
    # weights and stamping finalized_from_step on a run that did go the distance —
    # so every already-extracted feature would disagree with the on-disk SAE.
    if db_training.status == TrainingStatus.COMPLETED.value and not force:
        raise HTTPException(
            status_code=409,
            detail=(
                "This training already completed and has an exported SAE. "
                "Re-send with force=true to overwrite it from a checkpoint."
            ),
        )

    if db_training.status == TrainingStatus.FAILED.value and not allow_failed:
        raise HTTPException(
            status_code=409,
            detail=(
                "This training FAILED; its checkpoints may predate the crash. "
                "Re-send with allow_failed=true to finalize anyway."
            ),
        )

    from ....workers.training_finalize_tasks import (
        finalize_training_from_checkpoint_task,
    )

    task = finalize_training_from_checkpoint_task.delay(training_id, checkpoint_step)
    return {
        "data": {
            "training_id": training_id,
            "task_id": task.id,
            "checkpoint_step": checkpoint_step,
            "status": "queued",
        }
    }
