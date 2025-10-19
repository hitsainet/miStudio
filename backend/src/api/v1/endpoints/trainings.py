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
from ....workers.training_tasks import train_sae_task

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

        # Start training task asynchronously
        task = train_sae_task.delay(db_training.id)

        # Update training with celery task ID
        await TrainingService.start_training(db, db_training.id, task.id)

        # Refresh to get updated record
        db_training = await TrainingService.get_training(db, db_training.id)

        return db_training
    except Exception as e:
        logger.error(f"Failed to create training: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create training: {str(e)}"
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

    return {
        "data": trainings,
        "pagination": {
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit,
        }
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


@router.patch("/{training_id}", response_model=TrainingResponse)
async def update_training(
    training_update: TrainingUpdate,
    training_id: str = Path(..., description="Training job ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a training job.

    Args:
        training_id: Training job ID
        training_update: Update data
        db: Database session

    Returns:
        Updated training job

    Raises:
        HTTPException: If training not found
    """
    db_training = await TrainingService.update_training(db, training_id, training_update)
    if not db_training:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")

    return db_training


@router.delete("/{training_id}", status_code=204)
async def delete_training(
    training_id: str = Path(..., description="Training job ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a training job.

    Args:
        training_id: Training job ID
        db: Database session

    Raises:
        HTTPException: If training not found
    """
    success = await TrainingService.delete_training(db, training_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Training not found: {training_id}")


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
            # TODO: Resume training task
            message = "Training resumed"
        elif action == "stop":
            db_training = await TrainingService.stop_training(db, training_id)
            # TODO: Cancel Celery task
            message = "Training stopped"
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
        logger.error(f"Failed to {action} training {training_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to {action} training: {str(e)}"
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
