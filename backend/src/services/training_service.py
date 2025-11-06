"""
Training service layer for business logic.

This module contains the TrainingService class which handles all
SAE training-related business logic and database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, UTC
import asyncio

from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.training import Training, TrainingStatus
from ..models.training_metric import TrainingMetric
from ..schemas.training import TrainingCreate, TrainingUpdate


def _emit_training_event_sync(training_id: str, event: str, data: Dict[str, Any]):
    """Helper to emit WebSocket events synchronously from async context."""
    try:
        from ..workers.websocket_emitter import emit_training_progress
        emit_training_progress(training_id, event, data)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to emit WebSocket event: {e}")


class TrainingService:
    """Service class for training job operations."""

    @staticmethod
    async def create_training(
        db: AsyncSession,
        training_data: TrainingCreate
    ) -> Training:
        """
        Create a new training job.

        Args:
            db: Database session
            training_data: Training creation data

        Returns:
            Created training object
        """
        # Generate training ID
        training_id = f"train_{uuid4().hex[:8]}"

        # Convert hyperparameters to dict
        hyperparameters_dict = training_data.hyperparameters.model_dump()

        # Create training record
        db_training = Training(
            id=training_id,
            model_id=training_data.model_id,
            dataset_id=training_data.dataset_id,
            extraction_id=training_data.extraction_id,
            status=TrainingStatus.PENDING.value,
            progress=0.0,
            current_step=0,
            total_steps=hyperparameters_dict['total_steps'],
            hyperparameters=hyperparameters_dict,
        )

        db.add(db_training)
        await db.commit()
        await db.refresh(db_training)

        # Emit training:created event
        _emit_training_event_sync(
            training_id=training_id,
            event="created",
            data={
                "training_id": training_id,
                "model_id": training_data.model_id,
                "dataset_id": training_data.dataset_id,
                "status": TrainingStatus.PENDING.value,
                "total_steps": hyperparameters_dict['total_steps'],
            }
        )

        return db_training

    @staticmethod
    async def get_training(
        db: AsyncSession,
        training_id: str
    ) -> Optional[Training]:
        """
        Get a training job by ID.

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            Training object or None if not found
        """
        result = await db.execute(
            select(Training).where(Training.id == training_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_trainings(
        db: AsyncSession,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[List[Training], int]:
        """
        List training jobs with optional filtering.

        Args:
            db: Database session
            model_id: Filter by model ID
            dataset_id: Filter by dataset ID
            status: Filter by status
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Tuple of (list of trainings, total count)
        """
        # Build query with filters
        query = select(Training)
        filters = []

        if model_id:
            filters.append(Training.model_id == model_id)
        if dataset_id:
            filters.append(Training.dataset_id == dataset_id)
        if status:
            filters.append(Training.status == status.value)

        if filters:
            query = query.where(and_(*filters))

        # Get total count
        count_query = select(func.count()).select_from(Training)
        if filters:
            count_query = count_query.where(and_(*filters))
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Get paginated results
        query = query.order_by(Training.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        trainings = list(result.scalars().all())

        return trainings, total

    @staticmethod
    async def get_status_counts(
        db: AsyncSession,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Get counts of trainings by status, with optional filtering.

        Args:
            db: Database session
            model_id: Filter by model ID
            dataset_id: Filter by dataset ID

        Returns:
            Dictionary with counts for each status
        """
        # Build base filters (not including status)
        base_filters = []
        if model_id:
            base_filters.append(Training.model_id == model_id)
        if dataset_id:
            base_filters.append(Training.dataset_id == dataset_id)

        # Count all trainings
        count_query = select(func.count()).select_from(Training)
        if base_filters:
            count_query = count_query.where(and_(*base_filters))
        total_result = await db.execute(count_query)
        all_count = total_result.scalar() or 0

        # Count running trainings
        running_filters = base_filters + [Training.status == TrainingStatus.RUNNING.value]
        count_query = select(func.count()).select_from(Training).where(and_(*running_filters))
        running_result = await db.execute(count_query)
        running_count = running_result.scalar() or 0

        # Count completed trainings
        completed_filters = base_filters + [Training.status == TrainingStatus.COMPLETED.value]
        count_query = select(func.count()).select_from(Training).where(and_(*completed_filters))
        completed_result = await db.execute(count_query)
        completed_count = completed_result.scalar() or 0

        # Count failed trainings
        failed_filters = base_filters + [Training.status == TrainingStatus.FAILED.value]
        count_query = select(func.count()).select_from(Training).where(and_(*failed_filters))
        failed_result = await db.execute(count_query)
        failed_count = failed_result.scalar() or 0

        return {
            "all": all_count,
            "running": running_count,
            "completed": completed_count,
            "failed": failed_count,
        }

    @staticmethod
    async def update_training(
        db: AsyncSession,
        training_id: str,
        training_update: TrainingUpdate
    ) -> Optional[Training]:
        """
        Update a training job.

        Args:
            db: Database session
            training_id: Training job ID
            training_update: Update data

        Returns:
            Updated training object or None if not found
        """
        db_training = await TrainingService.get_training(db, training_id)
        if not db_training:
            return None

        # Update fields
        update_data = training_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if field == 'status' and value:
                setattr(db_training, field, value.value)
            else:
                setattr(db_training, field, value)

        await db.commit()
        await db.refresh(db_training)

        return db_training

    @staticmethod
    async def delete_training(
        db: AsyncSession,
        training_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Delete a training job from database and return file paths for background cleanup.

        This method:
        1. Deletes the database record
        2. Returns training directory path for the caller to queue background deletion

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            Dictionary with deletion info, or None if not found
        """
        db_training = await TrainingService.get_training(db, training_id)
        if not db_training:
            return None

        # Capture training directory path before deletion
        # Training directory structure: /data/trainings/{training_id}/
        # checkpoint_dir is: /data/trainings/{training_id}/checkpoints/
        from pathlib import Path
        from ..core.config import settings

        if db_training.checkpoint_dir:
            # Extract training dir from checkpoint_dir
            training_dir = str(Path(db_training.checkpoint_dir).parent)
        else:
            # Construct training dir if checkpoint_dir was never set
            training_dir = str(settings.data_dir / "trainings" / training_id)

        # Delete database record
        await db.delete(db_training)
        await db.commit()

        return {
            "deleted": True,
            "training_id": training_id,
            "training_dir": training_dir,
        }

    @staticmethod
    async def start_training(
        db: AsyncSession,
        training_id: str,
        celery_task_id: str
    ) -> Optional[Training]:
        """
        Mark a training job as started.

        Args:
            db: Database session
            training_id: Training job ID
            celery_task_id: Celery task ID

        Returns:
            Updated training object or None if not found
        """
        db_training = await TrainingService.get_training(db, training_id)
        if not db_training:
            return None

        db_training.status = TrainingStatus.INITIALIZING.value
        db_training.celery_task_id = celery_task_id
        db_training.started_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_training)

        return db_training

    @staticmethod
    async def pause_training(
        db: AsyncSession,
        training_id: str
    ) -> Optional[Training]:
        """
        Pause a training job.

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            Updated training object or None if not found
        """
        db_training = await TrainingService.get_training(db, training_id)
        if not db_training:
            return None

        if db_training.status not in [TrainingStatus.RUNNING.value, TrainingStatus.INITIALIZING.value]:
            return None  # Can only pause running/initializing jobs

        db_training.status = TrainingStatus.PAUSED.value

        await db.commit()
        await db.refresh(db_training)

        # Emit training:status_changed event
        _emit_training_event_sync(
            training_id=training_id,
            event="status_changed",
            data={
                "training_id": training_id,
                "status": TrainingStatus.PAUSED.value,
                "current_step": db_training.current_step,
                "progress": db_training.progress,
            }
        )

        return db_training

    @staticmethod
    async def resume_training(
        db: AsyncSession,
        training_id: str,
        celery_task_id: Optional[str] = None
    ) -> Optional[Training]:
        """
        Resume a paused training job.

        Args:
            db: Database session
            training_id: Training job ID
            celery_task_id: New Celery task ID (if restarting)

        Returns:
            Updated training object or None if not found
        """
        db_training = await TrainingService.get_training(db, training_id)
        if not db_training:
            return None

        if db_training.status != TrainingStatus.PAUSED.value:
            return None  # Can only resume paused jobs

        db_training.status = TrainingStatus.RUNNING.value
        if celery_task_id:
            db_training.celery_task_id = celery_task_id

        await db.commit()
        await db.refresh(db_training)

        # Emit training:status_changed event
        _emit_training_event_sync(
            training_id=training_id,
            event="status_changed",
            data={
                "training_id": training_id,
                "status": TrainingStatus.RUNNING.value,
                "current_step": db_training.current_step,
                "progress": db_training.progress,
            }
        )

        return db_training

    @staticmethod
    async def stop_training(
        db: AsyncSession,
        training_id: str
    ) -> Optional[Training]:
        """
        Stop a training job (cancel).

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            Updated training object or None if not found
        """
        db_training = await TrainingService.get_training(db, training_id)
        if not db_training:
            return None

        if db_training.status in [TrainingStatus.COMPLETED.value, TrainingStatus.FAILED.value, TrainingStatus.CANCELLED.value]:
            return None  # Already terminal state

        db_training.status = TrainingStatus.CANCELLED.value
        db_training.completed_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_training)

        # Emit training:status_changed event
        _emit_training_event_sync(
            training_id=training_id,
            event="status_changed",
            data={
                "training_id": training_id,
                "status": TrainingStatus.CANCELLED.value,
                "current_step": db_training.current_step,
                "progress": db_training.progress,
            }
        )

        return db_training

    @staticmethod
    async def mark_training_failed(
        db: AsyncSession,
        training_id: str,
        error_message: str,
        error_traceback: Optional[str] = None
    ) -> Optional[Training]:
        """
        Mark a training job as failed.

        Args:
            db: Database session
            training_id: Training job ID
            error_message: Error message
            error_traceback: Full error traceback

        Returns:
            Updated training object or None if not found
        """
        db_training = await TrainingService.get_training(db, training_id)
        if not db_training:
            return None

        db_training.status = TrainingStatus.FAILED.value
        db_training.error_message = error_message
        db_training.error_traceback = error_traceback
        db_training.completed_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_training)

        # Emit training:status_changed event
        _emit_training_event_sync(
            training_id=training_id,
            event="failed",
            data={
                "training_id": training_id,
                "status": TrainingStatus.FAILED.value,
                "error_message": error_message,
                "current_step": db_training.current_step,
                "progress": db_training.progress,
            }
        )

        return db_training

    @staticmethod
    async def mark_training_completed(
        db: AsyncSession,
        training_id: str
    ) -> Optional[Training]:
        """
        Mark a training job as completed.

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            Updated training object or None if not found
        """
        db_training = await TrainingService.get_training(db, training_id)
        if not db_training:
            return None

        db_training.status = TrainingStatus.COMPLETED.value
        db_training.progress = 100.0
        db_training.completed_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_training)

        # Emit training:completed event
        _emit_training_event_sync(
            training_id=training_id,
            event="completed",
            data={
                "training_id": training_id,
                "status": TrainingStatus.COMPLETED.value,
                "final_loss": db_training.current_loss,
                "total_steps": db_training.total_steps,
                "progress": 100.0,
            }
        )

        return db_training

    @staticmethod
    async def add_metric(
        db: AsyncSession,
        training_id: str,
        step: int,
        loss: float,
        **kwargs
    ) -> TrainingMetric:
        """
        Add a training metric record.

        Args:
            db: Database session
            training_id: Training job ID
            step: Training step
            loss: Total loss
            **kwargs: Additional metric fields

        Returns:
            Created training metric object
        """
        db_metric = TrainingMetric(
            training_id=training_id,
            step=step,
            loss=loss,
            **kwargs
        )

        db.add(db_metric)
        await db.commit()
        await db.refresh(db_metric)

        return db_metric

    @staticmethod
    async def get_metrics(
        db: AsyncSession,
        training_id: str,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        limit: int = 1000,
    ) -> List[TrainingMetric]:
        """
        Get training metrics for a job.

        Args:
            db: Database session
            training_id: Training job ID
            start_step: Start step (inclusive)
            end_step: End step (inclusive)
            limit: Maximum number of records

        Returns:
            List of training metrics
        """
        query = select(TrainingMetric).where(TrainingMetric.training_id == training_id)

        if start_step is not None:
            query = query.where(TrainingMetric.step >= start_step)
        if end_step is not None:
            query = query.where(TrainingMetric.step <= end_step)

        query = query.order_by(TrainingMetric.step).limit(limit)

        result = await db.execute(query)
        return list(result.scalars().all())
