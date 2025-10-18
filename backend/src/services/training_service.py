"""
Training service layer for business logic.

This module contains the TrainingService class which handles all
SAE training-related business logic and database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, UTC

from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.training import Training, TrainingStatus
from ..models.training_metric import TrainingMetric
from ..schemas.training import TrainingCreate, TrainingUpdate


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
    ) -> bool:
        """
        Delete a training job.

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            True if deleted, False if not found
        """
        db_training = await TrainingService.get_training(db, training_id)
        if not db_training:
            return False

        await db.delete(db_training)
        await db.commit()

        return True

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
