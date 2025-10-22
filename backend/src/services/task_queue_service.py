"""
TaskQueue service for managing background task operations and retries.

This service provides visibility into all background tasks and enables
user-controlled retry of failed operations.
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from ..models.task_queue import TaskQueue
from ..models.model import Model
from ..models.dataset import Dataset
from ..models.training import Training

logger = logging.getLogger(__name__)


class TaskQueueService:
    """Service for managing task queue operations."""

    @staticmethod
    async def create_task_entry(
        db: AsyncSession,
        task_type: str,
        entity_id: str,
        entity_type: str,
        task_id: Optional[str] = None,
        retry_params: Optional[Dict[str, Any]] = None,
    ) -> TaskQueue:
        """
        Create a new task queue entry.

        Args:
            db: Database session
            task_type: Type of task (download, training, extraction, tokenization)
            entity_id: ID of the entity being processed
            entity_type: Type of entity (model, dataset, training, extraction)
            task_id: Optional Celery task ID
            retry_params: Optional parameters for retry

        Returns:
            Created TaskQueue entry
        """
        entry = TaskQueue(
            id=f"tq_{uuid.uuid4().hex[:12]}",
            task_id=task_id,
            task_type=task_type,
            entity_id=entity_id,
            entity_type=entity_type,
            status="queued",
            retry_params=retry_params or {},
            retry_count=0,
        )
        db.add(entry)
        await db.commit()
        await db.refresh(entry)

        logger.info(f"Created task queue entry {entry.id} for {entity_type} {entity_id}")
        return entry

    @staticmethod
    async def update_task_status(
        db: AsyncSession,
        task_queue_id: str,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> Optional[TaskQueue]:
        """
        Update task queue entry status.

        Args:
            db: Database session
            task_queue_id: Task queue entry ID
            status: New status (queued, running, failed, completed, cancelled)
            progress: Optional progress percentage (0-100)
            error_message: Optional error message

        Returns:
            Updated TaskQueue entry or None if not found
        """
        result = await db.execute(
            select(TaskQueue).where(TaskQueue.id == task_queue_id)
        )
        entry = result.scalar_one_or_none()

        if not entry:
            logger.warning(f"Task queue entry {task_queue_id} not found")
            return None

        entry.status = status
        if progress is not None:
            entry.progress = progress
        if error_message:
            entry.error_message = error_message

        # Update timestamps
        if status == "running" and not entry.started_at:
            entry.started_at = datetime.utcnow()
        elif status in ("completed", "failed", "cancelled"):
            entry.completed_at = datetime.utcnow()

        await db.commit()
        await db.refresh(entry)

        logger.info(f"Updated task queue entry {task_queue_id} to status {status}")
        return entry

    @staticmethod
    async def get_all_tasks(
        db: AsyncSession,
        status: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> List[TaskQueue]:
        """
        Get all task queue entries with optional filtering.

        Args:
            db: Database session
            status: Optional status filter
            entity_type: Optional entity type filter

        Returns:
            List of TaskQueue entries
        """
        query = select(TaskQueue)

        filters = []
        if status:
            filters.append(TaskQueue.status == status)
        if entity_type:
            filters.append(TaskQueue.entity_type == entity_type)

        if filters:
            query = query.where(and_(*filters))

        query = query.order_by(TaskQueue.created_at.desc())

        result = await db.execute(query)
        return result.scalars().all()

    @staticmethod
    async def get_failed_tasks(db: AsyncSession) -> List[TaskQueue]:
        """
        Get all failed task queue entries.

        Args:
            db: Database session

        Returns:
            List of failed TaskQueue entries
        """
        return await TaskQueueService.get_all_tasks(db, status="failed")

    @staticmethod
    async def get_active_tasks(db: AsyncSession) -> List[TaskQueue]:
        """
        Get all active (queued or running) task queue entries.

        Args:
            db: Database session

        Returns:
            List of active TaskQueue entries
        """
        query = select(TaskQueue).where(
            or_(
                TaskQueue.status == "queued",
                TaskQueue.status == "running"
            )
        ).order_by(TaskQueue.created_at.desc())

        result = await db.execute(query)
        return result.scalars().all()

    @staticmethod
    async def get_task_by_id(
        db: AsyncSession,
        task_queue_id: str
    ) -> Optional[TaskQueue]:
        """
        Get a specific task queue entry by ID.

        Args:
            db: Database session
            task_queue_id: Task queue entry ID

        Returns:
            TaskQueue entry or None if not found
        """
        result = await db.execute(
            select(TaskQueue).where(TaskQueue.id == task_queue_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_task_by_entity(
        db: AsyncSession,
        entity_id: str,
        entity_type: str
    ) -> List[TaskQueue]:
        """
        Get all task queue entries for a specific entity.

        Args:
            db: Database session
            entity_id: Entity ID
            entity_type: Entity type

        Returns:
            List of TaskQueue entries for the entity
        """
        query = select(TaskQueue).where(
            and_(
                TaskQueue.entity_id == entity_id,
                TaskQueue.entity_type == entity_type
            )
        ).order_by(TaskQueue.created_at.desc())

        result = await db.execute(query)
        return result.scalars().all()

    @staticmethod
    async def delete_task(
        db: AsyncSession,
        task_queue_id: str
    ) -> bool:
        """
        Delete a task queue entry.

        Args:
            db: Database session
            task_queue_id: Task queue entry ID

        Returns:
            True if deleted, False if not found
        """
        result = await db.execute(
            select(TaskQueue).where(TaskQueue.id == task_queue_id)
        )
        entry = result.scalar_one_or_none()

        if not entry:
            logger.warning(f"Task queue entry {task_queue_id} not found")
            return False

        await db.delete(entry)
        await db.commit()

        logger.info(f"Deleted task queue entry {task_queue_id}")
        return True

    @staticmethod
    async def increment_retry_count(
        db: AsyncSession,
        task_queue_id: str
    ) -> Optional[TaskQueue]:
        """
        Increment the retry count for a task queue entry.

        Args:
            db: Database session
            task_queue_id: Task queue entry ID

        Returns:
            Updated TaskQueue entry or None if not found
        """
        result = await db.execute(
            select(TaskQueue).where(TaskQueue.id == task_queue_id)
        )
        entry = result.scalar_one_or_none()

        if not entry:
            logger.warning(f"Task queue entry {task_queue_id} not found")
            return None

        entry.retry_count += 1
        entry.status = "queued"
        entry.error_message = None
        entry.completed_at = None

        await db.commit()
        await db.refresh(entry)

        logger.info(f"Incremented retry count for task {task_queue_id} to {entry.retry_count}")
        return entry

    @staticmethod
    async def get_entity_info(
        db: AsyncSession,
        entity_id: str,
        entity_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about the entity associated with a task.

        Args:
            db: Database session
            entity_id: Entity ID
            entity_type: Entity type (model, dataset, training)

        Returns:
            Dictionary with entity information or None if not found
        """
        if entity_type == "model":
            result = await db.execute(
                select(Model).where(Model.id == entity_id)
            )
            entity = result.scalar_one_or_none()
            if entity:
                return {
                    "id": entity.id,
                    "name": entity.name,
                    "repo_id": entity.repo_id,
                    "status": entity.status,
                    "type": "model"
                }
        elif entity_type == "dataset":
            result = await db.execute(
                select(Dataset).where(Dataset.id == entity_id)
            )
            entity = result.scalar_one_or_none()
            if entity:
                return {
                    "id": entity.id,
                    "name": entity.name,
                    "hf_repo_id": entity.hf_repo_id,
                    "status": entity.status,
                    "type": "dataset"
                }
        elif entity_type == "training":
            result = await db.execute(
                select(Training).where(Training.id == entity_id)
            )
            entity = result.scalar_one_or_none()
            if entity:
                return {
                    "id": entity.id,
                    "name": entity.name,
                    "status": entity.status,
                    "type": "training"
                }

        return None

    @staticmethod
    async def cleanup_completed_tasks(
        db: AsyncSession,
        days_old: int = 7
    ) -> int:
        """
        Clean up old completed tasks from the queue.

        Args:
            db: Database session
            days_old: Delete tasks completed more than this many days ago

        Returns:
            Number of tasks deleted
        """
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        result = await db.execute(
            select(TaskQueue).where(
                and_(
                    TaskQueue.status == "completed",
                    TaskQueue.completed_at < cutoff_date
                )
            )
        )
        tasks_to_delete = result.scalars().all()

        count = len(tasks_to_delete)
        for task in tasks_to_delete:
            await db.delete(task)

        await db.commit()

        logger.info(f"Cleaned up {count} completed tasks older than {days_old} days")
        return count
