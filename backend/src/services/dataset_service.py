"""
Dataset service layer for business logic.

This module contains the DatasetService class which handles all
dataset-related business logic and database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, UTC

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.dataset import Dataset, DatasetStatus
from ..schemas.dataset import DatasetCreate, DatasetUpdate


class DatasetService:
    """Service class for dataset operations."""

    @staticmethod
    async def create_dataset(
        db: AsyncSession,
        dataset: DatasetCreate
    ) -> Dataset:
        """
        Create a new dataset.

        Args:
            db: Database session
            dataset: Dataset creation data

        Returns:
            Created dataset object
        """
        db_dataset = Dataset(
            name=dataset.name,
            source=dataset.source,
            hf_repo_id=dataset.hf_repo_id,
            raw_path=dataset.raw_path,
            status=DatasetStatus.DOWNLOADING if dataset.hf_repo_id else DatasetStatus.PROCESSING,
            metadata=dataset.metadata or {},
        )

        db.add(db_dataset)
        await db.commit()
        await db.refresh(db_dataset)

        return db_dataset

    @staticmethod
    async def get_dataset(
        db: AsyncSession,
        dataset_id: UUID
    ) -> Optional[Dataset]:
        """
        Get a dataset by ID.

        Args:
            db: Database session
            dataset_id: Dataset UUID

        Returns:
            Dataset object if found, None otherwise
        """
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_datasets(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        source: Optional[str] = None,
        status: Optional[DatasetStatus] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> tuple[List[Dataset], int]:
        """
        List datasets with filtering, pagination, and sorting.

        Args:
            db: Database session
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            search: Search query for name or hf_repo_id
            source: Filter by source type
            status: Filter by status
            sort_by: Column to sort by
            order: Sort order ('asc' or 'desc')

        Returns:
            Tuple of (list of datasets, total count)
        """
        # Build base query
        query = select(Dataset)
        count_query = select(func.count()).select_from(Dataset)

        # Apply filters
        filters = []

        if search:
            search_filter = or_(
                Dataset.name.ilike(f"%{search}%"),
                Dataset.hf_repo_id.ilike(f"%{search}%")
            )
            filters.append(search_filter)

        if source:
            filters.append(Dataset.source == source)

        if status:
            filters.append(Dataset.status == status)

        if filters:
            query = query.where(*filters)
            count_query = count_query.where(*filters)

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply sorting
        sort_column = getattr(Dataset, sort_by, Dataset.created_at)
        if order.lower() == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        datasets = result.scalars().all()

        return list(datasets), total

    @staticmethod
    async def update_dataset(
        db: AsyncSession,
        dataset_id: UUID,
        updates: DatasetUpdate
    ) -> Optional[Dataset]:
        """
        Update a dataset.

        Args:
            db: Database session
            dataset_id: Dataset UUID
            updates: Update data

        Returns:
            Updated dataset object if found, None otherwise
        """
        # Get existing dataset
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        db_dataset = result.scalar_one_or_none()

        if not db_dataset:
            return None

        # Apply updates
        update_data = updates.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            if field == "status" and isinstance(value, str):
                # Convert string to enum
                value = DatasetStatus[value.upper()]
            setattr(db_dataset, field, value)

        db_dataset.updated_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_dataset)

        return db_dataset

    @staticmethod
    async def delete_dataset(
        db: AsyncSession,
        dataset_id: UUID
    ) -> bool:
        """
        Delete a dataset.

        Args:
            db: Database session
            dataset_id: Dataset UUID

        Returns:
            True if deleted, False if not found
        """
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        db_dataset = result.scalar_one_or_none()

        if not db_dataset:
            return False

        await db.delete(db_dataset)
        await db.commit()

        return True

    @staticmethod
    async def get_dataset_by_repo_id(
        db: AsyncSession,
        repo_id: str
    ) -> Optional[Dataset]:
        """
        Get a dataset by HuggingFace repository ID.

        Args:
            db: Database session
            repo_id: HuggingFace repository ID

        Returns:
            Dataset object if found, None otherwise
        """
        result = await db.execute(
            select(Dataset).where(Dataset.hf_repo_id == repo_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update_dataset_progress(
        db: AsyncSession,
        dataset_id: UUID,
        progress: float,
        status: Optional[DatasetStatus] = None
    ) -> Optional[Dataset]:
        """
        Update dataset progress and optionally status.

        Args:
            db: Database session
            dataset_id: Dataset UUID
            progress: Progress value (0-100)
            status: Optional new status

        Returns:
            Updated dataset object if found, None otherwise
        """
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        db_dataset = result.scalar_one_or_none()

        if not db_dataset:
            return None

        db_dataset.progress = progress
        if status:
            db_dataset.status = status
        db_dataset.updated_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_dataset)

        return db_dataset

    @staticmethod
    async def mark_dataset_error(
        db: AsyncSession,
        dataset_id: UUID,
        error_message: str
    ) -> Optional[Dataset]:
        """
        Mark a dataset as errored with error message.

        Args:
            db: Database session
            dataset_id: Dataset UUID
            error_message: Error message

        Returns:
            Updated dataset object if found, None otherwise
        """
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        db_dataset = result.scalar_one_or_none()

        if not db_dataset:
            return None

        db_dataset.status = DatasetStatus.ERROR
        db_dataset.error_message = error_message
        db_dataset.updated_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_dataset)

        return db_dataset
