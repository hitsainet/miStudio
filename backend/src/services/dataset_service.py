"""
Dataset service layer for business logic.

This module contains the DatasetService class which handles all
dataset-related business logic and database operations.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, UTC
import copy

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.dataset import Dataset, DatasetStatus
from ..schemas.dataset import DatasetCreate, DatasetUpdate

logger = logging.getLogger(__name__)


def deep_merge_metadata(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge metadata dictionaries, preserving existing values.

    Rules:
    1. New non-None values overwrite existing values
    2. None values in new dict do NOT overwrite existing values
    3. Nested dicts are recursively merged
    4. Lists and other types are replaced (not merged)

    Args:
        existing: Existing metadata dictionary
        new: New metadata to merge in

    Returns:
        Merged metadata dictionary
    """
    if not existing:
        # Filter out None values from new metadata
        return {k: v for k, v in new.items() if v is not None}

    if not new:
        return copy.deepcopy(existing)

    # Start with a deep copy of existing to avoid mutations
    result = copy.deepcopy(existing)

    for key, new_value in new.items():
        # Skip None values - they don't overwrite existing data
        if new_value is None:
            continue

        # If key doesn't exist in result, just add it
        if key not in result:
            result[key] = new_value
            continue

        existing_value = result[key]

        # If both are dicts, recursively merge
        if isinstance(existing_value, dict) and isinstance(new_value, dict):
            result[key] = deep_merge_metadata(existing_value, new_value)
        else:
            # For non-dict types (lists, primitives), replace with new value
            result[key] = new_value

    return result


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
            extra_metadata=dataset.metadata or {},
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
        Get a dataset by ID with tokenizations eagerly loaded.

        Args:
            db: Database session
            dataset_id: Dataset UUID

        Returns:
            Dataset object with tokenizations relationship loaded if found, None otherwise
        """
        result = await db.execute(
            select(Dataset)
            .where(Dataset.id == dataset_id)
            .options(selectinload(Dataset.tokenizations))
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
        # Build base query with eager loading of tokenizations
        query = select(Dataset).options(selectinload(Dataset.tokenizations))
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
            elif field == "metadata":
                # Map 'metadata' to 'extra_metadata' (SQLAlchemy attribute name)
                # Deep merge metadata to preserve existing fields and skip None values
                existing_metadata = db_dataset.extra_metadata or {}
                if isinstance(value, dict):
                    # Use deep merge that skips None values and recursively merges nested dicts
                    merged_metadata = deep_merge_metadata(existing_metadata, value)
                    setattr(db_dataset, "extra_metadata", merged_metadata)
                    continue
                elif value is None:
                    # None means "no update to metadata" - skip it
                    continue

            # Map 'metadata' field to 'extra_metadata' attribute if not already handled
            attr_name = "extra_metadata" if field == "metadata" else field
            setattr(db_dataset, attr_name, value)

        db_dataset.updated_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_dataset)

        return db_dataset

    @staticmethod
    async def delete_dataset(
        db: AsyncSession,
        dataset_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """
        Delete a dataset from database and return file paths for background cleanup.

        This method:
        1. Deletes the database record
        2. Returns file paths for the caller to queue background deletion

        Args:
            db: Database session
            dataset_id: Dataset UUID

        Returns:
            Dict with deletion status and file paths, or None if not found
            Format: {
                "deleted": True,
                "dataset_id": str,
                "raw_path": Optional[str],
                "tokenized_path": Optional[str]
            }
        """
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        db_dataset = result.scalar_one_or_none()

        if not db_dataset:
            return None

        # Capture file paths before deletion
        raw_path = db_dataset.raw_path
        # Get tokenized paths from tokenizations relationship (if any)
        tokenized_paths = []
        if db_dataset.tokenizations:
            tokenized_paths = [t.tokenized_path for t in db_dataset.tokenizations if t.tokenized_path]

        # Delete database record (cascades to tokenizations)
        await db.delete(db_dataset)
        await db.commit()

        return {
            "deleted": True,
            "dataset_id": str(dataset_id),
            "raw_path": raw_path,
            "tokenized_paths": tokenized_paths,  # Return list instead of single path
        }

    @staticmethod
    async def clear_tokenization(
        db: AsyncSession,
        dataset_id: UUID
    ) -> Optional[Dataset]:
        """
        Clear tokenization data from a dataset while keeping raw data intact.

        This method:
        1. Deletes tokenized files from disk (if they exist)
        2. Clears tokenization-related database fields
        3. Removes tokenization metadata
        4. Resets status to READY

        Args:
            db: Database session
            dataset_id: Dataset UUID

        Returns:
            Updated dataset object, or None if not found
        """
        from pathlib import Path
        import shutil

        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        db_dataset = result.scalar_one_or_none()

        if not db_dataset:
            return None

        # Delete tokenized files and remove tokenization records from database
        # Use eager loading to get tokenizations
        result = await db.execute(
            select(Dataset)
            .where(Dataset.id == dataset_id)
            .options(selectinload(Dataset.tokenizations))
        )
        db_dataset = result.scalar_one_or_none()

        if not db_dataset:
            return None

        # Delete tokenized files from disk and remove tokenization records
        if db_dataset.tokenizations:
            from ..models.dataset_tokenization import DatasetTokenization

            for tokenization in db_dataset.tokenizations:
                # Delete tokenized files from disk if they exist
                if tokenization.tokenized_path:
                    tokenized_path = Path(tokenization.tokenized_path)
                    if tokenized_path.exists():
                        try:
                            if tokenized_path.is_dir():
                                shutil.rmtree(tokenized_path)
                            else:
                                tokenized_path.unlink()
                            logger.info(f"Deleted tokenized files at {tokenized_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete tokenized files at {tokenized_path}: {e}")

                # Delete the tokenization record from database
                await db.delete(tokenization)

            logger.info(f"Removed {len(db_dataset.tokenizations)} tokenization records from database")

        # Remove tokenization metadata
        if db_dataset.extra_metadata and 'tokenization' in db_dataset.extra_metadata:
            metadata = dict(db_dataset.extra_metadata)  # Create mutable copy
            del metadata['tokenization']
            db_dataset.extra_metadata = metadata

        # Clear task_id from metadata if it exists
        if db_dataset.extra_metadata and 'task_id' in db_dataset.extra_metadata:
            metadata = dict(db_dataset.extra_metadata)  # Create mutable copy
            del metadata['task_id']
            if 'task_type' in metadata:
                del metadata['task_type']
            db_dataset.extra_metadata = metadata

        # Reset status and progress
        db_dataset.status = DatasetStatus.READY
        db_dataset.progress = 0.0
        db_dataset.error_message = None

        # Commit changes
        await db.commit()
        await db.refresh(db_dataset)

        return db_dataset

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
