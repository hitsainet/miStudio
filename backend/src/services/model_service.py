"""
Model service layer.

This module contains business logic for model management operations.
"""

from typing import Optional, Tuple, List
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.model import Model, ModelStatus
from ..schemas.model import ModelCreate, ModelUpdate


class ModelService:
    """Service class for model operations."""

    @staticmethod
    async def create_model(db: AsyncSession, model_data: ModelCreate) -> Model:
        """
        Create a new model.

        Args:
            db: Database session
            model_data: Model creation data

        Returns:
            Created model
        """
        db_model = Model(
            name=model_data.name,
            architecture=model_data.architecture,
            repo_id=model_data.repo_id,
            quantization=model_data.quantization,
            status=ModelStatus.DOWNLOADING,
            extra_metadata=model_data.metadata or {}
        )

        db.add(db_model)
        await db.commit()
        await db.refresh(db_model)

        return db_model

    @staticmethod
    async def get_model(db: AsyncSession, model_id: UUID) -> Optional[Model]:
        """
        Get a model by ID.

        Args:
            db: Database session
            model_id: Model UUID

        Returns:
            Model if found, None otherwise
        """
        result = await db.execute(
            select(Model).where(Model.id == model_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_model_by_repo_id(db: AsyncSession, repo_id: str) -> Optional[Model]:
        """
        Get a model by HuggingFace repository ID.

        Args:
            db: Database session
            repo_id: HuggingFace repository ID

        Returns:
            Model if found, None otherwise
        """
        result = await db.execute(
            select(Model).where(Model.repo_id == repo_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_models(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        architecture: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> Tuple[List[Model], int]:
        """
        List models with filtering, pagination, and sorting.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            search: Search query for name or repo_id
            architecture: Filter by architecture type
            status: Filter by status
            sort_by: Column to sort by
            order: Sort order (asc or desc)

        Returns:
            Tuple of (list of models, total count)
        """
        # Build base query
        query = select(Model)

        # Apply filters
        if search:
            search_filter = or_(
                Model.name.ilike(f"%{search}%"),
                Model.repo_id.ilike(f"%{search}%")
            )
            query = query.where(search_filter)

        if architecture:
            query = query.where(Model.architecture == architecture)

        if status:
            query = query.where(Model.status == status)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply sorting
        sort_column = getattr(Model, sort_by, Model.created_at)
        if order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        models = result.scalars().all()

        return list(models), total

    @staticmethod
    async def update_model(
        db: AsyncSession,
        model_id: UUID,
        updates: ModelUpdate
    ) -> Optional[Model]:
        """
        Update a model.

        Args:
            db: Database session
            model_id: Model UUID
            updates: Update data

        Returns:
            Updated model if found, None otherwise
        """
        db_model = await ModelService.get_model(db, model_id)
        if not db_model:
            return None

        # Update fields
        update_data = updates.model_dump(exclude_unset=True)

        # Handle status conversion
        if "status" in update_data:
            status_value = update_data["status"]
            if isinstance(status_value, str):
                update_data["status"] = ModelStatus(status_value.lower())

        for field, value in update_data.items():
            if field == "metadata":
                setattr(db_model, "extra_metadata", value)
            else:
                setattr(db_model, field, value)

        db_model.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(db_model)

        return db_model

    @staticmethod
    async def delete_model(db: AsyncSession, model_id: UUID) -> bool:
        """
        Delete a model.

        Args:
            db: Database session
            model_id: Model UUID

        Returns:
            True if deleted, False if not found
        """
        db_model = await ModelService.get_model(db, model_id)
        if not db_model:
            return False

        await db.delete(db_model)
        await db.commit()

        return True
