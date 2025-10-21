"""
TrainingTemplate service layer for business logic.

This module contains the TrainingTemplateService class which handles all
training template-related business logic and database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, UTC
import json

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.training_template import TrainingTemplate
from ..schemas.training_template import (
    TrainingTemplateCreate,
    TrainingTemplateUpdate,
)


class TrainingTemplateService:
    """Service class for training template operations."""

    @staticmethod
    async def create_template(
        db: AsyncSession,
        template: TrainingTemplateCreate
    ) -> TrainingTemplate:
        """
        Create a new training template.

        Args:
            db: Database session
            template: Template creation data

        Returns:
            Created training template object
        """
        # Convert hyperparameters to dict for JSONB storage
        hyperparameters_dict = template.hyperparameters.model_dump()

        db_template = TrainingTemplate(
            name=template.name,
            description=template.description,
            model_id=template.model_id,
            dataset_id=template.dataset_id,
            encoder_type=template.encoder_type.value,
            hyperparameters=hyperparameters_dict,
            is_favorite=template.is_favorite,
            extra_metadata=template.extra_metadata or {},
        )

        db.add(db_template)
        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def get_template(
        db: AsyncSession,
        template_id: UUID
    ) -> Optional[TrainingTemplate]:
        """
        Get a training template by ID.

        Args:
            db: Database session
            template_id: Template UUID

        Returns:
            TrainingTemplate object if found, None otherwise
        """
        result = await db.execute(
            select(TrainingTemplate).where(TrainingTemplate.id == template_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_templates(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        is_favorite: Optional[bool] = None,
        encoder_type: Optional[str] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> tuple[List[TrainingTemplate], int]:
        """
        List training templates with filtering, pagination, and sorting.

        Args:
            db: Database session
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            search: Search query for name or description
            is_favorite: Filter by favorite status
            encoder_type: Filter by encoder architecture type
            sort_by: Column to sort by
            order: Sort order ('asc' or 'desc')

        Returns:
            Tuple of (list of templates, total count)
        """
        # Build base query
        query = select(TrainingTemplate)
        count_query = select(func.count()).select_from(TrainingTemplate)

        # Apply filters
        filters = []

        if search:
            search_filter = or_(
                TrainingTemplate.name.ilike(f"%{search}%"),
                TrainingTemplate.description.ilike(f"%{search}%")
            )
            filters.append(search_filter)

        if is_favorite is not None:
            filters.append(TrainingTemplate.is_favorite == is_favorite)

        if encoder_type is not None:
            filters.append(TrainingTemplate.encoder_type == encoder_type)

        if filters:
            query = query.where(*filters)
            count_query = count_query.where(*filters)

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply sorting
        sort_column = getattr(TrainingTemplate, sort_by, TrainingTemplate.created_at)
        if order.lower() == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        templates = result.scalars().all()

        return list(templates), total

    @staticmethod
    async def update_template(
        db: AsyncSession,
        template_id: UUID,
        updates: TrainingTemplateUpdate
    ) -> Optional[TrainingTemplate]:
        """
        Update a training template.

        Args:
            db: Database session
            template_id: Template UUID
            updates: Update data

        Returns:
            Updated TrainingTemplate object if found, None otherwise
        """
        # Get existing template
        result = await db.execute(
            select(TrainingTemplate).where(TrainingTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return None

        # Apply updates
        update_data = updates.model_dump(exclude_unset=True)

        # Convert hyperparameters if present
        if "hyperparameters" in update_data and update_data["hyperparameters"] is not None:
            update_data["hyperparameters"] = update_data["hyperparameters"].model_dump()

        # Convert encoder_type enum to string
        if "encoder_type" in update_data and update_data["encoder_type"] is not None:
            update_data["encoder_type"] = update_data["encoder_type"].value

        for field, value in update_data.items():
            setattr(db_template, field, value)

        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def delete_template(
        db: AsyncSession,
        template_id: UUID
    ) -> bool:
        """
        Delete a training template.

        Args:
            db: Database session
            template_id: Template UUID

        Returns:
            True if template was deleted, False if not found
        """
        result = await db.execute(
            select(TrainingTemplate).where(TrainingTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return False

        await db.delete(db_template)
        await db.commit()

        return True

    @staticmethod
    async def toggle_favorite(
        db: AsyncSession,
        template_id: UUID
    ) -> Optional[TrainingTemplate]:
        """
        Toggle the favorite status of a training template.

        Args:
            db: Database session
            template_id: Template UUID

        Returns:
            Updated TrainingTemplate object if found, None otherwise
        """
        result = await db.execute(
            select(TrainingTemplate).where(TrainingTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return None

        db_template.is_favorite = not db_template.is_favorite

        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def get_favorites(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[TrainingTemplate], int]:
        """
        Get all favorite training templates.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Tuple of (list of favorite templates, total count)
        """
        return await TrainingTemplateService.list_templates(
            db=db,
            skip=skip,
            limit=limit,
            is_favorite=True
        )

    @staticmethod
    async def export_templates(
        db: AsyncSession,
        template_ids: Optional[List[UUID]] = None
    ) -> Dict[str, Any]:
        """
        Export training templates to JSON format.

        Args:
            db: Database session
            template_ids: Optional list of template IDs to export. If None, exports all.

        Returns:
            Dictionary containing export data with version and templates
        """
        # Build query
        query = select(TrainingTemplate)

        if template_ids:
            query = query.where(TrainingTemplate.id.in_(template_ids))

        # Execute query
        result = await db.execute(query)
        templates = result.scalars().all()

        # Convert to response format
        templates_data = [
            {
                "id": str(template.id),
                "name": template.name,
                "description": template.description,
                "model_id": template.model_id,
                "dataset_id": template.dataset_id,
                "encoder_type": template.encoder_type,
                "hyperparameters": template.hyperparameters,
                "is_favorite": template.is_favorite,
                "extra_metadata": template.extra_metadata,
                "created_at": template.created_at.isoformat(),
                "updated_at": template.updated_at.isoformat(),
            }
            for template in templates
        ]

        return {
            "version": "1.0",
            "templates": templates_data,
            "exported_at": datetime.now(UTC).isoformat()
        }

    @staticmethod
    async def import_templates(
        db: AsyncSession,
        import_data: Dict[str, Any],
        overwrite_duplicates: bool = False
    ) -> Dict[str, Any]:
        """
        Import training templates from JSON format.

        Args:
            db: Database session
            import_data: Import data containing templates
            overwrite_duplicates: Whether to overwrite templates with the same name

        Returns:
            Dictionary containing import results (created, updated, skipped counts)

        Raises:
            ValueError: If import data is invalid
        """
        # Validate version
        version = import_data.get("version")
        if version not in ["1.0"]:
            raise ValueError(f"Unsupported import version: {version}")

        templates_data = import_data.get("templates", [])
        if not templates_data:
            raise ValueError("No templates found in import data")

        created_count = 0
        updated_count = 0
        skipped_count = 0

        for template_data in templates_data:
            # Check if template with same name exists
            name = template_data.get("name")
            result = await db.execute(
                select(TrainingTemplate).where(TrainingTemplate.name == name)
            )
            existing = result.scalar_one_or_none()

            if existing:
                if overwrite_duplicates:
                    # Update existing template
                    for field in ["description", "model_id", "dataset_id", "encoder_type", "hyperparameters", "is_favorite", "extra_metadata"]:
                        if field in template_data:
                            setattr(existing, field, template_data[field])
                    updated_count += 1
                else:
                    skipped_count += 1
                    continue
            else:
                # Create new template
                db_template = TrainingTemplate(
                    name=template_data["name"],
                    description=template_data.get("description"),
                    model_id=template_data.get("model_id"),
                    dataset_id=template_data.get("dataset_id"),
                    encoder_type=template_data["encoder_type"],
                    hyperparameters=template_data["hyperparameters"],
                    is_favorite=template_data.get("is_favorite", False),
                    extra_metadata=template_data.get("extra_metadata", {}),
                )
                db.add(db_template)
                created_count += 1

        await db.commit()

        return {
            "created": created_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "total": len(templates_data)
        }
