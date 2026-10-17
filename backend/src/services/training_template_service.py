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
from ..schemas.training import TrainingHyperparameters
from ..schemas.training_template import (
    TrainingTemplateCreate,
    TrainingTemplateUpdate,
)
from .template_updates import reject_null_updates


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

        # dataset_id kept for backward compatibility (first dataset in list)
        primary_dataset_id = template.dataset_ids[0] if template.dataset_ids else None
        db_template = TrainingTemplate(
            name=template.name,
            description=template.description,
            model_id=template.model_id,
            dataset_id=primary_dataset_id,  # Backward compat
            dataset_ids=template.dataset_ids,  # New multi-dataset field
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

        # Top level: a PATCH. Only the fields the request names are written.
        update_data = updates.model_dump(exclude_unset=True)

        # A null for a NOT NULL column is a bad request, not a 500 (review R2F-10).
        update_data = reject_null_updates(update_data, TrainingTemplate)

        # Hyperparameters: a REPLACE, stored as the complete validated dump, the same
        # as create_template stores (review round 2, R2D-4 / R2-F).
        # `exclude_unset` recursed into the nested model, so an update stored only
        # the keys it was sent. A key the form cleared was then missing, not null.
        # Loaded into the training panel, a missing key keeps the framework default
        # that loading applies first. For JumpReLU, `target_l0` null became 0.05.
        # The client sends the template's own keys with its edits applied (the
        # Templates form overlays them), so filling defaults only touches keys the
        # template never had. It never merges with the stored dict: that would bring
        # back a value the user cleared.
        if updates.hyperparameters is not None:
            update_data["hyperparameters"] = updates.hyperparameters.model_dump()

        # `dataset_id` is the first of `dataset_ids`, as on create. The training
        # panel's template selector matches on it, so an update that changed the
        # datasets left the template offered against the old dataset.
        if "dataset_ids" in update_data:
            dataset_ids = update_data["dataset_ids"] or []
            update_data["dataset_id"] = dataset_ids[0] if dataset_ids else None

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
                "dataset_ids": template.dataset_ids,  # Multi-dataset support
                "dataset_id": template.dataset_id,  # Backward compat
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
    def _validated_hyperparameters(template_data: Dict[str, Any]) -> Dict[str, Any]:
        """One imported template's hyperparameters, through the schema `create` uses.

        Raises ValueError naming the template, so a bad file is refused at the import
        endpoint (400) rather than becoming a row that breaks a training run later.
        """
        from pydantic import ValidationError

        name = template_data.get("name")
        raw = template_data.get("hyperparameters")
        if not isinstance(raw, dict):
            raise ValueError(
                f"Template {name!r}: 'hyperparameters' must be an object, got "
                f"{type(raw).__name__}"
            )
        try:
            return TrainingHyperparameters(**raw).model_dump()
        except ValidationError as exc:
            raise ValueError(
                f"Template {name!r} has invalid hyperparameters: {exc}"
            ) from exc

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
            # VALIDATED ON THE WAY IN (review R2F-7). An import used to store
            # `template_data["hyperparameters"]` exactly as the file gave it: any JSON
            # object at all, including one with no `hidden_dim`, a negative
            # `total_steps`, or a string where a number belongs. Nothing checked it,
            # so the file's contents became a template that looks like every other
            # template and fails only when a training built from it runs. A malformed
            # template is now refused here, naming itself.
            hyperparameters = TrainingTemplateService._validated_hyperparameters(template_data)

            # Check if template with same name exists
            name = template_data.get("name")
            result = await db.execute(
                select(TrainingTemplate).where(TrainingTemplate.name == name)
            )
            existing = result.scalar_one_or_none()

            if existing:
                if overwrite_duplicates:
                    # Update existing template
                    for field in ["description", "model_id", "dataset_ids", "dataset_id", "encoder_type", "hyperparameters", "is_favorite", "extra_metadata"]:
                        if field in template_data:
                            value = (
                                hyperparameters if field == "hyperparameters"
                                else template_data[field]
                            )
                            setattr(existing, field, value)
                    updated_count += 1
                else:
                    skipped_count += 1
                    continue
            else:
                # Handle backward compat: if dataset_ids not present, create from dataset_id
                dataset_ids = template_data.get("dataset_ids")
                if not dataset_ids and template_data.get("dataset_id"):
                    dataset_ids = [template_data["dataset_id"]]

                # Create new template
                db_template = TrainingTemplate(
                    name=template_data["name"],
                    description=template_data.get("description"),
                    model_id=template_data.get("model_id"),
                    dataset_ids=dataset_ids or [],
                    dataset_id=template_data.get("dataset_id"),  # Backward compat
                    encoder_type=template_data["encoder_type"],
                    hyperparameters=hyperparameters,
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
