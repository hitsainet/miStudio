"""
ExtractionTemplate service layer for business logic.

This module contains the ExtractionTemplateService class which handles all
extraction template-related business logic and database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, UTC
import json

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.extraction_template import ExtractionTemplate
from ..schemas.extraction_template import (
    ExtractionTemplateCreate,
    ExtractionTemplateUpdate,
)


class ExtractionTemplateService:
    """Service class for extraction template operations."""

    @staticmethod
    async def create_template(
        db: AsyncSession,
        template: ExtractionTemplateCreate
    ) -> ExtractionTemplate:
        """
        Create a new extraction template.

        Args:
            db: Database session
            template: Template creation data

        Returns:
            Created extraction template object
        """
        db_template = ExtractionTemplate(
            name=template.name,
            description=template.description,
            layer_indices=template.layer_indices,
            hook_types=template.hook_types,
            max_samples=template.max_samples,
            batch_size=template.batch_size,
            top_k_examples=template.top_k_examples,
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
    ) -> Optional[ExtractionTemplate]:
        """
        Get an extraction template by ID.

        Args:
            db: Database session
            template_id: Template UUID

        Returns:
            ExtractionTemplate object if found, None otherwise
        """
        result = await db.execute(
            select(ExtractionTemplate).where(ExtractionTemplate.id == template_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_templates(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        is_favorite: Optional[bool] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> tuple[List[ExtractionTemplate], int]:
        """
        List extraction templates with filtering, pagination, and sorting.

        Args:
            db: Database session
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            search: Search query for name or description
            is_favorite: Filter by favorite status
            sort_by: Column to sort by
            order: Sort order ('asc' or 'desc')

        Returns:
            Tuple of (list of templates, total count)
        """
        # Build base query
        query = select(ExtractionTemplate)
        count_query = select(func.count()).select_from(ExtractionTemplate)

        # Apply filters
        filters = []

        if search:
            search_filter = or_(
                ExtractionTemplate.name.ilike(f"%{search}%"),
                ExtractionTemplate.description.ilike(f"%{search}%")
            )
            filters.append(search_filter)

        if is_favorite is not None:
            filters.append(ExtractionTemplate.is_favorite == is_favorite)

        if filters:
            query = query.where(*filters)
            count_query = count_query.where(*filters)

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply sorting
        sort_column = getattr(ExtractionTemplate, sort_by, ExtractionTemplate.created_at)
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
        updates: ExtractionTemplateUpdate
    ) -> Optional[ExtractionTemplate]:
        """
        Update an extraction template.

        Args:
            db: Database session
            template_id: Template UUID
            updates: Update data

        Returns:
            Updated extraction template object if found, None otherwise
        """
        # Get existing template
        result = await db.execute(
            select(ExtractionTemplate).where(ExtractionTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return None

        # Apply updates
        update_data = updates.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            setattr(db_template, field, value)

        db_template.updated_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def delete_template(
        db: AsyncSession,
        template_id: UUID
    ) -> bool:
        """
        Delete an extraction template.

        Args:
            db: Database session
            template_id: Template UUID

        Returns:
            True if deleted, False if not found
        """
        result = await db.execute(
            select(ExtractionTemplate).where(ExtractionTemplate.id == template_id)
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
    ) -> Optional[ExtractionTemplate]:
        """
        Toggle the favorite status of an extraction template.

        Args:
            db: Database session
            template_id: Template UUID

        Returns:
            Updated extraction template object if found, None otherwise
        """
        result = await db.execute(
            select(ExtractionTemplate).where(ExtractionTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return None

        db_template.is_favorite = not db_template.is_favorite
        db_template.updated_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def get_favorites(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[ExtractionTemplate], int]:
        """
        Get all favorite extraction templates.

        Args:
            db: Database session
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return

        Returns:
            Tuple of (list of favorite templates, total count)
        """
        # Build query for favorites only
        query = select(ExtractionTemplate).where(
            ExtractionTemplate.is_favorite == True
        ).order_by(ExtractionTemplate.created_at.desc())

        count_query = select(func.count()).select_from(ExtractionTemplate).where(
            ExtractionTemplate.is_favorite == True
        )

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        templates = result.scalars().all()

        return list(templates), total

    @staticmethod
    async def export_templates(
        db: AsyncSession,
        template_ids: Optional[List[UUID]] = None
    ) -> Dict[str, Any]:
        """
        Export extraction templates to JSON format.

        Args:
            db: Database session
            template_ids: Optional list of template IDs to export. If None, exports all.

        Returns:
            Dictionary containing exported templates with version info
        """
        # Build query
        query = select(ExtractionTemplate)

        if template_ids:
            query = query.where(ExtractionTemplate.id.in_(template_ids))

        # Execute query
        result = await db.execute(query)
        templates = result.scalars().all()

        # Convert to export format
        exported_templates = []
        for template in templates:
            exported_templates.append({
                "name": template.name,
                "description": template.description,
                "layer_indices": template.layer_indices,
                "hook_types": template.hook_types,
                "max_samples": template.max_samples,
                "batch_size": template.batch_size,
                "top_k_examples": template.top_k_examples,
                "is_favorite": template.is_favorite,
                "extra_metadata": template.extra_metadata,
            })

        return {
            "version": "1.0",
            "export_date": datetime.now(UTC).isoformat(),
            "count": len(exported_templates),
            "templates": exported_templates
        }

    @staticmethod
    async def import_templates(
        db: AsyncSession,
        import_data: Dict[str, Any],
        overwrite_duplicates: bool = False
    ) -> Dict[str, Any]:
        """
        Import extraction templates from JSON format.

        Args:
            db: Database session
            import_data: Import data containing templates
            overwrite_duplicates: Whether to overwrite templates with same name

        Returns:
            Dictionary containing import results (created, skipped, updated counts)
        """
        # Validate import data
        if "templates" not in import_data:
            raise ValueError("Import data must contain 'templates' key")

        templates_data = import_data["templates"]
        created_count = 0
        updated_count = 0
        skipped_count = 0
        errors = []

        for template_data in templates_data:
            try:
                # Check if template with same name exists
                result = await db.execute(
                    select(ExtractionTemplate).where(
                        ExtractionTemplate.name == template_data["name"]
                    )
                )
                existing_template = result.scalar_one_or_none()

                if existing_template:
                    if overwrite_duplicates:
                        # Update existing template
                        existing_template.description = template_data.get("description")
                        existing_template.layer_indices = template_data["layer_indices"]
                        existing_template.hook_types = template_data["hook_types"]
                        existing_template.max_samples = template_data.get("max_samples", 1000)
                        existing_template.batch_size = template_data.get("batch_size", 32)
                        existing_template.top_k_examples = template_data.get("top_k_examples", 10)
                        existing_template.is_favorite = template_data.get("is_favorite", False)
                        existing_template.extra_metadata = template_data.get("extra_metadata", {})
                        existing_template.updated_at = datetime.now(UTC)
                        updated_count += 1
                    else:
                        # Skip duplicate
                        skipped_count += 1
                else:
                    # Create new template
                    new_template = ExtractionTemplate(
                        name=template_data["name"],
                        description=template_data.get("description"),
                        layer_indices=template_data["layer_indices"],
                        hook_types=template_data["hook_types"],
                        max_samples=template_data.get("max_samples", 1000),
                        batch_size=template_data.get("batch_size", 32),
                        top_k_examples=template_data.get("top_k_examples", 10),
                        is_favorite=template_data.get("is_favorite", False),
                        extra_metadata=template_data.get("extra_metadata", {}),
                    )
                    db.add(new_template)
                    created_count += 1

            except Exception as e:
                errors.append({
                    "template_name": template_data.get("name", "unknown"),
                    "error": str(e)
                })

        # Commit all changes
        await db.commit()

        return {
            "created": created_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "errors": errors,
            "total_processed": created_count + updated_count + skipped_count
        }
