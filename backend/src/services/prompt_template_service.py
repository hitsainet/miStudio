"""
PromptTemplate service layer for business logic.

This module contains the PromptTemplateService class which handles all
prompt template-related business logic and database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, UTC

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.prompt_template import PromptTemplate
from ..schemas.prompt_template import (
    PromptTemplateCreate,
    PromptTemplateUpdate,
)


class PromptTemplateService:
    """Service class for prompt template operations."""

    @staticmethod
    async def create_template(
        db: AsyncSession,
        template: PromptTemplateCreate
    ) -> PromptTemplate:
        """
        Create a new prompt template.

        Args:
            db: Database session
            template: Template creation data

        Returns:
            Created prompt template object
        """
        db_template = PromptTemplate(
            name=template.name,
            description=template.description,
            prompts=template.prompts,
            is_favorite=template.is_favorite,
            tags=template.tags or [],
        )

        db.add(db_template)
        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def get_template(
        db: AsyncSession,
        template_id: UUID
    ) -> Optional[PromptTemplate]:
        """
        Get a prompt template by ID.

        Args:
            db: Database session
            template_id: Template UUID

        Returns:
            PromptTemplate object if found, None otherwise
        """
        result = await db.execute(
            select(PromptTemplate).where(PromptTemplate.id == template_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_templates(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        is_favorite: Optional[bool] = None,
        tag: Optional[str] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> tuple[List[PromptTemplate], int]:
        """
        List prompt templates with filtering, pagination, and sorting.

        Args:
            db: Database session
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            search: Search query for name or description
            is_favorite: Filter by favorite status
            tag: Filter by tag (matches if template contains this tag)
            sort_by: Column to sort by
            order: Sort order ('asc' or 'desc')

        Returns:
            Tuple of (list of templates, total count)
        """
        # Build base query
        query = select(PromptTemplate)
        count_query = select(func.count()).select_from(PromptTemplate)

        # Apply filters
        filters = []

        if search:
            search_filter = or_(
                PromptTemplate.name.ilike(f"%{search}%"),
                PromptTemplate.description.ilike(f"%{search}%")
            )
            filters.append(search_filter)

        if is_favorite is not None:
            filters.append(PromptTemplate.is_favorite == is_favorite)

        if tag is not None:
            # JSONB contains operator for tag matching
            filters.append(PromptTemplate.tags.contains([tag]))

        if filters:
            query = query.where(*filters)
            count_query = count_query.where(*filters)

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply sorting
        sort_column = getattr(PromptTemplate, sort_by, PromptTemplate.created_at)
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
        updates: PromptTemplateUpdate
    ) -> Optional[PromptTemplate]:
        """
        Update a prompt template.

        Args:
            db: Database session
            template_id: Template UUID
            updates: Update data

        Returns:
            Updated PromptTemplate object if found, None otherwise
        """
        # Get existing template
        result = await db.execute(
            select(PromptTemplate).where(PromptTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return None

        # Apply updates
        update_data = updates.model_dump(exclude_unset=True)

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
        Delete a prompt template.

        Args:
            db: Database session
            template_id: Template UUID

        Returns:
            True if template was deleted, False if not found
        """
        result = await db.execute(
            select(PromptTemplate).where(PromptTemplate.id == template_id)
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
    ) -> Optional[PromptTemplate]:
        """
        Toggle the favorite status of a prompt template.

        Args:
            db: Database session
            template_id: Template UUID

        Returns:
            Updated PromptTemplate object if found, None otherwise
        """
        result = await db.execute(
            select(PromptTemplate).where(PromptTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return None

        db_template.is_favorite = not db_template.is_favorite

        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def duplicate_template(
        db: AsyncSession,
        template_id: UUID
    ) -> Optional[PromptTemplate]:
        """
        Duplicate a prompt template with a new name.

        Args:
            db: Database session
            template_id: Template UUID to duplicate

        Returns:
            New PromptTemplate object if source found, None otherwise
        """
        result = await db.execute(
            select(PromptTemplate).where(PromptTemplate.id == template_id)
        )
        source_template = result.scalar_one_or_none()

        if not source_template:
            return None

        # Create new template with "(Copy)" suffix
        db_template = PromptTemplate(
            name=f"{source_template.name} (Copy)",
            description=source_template.description,
            prompts=source_template.prompts.copy() if source_template.prompts else [],
            is_favorite=False,  # Don't copy favorite status
            tags=source_template.tags.copy() if source_template.tags else [],
        )

        db.add(db_template)
        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def get_favorites(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[PromptTemplate], int]:
        """
        Get all favorite prompt templates.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Tuple of (list of favorite templates, total count)
        """
        return await PromptTemplateService.list_templates(
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
        Export prompt templates to JSON format.

        Args:
            db: Database session
            template_ids: Optional list of template IDs to export. If None, exports all.

        Returns:
            Dictionary containing export data with version and templates
        """
        # Build query
        query = select(PromptTemplate)

        if template_ids:
            query = query.where(PromptTemplate.id.in_(template_ids))

        # Execute query
        result = await db.execute(query)
        templates = result.scalars().all()

        # Convert to response format
        templates_data = [
            {
                "id": str(template.id),
                "name": template.name,
                "description": template.description,
                "prompts": template.prompts,
                "is_favorite": template.is_favorite,
                "tags": template.tags or [],
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
        Import prompt templates from JSON format.

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
                select(PromptTemplate).where(PromptTemplate.name == name)
            )
            existing = result.scalar_one_or_none()

            if existing:
                if overwrite_duplicates:
                    # Update existing template
                    for field in ["description", "prompts", "is_favorite", "tags"]:
                        if field in template_data:
                            setattr(existing, field, template_data[field])
                    updated_count += 1
                else:
                    skipped_count += 1
                    continue
            else:
                # Create new template
                db_template = PromptTemplate(
                    name=template_data["name"],
                    description=template_data.get("description"),
                    prompts=template_data["prompts"],
                    is_favorite=template_data.get("is_favorite", False),
                    tags=template_data.get("tags", []),
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
