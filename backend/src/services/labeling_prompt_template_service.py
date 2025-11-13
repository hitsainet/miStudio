"""
LabelingPromptTemplate service layer for business logic.

This module contains the LabelingPromptTemplateService class which handles all
labeling prompt template-related business logic and database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, UTC

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from ..models.labeling_prompt_template import LabelingPromptTemplate
from ..models.labeling_job import LabelingJob
from ..schemas.labeling_prompt_template import (
    LabelingPromptTemplateCreate,
    LabelingPromptTemplateUpdate,
)


class LabelingPromptTemplateService:
    """Service class for labeling prompt template operations."""

    @staticmethod
    async def create_template(
        db: AsyncSession,
        template: LabelingPromptTemplateCreate
    ) -> LabelingPromptTemplate:
        """
        Create a new labeling prompt template.

        If is_default is True, unsets any existing default template first.

        Args:
            db: Database session
            template: Template creation data

        Returns:
            Created labeling prompt template object
        """
        # If this template should be default, unset existing default
        if template.is_default:
            await LabelingPromptTemplateService._unset_all_defaults(db)

        # Generate template ID
        template_id = f"lpt_{uuid4().hex[:16]}"

        db_template = LabelingPromptTemplate(
            id=template_id,
            name=template.name,
            description=template.description,
            system_message=template.system_message,
            user_prompt_template=template.user_prompt_template,
            temperature=template.temperature,
            max_tokens=template.max_tokens,
            top_p=template.top_p,
            is_default=template.is_default,
            is_system=False,  # User-created templates are never system templates
            created_by=None,  # TODO: Add user ID when auth is implemented
        )

        db.add(db_template)
        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def get_template(
        db: AsyncSession,
        template_id: str
    ) -> Optional[LabelingPromptTemplate]:
        """
        Get a labeling prompt template by ID.

        Args:
            db: Database session
            template_id: Template ID

        Returns:
            LabelingPromptTemplate object if found, None otherwise
        """
        result = await db.execute(
            select(LabelingPromptTemplate).where(LabelingPromptTemplate.id == template_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_default_template(
        db: AsyncSession
    ) -> Optional[LabelingPromptTemplate]:
        """
        Get the default labeling prompt template.

        Args:
            db: Database session

        Returns:
            Default LabelingPromptTemplate object if found, None otherwise
        """
        result = await db.execute(
            select(LabelingPromptTemplate).where(LabelingPromptTemplate.is_default == True)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_templates(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        include_system: bool = True,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> tuple[List[LabelingPromptTemplate], int]:
        """
        List labeling prompt templates with filtering, pagination, and sorting.

        Args:
            db: Database session
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            search: Search query for name or description
            include_system: Whether to include system templates
            sort_by: Column to sort by
            order: Sort order ('asc' or 'desc')

        Returns:
            Tuple of (list of templates, total count)
        """
        # Build base query
        query = select(LabelingPromptTemplate)
        count_query = select(func.count()).select_from(LabelingPromptTemplate)

        # Apply filters
        filters = []

        if search:
            search_filter = or_(
                LabelingPromptTemplate.name.ilike(f"%{search}%"),
                LabelingPromptTemplate.description.ilike(f"%{search}%")
            )
            filters.append(search_filter)

        if not include_system:
            filters.append(LabelingPromptTemplate.is_system == False)

        if filters:
            query = query.where(*filters)
            count_query = count_query.where(*filters)

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply sorting
        sort_column = getattr(LabelingPromptTemplate, sort_by, LabelingPromptTemplate.created_at)
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
        template_id: str,
        updates: LabelingPromptTemplateUpdate
    ) -> Optional[LabelingPromptTemplate]:
        """
        Update a labeling prompt template.

        System templates cannot be modified. If is_default is being set to True,
        unsets any existing default template first.

        Args:
            db: Database session
            template_id: Template ID
            updates: Update data

        Returns:
            Updated LabelingPromptTemplate object if found and not system, None otherwise
        """
        # Get existing template
        result = await db.execute(
            select(LabelingPromptTemplate).where(LabelingPromptTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return None

        # Cannot modify system templates
        if db_template.is_system:
            raise ValueError("Cannot modify system templates")

        # Apply updates
        update_data = updates.model_dump(exclude_unset=True)

        # If setting this as default, unset existing default
        if update_data.get("is_default") is True:
            await LabelingPromptTemplateService._unset_all_defaults(db)

        for field, value in update_data.items():
            setattr(db_template, field, value)

        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def delete_template(
        db: AsyncSession,
        template_id: str
    ) -> Dict[str, Any]:
        """
        Delete a labeling prompt template.

        System templates and templates in use cannot be deleted.
        Foreign key constraint (ON DELETE RESTRICT) prevents deletion
        if any labeling jobs reference this template.

        Args:
            db: Database session
            template_id: Template ID

        Returns:
            Dictionary with success status and message

        Raises:
            ValueError: If template is a system template
            IntegrityError: If template is in use by labeling jobs
        """
        result = await db.execute(
            select(LabelingPromptTemplate).where(LabelingPromptTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return {
                "success": False,
                "message": "Template not found"
            }

        # Cannot delete system templates
        if db_template.is_system:
            raise ValueError("Cannot delete system templates")

        try:
            await db.delete(db_template)
            await db.commit()
            return {
                "success": True,
                "message": f"Template '{db_template.name}' deleted successfully"
            }
        except IntegrityError:
            await db.rollback()
            return {
                "success": False,
                "message": f"Cannot delete template '{db_template.name}' because it is in use by one or more labeling jobs"
            }

    @staticmethod
    async def set_default_template(
        db: AsyncSession,
        template_id: str
    ) -> Optional[LabelingPromptTemplate]:
        """
        Set a template as the default.

        Unsets any existing default template first.

        Args:
            db: Database session
            template_id: Template ID to set as default

        Returns:
            Updated LabelingPromptTemplate object if found, None otherwise
        """
        # Get template
        result = await db.execute(
            select(LabelingPromptTemplate).where(LabelingPromptTemplate.id == template_id)
        )
        db_template = result.scalar_one_or_none()

        if not db_template:
            return None

        # Unset existing default
        await LabelingPromptTemplateService._unset_all_defaults(db)

        # Set this template as default
        db_template.is_default = True

        await db.commit()
        await db.refresh(db_template)

        return db_template

    @staticmethod
    async def get_template_usage_count(
        db: AsyncSession,
        template_id: str
    ) -> int:
        """
        Get the number of labeling jobs using a specific template.

        Args:
            db: Database session
            template_id: Template ID

        Returns:
            Count of labeling jobs using this template
        """
        result = await db.execute(
            select(func.count()).select_from(LabelingJob).where(
                LabelingJob.prompt_template_id == template_id
            )
        )
        return result.scalar() or 0

    @staticmethod
    async def _unset_all_defaults(db: AsyncSession) -> None:
        """
        Unset all default templates.

        Internal helper method to ensure only one default template exists.

        Args:
            db: Database session
        """
        result = await db.execute(
            select(LabelingPromptTemplate).where(LabelingPromptTemplate.is_default == True)
        )
        existing_defaults = result.scalars().all()

        for template in existing_defaults:
            template.is_default = False

        # Don't commit here - let the caller commit

    @staticmethod
    async def export_templates(
        db: AsyncSession,
        template_ids: Optional[list[str]] = None
    ) -> dict:
        """
        Export labeling prompt templates to a portable format.

        Args:
            db: Database session
            template_ids: Optional list of template IDs to export. If None, exports all custom templates.

        Returns:
            Dictionary with version, export timestamp, and list of templates
        """
        from datetime import datetime, timezone

        # Build query
        query = select(LabelingPromptTemplate).where(
            LabelingPromptTemplate.is_system == False
        )

        if template_ids:
            query = query.where(LabelingPromptTemplate.id.in_(template_ids))

        result = await db.execute(query)
        templates = result.scalars().all()

        # Convert to export format (exclude system fields)
        export_items = []
        for template in templates:
            export_items.append({
                "name": template.name,
                "description": template.description,
                "system_message": template.system_message,
                "user_prompt_template": template.user_prompt_template,
                "temperature": template.temperature,
                "max_tokens": template.max_tokens,
                "top_p": template.top_p,
                "is_default": template.is_default
            })

        return {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc),
            "templates": export_items
        }

    @staticmethod
    async def import_templates(
        db: AsyncSession,
        import_data: dict,
        overwrite_duplicates: bool = False
    ) -> dict:
        """
        Import labeling prompt templates from export data.

        Args:
            db: Database session
            import_data: Export data containing version, timestamp, and templates
            overwrite_duplicates: Whether to overwrite templates with same name

        Returns:
            Dictionary with import statistics and details
        """
        from datetime import datetime, timezone
        import uuid

        # Validate version
        if import_data.get("version") != "1.0":
            return {
                "success": False,
                "message": f"Unsupported export version: {import_data.get('version')}",
                "imported_count": 0,
                "skipped_count": 0,
                "overwritten_count": 0,
                "failed_count": 0,
                "details": []
            }

        templates_data = import_data.get("templates", [])
        imported_count = 0
        skipped_count = 0
        overwritten_count = 0
        failed_count = 0
        details = []

        for template_data in templates_data:
            try:
                template_name = template_data.get("name")

                # Check for existing template with same name
                result = await db.execute(
                    select(LabelingPromptTemplate).where(
                        LabelingPromptTemplate.name == template_name
                    )
                )
                existing_template = result.scalar_one_or_none()

                if existing_template:
                    if not overwrite_duplicates:
                        skipped_count += 1
                        details.append(f"Skipped '{template_name}' (already exists)")
                        continue
                    else:
                        # Update existing template
                        existing_template.description = template_data.get("description")
                        existing_template.system_message = template_data.get("system_message")
                        existing_template.user_prompt_template = template_data.get("user_prompt_template")
                        existing_template.temperature = template_data.get("temperature", 0.3)
                        existing_template.max_tokens = template_data.get("max_tokens", 50)
                        existing_template.top_p = template_data.get("top_p", 0.9)
                        existing_template.updated_at = datetime.now(timezone.utc)

                        # Handle default status
                        if template_data.get("is_default"):
                            await LabelingPromptTemplateService._unset_all_defaults(db)
                            existing_template.is_default = True

                        overwritten_count += 1
                        details.append(f"Overwritten '{template_name}'")
                else:
                    # Create new template
                    new_template = LabelingPromptTemplate(
                        id=f"tmpl_{uuid.uuid4().hex[:12]}",
                        name=template_name,
                        description=template_data.get("description"),
                        system_message=template_data.get("system_message"),
                        user_prompt_template=template_data.get("user_prompt_template"),
                        temperature=template_data.get("temperature", 0.3),
                        max_tokens=template_data.get("max_tokens", 50),
                        top_p=template_data.get("top_p", 0.9),
                        is_default=False,  # Don't import as default
                        is_system=False,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )

                    db.add(new_template)
                    imported_count += 1
                    details.append(f"Imported '{template_name}'")

            except Exception as e:
                failed_count += 1
                details.append(f"Failed to import '{template_data.get('name', 'unknown')}': {str(e)}")

        await db.commit()

        total_processed = imported_count + skipped_count + overwritten_count + failed_count
        success_message = f"Import completed: {imported_count} imported, {overwritten_count} overwritten, {skipped_count} skipped, {failed_count} failed"

        return {
            "success": True,
            "message": success_message,
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "overwritten_count": overwritten_count,
            "failed_count": failed_count,
            "details": details
        }
