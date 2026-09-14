"""
LabelingPromptTemplate service layer for business logic.

This module contains the LabelingPromptTemplateService class which handles all
labeling prompt template-related business logic and database operations.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, UTC

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

from ..models.labeling_prompt_template import LabelingPromptTemplate
from ..models.labeling_job import LabelingJob
from ..schemas.labeling_prompt_template import (
    LabelingPromptTemplateCreate,
    LabelingPromptTemplateUpdate,
)


#: Identity and bookkeeping — meaningless in another database.
#:
#: `is_system` is excluded deliberately: an import must never be able to mint an
#: undeletable template. Everything else is prompt-defining and travels.
_NEVER_EXPORTED = {
    "id",
    "is_system",
    "created_by",
    "created_at",
    "updated_at",
}


def _importable_fields(payload: dict) -> dict:
    """The columns an import payload may set, taken from the payload.

    DERIVED, for the same reason export is. The import path hand-listed six
    fields, so a round trip through export/import reset template_type,
    max_examples, the marker and context switches, both negative-example
    settings, include_nlp_analysis — and this arc's example_sampling and
    activation_display — to their column defaults.

    A key absent from the payload is LEFT ALONE rather than reset, so importing
    an older export cannot silently revert fields it never carried.

    `is_default` and `is_system` are excluded here and handled explicitly by the
    caller: an import must not be able to grant itself protection or seize the
    default slot as a side effect of carrying a field.
    """
    excluded = _NEVER_EXPORTED | {"is_default"}
    return {
        column.name: payload[column.name]
        for column in LabelingPromptTemplate.__table__.columns
        if column.name not in excluded and column.name in payload
    }


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

        # Copy EVERY field the request schema accepts.
        #
        # This previously listed eight fields by hand and silently discarded the
        # other twelve — template_type, max_examples, the prefix/suffix and
        # marker settings, the logit-effect counts, the negative-example
        # settings, is_detection_template and include_nlp_analysis. The API
        # accepted them, returned 201, and stored column defaults instead, so a
        # template created through the API or the UI did not behave the way it
        # was configured and nothing said so. Deriving the field list from the
        # schema means a new field cannot be forgotten here again.
        _NEVER_FROM_REQUEST = {"is_system", "created_by", "id"}
        values = {
            k: v for k, v in template.model_dump().items()
            if k not in _NEVER_FROM_REQUEST
        }
        db_template = LabelingPromptTemplate(
            id=template_id,
            is_system=False,  # User-created templates are never system templates
            created_by=None,  # TODO: Add user ID when auth is implemented
            **values,
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
    async def clone_template(
        db: AsyncSession,
        template_id: str,
        new_name: Optional[str] = None
    ) -> Optional[LabelingPromptTemplate]:
        """
        Clone an existing labeling prompt template.

        Useful for creating editable copies of system templates or duplicating user templates.
        The cloned template will always be a user template (is_system=False) and not default.

        Args:
            db: Database session
            template_id: ID of template to clone
            new_name: Optional custom name for the clone. If None, appends " (Copy)" to original name

        Returns:
            Cloned LabelingPromptTemplate object if source found, None otherwise
        """
        # Get source template
        result = await db.execute(
            select(LabelingPromptTemplate).where(LabelingPromptTemplate.id == template_id)
        )
        source_template = result.scalar_one_or_none()

        if not source_template:
            return None

        # Generate new template ID
        new_template_id = f"lpt_{uuid4().hex[:16]}"

        # Determine clone name
        clone_name = new_name if new_name else f"{source_template.name} (Copy)"

        # DERIVE THE FIELD LIST FROM THE COLUMNS, exactly as create_template
        # above already does — and for exactly the same reason.
        #
        # This listed eighteen fields by hand and silently dropped three:
        # include_negative_examples, num_negative_examples and
        # include_nlp_analysis. A clone therefore did not behave like its
        # source, and nothing said so.
        #
        # That is worse than an ordinary copy bug, because "duplicate the
        # baseline and change one field" is how an experimental arm gets built.
        # An A/B trial constructed this way would have moved FOUR variables
        # while its author believed one had moved, and the result would have
        # been attributed to the one they changed.
        _NEVER_CLONED = {
            "id",           # a clone gets its own
            "name",         # handled above
            "is_default",   # clones are never default
            "is_system",    # clones are always user-editable
            "created_by",
            "created_at",
            "updated_at",
        }
        carried = {
            column.name: getattr(source_template, column.name)
            for column in LabelingPromptTemplate.__table__.columns
            if column.name not in _NEVER_CLONED
        }

        cloned_template = LabelingPromptTemplate(
            id=new_template_id,
            name=clone_name,
            is_default=False,  # Clones are never default
            is_system=False,   # Clones are always user templates (editable)
            created_by=None,   # TODO: Add user ID when auth is implemented
            **carried,
        )

        db.add(cloned_template)
        await db.commit()
        await db.refresh(cloned_template)

        return cloned_template

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

        # DERIVED FROM THE COLUMNS, like create_template and clone_template.
        #
        # This hand-listed eight fields and silently dropped everything that
        # actually defines the prompt: template_type, max_examples, the
        # prefix/suffix and marker settings, the logit-effect counts, both
        # negative-example settings, include_nlp_analysis,
        # is_detection_template — and, once this arc added them,
        # example_sampling and activation_display.
        #
        # So the round trip destroyed the configuration while looking like it
        # worked. A researcher exports a validated stratified template, a
        # colleague imports it, and it runs top_k/absolute. Both believe they
        # are running the same judge; the export JSON carries no fingerprint to
        # contradict them.
        export_items = []
        for template in templates:
            item = {
                column.name: getattr(template, column.name)
                for column in LabelingPromptTemplate.__table__.columns
                if column.name not in _NEVER_EXPORTED
            }
            export_items.append(item)

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
                    # A SYSTEM TEMPLATE IS NOT OVERWRITABLE, HERE EITHER
                    # (MIS-E2E-108).
                    #
                    # `update_template` and `delete_template` both refuse one:
                    # `if db_template.is_system: raise ValueError(...)`. Import
                    # had neither guard. It matched on NAME alone and replaced
                    # `system_message`, `user_prompt_template` and the rest
                    # unconditionally — and would promote the row to
                    # `is_default`.
                    #
                    # So an import naming a seeded template (e.g. "Context-Aware
                    # Labeling", seeded `is_system=True`) with
                    # `overwrite_duplicates: true` replaced its prompt body and
                    # made it the default. Every subsequent bulk-labeling run —
                    # including runs billing the operator's OpenAI key against
                    # their corpus — then executed the imported instructions,
                    # while the UI still showed the template as protected and
                    # PATCH/DELETE still refused it. The tamper was invisible
                    # from the surface that is supposed to be authoritative.
                    #
                    # Note the create branch below already pins
                    # `is_default=False, is_system=False`; the overwrite branch
                    # simply was not given the same treatment.
                    if existing_template.is_system:
                        skipped_count += 1
                        details.append(
                            f"Skipped '{template_name}' (system template — "
                            f"protected from import overwrite)"
                        )
                        continue
                    if not overwrite_duplicates:
                        skipped_count += 1
                        details.append(f"Skipped '{template_name}' (already exists)")
                        continue
                    else:
                        # Update existing template — every importable column,
                        # not the six someone happened to list.
                        for column, value in _importable_fields(template_data).items():
                            setattr(existing_template, column, value)
                        existing_template.updated_at = datetime.now(timezone.utc)

                        # Handle default status. `is_system` is deliberately
                        # NOT read from the payload — an import must not be able
                        # to grant itself protection, any more than it may
                        # bypass it above.
                        if template_data.get("is_default"):
                            await LabelingPromptTemplateService._unset_all_defaults(db)
                            existing_template.is_default = True

                        overwritten_count += 1
                        details.append(f"Overwritten '{template_name}'")
                else:
                    # Create new template
                    fields = _importable_fields(template_data)
                    fields.pop("name", None)  # taken from template_name above
                    new_template = LabelingPromptTemplate(
                        id=f"tmpl_{uuid.uuid4().hex[:12]}",
                        name=template_name,
                        is_default=False,  # Don't import as default
                        is_system=False,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        **fields,
                    )

                    db.add(new_template)
                    imported_count += 1
                    details.append(f"Imported '{template_name}'")

            except Exception:
                template_name = template_data.get('name', 'unknown')
                logger.exception("Failed to import labeling prompt template %s", template_name)
                failed_count += 1
                details.append(f"Failed to import '{template_name}' — check the server log for details")

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
