"""
Labeling Prompt Template API endpoints.

This module contains all FastAPI routes for labeling prompt template management operations.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....schemas.labeling_prompt_template import (
    LabelingPromptTemplateCreate,
    LabelingPromptTemplateUpdate,
    LabelingPromptTemplateResponse,
    LabelingPromptTemplateListResponse,
    LabelingPromptTemplateDeleteResponse,
    LabelingPromptTemplateSetDefaultResponse,
)
from ....services.labeling_prompt_template_service import LabelingPromptTemplateService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/labeling-prompt-templates", tags=["labeling-prompt-templates"])


@router.post("", response_model=LabelingPromptTemplateResponse, status_code=201)
async def create_template(
    template: LabelingPromptTemplateCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new labeling prompt template.

    Args:
        template: Template creation data
        db: Database session

    Returns:
        Created labeling prompt template

    Raises:
        HTTPException: If template creation fails
    """
    try:
        db_template = await LabelingPromptTemplateService.create_template(db, template)
        return db_template
    except Exception as e:
        logger.error(f"Failed to create labeling prompt template: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create labeling prompt template: {str(e)}"
        )


@router.get("", response_model=LabelingPromptTemplateListResponse)
async def list_templates(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query for name or description"),
    include_system: bool = Query(True, description="Include system templates in results"),
    sort_by: str = Query("created_at", description="Sort by field"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    db: AsyncSession = Depends(get_db)
):
    """
    List labeling prompt templates with filtering, pagination, and sorting.

    Args:
        page: Page number (1-indexed)
        limit: Items per page
        search: Search query for name or description
        include_system: Include system templates in results
        sort_by: Column to sort by
        order: Sort order (asc or desc)
        db: Database session

    Returns:
        Paginated list of labeling prompt templates with metadata
    """
    skip = (page - 1) * limit

    templates, total = await LabelingPromptTemplateService.list_templates(
        db=db,
        skip=skip,
        limit=limit,
        search=search,
        include_system=include_system,
        sort_by=sort_by,
        order=order
    )

    total_pages = (total + limit - 1) // limit
    has_next = page < total_pages
    has_prev = page > 1

    return LabelingPromptTemplateListResponse(
        data=templates,
        meta={
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev
        }
    )


@router.get("/default", response_model=LabelingPromptTemplateResponse)
async def get_default_template(
    db: AsyncSession = Depends(get_db)
):
    """
    Get the default labeling prompt template.

    Args:
        db: Database session

    Returns:
        Default labeling prompt template

    Raises:
        HTTPException: If no default template exists
    """
    template = await LabelingPromptTemplateService.get_default_template(db)

    if not template:
        raise HTTPException(
            status_code=404,
            detail="No default labeling prompt template found"
        )

    return template


@router.get("/{template_id}", response_model=LabelingPromptTemplateResponse)
async def get_template(
    template_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a labeling prompt template by ID.

    Args:
        template_id: Template ID
        db: Database session

    Returns:
        Labeling prompt template details

    Raises:
        HTTPException: If template not found
    """
    template = await LabelingPromptTemplateService.get_template(db, template_id)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Labeling prompt template {template_id} not found"
        )

    return template


@router.patch("/{template_id}", response_model=LabelingPromptTemplateResponse)
async def update_template(
    template_id: str,
    updates: LabelingPromptTemplateUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a labeling prompt template.

    System templates cannot be modified.

    Args:
        template_id: Template ID
        updates: Update data
        db: Database session

    Returns:
        Updated labeling prompt template

    Raises:
        HTTPException: If template not found or is a system template
    """
    try:
        template = await LabelingPromptTemplateService.update_template(db, template_id, updates)

        if not template:
            raise HTTPException(
                status_code=404,
                detail=f"Labeling prompt template {template_id} not found"
            )

        return template
    except ValueError as e:
        # Raised when trying to modify a system template
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update labeling prompt template {template_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update template: {str(e)}"
        )


@router.delete("/{template_id}", response_model=LabelingPromptTemplateDeleteResponse)
async def delete_template(
    template_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a labeling prompt template.

    System templates and templates in use cannot be deleted.

    Args:
        template_id: Template ID
        db: Database session

    Returns:
        Deletion result with success status and message

    Raises:
        HTTPException: If template is a system template or in use
    """
    try:
        result = await LabelingPromptTemplateService.delete_template(db, template_id)

        if not result["success"]:
            if "not found" in result["message"]:
                raise HTTPException(
                    status_code=404,
                    detail=result["message"]
                )
            else:
                raise HTTPException(
                    status_code=409,
                    detail=result["message"]
                )

        return LabelingPromptTemplateDeleteResponse(
            id=template_id,
            message=result["message"],
            success=True
        )
    except ValueError as e:
        # Raised when trying to delete a system template
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete labeling prompt template {template_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete template: {str(e)}"
        )


@router.post("/{template_id}/set-default", response_model=LabelingPromptTemplateSetDefaultResponse)
async def set_default_template(
    template_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Set a template as the default.

    Unsets any existing default template first.

    Args:
        template_id: Template ID to set as default
        db: Database session

    Returns:
        Updated template with confirmation message

    Raises:
        HTTPException: If template not found
    """
    template = await LabelingPromptTemplateService.set_default_template(db, template_id)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Labeling prompt template {template_id} not found"
        )

    return LabelingPromptTemplateSetDefaultResponse(
        id=template.id,
        name=template.name,
        message=f"Template '{template.name}' set as default",
        success=True
    )


@router.get("/{template_id}/usage-count", response_model=dict)
async def get_template_usage_count(
    template_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the number of labeling jobs using a specific template.

    Args:
        template_id: Template ID
        db: Database session

    Returns:
        Dictionary with template_id and usage_count
    """
    count = await LabelingPromptTemplateService.get_template_usage_count(db, template_id)

    return {
        "template_id": template_id,
        "usage_count": count
    }


@router.post("/export")
async def export_templates(
    template_ids: Optional[list[str]] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Export labeling prompt templates to JSON format.

    Args:
        template_ids: Optional list of template IDs to export. If not provided, exports all custom templates.
        db: Database session

    Returns:
        Export data with version, timestamp, and templates in portable format
    """
    try:
        export_data = await LabelingPromptTemplateService.export_templates(db, template_ids)
        return export_data
    except Exception as e:
        logger.error(f"Failed to export templates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export templates: {str(e)}"
        )


@router.post("/import")
async def import_templates(
    import_request: dict,
    db: AsyncSession = Depends(get_db)
):
    """
    Import labeling prompt templates from JSON format.

    Args:
        import_request: Import data containing version, timestamp, templates, and overwrite_duplicates flag
        db: Database session

    Returns:
        Import result with statistics and details
    """
    try:
        overwrite_duplicates = import_request.get("overwrite_duplicates", False)
        result = await LabelingPromptTemplateService.import_templates(
            db,
            import_request,
            overwrite_duplicates
        )
        return result
    except Exception as e:
        logger.error(f"Failed to import templates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import templates: {str(e)}"
        )
