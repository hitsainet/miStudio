"""
Prompt Template API endpoints.

This module contains all FastAPI routes for prompt template management operations.
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....schemas.prompt_template import (
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptTemplateResponse,
    PromptTemplateListResponse,
)
from ....services.prompt_template_service import PromptTemplateService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompt-templates", tags=["prompt-templates"])


@router.post("", response_model=PromptTemplateResponse, status_code=201)
async def create_template(
    template: PromptTemplateCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new prompt template.

    Args:
        template: Template creation data
        db: Database session

    Returns:
        Created prompt template

    Raises:
        HTTPException: If template creation fails
    """
    try:
        db_template = await PromptTemplateService.create_template(db, template)
        return db_template
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create prompt template: {str(e)}"
        )


@router.get("", response_model=PromptTemplateListResponse)
async def list_templates(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query for name or description"),
    is_favorite: Optional[bool] = Query(None, description="Filter by favorite status"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    sort_by: str = Query("created_at", description="Sort by field"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    db: AsyncSession = Depends(get_db)
):
    """
    List prompt templates with filtering, pagination, and sorting.

    Args:
        page: Page number (1-indexed)
        limit: Items per page
        search: Search query for name or description
        is_favorite: Filter by favorite status
        tag: Filter by tag
        sort_by: Column to sort by
        order: Sort order (asc or desc)
        db: Database session

    Returns:
        Paginated list of prompt templates with metadata
    """
    skip = (page - 1) * limit

    templates, total = await PromptTemplateService.list_templates(
        db=db,
        skip=skip,
        limit=limit,
        search=search,
        is_favorite=is_favorite,
        tag=tag,
        sort_by=sort_by,
        order=order
    )

    total_pages = (total + limit - 1) // limit
    has_next = page < total_pages
    has_prev = page > 1

    return PromptTemplateListResponse(
        data=templates,
        pagination={
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev
        }
    )


@router.get("/favorites", response_model=PromptTemplateListResponse)
async def list_favorites(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db)
):
    """
    List only favorite prompt templates.

    Args:
        page: Page number (1-indexed)
        limit: Items per page
        db: Database session

    Returns:
        Paginated list of favorite prompt templates
    """
    skip = (page - 1) * limit

    templates, total = await PromptTemplateService.get_favorites(
        db=db,
        skip=skip,
        limit=limit
    )

    total_pages = (total + limit - 1) // limit
    has_next = page < total_pages
    has_prev = page > 1

    return PromptTemplateListResponse(
        data=templates,
        pagination={
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev
        }
    )


@router.get("/{template_id}", response_model=PromptTemplateResponse)
async def get_template(
    template_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a prompt template by ID.

    Args:
        template_id: Template UUID
        db: Database session

    Returns:
        Prompt template details

    Raises:
        HTTPException: If template not found
    """
    template = await PromptTemplateService.get_template(db, template_id)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt template {template_id} not found"
        )

    return template


@router.patch("/{template_id}", response_model=PromptTemplateResponse)
async def update_template(
    template_id: UUID,
    updates: PromptTemplateUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a prompt template.

    Args:
        template_id: Template UUID
        updates: Update data
        db: Database session

    Returns:
        Updated prompt template

    Raises:
        HTTPException: If template not found
    """
    template = await PromptTemplateService.update_template(db, template_id, updates)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt template {template_id} not found"
        )

    return template


@router.delete("/{template_id}", status_code=204)
async def delete_template(
    template_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a prompt template.

    Args:
        template_id: Template UUID
        db: Database session

    Raises:
        HTTPException: If template not found
    """
    deleted = await PromptTemplateService.delete_template(db, template_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt template {template_id} not found"
        )


@router.post("/{template_id}/favorite", response_model=PromptTemplateResponse)
async def toggle_favorite(
    template_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Toggle the favorite status of a prompt template.

    Args:
        template_id: Template UUID
        db: Database session

    Returns:
        Updated prompt template with toggled favorite status

    Raises:
        HTTPException: If template not found
    """
    template = await PromptTemplateService.toggle_favorite(db, template_id)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt template {template_id} not found"
        )

    return template


@router.post("/{template_id}/duplicate", response_model=PromptTemplateResponse, status_code=201)
async def duplicate_template(
    template_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Duplicate a prompt template with a new name.

    Args:
        template_id: Template UUID to duplicate
        db: Database session

    Returns:
        New prompt template created from the source

    Raises:
        HTTPException: If source template not found
    """
    template = await PromptTemplateService.duplicate_template(db, template_id)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt template {template_id} not found"
        )

    return template


@router.post("/export")
async def export_templates(
    template_ids: Optional[List[UUID]] = Body(None, description="List of template IDs to export. If empty, exports all."),
    db: AsyncSession = Depends(get_db)
):
    """
    Export prompt templates to JSON format.

    Args:
        template_ids: Optional list of template IDs to export. If None, exports all templates.
        db: Database session

    Returns:
        JSON response containing exported templates with version info

    Raises:
        HTTPException: If export fails
    """
    try:
        export_data = await PromptTemplateService.export_templates(db, template_ids)
        return JSONResponse(content=export_data)
    except Exception as e:
        logger.error(f"Failed to export templates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export templates: {str(e)}"
        )


@router.post("/import")
async def import_templates(
    import_data: Dict[str, Any] = Body(..., description="Import data containing templates"),
    overwrite_duplicates: bool = Body(False, description="Whether to overwrite templates with the same name"),
    db: AsyncSession = Depends(get_db)
):
    """
    Import prompt templates from JSON format.

    Args:
        import_data: Import data containing templates in JSON format
        overwrite_duplicates: Whether to overwrite templates with the same name
        db: Database session

    Returns:
        JSON response containing import results (created, updated, skipped counts)

    Raises:
        HTTPException: If import fails or data is invalid
    """
    try:
        result = await PromptTemplateService.import_templates(
            db,
            import_data,
            overwrite_duplicates
        )
        return JSONResponse(content=result)
    except ValueError as e:
        logger.error(f"Invalid import data: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid import data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to import templates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import templates: {str(e)}"
        )
