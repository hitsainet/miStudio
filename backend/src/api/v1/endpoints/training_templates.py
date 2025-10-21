"""
Training Template API endpoints.

This module contains all FastAPI routes for training template management operations.
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....schemas.training_template import (
    TrainingTemplateCreate,
    TrainingTemplateUpdate,
    TrainingTemplateResponse,
    TrainingTemplateListResponse,
)
from ....services.training_template_service import TrainingTemplateService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training-templates", tags=["training-templates"])


@router.post("", response_model=TrainingTemplateResponse, status_code=201)
async def create_template(
    template: TrainingTemplateCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new training template.

    Args:
        template: Template creation data
        db: Database session

    Returns:
        Created training template

    Raises:
        HTTPException: If template creation fails
    """
    try:
        db_template = await TrainingTemplateService.create_template(db, template)
        return db_template
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create training template: {str(e)}"
        )


@router.get("", response_model=TrainingTemplateListResponse)
async def list_templates(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query for name or description"),
    is_favorite: Optional[bool] = Query(None, description="Filter by favorite status"),
    encoder_type: Optional[str] = Query(None, description="Filter by encoder architecture type (standard/skip/transcoder)"),
    sort_by: str = Query("created_at", description="Sort by field"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    db: AsyncSession = Depends(get_db)
):
    """
    List training templates with filtering, pagination, and sorting.

    Args:
        page: Page number (1-indexed)
        limit: Items per page
        search: Search query for name or description
        is_favorite: Filter by favorite status
        encoder_type: Filter by encoder architecture type
        sort_by: Column to sort by
        order: Sort order (asc or desc)
        db: Database session

    Returns:
        Paginated list of training templates with metadata
    """
    skip = (page - 1) * limit

    templates, total = await TrainingTemplateService.list_templates(
        db=db,
        skip=skip,
        limit=limit,
        search=search,
        is_favorite=is_favorite,
        encoder_type=encoder_type,
        sort_by=sort_by,
        order=order
    )

    total_pages = (total + limit - 1) // limit
    has_next = page < total_pages
    has_prev = page > 1

    return TrainingTemplateListResponse(
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


@router.get("/favorites", response_model=TrainingTemplateListResponse)
async def list_favorites(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db)
):
    """
    List only favorite training templates.

    Args:
        page: Page number (1-indexed)
        limit: Items per page
        db: Database session

    Returns:
        Paginated list of favorite training templates
    """
    skip = (page - 1) * limit

    templates, total = await TrainingTemplateService.get_favorites(
        db=db,
        skip=skip,
        limit=limit
    )

    total_pages = (total + limit - 1) // limit
    has_next = page < total_pages
    has_prev = page > 1

    return TrainingTemplateListResponse(
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


@router.get("/{template_id}", response_model=TrainingTemplateResponse)
async def get_template(
    template_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a training template by ID.

    Args:
        template_id: Template UUID
        db: Database session

    Returns:
        Training template details

    Raises:
        HTTPException: If template not found
    """
    template = await TrainingTemplateService.get_template(db, template_id)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Training template {template_id} not found"
        )

    return template


@router.patch("/{template_id}", response_model=TrainingTemplateResponse)
async def update_template(
    template_id: UUID,
    updates: TrainingTemplateUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a training template.

    Args:
        template_id: Template UUID
        updates: Update data
        db: Database session

    Returns:
        Updated training template

    Raises:
        HTTPException: If template not found
    """
    template = await TrainingTemplateService.update_template(db, template_id, updates)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Training template {template_id} not found"
        )

    return template


@router.delete("/{template_id}", status_code=204)
async def delete_template(
    template_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a training template.

    Args:
        template_id: Template UUID
        db: Database session

    Raises:
        HTTPException: If template not found
    """
    deleted = await TrainingTemplateService.delete_template(db, template_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Training template {template_id} not found"
        )


@router.post("/{template_id}/favorite", response_model=TrainingTemplateResponse)
async def toggle_favorite(
    template_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Toggle the favorite status of a training template.

    Args:
        template_id: Template UUID
        db: Database session

    Returns:
        Updated training template with toggled favorite status

    Raises:
        HTTPException: If template not found
    """
    template = await TrainingTemplateService.toggle_favorite(db, template_id)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Training template {template_id} not found"
        )

    return template


@router.post("/export")
async def export_templates(
    template_ids: Optional[List[UUID]] = Body(None, description="List of template IDs to export. If empty, exports all."),
    db: AsyncSession = Depends(get_db)
):
    """
    Export training templates to JSON format.

    Args:
        template_ids: Optional list of template IDs to export. If None, exports all templates.
        db: Database session

    Returns:
        JSON response containing exported templates with version info

    Raises:
        HTTPException: If export fails
    """
    try:
        export_data = await TrainingTemplateService.export_templates(db, template_ids)
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
    Import training templates from JSON format.

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
        result = await TrainingTemplateService.import_templates(
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
