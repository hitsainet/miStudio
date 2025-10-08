"""
Model API endpoints.

This module contains all FastAPI routes for model management operations.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....models.model import ModelStatus
from ....schemas.model import (
    ModelCreate,
    ModelUpdate,
    ModelResponse,
    ModelListResponse,
    ModelDownloadRequest,
)
from ....services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=ModelResponse, status_code=201)
async def create_model(
    model: ModelCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new model.

    Args:
        model: Model creation data
        db: Database session

    Returns:
        Created model

    Raises:
        HTTPException: If model creation fails
    """
    try:
        db_model = await ModelService.create_model(db, model)
        return db_model
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create model: {str(e)}"
        )


@router.get("", response_model=ModelListResponse)
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query"),
    architecture: Optional[str] = Query(None, description="Filter by architecture"),
    status: Optional[ModelStatus] = Query(None, description="Filter by status"),
    sort_by: str = Query("created_at", description="Sort by field"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    db: AsyncSession = Depends(get_db)
):
    """
    List models with filtering, pagination, and sorting.

    Args:
        page: Page number (1-indexed)
        limit: Items per page
        search: Search query for name or repo_id
        architecture: Filter by architecture type
        status: Filter by status
        sort_by: Column to sort by
        order: Sort order (asc or desc)
        db: Database session

    Returns:
        Paginated list of models with metadata
    """
    skip = (page - 1) * limit

    models, total = await ModelService.list_models(
        db=db,
        skip=skip,
        limit=limit,
        search=search,
        architecture=architecture,
        status=status,
        sort_by=sort_by,
        order=order
    )

    total_pages = (total + limit - 1) // limit
    has_next = page < total_pages
    has_prev = page > 1

    return ModelListResponse(
        data=models,
        pagination={
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev
        }
    )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a model by ID.

    Args:
        model_id: Model UUID
        db: Database session

    Returns:
        Model details

    Raises:
        HTTPException: If model not found
    """
    model = await ModelService.get_model(db, model_id)

    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found"
        )

    return model


@router.patch("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: UUID,
    updates: ModelUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a model.

    Args:
        model_id: Model UUID
        updates: Update data
        db: Database session

    Returns:
        Updated model

    Raises:
        HTTPException: If model not found
    """
    model = await ModelService.update_model(db, model_id, updates)

    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found"
        )

    return model


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a model.

    Args:
        model_id: Model UUID
        db: Database session

    Raises:
        HTTPException: If model not found
    """
    deleted = await ModelService.delete_model(db, model_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found"
        )


@router.post("/download", response_model=ModelResponse, status_code=202)
async def download_model(
    request: ModelDownloadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Download a model from HuggingFace.

    This endpoint creates a model record and queues the download.
    The actual download happens asynchronously.

    Args:
        request: Download request with repo_id, quantization, and optional access_token
        db: Database session

    Returns:
        Created model with status 'downloading'

    Raises:
        HTTPException: If model already exists or download fails
    """
    # Check if model already exists
    existing = await ModelService.get_model_by_repo_id(db, request.repo_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Model {request.repo_id} already exists"
        )

    # Create model record
    model_create = ModelCreate(
        name=request.repo_id.split("/")[-1],
        repo_id=request.repo_id,
        quantization=request.quantization,
        architecture="auto",  # Will be determined during download
        metadata={
            "access_token_provided": bool(request.access_token)
        }
    )

    model = await ModelService.create_model(db, model_create)

    # TODO: Queue download job with Celery
    # download_model_task.delay(
    #     model_id=str(model.id),
    #     repo_id=request.repo_id,
    #     quantization=request.quantization,
    #     access_token=request.access_token
    # )

    return model
