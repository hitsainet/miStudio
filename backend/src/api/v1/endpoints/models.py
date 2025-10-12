"""
Model API endpoints.

This module contains all FastAPI routes for model management operations.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....models.model import ModelStatus, QuantizationFormat
from ....schemas.model import (
    ModelUpdate,
    ModelResponse,
    ModelListResponse,
    ModelDownloadRequest,
)
from ....services.model_service import ModelService
from ....workers.model_tasks import download_and_load_model
from ....core.celery_app import celery_app

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/download", response_model=ModelResponse, status_code=202)
async def download_model(
    request: ModelDownloadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Download a model from HuggingFace.

    This endpoint creates a model record and queues the download.
    The actual download happens asynchronously via Celery.

    Args:
        request: Download request with repo_id, quantization, and optional access_token
        db: Database session

    Returns:
        Created model with status 'downloading'

    Raises:
        HTTPException: If model with same name already exists or validation fails
    """
    # Extract model name from repo_id
    model_name = request.repo_id.split("/")[-1]

    # Check if model with this name already exists
    existing = await ModelService.get_model_by_name(db, model_name)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model_name}' already exists with ID {existing.id}"
        )

    try:
        # Initiate model download (creates DB record)
        model = await ModelService.initiate_model_download(db, request)

        # Queue download job with Celery
        download_and_load_model.delay(
            model_id=model.id,
            repo_id=request.repo_id,
            quantization=request.quantization.value,
            access_token=request.access_token
        )

        return model

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate model download: {str(e)}"
        )


@router.get("", response_model=ModelListResponse)
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query for name"),
    architecture: Optional[str] = Query(None, description="Filter by architecture"),
    quantization: Optional[QuantizationFormat] = Query(None, description="Filter by quantization"),
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
        search: Search query for name
        architecture: Filter by architecture type (llama, gpt2, phi, etc.)
        quantization: Filter by quantization format
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
        quantization=quantization,
        status=status,
        sort_by=sort_by,
        order=order
    )

    total_pages = (total + limit - 1) // limit if total > 0 else 0
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
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a model by ID.

    Args:
        model_id: Model ID (string format: m_{uuid})
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
            detail=f"Model '{model_id}' not found"
        )

    return model


@router.get("/{model_id}/architecture")
async def get_model_architecture(
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed architecture information for a model.

    Args:
        model_id: Model ID (string format: m_{uuid})
        db: Database session

    Returns:
        Detailed architecture configuration and metadata

    Raises:
        HTTPException: If model not found or not ready
    """
    arch_info = await ModelService.get_model_architecture_info(db, model_id)

    if not arch_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    # Check if model is ready (architecture info only available after loading)
    model = await ModelService.get_model(db, model_id)
    if model and model.status not in (ModelStatus.READY, ModelStatus.QUANTIZING):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not ready yet (status: {model.status.value})"
        )

    return arch_info


@router.patch("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    updates: ModelUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a model.

    Args:
        model_id: Model ID (string format: m_{uuid})
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
            detail=f"Model '{model_id}' not found"
        )

    return model


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a model.

    Args:
        model_id: Model ID (string format: m_{uuid})
        db: Database session

    Raises:
        HTTPException: If model not found
    """
    deleted = await ModelService.delete_model(db, model_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get Celery task status by task ID.

    This endpoint provides direct access to Celery task state for debugging.
    In most cases, clients should use GET /models/{model_id} instead,
    which provides model status with progress and error information.

    Args:
        task_id: Celery task ID (returned from POST /models/download)

    Returns:
        Task status information including state, result, and metadata
    """
    from celery.result import AsyncResult

    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "state": task_result.state,
        "ready": task_result.ready(),
        "successful": task_result.successful() if task_result.ready() else None,
        "failed": task_result.failed() if task_result.ready() else None,
    }

    # Add result if task is ready
    if task_result.ready():
        if task_result.successful():
            response["result"] = task_result.result
        elif task_result.failed():
            response["error"] = str(task_result.info)

    # Add task info if available
    if task_result.info:
        response["info"] = task_result.info

    return response
