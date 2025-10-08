"""
Dataset API endpoints.

This module contains all FastAPI routes for dataset management operations.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....models.dataset import DatasetStatus
from ....schemas.dataset import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetListResponse,
    DatasetDownloadRequest,
    DatasetTokenizeRequest,
)
from ....services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    dataset: DatasetCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new dataset.

    Args:
        dataset: Dataset creation data
        db: Database session

    Returns:
        Created dataset

    Raises:
        HTTPException: If dataset creation fails
    """
    try:
        db_dataset = await DatasetService.create_dataset(db, dataset)
        return db_dataset
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create dataset: {str(e)}"
        )


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query"),
    source: Optional[str] = Query(None, description="Filter by source"),
    status: Optional[DatasetStatus] = Query(None, description="Filter by status"),
    sort_by: str = Query("created_at", description="Sort by field"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    db: AsyncSession = Depends(get_db)
):
    """
    List datasets with filtering, pagination, and sorting.

    Args:
        page: Page number (1-indexed)
        limit: Items per page
        search: Search query for name or repo_id
        source: Filter by source type
        status: Filter by status
        sort_by: Column to sort by
        order: Sort order (asc or desc)
        db: Database session

    Returns:
        Paginated list of datasets with metadata
    """
    skip = (page - 1) * limit

    datasets, total = await DatasetService.list_datasets(
        db=db,
        skip=skip,
        limit=limit,
        search=search,
        source=source,
        status=status,
        sort_by=sort_by,
        order=order
    )

    total_pages = (total + limit - 1) // limit
    has_next = page < total_pages
    has_prev = page > 1

    return DatasetListResponse(
        data=datasets,
        pagination={
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev
        }
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a dataset by ID.

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Returns:
        Dataset details

    Raises:
        HTTPException: If dataset not found
    """
    dataset = await DatasetService.get_dataset(db, dataset_id)

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    return dataset


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: UUID,
    updates: DatasetUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a dataset.

    Args:
        dataset_id: Dataset UUID
        updates: Update data
        db: Database session

    Returns:
        Updated dataset

    Raises:
        HTTPException: If dataset not found
    """
    dataset = await DatasetService.update_dataset(db, dataset_id, updates)

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    return dataset


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a dataset.

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Raises:
        HTTPException: If dataset not found
    """
    deleted = await DatasetService.delete_dataset(db, dataset_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )


@router.post("/download", response_model=DatasetResponse, status_code=202)
async def download_dataset(
    request: DatasetDownloadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Download a dataset from HuggingFace.

    This endpoint creates a dataset record and queues the download.
    The actual download happens asynchronously.

    Args:
        request: Download request with repo_id and optional access_token
        db: Database session

    Returns:
        Created dataset with status 'downloading'

    Raises:
        HTTPException: If dataset already exists or download fails
    """
    # Check if dataset already exists
    existing = await DatasetService.get_dataset_by_repo_id(db, request.repo_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset {request.repo_id} already exists"
        )

    # Create dataset record
    dataset_create = DatasetCreate(
        name=request.repo_id.split("/")[-1],
        source="HuggingFace",
        hf_repo_id=request.repo_id,
        metadata={
            "split": request.split,
            "access_token_provided": bool(request.access_token)
        }
    )

    dataset = await DatasetService.create_dataset(db, dataset_create)

    # Queue download job with Celery
    from ....workers.dataset_tasks import download_dataset_task
    download_dataset_task.delay(
        dataset_id=str(dataset.id),
        repo_id=request.repo_id,
        access_token=request.access_token,
        split=request.split
    )

    return dataset


@router.post("/{dataset_id}/tokenize", response_model=DatasetResponse, status_code=202)
async def tokenize_dataset(
    dataset_id: UUID,
    request: DatasetTokenizeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Tokenize a dataset using a HuggingFace tokenizer.

    This endpoint queues the tokenization job to run asynchronously.
    The dataset status will be updated to 'processing' during tokenization.

    Args:
        dataset_id: Dataset UUID
        request: Tokenization request with tokenizer name and parameters
        db: Database session

    Returns:
        Dataset with status 'processing'

    Raises:
        HTTPException: If dataset not found or not ready for tokenization
    """
    # Get dataset
    dataset = await DatasetService.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Check if dataset is ready for tokenization
    if dataset.status != DatasetStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be in 'ready' status for tokenization (current: {dataset.status})"
        )

    if not dataset.raw_path:
        raise HTTPException(
            status_code=400,
            detail="Dataset has no raw_path - cannot tokenize"
        )

    # Update status to processing
    updates = DatasetUpdate(
        status=DatasetStatus.PROCESSING.value,
        progress=0.0,
    )
    dataset = await DatasetService.update_dataset(db, dataset_id, updates)

    # Queue tokenization job with Celery
    from ....workers.dataset_tasks import tokenize_dataset_task
    tokenize_dataset_task.delay(
        dataset_id=str(dataset_id),
        tokenizer_name=request.tokenizer_name,
        max_length=request.max_length,
        stride=request.stride,
    )

    return dataset
