"""
Dataset API endpoints.

This module contains all FastAPI routes for dataset management operations.
"""

import logging
from functools import lru_cache
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import AutoTokenizer

from ....core.deps import get_db
from ....core.celery_app import celery_app
from ....models.dataset import DatasetStatus
from ....schemas.dataset import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetListResponse,
    DatasetDownloadRequest,
    DatasetTokenizeRequest,
    TokenizePreviewRequest,
    TokenizePreviewResponse,
    TokenInfo,
)
from ....services.dataset_service import DatasetService

logger = logging.getLogger(__name__)

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


@router.get("/{dataset_id}/task-status")
async def get_dataset_task_status(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get Celery task status for a dataset operation (download/tokenization).

    This endpoint queries the Celery task state from Redis to provide
    real-time progress updates for long-running operations.

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Returns:
        Task status with progress information

    Raises:
        HTTPException: If dataset not found
    """
    # Get dataset to retrieve task_id
    dataset = await DatasetService.get_dataset(db, dataset_id)

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Check if dataset has an active task
    # Handle metadata as dict (already parsed by SQLAlchemy JSONB type)
    metadata = dataset.extra_metadata if isinstance(dataset.extra_metadata, dict) else {}
    task_id = metadata.get('task_id') if metadata else None

    if not task_id:
        # No active task - return status based on dataset state
        if dataset.status == DatasetStatus.READY:
            return {
                "state": "SUCCESS",
                "progress": 100.0,
                "status": "Complete",
                "task_id": None
            }
        elif dataset.status == DatasetStatus.ERROR:
            return {
                "state": "FAILURE",
                "progress": 0.0,
                "status": dataset.error_message or "Error occurred",
                "task_id": None
            }
        else:
            # Use database progress as fallback
            return {
                "state": "PROGRESS",
                "progress": dataset.progress or 0.0,
                "status": f"Status: {dataset.status}",
                "task_id": None
            }

    # Query Celery for task status
    task = celery_app.AsyncResult(task_id)

    if task.state == 'PENDING':
        # Task is waiting to start
        response = {
            "state": task.state,
            "progress": 0.0,
            "status": "Waiting to start...",
            "task_id": task_id
        }
    elif task.state == 'PROGRESS':
        # Task is running with progress updates
        info = task.info or {}
        response = {
            "state": task.state,
            "progress": info.get('percent', 0.0),
            "status": info.get('status', 'Processing...'),
            "current": info.get('current', 0),
            "total": info.get('total', 100),
            "task_id": task_id
        }
    elif task.state == 'SUCCESS':
        # Task completed successfully
        info = task.info or {}
        response = {
            "state": task.state,
            "progress": 100.0,
            "status": info.get('status', 'Complete'),
            "task_id": task_id
        }
    elif task.state == 'FAILURE':
        # Task failed
        response = {
            "state": task.state,
            "progress": 0.0,
            "status": f"Error: {str(task.info)}",
            "task_id": task_id
        }
    else:
        # Unknown state
        response = {
            "state": task.state,
            "progress": dataset.progress or 0.0,
            "status": f"State: {task.state}",
            "task_id": task_id
        }

    return response


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

    This endpoint deletes the database record and queues background file cleanup.

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Raises:
        HTTPException: If dataset not found
    """
    from ....workers.dataset_tasks import delete_dataset_files

    # Delete dataset record and get file paths
    deletion_info = await DatasetService.delete_dataset(db, dataset_id)

    if not deletion_info:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Queue background file cleanup if there are files to delete
    raw_path = deletion_info.get("raw_path")
    tokenized_path = deletion_info.get("tokenized_path")

    if raw_path or tokenized_path:
        logger.info(
            f"Queuing file cleanup for dataset {dataset_id} "
            f"(raw_path={raw_path}, tokenized_path={tokenized_path})"
        )
        delete_dataset_files.delay(
            dataset_id=str(dataset_id),
            raw_path=raw_path,
            tokenized_path=tokenized_path
        )
    else:
        logger.info(f"No files to clean up for dataset {dataset_id}")


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
            "config": request.config,
            "access_token_provided": bool(request.access_token)
        }
    )

    dataset = await DatasetService.create_dataset(db, dataset_create)

    # Queue download job with Celery
    from ....workers.dataset_tasks import download_dataset_task
    task = download_dataset_task.delay(
        dataset_id=str(dataset.id),
        repo_id=request.repo_id,
        access_token=request.access_token,
        split=request.split,
        config=request.config
    )

    # Store task_id in dataset metadata for progress tracking
    metadata = dataset.extra_metadata or {}
    metadata['task_id'] = task.id
    metadata['task_type'] = 'download'
    updates = DatasetUpdate(metadata=metadata)
    dataset = await DatasetService.update_dataset(db, dataset.id, updates)

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
    # Prevent duplicate requests while processing
    if dataset.status == DatasetStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="Dataset is already being processed. Please wait for the current operation to complete."
        )

    # Allow READY or ERROR status (ERROR allows retry after failure)
    if dataset.status not in [DatasetStatus.READY, DatasetStatus.ERROR]:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be in 'ready' or 'error' status for tokenization (current: {dataset.status})"
        )

    if not dataset.raw_path:
        raise HTTPException(
            status_code=400,
            detail="Dataset has no raw_path - cannot tokenize"
        )

    # Auto-clear existing tokenization before starting new job
    # This handles both ERROR status (failed tokenization) and already-tokenized datasets
    if dataset.status == DatasetStatus.ERROR or dataset.tokenized_path:
        logger.info(f"Auto-clearing existing tokenization for dataset {dataset_id} before re-tokenization")
        dataset = await DatasetService.clear_tokenization(db, dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear existing tokenization"
            )

    # Update status to processing
    updates = DatasetUpdate(
        status=DatasetStatus.PROCESSING.value,
        progress=0.0,
    )
    dataset = await DatasetService.update_dataset(db, dataset_id, updates)

    # Queue tokenization job with Celery
    from ....workers.dataset_tasks import tokenize_dataset_task
    task = tokenize_dataset_task.delay(
        dataset_id=str(dataset_id),
        tokenizer_name=request.tokenizer_name,
        max_length=request.max_length,
        stride=request.stride,
        padding=request.padding,
        truncation=request.truncation,
        add_special_tokens=request.add_special_tokens,
        return_attention_mask=request.return_attention_mask,
    )

    # Store task_id in dataset metadata for progress tracking
    metadata = dataset.extra_metadata or {}
    metadata['task_id'] = task.id
    metadata['task_type'] = 'tokenization'
    updates = DatasetUpdate(metadata=metadata)
    dataset = await DatasetService.update_dataset(db, dataset_id, updates)

    return dataset


@router.delete("/{dataset_id}/tokenization", response_model=DatasetResponse)
async def clear_tokenization(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Clear tokenization data from a dataset.

    This endpoint removes all tokenization-related data (tokenized files, metadata)
    while keeping the raw dataset intact. Resets the dataset to READY status.

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Returns:
        Updated dataset with tokenization cleared

    Raises:
        HTTPException: If dataset not found or cannot be cleared
    """
    # Get dataset
    dataset = await DatasetService.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Check if dataset is currently processing
    if dataset.status == DatasetStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="Cannot clear tokenization while dataset is processing. Cancel the job first."
        )

    # Clear tokenization
    try:
        dataset = await DatasetService.clear_tokenization(db, dataset_id)
        return dataset
    except Exception as e:
        logger.error(f"Failed to clear tokenization for dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear tokenization: {str(e)}"
        )


@router.delete("/{dataset_id}/cancel", status_code=200)
async def cancel_dataset_download(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an in-progress dataset download or tokenization.

    This endpoint cancels the processing task, cleans up partial files,
    and updates the dataset status to ERROR with "Cancelled by user".

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Returns:
        Cancellation status

    Raises:
        HTTPException: If dataset not found or not in cancellable state
    """
    from ....workers.dataset_tasks import cancel_dataset_download as cancel_task

    # Verify dataset exists
    dataset = await DatasetService.get_dataset(db, dataset_id)

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Check if dataset is in a cancellable state
    if dataset.status not in [DatasetStatus.DOWNLOADING, DatasetStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {dataset_id} cannot be cancelled (status: {dataset.status.value})"
        )

    try:
        # Call cancel_dataset_download task (runs synchronously for immediate response)
        # Note: We don't have task_id stored, so we can't revoke the specific task
        # Instead, the cancel task will clean up files and update database
        result = cancel_task(dataset_id=str(dataset_id))

        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        return {
            "dataset_id": str(dataset_id),
            "status": "cancelled",
            "message": "Download/processing cancelled successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel download/processing: {str(e)}"
        )


@router.get("/{dataset_id}/samples")
async def get_dataset_samples(
    dataset_id: UUID,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated samples from a dataset.

    Args:
        dataset_id: Dataset UUID
        page: Page number (1-indexed)
        limit: Items per page
        db: Database session

    Returns:
        Paginated list of dataset samples

    Raises:
        HTTPException: If dataset not found or not ready
    """
    from datasets import load_from_disk
    from pathlib import Path

    # Get dataset
    dataset = await DatasetService.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Check if dataset has raw data available (allow viewing during processing)
    if dataset.status not in [DatasetStatus.READY, DatasetStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be in 'ready' or 'processing' status to view samples (current: {dataset.status})"
        )

    if not dataset.raw_path:
        raise HTTPException(
            status_code=400,
            detail="Dataset has no raw_path"
        )

    try:
        # Load dataset from disk (Arrow format saved by download task)
        hf_dataset = load_from_disk(dataset.raw_path)

        # Handle DatasetDict (multi-split datasets) - use 'train' split by default
        from datasets import DatasetDict
        if isinstance(hf_dataset, DatasetDict):
            # Try to get train split, or first available split
            if 'train' in hf_dataset:
                hf_dataset = hf_dataset['train']
            else:
                # Get first available split
                first_split = next(iter(hf_dataset.keys()))
                hf_dataset = hf_dataset[first_split]

        # Calculate pagination
        total_samples = len(hf_dataset)
        start_idx = (page - 1) * limit
        end_idx = min(start_idx + limit, total_samples)

        # Get samples for current page
        samples = []
        for idx in range(start_idx, end_idx):
            sample = hf_dataset[idx]
            # Convert sample to dict if it's not already
            if not isinstance(sample, dict):
                sample = {"data": str(sample)}
            samples.append({
                "index": idx,
                "data": sample
            })

        # Calculate pagination metadata
        total_pages = (total_samples + limit - 1) // limit
        has_next = page < total_pages
        has_prev = page > 1

        return {
            "data": samples,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_samples,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load samples: {str(e)}"
        )


@lru_cache(maxsize=10)
def load_tokenizer_cached(tokenizer_name: str):
    """
    Load and cache a HuggingFace tokenizer.

    Uses LRU cache to keep the last 10 tokenizers in memory for fast previews.

    Args:
        tokenizer_name: HuggingFace tokenizer name

    Returns:
        Loaded tokenizer

    Raises:
        Exception: If tokenizer loading fails
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


@router.post("/tokenize-preview", response_model=TokenizePreviewResponse)
async def tokenize_preview(
    request: TokenizePreviewRequest
):
    """
    Preview tokenization on a small text sample.

    This endpoint provides fast tokenization preview without modifying any datasets.
    Tokenizers are cached for performance.

    Args:
        request: Tokenization preview request

    Returns:
        Tokenization result with token details

    Raises:
        HTTPException: If tokenization fails or tokenizer is invalid
    """
    try:
        # Load tokenizer (cached for performance)
        tokenizer = load_tokenizer_cached(request.tokenizer_name)

        # Set pad token if not already set (e.g., GPT-2 doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Map truncation strategy to tokenizer parameter
        truncation_config = {
            "longest_first": True,
            "only_first": "only_first",
            "only_second": "only_second",
            "do_not_truncate": False,
        }
        truncation_param = truncation_config.get(request.truncation, True)

        # Map padding strategy to tokenizer parameter
        padding_config = {
            "max_length": "max_length",
            "longest": "longest",
            "do_not_pad": False,
        }
        padding_param = padding_config.get(request.padding, "max_length")

        # Tokenize the text
        encoded = tokenizer(
            request.text,
            max_length=request.max_length,
            truncation=truncation_param,
            padding=padding_param,
            add_special_tokens=request.add_special_tokens,
            return_attention_mask=request.return_attention_mask,
        )

        # Get token IDs and convert to tokens
        input_ids = encoded["input_ids"]
        tokens_list = tokenizer.convert_ids_to_tokens(input_ids)

        # Identify special tokens
        special_token_ids = set()
        if hasattr(tokenizer, "all_special_ids"):
            special_token_ids = set(tokenizer.all_special_ids)

        # Build token info list
        token_infos = []
        special_count = 0
        for position, (token_id, token_text) in enumerate(zip(input_ids, tokens_list)):
            is_special = token_id in special_token_ids
            if is_special:
                special_count += 1

            token_infos.append(TokenInfo(
                id=token_id,
                text=token_text,
                type="special" if is_special else "regular",
                position=position
            ))

        # Build response
        response = TokenizePreviewResponse(
            tokens=token_infos,
            attention_mask=encoded.get("attention_mask") if request.return_attention_mask else None,
            token_count=len(input_ids),
            sequence_length=len(input_ids),
            special_token_count=special_count
        )

        return response

    except Exception as e:
        # Check if it's a tokenizer loading error
        error_msg = str(e).lower()
        if any(phrase in error_msg for phrase in [
            "not found",
            "does not exist",
            "not a valid model identifier",
            "is not a local folder"
        ]):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tokenizer name: {request.tokenizer_name}"
            )

        # General error
        raise HTTPException(
            status_code=500,
            detail=f"Tokenization failed: {str(e)}"
        )
