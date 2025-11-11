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
import redis

from ....core.deps import get_db
from ....core.celery_app import celery_app
from ....core.config import settings
from ....models.dataset import DatasetStatus
from ....models.dataset_tokenization import DatasetTokenization, TokenizationStatus
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
    DatasetTokenizationResponse,
    DatasetTokenizationListResponse,
)
from ....services.dataset_service import DatasetService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Initialize Redis client for distributed locking
_redis_client = None

def get_redis_client() -> redis.Redis:
    """Get Redis client for distributed locking."""
    global _redis_client
    if _redis_client is None:
        redis_url = str(settings.redis_url)
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client


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
    logger.info(f"=== START TOKENIZATION REQUEST for dataset {dataset_id} ===")

    # Get dataset
    try:
        logger.info(f"Fetching dataset {dataset_id} from database")
        dataset = await DatasetService.get_dataset(db, dataset_id)
        logger.info(f"Dataset fetched successfully: {dataset.id if dataset else 'None'}")

        if not dataset:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {dataset_id} not found"
            )

        logger.info(f"Dataset status: {dataset.status}, has tokenizations: {hasattr(dataset, 'tokenizations')}")
        if hasattr(dataset, 'tokenizations'):
            logger.info(f"Tokenizations count: {len(dataset.tokenizations) if dataset.tokenizations else 0}")
    except AttributeError as e:
        logger.error(f"AttributeError while fetching dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch dataset: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error while fetching dataset: {e}", exc_info=True)
        raise

    # Distributed lock: Prevent concurrent tokenization requests
    # This prevents race conditions where two requests arrive simultaneously
    redis_client = get_redis_client()
    lock_key = f"tokenization_lock:{dataset_id}"
    lock_timeout = 3600  # 1 hour - max expected tokenization time

    # Try to acquire lock (SET if Not eXists with EXpiration)
    if not redis_client.set(lock_key, "locked", nx=True, ex=lock_timeout):
        raise HTTPException(
            status_code=409,
            detail="Tokenization already in progress for this dataset. Please wait for the current operation to complete."
        )

    try:
        # Check if dataset is ready for tokenization
        # Prevent duplicate requests while processing
        if dataset.status == DatasetStatus.PROCESSING:
            redis_client.delete(lock_key)  # Release lock
            raise HTTPException(
                status_code=409,
                detail="Dataset is already being processed. Please wait for the current operation to complete."
            )

        # Allow READY or ERROR status (ERROR allows retry after failure)
        if dataset.status not in [DatasetStatus.READY, DatasetStatus.ERROR]:
            redis_client.delete(lock_key)  # Release lock
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must be in 'ready' or 'error' status for tokenization (current: {dataset.status})"
            )

        if not dataset.raw_path:
            redis_client.delete(lock_key)  # Release lock
            raise HTTPException(
                status_code=400,
                detail="Dataset has no raw_path - cannot tokenize"
            )

        # Auto-clear existing tokenization before starting new job
        # This handles both ERROR status (failed tokenization) and already-tokenized datasets
        # Check if any tokenization exists by querying the relationship
        try:
            logger.info(f"Checking for existing tokenizations on dataset {dataset_id}")
            has_tokenization = dataset.tokenizations and len(dataset.tokenizations) > 0
            logger.info(f"Dataset {dataset_id} has_tokenization={has_tokenization}, status={dataset.status}")

            if dataset.status == DatasetStatus.ERROR or has_tokenization:
                logger.info(f"Auto-clearing existing tokenization for dataset {dataset_id} before re-tokenization")
                dataset = await DatasetService.clear_tokenization(db, dataset_id)
                if not dataset:
                    redis_client.delete(lock_key)  # Release lock
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to clear existing tokenization"
                    )
        except AttributeError as e:
            logger.error(f"AttributeError in tokenization check: {e}", exc_info=True)
            redis_client.delete(lock_key)  # Release lock
            raise HTTPException(
                status_code=500,
                detail=f"Failed to check tokenization status: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in tokenization check: {e}", exc_info=True)
            redis_client.delete(lock_key)  # Release lock
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start tokenization: {str(e)}"
            )

        # Update status to processing and save filter configuration
        updates = DatasetUpdate(
            status=DatasetStatus.PROCESSING.value,
            progress=0.0,
            tokenization_filter_enabled=request.tokenization_filter_enabled,
            tokenization_filter_mode=request.tokenization_filter_mode,
            tokenization_junk_ratio_threshold=request.tokenization_junk_ratio_threshold,
        )
        dataset = await DatasetService.update_dataset(db, dataset_id, updates)

        # Queue tokenization job with Celery
        from ....workers.dataset_tasks import tokenize_dataset_task
        task = tokenize_dataset_task.delay(
            dataset_id=str(dataset_id),
            model_id=request.model_id,
            max_length=request.max_length,
            stride=request.stride,
            padding=request.padding,
            truncation=request.truncation,
            add_special_tokens=request.add_special_tokens,
            return_attention_mask=request.return_attention_mask,
            enable_cleaning=request.enable_cleaning,
        )

        # Store task_id in dataset metadata for progress tracking
        metadata = dataset.extra_metadata or {}
        metadata['task_id'] = task.id
        metadata['task_type'] = 'tokenization'
        metadata['lock_key'] = lock_key  # Store lock key for cleanup
        updates = DatasetUpdate(metadata=metadata)
        dataset = await DatasetService.update_dataset(db, dataset_id, updates)

        # Lock will be released by the Celery task on completion/failure
        # Don't delete lock_key here - let the task do it

        return dataset

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        # Release lock on unexpected errors
        redis_client.delete(lock_key)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start tokenization: {str(e)}"
        )


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
    from sqlalchemy import select
    from ....models.dataset import Dataset as DatasetModel

    # Get dataset with metadata (direct query to ensure metadata is loaded)
    result = await db.execute(
        select(DatasetModel).where(DatasetModel.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()

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

    try:
        # Try to load from raw path first (preferred for viewing text)
        dataset_path = None
        is_tokenized = False

        if dataset.raw_path and Path(dataset.raw_path).exists():
            dataset_path = dataset.raw_path
            is_tokenized = False
        else:
            # Try HuggingFace cache format path (repo_id slash becomes triple underscore)
            # Example: vietgpt_openwebtext_en → vietgpt___openwebtext_en
            if dataset.raw_path:
                raw_path_obj = Path(dataset.raw_path)
                hf_cache_name = raw_path_obj.name.replace('_', '___', 1)
                hf_cache_path = raw_path_obj.parent / hf_cache_name

                if hf_cache_path.exists():
                    dataset_path = str(hf_cache_path)
                    is_tokenized = False

        # If still not found, try tokenized path from tokenizations relationship
        if not dataset_path and dataset.tokenizations and len(dataset.tokenizations) > 0:
            # Fallback to tokenized dataset if raw was cleaned up
            # Use the first tokenization's path (most common case is one tokenization per dataset)
            tokenization = dataset.tokenizations[0]
            if tokenization.tokenized_path and Path(tokenization.tokenized_path).exists():
                dataset_path = tokenization.tokenized_path
                is_tokenized = True

        # Try to load dataset from local path first
        hf_dataset = None
        if dataset_path:
            try:
                # Load dataset from disk (Arrow format)
                hf_dataset = load_from_disk(dataset_path)
                logger.info(f"Loaded dataset from disk: {dataset_path}")
            except Exception as e:
                logger.warning(f"Failed to load from disk path {dataset_path}: {e}")
                # Fall through to HuggingFace fallback

        # If loading from disk failed or no path, try loading from HuggingFace
        if not hf_dataset and dataset.hf_repo_id:
            from datasets import load_dataset as hf_load_dataset
            try:
                # Extract split from metadata (safely handle dict or SQLAlchemy type)
                split_name = None
                if dataset.metadata:
                    if isinstance(dataset.metadata, dict):
                        split_name = dataset.metadata.get('split')
                    elif hasattr(dataset.metadata, 'get'):
                        split_name = dataset.metadata.get('split')

                # Load from HuggingFace (will use cached files if available)
                hf_dataset = hf_load_dataset(
                    dataset.hf_repo_id,
                    split=split_name,
                    trust_remote_code=True
                )
                logger.info(f"Loaded dataset {dataset.hf_repo_id} from HuggingFace cache (split={split_name})")
            except Exception as e:
                logger.error(f"Failed to load from HuggingFace: {e}", exc_info=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load dataset from HuggingFace: {str(e) or type(e).__name__}"
                )

        if not hf_dataset:
            raise HTTPException(
                status_code=400,
                detail="Dataset files not found. Raw dataset may have been cleaned up and tokenized dataset is not available."
            )

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

        # If using tokenized dataset, load tokenizer for decoding
        tokenizer = None
        if is_tokenized:
            try:
                from transformers import AutoTokenizer
                # Get tokenizer name from metadata (handle both dict and SQLAlchemy types)
                metadata_dict = dataset.metadata if isinstance(dataset.metadata, dict) else {}
                tokenization_meta = metadata_dict.get('tokenization', {}) if metadata_dict else {}
                tokenizer_name = tokenization_meta.get('tokenizer_name') if tokenization_meta else None

                if tokenizer_name:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    print(f"Loaded tokenizer '{tokenizer_name}' for decoding tokenized samples")
            except Exception as e:
                # If tokenizer fails to load, we'll show raw token IDs
                print(f"Warning: Failed to load tokenizer for decoding: {e}")

        for idx in range(start_idx, end_idx):
            sample = hf_dataset[idx]
            # Convert sample to dict if it's not already
            if not isinstance(sample, dict):
                sample = {"data": str(sample)}

            # If this is a tokenized dataset, decode input_ids to text
            if is_tokenized and tokenizer and 'input_ids' in sample:
                try:
                    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                    # Replace input_ids with decoded text for display
                    sample = {
                        "text": decoded_text,
                        "_metadata": {
                            "source": "tokenized_dataset_decoded",
                            "num_tokens": len(sample['input_ids']),
                            "has_attention_mask": 'attention_mask' in sample
                        }
                    }
                except Exception as e:
                    # If decoding fails, show truncated token IDs
                    sample = {
                        "input_ids": sample['input_ids'][:50],  # Show first 50 tokens
                        "_metadata": {
                            "source": "tokenized_dataset_raw",
                            "error": f"Failed to decode: {str(e)}",
                            "total_tokens": len(sample['input_ids'])
                        }
                    }

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
        logger.error(f"Samples endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load samples: {str(e) or type(e).__name__} - {repr(e)}"
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


@router.get("/{dataset_id}/tokenizations", response_model=DatasetTokenizationListResponse)
async def list_dataset_tokenizations(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    List all tokenizations for a dataset.

    This endpoint returns all tokenization records for a specific dataset,
    showing which models have been used to tokenize it and their status.

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Returns:
        List of tokenizations with their status and statistics

    Raises:
        HTTPException: If dataset not found
    """
    # Verify dataset exists
    dataset = await DatasetService.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

    # Get all tokenizations for this dataset
    from sqlalchemy import select
    from ....models.dataset_tokenization import DatasetTokenization

    result = await db.execute(
        select(DatasetTokenization)
        .where(DatasetTokenization.dataset_id == dataset_id)
        .order_by(DatasetTokenization.created_at.desc())
    )
    tokenizations = result.scalars().all()

    return DatasetTokenizationListResponse(
        data=tokenizations,
        total=len(tokenizations)
    )


@router.get("/{dataset_id}/tokenizations/{model_id}", response_model=DatasetTokenizationResponse)
async def get_dataset_tokenization(
    dataset_id: UUID,
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific tokenization for a dataset and model.

    This endpoint returns detailed information about a specific tokenization,
    including its path, statistics, and current status.

    Args:
        dataset_id: Dataset UUID
        model_id: Model ID
        db: Database session

    Returns:
        Tokenization details

    Raises:
        HTTPException: If tokenization not found
    """
    # Query tokenization
    from sqlalchemy import select
    from ....models.dataset_tokenization import DatasetTokenization

    result = await db.execute(
        select(DatasetTokenization)
        .where(
            DatasetTokenization.dataset_id == dataset_id,
            DatasetTokenization.model_id == model_id
        )
    )
    tokenization = result.scalar_one_or_none()

    if not tokenization:
        raise HTTPException(
            status_code=404,
            detail=f"No tokenization found for dataset {dataset_id} with model {model_id}"
        )

    return tokenization


@router.delete("/{dataset_id}/tokenizations/{model_id}", status_code=204)
async def delete_dataset_tokenization(
    dataset_id: UUID,
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a specific tokenization.

    This endpoint removes a tokenization record and queues deletion of
    the associated tokenized files. The raw dataset remains intact.

    Args:
        dataset_id: Dataset UUID
        model_id: Model ID
        db: Database session

    Raises:
        HTTPException: If tokenization not found or currently processing
    """
    from pathlib import Path
    from sqlalchemy import select
    from ....models.dataset_tokenization import DatasetTokenization

    # Query tokenization
    result = await db.execute(
        select(DatasetTokenization)
        .where(
            DatasetTokenization.dataset_id == dataset_id,
            DatasetTokenization.model_id == model_id
        )
    )
    tokenization = result.scalar_one_or_none()

    if not tokenization:
        raise HTTPException(
            status_code=404,
            detail=f"No tokenization found for dataset {dataset_id} with model {model_id}"
        )

    # Prevent deletion of tokenization currently being processed
    if tokenization.status == TokenizationStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete tokenization while it is being processed. Please wait for completion or cancel the job first."
        )

    # Store tokenized_path before deletion
    tokenized_path = tokenization.tokenized_path

    # Delete tokenization record
    await db.delete(tokenization)
    await db.commit()

    # Queue background file cleanup if there are files to delete
    if tokenized_path and Path(tokenized_path).exists():
        from ....workers.dataset_tasks import delete_dataset_files
        logger.info(f"Queuing file cleanup for tokenization {tokenization.id} (path={tokenized_path})")
        delete_dataset_files.delay(
            dataset_id=str(dataset_id),
            raw_path=None,  # Don't delete raw files
            tokenized_path=tokenized_path
        )
    else:
        logger.info(f"No files to clean up for tokenization {tokenization.id}")

    return None


@router.post("/{dataset_id}/tokenizations/{model_id}/cancel", status_code=200)
async def cancel_dataset_tokenization(
    dataset_id: UUID,
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an in-progress tokenization job.

    This endpoint:
    1. Revokes the Celery task
    2. Updates tokenization status to ERROR with "Cancelled by user"
    3. Cleans up partial tokenization files

    Args:
        dataset_id: Dataset UUID
        model_id: Model ID
        db: Database session

    Returns:
        dict: Cancellation status

    Raises:
        HTTPException: If tokenization not found or not cancellable
    """
    from pathlib import Path
    from sqlalchemy import select
    from ....models.dataset_tokenization import DatasetTokenization, TokenizationStatus
    from celery import current_app

    # Query tokenization
    result = await db.execute(
        select(DatasetTokenization)
        .where(
            DatasetTokenization.dataset_id == dataset_id,
            DatasetTokenization.model_id == model_id
        )
    )
    tokenization = result.scalar_one_or_none()

    if not tokenization:
        raise HTTPException(
            status_code=404,
            detail=f"No tokenization found for dataset {dataset_id} with model {model_id}"
        )

    # Check if tokenization is in a cancellable state
    if tokenization.status not in [TokenizationStatus.PROCESSING, TokenizationStatus.QUEUED]:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel tokenization with status '{tokenization.status.value}'. Only PROCESSING or QUEUED jobs can be cancelled."
        )

    # Revoke the Celery task if task_id exists
    if tokenization.celery_task_id:
        try:
            current_app.control.revoke(tokenization.celery_task_id, terminate=True, signal='SIGKILL')
            logger.info(f"Revoked Celery task {tokenization.celery_task_id} for tokenization {tokenization.id}")
        except Exception as e:
            logger.warning(f"Failed to revoke Celery task {tokenization.celery_task_id}: {e}")

    # Update tokenization status to ERROR
    tokenization.status = TokenizationStatus.ERROR
    tokenization.error_message = "Cancelled by user"
    tokenization.progress = None
    await db.commit()

    # Clean up partial tokenization files if they exist
    if tokenization.tokenized_path:
        tokenized_path = Path(tokenization.tokenized_path)
        if tokenized_path.exists():
            from ....workers.dataset_tasks import delete_dataset_files
            logger.info(f"Queuing cleanup of partial tokenization files: {tokenized_path}")
            delete_dataset_files.delay(
                dataset_id=str(dataset_id),
                raw_path=None,
                tokenized_path=str(tokenized_path)
            )

    logger.info(f"Cancelled tokenization {tokenization.id} for dataset {dataset_id}")

    return {
        "dataset_id": str(dataset_id),
        "model_id": model_id,
        "tokenization_id": tokenization.id,
        "status": "cancelled",
        "message": "Tokenization cancelled successfully"
    }
