"""
Model API endpoints.

This module contains all FastAPI routes for model management operations.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from ....core.deps import get_db
from ....models.model import ModelStatus, QuantizationFormat
from ....models.activation_extraction import ActivationExtraction, ExtractionStatus
from ....schemas.model import (
    ModelUpdate,
    ModelResponse,
    ModelListResponse,
    ModelDownloadRequest,
    ModelRedownloadRequest,
    ModelRedownloadResponse,
    ActivationExtractionRequest,
    ExtractionCancelResponse,
    ExtractionRetryRequest,
    ExtractionRetryResponse,
    ExtractionDeleteRequest,
    ExtractionDeleteResponse,
)
from ....services.model_service import ModelService
from ....services.extraction_db_service import ExtractionDatabaseService
from ....core.database import get_sync_db
from ....workers.model_tasks import download_and_load_model, extract_activations, delete_model_files
from ....core.celery_app import celery_app
from sqlalchemy import select, exists

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
            access_token=request.access_token,
            trust_remote_code=request.trust_remote_code
        )

        return model

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate model download: {str(e)}"
        )


@router.post("/{model_id}/redownload", response_model=ModelRedownloadResponse, status_code=202)
async def redownload_model(
    model_id: str,
    request: ModelRedownloadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Re-download a model with different quantization.

    This endpoint allows changing a model's quantization without losing references
    from trainings and tokenizations. It:
    1. Deletes existing model files from disk
    2. Updates the model record with new quantization and DOWNLOADING status
    3. Queues a new download job with the requested quantization

    Args:
        model_id: Model ID to re-download
        request: Re-download request with new quantization
        db: Database session

    Returns:
        Re-download confirmation with old/new quantization

    Raises:
        HTTPException: If model not found, not ready, or already downloading
    """
    import shutil
    from pathlib import Path

    # Get model
    model = await ModelService.get_model(db, model_id)

    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    # Check if model is in a re-downloadable state (must be READY or ERROR)
    if model.status not in [ModelStatus.READY, ModelStatus.ERROR]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' cannot be re-downloaded (status: {model.status.value}). "
                   f"Only models with status READY or ERROR can be re-downloaded."
        )

    # Must have repo_id to re-download
    if not model.repo_id:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' cannot be re-downloaded: no repo_id stored"
        )

    old_quantization = model.quantization.value
    new_quantization = request.quantization.value

    try:
        # Delete existing model files
        if model.file_path:
            file_path = Path(model.file_path)
            if file_path.exists():
                logger.info(f"Deleting existing model files: {file_path}")
                shutil.rmtree(file_path)

        if model.quantized_path:
            quantized_path = Path(model.quantized_path)
            if quantized_path.exists():
                logger.info(f"Deleting existing quantized files: {quantized_path}")
                shutil.rmtree(quantized_path)

        # Update model record
        from ....models.model import Model

        # Direct database update for status and quantization
        model.status = ModelStatus.DOWNLOADING
        model.quantization = request.quantization
        model.progress = 0.0
        model.error_message = None
        await db.commit()
        await db.refresh(model)

        # Queue download job with Celery
        download_and_load_model.delay(
            model_id=model.id,
            repo_id=model.repo_id,
            quantization=new_quantization,
            access_token=request.access_token,
            trust_remote_code=request.trust_remote_code
        )

        logger.info(
            f"Initiated re-download for model {model_id}: "
            f"{old_quantization} -> {new_quantization}"
        )

        return ModelRedownloadResponse(
            model_id=model_id,
            repo_id=model.repo_id,
            old_quantization=old_quantization,
            new_quantization=new_quantization,
            status="downloading",
            message=f"Re-downloading model with {new_quantization} quantization"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate re-download for model {model_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate model re-download: {str(e)}"
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

    # Add extraction status to each model
    # Check if model has any completed activation extractions
    for model in models:
        has_completed = await db.execute(
            select(exists().where(
                ActivationExtraction.model_id == model.id,
                ActivationExtraction.status == ExtractionStatus.COMPLETED.value
            ))
        )
        # Add as dynamic attribute for Pydantic serialization
        model.has_completed_extractions = has_completed.scalar()

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


@router.get("/local-cache/list", response_model=dict)
async def list_local_models(db: AsyncSession = Depends(get_db)):
    """
    List models downloaded through the application.

    Returns models from the database that have been successfully downloaded
    and are available for local LLM labeling.

    Returns:
        dict with 'models' list containing model names
    """
    try:
        # Query database for downloaded models
        from ....models.model import Model

        result = await db.execute(
            select(Model.name)
            .where(Model.status == ModelStatus.READY.value)
            .order_by(Model.name)
        )

        model_names = [row[0] for row in result.fetchall()]

        return {"models": model_names}

    except Exception as e:
        logger.error(f"Error listing local models: {e}")
        return {"models": [], "error": str(e)}


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

    # Add extraction status
    has_completed = await db.execute(
        select(exists().where(
            ActivationExtraction.model_id == model.id,
            ActivationExtraction.status == ExtractionStatus.COMPLETED.value
        ))
    )
    model.has_completed_extractions = has_completed.scalar()

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
    Delete a model and queue file cleanup task.

    This endpoint deletes the model record from the database and queues
    a background task to remove associated files from disk.

    Args:
        model_id: Model ID (string format: m_{uuid})
        db: Database session

    Raises:
        HTTPException: If model not found
    """
    result = await ModelService.delete_model(db, model_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    # Queue background file cleanup task
    try:
        delete_model_files.delay(
            model_id=result["model_id"],
            file_path=result.get("file_path"),
            quantized_path=result.get("quantized_path")
        )
        logger.info(f"Queued file cleanup for model {model_id}")
    except Exception as e:
        # Log error but don't fail the deletion
        # (database record is already deleted)
        logger.error(f"Failed to queue file cleanup for model {model_id}: {e}")


@router.post("/{model_id}/estimate-extraction", status_code=200)
async def estimate_extraction_resources(
    model_id: str,
    request: ActivationExtractionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Estimate resource requirements for an activation extraction job.

    This endpoint calculates GPU memory, disk space, and processing time
    estimates for a proposed extraction configuration without actually
    starting the extraction.

    Args:
        model_id: Model ID (string format: m_{uuid})
        request: Extraction configuration (dataset_id, layers, hook_types, etc.)
        db: Database session

    Returns:
        Resource estimates including:
        - GPU memory required (MB/GB)
        - Disk space required (MB/GB)
        - Estimated processing time
        - Warnings if resources are excessive

    Raises:
        HTTPException: If model not found or dataset not found
    """
    from ....utils.resource_estimation import estimate_extraction_resources
    from ....services.dataset_service import DatasetService

    # Verify model exists
    model = await ModelService.get_model(db, model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    # Get dataset info
    dataset = await DatasetService.get_dataset(db, request.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_id}' not found"
        )

    # Get tokenization for this dataset and model to get avg_seq_length
    from ....models.dataset_tokenization import DatasetTokenization
    from sqlalchemy import select

    result = await db.execute(
        select(DatasetTokenization).where(
            DatasetTokenization.dataset_id == request.dataset_id,
            DatasetTokenization.model_id == model_id
        )
    )
    tokenization = result.scalar_one_or_none()

    # Get avg_seq_length from tokenization, or use default
    avg_seq_length = 512  # Default
    if tokenization and tokenization.avg_seq_length:
        avg_seq_length = tokenization.avg_seq_length

    # Build model config
    # NOTE: Always use FP16 for GPU memory estimation because activation_service.py
    # loads all models as FP16 (torch_dtype=torch.float16) regardless of storage quantization
    model_config = {
        "hidden_size": model.architecture_config.get("hidden_size") if model.architecture_config else 768,
        "num_layers": model.architecture_config.get("num_hidden_layers") or model.architecture_config.get("num_layers") if model.architecture_config else 12,
        "params_count": model.params_count,
        "quantization": "FP16"  # Always FP16 - models are loaded with torch_dtype=torch.float16
    }

    # Build extraction config
    extraction_config = {
        "layer_indices": request.layer_indices,
        "hook_types": request.hook_types,
        "batch_size": request.batch_size or 8,
        "max_samples": request.max_samples
    }

    # Build dataset config
    dataset_config = {
        "avg_sequence_length": avg_seq_length
    }

    # Calculate estimates
    estimates = estimate_extraction_resources(
        model_config=model_config,
        extraction_config=extraction_config,
        dataset_config=dataset_config
    )

    return {
        "model_id": model_id,
        "dataset_id": request.dataset_id,
        "estimates": estimates
    }


@router.post("/{model_id}/extract-activations", status_code=202)
async def extract_model_activations(
    model_id: str,
    request: ActivationExtractionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract activations from a model using a tokenized dataset.

    This endpoint queues an activation extraction job via Celery.
    The actual extraction happens asynchronously.

    Args:
        model_id: Model ID (string format: m_{uuid})
        request: Extraction configuration (dataset_id, layers, hook_types, etc.)
        db: Database session

    Returns:
        Job information with job_id for tracking

    Raises:
        HTTPException: If model not found or not ready
    """
    # Verify model exists and is ready
    model = await ModelService.get_model(db, model_id)

    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not ready (status: {model.status.value})"
        )

    try:
        # Generate extraction ID (same format as in extract_activations task)
        import datetime
        extraction_id = f"ext_{model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Queue extraction job with Celery
        task = extract_activations.delay(
            model_id=model_id,
            dataset_id=request.dataset_id,
            layer_indices=request.layer_indices,
            hook_types=request.hook_types,
            max_samples=request.max_samples,
            batch_size=request.batch_size or 8,
            micro_batch_size=request.micro_batch_size,
            gpu_id=request.gpu_id or 0,
        )

        # Emit immediate progress update to show job has started
        from ....workers.websocket_emitter import emit_extraction_progress
        emit_extraction_progress(
            model_id=model_id,
            extraction_id=extraction_id,
            progress=0,
            status="starting",
            message="Extraction job queued, waiting for worker..."
        )

        return {
            "job_id": task.id,
            "model_id": model_id,
            "dataset_id": request.dataset_id,
            "extraction_id": extraction_id,
            "status": "queued",
            "message": "Activation extraction job queued successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue extraction job: {str(e)}"
        )


@router.delete("/{model_id}/cancel", status_code=200)
async def cancel_model_download(
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an in-progress model download.

    This endpoint cancels the download task, cleans up partial files,
    and updates the model status to ERROR with "Cancelled by user".

    Args:
        model_id: Model ID (string format: m_{uuid})
        db: Database session

    Returns:
        Cancellation status

    Raises:
        HTTPException: If model not found or not in cancellable state
    """
    from ....workers.model_tasks import cancel_download

    # Verify model exists
    model = await ModelService.get_model(db, model_id)

    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    # Check if model is in a cancellable state
    if model.status not in [ModelStatus.DOWNLOADING, ModelStatus.LOADING, ModelStatus.QUANTIZING]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' cannot be cancelled (status: {model.status.value})"
        )

    try:
        # Call cancel_download task (runs synchronously for immediate response)
        # Note: We don't have task_id stored, so we can't revoke the specific task
        # Instead, the cancel task will clean up files and update database
        result = cancel_download(model_id=model_id)

        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        return {
            "model_id": model_id,
            "status": "cancelled",
            "message": "Download cancelled successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel download: {str(e)}"
        )


@router.get("/{model_id}/extractions/active")
async def get_active_extraction(
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the currently active extraction for a model.

    This endpoint returns the active extraction (if any) with its current progress,
    allowing the frontend to restore extraction state after page refresh.

    Args:
        model_id: Model ID (string format: m_{uuid})
        db: Database session

    Returns:
        Active extraction details with progress, or 404 if no active extraction

    Raises:
        HTTPException: If model not found or no active extraction
    """
    # Verify model exists
    model = await ModelService.get_model(db, model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    # Query database for active extraction
    with get_sync_db() as sync_db:
        active_extraction = ExtractionDatabaseService.get_active_extraction_for_model(
            sync_db, model_id
        )

    if not active_extraction:
        # Return null instead of 404 to avoid console spam
        # The frontend expects this when no extraction is active
        return {"data": None}

    # Return extraction details
    return {
        "data": {
            "extraction_id": active_extraction.id,
            "model_id": active_extraction.model_id,
            "dataset_id": active_extraction.dataset_id,
            "celery_task_id": active_extraction.celery_task_id,
            "status": active_extraction.status.value,
            "progress": active_extraction.progress,
            "samples_processed": active_extraction.samples_processed,
            "max_samples": active_extraction.max_samples,
            "layer_indices": active_extraction.layer_indices,
            "hook_types": active_extraction.hook_types,
            "batch_size": active_extraction.batch_size,
            "created_at": active_extraction.created_at.isoformat(),
            "updated_at": active_extraction.updated_at.isoformat(),
            "error_message": active_extraction.error_message
        }
    }


@router.get("/{model_id}/extractions")
async def list_model_extractions(
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    List all activation extractions for a model.

    This endpoint returns all extractions (completed and in-progress) with their
    metadata and statistics, combining database records with filesystem data.

    Args:
        model_id: Model ID (string format: m_{uuid})
        db: Database session

    Returns:
        List of extraction records with statistics

    Raises:
        HTTPException: If model not found
    """
    from ....services.activation_service import ActivationService

    # Verify model exists
    model = await ModelService.get_model(db, model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    # Get database records for all extractions (including in-progress)
    with get_sync_db() as sync_db:
        db_extractions = ExtractionDatabaseService.list_extractions_for_model(
            sync_db, model_id, limit=50
        )

    # Get filesystem-based extractions (completed extractions with metadata.json)
    activation_service = ActivationService()
    fs_extractions = activation_service.list_extractions()

    # Filter filesystem extractions for this model
    fs_extractions = [
        ext for ext in fs_extractions
        if ext.get("model_id") == model_id
    ]

    # Create a merged list:
    # - Database records provide authoritative status and progress
    # - Filesystem records provide full statistics and metadata
    extraction_map = {}

    # First, add all database records
    for db_ext in db_extractions:
        extraction_map[db_ext.id] = {
            "extraction_id": db_ext.id,
            "model_id": db_ext.model_id,
            "dataset_id": db_ext.dataset_id,
            "status": db_ext.status.value,
            "progress": db_ext.progress,
            "samples_processed": db_ext.samples_processed,
            "max_samples": db_ext.max_samples,
            "batch_size": db_ext.batch_size,
            "layer_indices": db_ext.layer_indices,
            "hook_types": db_ext.hook_types,
            "created_at": db_ext.created_at.isoformat(),
            "completed_at": db_ext.completed_at.isoformat() if db_ext.completed_at else None,
            "error_message": db_ext.error_message,
            "statistics": db_ext.statistics if db_ext.statistics else {},
            "saved_files": db_ext.saved_files if db_ext.saved_files else []
        }

    # Then, merge in filesystem data (for completed extractions)
    for fs_ext in fs_extractions:
        ext_id = fs_ext.get("extraction_id")
        if ext_id in extraction_map:
            # Update with full statistics and metadata from filesystem
            extraction_map[ext_id].update({
                "num_samples_processed": fs_ext.get("num_samples_processed"),
                "statistics": fs_ext.get("statistics", {}),
                "saved_files": fs_ext.get("saved_files", []),
                "dataset_path": fs_ext.get("dataset_path"),
                "architecture": fs_ext.get("architecture"),
                "quantization": fs_ext.get("quantization")
            })
        else:
            # Filesystem-only extraction (old extraction without database record)
            extraction_map[ext_id] = {
                "extraction_id": ext_id,
                "status": "completed",  # Must be completed if metadata.json exists
                "progress": 100.0,
                "created_at": fs_ext.get("created_at"),
                "num_samples_processed": fs_ext.get("num_samples_processed"),
                "layer_indices": fs_ext.get("layer_indices", []),
                "hook_types": fs_ext.get("hook_types", []),
                "max_samples": fs_ext.get("max_samples"),
                "statistics": fs_ext.get("statistics", {}),
                "saved_files": fs_ext.get("saved_files", []),
                "dataset_path": fs_ext.get("dataset_path"),
                "architecture": fs_ext.get("architecture"),
                "quantization": fs_ext.get("quantization")
            }

    # Convert to list and sort by created_at descending (newest first)
    extractions = list(extraction_map.values())
    extractions.sort(
        key=lambda x: x.get("created_at", ""),
        reverse=True
    )

    # Check deletion eligibility for each extraction
    # Query trainings table to find which extractions are in use
    with get_sync_db() as sync_db:
        from ....models.training import Training

        # Get all extraction IDs
        extraction_ids = [ext["extraction_id"] for ext in extractions]

        # Query trainings that reference these extractions
        trainings_using_extractions = sync_db.query(
            Training.extraction_id,
            Training.id,
            Training.status
        ).filter(
            Training.extraction_id.in_(extraction_ids)
        ).all()

        # Build a map: extraction_id -> list of training info
        extraction_usage_map = {}
        for training in trainings_using_extractions:
            if training.extraction_id not in extraction_usage_map:
                extraction_usage_map[training.extraction_id] = []
            extraction_usage_map[training.extraction_id].append({
                "training_id": training.id,
                "status": training.status
            })

        # Add deletion eligibility to each extraction
        for extraction in extractions:
            ext_id = extraction["extraction_id"]
            used_by_trainings = extraction_usage_map.get(ext_id, [])
            extraction["can_delete"] = len(used_by_trainings) == 0
            extraction["used_by_trainings"] = used_by_trainings

    return {
        "model_id": model_id,
        "model_name": model.name,
        "extractions": extractions,
        "count": len(extractions)
    }


@router.post("/{model_id}/extractions/{extraction_id}/cancel", response_model=ExtractionCancelResponse)
async def cancel_extraction(
    model_id: str,
    extraction_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an in-progress activation extraction.

    This endpoint cancels the extraction Celery task, updates the database
    record to CANCELLED status, and emits a WebSocket event.

    Args:
        model_id: Model ID (string format: m_{uuid})
        extraction_id: Extraction ID (string format: ext_m_{uuid}_{timestamp})
        db: Database session

    Returns:
        Cancellation confirmation with extraction_id and status

    Raises:
        HTTPException: If model or extraction not found, or extraction not cancellable
    """
    # Verify model exists
    model = await ModelService.get_model(db, model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    # Get extraction from database
    with get_sync_db() as sync_db:
        from ....models.extraction import ActivationExtraction
        extraction = sync_db.query(ActivationExtraction).filter(
            ActivationExtraction.id == extraction_id,
            ActivationExtraction.model_id == model_id
        ).first()

        if not extraction:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction '{extraction_id}' not found for model '{model_id}'"
            )

        # Check if extraction is in a cancellable state
        from ....models.extraction import ExtractionStatus
        if extraction.status not in [
            ExtractionStatus.QUEUED,
            ExtractionStatus.LOADING,
            ExtractionStatus.EXTRACTING,
            ExtractionStatus.SAVING
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Extraction '{extraction_id}' cannot be cancelled (status: {extraction.status.value})"
            )

        # Revoke Celery task if task_id is available
        if extraction.celery_task_id:
            try:
                celery_app.control.revoke(
                    extraction.celery_task_id,
                    terminate=True,
                    signal='SIGTERM'
                )
                logger.info(f"Revoked Celery task {extraction.celery_task_id} for extraction {extraction_id}")
            except Exception as e:
                logger.error(f"Failed to revoke Celery task: {e}")
                # Continue anyway - will update database status

        # Update database to CANCELLED
        extraction.status = ExtractionStatus.CANCELLED
        extraction.error_message = "Extraction cancelled by user"
        sync_db.commit()

        logger.info(f"Cancelled extraction {extraction_id} for model {model_id}")

    # Emit WebSocket event
    try:
        from ....workers.websocket_emitter import emit_extraction_progress
        emit_extraction_progress(
            model_id=model_id,
            extraction_id=extraction_id,
            progress=extraction.progress,
            status="cancelled",
            message="Extraction cancelled by user"
        )
    except Exception as e:
        logger.error(f"Failed to emit cancellation event: {e}")
        # Don't fail the request - cancellation was successful

    return ExtractionCancelResponse(
        extraction_id=extraction_id,
        status="cancelled",
        message="Extraction cancelled successfully"
    )


@router.post("/{model_id}/extractions/{extraction_id}/retry", response_model=ExtractionRetryResponse)
async def retry_extraction(
    model_id: str,
    extraction_id: str,
    request: ExtractionRetryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Retry a failed or cancelled extraction with optional parameter overrides.

    This endpoint creates a new extraction job based on the parameters of an
    existing extraction, with optional overrides for batch_size and max_samples.

    Args:
        model_id: Model ID (string format: m_{uuid})
        extraction_id: Original extraction ID (string format: ext_m_{uuid}_{timestamp})
        request: Optional parameter overrides (batch_size, max_samples)
        db: Database session

    Returns:
        Retry confirmation with new extraction_id and job_id

    Raises:
        HTTPException: If model or extraction not found, or model not ready
    """
    # Verify model exists and is ready
    model = await ModelService.get_model(db, model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not ready (status: {model.status.value})"
        )

    # Get original extraction from database
    with get_sync_db() as sync_db:
        from ....models.extraction import ActivationExtraction
        original_extraction = sync_db.query(ActivationExtraction).filter(
            ActivationExtraction.id == extraction_id,
            ActivationExtraction.model_id == model_id
        ).first()

        if not original_extraction:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction '{extraction_id}' not found for model '{model_id}'"
            )

    # Copy parameters from original extraction, applying overrides
    batch_size = request.batch_size if request.batch_size is not None else original_extraction.batch_size
    micro_batch_size = request.micro_batch_size if request.micro_batch_size is not None else original_extraction.micro_batch_size
    max_samples = request.max_samples if request.max_samples is not None else original_extraction.max_samples

    try:
        # Generate new extraction ID
        import datetime
        new_extraction_id = f"ext_{model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Queue new extraction job with Celery
        # Use the original gpu_id if available, default to 0
        gpu_id = getattr(original_extraction, 'gpu_id', 0) or 0
        task = extract_activations.delay(
            model_id=model_id,
            dataset_id=original_extraction.dataset_id,
            layer_indices=original_extraction.layer_indices,
            hook_types=original_extraction.hook_types,
            max_samples=max_samples,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            gpu_id=gpu_id,
        )

        logger.info(
            f"Queued retry extraction {new_extraction_id} for model {model_id} "
            f"(original: {extraction_id}, batch_size: {batch_size}, max_samples: {max_samples})"
        )

        # Emit immediate progress update to show job has started
        from ....workers.websocket_emitter import emit_extraction_progress
        emit_extraction_progress(
            model_id=model_id,
            extraction_id=new_extraction_id,
            progress=0,
            status="starting",
            message=f"Retry extraction queued (retry of {extraction_id})"
        )

        return ExtractionRetryResponse(
            original_extraction_id=extraction_id,
            new_extraction_id=new_extraction_id,
            job_id=task.id,
            status="queued",
            message="Extraction retry queued successfully"
        )

    except Exception as e:
        logger.error(f"Failed to queue retry extraction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue extraction retry: {str(e)}"
        )


@router.delete("/{model_id}/extractions", response_model=ExtractionDeleteResponse)
async def delete_extractions(
    model_id: str,
    request: ExtractionDeleteRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete multiple activation extractions for a model.

    This endpoint deletes extraction records from the database and removes
    the associated files from the filesystem. It performs batch deletion
    and reports success/failure for each extraction.

    Args:
        model_id: Model ID (string format: m_{uuid})
        request: List of extraction IDs to delete
        db: Database session

    Returns:
        Deletion summary with counts of successful and failed deletions

    Raises:
        HTTPException: If model not found
    """
    from ....services.activation_service import ActivationService
    from ....models.activation_extraction import ActivationExtraction

    # Verify model exists
    model = await ModelService.get_model(db, model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    activation_service = ActivationService()
    deleted_ids = []
    failed_ids = []
    errors = {}

    # Process each extraction ID
    for extraction_id in request.extraction_ids:
        try:
            # Delete from database
            with get_sync_db() as sync_db:
                extraction = sync_db.query(ActivationExtraction).filter(
                    ActivationExtraction.id == extraction_id,
                    ActivationExtraction.model_id == model_id
                ).first()

                if extraction:
                    sync_db.delete(extraction)
                    sync_db.commit()
                    logger.info(f"Deleted extraction record from database: {extraction_id}")

            # Delete from filesystem
            try:
                activation_service.delete_extraction(extraction_id)
                logger.info(f"Deleted extraction files from filesystem: {extraction_id}")
            except Exception as fs_error:
                # Log filesystem deletion error but don't fail the request
                # (database record is already deleted)
                logger.warning(f"Failed to delete extraction files for {extraction_id}: {fs_error}")

            deleted_ids.append(extraction_id)

        except Exception as e:
            logger.error(f"Failed to delete extraction {extraction_id}: {e}")
            failed_ids.append(extraction_id)
            errors[extraction_id] = str(e)

    deleted_count = len(deleted_ids)
    failed_count = len(failed_ids)

    # Generate message
    if failed_count == 0:
        message = f"Successfully deleted {deleted_count} extraction(s)"
    elif deleted_count == 0:
        message = f"Failed to delete all {failed_count} extraction(s)"
    else:
        message = f"Deleted {deleted_count} extraction(s), failed {failed_count}"

    logger.info(
        f"Batch deletion for model {model_id}: "
        f"deleted={deleted_count}, failed={failed_count}"
    )

    return ExtractionDeleteResponse(
        model_id=model_id,
        deleted_count=deleted_count,
        failed_count=failed_count,
        deleted_ids=deleted_ids,
        failed_ids=failed_ids,
        errors=errors,
        message=message
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
