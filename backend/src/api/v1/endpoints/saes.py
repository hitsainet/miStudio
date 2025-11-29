"""
SAE management API endpoints.

This module defines REST API endpoints for SAE operations including:
- Listing and searching SAEs
- Downloading SAEs from HuggingFace
- Uploading SAEs to HuggingFace
- Importing SAEs from training
- Deleting SAEs
- Feature extraction from SAEs
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ....core.database import get_db
from ....models.dataset import Dataset
from ....models.external_sae import SAESource, SAEStatus
from ....schemas.sae import (
    HFRepoPreviewRequest,
    HFRepoPreviewResponse,
    SAEDownloadRequest,
    SAEUploadRequest,
    SAEUploadResponse,
    SAEImportFromTrainingRequest,
    SAEImportFromFileRequest,
    SAEResponse,
    SAEListResponse,
    SAEDeleteRequest,
    SAEDeleteResponse,
    SAEFeatureBrowserResponse,
)
from ....schemas.extraction import ExtractionConfigRequest, ExtractionStatusResponse
from ....services.huggingface_sae_service import HuggingFaceSAEService
from ....services.sae_manager_service import SAEManagerService
from ....services.extraction_service import ExtractionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/saes", tags=["SAEs"])


# ============================================================================
# List and Search
# ============================================================================

@router.get("", response_model=SAEListResponse)
async def list_saes(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum records to return"),
    search: Optional[str] = Query(None, description="Search query"),
    source: Optional[str] = Query(None, description="Filter by source (huggingface, local, trained)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    sort_by: str = Query("created_at", description="Sort column"),
    order: str = Query("desc", description="Sort order (asc, desc)"),
    db: AsyncSession = Depends(get_db)
):
    """
    List SAEs with filtering and pagination.

    Returns a paginated list of SAEs. Can filter by source type, status,
    and model name. Search searches name, description, and HuggingFace repo.
    """
    # Parse source enum if provided
    source_enum = None
    if source:
        try:
            source_enum = SAESource(source)
        except ValueError:
            raise HTTPException(400, f"Invalid source: {source}")

    # Parse status enum if provided
    status_enum = None
    if status:
        try:
            status_enum = SAEStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    saes, total = await SAEManagerService.list_saes(
        db=db,
        skip=skip,
        limit=limit,
        search=search,
        source=source_enum,
        status=status_enum,
        model_name=model_name,
        sort_by=sort_by,
        order=order
    )

    return SAEListResponse(
        data=[SAEResponse.model_validate(sae) for sae in saes],
        pagination={
            "skip": skip,
            "limit": limit,
            "total": total,
            "has_more": skip + len(saes) < total
        }
    )


@router.get("/{sae_id}", response_model=SAEResponse)
async def get_sae(
    sae_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a single SAE by ID.
    """
    sae = await SAEManagerService.get_sae(db, sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {sae_id}")

    return SAEResponse.model_validate(sae)


# ============================================================================
# HuggingFace Operations
# ============================================================================

@router.post("/hf/preview", response_model=HFRepoPreviewResponse)
async def preview_hf_repository(request: HFRepoPreviewRequest):
    """
    Preview a HuggingFace repository to discover available SAEs.

    Returns list of files and detected SAE paths in the repository.
    Use this before downloading to see what's available.
    """
    try:
        preview = await HuggingFaceSAEService.preview_repository(
            repo_id=request.repo_id,
            access_token=request.access_token
        )
        return preview
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Error previewing repository: {e}")
        raise HTTPException(500, f"Error previewing repository: {str(e)}")


@router.post("/download", response_model=SAEResponse)
async def download_sae_from_hf(
    request: SAEDownloadRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Initiate an SAE download from HuggingFace.

    Creates an SAE record and starts the download in the background.
    Returns immediately with the SAE in PENDING status.
    Use the WebSocket or polling to track download progress.
    """
    try:
        # Create SAE record
        sae = await SAEManagerService.initiate_download(db, request)

        # Start download task in background
        # TODO: Replace with Celery task for better reliability
        background_tasks.add_task(
            _download_sae_background,
            sae_id=sae.id,
            request=request
        )

        return SAEResponse.model_validate(sae)

    except Exception as e:
        logger.error(f"Error initiating SAE download: {e}")
        raise HTTPException(500, f"Error initiating download: {str(e)}")


async def _download_sae_background(sae_id: str, request: SAEDownloadRequest):
    """Background task to download SAE from HuggingFace."""
    from ....core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            # Update status to downloading
            await SAEManagerService.update_download_progress(
                db, sae_id, progress=0.0, status=SAEStatus.DOWNLOADING
            )

            # Get storage path
            local_path = HuggingFaceSAEService.get_sae_storage_path(sae_id)

            # Download files
            result = await HuggingFaceSAEService.download_sae(
                repo_id=request.repo_id,
                filepath=request.filepath,
                local_dir=local_path,
                revision=request.revision,
                access_token=request.access_token
            )

            # Update with download info
            await SAEManagerService.update_download_progress(
                db, sae_id,
                progress=100.0,
                status=SAEStatus.READY,
                metadata_updates={
                    "files_downloaded": result.get("files_downloaded", []),
                    "is_directory": result.get("is_directory", False)
                }
            )

            # Update file size
            await SAEManagerService.update_sae_info(
                db, sae_id,
                file_size_bytes=result.get("file_size_bytes")
            )

            logger.info(f"SAE download completed: {sae_id}")

        except Exception as e:
            logger.error(f"SAE download failed: {sae_id} - {e}")
            await SAEManagerService.update_download_progress(
                db, sae_id,
                progress=0.0,
                status=SAEStatus.ERROR,
                error_message=str(e)
            )


@router.post("/upload", response_model=SAEUploadResponse)
async def upload_sae_to_hf(
    request: SAEUploadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Upload an SAE to HuggingFace.

    Uploads a local SAE to the specified HuggingFace repository.
    Requires a HuggingFace access token with write permissions.
    """
    # Get the SAE
    sae = await SAEManagerService.get_sae(db, request.sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {request.sae_id}")

    if sae.status != SAEStatus.READY.value:
        raise HTTPException(400, f"SAE is not ready for upload: {sae.status}")

    if not sae.local_path:
        raise HTTPException(400, "SAE has no local files to upload")

    try:
        from pathlib import Path
        local_path = Path(sae.local_path)

        if not local_path.exists():
            raise HTTPException(400, f"SAE files not found at: {sae.local_path}")

        result = await HuggingFaceSAEService.upload_sae(
            local_path=local_path,
            repo_id=request.repo_id,
            filepath=request.filepath,
            access_token=request.access_token,
            create_repo=request.create_repo,
            private=request.private,
            commit_message=request.commit_message
        )

        return SAEUploadResponse(
            sae_id=request.sae_id,
            repo_id=result["repo_id"],
            filepath=result["filepath"],
            url=result["url"],
            commit_hash=result.get("commit_hash")
        )

    except Exception as e:
        logger.error(f"Error uploading SAE: {e}")
        raise HTTPException(500, f"Error uploading SAE: {str(e)}")


# ============================================================================
# Import Operations
# ============================================================================

@router.post("/import/training", response_model=SAEResponse)
async def import_sae_from_training(
    request: SAEImportFromTrainingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Import an SAE from a completed training job.

    Creates a copy of the trained SAE in the SAE storage directory.
    The SAE is immediately ready for use in steering.
    """
    try:
        sae = await SAEManagerService.import_from_training(db, request)
        return SAEResponse.model_validate(sae)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Error importing SAE from training: {e}")
        raise HTTPException(500, f"Error importing SAE: {str(e)}")


@router.post("/import/file", response_model=SAEResponse)
async def import_sae_from_file(
    request: SAEImportFromFileRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Import an SAE from a local file.

    Creates a copy of the SAE file in the SAE storage directory.
    The SAE is immediately ready for use in steering.
    """
    try:
        sae = await SAEManagerService.import_from_file(db, request)
        return SAEResponse.model_validate(sae)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Error importing SAE from file: {e}")
        raise HTTPException(500, f"Error importing SAE: {str(e)}")


# ============================================================================
# Delete Operations
# ============================================================================

@router.delete("/{sae_id}")
async def delete_sae(
    sae_id: str,
    delete_files: bool = Query(True, description="Delete local files"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a single SAE.

    Soft-deletes the SAE record. Optionally deletes local files.
    """
    success = await SAEManagerService.delete_sae(db, sae_id, delete_files)
    if not success:
        raise HTTPException(404, f"SAE not found: {sae_id}")

    return {"message": f"SAE {sae_id} deleted successfully"}


@router.post("/delete", response_model=SAEDeleteResponse)
async def delete_saes_batch(
    request: SAEDeleteRequest,
    delete_files: bool = Query(True, description="Delete local files"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete multiple SAEs.

    Soft-deletes multiple SAE records. Returns count of successful/failed deletions.
    """
    result = await SAEManagerService.delete_saes_batch(
        db, request.sae_ids, delete_files
    )

    return SAEDeleteResponse(
        deleted_count=result["deleted_count"],
        failed_count=result["failed_count"],
        deleted_ids=result["deleted_ids"],
        failed_ids=result["failed_ids"],
        errors=result["errors"],
        message=f"Deleted {result['deleted_count']} SAE(s)"
    )


# ============================================================================
# Feature Browser (for Steering integration)
# ============================================================================

@router.get("/{sae_id}/features", response_model=SAEFeatureBrowserResponse)
async def browse_sae_features(
    sae_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=1000),
    search: Optional[str] = Query(None, description="Search feature by index or label"),
    db: AsyncSession = Depends(get_db)
):
    """
    Browse features in an SAE for steering selection.

    Returns paginated list of features with their labels and statistics.
    This is used by the Steering tab's feature browser.
    """
    from sqlalchemy import select, func, or_
    from ....schemas.sae import SAEFeatureSummary
    from ....models.feature import Feature

    sae = await SAEManagerService.get_sae(db, sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {sae_id}")

    if sae.status != SAEStatus.READY.value:
        raise HTTPException(400, f"SAE is not ready: {sae.status}")

    n_features = sae.n_features or 8192
    layer = sae.layer if sae.layer is not None else 0
    training_id = sae.training_id

    # If SAE has a linked training, try to fetch real feature data
    features_from_db = {}
    if training_id:
        # Build query for features
        query = select(Feature).where(Feature.training_id == training_id)

        # Apply search filter
        if search:
            search = search.strip()
            if search.isdigit():
                # Search by exact index
                search_idx = int(search)
                query = query.where(Feature.neuron_index == search_idx)
            else:
                # Search by label (name, category, or description)
                search_pattern = f"%{search}%"
                query = query.where(
                    or_(
                        Feature.name.ilike(search_pattern),
                        Feature.category.ilike(search_pattern),
                        Feature.description.ilike(search_pattern)
                    )
                )

        # Get total count for pagination
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination and ordering
        query = query.order_by(Feature.neuron_index).offset(skip).limit(limit)

        result = await db.execute(query)
        db_features = result.scalars().all()

        # Build feature summaries from DB
        features = []
        for f in db_features:
            # Parse top_tokens - could be a list (JSONB) or comma-separated string
            top_tokens = []
            if f.example_tokens_summary:
                if isinstance(f.example_tokens_summary, list):
                    top_tokens = f.example_tokens_summary[:5]  # Limit to 5
                elif isinstance(f.example_tokens_summary, str):
                    # Parse comma-separated string
                    top_tokens = [t.strip() for t in f.example_tokens_summary.split(",")][:5]

            features.append(
                SAEFeatureSummary(
                    feature_idx=f.neuron_index,
                    layer=layer,
                    label=f.name if f.name and not f.name.startswith("feature_") else None,
                    activation_count=None,  # Not stored directly
                    mean_activation=f.mean_activation,
                    max_activation=f.max_activation,
                    top_tokens=top_tokens,
                    neuronpedia_url=None,
                    feature_id=f.id
                )
            )

        return SAEFeatureBrowserResponse(
            sae_id=sae_id,
            n_features=n_features,
            features=features,
            pagination={
                "skip": skip,
                "limit": limit,
                "total": total,
                "has_more": skip + limit < total
            }
        )

    # Fallback: No training linked, return placeholder indices
    all_indices = list(range(n_features))

    # Apply search filter (only numeric search for unlabeled SAEs)
    if search:
        search = search.strip()
        if search.isdigit():
            search_idx = int(search)
            if 0 <= search_idx < n_features:
                all_indices = [search_idx]
            else:
                all_indices = []
        else:
            all_indices = []

    total = len(all_indices)
    paginated_indices = all_indices[skip:skip + limit]

    features = [
        SAEFeatureSummary(
            feature_idx=idx,
            layer=layer,
            label=None,
            activation_count=None,
            mean_activation=None,
            max_activation=None,
            top_tokens=[],
            neuronpedia_url=None
        )
        for idx in paginated_indices
    ]

    return SAEFeatureBrowserResponse(
        sae_id=sae_id,
        n_features=n_features,
        features=features,
        pagination={
            "skip": skip,
            "limit": limit,
            "total": total,
            "has_more": skip + limit < total
        }
    )


# ============================================================================
# Feature Extraction Operations
# ============================================================================

@router.post("/{sae_id}/extract-features", response_model=ExtractionStatusResponse)
async def start_sae_extraction(
    sae_id: str,
    config: ExtractionConfigRequest,
    dataset_id: str = Query(..., description="Dataset ID to use for extraction"),
    db: AsyncSession = Depends(get_db)
):
    """
    Start feature extraction from an external SAE.

    Runs activations through the SAE and stores top-k activating examples
    for each feature. This enables feature browsing and labeling.

    Requires:
    - SAE must be in READY status
    - Dataset must exist and be downloaded

    Args:
        sae_id: ID of the SAE to extract features from
        dataset_id: ID of the dataset to use for extraction
        config: Extraction configuration (evaluation_samples, top_k_examples, etc.)
    """
    sae = await SAEManagerService.get_sae(db, sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {sae_id}")

    if sae.status != SAEStatus.READY.value:
        raise HTTPException(400, f"SAE is not ready for extraction: {sae.status}")

    try:
        extraction_service = ExtractionService(db)

        # Merge dataset_id into config
        config_dict = config.model_dump()
        config_dict["dataset_id"] = dataset_id

        extraction_job = await extraction_service.start_extraction_for_sae(
            sae_id=sae_id,
            config=config_dict
        )

        # Lookup dataset name
        dataset_name = None
        if dataset_id:
            dataset_result = await db.execute(
                select(Dataset).where(Dataset.id == dataset_id)
            )
            dataset = dataset_result.scalar_one_or_none()
            if dataset:
                dataset_name = dataset.name

        return ExtractionStatusResponse(
            id=extraction_job.id,
            training_id=None,
            external_sae_id=sae_id,
            source_type="external_sae",
            model_name=sae.model_id,
            dataset_name=dataset_name,
            sae_name=sae.name,
            status=extraction_job.status,
            progress=extraction_job.progress,
            features_extracted=extraction_job.features_extracted,
            total_features=extraction_job.total_features,
            config=extraction_job.config or {},
            created_at=extraction_job.created_at,
            updated_at=extraction_job.updated_at,
            completed_at=extraction_job.completed_at
        )

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        error_message = f"Error starting extraction: {str(e)}"
        logger.error(error_message, exc_info=True)

        # Try to update the extraction job with error (if it was created)
        try:
            from sqlalchemy import desc
            from ....models.extraction_job import ExtractionJob, ExtractionStatus
            result = await db.execute(
                select(ExtractionJob)
                .where(ExtractionJob.external_sae_id == sae_id)
                .order_by(desc(ExtractionJob.created_at))
                .limit(1)
            )
            extraction_job = result.scalar_one_or_none()
            if extraction_job and extraction_job.status in [ExtractionStatus.QUEUED.value, ExtractionStatus.EXTRACTING.value]:
                extraction_job.status = ExtractionStatus.FAILED.value
                extraction_job.error_message = str(e)
                await db.commit()
        except Exception:
            pass  # Best effort

        raise HTTPException(500, error_message)


@router.get("/{sae_id}/extraction-status", response_model=ExtractionStatusResponse)
async def get_sae_extraction_status(
    sae_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the status of a feature extraction job for an SAE.

    Returns the most recent extraction job for this SAE.
    """
    sae = await SAEManagerService.get_sae(db, sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {sae_id}")

    try:
        extraction_service = ExtractionService(db)
        status = await extraction_service.get_extraction_status_for_sae(sae_id)

        if not status:
            raise HTTPException(404, f"No extraction found for SAE: {sae_id}")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SAE extraction status: {e}", exc_info=True)
        raise HTTPException(500, f"Error getting extraction status: {str(e)}")


@router.post("/{sae_id}/cancel-extraction")
async def cancel_sae_extraction(
    sae_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an in-progress feature extraction for an SAE.

    Only works for extractions in QUEUED or EXTRACTING status.
    """
    from ....models.extraction_job import ExtractionJob, ExtractionStatus
    from sqlalchemy import select, desc
    from datetime import datetime, timezone

    sae = await SAEManagerService.get_sae(db, sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {sae_id}")

    # Find the most recent extraction job for this SAE
    query = select(ExtractionJob).where(
        ExtractionJob.external_sae_id == sae_id
    ).order_by(desc(ExtractionJob.created_at)).limit(1)

    result = await db.execute(query)
    extraction_job = result.scalar_one_or_none()

    if not extraction_job:
        raise HTTPException(404, f"No extraction found for SAE: {sae_id}")

    if extraction_job.status not in [ExtractionStatus.QUEUED.value, ExtractionStatus.EXTRACTING.value]:
        raise HTTPException(
            400,
            f"Cannot cancel extraction in status: {extraction_job.status}"
        )

    # Revoke Celery task if task_id is available
    if extraction_job.celery_task_id:
        try:
            from ....core.celery_app import celery_app
            celery_app.control.revoke(
                extraction_job.celery_task_id,
                terminate=True,
                signal='SIGTERM'
            )
            logger.info(f"Revoked Celery task {extraction_job.celery_task_id} for extraction {extraction_job.id}")
        except Exception as e:
            logger.error(f"Failed to revoke Celery task: {e}")
            # Continue anyway - will update database status

    # Update status to FAILED with cancellation message
    extraction_job.status = ExtractionStatus.FAILED.value
    extraction_job.error_message = "Cancelled by user"
    extraction_job.updated_at = datetime.now(timezone.utc)

    await db.commit()

    return {"message": f"Extraction {extraction_job.id} cancelled"}
