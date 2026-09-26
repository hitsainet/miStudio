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

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from sqlalchemy import select, or_

from starlette.concurrency import run_in_threadpool

from ....core.cancellation import request_cancel
from ....core.config import settings
from ....core.database import get_db
from ..gpu_request import resolve_gpu_request
from ....models.dataset import Dataset, DatasetStatus
from ....models.external_sae import SAESource, SAEStatus
from ....services.sae_hook_support import (
    UnsupportedSae,
    non_residual_hook_reason,
    unrecorded_layer_reason,
)
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
    TrainingAvailableSAEsResponse,
    SAEImportFromTrainingResponse,
)
from ....schemas.extraction import (
    ExtractionConfigRequest,
    ExtractionStatusResponse,
    BatchExtractionRequest,
    BatchExtractionResponse,
    BatchExtractionJobInfo,
    BatchExtractionSkippedInfo,
)
from ....services.huggingface_sae_service import HuggingFaceSAEService
from ....services.sae_manager_service import SAEManagerService
from ....workers.sae_tasks import download_sae_task
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
        logger.exception("Error previewing repository")
        raise HTTPException(500, f"Error previewing repository: {str(e)}")


@router.post("/download", response_model=SAEResponse)
async def download_sae_from_hf(
    request: SAEDownloadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Initiate an SAE download from HuggingFace.

    Creates an SAE record and starts the download as a Celery task.
    Returns immediately with the SAE in PENDING status.
    Use the WebSocket or polling to track download progress.
    """
    try:
        # Create SAE record
        sae = await SAEManagerService.initiate_download(db, request)

        # Start download as a Celery task for reliability and retry support
        download_sae_task.delay(
            sae_id=sae.id,
            repo_id=request.repo_id,
            filepath=request.filepath,
            access_token=request.access_token,
            revision=request.revision,
        )

        return SAEResponse.model_validate(sae)

    except Exception as e:
        logger.exception("Error initiating SAE download")
        raise HTTPException(500, f"Error initiating download: {str(e)}")



@router.post("/upload", response_model=SAEUploadResponse)
async def upload_sae_to_hf(
    request: SAEUploadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Upload an SAE to HuggingFace.

    Uploads a local SAE to the specified HuggingFace repository.
    Requires a HuggingFace access token with write permissions.

    ⚠ THE TOKEN IS CHECKED BEFORE ANY WORK, and it may come from Settings → API Keys rather than
    the request — `resolve_hf_token` is the resolver the preview and download routes have always
    used, and this route is the one that did not. A missing token is a 401 here, not a 500 raised
    out of the executor after the repository has already been created.
    """
    from ....services.huggingface_sae_service import resolve_hf_token

    if not resolve_hf_token(request.access_token):
        raise HTTPException(
            401,
            "no HuggingFace token with write permissions: pass one in the request or store it in "
            "Settings → API Keys",
        )

    # Get the SAE
    sae = await SAEManagerService.get_sae(db, request.sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {request.sae_id}")

    if sae.status != SAEStatus.READY.value:
        raise HTTPException(400, f"SAE is not ready for upload: {sae.status}")

    if not sae.local_path:
        raise HTTPException(400, "SAE has no local files to upload")

    try:
        local_path = settings.resolve_data_path(sae.local_path)

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
        logger.exception("Error uploading SAE")
        raise HTTPException(500, f"Error uploading SAE: {str(e)}")


# ============================================================================
# Import Operations
# ============================================================================

@router.get("/training/{training_id}/available", response_model=TrainingAvailableSAEsResponse)
async def get_available_saes_from_training(
    training_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    List available SAEs in a completed training for import.

    Scans the training checkpoint directory and returns information about
    each available SAE, including layer index, hook type, and file size.
    Use this to preview what can be imported before calling the import endpoint.
    """
    try:
        response = await SAEManagerService.get_available_saes_from_training(db, training_id)
        return response
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Error getting available SAEs from training")
        raise HTTPException(500, f"Error listing available SAEs: {str(e)}")


@router.post("/import/training", response_model=SAEImportFromTrainingResponse)
async def import_sae_from_training(
    request: SAEImportFromTrainingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Import SAE(s) from a completed training job.

    Supports importing multiple SAEs from multi-layer/multi-hook trainings.
    By default (import_all=True), imports all available SAEs.
    Use layers and hook_types filters to import specific SAEs.

    Creates copies of the trained SAE(s) in the SAE storage directory.
    Each SAE is immediately ready for use in steering.
    """
    try:
        response = await SAEManagerService.import_from_training(db, request)
        return response
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Error importing SAE from training")
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
        logger.exception("Error importing SAE from file")
        raise HTTPException(500, f"Error importing SAE: {str(e)}")


# ============================================================================
# Delete Operations
# ============================================================================

@router.delete("/{sae_id}")
async def delete_sae(
    sae_id: str,
    delete_files: bool = Query(True, description="Delete local files"),
    force: bool = Query(False, description="Unbind cluster profiles and delete anyway"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a single SAE.

    Soft-deletes the SAE record. Optionally deletes local files.
    Guarded (Feature 014): SAEs with bound cluster profiles return a structured
    409 unless ``force`` — force unbinds the profiles (they survive as unbound,
    steerable after re-binding) rather than destroying user-authored work.
    """
    from src.services.cluster_profile_service import ClusterProfileService

    profile_count = await ClusterProfileService.count_for_sae(db, sae_id)
    if profile_count > 0:
        if not force:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "PROFILES_BOUND",
                    "profile_count": profile_count,
                    "message": (
                        f"{profile_count} cluster profile(s) are bound to this SAE. "
                        "Delete them, or retry with force=true to unbind them "
                        "(profiles are kept, marked unbound)."
                    ),
                },
            )
        await ClusterProfileService.unbind_for_sae(db, sae_id)

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
    # EVERY summary below carries this layer, and the steering panel routes the features
    # the user picks to it -- so ``else 0`` did not merely mislabel a listing, it steered
    # layer 0 (review R3-B, R3B-11).
    browser_layer_reason = unrecorded_layer_reason(sae.layer, "The feature browser", sae.id)
    if browser_layer_reason:
        raise HTTPException(422, browser_layer_reason)
    layer = sae.layer
    # SCOPE BY THE SAE ITSELF, NOT BY ITS TRAINING (MIS-E2E-100).
    #
    # This resolved features only through `sae.training_id`. A downloaded or
    # externally imported SAE has `external_sae_id` set and `training_id` NULL,
    # so no feature ever matched and the handler fell through to a placeholder
    # branch: every feature rendered with no label, no statistics, and no
    # `activation_frequency`.
    #
    # That last one is the sharp end. The frequency-derived auto-baseline
    # (`S = clamp(2.9 − 2.6·freq, 1, 3)`, IDL-27) has nothing to compute from
    # without it, so every feature of a community SAE silently took the default
    # strength of 10 instead of its measured one — and downloading a community
    # SAE from HuggingFace is a first-class documented workflow.
    #
    # Features carry `external_sae_id` pointing at this registry row, which is
    # the direct link the training hop was standing in for.
    training_id = sae.training_id

    features_from_db = {}
    scopes = [Feature.external_sae_id == sae.id]
    if training_id:
        # EITHER link, not just the new one. A feature extracted before the
        # registry existed carries only `training_id`; narrowing to
        # `external_sae_id` alone would have traded one silent empty result for
        # another. `or_` covers both without needing to know which era a row
        # came from.
        scopes.append(Feature.training_id == training_id)
    feature_scope = or_(*scopes) if len(scopes) > 1 else scopes[0]

    if feature_scope is not None:
        # Build query for features
        query = select(Feature).where(feature_scope)

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
                    activation_frequency=f.activation_frequency,  # Feature 011: steering auto-baseline
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
            neuronpedia_url=None,
            feature_id=None  # Explicitly set for consistency
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

def _cannot_be_a_dataset_id(value) -> bool:
    """True when `value` cannot possibly name a dataset row.

    `Dataset.id` is a UUID column, so an id that is not UUID-shaped cannot be
    BOUND: asyncpg raises while encoding the parameter, before any query runs.
    Such an id matches nothing, so callers treat it as not-found.

    Checked here rather than caught downstream, deliberately. The bind failure
    surfaces as a bare `sqlalchemy.exc.DBAPIError` (measured, with a control),
    and DBAPIError is the BASE class of OperationalError and InterfaceError —
    catching it would report a database outage as "dataset not found", a silent
    wrong answer. A shape check cannot do that.
    """
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return True
    return False


@router.post("/{sae_id}/extract-features", response_model=ExtractionStatusResponse)
async def start_sae_extraction(
    sae_id: str,
    config: ExtractionConfigRequest,
    dataset_id: Optional[str] = Query(
        None,
        description="Single dataset ID (legacy single-corpus form). Ignored when config.dataset_ids is supplied.",
    ),
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

    # Resolved before the job row exists, and OUTSIDE the try below: its
    # generic handler would turn the 400 for an unknown card into a 500.
    gpu_request = resolve_gpu_request(config.gpu, can_split=True)

    # AN UNUSABLE SAE IS REFUSED BEFORE THE REQUEST'S DATA IS.
    #
    # `start_extraction_for_sae` raises these, and the handler below answers
    # them with 422 — but that is AFTER the corpus resolution added for the
    # mixture, which answers first for a bad dataset id. The precedence matters:
    # a non-residual hook or an unrecorded layer is a PERMANENT property of the
    # SAE, while a bad dataset id is a transient property of one request, and
    # reporting the transient one hides the fact that this SAE can never be
    # extracted at all. Same consumer string and same helpers as the service, so
    # the message is identical wherever it surfaces; same shape as the feature
    # browser's early refusal above.
    sae_hook_reason = non_residual_hook_reason(sae.hook_type, "Feature extraction")
    if sae_hook_reason:
        raise HTTPException(422, sae_hook_reason)
    sae_layer_reason = unrecorded_layer_reason(sae.layer, "Feature extraction", sae.id)
    if sae_layer_reason:
        raise HTTPException(422, sae_layer_reason)

    # Which corpora this run reads. Resolved and validated OUT HERE for the
    # same reason as gpu_request above: the try below has a generic handler
    # that would turn each of these 400s into a 500.
    dataset_ids = list(config.dataset_ids) if config.dataset_ids else ([dataset_id] if dataset_id else [])
    if not dataset_ids:
        raise HTTPException(
            400,
            "No dataset selected: supply dataset_ids in the body or dataset_id in the query string",
        )
    if len(set(dataset_ids)) != len(dataset_ids):
        raise HTTPException(400, f"dataset_ids contains duplicates: {dataset_ids}")
    if config.dataset_weights is not None:
        if len(config.dataset_weights) != len(dataset_ids):
            raise HTTPException(
                400,
                f"dataset_ids and dataset_weights must correspond one-to-one, got "
                f"{len(dataset_ids)} ids and {len(config.dataset_weights)} weights",
            )
        if any(w < 0 for w in config.dataset_weights):
            raise HTTPException(400, "dataset_weights must be non-negative")
        if sum(config.dataset_weights) <= 0:
            raise HTTPException(400, "dataset_weights must not sum to zero")
    # Fetch each corpus ONCE and keep the row: the name lookup below reuses it
    # rather than re-querying the same id, which is both a round trip saved and
    # one fewer place for the two reads to disagree.
    #
    # `Dataset.id` is a UUID column, so an id that is not UUID-shaped cannot be
    # BOUND at all: asyncpg raises while encoding the parameter, before the
    # query runs. An unparseable identifier is simply one that matches nothing,
    # so it is treated as not-found and never sent to the driver.
    #
    # Checked in Python rather than caught, deliberately. The bind failure
    # surfaces as a bare `sqlalchemy.exc.DBAPIError` — measured, with a control
    # — and DBAPIError is the BASE class of OperationalError and
    # InterfaceError. Catching it would report a database outage as "dataset
    # not found", which is a silent wrong answer. A shape check cannot do that.
    dataset_rows = []
    for ds_id in dataset_ids:
        if _cannot_be_a_dataset_id(ds_id):
            dataset_rows.append(None)
            continue
        row = (await db.execute(select(Dataset).where(Dataset.id == ds_id))).scalar_one_or_none()
        dataset_rows.append(row)
    missing = [ds_id for ds_id, row in zip(dataset_ids, dataset_rows) if row is None]
    if missing:
        # 400, NOT 404, and this EXACT wording — both are existing contract.
        # An unknown dataset already produced a 400 here (the service raised
        # ValueError into the generic handler) with the message
        # "Dataset {id} not found", which `test_sae_hook_recorded_at_import`
        # asserts verbatim. Keeping the singular phrasing per id means the
        # single-corpus case is byte-identical to what callers have always seen,
        # and the multi-corpus case just lists more of the same sentence.
        raise HTTPException(400, "; ".join(f"Dataset {d} not found" for d in missing))

    try:
        extraction_service = ExtractionService(db)

        # Merge dataset_id into config. `gpu` stays out of it: the job row's
        # `gpu_request` column is the one record the worker reads, and a copy
        # in `config` could only ever disagree with it.
        config_dict = config.model_dump(exclude={"gpu"})
        # CANONICAL DUAL WRITE. `dataset_ids` is what the worker plans the
        # mixture from; `dataset_id` is kept populated with the FIRST corpus so
        # that every legacy reader — list_extractions, the dataset_name lookup
        # below, the batch chain — keeps working with no change at all.
        config_dict["dataset_ids"] = dataset_ids
        config_dict["dataset_id"] = dataset_ids[0]

        extraction_job = await extraction_service.start_extraction_for_sae(
            sae_id=sae_id,
            config=config_dict,
            gpu_request=gpu_request,
        )

        # The name of the FIRST corpus, from the row already fetched above.
        # Under a mixture this is one of several; the full list travels in
        # config["dataset_ids"], and `dataset_names` on the response carries
        # them all. This single field stays first-corpus so existing readers
        # keep rendering exactly what they rendered before.
        dataset_name = dataset_rows[0].name if dataset_rows and dataset_rows[0] else None
        dataset_names = [row.name for row in dataset_rows if row is not None]

        return ExtractionStatusResponse(
            id=extraction_job.id,
            training_id=None,
            external_sae_id=sae_id,
            source_type="external_sae",
            model_name=sae.model_id,
            dataset_name=dataset_name,
            dataset_names=dataset_names,
            sae_name=sae.name,
            status=extraction_job.status,
            progress=extraction_job.progress,
            features_extracted=extraction_job.features_extracted,
            total_features=extraction_job.total_features,
            config=extraction_job.config or {},
            gpu_request=extraction_job.gpu_request,
            gpu_uuid=extraction_job.gpu_uuid,
            created_at=extraction_job.created_at,
            updated_at=extraction_job.updated_at,
            completed_at=extraction_job.completed_at
        )

    except UnsupportedSae as e:
        # A multi-hook training's MLP and attention SAEs import like any other, and
        # extraction hooks each layer's decoder output for every SAE (review R1-A, A5) --
        # at the layer the SAE records, or, when it records none, at layer 0 without a
        # word (review R3-B, R3B-11). Both refusals are UnsupportedSae.
        raise HTTPException(422, str(e))
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

    # CANCELLED, NOT FAILED. This wrote `status = FAILED` with the message
    # "Cancelled by user" while the sibling lifecycle (`models.py`) wrote
    # CANCELLED for the same operator action. `ExtractionStatus.CANCELLED` has
    # existed the whole time. Two consequences, both real: the UI showed a
    # deliberate stop as a failure, and — once a worker polls this row — a
    # checker looking for "cancelled" could never see a cancel expressed as
    # "failed". That vocabulary split is what Shape C tests for.
    #
    # `request_cancel` also issues a plain `revoke()` centrally, for the one
    # case it genuinely handles: a task that has not started never will.
    # `terminate=True, signal='SIGTERM'` is gone — it signals a POOL CHILD and
    # this worker is `--pool=solo`, so it returned cleanly and did nothing.
    outcome = await run_in_threadpool(
        request_cancel,
        "sae_extraction",
        extraction_job.id,
        reason="Cancelled by user",
        celery_task_id=extraction_job.celery_task_id,
    )

    return {"message": outcome.detail, "extraction_id": extraction_job.id}


@router.post("/batch-extract-features", response_model=BatchExtractionResponse)
async def start_batch_sae_extraction(
    request: BatchExtractionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start feature extraction from multiple SAEs in a single batch.

    Creates extraction jobs for all specified SAEs using the same dataset
    and configuration. Jobs are queued and processed sequentially.

    Requires:
    - All SAEs must be in READY status
    - Dataset must exist and be in READY status

    Args:
        request: Batch extraction request with SAE IDs, dataset ID, and config

    Returns:
        Batch extraction response with created jobs and any skipped SAEs
    """

    # Every SAE in the batch reads the SAME corpora. `dataset_ids` is canonical;
    # `dataset_id` stays the legacy single-corpus form and the first entry.
    batch_dataset_ids = list(request.dataset_ids) if request.dataset_ids else [request.dataset_id]
    if len(set(batch_dataset_ids)) != len(batch_dataset_ids):
        raise HTTPException(400, f"dataset_ids contains duplicates: {batch_dataset_ids}")
    if request.dataset_weights is not None:
        if len(request.dataset_weights) != len(batch_dataset_ids):
            raise HTTPException(
                400,
                f"dataset_ids and dataset_weights must correspond one-to-one, got "
                f"{len(batch_dataset_ids)} ids and {len(request.dataset_weights)} weights",
            )
        if any(w < 0 for w in request.dataset_weights):
            raise HTTPException(400, "dataset_weights must be non-negative")
        if sum(request.dataset_weights) <= 0:
            raise HTTPException(400, "dataset_weights must not sum to zero")

    # Validate EVERY corpus exists and is ready — checking only the first would
    # let a bad id through and fail in the worker, after the job rows exist.
    batch_dataset_rows = []
    for ds_id in batch_dataset_ids:
        # Same shape check as the single-SAE endpoint. PRE-EXISTING exposure —
        # the original code bound `request.dataset_id` to this UUID column with
        # no check either — but this is the loop the mixture rewrote, so it is
        # fixed here rather than recorded as debt. Without it the batch path
        # crashes on a malformed id while the single path answers cleanly.
        dataset = None
        if not _cannot_be_a_dataset_id(ds_id):
            dataset_result = await db.execute(
                select(Dataset).where(Dataset.id == ds_id)
            )
            dataset = dataset_result.scalar_one_or_none()
        if not dataset:
            raise HTTPException(404, f"Dataset not found: {ds_id}")
        if dataset.status != DatasetStatus.READY:
            raise HTTPException(400, f"Dataset is not ready: {ds_id} is {dataset.status.value}")
        batch_dataset_rows.append(dataset)

    # Before any job row is written: an unknown card is a 400 and the batch
    # creates nothing. Every job in the batch stores this request on its own
    # row, and a chained job is placed from that row when it starts.
    gpu_request = resolve_gpu_request(request.gpu, can_split=True)

    extraction_service = ExtractionService(db)

    # Build config from request (no `gpu`: each row's gpu_request is the record)
    config_dict = {
        # Canonical dual write, same as the single-SAE endpoint.
        "dataset_id": batch_dataset_ids[0],
        "dataset_ids": batch_dataset_ids,
        "dataset_weights": list(request.dataset_weights) if request.dataset_weights else None,
        "evaluation_samples": request.evaluation_samples,
        "top_k_examples": request.top_k_examples,
        "filter_special": request.filter_special,
        "filter_single_char": request.filter_single_char,
        "filter_punctuation": request.filter_punctuation,
        "filter_numbers": request.filter_numbers,
        "filter_fragments": request.filter_fragments,
        "filter_stop_words": request.filter_stop_words,
        "context_prefix_tokens": request.context_prefix_tokens,
        "context_suffix_tokens": request.context_suffix_tokens,
        "min_activation_frequency": request.min_activation_frequency,
        "auto_nlp": request.auto_nlp,
    }

    # Create batch extraction
    result = await extraction_service.start_batch_extraction_for_saes(
        sae_ids=request.sae_ids,
        config=config_dict,
        gpu_request=gpu_request,
    )

    return BatchExtractionResponse(
        batch_id=result["batch_id"],
        created_jobs=[
            BatchExtractionJobInfo(**job) for job in result["created_jobs"]
        ],
        skipped_saes=[
            BatchExtractionSkippedInfo(**skip) for skip in result["skipped_saes"]
        ],
        total_requested=result["total_requested"],
        total_created=result["total_created"],
        total_skipped=result["total_skipped"],
        # FIRST corpus, not `dataset` — that name is the loop variable from the
        # validation above and would report the LAST corpus under a mixture.
        dataset_id=batch_dataset_ids[0],
        dataset_name=batch_dataset_rows[0].name,
        dataset_names=[row.name for row in batch_dataset_rows],
    )
