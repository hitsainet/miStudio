"""
Feature discovery API endpoints.

Provides REST API for feature extraction, search, and management.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.core.deps import get_db
from src.services.extraction_service import ExtractionService
from src.models.feature import Feature
from src.workers.extraction_tasks import delete_extraction_task
from src.services.feature_service import FeatureService
from src.services.analysis_service import AnalysisService
from src.schemas.extraction import (
    ExtractionConfigRequest,
    ExtractionStatusResponse,
    ExtractionListResponse
)
from src.schemas.feature import (
    FeatureSearchRequest,
    FeatureListResponse,
    FeatureDetailResponse,
    FeatureResponse,
    FeatureUpdateRequest,
    FeatureActivationExample,
    LogitLensResponse,
    CorrelationsResponse,
    AblationResponse,
    NLPAnalysisRequest,
    NLPAnalysisStatusResponse,
    NLPAnalysisResultResponse,
    NLPResetRequest,
    NLPControlResponse
)
from src.workers.nlp_analysis_tasks import (
    analyze_features_nlp_task,
    analyze_single_feature_nlp_task
)
from src.models.feature_analysis_cache import FeatureAnalysisCache, AnalysisType


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/trainings/{training_id}/extract-features",
    response_model=ExtractionStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start feature extraction"
)
async def start_feature_extraction(
    training_id: str,
    config: ExtractionConfigRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start a feature extraction job for a completed training.

    Args:
        training_id: ID of the training to extract features from
        config: Extraction configuration (evaluation_samples, top_k_examples)

    Returns:
        ExtractionStatusResponse with job details

    Raises:
        404: Training not found
        409: Active extraction already exists for this training
        422: Training not completed or has no checkpoint
    """
    extraction_service = ExtractionService(db)

    try:
        # Start extraction job
        extraction_job = await extraction_service.start_extraction(
            training_id=training_id,
            config=config.model_dump()
        )

        # Get status to return
        status_dict = await extraction_service.get_extraction_status(training_id)

        if not status_dict:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get extraction status after creation"
            )

        return ExtractionStatusResponse(**status_dict)

    except ValueError as e:
        error_message = str(e)

        # Check for specific error conditions
        if "not found" in error_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        elif "already has an active extraction" in error_message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_message
            )
        else:
            # Must be completed, has checkpoint, etc.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_message
            )


@router.get(
    "/trainings/{training_id}/extraction-status",
    response_model=Optional[ExtractionStatusResponse],
    summary="Get extraction status"
)
async def get_extraction_status(
    training_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the status of the most recent extraction job for a training.

    Args:
        training_id: ID of the training

    Returns:
        ExtractionStatusResponse with status, progress, and statistics, or null if no extraction exists
    """
    extraction_service = ExtractionService(db)

    status_dict = await extraction_service.get_extraction_status(training_id)

    if not status_dict:
        return None

    return ExtractionStatusResponse(**status_dict)


@router.get(
    "/extractions",
    response_model=ExtractionListResponse,
    summary="List all extraction jobs"
)
async def list_extractions(
    status_filter: Optional[str] = Query(None, description="Comma-separated list of statuses to filter by"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a paginated list of all extraction jobs.

    Args:
        status_filter: Optional comma-separated list of statuses (e.g., "queued,extracting")
        limit: Maximum number of results to return (1-100)
        offset: Number of results to skip for pagination

    Returns:
        ExtractionListResponse with list of extraction jobs and metadata
    """
    extraction_service = ExtractionService(db)

    # Parse status filter
    status_list = None
    if status_filter:
        status_list = [s.strip() for s in status_filter.split(",")]

    # Get extractions
    extractions_list, total = await extraction_service.list_extractions(
        status_filter=status_list,
        limit=limit,
        offset=offset
    )

    return ExtractionListResponse(
        data=[ExtractionStatusResponse(**e) for e in extractions_list],
        meta={
            "total": total,
            "limit": limit,
            "offset": offset
        }
    )


@router.post(
    "/trainings/{training_id}/cancel-extraction",
    status_code=status.HTTP_200_OK,
    summary="Cancel extraction"
)
async def cancel_extraction(
    training_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an active extraction job for a training.

    Args:
        training_id: ID of the training

    Returns:
        Success message

    Raises:
        404: No active extraction job found for this training
    """
    extraction_service = ExtractionService(db)

    try:
        await extraction_service.cancel_extraction(training_id)
        return {"message": "Extraction cancelled successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.delete(
    "/extractions/{extraction_id}",
    summary="Delete extraction job"
)
async def delete_extraction(
    extraction_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an extraction job and all associated features.

    For large extractions (>5000 features), deletion is performed in the background
    and returns 202 Accepted. For smaller extractions, deletion is synchronous
    and returns 204 No Content.

    Args:
        extraction_id: ID of the extraction job

    Raises:
        404: Extraction job not found
        409: Cannot delete active extraction (must cancel first)

    Returns:
        204 No Content for sync deletion (small extractions)
        202 Accepted for async deletion (large extractions)
    """
    extraction_service = ExtractionService(db)

    try:
        # Count features to determine if we need background deletion
        result = await db.execute(
            select(func.count(Feature.id)).where(Feature.extraction_job_id == extraction_id)
        )
        feature_count = result.scalar() or 0

        # Use background deletion for large extractions (>5000 features)
        if feature_count > 5000:
            logger.info(f"Large extraction ({feature_count} features) - using background deletion")

            # Queue background deletion task
            delete_extraction_task.delay(extraction_id)

            # Return 202 Accepted to indicate async processing
            return Response(
                content=f"Deletion queued for extraction with {feature_count} features",
                status_code=status.HTTP_202_ACCEPTED
            )
        else:
            logger.info(f"Small extraction ({feature_count} features) - using sync deletion")

            # Perform synchronous deletion for small extractions
            await extraction_service.delete_extraction(extraction_id)

            # Return 204 No Content for successful sync deletion
            return Response(status_code=status.HTTP_204_NO_CONTENT)

    except ValueError as e:
        error_message = str(e)
        if "not found" in error_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        else:
            # Must be active extraction
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_message
            )


@router.get(
    "/trainings/{training_id}/features",
    response_model=FeatureListResponse,
    summary="List and search features"
)
async def list_features(
    training_id: str,
    search: str = Query(None, max_length=500, description="Full-text search query"),
    sort_by: str = Query("activation_freq", pattern="^(activation_freq|max_activation|feature_id|name|category)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    is_favorite: bool = Query(None, description="Filter by favorite status"),
    limit: int = Query(50, ge=1, le=500, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db: AsyncSession = Depends(get_db)
):
    """
    List and search features for a training with filtering, sorting, and pagination.

    Args:
        training_id: ID of the training
        search: Full-text search query on feature name and description
        sort_by: Sort field (activation_freq, max_activation, feature_id)
        sort_order: Sort order (asc, desc)
        is_favorite: Filter by favorite status (None = all)
        limit: Maximum number of results (1-500)
        offset: Number of results to skip

    Returns:
        FeatureListResponse with features, pagination info, and statistics
    """
    feature_service = FeatureService(db)

    # Build search params
    search_params = FeatureSearchRequest(
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        is_favorite=is_favorite,
        limit=limit,
        offset=offset
    )

    return await feature_service.list_features(training_id, search_params)


@router.get(
    "/extractions/{extraction_id}/features",
    response_model=FeatureListResponse,
    summary="List and search features for an extraction"
)
async def list_extraction_features(
    extraction_id: str,
    search: str = Query(None, max_length=500, description="Full-text search query"),
    sort_by: str = Query("activation_freq", pattern="^(activation_freq|max_activation|feature_id|name|category)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    is_favorite: bool = Query(None, description="Filter by favorite status"),
    limit: int = Query(50, ge=1, le=500, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    min_activation_freq: float = Query(None, ge=0, le=100, description="Minimum activation frequency (0-100)"),
    max_activation_freq: float = Query(None, ge=0, le=100, description="Maximum activation frequency (0-100)"),
    min_max_activation: float = Query(None, ge=0, description="Minimum max activation value"),
    max_max_activation: float = Query(None, ge=0, description="Maximum max activation value"),
    db: AsyncSession = Depends(get_db)
):
    """
    List and search features for a specific extraction job with filtering, sorting, and pagination.

    Args:
        extraction_id: ID of the extraction job
        search: Full-text search query on feature name and description
        sort_by: Sort field (activation_freq, max_activation, feature_id)
        sort_order: Sort order (asc, desc)
        is_favorite: Filter by favorite status (None = all)
        limit: Maximum number of results (1-500)
        offset: Number of results to skip
        min_activation_freq: Minimum activation frequency filter (0-100)
        max_activation_freq: Maximum activation frequency filter (0-100)
        min_max_activation: Minimum max activation value filter
        max_max_activation: Maximum max activation value filter

    Returns:
        FeatureListResponse with features, pagination info, and statistics
    """
    feature_service = FeatureService(db)

    # Build search params
    search_params = FeatureSearchRequest(
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        is_favorite=is_favorite,
        limit=limit,
        offset=offset,
        min_activation_freq=min_activation_freq,
        max_activation_freq=max_activation_freq,
        min_max_activation=min_max_activation,
        max_max_activation=max_max_activation
    )

    return await feature_service.list_features_by_extraction(extraction_id, search_params)


@router.get(
    "/features/{feature_id}",
    response_model=FeatureDetailResponse,
    summary="Get feature details"
)
async def get_feature_detail(
    feature_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a feature.

    Args:
        feature_id: ID of the feature

    Returns:
        FeatureDetailResponse with all feature metadata and computed fields

    Raises:
        404: Feature not found
    """
    feature_service = FeatureService(db)

    feature_detail = await feature_service.get_feature_detail(feature_id)

    if not feature_detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature {feature_id} not found"
        )

    return feature_detail


@router.get(
    "/trainings/{training_id}/features/by-index/{feature_idx}",
    response_model=dict,
    summary="Lookup feature ID by index"
)
async def get_feature_id_by_index(
    training_id: str,
    feature_idx: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Look up feature ID by training_id and feature index (neuron_index).

    This endpoint is used when a feature was added by manual index input
    and we need to fetch its database feature_id for viewing details.

    Args:
        training_id: ID of the training
        feature_idx: Feature index (neuron_index) in the SAE

    Returns:
        {"feature_id": "..."} if found, {"feature_id": null} if not found
    """
    feature_service = FeatureService(db)

    feature_id = await feature_service.get_feature_by_index(training_id, feature_idx)

    return {"feature_id": feature_id}


@router.patch(
    "/features/{feature_id}",
    response_model=FeatureResponse,
    summary="Update feature metadata"
)
async def update_feature(
    feature_id: str,
    updates: FeatureUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update feature metadata (name, description, notes).

    Args:
        feature_id: ID of the feature to update
        updates: Fields to update

    Returns:
        Updated FeatureResponse

    Raises:
        404: Feature not found
    """
    feature_service = FeatureService(db)

    updated_feature = await feature_service.update_feature(feature_id, updates)

    if not updated_feature:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature {feature_id} not found"
        )

    return updated_feature


@router.post(
    "/features/{feature_id}/favorite",
    response_model=dict,
    summary="Toggle feature favorite status"
)
async def toggle_favorite(
    feature_id: str,
    is_favorite: bool = Query(..., description="New favorite status"),
    db: AsyncSession = Depends(get_db)
):
    """
    Toggle favorite status for a feature.

    Args:
        feature_id: ID of the feature
        is_favorite: New favorite status

    Returns:
        Dict with new is_favorite value

    Raises:
        404: Feature not found
    """
    feature_service = FeatureService(db)

    new_favorite_status = await feature_service.toggle_favorite(feature_id, is_favorite)

    if new_favorite_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature {feature_id} not found"
        )

    return {"is_favorite": new_favorite_status}


@router.get(
    "/features/{feature_id}/examples",
    response_model=List[FeatureActivationExample],
    summary="Get max-activating examples"
)
async def get_feature_examples(
    feature_id: str,
    limit: int = Query(100, ge=10, le=1000, description="Number of examples to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get max-activating examples for a feature.

    Args:
        feature_id: ID of the feature
        limit: Maximum number of examples to return (10-1000)

    Returns:
        List of FeatureActivationExample with tokens and activations
    """
    feature_service = FeatureService(db)

    examples = await feature_service.get_feature_examples(feature_id, limit)

    return examples


@router.get(
    "/features/{feature_id}/token-analysis",
    summary="Get token analysis for feature"
)
async def get_token_analysis(
    feature_id: str,
    apply_filters: bool = Query(True, description="Master switch for all filtering"),
    filter_special: bool = Query(True, description="Filter special tokens (<s>, </s>, etc.)"),
    filter_single_char: bool = Query(True, description="Filter single character tokens"),
    filter_punctuation: bool = Query(True, description="Filter pure punctuation"),
    filter_numbers: bool = Query(True, description="Filter pure numeric tokens"),
    filter_fragments: bool = Query(True, description="Filter word fragments (BPE subwords)"),
    filter_stop_words: bool = Query(False, description="Filter common stop words (the, and, is, etc.)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get token analysis for a feature's activation examples with granular filter control.

    Analyzes all tokens from the feature's max-activating examples,
    applies filtering to remove junk tokens based on selected categories,
    and returns statistics and a ranked token list.

    Args:
        feature_id: ID of the feature
        apply_filters: Master switch for all filtering (default: True)
        filter_special: Filter special tokens (<s>, </s>, <pad>, <unk>, etc.)
        filter_single_char: Filter single character tokens
        filter_punctuation: Filter pure punctuation tokens
        filter_numbers: Filter pure numeric tokens
        filter_fragments: Filter word fragments (BPE subwords like "tion", "ing")
        filter_stop_words: Filter common stop words (a, the, and, is, it, etc.)

    Returns:
        Dictionary with:
        - summary: Statistics (total_examples, original_token_count, filtered_token_count,
                  junk_removed, total_token_occurrences, filtered_token_occurrences,
                  diversity_percent, filter_stats per category)
        - tokens: List of tokens with rank, token, count, and percentage

    Raises:
        404: Feature not found

    Example:
        # Filter everything except stop words
        GET /features/feat_123/token-analysis?filter_stop_words=false

        # Only filter special tokens and punctuation
        GET /features/feat_123/token-analysis?filter_single_char=false&filter_numbers=false&filter_fragments=false
    """
    feature_service = FeatureService(db)

    analysis = await feature_service.get_feature_token_analysis(
        feature_id,
        apply_filters=apply_filters,
        filter_special=filter_special,
        filter_single_char=filter_single_char,
        filter_punctuation=filter_punctuation,
        filter_numbers=filter_numbers,
        filter_fragments=filter_fragments,
        filter_stop_words=filter_stop_words
    )

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature {feature_id} not found"
        )

    return analysis


@router.get(
    "/features/{feature_id}/logit-lens",
    response_model=LogitLensResponse,
    summary="Get logit lens analysis"
)
async def get_logit_lens(
    feature_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get logit lens analysis for a feature.

    Analyzes what the feature contributes to the model's output predictions
    by passing a synthetic activation through the SAE decoder and model head.

    Returns top predicted tokens and an interpretation of the feature's role.

    Args:
        feature_id: ID of the feature

    Returns:
        LogitLensResponse with top tokens, probabilities, and interpretation

    Raises:
        404: Feature not found
        500: Analysis computation error
    """
    analysis_service = AnalysisService(db)

    result = await analysis_service.calculate_logit_lens(feature_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature {feature_id} not found"
        )

    return result


@router.get(
    "/features/{feature_id}/correlations",
    response_model=CorrelationsResponse,
    summary="Get feature correlations"
)
async def get_correlations(
    feature_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get correlation analysis for a feature.

    Finds features with similar activation patterns by computing
    Pearson correlation coefficients on activation vectors.

    Returns up to 10 features with correlation > 0.5.

    Args:
        feature_id: ID of the feature

    Returns:
        CorrelationsResponse with list of correlated features

    Raises:
        404: Feature not found
        500: Analysis computation error
    """
    analysis_service = AnalysisService(db)

    result = await analysis_service.calculate_correlations(feature_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature {feature_id} not found"
        )

    return result


@router.get(
    "/features/{feature_id}/ablation",
    response_model=AblationResponse,
    summary="Get ablation analysis"
)
async def get_ablation(
    feature_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get ablation analysis for a feature.

    Measures the feature's importance by comparing model performance
    with the feature active vs. ablated (set to zero).

    Returns perplexity delta and normalized impact score (0-1).

    Args:
        feature_id: ID of the feature

    Returns:
        AblationResponse with perplexity metrics and impact score

    Raises:
        404: Feature not found
        500: Analysis computation error
    """
    analysis_service = AnalysisService(db)

    result = await analysis_service.calculate_ablation(feature_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature {feature_id} not found"
        )

    return result


@router.post(
    "/analysis/cleanup",
    status_code=status.HTTP_200_OK,
    summary="Clean up GPU memory from analysis operations"
)
async def cleanup_analysis_gpu():
    """
    Clean up GPU memory used by analysis operations (logit lens, etc.).

    Call this endpoint when closing the feature detail modal to free
    GPU memory that may still be allocated from model loading.

    Returns:
        Dict with cleanup status and memory freed
    """
    import torch
    import gc

    result = {
        "cleaned": False,
        "vram_before_gb": 0.0,
        "vram_after_gb": 0.0,
        "vram_freed_gb": 0.0,
    }

    try:
        if torch.cuda.is_available():
            # Get memory before cleanup
            result["vram_before_gb"] = round(
                torch.cuda.memory_allocated() / (1024**3), 2
            )

            # Force garbage collection first
            gc.collect()

            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()

            # Synchronize to ensure cleanup is complete
            torch.cuda.synchronize()

            # Get memory after cleanup
            result["vram_after_gb"] = round(
                torch.cuda.memory_allocated() / (1024**3), 2
            )
            result["vram_freed_gb"] = round(
                result["vram_before_gb"] - result["vram_after_gb"], 2
            )
            result["cleaned"] = True

            logger.info(
                f"Analysis GPU cleanup: freed {result['vram_freed_gb']} GB "
                f"({result['vram_before_gb']} -> {result['vram_after_gb']} GB)"
            )
        else:
            result["message"] = "CUDA not available"

    except Exception as e:
        logger.error(f"Error during analysis GPU cleanup: {e}")
        result["error"] = str(e)

    return result


# =============================================================================
# NLP Analysis Endpoints
# =============================================================================


@router.post(
    "/extractions/{extraction_id}/analyze-nlp",
    response_model=NLPAnalysisStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger NLP analysis for extraction"
)
async def trigger_nlp_analysis(
    extraction_id: str,
    request: NLPAnalysisRequest = NLPAnalysisRequest(),
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger NLP analysis for all features in an extraction job.

    This queues a background Celery task to compute NLP analysis (POS tagging,
    NER, context patterns, semantic clusters) for each feature's activation
    examples. Results are cached in the feature_analysis_cache table.

    Progress can be monitored via WebSocket channel: `nlp_analysis/{extraction_id}`

    Args:
        extraction_id: ID of the extraction job
        request: Optional configuration (feature_ids to analyze, batch_size)

    Returns:
        NLPAnalysisStatusResponse with task_id for tracking

    Raises:
        404: Extraction job not found
    """
    from src.models.extraction_job import ExtractionJob

    # Verify extraction exists
    result = await db.execute(
        select(ExtractionJob).where(ExtractionJob.id == extraction_id)
    )
    extraction = result.scalar_one_or_none()

    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction job {extraction_id} not found"
        )

    # Queue the Celery task
    task = analyze_features_nlp_task.delay(
        extraction_job_id=extraction_id,
        feature_ids=request.feature_ids,
        batch_size=request.batch_size,
        force_reprocess=request.force_reprocess
    )

    mode = "restart from scratch" if request.force_reprocess else "resume/start"
    return NLPAnalysisStatusResponse(
        task_id=task.id,
        extraction_job_id=extraction_id,
        status="queued",
        message=f"NLP analysis queued ({mode}) for extraction {extraction_id}"
    )


@router.post(
    "/extractions/{extraction_id}/cancel-nlp",
    response_model=NLPControlResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel NLP analysis for extraction"
)
async def cancel_nlp_analysis(
    extraction_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an in-progress NLP analysis job.

    Sets the nlp_status to 'cancelled', preserving any progress already made.
    Features that have already been analyzed will retain their NLP analysis.
    The task will detect the cancellation on its next batch check and exit cleanly.

    Args:
        extraction_id: ID of the extraction job

    Returns:
        NLPControlResponse with cancellation status

    Raises:
        404: Extraction job not found
        400: NLP analysis is not currently running
    """
    from src.models.extraction_job import ExtractionJob

    # Get extraction job
    result = await db.execute(
        select(ExtractionJob).where(ExtractionJob.id == extraction_id)
    )
    extraction = result.scalar_one_or_none()

    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction job {extraction_id} not found"
        )

    # Store previous status
    previous_status = extraction.nlp_status
    previous_progress = extraction.nlp_progress

    # Only allow cancellation if currently processing
    if previous_status != "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel NLP analysis: status is '{previous_status}', not 'processing'"
        )

    # Update status to cancelled
    extraction.nlp_status = "cancelled"
    extraction.nlp_error_message = "Cancelled by user"
    await db.commit()

    logger.info(f"NLP analysis cancelled for extraction {extraction_id} at {previous_progress:.1%} progress")

    return NLPControlResponse(
        extraction_job_id=extraction_id,
        action="cancelled",
        previous_status=previous_status,
        previous_progress=previous_progress,
        message=f"NLP analysis cancelled. Progress preserved at {previous_progress:.1%} ({extraction.nlp_processed_count} features processed)"
    )


@router.post(
    "/extractions/{extraction_id}/reset-nlp",
    response_model=NLPControlResponse,
    status_code=status.HTTP_200_OK,
    summary="Reset NLP analysis status for extraction"
)
async def reset_nlp_analysis(
    extraction_id: str,
    request: NLPResetRequest = NLPResetRequest(),
    db: AsyncSession = Depends(get_db)
):
    """
    Reset the NLP analysis status for an extraction job.

    This allows re-running NLP analysis from scratch or resuming from where it left off.

    If clear_feature_analysis=False (default):
        - Resets nlp_status to null (ready to restart)
        - Preserves NLP analysis already computed on individual features
        - Re-running analyze-nlp will RESUME (skip already processed features)

    If clear_feature_analysis=True:
        - Resets nlp_status to null
        - Clears NLP analysis from all features (start from scratch)
        - Re-running analyze-nlp will process ALL features

    Args:
        extraction_id: ID of the extraction job
        request: Reset options (clear_feature_analysis flag)

    Returns:
        NLPControlResponse with reset status

    Raises:
        404: Extraction job not found
        400: Cannot reset while NLP analysis is actively processing
    """
    from src.models.extraction_job import ExtractionJob

    # Get extraction job
    result = await db.execute(
        select(ExtractionJob).where(ExtractionJob.id == extraction_id)
    )
    extraction = result.scalar_one_or_none()

    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction job {extraction_id} not found"
        )

    # Store previous status
    previous_status = extraction.nlp_status
    previous_progress = extraction.nlp_progress

    # Don't allow reset while actively processing (use cancel first)
    if previous_status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot reset while NLP analysis is processing. Use cancel-nlp first."
        )

    features_affected = 0

    # Optionally clear feature-level NLP analysis
    if request.clear_feature_analysis:
        # Update all features for this extraction to clear NLP analysis
        from sqlalchemy import update
        stmt = (
            update(Feature)
            .where(Feature.extraction_job_id == extraction_id)
            .values(nlp_analysis=None, nlp_processed_at=None)
        )
        result = await db.execute(stmt)
        features_affected = result.rowcount

    # Reset extraction job NLP status
    extraction.nlp_status = None
    extraction.nlp_progress = None
    extraction.nlp_processed_count = None
    extraction.nlp_error_message = None
    await db.commit()

    action_detail = "with feature analysis cleared" if request.clear_feature_analysis else "preserving feature analysis"
    logger.info(f"NLP analysis reset for extraction {extraction_id} {action_detail}")

    return NLPControlResponse(
        extraction_job_id=extraction_id,
        action="reset",
        previous_status=previous_status,
        previous_progress=previous_progress,
        features_affected=features_affected if request.clear_feature_analysis else None,
        message=f"NLP analysis status reset {action_detail}. Ready to restart."
    )


@router.post(
    "/features/{feature_id}/analyze-nlp",
    response_model=NLPAnalysisStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger NLP analysis for single feature"
)
async def trigger_single_feature_nlp_analysis(
    feature_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger NLP analysis for a single feature.

    Queues a background task to compute NLP analysis for one feature.
    Useful for re-computing analysis or analyzing newly added features.

    Args:
        feature_id: ID of the feature

    Returns:
        NLPAnalysisStatusResponse with task_id for tracking

    Raises:
        404: Feature not found
    """
    # Verify feature exists
    result = await db.execute(
        select(Feature).where(Feature.id == feature_id)
    )
    feature = result.scalar_one_or_none()

    if not feature:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature {feature_id} not found"
        )

    # Queue the Celery task
    task = analyze_single_feature_nlp_task.delay(feature_id=feature_id)

    return NLPAnalysisStatusResponse(
        task_id=task.id,
        extraction_job_id=feature.extraction_job_id,
        status="queued",
        message=f"NLP analysis queued for feature {feature_id}"
    )


@router.get(
    "/features/{feature_id}/nlp-analysis",
    response_model=Optional[NLPAnalysisResultResponse],
    summary="Get cached NLP analysis"
)
async def get_nlp_analysis(
    feature_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get cached NLP analysis for a feature.

    Returns the pre-computed NLP analysis if available and not expired.
    Analysis includes POS tagging, NER, context patterns, semantic clusters,
    and a formatted summary suitable for LLM prompt inclusion.

    Args:
        feature_id: ID of the feature

    Returns:
        NLPAnalysisResultResponse if cached, null if not available

    Raises:
        404: Feature not found
    """
    from datetime import datetime, timezone

    # Verify feature exists
    result = await db.execute(
        select(Feature).where(Feature.id == feature_id)
    )
    feature = result.scalar_one_or_none()

    if not feature:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature {feature_id} not found"
        )

    # Check for cached analysis
    result = await db.execute(
        select(FeatureAnalysisCache).where(
            FeatureAnalysisCache.feature_id == feature_id,
            FeatureAnalysisCache.analysis_type == AnalysisType.NLP_ANALYSIS,
            FeatureAnalysisCache.expires_at > datetime.now(timezone.utc)
        )
    )
    cache_entry = result.scalar_one_or_none()

    if not cache_entry:
        return None

    # Return the cached result
    analysis_data = cache_entry.result
    return NLPAnalysisResultResponse(
        feature_id=feature_id,
        prime_token_analysis=analysis_data.get("prime_token_analysis", {}),
        context_patterns=analysis_data.get("context_patterns", {}),
        activation_stats=analysis_data.get("activation_stats", {}),
        semantic_clusters=analysis_data.get("semantic_clusters", []),
        summary_for_prompt=analysis_data.get("summary_for_prompt", ""),
        computed_at=cache_entry.computed_at
    )
