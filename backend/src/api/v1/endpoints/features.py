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
    AblationResponse
)


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
    sort_by: str = Query("activation_freq", regex="^(activation_freq|max_activation|feature_id|name|category)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
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
    sort_by: str = Query("activation_freq", regex="^(activation_freq|max_activation|feature_id|name|category)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    is_favorite: bool = Query(None, description="Filter by favorite status"),
    limit: int = Query(50, ge=1, le=500, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
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
