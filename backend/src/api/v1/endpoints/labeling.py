"""
Feature labeling API endpoints.

Provides REST API for independent semantic labeling of extracted SAE features.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.deps import get_db
from src.services.labeling_service import LabelingService
from src.workers.labeling_tasks import label_features_task
from src.schemas.labeling import (
    LabelingConfigRequest,
    LabelingStatusResponse,
    LabelingListResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/labeling",
    response_model=LabelingStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start feature labeling"
)
async def start_labeling(
    config: LabelingConfigRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start a semantic labeling job for a completed extraction.

    This creates a labeling job and queues it for async processing. Features
    are labeled independently from extraction, allowing re-labeling without
    re-extraction.

    Args:
        config: Labeling configuration (extraction_job_id, labeling_method, etc.)

    Returns:
        LabelingStatusResponse with job details

    Raises:
        404: Extraction not found
        409: Active labeling already exists for this extraction
        422: Extraction not completed or has no features
    """
    labeling_service = LabelingService(db)

    try:
        # Start labeling job (creates record in QUEUED status)
        labeling_job = await labeling_service.start_labeling(
            extraction_job_id=config.extraction_job_id,
            config=config.model_dump()
        )

        # Queue Celery task for async labeling
        task = label_features_task.delay(labeling_job.id)

        # Update with Celery task ID
        labeling_job.celery_task_id = task.id
        await db.commit()
        await db.refresh(labeling_job)

        logger.info(
            f"Started labeling job {labeling_job.id} for extraction "
            f"{config.extraction_job_id} with task {task.id}"
        )

        return LabelingStatusResponse.model_validate(labeling_job)

    except ValueError as e:
        error_message = str(e)

        # Check for specific error conditions
        if "not found" in error_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        elif "already has an active labeling" in error_message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_message
            )
        else:
            # Must be completed, has features, etc.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_message
            )


@router.get(
    "/labeling/{labeling_job_id}",
    response_model=LabelingStatusResponse,
    summary="Get labeling job status"
)
async def get_labeling_status(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the status of a specific labeling job.

    Args:
        labeling_job_id: ID of the labeling job

    Returns:
        LabelingStatusResponse with status, progress, and statistics

    Raises:
        404: Labeling job not found
    """
    labeling_service = LabelingService(db)

    labeling_job = await labeling_service.get_labeling_job(labeling_job_id)

    if not labeling_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Labeling job {labeling_job_id} not found"
        )

    return LabelingStatusResponse.model_validate(labeling_job)


@router.get(
    "/labeling",
    response_model=LabelingListResponse,
    summary="List labeling jobs"
)
async def list_labeling_jobs(
    extraction_job_id: Optional[str] = Query(None, description="Filter by extraction job ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a paginated list of labeling jobs.

    Args:
        extraction_job_id: Optional filter by extraction job ID
        limit: Maximum number of results to return (1-100)
        offset: Number of results to skip for pagination

    Returns:
        LabelingListResponse with list of labeling jobs and metadata
    """
    labeling_service = LabelingService(db)

    # Get labeling jobs
    jobs_list, total = await labeling_service.list_labeling_jobs(
        extraction_job_id=extraction_job_id,
        limit=limit,
        offset=offset
    )

    return LabelingListResponse(
        data=[LabelingStatusResponse.model_validate(job) for job in jobs_list],
        meta={
            "total": total,
            "limit": limit,
            "offset": offset
        }
    )


@router.post(
    "/labeling/{labeling_job_id}/cancel",
    status_code=status.HTTP_200_OK,
    summary="Cancel labeling job"
)
async def cancel_labeling(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an active labeling job.

    Args:
        labeling_job_id: ID of the labeling job to cancel

    Returns:
        Success message

    Raises:
        404: Labeling job not found
        409: Labeling job not in cancellable state
    """
    labeling_service = LabelingService(db)

    try:
        await labeling_service.cancel_labeling_job(labeling_job_id)
        logger.info(f"Cancelled labeling job {labeling_job_id}")
        return {"message": "Labeling job cancelled successfully"}
    except ValueError as e:
        error_message = str(e)
        if "not found" in error_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        else:
            # Cannot cancel due to status
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_message
            )


@router.delete(
    "/labeling/{labeling_job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete labeling job"
)
async def delete_labeling(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a labeling job record.

    This does NOT delete the features or their labels, only the labeling job
    record itself. Feature labels will remain intact.

    If the job is currently active (queued or labeling), it will be automatically
    cancelled before deletion by revoking the Celery task.

    Args:
        labeling_job_id: ID of the labeling job to delete

    Raises:
        404: Labeling job not found
    """
    labeling_service = LabelingService(db)

    try:
        await labeling_service.delete_labeling_job(labeling_job_id)
        logger.info(f"Deleted labeling job {labeling_job_id}")
        return None  # 204 No Content
    except ValueError as e:
        error_message = str(e)
        # Only possible error now is "not found"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_message
        )


@router.post(
    "/extractions/{extraction_id}/label",
    response_model=LabelingStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Label extraction (convenience endpoint)"
)
async def label_extraction(
    extraction_id: str,
    config: LabelingConfigRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Convenience endpoint to start labeling for an extraction.

    This is a shorthand for POST /labeling with extraction_job_id in the body.
    The extraction_id from the URL takes precedence over config.extraction_job_id.

    Args:
        extraction_id: ID of the extraction to label
        config: Labeling configuration (labeling_method, openai_model, etc.)

    Returns:
        LabelingStatusResponse with job details

    Raises:
        404: Extraction not found
        409: Active labeling already exists
        422: Extraction not completed or has no features
    """
    # Override extraction_job_id with URL parameter
    config.extraction_job_id = extraction_id

    # Delegate to main labeling endpoint
    return await start_labeling(config, db)
