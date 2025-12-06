"""
Neuronpedia Export API endpoints.

This module defines REST API endpoints for Neuronpedia export operations including:
- Starting export jobs
- Checking export job status
- Downloading export archives
- Computing dashboard data on-demand
"""

import logging
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from ....core.database import get_db
from ....core.config import settings
from ....models.neuronpedia_export import NeuronpediaExportJob, ExportStatus
from ....schemas.neuronpedia import (
    NeuronpediaExportRequest,
    NeuronpediaExportJobResponse,
    NeuronpediaExportJobStatus,
    NeuronpediaExportJobListResponse,
    ComputeDashboardDataRequest,
    ComputeDashboardDataResponse,
    FeatureDashboardDataResponse,
)
from ....services.neuronpedia_export_service import (
    get_neuronpedia_export_service,
    ExportConfig,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/neuronpedia", tags=["Neuronpedia"])


@router.post("/export", response_model=NeuronpediaExportJobResponse)
async def start_export(
    request: NeuronpediaExportRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Start a new Neuronpedia export job.

    Creates a background job that:
    1. Computes missing dashboard data (logit lens, histograms, top tokens)
    2. Generates Neuronpedia JSON files
    3. Creates SAELens-compatible format
    4. Packages everything into a ZIP archive

    Returns a job_id for status polling.
    """
    service = get_neuronpedia_export_service()

    try:
        # Convert request config to ExportConfig
        config = ExportConfig(
            feature_selection=request.config.feature_selection,
            feature_indices=request.config.feature_indices,
            include_logit_lens=request.config.include_logit_lens,
            include_histograms=request.config.include_histograms,
            include_top_tokens=request.config.include_top_tokens,
            logit_lens_k=request.config.logit_lens_k,
            histogram_bins=request.config.histogram_bins,
            top_tokens_k=request.config.top_tokens_k,
            include_saelens_format=request.config.include_saelens_format,
            include_explanations=request.config.include_explanations,
        )

        job_id = await service.start_export(db, request.sae_id, config)

        # Start background task
        background_tasks.add_task(
            _execute_export_task,
            job_id=job_id,
        )

        return NeuronpediaExportJobResponse(
            job_id=job_id,
            status="pending",
            message="Export job started. Poll /neuronpedia/export/{job_id} for status.",
        )

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception(f"Error starting export: {e}")
        raise HTTPException(500, f"Failed to start export: {str(e)}")


async def _execute_export_task(job_id: str):
    """Background task to execute export."""
    from src.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            service = get_neuronpedia_export_service()
            await service.execute_export(db, job_id)
        except Exception as e:
            logger.exception(f"Export job {job_id} failed: {e}")


@router.get("/export/{job_id}", response_model=NeuronpediaExportJobStatus)
async def get_export_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get status of an export job.

    Returns current progress, stage, and results when complete.
    """
    service = get_neuronpedia_export_service()

    try:
        status = await service.get_job_status(db, job_id)

        # Add download URL if completed
        if status["status"] == "completed" and status["output_path"]:
            status["download_url"] = f"/api/v1/neuronpedia/export/{job_id}/download"

        return NeuronpediaExportJobStatus(**status)

    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception(f"Error getting export status: {e}")
        raise HTTPException(500, str(e))


@router.get("/export/{job_id}/download")
async def download_export(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Download completed export archive.

    Returns the ZIP archive file for completed exports.
    """
    job = await db.get(NeuronpediaExportJob, job_id)
    if not job:
        raise HTTPException(404, f"Export job not found: {job_id}")

    if job.status != ExportStatus.COMPLETED.value:
        raise HTTPException(400, f"Export not completed: {job.status}")

    if not job.output_path:
        raise HTTPException(500, "Export completed but no output file found")

    output_path = settings.resolve_data_path(job.output_path)
    if not output_path.exists():
        raise HTTPException(404, "Export file not found")

    return FileResponse(
        path=output_path,
        filename=output_path.name,
        media_type="application/zip",
    )


@router.post("/export/{job_id}/cancel")
async def cancel_export(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel an in-progress export job.
    """
    service = get_neuronpedia_export_service()

    try:
        cancelled = await service.cancel_export(db, job_id)
        if cancelled:
            return {"message": f"Export job {job_id} cancelled"}
        else:
            return {"message": f"Export job {job_id} could not be cancelled (already completed or failed)"}

    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/export/{job_id}")
async def delete_export(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete an export job and its archive.
    """
    job = await db.get(NeuronpediaExportJob, job_id)
    if not job:
        raise HTTPException(404, f"Export job not found: {job_id}")

    # Delete archive file if exists
    if job.output_path:
        output_path = settings.resolve_data_path(job.output_path)
        if output_path.exists():
            output_path.unlink()

    # Delete job record
    await db.delete(job)
    await db.commit()

    return {"message": f"Export job {job_id} deleted"}


@router.get("/exports", response_model=NeuronpediaExportJobListResponse)
async def list_exports(
    sae_id: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """
    List export jobs with optional filters.
    """
    stmt = select(NeuronpediaExportJob)

    if sae_id:
        stmt = stmt.where(NeuronpediaExportJob.sae_id == sae_id)

    if status:
        stmt = stmt.where(NeuronpediaExportJob.status == status)

    stmt = stmt.order_by(desc(NeuronpediaExportJob.created_at))

    # Get total count
    from sqlalchemy import func
    count_stmt = select(func.count()).select_from(stmt.subquery())
    result = await db.execute(count_stmt)
    total = result.scalar()

    # Get paginated results
    stmt = stmt.offset(skip).limit(limit)
    result = await db.execute(stmt)
    jobs = result.scalars().all()

    job_statuses = []
    for job in jobs:
        status_dict = {
            "id": str(job.id),
            "sae_id": job.sae_id,
            "status": job.status,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "feature_count": job.feature_count,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "output_path": job.output_path,
            "file_size_bytes": job.file_size_bytes,
            "error_message": job.error_message,
        }
        if job.status == ExportStatus.COMPLETED.value and job.output_path:
            status_dict["download_url"] = f"/api/v1/neuronpedia/export/{job.id}/download"
        job_statuses.append(NeuronpediaExportJobStatus(**status_dict))

    return NeuronpediaExportJobListResponse(
        jobs=job_statuses,
        total=total,
    )


# ============================================================================
# Dashboard Data Computation Endpoints
# ============================================================================

@router.post("/compute-dashboard-data", response_model=ComputeDashboardDataResponse)
async def compute_dashboard_data(
    request: ComputeDashboardDataRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Compute dashboard data for features on-demand.

    This computes logit lens, histograms, and top tokens for the specified
    features without creating a full export archive. Useful for populating
    dashboard data before export.
    """
    # For now, compute synchronously for small feature sets
    # TODO: Add background task for large feature sets

    from ....services.logit_lens_service import get_logit_lens_service
    from ....services.histogram_service import get_histogram_service
    from ....services.token_aggregator_service import get_token_aggregator_service

    computed = 0

    try:
        if request.include_logit_lens:
            service = get_logit_lens_service()
            results = await service.compute_logit_lens_for_sae(
                db,
                request.sae_id,
                request.feature_indices,
                force_recompute=request.force_recompute,
            )
            await service.save_logit_lens_results(db, request.sae_id, results)
            computed += len(results)

        if request.include_histograms:
            service = get_histogram_service()
            results = await service.compute_histograms_for_sae(
                db,
                request.sae_id,
                force_recompute=request.force_recompute,
            )
            await service.save_histogram_results(db, request.sae_id, results)
            computed = max(computed, len(results))

        if request.include_top_tokens:
            service = get_token_aggregator_service()
            results = await service.aggregate_tokens_for_sae(
                db,
                request.sae_id,
                force_recompute=request.force_recompute,
            )
            await service.save_token_aggregation_results(db, request.sae_id, results)
            computed = max(computed, len(results))

        return ComputeDashboardDataResponse(
            features_computed=computed,
            status="completed",
            message=f"Computed dashboard data for {computed} features",
        )

    except Exception as e:
        logger.exception(f"Error computing dashboard data: {e}")
        raise HTTPException(500, str(e))
