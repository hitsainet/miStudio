"""
Celery tasks for Neuronpedia export operations.

This module contains Celery tasks for exporting SAEs to Neuronpedia format,
including dashboard data computation (logit lens, histograms, top tokens)
and archive generation.
"""

import logging
import traceback
from typing import Optional, Dict, Any
from datetime import datetime

from .base_task import DatabaseTask
from ..models.neuronpedia_export import NeuronpediaExportJob, ExportStatus
from ..services.neuronpedia_export_service import (
    get_neuronpedia_export_service,
    ExportConfig,
)
from .websocket_emitter import emit_export_progress

logger = logging.getLogger(__name__)


def get_celery_app():
    """Import celery app lazily to avoid circular imports."""
    from ..core.celery_app import celery_app
    return celery_app


class NeuronpediaTask(DatabaseTask):
    """Base class for Neuronpedia export tasks with progress utilities."""

    def update_export_progress(
        self,
        job_id: str,
        progress: float,
        stage: str,
        message: Optional[str] = None,
    ):
        """
        Update export job progress in database and emit WebSocket event.

        Args:
            job_id: Export job ID
            progress: Progress percentage (0-100)
            stage: Current processing stage
            message: Optional status message
        """
        with self.get_db() as db:
            job = db.query(NeuronpediaExportJob).filter_by(id=job_id).first()
            if job:
                job.progress = progress
                job.current_stage = stage
                db.commit()

                # Emit WebSocket event
                emit_export_progress(
                    job_id=job_id,
                    progress=progress,
                    stage=stage,
                    status=job.status,
                    message=message,
                )

    def mark_export_failed(
        self,
        job_id: str,
        error_message: str,
    ):
        """
        Mark export job as failed.

        Args:
            job_id: Export job ID
            error_message: Error message to store
        """
        with self.get_db() as db:
            job = db.query(NeuronpediaExportJob).filter_by(id=job_id).first()
            if job:
                job.status = ExportStatus.FAILED.value
                job.error_message = error_message
                job.completed_at = datetime.utcnow()
                db.commit()

                # Emit WebSocket event
                emit_export_progress(
                    job_id=job_id,
                    progress=job.progress,
                    stage=job.current_stage or "failed",
                    status="failed",
                    message=error_message,
                )


# Get celery app for task registration
celery_app = get_celery_app()


@celery_app.task(
    base=NeuronpediaTask,
    bind=True,
    name="neuronpedia.execute_export",
    max_retries=0,  # Don't retry on failure - exports are expensive
    soft_time_limit=7200,  # 2 hour soft limit
    time_limit=10800,  # 3 hour hard limit
)
def execute_neuronpedia_export(self, job_id: str):
    """
    Execute a Neuronpedia export job.

    This task handles the complete export pipeline:
    1. Computing dashboard data (logit lens, histograms, top tokens)
    2. Generating Neuronpedia JSON files
    3. Creating SAELens-compatible format
    4. Packaging everything into a ZIP archive

    Args:
        job_id: Export job ID to execute

    Returns:
        dict: Export result with status and output path
    """
    logger.info(f"Starting Neuronpedia export job: {job_id}")

    try:
        # Import here to avoid circular imports
        from ..core.database import sync_session_maker

        with sync_session_maker() as db:
            # Get the job
            job = db.query(NeuronpediaExportJob).filter_by(id=job_id).first()
            if not job:
                raise ValueError(f"Export job not found: {job_id}")

            if job.status == ExportStatus.CANCELLED.value:
                logger.info(f"Export job {job_id} was cancelled, skipping")
                return {"status": "cancelled", "job_id": job_id}

            # Mark as started
            job.status = ExportStatus.COMPUTING.value
            job.started_at = datetime.utcnow()
            db.commit()

            # Update progress
            self.update_export_progress(
                job_id=job_id,
                progress=0,
                stage="initializing",
                message="Starting export process",
            )

            # Execute the export using the service
            # Note: We run this synchronously since we're already in a Celery task
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Create an async session for the service
                from ..core.database import async_session_maker

                async def run_export():
                    async with async_session_maker() as async_db:
                        service = get_neuronpedia_export_service()
                        await service.execute_export(async_db, job_id)

                loop.run_until_complete(run_export())
            finally:
                loop.close()

            # Refresh job to get final status
            db.refresh(job)

            logger.info(f"Neuronpedia export job {job_id} completed with status: {job.status}")

            return {
                "status": job.status,
                "job_id": job_id,
                "output_path": job.output_path,
                "feature_count": job.feature_count,
            }

    except Exception as e:
        logger.exception(f"Neuronpedia export job {job_id} failed: {e}")

        # Mark as failed
        self.mark_export_failed(
            job_id=job_id,
            error_message=str(e),
        )

        raise


@celery_app.task(
    base=NeuronpediaTask,
    bind=True,
    name="neuronpedia.compute_dashboard_data",
    max_retries=0,
    soft_time_limit=3600,  # 1 hour soft limit
    time_limit=7200,  # 2 hour hard limit
)
def compute_dashboard_data_task(
    self,
    sae_id: str,
    feature_indices: Optional[list] = None,
    include_logit_lens: bool = True,
    include_histograms: bool = True,
    include_top_tokens: bool = True,
    force_recompute: bool = False,
):
    """
    Compute dashboard data for SAE features.

    This task computes individual dashboard data components without
    creating a full export. Useful for pre-computing data.

    Args:
        sae_id: SAE to compute data for
        feature_indices: Optional list of feature indices to compute
        include_logit_lens: Whether to compute logit lens data
        include_histograms: Whether to compute activation histograms
        include_top_tokens: Whether to compute top activating tokens
        force_recompute: Whether to recompute even if data exists

    Returns:
        dict: Computation results with counts
    """
    logger.info(f"Computing dashboard data for SAE: {sae_id}")

    try:
        import asyncio
        from ..core.database import async_session_maker

        computed = 0

        async def run_computation():
            nonlocal computed

            async with async_session_maker() as db:
                if include_logit_lens:
                    from ..services.logit_lens_service import get_logit_lens_service
                    service = get_logit_lens_service()
                    results = await service.compute_logit_lens_for_sae(
                        db,
                        sae_id,
                        feature_indices,
                        force_recompute=force_recompute,
                    )
                    await service.save_logit_lens_results(db, sae_id, results)
                    computed += len(results)
                    logger.info(f"Computed logit lens for {len(results)} features")

                if include_histograms:
                    from ..services.histogram_service import get_histogram_service
                    service = get_histogram_service()
                    results = await service.compute_histograms_for_sae(
                        db,
                        sae_id,
                        force_recompute=force_recompute,
                    )
                    await service.save_histogram_results(db, sae_id, results)
                    computed = max(computed, len(results))
                    logger.info(f"Computed histograms for {len(results)} features")

                if include_top_tokens:
                    from ..services.token_aggregator_service import get_token_aggregator_service
                    service = get_token_aggregator_service()
                    results = await service.aggregate_tokens_for_sae(
                        db,
                        sae_id,
                        force_recompute=force_recompute,
                    )
                    await service.save_token_aggregation_results(db, sae_id, results)
                    computed = max(computed, len(results))
                    logger.info(f"Computed top tokens for {len(results)} features")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_computation())
        finally:
            loop.close()

        logger.info(f"Dashboard data computation completed for SAE {sae_id}: {computed} features")

        return {
            "status": "completed",
            "sae_id": sae_id,
            "features_computed": computed,
        }

    except Exception as e:
        logger.exception(f"Dashboard data computation failed for SAE {sae_id}: {e}")
        raise
