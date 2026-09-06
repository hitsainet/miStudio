"""
Celery tasks for Neuronpedia export operations.

This module contains Celery tasks for exporting SAEs to Neuronpedia format,
including dashboard data computation (logit lens, histograms, top tokens)
and archive generation.
"""

import logging
import shutil
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
from ..core.cancellation import (
    OperatorCancelled,
    guard_allows,
    is_cancelled,
)
from ..core.clock import utc_now

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

        WRITES NOTHING ONTO A TERMINAL ROW. This writer never sets `status`, so
        it cannot lose a cancellation by overwriting it — it loses it by
        contradicting it: a cancelled export that goes on reporting
        "packaging, 60%" reads to the operator as a cancel that did not take.
        `guard_allows` refuses a progress move on a terminal row for exactly
        this shape.
        """
        with self.get_db() as db:
            job = (
                db.query(NeuronpediaExportJob)
                .filter_by(id=job_id)
                .populate_existing()
                .first()
            )
            if job and not guard_allows(
                "neuronpedia_export", job.status, writes_progress=True
            ):
                logger.info(
                    "Ignoring progress update for export %s — row is already %s",
                    job_id, job.status,
                )
                return
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
        # THROUGH THE GUARD. This wrote FAILED unconditionally, so the bare
        # `except Exception` that calls it would relabel a row the operator had
        # just CANCELLED. terminal -> terminal is still permitted, so a genuine
        # failure after a cancel is recorded; what is refused is dragging a live
        # cancellation into a crash report.
        with self.get_db() as db:
            job = (
                db.query(NeuronpediaExportJob)
                .filter_by(id=job_id)
                .populate_existing()
                .first()
            )
            if job and is_cancelled("neuronpedia_export", job.status):
                logger.info(
                    "Export %s was cancelled; not relabelling it failed", job_id
                )
                return
            if job:
                job.status = ExportStatus.FAILED.value
                job.error_message = error_message
                job.completed_at = utc_now()
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
        from ..core.database import SyncSessionLocal as sync_session_maker

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
            job.started_at = utc_now()
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
                from ..core.database import AsyncSessionLocal as async_session_maker

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

    except OperatorCancelled as cancelled:
        # NO HANDLER EXISTED, so the raise from `_cancel_point` escaped the task
        # and the acks_late message was never acked. `mark_export_failed` is
        # correctly NOT called — the row is already cancelled — but the export's
        # half-built tree has to go, or a cancelled export leaks its JSON files
        # and SAELens output with nothing left to clean them.
        logger.info("Neuronpedia export %s cancelled: %s", job_id, cancelled.detail)
        try:
            # THE PATH THE SERVICE ACTUALLY WRITES TO. R1 invented
            # `data_dir/"neuronpedia_exports"`, which appears nowhere else in
            # the repo — so `exists()` was always False, the rmtree never ran,
            # and the tree leaked exactly as before. Taken from the service so
            # the two cannot drift again.
            from ..services.neuronpedia_export_service import (
                get_neuronpedia_export_service,
            )

            from ..core.config import settings

            exports_dir = get_neuronpedia_export_service()._exports_dir
            if not str(job_id):
                # `Path("/a/b") / "" == Path("/a/b")` — an empty id collapses the
                # target onto the exports root, i.e. every completed archive.
                raise ValueError("refusing to clean up an export with no id")
            for partial in (
                exports_dir / str(job_id),
                exports_dir / f"neuronpedia_export_{job_id}.zip",
            ):
                # Through the MIS-E2E-071 guard, like every other deletion in
                # this change. R2 closed exactly this hole in `model_tasks` and
                # left it open here — the guard refuses the trusted roots and
                # their top-level directories, so a collapsed target cannot
                # take the whole exports tree with it.
                try:
                    target = settings.resolve_deletable_path(str(partial))
                except ValueError as guard_exc:
                    logger.error("Refusing to delete %s: %s", partial, guard_exc)
                    continue
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                    logger.info(
                        "Removed the cancelled export's partial output: %s", target
                    )
        except Exception as cleanup_exc:  # noqa: BLE001 - must not mask the cancel
            logger.warning("Could not remove the partial export tree: %s", cleanup_exc)
        return {"status": "cancelled", "job_id": job_id, "detail": cancelled.detail}

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
        from ..core.database import AsyncSessionLocal as async_session_maker

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
