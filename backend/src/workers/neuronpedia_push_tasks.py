"""
Celery tasks for Neuronpedia push operations.

These tasks handle asynchronous pushing of SAE features to the local
Neuronpedia instance, with real-time progress updates via WebSocket.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from celery import shared_task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.database import AsyncSessionLocal
from ..models.external_sae import ExternalSAE
from ..services.neuronpedia_local_service import (
    LocalPushConfig,
    LocalPushResult,
    NeuronpediaLocalPushService,
)
from .websocket_emitter import emit_neuronpedia_push_progress

logger = logging.getLogger(__name__)


@dataclass
class PushProgress:
    """Track push progress for WebSocket emission."""
    push_job_id: str
    sae_id: str
    start_time: float
    total_features: int = 0
    features_pushed: int = 0
    activations_pushed: int = 0
    explanations_pushed: int = 0
    current_stage: str = "initializing"

    def emit(self, message: str, status: str = "pushing") -> None:
        """Emit current progress via WebSocket."""
        elapsed = time.time() - self.start_time

        # Calculate progress percentage (simple feature ratio)
        if self.total_features > 0:
            progress = (self.features_pushed / self.total_features) * 100
        else:
            progress = 0

        # Estimate ETA
        eta_seconds = None
        if self.features_pushed > 0 and elapsed > 0:
            features_per_sec = self.features_pushed / elapsed
            remaining = self.total_features - self.features_pushed
            if features_per_sec > 0:
                eta_seconds = remaining / features_per_sec

        emit_neuronpedia_push_progress(
            push_job_id=self.push_job_id,
            sae_id=self.sae_id,
            stage=self.current_stage,
            progress=min(progress, 99) if status == "pushing" else 100,
            status=status,
            message=message,
            features_pushed=self.features_pushed,
            total_features=self.total_features,
            activations_pushed=self.activations_pushed,
            explanations_pushed=self.explanations_pushed,
            elapsed_seconds=elapsed,
            eta_seconds=eta_seconds,
        )


@shared_task(bind=True, name="push_to_neuronpedia_local")
def push_to_neuronpedia_local_task(
    self,
    push_job_id: str,
    sae_id: str,
    include_activations: bool = True,
    include_explanations: bool = True,
    max_activations_per_feature: int = 20,
) -> dict:
    """
    Celery task to push SAE features to local Neuronpedia.

    Args:
        push_job_id: Unique push job identifier
        sae_id: SAE to push
        include_activations: Whether to include activations
        include_explanations: Whether to include explanations
        max_activations_per_feature: Max activations per feature

    Returns:
        dict with push results
    """
    import asyncio

    async def _do_push():
        progress = PushProgress(
            push_job_id=push_job_id,
            sae_id=sae_id,
            start_time=time.time(),
        )

        try:
            # Emit initial progress
            progress.emit("Initializing push...", "pushing")

            async with AsyncSessionLocal() as db:
                # Load SAE to get feature count
                sae = await db.get(ExternalSAE, sae_id)
                if not sae:
                    progress.emit(f"SAE not found: {sae_id}", "failed")
                    emit_neuronpedia_push_progress(
                        push_job_id=push_job_id,
                        sae_id=sae_id,
                        stage="failed",
                        progress=0,
                        status="failed",
                        error=f"SAE not found: {sae_id}",
                    )
                    return {"success": False, "error": f"SAE not found: {sae_id}"}

                progress.total_features = sae.n_features or 0
                progress.current_stage = "creating_source"
                progress.emit(f"Creating Neuronpedia source for {sae.name}...")

                # Create push service
                service = NeuronpediaLocalPushService()

                # Create config
                config = LocalPushConfig(
                    include_activations=include_activations,
                    include_explanations=include_explanations,
                    max_activations_per_feature=max_activations_per_feature,
                )

                # Create a progress callback that emits WebSocket events
                def progress_callback(pct: int, message: str):
                    """Convert service progress to WebSocket emissions."""
                    # Parse stage from message
                    if "Creating model" in message:
                        progress.current_stage = "creating_model"
                    elif "Creating source" in message:
                        progress.current_stage = "creating_source"
                    elif "Loading features" in message:
                        progress.current_stage = "loading_features"
                    elif "Processing feature" in message:
                        progress.current_stage = "pushing_features"
                        # Extract feature count from message
                        try:
                            parts = message.split()
                            for p in parts:
                                if "/" in p:
                                    current, total = p.split("/")
                                    progress.features_pushed = int(current)
                                    progress.total_features = int(total)
                                    break
                        except (ValueError, IndexError):
                            pass
                    elif "complete" in message.lower():
                        progress.current_stage = "completed"

                    progress.emit(message)

                # Perform the push
                result = await service.push_sae_to_local(
                    db=db,
                    sae_id=sae_id,
                    config=config,
                    progress_callback=progress_callback,
                )

                # Close service
                await service.close()

                # Emit final result
                if result.success:
                    progress.features_pushed = result.neurons_created
                    progress.activations_pushed = result.activations_created
                    progress.explanations_pushed = result.explanations_created
                    progress.current_stage = "completed"

                    emit_neuronpedia_push_progress(
                        push_job_id=push_job_id,
                        sae_id=sae_id,
                        stage="completed",
                        progress=100,
                        status="completed",
                        message="Push completed successfully",
                        features_pushed=result.neurons_created,
                        total_features=progress.total_features,
                        activations_pushed=result.activations_created,
                        explanations_pushed=result.explanations_created,
                        elapsed_seconds=time.time() - progress.start_time,
                    )

                    return {
                        "success": True,
                        "model_id": result.model_id,
                        "source_id": result.source_id,
                        "neurons_created": result.neurons_created,
                        "activations_created": result.activations_created,
                        "explanations_created": result.explanations_created,
                        "neuronpedia_url": result.neuronpedia_url,
                    }
                else:
                    emit_neuronpedia_push_progress(
                        push_job_id=push_job_id,
                        sae_id=sae_id,
                        stage="failed",
                        progress=0,
                        status="failed",
                        message=result.error_message or "Unknown error",
                        error=result.error_message,
                    )

                    return {
                        "success": False,
                        "error": result.error_message,
                    }

        except Exception as e:
            logger.exception(f"Neuronpedia push task failed: {e}")
            emit_neuronpedia_push_progress(
                push_job_id=push_job_id,
                sae_id=sae_id,
                stage="failed",
                progress=0,
                status="failed",
                message=str(e),
                error=str(e),
            )
            return {"success": False, "error": str(e)}

    # Run the async push
    return asyncio.run(_do_push())
