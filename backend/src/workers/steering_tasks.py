"""
Celery tasks for steering operations.

These tasks run in a dedicated GPU worker process, providing:
- Process isolation (crashes don't affect API)
- Proper timeout handling via SIGKILL
- Worker recycling to prevent memory leaks

Timeout behavior:
- soft_time_limit: SIGTERM sent, SoftTimeLimitExceeded raised
- time_limit: SIGKILL sent, process terminated, GPU memory released by kernel

Worker configuration (via celery.sh):
    celery -A src.core.celery_app worker \
        -Q steering \
        --pool=solo \
        --concurrency=1 \
        --max-tasks-per-child=50 \
        --loglevel=info \
        --hostname=steering@%h
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from ..core.celery_app import celery_app
from .websocket_emitter import emit_steering_progress

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="steering.compare",
    queue="steering",
    soft_time_limit=150,  # SIGTERM after 150s
    time_limit=180,       # SIGKILL after 180s (guaranteed termination)
    max_retries=0,        # No retries for GPU tasks
    acks_late=True,       # Acknowledge after completion
    reject_on_worker_lost=True,  # Requeue if worker dies
    track_started=True,   # Track when task starts
)
def steering_compare_task(
    self,
    request_dict: Dict[str, Any],
    sae_id: str,
    model_id: str,
    sae_path: str,
    model_path: Optional[str] = None,
    sae_layer: Optional[int] = None,
    sae_d_model: Optional[int] = None,
    sae_n_features: Optional[int] = None,
    sae_architecture: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute steering comparison in isolated worker process.

    This task:
    1. Loads model and SAE to GPU
    2. Generates unsteered baseline
    3. Generates steered outputs for each feature/strength
    4. Emits progress via WebSocket
    5. Returns result dict

    Args:
        request_dict: Serialized SteeringComparisonRequest as dict
        sae_id: SAE database ID
        model_id: Model identifier for HuggingFace loading
        sae_path: Path to SAE weights file
        model_path: Optional local path to model weights
        sae_layer: SAE layer index
        sae_d_model: SAE model dimension
        sae_n_features: Number of SAE features
        sae_architecture: SAE architecture type

    Returns:
        Dict containing steered and unsteered outputs

    Timeout behavior:
        - At 150s: SIGTERM sent, SoftTimeLimitExceeded raised
        - At 180s: SIGKILL sent, process terminated, GPU memory released
    """
    task_id = self.request.id
    logger.info(f"[Steering Task {task_id}] Starting steering comparison")

    try:
        # Emit initial progress
        emit_steering_progress(task_id, 0, "Initializing...")

        # Import service lazily to avoid GPU initialization at module load
        from ..services.steering_service import get_steering_service

        service = get_steering_service()

        # Define progress callback for WebSocket updates
        def progress_callback(percent: int, message: str):
            emit_steering_progress(
                task_id=task_id,
                percent=percent,
                message=message,
            )

        # Run the synchronous steering operation
        result = service.generate_comparison_sync(
            request_dict=request_dict,
            sae_path=sae_path,
            model_id=model_id,
            model_path=model_path,
            sae_layer=sae_layer,
            sae_d_model=sae_d_model,
            sae_n_features=sae_n_features,
            sae_architecture=sae_architecture,
            progress_callback=progress_callback,
        )

        # Emit completion
        emit_steering_progress(
            task_id=task_id,
            percent=100,
            message="Complete",
            result=result,
        )

        logger.info(f"[Steering Task {task_id}] Completed successfully")
        return result

    except SoftTimeLimitExceeded:
        logger.warning(f"[Steering Task {task_id}] Soft time limit exceeded, cleaning up...")
        emit_steering_progress(
            task_id=task_id,
            percent=-1,
            message="Timeout - cleaning up",
            error="Task exceeded time limit (150s). Try reducing max_new_tokens or using fewer features.",
        )

        # Attempt graceful cleanup before SIGKILL
        try:
            from ..services.steering_service import get_steering_service
            service = get_steering_service()
            service.cleanup_gpu()
        except Exception as e:
            logger.error(f"[Steering Task {task_id}] Cleanup failed: {e}")

        raise  # Re-raise to mark task as failed

    except Exception as e:
        logger.exception(f"[Steering Task {task_id}] Failed: {e}")
        emit_steering_progress(
            task_id=task_id,
            percent=-1,
            message=f"Failed: {str(e)[:100]}",
            error=str(e),
        )
        raise


@celery_app.task(
    bind=True,
    name="steering.sweep",
    queue="steering",
    soft_time_limit=300,  # Sweeps take longer - 5 min soft limit
    time_limit=360,       # 6 min hard limit
    max_retries=0,
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True,
)
def steering_sweep_task(
    self,
    request_dict: Dict[str, Any],
    sae_id: str,
    model_id: str,
    sae_path: str,
    model_path: Optional[str] = None,
    sae_layer: Optional[int] = None,
    sae_d_model: Optional[int] = None,
    sae_n_features: Optional[int] = None,
    sae_architecture: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute strength sweep in isolated worker process.

    Tests a single feature at multiple strength values and returns
    outputs for comparison. Useful for finding optimal steering strength.

    Args:
        request_dict: Serialized SteeringStrengthSweepRequest as dict
        sae_id: SAE database ID
        model_id: Model identifier for HuggingFace loading
        sae_path: Path to SAE weights file
        model_path: Optional local path to model weights
        sae_layer: SAE layer index
        sae_d_model: SAE model dimension
        sae_n_features: Number of SAE features
        sae_architecture: SAE architecture type

    Returns:
        Dict containing sweep results at different strength values
    """
    task_id = self.request.id
    logger.info(f"[Sweep Task {task_id}] Starting strength sweep")

    try:
        emit_steering_progress(task_id, 0, "Initializing sweep...")

        from ..services.steering_service import get_steering_service

        service = get_steering_service()

        def progress_callback(percent: int, message: str):
            emit_steering_progress(
                task_id=task_id,
                percent=percent,
                message=message,
            )

        result = service.generate_strength_sweep_sync(
            request_dict=request_dict,
            sae_path=sae_path,
            model_id=model_id,
            model_path=model_path,
            sae_layer=sae_layer,
            sae_d_model=sae_d_model,
            sae_n_features=sae_n_features,
            sae_architecture=sae_architecture,
            progress_callback=progress_callback,
        )

        emit_steering_progress(
            task_id=task_id,
            percent=100,
            message="Complete",
            result=result,
        )

        logger.info(f"[Sweep Task {task_id}] Completed successfully")
        return result

    except SoftTimeLimitExceeded:
        logger.warning(f"[Sweep Task {task_id}] Soft time limit exceeded")
        emit_steering_progress(
            task_id=task_id,
            percent=-1,
            message="Timeout",
            error="Task exceeded time limit (300s). Try reducing number of strength values or max_new_tokens.",
        )
        try:
            from ..services.steering_service import get_steering_service
            get_steering_service().cleanup_gpu()
        except:
            pass
        raise

    except Exception as e:
        logger.exception(f"[Sweep Task {task_id}] Failed: {e}")
        emit_steering_progress(
            task_id=task_id,
            percent=-1,
            message=f"Failed: {str(e)[:100]}",
            error=str(e),
        )
        raise
