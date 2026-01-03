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
        --max-tasks-per-child=1 \
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
        # CRITICAL: Check for zombie processes holding GPU memory
        # This is a common issue when previous tasks were killed unexpectedly
        try:
            import subprocess
            gpu_apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if gpu_apps.returncode == 0 and gpu_apps.stdout.strip():
                # Check if any process holding GPU memory is a zombie
                import os
                for line in gpu_apps.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split(",")
                        if len(parts) >= 2:
                            pid = int(parts[0].strip())
                            mem_mb = int(parts[1].strip())
                            # Check if this PID is a zombie
                            try:
                                with open(f"/proc/{pid}/status", "r") as f:
                                    status = f.read()
                                    if "State:\tZ" in status:
                                        raise RuntimeError(
                                            f"Zombie process {pid} is holding {mem_mb}MB GPU memory. "
                                            f"A system reboot is required to free this memory. "
                                            f"Steering cannot proceed until GPU memory is available."
                                        )
                            except FileNotFoundError:
                                pass  # Process doesn't exist, nvidia-smi data may be stale
        except subprocess.TimeoutExpired:
            logger.warning(f"[Steering Task {task_id}] nvidia-smi timeout during zombie check")
        except RuntimeError:
            raise  # Re-raise zombie detection errors
        except Exception as e:
            logger.warning(f"[Steering Task {task_id}] Zombie check failed: {e}")

        # CRITICAL: Clear GPU state at task start to prevent orphan context issues
        # This ensures we start with a clean GPU regardless of previous task state
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Force garbage collection to release any lingering tensors
                import gc
                gc.collect()
                logger.info(f"[Steering Task {task_id}] GPU cache cleared at task start")
        except Exception as e:
            logger.warning(f"[Steering Task {task_id}] Failed to clear GPU cache: {e}")

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
        error_str = str(e)
        emit_steering_progress(
            task_id=task_id,
            percent=-1,
            message=f"Failed: {error_str[:200]}" if len(error_str) > 200 else f"Failed: {error_str}",
            error=error_str,  # Full error preserved for frontend error display
        )
        raise

    finally:
        # =================================================================
        # Cleanup: Clear hooks and reset state (keep model cached)
        # =================================================================
        # IMPORTANT: Do NOT unload models - models loaded with device_map="auto"
        # use accelerate hooks, and unloading them corrupts state.
        # Instead: clear hooks, reset model state, clean GPU cache.
        # =================================================================
        try:
            import torch
            import gc

            # 1. Clear hooks from cached models (but keep them loaded)
            try:
                from ..services.steering_service import get_steering_service
                service = get_steering_service()

                hooks_cleared = 0
                for mid, (model, _) in list(service._loaded_models.items()):
                    try:
                        cleared = service._clear_all_model_hooks(model)
                        hooks_cleared += cleared
                        service._reset_model_state(model)
                    except Exception:
                        pass

                if hooks_cleared > 0:
                    logger.info(f"[Steering Task {task_id}] Cleared {hooks_cleared} hooks")
            except Exception as e:
                logger.warning(f"[Steering Task {task_id}] Hook cleanup failed: {e}")

            # 2. Reset watchdog state
            try:
                from ..services.steering_service import _generation_watchdog
                if _generation_watchdog is not None:
                    _generation_watchdog._generation_active = False
                    _generation_watchdog._generation_start = None
            except Exception:
                pass

            # 3. GC and GPU cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info(f"[Steering Task {task_id}] Cleanup finished")

        except Exception as cleanup_error:
            logger.warning(f"[Steering Task {task_id}] Cleanup failed: {cleanup_error}")


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
        # CRITICAL: Check for zombie processes holding GPU memory
        try:
            import subprocess
            gpu_apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if gpu_apps.returncode == 0 and gpu_apps.stdout.strip():
                import os
                for line in gpu_apps.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split(",")
                        if len(parts) >= 2:
                            pid = int(parts[0].strip())
                            mem_mb = int(parts[1].strip())
                            try:
                                with open(f"/proc/{pid}/status", "r") as f:
                                    status = f.read()
                                    if "State:\tZ" in status:
                                        raise RuntimeError(
                                            f"Zombie process {pid} is holding {mem_mb}MB GPU memory. "
                                            f"A system reboot is required to free this memory."
                                        )
                            except FileNotFoundError:
                                pass
        except subprocess.TimeoutExpired:
            logger.warning(f"[Sweep Task {task_id}] nvidia-smi timeout during zombie check")
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"[Sweep Task {task_id}] Zombie check failed: {e}")

        # CRITICAL: Clear GPU state at task start to prevent orphan context issues
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info(f"[Sweep Task {task_id}] GPU cache cleared at task start")
        except Exception as e:
            logger.warning(f"[Sweep Task {task_id}] Failed to clear GPU cache: {e}")

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
        error_str = str(e)
        emit_steering_progress(
            task_id=task_id,
            percent=-1,
            message=f"Failed: {error_str[:200]}" if len(error_str) > 200 else f"Failed: {error_str}",
            error=error_str,  # Full error preserved for frontend error display
        )
        raise

    finally:
        # =================================================================
        # Cleanup: Clear hooks and reset state (keep model cached)
        # =================================================================
        try:
            import torch
            import gc

            try:
                from ..services.steering_service import get_steering_service, _generation_watchdog
                service = get_steering_service()

                hooks_cleared = 0
                for mid, (model, _) in list(service._loaded_models.items()):
                    try:
                        cleared = service._clear_all_model_hooks(model)
                        hooks_cleared += cleared
                        service._reset_model_state(model)
                    except Exception:
                        pass

                if hooks_cleared > 0:
                    logger.info(f"[Sweep Task {task_id}] Cleared {hooks_cleared} hooks")
            except Exception as e:
                logger.warning(f"[Sweep Task {task_id}] Hook cleanup failed: {e}")

            try:
                from ..services.steering_service import _generation_watchdog
                if _generation_watchdog is not None:
                    _generation_watchdog._generation_active = False
                    _generation_watchdog._generation_start = None
            except Exception:
                pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info(f"[Sweep Task {task_id}] Cleanup finished")

        except Exception as cleanup_error:
            logger.warning(f"[Sweep Task {task_id}] Cleanup failed: {cleanup_error}")


@celery_app.task(
    bind=True,
    name="steering.cleanup",
    queue="steering",
    soft_time_limit=30,
    time_limit=60,
)
def cleanup_steering_gpu(self) -> Dict[str, Any]:
    """
    Release GPU memory held by the steering worker.

    This task runs on the steering queue and unloads all cached models
    and SAEs from GPU memory. Use this when done with steering to free
    VRAM for other tasks.

    Returns:
        Dict with counts of unloaded models and SAEs, plus memory stats.
    """
    import torch
    from ..services.steering_service import get_steering_service

    task_id = self.request.id
    logger.info(f"[Cleanup Task {task_id}] Starting GPU memory cleanup")

    try:
        service = get_steering_service()

        # Get memory before cleanup
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1024**3
        else:
            memory_before = 0

        # Unload all models and SAEs
        result = service.unload_all()

        # Get memory after cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_after = torch.cuda.memory_allocated() / 1024**3
        else:
            memory_after = 0

        memory_freed = memory_before - memory_after

        logger.info(
            f"[Cleanup Task {task_id}] Complete: "
            f"unloaded {result['models_unloaded']} models, "
            f"{result['saes_unloaded']} SAEs, "
            f"freed {memory_freed:.2f}GB VRAM"
        )

        return {
            "success": True,
            "models_unloaded": result["models_unloaded"],
            "saes_unloaded": result["saes_unloaded"],
            "memory_freed_gb": round(memory_freed, 2),
        }

    except Exception as e:
        logger.exception(f"[Cleanup Task {task_id}] Failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "models_unloaded": 0,
            "saes_unloaded": 0,
            "memory_freed_gb": 0,
        }
