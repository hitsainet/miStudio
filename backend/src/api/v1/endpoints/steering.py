"""
Steering API endpoints.

This module defines REST API endpoints for model steering operations including:
- Generating steered and unsteered text comparisons
- Running strength sweeps to test different steering intensities
- Managing steering experiments

Resilience features:
- Circuit breaker: Temporarily disables steering after repeated failures
- Concurrency limiter: Ensures only one steering request at a time
- Process isolation: Timeout and cleanup for stuck operations
"""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....core.config import settings
from ....models.external_sae import SAEStatus
from ....schemas.steering import (
    SteeringComparisonRequest,
    SteeringComparisonResponse,
    SteeringStrengthSweepRequest,
    StrengthSweepResponse,
    SteeringTaskResponse,
    SteeringResultResponse,
    SteeringTaskStatus,
    SteeringCancelResponse,
    SteeringExperimentSaveRequest,
)
from ....services.sae_manager_service import SAEManagerService
from ....services.steering_service import get_steering_service
from ....services.model_service import ModelService
from ....services.steering_resilience import (
    get_circuit_breaker,
    get_concurrency_limiter,
    get_process_isolation,
    get_resilience_status,
    reset_resilience,
)

logger = logging.getLogger(__name__)

# Steering configuration - now configurable via settings
STEERING_TIMEOUT_SECONDS = settings.steering_timeout_seconds
RATE_LIMIT_REQUESTS = 5  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds


class RateLimiter:
    """Simple in-memory rate limiter per client IP."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make a request."""
        now = time.time()
        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id]
            if now - t < self.window_seconds
        ]
        # Check limit
        if len(self._requests[client_id]) >= self.max_requests:
            return False
        # Record request
        self._requests[client_id].append(now)
        return True

    def time_until_allowed(self, client_id: str) -> float:
        """Get seconds until client can make another request."""
        if not self._requests[client_id]:
            return 0
        oldest = min(self._requests[client_id])
        return max(0, self.window_seconds - (time.time() - oldest))


# Global rate limiter for steering endpoints
_rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)


def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Use X-Forwarded-For if behind proxy, otherwise use client host
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


router = APIRouter(prefix="/steering", tags=["Steering"])


@router.post("/compare")
async def generate_steering_comparison_removed():
    """
    [REMOVED] This synchronous endpoint has been removed.

    Use POST /steering/async/compare instead.
    If you see this error, please hard refresh your browser (Ctrl+Shift+R).
    """
    raise HTTPException(
        410,
        "This endpoint has been removed. Please hard refresh your browser (Ctrl+Shift+R) "
        "to load the updated frontend that uses the async steering API."
    )


@router.post("/sweep")
async def generate_strength_sweep_removed():
    """
    [REMOVED] This synchronous endpoint has been removed.

    Use POST /steering/async/sweep instead.
    If you see this error, please hard refresh your browser (Ctrl+Shift+R).
    """
    raise HTTPException(
        410,
        "This endpoint has been removed. Please hard refresh your browser (Ctrl+Shift+R) "
        "to load the updated frontend that uses the async steering API."
    )


@router.get("/status")
async def get_steering_status():
    """
    Get steering service status including resilience metrics.

    Returns comprehensive status information about:
    - Circuit breaker state and failure counts
    - Concurrency limiter status
    - Process isolation statistics
    - Cache contents

    Use this endpoint to monitor steering health and diagnose issues.
    """
    steering_service = get_steering_service()

    # Get resilience status
    resilience = await get_resilience_status()

    # Add cache info
    cache_info = {
        "loaded_models": len(steering_service._loaded_models),
        "loaded_saes": len(steering_service._loaded_saes),
        "model_ids": list(steering_service._loaded_models.keys()),
        "sae_ids": list(steering_service._loaded_saes.keys()),
    }

    return {
        "status": "healthy" if resilience["circuit_breaker"]["state"] == "closed" else "degraded",
        "resilience": resilience,
        "cache": cache_info,
        "timeout_seconds": STEERING_TIMEOUT_SECONDS,
    }


@router.post("/reset")
async def reset_steering_resilience():
    """
    Reset steering resilience mechanisms.

    Resets the circuit breaker to closed state, allowing requests
    to flow again after failures. Use this after fixing underlying
    issues that caused the circuit to open.

    Returns:
        Dict with reset confirmation for each component.
    """
    result = await reset_resilience()

    # Also clear any stale state in steering service
    steering_service = get_steering_service()

    return {
        "message": "Resilience mechanisms reset",
        "details": result,
    }


@router.post("/cleanup")
async def cleanup_steering_gpu():
    """
    Release GPU memory held by the steering worker.

    Submits a cleanup task to the steering Celery worker that unloads
    all cached models and SAEs from GPU memory. Use this when done
    with steering to free VRAM for other tasks.

    Returns:
        Dict with task_id for tracking and immediate acknowledgment.
    """
    from ....workers.steering_tasks import cleanup_steering_gpu as cleanup_task

    # Submit cleanup task to steering queue
    result = cleanup_task.delay()

    # Wait briefly for result (cleanup is fast)
    try:
        cleanup_result = result.get(timeout=30)
        return {
            "message": "GPU memory released",
            "task_id": result.id,
            **cleanup_result,
        }
    except Exception as e:
        return {
            "message": "Cleanup task submitted but result pending",
            "task_id": result.id,
            "error": str(e),
        }


# =============================================================================
# STEERING MODE CONTROL
# =============================================================================
# These endpoints control whether steering mode is active.
# IN mode: Worker running, model loaded on GPU, can execute tasks.
# OUT of mode: No worker, no model, GPU free, tasks disabled.

PID_FILE = "/tmp/mistudio-celery-steering.pid"
STEERING_LOG = "/tmp/celery-steering.log"


def _get_gpu_memory_mb() -> Optional[int]:
    """Get current GPU memory usage in MB."""
    import subprocess
    try:
        gpu_output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if gpu_output.returncode == 0:
            return int(gpu_output.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def _is_steering_worker_running() -> tuple[bool, Optional[int]]:
    """Check if steering worker is running. Returns (is_running, pid)."""
    import os
    import signal

    # Check PID file
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            # Check if process is actually running
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            return True, pid
        except (ProcessLookupError, ValueError, OSError):
            # Process not running or invalid PID
            pass

    # Also check by process name pattern
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "steering@"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            pid = int(result.stdout.strip().split("\n")[0])
            return True, pid
    except Exception:
        pass

    return False, None


async def _ensure_steering_worker_running() -> tuple[bool, Optional[int]]:
    """
    Ensure a FRESH steering worker is running.

    Returns (success, pid) - success=True if worker is running,
    pid is the worker PID if known.

    IMPORTANT: This function ALWAYS kills any existing worker before starting
    a new one. This ensures each task gets a completely fresh Python/CUDA
    environment, avoiding state corruption issues with --pool=solo.
    """
    import subprocess
    import os
    import signal

    # ALWAYS kill existing worker to ensure fresh state
    is_active, existing_pid = _is_steering_worker_running()
    if is_active and existing_pid:
        logger.info(f"Killing existing steering worker PID {existing_pid} for fresh start")
        try:
            os.kill(existing_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Already dead
        except Exception as e:
            logger.warning(f"Could not kill worker {existing_pid}: {e}")

        # Also kill by pattern to catch orphans
        try:
            subprocess.run(["pkill", "-9", "-f", "steering@"], timeout=5, capture_output=True)
        except Exception:
            pass

        # Wait for process to fully terminate
        await asyncio.sleep(1)

    # Clean up stale PID file
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except Exception:
            pass

    # Start new steering worker
    # Use settings.backend_dir which defaults to /app in containers
    # In development, set BACKEND_DIR env var to your backend path
    backend_dir = settings.backend_dir

    try:
        # Check if running in container (no venv) or development (with venv)
        venv_path = backend_dir / "venv" / "bin" / "activate"
        venv_activate = f"source {venv_path} && " if venv_path.exists() else ""

        # CUDA_VISIBLE_DEVICES=0 restricts to first GPU only
        start_cmd = (
            f"cd {backend_dir} && {venv_activate}"
            f"CUDA_VISIBLE_DEVICES=0 celery -A src.core.celery_app worker "
            f"-Q steering -c 1 --pool=solo --loglevel=info "
            f'--hostname="steering@%h" --max-tasks-per-child=1 '
            f'--pidfile="{PID_FILE}" > "{STEERING_LOG}" 2>&1'
        )

        # Start process in background
        subprocess.Popen(
            start_cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait for worker to initialize
        for i in range(10):  # Try for 10 seconds
            await asyncio.sleep(1)
            is_running, pid = _is_steering_worker_running()
            if is_running:
                logger.info(f"Auto-started steering worker PID {pid}")
                return True, pid

        logger.error("Failed to auto-start steering worker within 10s")
        return False, None

    except Exception as e:
        logger.error(f"Failed to auto-start steering worker: {e}")
        return False, None


@router.get("/mode")
async def get_steering_mode_status():
    """
    Get current steering mode status.

    Returns whether steering mode is active (worker running) and GPU memory usage.
    """
    is_active, pid = _is_steering_worker_running()
    gpu_memory = _get_gpu_memory_mb()

    return {
        "active": is_active,
        "worker_pid": pid,
        "gpu_memory_mb": gpu_memory,
    }


@router.post("/enter-mode")
async def enter_steering_mode():
    """
    Enter steering mode by starting the steering worker.

    Starts a dedicated Celery worker for steering operations. The worker will
    load models on first use and keep them cached for fast subsequent generations.

    Returns:
        Dict with status of the enter operation.
    """
    import subprocess
    import os

    # Check if already in steering mode
    is_active, existing_pid = _is_steering_worker_running()
    if is_active:
        return {
            "success": True,
            "message": f"Already in steering mode (worker PID: {existing_pid})",
            "worker_pid": existing_pid,
            "already_active": True,
        }

    result = {
        "success": False,
        "message": "",
        "worker_pid": None,
        "already_active": False,
    }

    # Start new steering worker
    # Use settings.backend_dir which defaults to /app in containers
    # In development, set BACKEND_DIR env var to your backend path
    backend_dir = settings.backend_dir

    try:
        # Check if running in container (no venv) or development (with venv)
        venv_path = backend_dir / "venv" / "bin" / "activate"
        venv_activate = f"source {venv_path} && " if venv_path.exists() else ""

        # Use Popen to start worker in background without waiting
        # This avoids timeout issues with subprocess.run
        # CUDA_VISIBLE_DEVICES=0 restricts to first GPU only
        start_cmd = (
            f"cd {backend_dir} && {venv_activate}"
            f"CUDA_VISIBLE_DEVICES=0 celery -A src.core.celery_app worker "
            f"-Q steering -c 1 --pool=solo --loglevel=info "
            f'--hostname="steering@%h" --max-tasks-per-child=1 '
            f'--pidfile="{PID_FILE}" > "{STEERING_LOG}" 2>&1'
        )

        # Start process in background - Popen returns immediately
        process = subprocess.Popen(
            start_cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent
        )

        # Wait for worker to initialize and create PID file
        for i in range(10):  # Try for 10 seconds
            await asyncio.sleep(1)
            is_running, pid = _is_steering_worker_running()
            if is_running:
                result["success"] = True
                result["message"] = f"Entered steering mode (worker PID: {pid})"
                result["worker_pid"] = pid
                logger.info(f"Started steering worker PID {pid}")
                break
        else:
            result["message"] = "Failed to start steering worker within 10s - check logs"
            logger.error("Steering worker failed to start within timeout")

    except Exception as e:
        logger.error(f"Failed to start steering worker: {e}")
        result["message"] = f"Failed to start worker: {e}"

    return result


@router.post("/exit-mode")
async def exit_steering_mode():
    """
    Exit steering mode by killing the steering worker.

    This forcefully terminates the steering worker process, releasing ALL
    GPU memory held by steering operations. Steering will be unavailable
    until enter-mode is called again.

    Returns:
        Dict with status of the exit operation.
    """
    import subprocess
    import os
    import signal

    # Check if already out of steering mode
    is_active, existing_pid = _is_steering_worker_running()
    if not is_active:
        return {
            "success": True,
            "message": "Already out of steering mode",
            "killed_pid": None,
            "gpu_memory_freed_mb": 0,
            "already_inactive": True,
        }

    result = {
        "success": False,
        "message": "",
        "killed_pid": None,
        "gpu_memory_before": _get_gpu_memory_mb(),
        "gpu_memory_after": None,
        "gpu_memory_freed_mb": 0,
        "already_inactive": False,
    }

    # Kill the steering worker
    killed = False

    # Kill by PID if we have it
    if existing_pid:
        try:
            os.kill(existing_pid, signal.SIGKILL)
            killed = True
            result["killed_pid"] = existing_pid
            logger.info(f"Killed steering worker PID {existing_pid}")
        except ProcessLookupError:
            logger.info(f"Process {existing_pid} not found (already dead)")
            killed = True
        except Exception as e:
            logger.warning(f"Could not kill PID {existing_pid}: {e}")

    # Also kill by pattern to catch any orphans
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "steering@"],
            timeout=5, capture_output=True
        )
        killed = True
    except Exception:
        pass

    # Clean up PID file
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass

    # Wait for process to fully terminate and GPU to release
    await asyncio.sleep(3)

    # Get GPU memory after
    result["gpu_memory_after"] = _get_gpu_memory_mb()

    # Calculate memory freed
    if result["gpu_memory_before"] and result["gpu_memory_after"]:
        freed = result["gpu_memory_before"] - result["gpu_memory_after"]
        result["gpu_memory_freed_mb"] = freed

    # Verify we're out of steering mode
    is_still_active, _ = _is_steering_worker_running()
    if not is_still_active:
        result["success"] = True
        freed_msg = f" - Freed {result['gpu_memory_freed_mb']}MB" if result["gpu_memory_freed_mb"] > 0 else ""
        result["message"] = f"Exited steering mode{freed_msg}"
    else:
        result["message"] = "Worker may still be running - try again"

    return result


# =============================================================================
# ASYNC CELERY-BASED ENDPOINTS
# =============================================================================
# These endpoints submit tasks to Celery workers for isolated GPU execution.
# Benefits: process isolation, SIGKILL timeout, worker recycling, no zombies.


@router.post("/async/compare", response_model=SteeringTaskResponse)
async def submit_async_steering_comparison(
    request: SteeringComparisonRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit an async steering comparison task.

    This endpoint submits the steering task to a Celery worker running in a
    separate process. The worker provides:
    - Process isolation (crashes don't affect API)
    - SIGKILL timeout (guaranteed termination)
    - Worker recycling (prevents memory leaks)

    After submission:
    1. Subscribe to WebSocket channel steering/{task_id} for progress
    2. Or poll GET /steering/async/result/{task_id}

    Rate limited to 5 requests per minute per client.

    NOTE: The steering worker automatically exits after each task to ensure
    a fresh Python/CUDA environment. This endpoint auto-starts the worker
    if it's not running.
    """
    from datetime import datetime
    from ....workers.steering_tasks import steering_compare_task

    # Rate limiting
    client_id = get_client_id(http_request)
    if not _rate_limiter.is_allowed(client_id):
        retry_after = int(_rate_limiter.time_until_allowed(client_id)) + 1
        raise HTTPException(
            429,
            f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    # Ensure steering worker is running (it exits after each task)
    worker_ok, worker_pid = await _ensure_steering_worker_running()
    if not worker_ok:
        raise HTTPException(
            503,
            "Steering worker failed to start. Check server logs for details.",
        )

    # Get SAE from database
    sae = await SAEManagerService.get_sae(db, request.sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {request.sae_id}")

    if sae.status != SAEStatus.READY.value:
        raise HTTPException(400, f"SAE is not ready: {sae.status}")

    if not sae.local_path:
        raise HTTPException(400, "SAE has no local path")

    sae_path = settings.resolve_data_path(sae.local_path)
    if not sae_path.exists():
        raise HTTPException(400, f"SAE path does not exist: {sae.local_path}")

    # Validate feature indices against SAE dimension
    if sae.n_features:
        invalid_features = [
            f for f in request.selected_features
            if f.feature_idx >= sae.n_features
        ]
        if invalid_features:
            invalid_indices = [f.feature_idx for f in invalid_features]
            raise HTTPException(
                400,
                f"Invalid feature indices: {invalid_indices}. "
                f"SAE only has {sae.n_features} features."
            )

    # Determine model to use
    model_id = request.model_id
    if not model_id:
        if sae.model_id:
            model_id = sae.model_id
        elif sae.model_name:
            model_id = sae.model_name
        else:
            raise HTTPException(
                400,
                "No model specified and SAE has no linked model."
            )

    # Look up model from database to get actual file_path
    model_path = None
    model = await ModelService.get_model(db, model_id)
    if model and model.file_path:
        model_path = str(settings.resolve_data_path(model.file_path))
        model_id = model.repo_id or model.name

    # Submit task to Celery
    task = steering_compare_task.apply_async(
        kwargs={
            "request_dict": request.model_dump(mode="json"),
            "sae_id": request.sae_id,
            "model_id": model_id,
            "sae_path": str(sae_path),
            "model_path": model_path,
            "sae_layer": sae.layer,
            "sae_d_model": sae.d_model,
            "sae_n_features": sae.n_features,
            "sae_architecture": sae.architecture,
        }
    )

    return SteeringTaskResponse(
        task_id=task.id,
        task_type="compare",
        status="pending",
        websocket_channel=f"steering/{task.id}",
        message="Steering comparison task submitted",
        submitted_at=datetime.utcnow(),
    )


@router.post("/async/sweep", response_model=SteeringTaskResponse)
async def submit_async_strength_sweep(
    request: SteeringStrengthSweepRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit an async strength sweep task.

    Similar to /async/compare but for strength sweeps.

    NOTE: The steering worker automatically exits after each task to ensure
    a fresh Python/CUDA environment. This endpoint auto-starts the worker
    if it's not running.
    """
    from datetime import datetime
    from ....workers.steering_tasks import steering_sweep_task

    # Rate limiting
    client_id = get_client_id(http_request)
    if not _rate_limiter.is_allowed(client_id):
        retry_after = int(_rate_limiter.time_until_allowed(client_id)) + 1
        raise HTTPException(
            429,
            f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    # Ensure steering worker is running (it exits after each task)
    worker_ok, worker_pid = await _ensure_steering_worker_running()
    if not worker_ok:
        raise HTTPException(
            503,
            "Steering worker failed to start. Check server logs for details.",
        )

    # Get SAE from database
    sae = await SAEManagerService.get_sae(db, request.sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {request.sae_id}")

    if sae.status != SAEStatus.READY.value:
        raise HTTPException(400, f"SAE is not ready: {sae.status}")

    if not sae.local_path:
        raise HTTPException(400, "SAE has no local path")

    sae_path = settings.resolve_data_path(sae.local_path)
    if not sae_path.exists():
        raise HTTPException(400, f"SAE path does not exist")

    # Determine model
    model_id = request.model_id
    if not model_id:
        if sae.model_id:
            model_id = sae.model_id
        elif sae.model_name:
            model_id = sae.model_name
        else:
            raise HTTPException(400, "No model specified and SAE has no linked model.")

    model_path = None
    model = await ModelService.get_model(db, model_id)
    if model and model.file_path:
        model_path = str(settings.resolve_data_path(model.file_path))
        model_id = model.repo_id or model.name

    # Submit task
    task = steering_sweep_task.apply_async(
        kwargs={
            "request_dict": request.model_dump(mode="json"),
            "sae_id": request.sae_id,
            "model_id": model_id,
            "sae_path": str(sae_path),
            "model_path": model_path,
            "sae_layer": sae.layer,
            "sae_d_model": sae.d_model,
            "sae_n_features": sae.n_features,
            "sae_architecture": sae.architecture,
        }
    )

    return SteeringTaskResponse(
        task_id=task.id,
        task_type="sweep",
        status="pending",
        websocket_channel=f"steering/{task.id}",
        message="Strength sweep task submitted",
        submitted_at=datetime.utcnow(),
    )


@router.get("/async/result/{task_id}", response_model=SteeringResultResponse)
async def get_steering_task_result(task_id: str):
    """
    Get the result of an async steering task.

    Returns the current task status and, if complete, the result.

    Task statuses:
    - pending: Task waiting in queue
    - started: Task picked up by worker
    - progress: Task in progress (check percent)
    - success: Task completed successfully
    - failure: Task failed (check error)
    - revoked: Task was cancelled
    """
    from datetime import datetime
    from ....core.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)

    # Map Celery state to our status
    status_map = {
        "PENDING": "pending",
        "STARTED": "started",
        "PROGRESS": "progress",
        "SUCCESS": "success",
        "FAILURE": "failure",
        "REVOKED": "revoked",
        "RETRY": "pending",
    }

    status = status_map.get(result.state, "pending")

    # Build status object
    task_status = SteeringTaskStatus(
        task_id=task_id,
        status=status,
        percent=0,
        message="",
    )

    # Get additional info from result.info if available
    if result.info:
        if isinstance(result.info, dict):
            task_status.percent = result.info.get("percent", 0)
            task_status.message = result.info.get("message", "")
        elif isinstance(result.info, Exception):
            task_status.error = str(result.info)
            task_status.message = str(result.info)
            task_status.percent = -1

    # Handle success
    task_result = None
    if result.successful():
        task_status.percent = 100
        task_status.message = "Complete"
        task_status.completed_at = datetime.utcnow()
        task_result = result.result

    # Handle failure
    if result.failed():
        task_status.percent = -1
        task_status.error = str(result.result) if result.result else "Unknown error"
        task_status.message = f"Failed: {task_status.error}"
        task_status.completed_at = datetime.utcnow()

    return SteeringResultResponse(
        task_id=task_id,
        status=task_status,
        result=task_result,
    )


@router.delete("/async/task/{task_id}", response_model=SteeringCancelResponse)
async def cancel_steering_task(task_id: str):
    """
    Cancel a steering task.

    If the task is pending, it will be removed from the queue.
    If the task is running, it will be terminated (SIGTERM, then SIGKILL).

    Note: Running tasks may not terminate immediately. The worker will
    attempt graceful shutdown first, then force terminate.
    """
    from ....core.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)

    if result.state == "SUCCESS":
        return SteeringCancelResponse(
            task_id=task_id,
            status="already_complete",
            message="Task already completed successfully",
        )

    if result.state == "FAILURE":
        return SteeringCancelResponse(
            task_id=task_id,
            status="already_complete",
            message="Task already failed",
        )

    if result.state == "REVOKED":
        return SteeringCancelResponse(
            task_id=task_id,
            status="already_cancelled",
            message="Task was already cancelled",
        )

    # Revoke the task (terminate if running)
    celery_app.control.revoke(task_id, terminate=True, signal="SIGKILL")

    return SteeringCancelResponse(
        task_id=task_id,
        status="cancelled",
        message="Task cancellation requested. Worker will terminate if running.",
    )


# =============================================================================
# EXPERIMENTS ENDPOINTS
# =============================================================================
# These endpoints manage saved steering experiments for later viewing.


@router.get("/experiments")
async def list_steering_experiments(
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    sae_id: Optional[str] = None,
    model_id: Optional[str] = None,
    tag: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List saved steering experiments with filtering and pagination.

    Query parameters:
    - skip: Number of records to skip (default 0)
    - limit: Max records to return (default 50)
    - search: Search in name, description, or prompt
    - sae_id: Filter by SAE ID
    - model_id: Filter by model ID
    - tag: Filter by tag
    """
    from ....services.steering_experiments_service import SteeringExperimentsService

    experiments, total = await SteeringExperimentsService.list_experiments(
        db=db,
        skip=skip,
        limit=limit,
        search=search,
        sae_id=sae_id,
        model_id=model_id,
        tag=tag,
    )

    return {
        "data": [exp.to_dict() for exp in experiments],
        "pagination": {
            "skip": skip,
            "limit": limit,
            "total": total,
            "has_more": skip + len(experiments) < total,
        },
    }


@router.post("/experiments")
async def save_steering_experiment(
    request: SteeringExperimentSaveRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Save a steering experiment for later viewing.

    The request must include the full comparison result since
    comparisons are ephemeral (stored in Redis with TTL).
    """
    from ....services.steering_experiments_service import SteeringExperimentsService

    if not request.result:
        raise HTTPException(400, "Result is required to save an experiment")

    # Check if experiment with this comparison_id already exists
    existing = await SteeringExperimentsService.get_experiment_by_comparison_id(
        db, request.comparison_id
    )
    if existing:
        raise HTTPException(
            409,
            f"Experiment with comparison_id {request.comparison_id} already exists"
        )

    experiment = await SteeringExperimentsService.create_experiment(
        db=db,
        name=request.name,
        comparison_id=request.comparison_id,
        results=request.result,
        description=request.description,
        tags=request.tags,
    )

    return experiment.to_dict()


@router.get("/experiments/{experiment_id}")
async def get_steering_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a single steering experiment by ID.
    """
    from uuid import UUID
    from ....services.steering_experiments_service import SteeringExperimentsService

    try:
        exp_uuid = UUID(experiment_id)
    except ValueError:
        raise HTTPException(400, f"Invalid experiment ID: {experiment_id}")

    experiment = await SteeringExperimentsService.get_experiment(db, exp_uuid)
    if not experiment:
        raise HTTPException(404, f"Experiment not found: {experiment_id}")

    return experiment.to_dict()


@router.delete("/experiments/{experiment_id}")
async def delete_steering_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a steering experiment.
    """
    from uuid import UUID
    from ....services.steering_experiments_service import SteeringExperimentsService

    try:
        exp_uuid = UUID(experiment_id)
    except ValueError:
        raise HTTPException(400, f"Invalid experiment ID: {experiment_id}")

    deleted = await SteeringExperimentsService.delete_experiment(db, exp_uuid)
    if not deleted:
        raise HTTPException(404, f"Experiment not found: {experiment_id}")

    return {"message": f"Experiment {experiment_id} deleted"}


@router.post("/experiments/delete")
async def delete_steering_experiments_batch(
    request: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete multiple steering experiments.

    Request body: {"ids": ["uuid1", "uuid2", ...]}
    """
    from uuid import UUID
    from ....services.steering_experiments_service import SteeringExperimentsService

    ids = request.get("ids", [])
    if not ids:
        raise HTTPException(400, "No experiment IDs provided")

    try:
        exp_uuids = [UUID(id) for id in ids]
    except ValueError as e:
        raise HTTPException(400, f"Invalid experiment ID: {e}")

    deleted_count = await SteeringExperimentsService.delete_experiments_batch(
        db, exp_uuids
    )

    return {
        "deleted_count": deleted_count,
        "message": f"Deleted {deleted_count} experiments",
    }
