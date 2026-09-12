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
import os
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    CombinedSteeringRequest,
    CombinedSteeringResponse,
    ClusterAllocationRequest,
    ClusterAllocationResponse,
    PerLayerAllocation,
    MultiLayerAllocationResponse,
    HazardModel,
    AllocationResponseUnion,
)
from ....services.sae_manager_service import SAEManagerService
from ....services.steering_service import (
    get_steering_service,
    load_sae_weights_cpu,
    resolve_decoder_weight,
    resolve_encoder_weight,
)
from ....services.cluster_allocation_service import (
    AllocationMember,
    MultiLayerMember,
    compute_allocation,
    compute_multi_layer_allocation,
    resolve_constants,
)
from ....services import steering_hazards
from ....services.model_service import ModelService
from ....services.steering_resilience import (
    get_circuit_breaker,
    get_resilience_status,
    reset_resilience,
)
from ....core.clock import utc_now

logger = logging.getLogger(__name__)


# MIS-E2E-062. The circuit breaker existed with no caller, so `/steering/status`
# could only ever report "healthy" and `/steering/reset` was a no-op that
# reported success. These two helpers are the wiring; removing either one turns
# the endpoint back into a constant, and
# `tests/unit/test_steering_resilience_wired.py` fails if you do.
#
# Both live in the API process, which is what makes the reported state the state
# that was actually recorded: dispatch happens here, and `GET
# /async/result/{task_id}` is where the API first learns a task's outcome.


async def _guard_steering_dispatch() -> None:
    """Refuse a new steering dispatch while the breaker is open.

    Raises 503 rather than queueing work behind a GPU that has already failed
    repeatedly — which is the situation the breaker exists to notice.
    """
    breaker = get_circuit_breaker()
    allowed, reason = await breaker.can_execute()
    if not allowed:
        raise HTTPException(
            503,
            detail={
                "code": "STEERING_CIRCUIT_OPEN",
                "message": reason or "Steering is temporarily unavailable",
                "hint": "Check GET /api/v1/steering/status; POST /steering/reset "
                        "once the underlying issue is fixed",
            },
        )


# A terminal outcome must be recorded ONCE, however many times the client polls.
# Without this the breaker counts a poll loop, not a failure.
_recorded_task_outcomes: set[str] = set()
_MAX_RECORDED_OUTCOMES = 10_000


async def _record_steering_outcome(task_id: str, *, succeeded: bool, error: str | None = None) -> None:
    """Feed a terminal task outcome to the breaker, exactly once per task."""
    if task_id in _recorded_task_outcomes:
        return
    if len(_recorded_task_outcomes) >= _MAX_RECORDED_OUTCOMES:
        # Bounded: this is a de-duplication aid, not a ledger.
        _recorded_task_outcomes.clear()
    _recorded_task_outcomes.add(task_id)

    breaker = get_circuit_breaker()
    if succeeded:
        await breaker.record_success()
    else:
        await breaker.record_failure(Exception(error or "steering task failed"))

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


def _routed_features(request) -> list:
    """Normalise a steering request to `[(feature_idx, layer, sae_id_or_None)]`.

    Compare and combined carry `selected_features`; sweep carries a single
    `feature_idx` / `layer` pair on the request itself. Both need the same
    layer check, and giving them one shape is what lets one helper serve all
    three rather than three near-copies that drift.
    """
    features = getattr(request, "selected_features", None)
    if features is not None:
        return [(f.feature_idx, f.layer, getattr(f, "sae_id", None)) for f in features]
    return [(request.feature_idx, request.layer, None)]


async def resolve_referenced_saes(request, sae, db) -> dict:
    """Resolve, validate and package EVERY SAE a steering request references.

    MIS-E2E-064. This was the combined endpoint's code, inline, and only the
    combined endpoint's. Compare and sweep placed their hook at `feature.layer`
    while always steering through the REQUEST-level SAE, discarding each
    feature's own `sae_id` — which `SelectedFeature` has carried since Feature
    015. Nothing validated `feature.layer == sae.layer` on those paths either.

    The consequence is the worst failure mode an interpretability tool has:
    because `d_model` is uniform across layers, the `hidden_dim != sae.d_in`
    shape guard NEVER fires, so a feature from layer 20's dictionary is decoded
    through layer 12's SAE and applied at layer 20 — silently, in the correct
    shape and the wrong basis. Plausible output, meaningless, no error.

    Extracted rather than copied. "Fixed one representative, never generalized"
    is this audit's most repeated anti-pattern — five independent instances,
    and this finding IS one of them. A shared helper is the only version of the
    fix that cannot drift back apart.

    Validation happens at SUBMIT time, never in the worker: failing in the
    worker burns a GPU slot to produce a 500 instead of a 422.

    Returns:
        {sae_id -> SaeMeta dict} for exactly the referenced ids, JSON-safe for
        the Celery kwargs.

    Raises:
        HTTPException: 404 / 400 for an unusable SAE, 422 for a layer mismatch.
    """
    # Each feature steers through the SAE trained on ITS layer (feature.sae_id ??
    # request.sae_id). Validate — at SUBMIT time, never in the worker (which
    # would burn a GPU slot to fail) — that each referenced SAE exists, is READY,
    # and its layer matches every feature routed to it. Build a JSON-serializable
    # SaeMeta map for the worker so it can load them all.
    routed = _routed_features(request)

    referenced_ids: list[str] = []
    for _idx, _layer, f_sae_id in routed:
        sid = f_sae_id or request.sae_id
        if sid not in referenced_ids:
            referenced_ids.append(sid)

    sae_records = {request.sae_id: sae}
    for sid in referenced_ids:
        if sid not in sae_records:
            rec = await SAEManagerService.get_sae(db, sid)
            if not rec:
                raise HTTPException(404, f"SAE not found: {sid}")
            if rec.status != SAEStatus.READY.value:
                raise HTTPException(400, f"SAE is not ready: {sid} ({rec.status})")
            if not rec.local_path:
                raise HTTPException(400, f"SAE has no local path: {sid}")
            sae_records[sid] = rec

    # Per-feature layer/SAE mismatch → 422 listing offenders.
    offenders = []
    for f_idx, f_layer, f_sae_id in routed:
        sid = f_sae_id or request.sae_id
        rec = sae_records[sid]
        if rec.layer is not None and f_layer != rec.layer:
            offenders.append({
                "feature_idx": f_idx,
                "layer": f_layer,
                "sae_id": sid,
                "sae_layer": rec.layer,
            })
    if offenders:
        raise HTTPException(
            422,
            {
                "code": "sae_layer_mismatch",
                "message": "One or more features are routed to an SAE trained on a different layer.",
                "offenders": offenders,
            },
        )

    # Validate feature indices against each routed SAE's dimension.
    bad_by_sae: dict = {}
    for f_idx, _f_layer, f_sae_id in routed:
        sid = f_sae_id or request.sae_id
        rec = sae_records[sid]
        if rec.n_features and f_idx >= rec.n_features:
            bad_by_sae.setdefault(sid, []).append(f_idx)
    if bad_by_sae:
        raise HTTPException(
            400,
            f"Invalid feature indices per SAE: {bad_by_sae}. "
            "Each index must be within its SAE's feature count.",
        )

    # Build the SaeMeta map (JSON dicts) for the worker — one entry per
    # referenced SAE. Single-SAE requests produce a one-entry map, keeping the
    # worker on the byte-identical single-SAE codepath.
    sae_meta_map: dict = {}
    for sid in referenced_ids:
        rec = sae_records[sid]
        rec_path = settings.resolve_data_path(rec.local_path)
        if not rec_path.exists():
            raise HTTPException(400, f"SAE path does not exist: {rec.local_path}")
        sae_meta_map[sid] = {
            "sae_id": sid,
            "sae_path": str(rec_path),
            "layer": rec.layer,
            "d_model": rec.d_model,
            "n_features": rec.n_features,
            "architecture": rec.architecture,
        }

    return sae_meta_map


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

    # Wait briefly for result (cleanup is fast). result.get() is a blocking
    # Celery wait — it froze the event loop once (2026-07); keep it in a thread.
    try:
        cleanup_result = await asyncio.wait_for(
            asyncio.to_thread(result.get, timeout=30), timeout=35
        )
        return {
            "message": "GPU memory released",
            "task_id": result.id,
            **cleanup_result,
        }
    except Exception:
        logger.exception("GPU cleanup task did not complete in time")
        return {
            "message": "Cleanup task submitted but result pending",
            "task_id": result.id,
            "error": "Task is still running or failed — check the server log for details",
        }


# =============================================================================
# STEERING MODE CONTROL
# =============================================================================
# These endpoints control whether steering mode is active.
# IN mode: Worker running, model loaded on GPU, can execute tasks.
# OUT of mode: No worker, no model, GPU free, tasks disabled.

# Use configurable run_dir for PID files and logs (works across native/docker/k8s)
PID_FILE = str(settings.run_dir / "mistudio-celery-steering.pid")

#: PIDs of steering workers THIS process spawned, so orphan cleanup can be
#: precise (MIS-E2E-003).
#:
#: `pkill -9 -f steering@` SIGKILLs any process on the host whose COMMAND LINE
#: contains "steering@" — another user's shell, an unrelated container sharing
#: the PID namespace, someone's `grep steering@`. A pattern kill reachable over
#: HTTP is a privilege operation regardless of who can reach the port, which is
#: why it is not covered by the accepted network-boundary posture.
#:
#: The worker is already started with `--pidfile`, so the precise handle exists;
#: nothing was using it for the orphan sweep.
_SPAWNED_WORKER_PIDS: set[int] = set()


async def _kill_orphan_steering_workers() -> int:
    """SIGKILL steering workers this process started, and nothing else.

    Replaces `pkill -9 -f steering@`. Returns how many were signalled.
    """
    killed = 0
    for pid in sorted(_SPAWNED_WORKER_PIDS):
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
            logger.info(f"Killed orphaned steering worker PID {pid}")
        except ProcessLookupError:
            pass          # already gone — the normal case
        except PermissionError:
            logger.warning(f"Not permitted to kill PID {pid}; leaving it")
        except Exception:
            logger.exception(f"Could not kill steering worker PID {pid}")
    _SPAWNED_WORKER_PIDS.clear()
    return killed
STEERING_LOG = str(settings.run_dir / "celery-steering.log")

# Ceiling on sync result-backend reads executed via asyncio.to_thread —
# protects the event loop from wedged Redis connections (2026-07 freeze).
RESULT_BACKEND_TIMEOUT_SECONDS = 10


def _read_celery_result(task_id: str):
    """Sync AsyncResult read. Must ONLY be called via asyncio.to_thread."""
    from ....core.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)
    state = result.state
    info = result.info
    succeeded = result.successful()
    failed = result.failed()
    raw = result.result if (succeeded or failed) else None
    return state, info, succeeded, failed, raw


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


async def _steering_queue_depth() -> int:
    """Ready (not yet delivered) messages on the steering queue.

    Celery's redis transport keeps each queue as a list named after it.
    Used by the reconcile endpoint; errors report 0 (no spawn) rather than
    failing the beat cycle.
    """
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(str(settings.redis_url))
        try:
            return int(await client.llen("steering"))
        finally:
            await client.aclose()
    except Exception:
        logger.exception("Could not read steering queue depth")
        return 0


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

    Kills any existing IDLE worker before starting a new one, so each task
    gets a completely fresh Python/CUDA environment (--pool=solo state
    corruption). A worker that is MID-GENERATION is left alone: SIGKILLing
    it stranded the in-flight acks_late message for the 12h visibility
    timeout (zombie "started 0%" tasks + leaked guardrail slots). The
    submitted task simply queues behind the running one; the post-task
    self-exit + reconcile loop give it a fresh worker afterwards.
    """
    import subprocess
    import os
    import signal

    from ....workers.steering_worker_state import read_busy_marker

    is_active, existing_pid = await asyncio.to_thread(_is_steering_worker_running)
    if is_active and existing_pid:
        busy = await asyncio.to_thread(read_busy_marker)
        # Honor the marker only when it was written by the live worker —
        # a killed worker's leftover marker must not shield its successor.
        # (Unreadable markers report pid -1 and are honored: fail busy-safe.)
        if busy is not None and busy.get("pid") in (existing_pid, -1):
            logger.info(
                "Steering worker PID %s is mid-task %s — not killing; "
                "new task will queue behind it",
                existing_pid, busy.get("task_id"),
            )
            return True, existing_pid
        logger.info(f"Killing existing steering worker PID {existing_pid} for fresh start")
        try:
            os.kill(existing_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Already dead
        except Exception as e:
            logger.exception(f"Could not kill worker {existing_pid}")

        # Also clear orphans WE spawned — by PID, never by cmdline pattern.
        await _kill_orphan_steering_workers()

        # Wait for process to fully terminate
        await asyncio.sleep(1)

    # Clean up stale PID file + any leftover busy marker from the old worker
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except Exception:
            pass
    try:
        from ....workers.steering_worker_state import _marker_path

        _marker_path().unlink(missing_ok=True)
    except Exception:
        pass

    # Start new steering worker
    # Use settings.backend_dir which defaults to /app in containers
    # In development, set BACKEND_DIR env var to your backend path
    backend_dir = settings.backend_dir

    try:
        # Use venv celery binary if present (dev), otherwise system celery (container)
        venv_celery = backend_dir / "venv" / "bin" / "celery"
        celery_bin = str(venv_celery) if venv_celery.exists() else "celery"

        # Pass CUDA restriction via env, not shell interpolation
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        # Redirect output to log file; child inherits fd after fork
        log_file = open(STEERING_LOG, "a")
        try:
            # RECORD THE PID (MIS-E2E-003). The orphan sweep kills what this
            # process started, by pid — never `pkill -f steering@`, which would
            # SIGKILL any process on the host whose cmdline happens to match.
            _spawned = subprocess.Popen(
                [
                    celery_bin, "-A", "src.core.celery_app", "worker",
                    "-Q", "steering", "-c", "1", "--pool=solo", "--loglevel=info",
                    "--hostname=steering@%h", "--max-tasks-per-child=1",
                    f"--pidfile={PID_FILE}",
                ],
                cwd=str(backend_dir),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _SPAWNED_WORKER_PIDS.add(_spawned.pid)
        finally:
            log_file.close()

        # Wait for worker to initialize
        for i in range(10):  # Try for 10 seconds
            await asyncio.sleep(1)
            is_running, pid = await asyncio.to_thread(_is_steering_worker_running)
            if is_running:
                logger.info(f"Auto-started steering worker PID {pid}")
                return True, pid

        logger.error("Failed to auto-start steering worker within 10s")
        return False, None

    except Exception as e:
        logger.exception("Failed to auto-start steering worker")
        return False, None


@router.get("/mode")
async def get_steering_mode_status():
    """
    Get current steering mode status.

    Returns whether steering mode is active (worker running) and GPU memory usage.
    """
    is_active, pid = await asyncio.to_thread(_is_steering_worker_running)
    gpu_memory = await asyncio.to_thread(_get_gpu_memory_mb)

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
    is_active, existing_pid = await asyncio.to_thread(_is_steering_worker_running)
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
        # Use venv celery binary if present (dev), otherwise system celery (container)
        venv_celery = backend_dir / "venv" / "bin" / "celery"
        celery_bin = str(venv_celery) if venv_celery.exists() else "celery"

        # Pass CUDA restriction via env, not shell interpolation
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        # Use Popen to start worker in background without waiting
        # This avoids timeout issues with subprocess.run
        # Redirect output to log file; child inherits fd after fork
        log_file = open(STEERING_LOG, "a")
        try:
            # RECORD THE PID (MIS-E2E-003). The orphan sweep kills what this
            # process started, by pid — never `pkill -f steering@`, which would
            # SIGKILL any process on the host whose cmdline happens to match.
            _spawned = subprocess.Popen(
                [
                    celery_bin, "-A", "src.core.celery_app", "worker",
                    "-Q", "steering", "-c", "1", "--pool=solo", "--loglevel=info",
                    "--hostname=steering@%h", "--max-tasks-per-child=1",
                    f"--pidfile={PID_FILE}",
                ],
                cwd=str(backend_dir),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
            )
            _SPAWNED_WORKER_PIDS.add(_spawned.pid)   # MIS-E2E-003
        finally:
            log_file.close()

        # Wait for worker to initialize and create PID file
        for i in range(10):  # Try for 10 seconds
            await asyncio.sleep(1)
            is_running, pid = await asyncio.to_thread(_is_steering_worker_running)
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
        logger.exception("Failed to start steering worker")
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
    is_active, existing_pid = await asyncio.to_thread(_is_steering_worker_running)
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
        "gpu_memory_before": await asyncio.to_thread(_get_gpu_memory_mb),
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
            logger.exception(f"Could not kill PID {existing_pid}")

    # Also clear orphans WE spawned — by PID, never by cmdline pattern.
    if await _kill_orphan_steering_workers():
        killed = True

    # Clean up PID file
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass

    # Wait for process to fully terminate and GPU to release
    await asyncio.sleep(3)

    # Get GPU memory after
    result["gpu_memory_after"] = await asyncio.to_thread(_get_gpu_memory_mb)

    # Calculate memory freed
    if result["gpu_memory_before"] and result["gpu_memory_after"]:
        freed = result["gpu_memory_before"] - result["gpu_memory_after"]
        result["gpu_memory_freed_mb"] = freed

    # Verify we're out of steering mode
    is_still_active, _ = await asyncio.to_thread(_is_steering_worker_running)
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

    # MIS-E2E-064: the shared resolver. This used to validate feature indices
    # against the REQUEST-level SAE only, and never checked that a feature's
    # layer matched the SAE it would be decoded through — so a cross-layer
    # feature steered in the wrong basis with no error. The resolver validates
    # per routed SAE and returns the map the worker needs to honour
    # `feature.sae_id`.
    sae_meta_map = await resolve_referenced_saes(request, sae, db)

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
    # MIS-E2E-062: the breaker gates the dispatch. Nothing called this
    # before, which is why /steering/status was a constant.
    await _guard_steering_dispatch()

    task = steering_compare_task.apply_async(
        kwargs={
            "request_dict": request.model_dump(mode="json"),
            "sae_id": request.sae_id,
            # MIS-E2E-064: every referenced SAE, so each feature
            # steers through the dictionary trained on ITS layer.
            "sae_meta_map": sae_meta_map,
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
        submitted_at=utc_now(),
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

    # MIS-E2E-064. Sweep had NO feature validation at all: it steered at
    # `request.layer` through `request.sae_id`'s SAE without ever checking the
    # two agree, and `d_model` being uniform across layers means the shape guard
    # never catches it. Same wrong-basis defect as compare, simpler shape.
    sae_meta_map = await resolve_referenced_saes(request, sae, db)

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
    # MIS-E2E-062: the breaker gates the dispatch. Nothing called this
    # before, which is why /steering/status was a constant.
    await _guard_steering_dispatch()

    task = steering_sweep_task.apply_async(
        kwargs={
            "request_dict": request.model_dump(mode="json"),
            "sae_id": request.sae_id,
            # MIS-E2E-064: every referenced SAE, so each feature
            # steers through the dictionary trained on ITS layer.
            "sae_meta_map": sae_meta_map,
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
        submitted_at=utc_now(),
    )


@router.post("/async/combined", response_model=SteeringTaskResponse)
async def submit_async_combined_steering(
    request: CombinedSteeringRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit an async combined multi-feature steering task.

    This endpoint applies ALL selected features simultaneously in a single
    generation pass, enabling exploration of synergistic effects and
    feature interactions.

    Use cases:
    - Test synergistic effects (e.g., "formal" + "positive" = professional tone)
    - Create complex behavioral changes with multiple influences
    - Explore feature interactions and emergent behaviors

    After submission:
    1. Subscribe to WebSocket channel steering/{task_id} for progress
    2. Or poll GET /steering/async/result/{task_id}

    Rate limited to 5 requests per minute per client.
    """
    from datetime import datetime
    from ....workers.steering_tasks import steering_combined_task

    # Rate limiting
    client_id = get_client_id(http_request)
    if not _rate_limiter.is_allowed(client_id):
        retry_after = int(_rate_limiter.time_until_allowed(client_id)) + 1
        raise HTTPException(
            429,
            f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    # Ensure steering worker is running
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

    # MIS-E2E-064: one shared resolver for all three steering endpoints.
    sae_meta_map = await resolve_referenced_saes(request, sae, db)

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
    # MIS-E2E-062: the breaker gates the dispatch. Nothing called this
    # before, which is why /steering/status was a constant.
    await _guard_steering_dispatch()

    task = steering_combined_task.apply_async(
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
            # Feature 015: load metadata for every referenced SAE (one entry for
            # single-SAE requests → byte-identical worker behaviour).
            "sae_meta_map": sae_meta_map,
        }
    )

    return SteeringTaskResponse(
        task_id=task.id,
        task_type="combined",
        status="pending",
        websocket_channel=f"steering/{task.id}",
        message="Combined multi-feature steering task submitted",
        submitted_at=utc_now(),
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

    # AsyncResult does sync Redis I/O (and can enter unbounded kombu retry
    # loops on a bad connection) — never run it on the event loop. A wedged
    # result-backend read now returns 503 instead of freezing the whole API.
    try:
        state, info, succeeded, failed, raw_result = await asyncio.wait_for(
            asyncio.to_thread(_read_celery_result, task_id),
            timeout=RESULT_BACKEND_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            503,
            detail={
                "code": "RESULT_BACKEND_TIMEOUT",
                "message": f"Result backend did not respond within {RESULT_BACKEND_TIMEOUT_SECONDS}s",
                "hint": "Retry shortly; check Redis health if this persists",
            },
        )

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

    status = status_map.get(state, "pending")

    # Build status object
    task_status = SteeringTaskStatus(
        task_id=task_id,
        status=status,
        percent=0,
        message="",
    )

    # Get additional info from result.info if available
    if info:
        if isinstance(info, dict):
            task_status.percent = info.get("percent", 0)
            task_status.message = info.get("message", "")
        elif isinstance(info, Exception):
            task_status.error = str(info)
            task_status.message = str(info)
            task_status.percent = -1

    # Handle success
    task_result = None
    if succeeded:
        task_status.percent = 100
        task_status.message = "Complete"
        task_status.completed_at = utc_now()
        task_result = raw_result

    # Handle failure
    if failed:
        task_status.percent = -1
        task_status.error = str(raw_result) if raw_result else "Unknown error"
        task_status.message = f"Failed: {task_status.error}"
        task_status.completed_at = utc_now()

    # MIS-E2E-062: this is where the API first learns a task's outcome, so this
    # is where the breaker learns it. Recorded once per task id, however many
    # times the client polls — otherwise the breaker counts a poll loop rather
    # than a failure, and three polls of one failed task would open it.
    if succeeded or failed:
        await _record_steering_outcome(
            task_id, succeeded=succeeded, error=task_status.error
        )

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

    try:
        state, _, _, _, _ = await asyncio.wait_for(
            asyncio.to_thread(_read_celery_result, task_id),
            timeout=RESULT_BACKEND_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            503,
            detail={
                "code": "RESULT_BACKEND_TIMEOUT",
                "message": f"Result backend did not respond within {RESULT_BACKEND_TIMEOUT_SECONDS}s",
                "hint": "Retry shortly; check Redis health if this persists",
            },
        )

    if state == "SUCCESS":
        return SteeringCancelResponse(
            task_id=task_id,
            status="already_complete",
            message="Task already completed successfully",
        )

    if state == "FAILURE":
        return SteeringCancelResponse(
            task_id=task_id,
            status="already_complete",
            message="Task already failed",
        )

    if state == "REVOKED":
        return SteeringCancelResponse(
            task_id=task_id,
            status="already_cancelled",
            message="Task was already cancelled",
        )

    # Revoke the task (terminate if running); broker I/O off the loop too
    await asyncio.to_thread(
        celery_app.control.revoke, task_id, terminate=True, signal="SIGKILL"
    )

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

@router.post("/cluster-allocation", response_model=AllocationResponseUnion)
async def compute_cluster_strength_allocation(
    request: ClusterAllocationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Compute a principled starting strength allocation for a cluster (Feature 013,
    per-layer for a multi-layer circuit — Feature 015).

    Loads only the SAE(s) (through the shared steering cache — the same load any
    steering prep performs; no LLM load, no generation, no steering-mode
    requirement, no Celery). The formula is the single server-side source of
    truth (IDL-29); the frontend only performs budget-preserving rebalance on
    the returned values. Decoder columns are sliced in-place — no full-matrix
    copies.

    Response shape (union, single-layer FIRST for 013 back-compat):
      * single distinct layer  → the 013 ``ClusterAllocationResponse`` (unchanged).
      * multiple layers        → ``MultiLayerAllocationResponse``: a per-layer
        map (each entry the 013 allocation + its ``sae_id``), cross-layer
        ``hazards``, and ``strengths`` flattened in request-member order.
    """
    distinct_layers = sorted({m.layer for m in request.members})

    # ── Multi-layer branch (Feature 015) ───────────────────────────────────────
    if len(distinct_layers) > 1:
        return await _compute_multi_layer_allocation_response(request, db)

    # ── Single-layer branch (Feature 013 — BYTE-IDENTICAL to pre-015) ──────────
    sae = await SAEManagerService.get_sae(db, request.sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {request.sae_id}")
    if str(sae.status.value if hasattr(sae.status, "value") else sae.status) != "ready":
        raise HTTPException(400, f"SAE is not ready: {sae.status}")
    if not sae.local_path:
        raise HTTPException(400, "SAE has no local path")

    # Members must target the SAE's own layer — the decoder defines directions
    # for exactly that layer; a mismatched request would return a "principled"
    # allocation computed against the wrong residual space.
    if sae.layer is not None:
        wrong_layer = sorted({m.layer for m in request.members if m.layer != sae.layer})
        if wrong_layer:
            raise HTTPException(
                400,
                f"Members target layer(s) {wrong_layer} but SAE {request.sae_id} is layer {sae.layer}",
            )

    # Bounds check against DB metadata up-front (also enforced against the real
    # decoder inside compute_allocation when it loads).
    if sae.n_features:
        bad = [m.feature_idx for m in request.members if m.feature_idx >= sae.n_features]
        if bad:
            raise HTTPException(
                400,
                f"Feature indices out of bounds for this SAE ({sae.n_features} features): {bad}",
            )

    # Resolve the decoder via the SAME loader + orientation logic the steering
    # hook uses. Failure degrades to the approximate (G=1) allocation.
    decoder = None
    try:
        sae_path = settings.resolve_data_path(sae.local_path)
        if sae_path.exists():
            # CPU weight-only load — this read-only allocation endpoint must NOT
            # resident-load the SAE onto the GPU (R2 F5: 015's "Steer this
            # circuit" button makes this a one-click browser action; the
            # multi-layer path was already CPU-only, so make single-layer match).
            dw, _ew = load_sae_weights_cpu(
                sae_path, d_model=sae.d_model,
                n_features=sae.n_features, architecture=sae.architecture)
            if dw is not None:
                decoder = dw
    except Exception as e:
        logger.warning(f"[ClusterAllocation] Decoder unavailable, using approximate G=1: {e}")

    members = [
        AllocationMember(
            feature_idx=m.feature_idx,
            layer=m.layer,
            similarity=m.similarity,
            activation_frequency=m.activation_frequency,
            sign=m.sign,
        )
        for m in request.members
    ]
    constants = resolve_constants(settings.steering_cluster_constants_json, request.sae_id)

    try:
        result = compute_allocation(
            members,
            decoder=decoder,
            constants=constants,
            group_cohesion=request.group_cohesion,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return ClusterAllocationResponse(
        B=result.B,
        B_dir=result.B_dir,
        G=result.G,
        f_eff=result.f_eff,
        weights=result.weights,
        strengths=result.strengths,
        flags=result.flags,
        cancellation_pair=list(result.cancellation_pair) if result.cancellation_pair else None,
        constants_used=result.constants_used,
        formula_id=result.formula_id,
        approximate=result.approximate,
    )


async def _load_circuit_edges(db: AsyncSession, circuit_id: str) -> Optional[list]:
    """Fetch a circuit's stored edges (JSONB list of edge dicts) for hazard-v2.

    Returns None if the circuit is missing — hazards then fall back to the
    labeled weight-prior heuristic. Never raises: an edge-fetch failure must not
    fail the allocation.
    """
    try:
        from ....models.circuit import Circuit
        from sqlalchemy import select

        row = (await db.execute(select(Circuit).where(Circuit.id == circuit_id))).scalar_one_or_none()
        if row is None:
            return None
        return list(row.edges or [])
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[ClusterAllocation] Could not load circuit {circuit_id} edges: {e}")
        return None


async def _resolve_cluster_members(
    db: AsyncSession, profile_ids
) -> Dict[Any, List[int]]:
    """`{profile_id: [feature_idx]}` for the profiles named by cluster edges.

    ONE query for all of them, resolved BEFORE expansion rather than during it:
    `expand_cluster_edges` is a synchronous pure function (which is what makes
    it testable without a database), so it cannot await, and resolving per edge
    inside a loop would be a query per edge besides.

    Never raises. A hazard analysis that fails must degrade to "could not
    check", which the caller reports, and never take the allocation down with
    it.
    """
    ids = [p for p in dict.fromkeys(profile_ids) if p is not None]
    if not ids:
        return {}
    try:
        from ....models.cluster_profile import ClusterProfile
        from sqlalchemy import select

        rows = (
            await db.execute(
                select(ClusterProfile).where(ClusterProfile.id.in_(ids))
            )
        ).scalars().all()
        return {
            r.id: [
                int(m["feature_idx"])
                for m in (r.members or [])
                if m.get("feature_idx") is not None
            ]
            for r in rows
        }
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[ClusterAllocation] Could not resolve cluster profiles: {e}")
        return {}


async def _compute_multi_layer_allocation_response(
    request: ClusterAllocationRequest,
    db: AsyncSession,
) -> MultiLayerAllocationResponse:
    """Per-layer allocation for a multi-layer cluster (Feature 015, IDL-29 reuse).

    Partitions members by layer, resolves each layer's SAE (per-member sae_id ??
    request sae_id), loads its decoder (for the gain) and encoder (for the
    hazard weight prior), runs the UNCHANGED per-layer ``compute_allocation``,
    then detects cross-layer hazards.
    """
    # Resolve the SAE for each layer from the members (per-member sae_id first).
    layer_sae_id: dict = {}
    for m in request.members:
        sid = getattr(m, "sae_id", None) or request.sae_id
        prev = layer_sae_id.get(m.layer)
        if prev is not None and prev != sid:
            raise HTTPException(
                422,
                {
                    "code": "mixed_sae_within_layer",
                    "message": f"Layer {m.layer} members reference more than one SAE ({prev}, {sid}); one SAE per layer.",
                },
            )
        layer_sae_id[m.layer] = sid

    # Cap the number of distinct SAEs (015 R1 QA-1): each one is loaded for the
    # gain/prior math; an unbounded fat circuit would thrash the GPU/CPU.
    MAX_ALLOCATION_SAES = 8
    if len(set(layer_sae_id.values())) > MAX_ALLOCATION_SAES:
        raise HTTPException(
            422, {"code": "too_many_saes",
                  "message": f"Multi-layer allocation references "
                             f"{len(set(layer_sae_id.values()))} SAEs — "
                             f"the cap is {MAX_ALLOCATION_SAES}."})

    decoders: dict = {}
    encoders: dict = {}
    offenders: list = []
    constants_by_layer: dict = {}

    for layer, sid in layer_sae_id.items():
        rec = await SAEManagerService.get_sae(db, sid)
        if not rec:
            raise HTTPException(404, f"SAE not found: {sid}")
        if str(rec.status.value if hasattr(rec.status, "value") else rec.status) != "ready":
            raise HTTPException(400, f"SAE is not ready: {sid} ({rec.status})")
        if not rec.local_path:
            raise HTTPException(400, f"SAE has no local path: {sid}")

        # Layer/SAE mismatch → 422 (collect all offenders).
        if rec.layer is not None and rec.layer != layer:
            for m in request.members:
                if m.layer == layer:
                    offenders.append({
                        "feature_idx": m.feature_idx,
                        "layer": m.layer,
                        "sae_id": sid,
                        "sae_layer": rec.layer,
                    })
            continue

        # Feature-index bounds against this SAE.
        if rec.n_features:
            bad = [m.feature_idx for m in request.members
                   if m.layer == layer and m.feature_idx >= rec.n_features]
            if bad:
                raise HTTPException(
                    400,
                    f"Feature indices out of bounds for SAE {sid} ({rec.n_features} features): {bad}",
                )

        constants_by_layer[layer] = resolve_constants(
            settings.steering_cluster_constants_json, sid)

        # Load ONLY the decoder (gain) + encoder (hazard prior) weight tensors
        # on CPU — never onto the GPU, never into the SAE cache (015 R1 QA-1:
        # this "read-only" endpoint used to force-load N full SAEs onto the
        # 24 GB card). Failure degrades that layer to approximate G=1.
        try:
            rec_path = settings.resolve_data_path(rec.local_path)
            if rec_path.exists():
                dw, ew = load_sae_weights_cpu(
                    rec_path, d_model=rec.d_model,
                    n_features=rec.n_features, architecture=rec.architecture)
                if dw is not None:
                    decoders[layer] = dw
                if ew is not None:
                    encoders[layer] = ew
        except Exception as e:
            logger.warning(f"[ClusterAllocation] Layer {layer} SAE {sid} unavailable, approximate G=1: {e}")

    if offenders:
        raise HTTPException(
            422,
            {
                "code": "sae_layer_mismatch",
                "message": "One or more members are routed to an SAE trained on a different layer.",
                "offenders": offenders,
            },
        )

    # The IDL-29 constants are (currently) per-SAE identical across layers unless
    # a per-SAE override exists; pass the request-level SAE's resolved constants
    # as the shared set (the single-layer path uses request.sae_id likewise).
    constants = resolve_constants(settings.steering_cluster_constants_json, request.sae_id)

    ml_members = [
        MultiLayerMember(
            feature_idx=m.feature_idx,
            layer=m.layer,
            similarity=m.similarity,
            activation_frequency=m.activation_frequency,
            sign=m.sign,
            sae_id=(getattr(m, "sae_id", None) or request.sae_id),
        )
        for m in request.members
    ]

    try:
        ml = compute_multi_layer_allocation(
            ml_members,
            decoders=decoders,
            constants=constants,
            constants_by_layer=constants_by_layer,  # per-SAE overrides (R1 ARCH-4)
            group_cohesion=request.group_cohesion,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Cross-layer hazards. PRIMARY = stored circuit edges at rung ≥2 (when a
    # circuit_id is supplied); FALLBACK = the labeled weight-prior heuristic.
    circuit_edges = None
    if request.circuit_id:
        circuit_edges = await _load_circuit_edges(db, request.circuit_id)

    # ml.strengths is returned in request-member order (R2 F8: a REAL runtime
    # check — not an assert, which `python -O` strips — so a future allocation-
    # partitioning refactor can't silently mis-pair strengths to members for
    # hazard sign detection).
    if len(ml.strengths) != len(request.members):
        raise HTTPException(
            500, "allocation strengths are not 1:1 with request members")
    steered = [
        {"layer": m.layer, "feature_idx": m.feature_idx, "strength": s}
        for m, s in zip(request.members, ml.strengths)
    ]

    # BR-016: a circuit's edges can be CLUSTER-level, and those are the ones
    # most worth having — an edge at rung >= 2 carries a measured effect size,
    # where the fallback is a weight-prior heuristic. `detect_hazards` keys on
    # `feature_idx` and a cluster endpoint has none, so every such edge used to
    # be skipped: steering a cluster-membered circuit silently discarded its
    # best evidence and still reported an empty hazard list. Expanded here,
    # where the session exists, and bounded to what is actually being steered.
    unresolved_edges: list = []
    if circuit_edges:
        keep = {
            (m["layer"], m["feature_idx"])
            for m in steered
            if m.get("feature_idx") is not None
        }
        wanted = [
            side.get("cluster_profile_id")
            for e in circuit_edges
            for side in ((e.get("up") or {}), (e.get("down") or {}))
        ]
        resolved = await _resolve_cluster_members(db, wanted)
        circuit_edges, unresolved_edges = steering_hazards.expand_cluster_edges(
            circuit_edges,
            resolved.get,
            keep=keep,
        )

    hazards = steering_hazards.detect_hazards(
        steered,
        circuit_edges=circuit_edges,
        decoders=decoders or None,
        encoders=encoders or None,
        prior_threshold=settings.steering_hazard_prior_threshold,
    )

    layers_out: dict = {}
    for layer, res in ml.layers.items():
        layers_out[str(layer)] = PerLayerAllocation(
            sae_id=ml.layer_sae_ids.get(layer),
            B=res.B,
            B_dir=res.B_dir,
            G=res.G,
            f_eff=res.f_eff,
            weights=res.weights,
            strengths=res.strengths,
            flags=res.flags,
            cancellation_pair=list(res.cancellation_pair) if res.cancellation_pair else None,
            constants_used=res.constants_used,
            approximate=res.approximate,
        )

    return MultiLayerAllocationResponse(
        formula_id=ml.formula_id,
        layers=layers_out,
        hazards=[HazardModel(**h.to_dict()) for h in hazards],
        strengths=ml.strengths,
        # "No hazards" and "not analysed" are different claims. An edge whose
        # cluster profile is missing or empty could not be checked, and saying
        # so is the whole point — an empty list that reads as safety is a bug
        # this project has shipped before.
        unchecked_edges=unresolved_edges,
    )

