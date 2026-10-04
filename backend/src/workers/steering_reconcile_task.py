"""
Steering worker reconcile beat task.

The dedicated steering worker exits after every generation task (the solo
pool ignores --max-tasks-per-child, so it self-terminates to free VRAM).
Any death mode — self-exit, crash, external kill — can leave queued or
broker-restored steering messages with no consumer; historically they sat
until the next submit or a manual enter-mode.

This beat task closes the gap from the main worker: every cycle it asks the
backend (which shares a PID namespace with the steering worker — this
container does NOT) to spawn a worker if the steering queue is non-empty
and no worker is alive. The decision logic lives server-side in
/api/internal/steering/reconcile-worker.
"""

import logging

import httpx
from celery import shared_task

from ..core.config import settings

logger = logging.getLogger(__name__)


@shared_task(name="steering_worker_reconcile", queue="low_priority", ignore_result=True)
def steering_worker_reconcile() -> dict:
    url = f"{settings.internal_api_url}/api/internal/steering/reconcile-worker"
    try:
        response = httpx.post(
            url,
            headers={"X-Internal-Token": settings.internal_api_secret},
            timeout=15.0,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("action") == "spawned":
            logger.warning(
                "Steering reconcile: spawned worker PID %s for %s stranded task(s)",
                result.get("worker_pid"), result.get("queue_depth"),
            )
        return result
    except Exception as e:
        # Never crash the beat cycle — the next tick retries.
        logger.warning("Steering reconcile call failed: %s", e)
        return {"status": "error", "error": str(e)}
