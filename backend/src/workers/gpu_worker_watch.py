"""The beat check for a worker that finished a task and never acked it (ENH-006 D116).

Seen live on 2026-11-02: "succeeded" logged, the message still in `unacked`, messages queued, the GPU idle and the log
silent. Nothing noticed. This runs every minute on the CPU janitors' queue, so it does not depend on the worker it watches.

It only reports: `logger.error` every pass while it lasts, and one failed `task_queue` row per stuck task, which is what
the System Monitor's Failed Operations section shows. MCP callers read `services.gpu_queue_state.assess` through
`get_gpu_status`, `list_active_jobs` and `gpu_preflight`. IT NEVER RESTARTS ANYTHING: a restart would destroy the only
evidence of why, and kills whatever else that process is doing. That is the operator's.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from ..core.celery_app import celery_app
from ..core.clock import utc_now
from ..core.database import get_sync_db
from ..models.task_queue import TaskQueue
from ..services import gpu_queue_state

logger = logging.getLogger(__name__)

#: `task_queue.task_type` of the row an operator sees. `entity_id` is the stuck task's id.
WEDGE_TASK_TYPE = "gpu_worker_wedged"


def _report(db: Any, answer: Dict[str, Any]) -> None:
    for item in answer["stuck"]:
        existing = db.query(TaskQueue).filter(
            TaskQueue.task_type == WEDGE_TASK_TYPE, TaskQueue.entity_id == item["task_id"]
        ).first()
        if existing is None:
            db.add(TaskQueue(id=f"tq_{uuid.uuid4().hex[:12]}", task_id=None, task_type=WEDGE_TASK_TYPE,
                             entity_id=item["task_id"], entity_type="worker", status="failed",
                             error_message=answer["message"], retry_params={}, retry_count=0))
        else:
            existing.status = "failed"
            existing.error_message = answer["message"]
    db.commit()


def _resolve(db: Any) -> None:
    """The state has cleared: a row saying a worker is wedged must not outlive it."""
    rows = db.query(TaskQueue).filter(TaskQueue.task_type == WEDGE_TASK_TYPE, TaskQueue.status == "failed").all()
    for row in rows:
        row.status = "completed"
        row.completed_at = utc_now()
    if rows:
        db.commit()


@celery_app.task(name="watch_gpu_workers", ignore_result=True)
def watch_gpu_workers(*, client: Optional[Any] = None) -> Dict[str, Any]:
    """One pass. Keeps the first-seen mark, and reports while the state is `wedged`."""
    client = client if client is not None else gpu_queue_state._client()
    now = time.time()
    answer = gpu_queue_state.assess(client=client, now=now)
    if answer["state"] == "unknown":
        logger.warning("Worker watch: %s", answer.get("reason"))
        return answer
    if answer.get("condition_holds"):
        client.set(gpu_queue_state.FIRST_SEEN_KEY, str(now), nx=True, ex=gpu_queue_state.FIRST_SEEN_TTL_S)
    else:
        client.delete(gpu_queue_state.FIRST_SEEN_KEY)
    with get_sync_db() as db:
        if answer["state"] == "wedged":
            logger.error("WORKER WEDGED: %s", answer["message"])
            _report(db, answer)
        elif answer["state"] == "ok":
            _resolve(db)
    return answer
