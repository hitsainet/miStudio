"""A task id that has already completed does nothing when its message comes back (ENH-006 D116).

A row can say a job has ended, and `core.cancellation.refuses_terminal_row` reads it. An evaluation, a re-cut, a
score or a readout has no such row: a redelivered copy repeats the work, and a re-cut moves a probe's bars again.
A restart redelivers any message that was never acked, and a worker that wedges after finishing leaves one behind.

So the wrapper every GPU task runs under (`workers/gpu_job.py`) stamps the task id here when the task returns, and a
later delivery of that id does nothing. A second request made on purpose is a new Celery task with a new id, so it
runs. A hand-off keeps its id and raises rather than returns, so it is never stamped.

KEYED IN THE BROKER'S REDIS, with a TTL well past the 12-hour visibility timeout: a message restored at the timeout can
then wait behind other work. The result backend cannot do this job: its entries expire in an hour (`result_expires`).

Never raises. Unreadable means "not known to have completed", and the task runs, as it did before this existed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

DONE_PREFIX = "mistudio:task:done:"

#: 7 days: past the 43,200 s visibility timeout, plus time queued behind other work once restored.
DONE_TTL_S = 7 * 24 * 3600


def _client() -> Any:
    import redis

    from ..core.config import settings

    return redis.Redis.from_url(str(settings.celery_broker_url), socket_connect_timeout=1.0, socket_timeout=2.0)


def record_completed(task_id: Optional[str], name: Optional[str] = None, *, client: Any = None) -> None:
    """The task returned. Never raises: at worst a redelivery repeats the work, as it always did."""
    if not task_id:
        return
    stamp = {"completed_at": datetime.now(timezone.utc).isoformat(), "task": name}
    try:
        (client if client is not None else _client()).set(DONE_PREFIX + str(task_id), json.dumps(stamp), ex=DONE_TTL_S)
    except Exception:  # noqa: BLE001
        logger.warning("Could not record that task %s completed; a redelivery would repeat it", task_id, exc_info=True)


def completed_at(task_id: Optional[str], *, client: Any = None) -> Optional[str]:
    """When this task id completed, ISO 8601, or None when it has not or the broker cannot say."""
    if not task_id:
        return None
    try:
        raw = (client if client is not None else _client()).get(DONE_PREFIX + str(task_id))
    except Exception:  # noqa: BLE001
        logger.warning("Could not read whether task %s already completed; running it", task_id, exc_info=True)
        return None
    if raw is None:
        return None
    try:
        return str(json.loads(raw)["completed_at"])
    except (ValueError, KeyError, TypeError):
        return "unknown"
