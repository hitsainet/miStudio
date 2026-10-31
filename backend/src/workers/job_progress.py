"""Is the work advancing? — the question the stuck-job reapers never asked.

Every janitor decides a job is dead from ONE signal: `updated_at` has been quiet
for longer than a fixed threshold. That answers "has this row been written to",
which is not the same question, and it is wrong in both directions:

  * A job legitimately waiting its turn is condemned. On 2026-08-28 the third
    member of a three-SAE extraction batch was failed at 186 minutes with
    "no progress ... may indicate a crashed worker". Nothing had crashed. Job 1
    had completed, job 2 was mid-run, and job 3 had never started because only
    the first member of a batch is dispatched. One job alone takes ~169 minutes,
    so the third of any three-job batch is structurally guaranteed to die
    against a 180-minute grace period.

  * A job wedged mid-run is NOT caught. `update_extraction_status_sync`
    reassigns `status` and `updated_at` unconditionally, so a task looping
    without advancing keeps its row fresh forever and looks healthy.

This module supplies the missing evidence. It is ADDITIVE: callers require both
the existing clock/liveness verdict AND a stalled counter before condemning
anything, so a slow-but-advancing job is protected while a wedged one is newly
detectable.

State lives in Redis rather than a new column: it is derived, cheap to lose (a
lost marker just re-arms the grace period), and needs no migration across the
seven lifecycles that want it.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import redis

from src.core.config import settings

logger = logging.getLogger(__name__)

#: Markers outlive any plausible janitor period but not a stale deployment.
MARKER_TTL_SECONDS = 24 * 60 * 60

_KEY_PREFIX = "janitor:progress"


def _client() -> Optional["redis.Redis"]:
    """A Redis client, or None if one cannot be had.

    Returning None rather than raising matters: this is supporting evidence for
    a DESTRUCTIVE decision. If Redis is unreachable the caller must fall back to
    its previous behaviour, not gain or lose the ability to reap.
    """
    try:
        return redis.from_url(str(settings.redis_url), decode_responses=True)
    except Exception as exc:  # pragma: no cover - configuration/runtime only
        logger.warning("job_progress: no Redis client (%s)", exc)
        return None


def _key(kind: str, row_id: Any) -> str:
    return f"{_KEY_PREFIX}:{kind}:{row_id}"


def progress_stalled_seconds(
    kind: str,
    row_id: Any,
    counter: Any,
    *,
    now: Optional[datetime] = None,
    client: Optional["redis.Redis"] = None,
) -> Optional[float]:
    """How long `counter` has been unchanged for this row, in seconds.

    Returns None when that cannot be answered — no Redis, an unusable counter,
    or the first sighting of this row. None means "no evidence", and callers
    MUST treat it as "do not reap on this basis", never as zero.

    `counter` is whatever monotonic value the lifecycle advances: `progress`,
    `current_step`, `samples_processed`, `examples_completed`. Each janitor
    names its own; this only needs to compare it with the previous sighting.
    """
    if counter is None:
        return None

    now = now or datetime.now(timezone.utc)
    client = client or _client()
    if client is None:
        return None

    key = _key(kind, row_id)
    marker = {"value": str(counter), "first_seen_at": now.isoformat()}

    try:
        raw = client.get(key)
        if raw is None:
            # First sighting: start the clock, claim nothing.
            client.set(key, json.dumps(marker), ex=MARKER_TTL_SECONDS)
            return None

        previous = json.loads(raw)
        if previous.get("value") != marker["value"]:
            # It moved. Re-arm and report no stall.
            client.set(key, json.dumps(marker), ex=MARKER_TTL_SECONDS)
            return 0.0

        first_seen = datetime.fromisoformat(previous["first_seen_at"])
        if first_seen.tzinfo is None:
            first_seen = first_seen.replace(tzinfo=timezone.utc)

        # Refresh the TTL without moving first_seen_at, so a long stall on a
        # quiet system cannot expire its own evidence.
        client.expire(key, MARKER_TTL_SECONDS)
        return max(0.0, (now - first_seen).total_seconds())
    except Exception as exc:
        logger.warning("job_progress: marker read failed for %s (%s)", key, exc)
        return None


def clear_progress_marker(
    kind: str, row_id: Any, *, client: Optional["redis.Redis"] = None
) -> None:
    """Forget a row, so a retry of the same id starts with a clean clock."""
    client = client or _client()
    if client is None:
        return
    try:
        client.delete(_key(kind, row_id))
    except Exception as exc:  # pragma: no cover
        logger.debug("job_progress: could not clear marker (%s)", exc)


def batch_last_activity(db, batch_id: str) -> Optional[datetime]:
    """The most recent `updated_at` across every member of an extraction batch.

    A queued batch member has no Celery task and no progress of its own — only
    the FIRST member is dispatched, and `_start_next_batch_job` starts each
    successor after the previous one's NLP completes. Its own age therefore says
    nothing about its health; the batch's does.

    Used so a member waiting its turn is judged by whether the CHAIN is
    advancing. Self-scaling: a three-job or thirty-job batch needs no threshold
    tuning, and a genuinely broken chain still goes silent and is still
    reclaimed.
    """
    from sqlalchemy import func

    from src.models.extraction_job import ExtractionJob

    if not batch_id:
        return None

    latest = (
        db.query(func.max(ExtractionJob.updated_at))
        .filter(ExtractionJob.batch_id == batch_id)
        .scalar()
    )
    if latest is None:
        return None
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    return latest


def batch_has_live_sibling(db, batch_id: str, exclude_id: Any = None) -> bool:
    """Is any other member of this batch still running or about to?

    Distinguishes "waiting behind live work" from "the chain died". Only the
    second justifies condemning a queued member.
    """
    from src.models.extraction_job import ExtractionJob, ExtractionStatus

    if not batch_id:
        return False

    query = db.query(ExtractionJob).filter(
        ExtractionJob.batch_id == batch_id,
        ExtractionJob.status.in_(
            [ExtractionStatus.QUEUED.value, ExtractionStatus.EXTRACTING.value]
        ),
        ExtractionJob.celery_task_id.isnot(None),
    )
    if exclude_id is not None:
        query = query.filter(ExtractionJob.id != exclude_id)
    return db.query(query.exists()).scalar() is True
