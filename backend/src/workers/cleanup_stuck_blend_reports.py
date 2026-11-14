"""Reclaim blend reports whose worker is gone (Feature 041 S2; T31, FTID §12).

A blend report counts against `blend_max_concurrent` while it is `queued` or `running`, so a report whose
worker vanished (a rollout, an OOM kill) would hold a slot forever and refuse every later start. Swept
every ten minutes from beat (`cleanup-stuck-blend-reports`):

* `running` whose heartbeat (written between folds and candidates) is older than `STALE_HEARTBEAT_S` →
  `failed` / `worker_lost`;
* `queued` or `running` older than the runtime setting `blend_evaluation_time_limit_s` (by heartbeat, else
  creation) → `failed` / `time_limit` — this is how a report queued while the blend worker is down frees its
  slot.

⚠ NEVER REQUEUES. A reaped report stays failed with its reason; `rerun_blend_report` is the agent's call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ..core.celery_app import celery_app
from .base_task import DatabaseTask

logger = logging.getLogger(__name__)

#: The task beats between folds and between candidates; no single step is near this long.
HEARTBEAT_INTERVAL_S = 200
STALE_HEARTBEAT_S = 3 * HEARTBEAT_INTERVAL_S

WORDS = {
    "worker_lost": "the worker running this report stopped reporting; nothing was written. Rerun the report.",
    "time_limit": "this report did not finish within the blend evaluation time limit; rerun it (and check the "
                  "blend worker is running).",
}


def sweep(db, now=None) -> dict:
    from ..core.runtime_settings import effective
    from ..models.probe_blend import ProbeBlendReport
    from ..services.probe_blend.service import remove_staging

    now = now or datetime.now(timezone.utc)
    limit = timedelta(seconds=int(effective().blend_evaluation_time_limit_s))
    reaped = {"worker_lost": 0, "time_limit": 0}
    for row in db.query(ProbeBlendReport).filter(ProbeBlendReport.status.in_(("queued", "running"))).all():
        last = row.heartbeat_at or row.created_at
        if last is not None and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        code = None
        if last is not None and now - last > limit:
            code = "time_limit"
        elif row.status == "running" and row.heartbeat_at is not None and \
                now - last > timedelta(seconds=STALE_HEARTBEAT_S):
            code = "worker_lost"
        if code is None:
            continue
        row.status, row.failure_code, row.failure_message = "failed", code, WORDS[code]
        row.completed_at = now
        db.commit()
        remove_staging(row.id)
        logger.warning("blend report %s reaped: %s", row.id, code)
        reaped[code] += 1
    return reaped


@celery_app.task(bind=True, base=DatabaseTask, name="cleanup_stuck_blend_reports")
def cleanup_stuck_blend_reports(self) -> dict:
    with self.get_db() as db:
        return sweep(db)
