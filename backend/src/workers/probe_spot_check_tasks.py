"""Spot-check tasks (Feature 038; FTDD §6). CPU and HTTP only: no GPU lease (D-T3).

⚠ IDS IN, IDS AND COUNTS OUT (FR-12, D35). The arguments are a spot-check id; the result is its status
and item count. No transcript, judged excerpt or judge reply is ever an argument, a result or a log line.

⚠ THE ROW IS COMMITTED BY THE API BEFORE DISPATCH, AND THIS TASK REFUSES WITHOUT ONE (the
probe_monitor_tasks precondition, kept beside the guard).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..core.celery_app import celery_app
from ..core.clock import utc_now
from .base_task import DatabaseTask

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.probe_spot_check_tasks.run_probe_spot_check",
    max_retries=0,
)
def run_probe_spot_check(self, spot_check_id: str) -> Dict[str, Any]:
    from ..models.probe_spot_check import ProbeSpotCheck
    from ..services.probe_spot_check import execute

    with self.get_db() as db:
        if db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == spot_check_id).first() is None:
            raise ValueError(
                f"spot-check {spot_check_id} has no row. The API commits it before dispatch, so it was "
                f"deleted — the task stops rather than doing work nothing can read"
            )
        try:
            return execute(db, spot_check_id)
        except Exception as exc:
            logger.error("spot-check %s failed: %s", spot_check_id, type(exc).__name__)
            db.rollback()
            row = db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == spot_check_id).first()
            if row is not None:
                row.status = "failed"
                row.error_message = f"{type(exc).__name__}: {exc}"[:2000]
                row.completed_at = utc_now()
                db.commit()
            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.probe_spot_check_tasks.expire_spot_check_excerpts",
)
def expire_spot_check_excerpts(self) -> Dict[str, int]:
    """D39: judged text and replies are kept seven days unless the run is pinned."""
    from ..services.probe_spot_check import expire_excerpts

    with self.get_db() as db:
        return {"items_cleared": expire_excerpts(db)}
