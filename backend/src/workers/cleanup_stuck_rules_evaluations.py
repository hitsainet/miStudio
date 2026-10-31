"""Reclaim rules evaluations whose worker is gone, and prune the verdict cache (Feature 040 S3; FR-37; FTDD §6.6).

Swept every five minutes from beat (`cleanup-stuck-rules-evaluations`, queue `low_priority`):

* a `running` row whose heartbeat is older than the runtime setting `rules_evaluation_stale_after_seconds` →
  `failed`, `worker_lost`, with a rerun `next` (a rollout or an OOM kill took its worker);
* the verdict cache is pruned by age (`rule_verdict_cache_max_age_days`) and size
  (`rule_verdict_cache_max_bytes`), least recently used first — NEVER an entry a pending or running
  evaluation is reading.

⚠ NEVER REQUEUES. A reaped row stays failed with its reason; `evaluate_detector_with_rules` is the rerun.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ..core.celery_app import celery_app
from .base_task import DatabaseTask

logger = logging.getLogger(__name__)

WORKER_LOST = "the worker running this evaluation stopped reporting; nothing was written. Request it again."


def sweep(db, now=None) -> dict:
    from ..core.runtime_settings import effective
    from ..models.rules_evaluation import ProbeRulesEvaluation
    from ..services import rules_evaluation
    from ..services.detector_next import next_rules_evaluation

    settings = effective()
    now = now or datetime.now(timezone.utc)
    stale_after = timedelta(seconds=int(settings.rules_evaluation_stale_after_seconds))
    reaped = 0
    for row in db.query(ProbeRulesEvaluation).filter(ProbeRulesEvaluation.status == "running").all():
        last = row.heartbeat_at or row.started_at or row.created_at
        if last is not None and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if last is None or now - last <= stale_after:
            continue
        row.status, row.completed_at = "failed", now
        row.error = {"code": "worker_lost", "message": WORKER_LOST,
                     "next": [next_rules_evaluation(row.probe_id, list(row.view_ids), row.behaviour_version)]}
        db.commit()
        logger.warning("rules evaluation %s reaped: worker_lost", row.id)
        reaped += 1
    pruned = rules_evaluation.prune_cache(db, max_bytes=int(settings.rule_verdict_cache_max_bytes),
                                          max_age_days=int(settings.rule_verdict_cache_max_age_days), now=now)
    return {"worker_lost": reaped, "cache_pruned": pruned}


@celery_app.task(bind=True, base=DatabaseTask, name="cleanup_stuck_rules_evaluations")
def cleanup_stuck_rules_evaluations(self) -> dict:
    with self.get_db() as db:
        return sweep(db)
