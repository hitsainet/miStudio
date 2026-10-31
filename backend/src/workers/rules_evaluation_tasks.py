"""The rules ∪ probe evaluation task (Feature 040 S3; FR-37; FTDD §6.6; FTID §3.6, TI-19).

CPU only, on the `rules` queue consumed by its own `celery-worker-rules` (ENH-006 D52; `k8s/base/backend.yaml`, `docker-compose.yml`):
routed by its FULL name (`core/celery_app.py`'s `task_routes` — a short name silently lands on the default
queue, miStudio memory `celery-queue-split-and-name-routing`).

⚠ IDS IN, IDS AND COUNTS OUT. The argument is an evaluation id; the body is counts, rates, digests and positions.
Row text is read in chunks of `CHUNK_ROWS`, evaluated, and dropped; nothing logs it (FTID §13).

⚠ THE ROW IS COMMITTED BY THE API BEFORE DISPATCH, and a terminal row returns at once: an acks-late redelivery
after a lost worker is not a restart (miStudio memory `cancel-must-not-be-its-own-restart`).

⚠ THE CANCEL POLL LIVES HERE: `cancel_checker("probe_rules_evaluation", …)` is built in this module and handed
to the service as `should_stop`, checked between chunks of rows, so a cancel lands within one chunk.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..core.cancellation import cooperative_cancel
from ..core.celery_app import celery_app
from ..core.clock import utc_now
from .base_task import DatabaseTask

logger = logging.getLogger(__name__)

#: Words only: a failure names what happened, never a path, a value or an exception's text.
FAILURE_WORDS = {
    "split_changed": "a view's split changed since it was registered; register it again",
    "column_absent": "the column the rules read is not in a view's split",
    "inputs_removed": "an evaluation or view this request recorded is gone; request it again",
    "rule_set_cases_failed": "the rule set's own cases fail on this engine; fix the set",
    "rule_signal_error": "a validator failed on a row; see the miStudio log for the signal",
    "engine_error": "the rules evaluation could not complete; see the miStudio log",
}


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.rules_evaluation_tasks.run_rules_evaluation",
    max_retries=0,
)
@cooperative_cancel("probe_rules_evaluation")
def run_rules_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
    from ..core.cancellation import cancel_checker, record_progress
    from ..models.rules_evaluation import TERMINAL_STATUSES, ProbeRulesEvaluation
    from ..services import rules_evaluation
    from ..services.detector_next import DetectorRefused

    with self.get_db() as db:
        row = db.query(ProbeRulesEvaluation).filter(ProbeRulesEvaluation.id == evaluation_id).first()
        if row is None:
            raise ValueError(
                f"rules evaluation {evaluation_id} has no row. The API commits it before dispatch, so it was "
                "deleted — the task stops rather than doing work nothing can read"
            )
        if row.status in TERMINAL_STATUSES:
            return {"evaluation_id": evaluation_id, "status": row.status}
        row.status, row.started_at, row.heartbeat_at = "running", utc_now(), utc_now()
        db.commit()
        _task_queue(db, row.task_queue_id, "running")
        # Polled at EVERY chunk (no time throttle): one indexed read per `CHUNK_ROWS` rows is cheap, and a cancel
        # must land at the next chunk, not seconds later.
        checker = cancel_checker("probe_rules_evaluation", evaluation_id, min_interval_s=0)

        def should_stop() -> None:
            checker.raise_if_cancelled("between chunks of rows")

        def beat(progress: Dict[str, Any]) -> None:
            record_progress("probe_rules_evaluation", evaluation_id, heartbeat_at=utc_now(), progress=progress)

        try:
            body = rules_evaluation.run(db, evaluation_id, should_stop=should_stop, beat=beat)
            should_stop()
            outcome = rules_evaluation.finish(db, evaluation_id, body)
            _task_queue(db, row.task_queue_id, "completed" if outcome["status"] == "completed" else "failed")
            logger.info("rules evaluation %s %s (%d views)", evaluation_id, outcome["status"], len(body["views"]))
            return outcome
        except BaseException as exc:
            from ..core.cancellation import OperatorCancelled

            if isinstance(exc, OperatorCancelled):
                _task_queue(db, row.task_queue_id, "cancelled")
                raise
            code = exc.code if isinstance(exc, DetectorRefused) and exc.code in FAILURE_WORDS else "engine_error"
            if type(exc).__name__ == "RuleSignalError":
                code = "rule_signal_error"
            logger.error("rules evaluation %s failed: %s (%s)", evaluation_id, code, type(exc).__name__)
            _fail(db, evaluation_id, code)
            _task_queue(db, row.task_queue_id, "failed")
            return {"evaluation_id": evaluation_id, "status": "failed", "failure_code": code}


def _fail(db: Any, evaluation_id: str, code: str) -> None:
    from ..models.rules_evaluation import ProbeRulesEvaluation

    db.rollback()
    row = db.query(ProbeRulesEvaluation).filter(ProbeRulesEvaluation.id == evaluation_id).populate_existing().first()
    if row is None or row.status in ("completed", "cancelled"):
        return
    row.status, row.completed_at = "failed", utc_now()
    row.error = {"code": code, "message": FAILURE_WORDS.get(code, FAILURE_WORDS["engine_error"])}
    db.commit()


def _task_queue(db: Any, task_queue_id: Any, status: str) -> None:
    """Mirror the status on the `task_queue` row `get_task_status` polls (040-P7)."""
    if not task_queue_id:
        return
    from ..models.task_queue import TaskQueue

    entry = db.query(TaskQueue).filter(TaskQueue.id == task_queue_id).first()
    if entry is None:
        return
    entry.status = status
    if status == "running":
        entry.started_at = entry.started_at or utc_now()
    elif status in ("completed", "failed", "cancelled"):
        entry.completed_at = utc_now()
        entry.progress = 100.0 if status == "completed" else entry.progress
    db.commit()
