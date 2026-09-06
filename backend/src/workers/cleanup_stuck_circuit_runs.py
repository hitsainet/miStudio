"""
Periodic cleanup for stuck circuit capture/discovery/attribution runs
(Feature 016, R2 Q3).

Without this, an OOM-killed or pod-restarted capture leaves its row in
'running' forever — and `assert_no_active_gpu_run` then rejects EVERY future
capture with a 409 (a permanent lockout). Mirrors cleanup_stuck_extractions:
if a run has had no update past a threshold AND its Celery task is no longer
active, mark it failed and rmtree any partial store.
"""

import logging
import shutil
from datetime import datetime, timedelta, timezone

from src.core.celery_app import celery_app
from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from src.workers.base_task import DatabaseTask

from src.workers.job_progress import progress_stalled_seconds

logger = logging.getLogger(__name__)

STUCK_THRESHOLD_MINUTES = 60  # no update in an hour + task not active → stuck
#: Statuses a run only reaches once its task has REPORTED progress. The row's
#: own clock is meaningful for these and meaningless for "pending", which is
#: what a run looks like while queued behind a 45-minute fit — condemning those
#: on row age would clear the queue every time one job ran long.
_STARTED_STATUSES = ("estimating", "running")


class _SubLifecycleView:
    """Presents ONE lifecycle of a multi-lifecycle row to `_is_abandoned`.

    Attribution, validation, faithfulness and calibration each carry their own
    task id and status on a row shared with something else, so each needs the
    same treatment without pretending to be the row's primary task.
    `updated_at` is shared, which is the best clock available.

    APPLIED TO EVERY ONE OF THEM, not just the first. Fixing one representative
    and leaving the siblings is the failure mode this repo has hit repeatedly —
    and here the siblings are worse than cosmetic: calibration, faithfulness and
    recording all hold the single-GPU guard, so any one of them stuck at PENDING
    locks out every circuit task exactly like the capture did.
    """

    def __init__(self, task_id, updated_at):
        self.celery_task_id = task_id
        self.updated_at = updated_at


def _is_abandoned(run, status: str) -> bool:
    """Whether this row's task is gone, including when Celery cannot say so.

    THE REASON THIS EXISTS. The rule this replaced treated PENDING as alive,
    and
    Celery reports PENDING for any task id it holds no result for — which is
    exactly what an OOM-killed worker leaves behind, indistinguishable from a
    task that has not started yet. So the one failure this janitor was written
    for was the one it could never clear: `cap_cda1e1da6a0a` sat at "running
    45.6%" with the GPU idle while this task ran every ten minutes returning
    `{'cleaned': 0}`, and because `assert_no_active_gpu_run` counts any
    non-terminal row, every new capture was refused with a 409.

    `looks_abandoned` already solved this for the task-queue surface, down to a
    docstring describing the same symptom on a J-lens fit. It was never carried
    across. This is that fix, generalised.

    Row age is only consulted for statuses that mean the task HAS run — see
    `_STARTED_STATUSES`. A "pending" row keeps the old, conservative rule.
    """
    from src.workers.task_heartbeat import looks_abandoned, seconds_since_row_update

    # IS THE WORK ADVANCING? Asked before anything else, because a row whose
    # counter is still moving is alive whatever Celery says about it. Additive
    # and conservative: only a POSITIVE recent advance spares the row. None —
    # no counter on this lifecycle (steering record runs have none), a first
    # sighting, or Redis unreachable — falls through to the rule below
    # unchanged.
    counter = getattr(run, "progress", None)
    if counter is not None:
        stalled = progress_stalled_seconds(
            f"circuit:{type(run).__name__}", getattr(run, "id", None), counter
        )
        if stalled is not None and stalled < STUCK_THRESHOLD_MINUTES * 60:
            return False

    task_id = getattr(run, "celery_task_id", None)
    if not task_id:
        # No task was ever recorded, and the row is already past the staleness
        # filter — nothing can be waiting on it.
        return True
    try:
        result = celery_app.AsyncResult(task_id, app=celery_app)
        state, info = result.state, result.info
    except Exception:  # broker hiccup — treat as alive, never false-kill
        return False

    age = seconds_since_row_update(run) if status in _STARTED_STATUSES else None
    return looks_abandoned(state, info, age)


@celery_app.task(bind=True, base=DatabaseTask, name="cleanup_stuck_circuit_runs")
def cleanup_stuck_circuit_runs_task(self):
    """Fail circuit runs stuck past the threshold with no active task."""
    threshold = datetime.now(timezone.utc) - timedelta(
        minutes=STUCK_THRESHOLD_MINUTES)
    cleaned = 0
    with self.get_db() as db:
        # ── captures ──
        for run in db.query(CircuitCaptureRun).filter(
                CircuitCaptureRun.status.in_(("pending", "estimating", "running")),
                CircuitCaptureRun.updated_at < threshold).all():
            if not _is_abandoned(run, run.status):
                continue
            run.status = "failed"
            run.error_message = "Stuck run reclaimed by cleanup (worker died?)"
            if run.store_path:
                try:
                    from src.core.config import settings
                    p = settings.resolve_data_path(run.store_path)
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    logger.exception("rmtree failed for %s", run.id)
            cleaned += 1
        # ── discovery + attribution lifecycles ──
        for run in db.query(CircuitDiscoveryRun).filter(
                CircuitDiscoveryRun.updated_at < threshold).all():
            if run.status in ("pending", "running") and _is_abandoned(
                    run, run.status):
                run.status = "failed"
                run.error_message = "Stuck discovery reclaimed by cleanup"
                cleaned += 1
            if run.attribution_status in ("pending", "running") and _is_abandoned(
                    _SubLifecycleView(run.attribution_task_id, run.updated_at),
                    run.attribution_status):
                run.attribution_status = "failed"
                run.attribution_error = "Stuck attribution reclaimed by cleanup"
                cleaned += 1
            if run.validation_status in ("pending", "running") and _is_abandoned(
                    _SubLifecycleView(run.validation_task_id, run.updated_at),
                    run.validation_status):
                run.validation_status = "failed"
                run.validation_error = "Stuck validation reclaimed by cleanup"
                cleaned += 1
        # ── faithfulness (runs on a circuit, not a discovery run) — R2 B-5 ──
        from src.models.circuit import Circuit
        for circuit in db.query(Circuit).filter(
                Circuit.faithfulness_status.in_(("pending", "running")),
                Circuit.updated_at < threshold).all():
            if _is_abandoned(
                    _SubLifecycleView(circuit.faithfulness_task_id,
                                      circuit.updated_at),
                    circuit.faithfulness_status):
                circuit.faithfulness_status = "failed"
                cleaned += 1
        # ── calibration (also runs on a circuit + holds the GPU) — Feature 20 ──
        # A crashed calibration would otherwise wedge the single-GPU guard for
        # every circuit task, since assert_no_active_gpu_run checks this status.
        for circuit in db.query(Circuit).filter(
                Circuit.calibration_status.in_(("pending", "running")),
                Circuit.updated_at < threshold).all():
            if _is_abandoned(
                    _SubLifecycleView(circuit.calibration_task_id,
                                      circuit.updated_at),
                    circuit.calibration_status):
                circuit.calibration_status = "failed"
                cleaned += 1
        # ── steered-transcript recording (own marker table) — Recorder ──
        # A crashed record job wedges the single-GPU guard just like calibration.
        from src.models.steering_record_run import SteeringRecordRun
        for rec in db.query(SteeringRecordRun).filter(
                SteeringRecordRun.status.in_(("pending", "running")),
                SteeringRecordRun.updated_at < threshold).all():
            if _is_abandoned(_SubLifecycleView(rec.task_id, rec.updated_at),
                             rec.status):
                rec.status = "failed"
                cleaned += 1
        if cleaned:
            db.commit()
            logger.info("Reclaimed %d stuck circuit run(s)", cleaned)
    return {"cleaned": cleaned}
