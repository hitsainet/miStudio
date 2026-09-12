"""
Celery tasks for circuit capture + discovery + attribution (Feature 016).

Routing: extraction queue (GPU profile — same as activation/feature
extraction; the steering worker's busy-marker machinery is steering-specific
and deliberately NOT used here). Cancellation: DB-status polling between
batches (house pattern — training_tasks precedent).
"""

import logging
from typing import Any, Dict, Optional

from ..core.celery_app import celery_app
from .base_task import DatabaseTask
from .websocket_emitter import (
    emit_circuit_run_completed,
    emit_circuit_run_failed,
    emit_circuit_run_progress,
)

logger = logging.getLogger(__name__)


#: Which registered scope each (model, status_field) pair corresponds to. The
#: shim keeps `_cancel_checker`'s signature so its four call sites are
#: untouched, but the vocabulary now lives in ONE place.
_SCOPE_FOR = {
    ("CircuitCaptureRun", "status"): "circuit_capture",
    ("CircuitDiscoveryRun", "status"): "circuit_discovery",
    ("CircuitDiscoveryRun", "attribution_status"): "circuit_attribution",
    ("CircuitDiscoveryRun", "validation_status"): "circuit_validation",
}


def _cancel_checker(db, model_cls, run_id, status_field="status"):
    """Throttled DB-status poll — returns a callable for the service loop.

    NOW A SHIM OVER `core.cancellation.cancel_checker`. This convention was
    already healthy; what it was not was shared. Three near-identical copies of
    the same poll existed, each with its own guess at a throttle.

    THE THROTTLE IS NOW TIME, NOT COUNT. This polled every 5th call, which is a
    number chosen against one loop's unit cost and true of no other: over
    attribution batches on a large model `% 5` is up to twenty minutes of
    latency. A 2-second budget makes the caller's rule simply "call me at the
    finest boundary you can cleanly abandon work at".

    `db` is still accepted and still passed through — these services poll on the
    caller's task session by design, and `CancelCheck` re-reads with
    `populate_existing()` so that remains safe.
    """
    from ..core.cancellation import cancel_checker

    try:
        kind = _SCOPE_FOR[(model_cls.__name__, status_field)]
    except KeyError:
        raise KeyError(
            f"no cancel scope for {model_cls.__name__}.{status_field}; add one "
            f"in src/core/cancellation.py rather than inventing a second "
            f"convention"
        ) from None
    return cancel_checker(kind, run_id, db=db)


def _failure_detail(exc: BaseException) -> str:
    """`TypeName: message (file:line)` — enough to act on without the pod log."""
    import traceback

    where = ""
    tb = exc.__traceback__
    if tb is not None:
        frame = traceback.extract_tb(tb)[-1]
        where = f" ({frame.filename.rsplit('/', 1)[-1]}:{frame.lineno})"
    return f"{type(exc).__name__}: {exc}{where}"


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_capture_tasks.capture_circuit_activations",
                 max_retries=0)
def capture_circuit_activations(self, run_id: str, confirmed: bool = False) -> Dict[str, Any]:
    """Probe (estimate) and, when confirmed, full capture for one run."""
    from ..models.circuit_runs import CircuitCaptureRun
    from ..services.circuit_capture_service import CircuitCaptureService

    with self.get_db() as db:
        try:
            result = CircuitCaptureService.run_capture(
                db, run_id, confirmed=confirmed,
                cancel_check=_cancel_checker(db, CircuitCaptureRun, run_id),
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "capture", run_id, pct),
            )
            emit_circuit_run_completed("capture", run_id, summary=result)
            return result
        except Exception as e:
            logger.exception("Circuit capture %s failed", run_id)
            # TYPE AND FRAME, NOT JUST THE MESSAGE. `str(IndexError(...))` is
            # exactly "tuple index out of range" — no exception type, no file,
            # no line — and that string is the whole of what the UI and the API
            # ever showed. A real failure took a log dive on the worker pod to
            # identify. The frame is what makes the next one self-diagnosing.
            detail = _failure_detail(e)
            run = db.query(CircuitCaptureRun).filter(
                CircuitCaptureRun.id == run_id).first()
            if run is not None:
                run.status = "failed"
                run.error_message = detail[:2000]
                db.commit()
            emit_circuit_run_failed("capture", run_id, detail[:500])
            raise


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_capture_tasks.run_circuit_discovery",
                 max_retries=0)
def run_circuit_discovery(self, run_id: str) -> Dict[str, Any]:
    """Statistical mining over a completed capture store (CPU-heavy, no GPU)."""
    from ..models.circuit_runs import CircuitDiscoveryRun
    from ..services.circuit_discovery_service import CircuitDiscoveryService

    with self.get_db() as db:
        try:
            result = CircuitDiscoveryService.run(
                db, run_id,
                cancel_check=_cancel_checker(db, CircuitDiscoveryRun, run_id),
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "discovery", run_id, pct),
            )
            emit_circuit_run_completed("discovery", run_id, summary=result)
            return result
        except Exception as e:
            logger.exception("Circuit discovery %s failed", run_id)
            run = db.query(CircuitDiscoveryRun).filter(
                CircuitDiscoveryRun.id == run_id).first()
            if run is not None:
                run.status = "failed"
                run.error_message = str(e)[:2000]
                db.commit()
            emit_circuit_run_failed("discovery", run_id, str(e)[:500])
            raise


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_capture_tasks.run_circuit_attribution",
                 max_retries=0)
def run_circuit_attribution(self, run_id: str,
                            prompt_limit: Optional[int] = None) -> Dict[str, Any]:
    """Tier-2 gradient attribution pass over a discovery run's candidates (GPU)."""
    from ..models.circuit_runs import CircuitDiscoveryRun
    from ..services.circuit_attribution_service import CircuitAttributionService

    with self.get_db() as db:
        try:
            result = CircuitAttributionService.run(
                db, run_id, prompt_limit=prompt_limit,
                cancel_check=_cancel_checker(db, CircuitDiscoveryRun, run_id,
                                             status_field="attribution_status"),
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "attribution", run_id, pct),
            )
            emit_circuit_run_completed("attribution", run_id, summary=result)
            return result
        except Exception as e:
            logger.exception("Circuit attribution %s failed", run_id)
            run = db.query(CircuitDiscoveryRun).filter(
                CircuitDiscoveryRun.id == run_id).first()
            if run is not None:
                # Attribution's OWN lifecycle — the completed discovery's
                # status/report/candidates are untouched (R1 QA-P2).
                run.attribution_status = "failed"
                run.attribution_error = str(e)[:2000]
                db.commit()
            emit_circuit_run_failed("attribution", run_id, str(e)[:500])
            raise
