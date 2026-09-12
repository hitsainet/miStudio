"""Celery task for circuit strength calibration (Feature 20 / IDL-37).

Calibration runs ON a circuit and holds the GPU like faithfulness, so it shares
the same lifecycle: a `calibration_status`/`_task_id` pair on the circuit row
(seen by the single-GPU guard + reclaimed by cleanup), a `calibration` manifest
as the reproducible record, and the band written onto the circuit through the
contract (apply_calibration → update() → validators + version bump).

GPU profile → extraction queue (same as capture/faithfulness). The single-GPU
guard is asserted at the endpoint (advisory lock) before dispatch.
"""

import logging
from typing import Any, Dict, Optional

from ..core.cancellation import OperatorCancelled, cancel_checker
from ..core.celery_app import celery_app
from .base_task import DatabaseTask
from .websocket_emitter import (
    emit_circuit_run_completed,
    emit_circuit_run_failed,
    emit_circuit_run_progress,
)

logger = logging.getLogger(__name__)


def _calibration_progress(db, circuit_id):
    """Progress callback that is ALSO the calibration's cancel checkpoint.

    Calibration has no loop this task can see — the bisection lives inside the
    service — but it reports progress at every bisection step, and a callback
    is the one hook already threaded through. Raising from it stops the search
    at the next step rather than at the next full sweep.
    """
    check = cancel_checker("circuit_calibration", circuit_id, db=db)

    def _cb(pct):
        check.raise_if_cancelled(f"stopped at {pct}% of the band search")
        emit_circuit_run_progress("calibration", circuit_id, pct)

    return _cb


def _refuse_if_cancelled(db, target_id) -> bool:
    """False when the row is already cancelled, so the caller should not start."""
    from ..core.cancellation import is_cancelled

    scope = "circuit_calibration"
    try:
        from ..core.cancellation import get_scope

        sc = get_scope(scope)
        model = sc.model()
        row = (
            db.query(model)
            .filter(getattr(model, sc.id_field) == target_id)
            .populate_existing()
            .first()
        )
    except Exception:  # noqa: BLE001 - a failed check must not block the work
        # LOUD, even though it fails open. Returning True silently is how a
        # guard becomes decorative: a rename of `get_scope` or a scope's model
        # would make this permanently inert with nothing in the log to say so.
        logger.exception(
            "Could not check whether %s was already cancelled; starting anyway",
            target_id,
        )
        return True
    if row is None:
        return True
    return not is_cancelled(scope, getattr(row, sc.status_field, None))


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_calibration_tasks.run_circuit_calibration",
                 max_retries=0)
def run_circuit_calibration(self, circuit_id: str,
                            config: Dict[str, Any]) -> Dict[str, Any]:
    """Find the usable band and clamp the served dial to it. WS on the
    "calibration" channel (run_id = circuit_id)."""
    from ..services.circuit_calibration_service import CircuitCalibrationService

    with self.get_db() as db:
        # A CANCEL-WHILE-QUEUED MUST NOT BE STAMPED OVER.
        #
        # `request_cancel` writes "cancelled" and issues a plain revoke(), but a
        # solo worker busy on another job is not reading the control queue — so
        # the revoke can land late or not at all, this task starts anyway, and
        # this write turns "cancelled" back into "running". Every subsequent
        # poll then reads running and the cancellation is simply gone, while
        # the endpoint has already told the operator "it will not run".
        if not _refuse_if_cancelled(db, circuit_id):
            return {"status": "cancelled", "id": circuit_id}
        _set_status(db, circuit_id, "running")
        try:
            result = CircuitCalibrationService.run(
                db, circuit_id, config,
                progress_cb=_calibration_progress(db, circuit_id))
            emit_circuit_run_completed("calibration", circuit_id, summary=result)
            return result
        except OperatorCancelled as cancelled:
            _set_status(db, circuit_id, "cancelled")
            emit_circuit_run_failed("calibration", circuit_id, "cancelled")
            return {"status": "cancelled", "circuit_id": circuit_id,
                    "detail": cancelled.detail}
        except Exception as e:
            logger.exception("Circuit calibration %s failed", circuit_id)
            # Clear the in-flight marker so the single-GPU guard isn't wedged.
            _set_status(db, circuit_id, "failed")
            emit_circuit_run_failed("calibration", circuit_id, str(e)[:500])
            raise


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_calibration_tasks.reproduce_circuit_calibration",
                 max_retries=0)
def reproduce_circuit_calibration(self, manifest_id: str,
                                  prior_status: Optional[str] = None
                                  ) -> Dict[str, Any]:
    """Re-run a calibration from a stored manifest and record a reproduction
    manifest with the band-delta verdict. Holds the GPU like a fresh run.

    Reproduction only CHECKS the number — it does not re-calibrate the circuit —
    so it must RESTORE the circuit's prior calibration_status rather than stamp
    'completed' over it (R3: a circuit whose last real run failed must not be
    relabeled completed by a reproduce)."""
    from ..services.circuit_calibration_service import CircuitCalibrationService

    with self.get_db() as db:
        circuit_id = _circuit_of_manifest(db, manifest_id)
        try:
            result = CircuitCalibrationService.reproduce(
                db, manifest_id,
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "calibration-reproduce", manifest_id, pct))
            # Restore the circuit's real status (the endpoint set it 'pending'
            # only to hold the GPU guard); reproduce changed nothing on it.
            if circuit_id:
                _set_status(db, circuit_id, prior_status)
            emit_circuit_run_completed("calibration-reproduce", manifest_id,
                                       summary=result)
            return result
        except Exception as e:
            logger.exception("Calibration reproduce %s failed", manifest_id)
            # The reproduce failed, not the circuit's calibration — restore its
            # prior status (still clears the in-flight 'pending' marker).
            if circuit_id:
                _set_status(db, circuit_id, prior_status)
            emit_circuit_run_failed("calibration-reproduce", manifest_id,
                                    str(e)[:500])
            raise


def _circuit_of_manifest(db, manifest_id):
    from ..models.validation_manifest import ValidationManifest
    try:
        db.rollback()
    except Exception:
        pass
    m = db.query(ValidationManifest).filter(
        ValidationManifest.id == manifest_id).first()
    return m.circuit_id if m is not None else None


def _set_status(db, circuit_id, status):
    from ..models.circuit import Circuit
    # If the run failed on a DB error, `db` is in an aborted transaction and any
    # query on it raises "current transaction is aborted". Roll back FIRST so the
    # status write lands — otherwise the in-flight marker is never cleared and the
    # single-GPU guard stays wedged until the 60-min cleanup.
    try:
        db.rollback()
    except Exception:
        logger.exception("Rollback before calibration_status write failed for %s",
                         circuit_id)
    try:
        row = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if row is not None:
            row.calibration_status = status
            db.commit()
    except Exception:
        logger.exception("Could not set calibration_status for %s", circuit_id)
