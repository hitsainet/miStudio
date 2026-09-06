"""
Celery tasks for circuit edge validation + faithfulness (Feature 017).

GPU profile → extraction queue (same as capture/attribution). Separate
lifecycle on the discovery run's validation_* columns (a failed pass never
corrupts the completed discovery). Cancellation via DB-status polling.
"""

import logging
from typing import Any, Dict

from ..core.cancellation import cancel_checker
from ..core.celery_app import celery_app
from .base_task import DatabaseTask
from .circuit_capture_tasks import _cancel_checker
from .websocket_emitter import (
    emit_circuit_run_completed,
    emit_circuit_run_failed,
    emit_circuit_run_progress,
)

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_validation_tasks.validate_circuit_edges",
                 max_retries=0)
def validate_circuit_edges(self, run_id: str, scope: Dict[str, Any]) -> Dict[str, Any]:
    from ..models.circuit_runs import CircuitDiscoveryRun
    from ..services.circuit_intervention_service import CircuitInterventionService

    with self.get_db() as db:
        try:
            result = CircuitInterventionService.run(
                db, run_id, scope,
                cancel_check=_cancel_checker(db, CircuitDiscoveryRun, run_id,
                                             status_field="validation_status"),
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "validation", run_id, pct))
            emit_circuit_run_completed("validation", run_id, summary=result)
            return result
        except Exception as e:
            logger.exception("Circuit validation %s failed", run_id)
            run = db.query(CircuitDiscoveryRun).filter(
                CircuitDiscoveryRun.id == run_id).first()
            if run is not None:
                run.validation_status = "failed"
                run.validation_error = str(e)[:2000]
                db.commit()
            emit_circuit_run_failed("validation", run_id, str(e)[:500])
            raise


@celery_app.task(bind=True, base=DatabaseTask,
                 name="src.workers.circuit_validation_tasks.run_circuit_faithfulness",
                 max_retries=0)
def run_circuit_faithfulness(self, circuit_id: str,
                             config: Dict[str, Any]) -> Dict[str, Any]:
    """Faithfulness (rung 3) runs on a CIRCUIT, not a discovery run — its own
    lifecycle: the result + a `faithfulness` manifest are the record (no
    discovery-run status columns), and circuit.faithfulness is written through
    the contract. WS on the "faithfulness" channel (run_id = circuit_id). The
    single-GPU guard is asserted at the endpoint (advisory lock) before dispatch.
    """
    from ..services.circuit_faithfulness_service import (
        CircuitFaithfulnessService, _FaithfulnessCancelled)

    with self.get_db() as db:
        try:
            result = CircuitFaithfulnessService.run(
                db, circuit_id, config,
                # WAS `cancel_check=None`. The entire cancellation path in
                # `circuit_faithfulness_service` — the poll, the raise, and the
                # `_FaithfulnessCancelled` handler five lines below — was
                # written, tested and DEAD, because the one caller passed None.
                # Nothing else had to change to bring it back.
                #
                # `db=db` deliberately: this service holds one long-lived task
                # session and the checker re-reads with populate_existing, so
                # it sees the API process's write rather than the row as it
                # looked when the run started.
                cancel_check=cancel_checker("circuit_faithfulness", circuit_id, db=db),
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "faithfulness", circuit_id, pct))
            emit_circuit_run_completed("faithfulness", circuit_id,
                                       summary=result)
            return result
        except _FaithfulnessCancelled:
            _set_faithfulness_status(db, circuit_id, "cancelled")
            emit_circuit_run_failed("faithfulness", circuit_id, "cancelled")
            return {"status": "cancelled", "circuit_id": circuit_id}
        except Exception as e:
            logger.exception("Circuit faithfulness %s failed", circuit_id)
            # Clear the in-flight marker so the GPU guard isn't wedged (R2 B-5).
            _set_faithfulness_status(db, circuit_id, "failed")
            emit_circuit_run_failed("faithfulness", circuit_id, str(e)[:500])
            raise


def _set_faithfulness_status(db, circuit_id, status):
    from ..models.circuit import Circuit
    # Roll back FIRST: if the run failed on a DB error, `db` is in an aborted
    # transaction and the status write below would itself raise, leaving the
    # in-flight marker set and wedging the single-GPU guard (Feature 20 review —
    # pre-existing, same shape as the calibration task).
    try:
        db.rollback()
    except Exception:
        logger.exception("Rollback before faithfulness_status write failed for %s",
                         circuit_id)
    try:
        row = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if row is not None:
            row.faithfulness_status = status
            db.commit()
    except Exception:
        logger.exception("Could not set faithfulness_status for %s", circuit_id)
