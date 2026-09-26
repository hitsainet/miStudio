"""Celery task for steered-transcript recording (Steered Transcript Recorder).

Records (dial, prompt, unsteered, steered) transcripts for a circuit / cluster /
feature set on the GPU and persists a `steering_samples` manifest. Holds the
single GPU like calibration; its in-flight marker is a `steering_record_runs` row
(cluster/feature jobs have no circuit row). GPU profile → extraction queue.
"""

import logging
from typing import Any, Dict

from ..core.cancellation import (
    OperatorCancelled,
    cancel_checker,
    is_cancelled,
)
from ..core.celery_app import celery_app
from .base_task import DatabaseTask
from .gpu_job import gpu_job, record_request
from .circuit_gpu import (
    circuit_required_mb,
    place_circuit_job,
    release_circuit_job,
    steering_sizing_manifest,
)
from .websocket_emitter import (
    emit_circuit_run_completed,
    emit_circuit_run_failed,
    emit_circuit_run_progress,
)

logger = logging.getLogger(__name__)


def _refuse_if_cancelled(db, target_id) -> bool:
    """False when the row is already cancelled, so the caller should not start."""
    from ..core.cancellation import is_cancelled

    scope = "steering_record"
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
                 name="src.workers.circuit_record_tasks.run_circuit_record",
                 max_retries=0)
@gpu_job("steering_record", request_from=record_request)
def run_circuit_record(self, record_run_id: str,
                       config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate + record steered transcripts. WS on the "steering-record"
    channel (run_id = record_run_id)."""
    from ..services.steering_recorder_service import SteeringRecorderService

    with self.get_db() as db:
        # A CANCEL-WHILE-QUEUED MUST NOT BE STAMPED OVER.
        #
        # `request_cancel` writes "cancelled" and issues a plain revoke(), but a
        # solo worker busy on another job is not reading the control queue — so
        # the revoke can land late or not at all, this task starts anyway, and
        # this write turns "cancelled" back into "running". Every subsequent
        # poll then reads running and the cancellation is simply gone, while
        # the endpoint has already told the operator "it will not run".
        if not _refuse_if_cancelled(db, record_run_id):
            return {"status": "cancelled", "id": record_run_id}
        _set_status(db, record_run_id, "running")
        placement = None
        try:
            placement, gpu = _place_and_record(db, record_run_id, config)
            # A split hands the recorder its placement (the loader needs the
            # budget, the resolvers each layer's card). One card keeps the exact
            # call it always made.
            split = {"placement": placement} if placement.is_shard else {}
            result = SteeringRecorderService.record_samples(
                db, config, device=placement.device, gpu=gpu,
                progress_cb=lambda pct: emit_circuit_run_progress(
                    "steering-record", record_run_id, pct),
                cancel_check=cancel_checker(
                    "steering_record", record_run_id, db=db),
                run_id=record_run_id, **split)
            # `_complete` returns without setting "completed" when the run was
            # cancelled during its tail; emitting a completion anyway shows the
            # operator a finished job over a cancelled row.
            # `_complete` ALREADY performed this read, inside its own
            # try/except. Repeating it here put an unprotected query inside the
            # outer try whose handler writes FAILED — so a recycled connection
            # on the duplicate read would relabel a run `_complete` had just
            # committed as completed. It returns the answer now.
            if not _complete(db, record_run_id, result.get("manifest_ref")):
                return {"status": "cancelled", "id": record_run_id}
            emit_circuit_run_completed("steering-record", record_run_id,
                                       summary=result)
            return result
        except OperatorCancelled as cancelled:
            from ..core.cancellation import GPU_LEASE_LOST

            # The row is already CANCELLED — the endpoint set it, which is how
            # this task found out. Returning (not raising) is what acks the
            # acks_late message. A lost GPU lease (multi-GPU Phase 3) is the
            # exception: nothing set the row, and it is the job's failure.
            outcome = "failed" if cancelled.reason == GPU_LEASE_LOST else "cancelled"
            _set_status(db, record_run_id, outcome, error=cancelled.detail[:500])
            emit_circuit_run_failed(
                "steering-record", record_run_id,
                f"stopped: {cancelled.detail}"[:500] if outcome == "failed" else "cancelled",
            )
            return {"status": outcome, "id": record_run_id,
                    "reason": cancelled.reason, "detail": cancelled.detail}
        except Exception as e:
            logger.exception("Steering record %s failed", record_run_id)
            _set_status(db, record_run_id, "failed", error=str(e)[:500])
            emit_circuit_run_failed("steering-record", record_run_id, str(e)[:500])
            raise
        finally:
            # Every card of the job, once the model is unreachable; see
            # `circuit_gpu.release_circuit_job`.
            release_circuit_job(placement, kind="steering-record", run_id=record_run_id)


def _place_and_record(db, record_run_id, config):
    """Place the recording with the request stored on its row, and record the
    card on the row BEFORE the model load (so a load that dies names its card).

    A row from before GPUs could be chosen has a NULL request, which is auto.
    The recording may split, sized from the artifact's model and SAEs.
    """
    from ..models.steering_record_run import SteeringRecordRun
    from ..services.steering_recorder_service import SteeringRecorderService

    row = db.query(SteeringRecordRun).filter(
        SteeringRecordRun.id == record_run_id).first()
    if row is None:
        raise ValueError(f"Record run {record_run_id} not found")
    refs = SteeringRecorderService.sizing_refs(config, db)
    placement, gpu = place_circuit_job(
        row.gpu_request, kind="steering-record", run_id=record_run_id, allow_shard=True,
        required_mb=circuit_required_mb(db, steering_sizing_manifest(*refs) if refs else None))
    columns = placement.gpu_columns()
    row.gpu_uuid = columns["gpu_uuid"]
    # Every card of a split, most free first; None for one card. `gpu_uuid` alone
    # named only the first card of a model that ran on two.
    row.gpu_uuids = columns["gpu_uuids"]
    if placement.is_shard:
        # The manifest is read without the row (cluster/feature manifests have
        # no circuit to join), so it carries the split's cards too.
        gpu = {**gpu, "uuids": placement.uuids}
    db.commit()
    return placement, gpu


def _set_status(db, record_run_id, status, error=None):
    from ..models.steering_record_run import SteeringRecordRun
    # Roll back first: a DB-error failure leaves the session aborted, and the
    # status write below would itself raise, leaving the marker set and wedging
    # the single-GPU guard (Feature 20 R2 lesson).
    try:
        db.rollback()
    except Exception:
        logger.exception("Rollback before record status write failed for %s",
                         record_run_id)
    try:
        row = db.query(SteeringRecordRun).filter(
            SteeringRecordRun.id == record_run_id).first()
        if row is not None:
            row.status = status
            if error is not None:
                row.error = error
            db.commit()
    except Exception:
        logger.exception("Could not set record status for %s", record_run_id)


def _complete(db, record_run_id, manifest_ref) -> bool:
    """Write the completion. Returns False when the run was cancelled in its
    tail, so the caller can skip announcing a completion the row refused."""
    from ..models.steering_record_run import SteeringRecordRun
    # record_samples committed the manifest, so the session is clean here — but
    # roll back defensively so a lingering aborted state can't block the status
    # write (a completed job whose marker stays 'running' would wedge the GPU
    # guard until cleanup; R1).
    try:
        db.rollback()
    except Exception:
        logger.exception("Rollback before record completion failed for %s",
                         record_run_id)
    try:
        # populate_existing: this runs on the long-lived task session, which
        # holds the row as it looked when the recording started — so without it
        # the check below reads a stale status and can never see a cancel.
        row = db.query(SteeringRecordRun).filter(
            SteeringRecordRun.id == record_run_id).populate_existing().first()
        completed = False
        if row is not None and is_cancelled("steering_record", row.status):
            # The last checkpoint is the top of the prompt loop, so a cancel
            # arriving during the final prompt's generations, or during manifest
            # persistence, lands on this write — which was unguarded, and would
            # have relabelled the operator's stop as a success.
            logger.info(
                "Record run %s finished after it was cancelled; keeping the "
                "cancelled status", record_run_id,
            )
            row.manifest_ref = manifest_ref
            db.commit()
            return False
        if row is not None:
            row.status = "completed"
            row.manifest_ref = manifest_ref
            db.commit()
            completed = True
    except Exception:
        logger.exception("Could not complete record run %s", record_run_id)
    return completed
