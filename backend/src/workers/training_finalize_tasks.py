"""
Celery task: finalize a training from a saved checkpoint.

Runs on the CPU queue — rebuilding SAE modules and writing the Community
Standard export never touches the GPU, so a finalize can proceed while another
training owns the device.
"""

import logging
from datetime import datetime, timezone

from src.core.celery_app import celery_app
from src.models.training import Training, TrainingStatus
from src.services.training_finalize_service import (
    FinalizeError,
    finalize_from_checkpoint,
)
from src.workers.base_task import DatabaseTask

logger = logging.getLogger(__name__)


def _emit_finalize_failed(training_id: str, message: str) -> None:
    """Tell the UI a finalize failed.

    Without this the endpoint's 202 is the last thing the user ever hears: the
    button clears, the card never changes, and the only record is the worker log
    — the "goes quietly dark" failure mode this feature exists to remove.
    """
    try:
        from src.workers.websocket_emitter import emit_training_progress

        emit_training_progress(
            training_id=training_id,
            event="training:finalize_failed",
            data={"training_id": training_id, "error": message},
            retries=2,  # terminal event
        )
    except Exception:  # noqa: BLE001 - notification is best-effort
        logger.warning("Could not emit finalize failure for %s", training_id)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    # Fully-qualified: Celery routes on the TASK NAME, so a short name would
    # never match the "src.workers.training_finalize_tasks.*" route glob and
    # the task would silently fall back to the default queue.
    name="src.workers.training_finalize_tasks.finalize_training_from_checkpoint",
)
def finalize_training_from_checkpoint_task(
    self,
    training_id: str,
    checkpoint_step: int = None,
):
    """Write community_format for a stopped training, then mark it COMPLETED.

    Marking COMPLETED is what unlocks the downstream SAE import path. The run's
    progress/current_step are deliberately left untouched and
    ``finalized_from_step`` records where it actually stopped, so nothing
    pretends a partial run went the distance.
    """
    logger.info(
        "Finalizing training %s from checkpoint_step=%s", training_id, checkpoint_step
    )

    with self.get_db() as db:
        try:
            result = finalize_from_checkpoint(
                db, training_id, checkpoint_step=checkpoint_step
            )

            training = db.query(Training).filter_by(id=training_id).first()
            if training is None:
                # finalize_from_checkpoint already proved it exists; a delete
                # racing us is the only way here.
                raise FinalizeError(f"Training disappeared during finalize: {training_id}")

            # GUARDED WRITE: stop_and_finalize revokes the trainer, whose own
            # except-handler races us to write FAILED. Whoever commits last used
            # to win, so a valid export could end up on a run showing FAILED
            # (import locked), or a crash record could be silently erased.
            # Only promote a run that is still in the terminal state we expect.
            if training.status not in (
                TrainingStatus.CANCELLED.value,
                TrainingStatus.FAILED.value,
            ):
                logger.warning(
                    "Not promoting training %s: status is %s, not a terminal "
                    "state this task may overwrite. Export was still written.",
                    training_id, training.status,
                )
                db.rollback()
                return result

            completed_at = datetime.now(timezone.utc)
            # Finalizing over a FAILED run must not leave its crash record
            # attached: the card would render an error box beside a green
            # Completed badge, and anything reading error_message as a failure
            # signal sees a contradiction. Preserve it in hyperparameters so the
            # crash is not simply erased.
            if training.error_message:
                hp = dict(training.hyperparameters or {})
                hp["finalized_over_error"] = training.error_message
                training.hyperparameters = hp
                training.error_message = None
                training.error_traceback = None

            training.status = TrainingStatus.COMPLETED.value
            training.completed_at = completed_at
            training.finalized_from_step = result["checkpoint_step"]
            # progress / current_step intentionally NOT modified — a salvaged run
            # must not claim it went the distance. Captured here for the emit
            # below, which happens after the session closes.
            final_progress = training.progress
            final_current_step = training.current_step
            db.commit()

            logger.info(
                "Finalized training %s from step %s (%d SAEs)",
                training_id, result["checkpoint_step"], result["sae_count"],
            )

        except FinalizeError as e:
            db.rollback()
            logger.error("Cannot finalize training %s: %s", training_id, e)
            _emit_finalize_failed(training_id, str(e))
            raise
        except Exception as e:
            db.rollback()
            logger.error(
                "Error finalizing training %s: %s", training_id, e, exc_info=True
            )
            _emit_finalize_failed(training_id, str(e))
            raise

    # Emitted outside the DB block so a websocket failure can never roll back a
    # successful finalize. Mirrors the success path's terminal event (same name
    # and retries) so the frontend's existing handler clears "running".
    try:
        from src.workers.websocket_emitter import emit_training_progress

        emit_training_progress(
            training_id=training_id,
            event="training:completed",
            data={
                "training_id": training_id,
                "status": TrainingStatus.COMPLETED.value,
                "finalized_from_step": result["checkpoint_step"],
                # Send the REAL progress so the UI does not paint a full bar on a
                # run that stopped early. The frontend falls back to 100 only
                # when progress is absent (normal completion).
                "progress": final_progress,
                "current_step": final_current_step,
                "completed_at": completed_at.isoformat(),
            },
            retries=2,  # Terminal event: frontend stays "running" if this is lost
        )
    except Exception as emit_exc:  # noqa: BLE001 - notification is best-effort
        logger.warning(
            "Finalize succeeded but progress emit failed for %s: %s",
            training_id, emit_exc,
        )

    return result
