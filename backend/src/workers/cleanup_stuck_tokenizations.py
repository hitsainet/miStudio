"""
Periodic task to reclaim tokenization jobs whose worker died.

Tokenization was the one long-running status with no janitor. Every other such
status has one (extractions, trainings, activations, NLP, circuit runs, enhanced
labeling), so a tokenization whose worker vanished held PROCESSING forever: the
dataset card showed a progress bar frozen at whatever percentage was last
written, with no error, no retry affordance, and nothing that would ever change
it. Worse, the parent dataset stayed PROCESSING too, so the whole dataset looked
busy indefinitely.

A worker can vanish for reasons the task cannot catch -- a rolling deploy
SIGTERMing a single-slot worker mid-job, an OOM kill, a pod eviction, a node
reboot. None of those give the task a chance to write a terminal status, so the
sweep has to come from outside.

Runs every 10 minutes.
"""

import logging
from datetime import datetime, timedelta, timezone

from src.core.celery_app import celery_app
from src.models.dataset import Dataset, DatasetStatus
from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from src.workers.base_task import DatabaseTask
from src.workers.websocket_emitter import emit_tokenization_status
from src.workers.job_progress import progress_stalled_seconds

logger = logging.getLogger(__name__)

# A tokenization writes progress continuously while it runs, so silence this
# long means the process is gone rather than slow.
STUCK_THRESHOLD_MINUTES = 60


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_tokenizations",
)
def cleanup_stuck_tokenizations_task(self):
    """Mark tokenizations as ERROR when their worker is gone, and free the dataset."""
    logger.info("Running stuck tokenization cleanup task")

    with self.get_db() as db:
        try:
            threshold = datetime.now(timezone.utc) - timedelta(
                minutes=STUCK_THRESHOLD_MINUTES
            )

            candidates = (
                db.query(DatasetTokenization)
                .filter(
                    DatasetTokenization.status.in_(
                        [
                            TokenizationStatus.QUEUED,
                            TokenizationStatus.PROCESSING,
                        ]
                    ),
                    DatasetTokenization.updated_at < threshold,
                )
                .all()
            )

            cleaned_count = 0
            for tok in candidates:
                age_minutes = (
                    datetime.now(timezone.utc) - tok.updated_at
                ).total_seconds() / 60

                if tok.celery_task_id:
                    # MIS-E2E-092: a dead worker and a queued task are both
                    # PENDING, so a bare state check can never be false here.
                    from src.workers.task_heartbeat import task_looks_alive

                    if task_looks_alive(
                        tok.celery_task_id,
                        tok,
                        started=str(getattr(tok, "status", "")).lower()
                        in ("processing", "tokenizationstatus.processing"),
                    ):
                        logger.info(
                            f"Tokenization {tok.id} has an active Celery task "
                            f"{tok.celery_task_id}, skipping cleanup"
                        )
                        continue

                # IS THE WORK ADVANCING? Additive to the clock — reap only
                # when stale AND the counter has not moved. None means "no
                # evidence" and must never read as "stalled".
                _stalled = progress_stalled_seconds("tokenization", tok.id, getattr(tok, "progress", None))
                if _stalled is not None and _stalled < STUCK_THRESHOLD_MINUTES * 60:
                    logger.info(
                        "%s advanced %.0fs ago; sparing it despite a stale row",
                        tok.id, _stalled,
                    )
                    continue

                logger.warning(
                    f"Marking stuck tokenization {tok.id} as ERROR "
                    f"(status: {tok.status}, age: {age_minutes:.0f}min, "
                    f"task_id: {tok.celery_task_id or 'None'})"
                )

                tok.status = TokenizationStatus.ERROR
                tok.error_message = (
                    f"Tokenization stuck - no progress for more than "
                    f"{int(age_minutes)} minutes. The worker was lost before it "
                    "could record an outcome; no output was written. Re-run it."
                )
                tok.completed_at = datetime.now(timezone.utc)
                tok.updated_at = datetime.now(timezone.utc)

                # Release the parent dataset. Another tokenization of the same
                # dataset may still be live, so only clear it when none is.
                dataset = (
                    db.query(Dataset).filter(Dataset.id == tok.dataset_id).first()
                )
                if dataset is not None and dataset.status == DatasetStatus.PROCESSING:
                    still_busy = (
                        db.query(DatasetTokenization)
                        .filter(
                            DatasetTokenization.dataset_id == tok.dataset_id,
                            DatasetTokenization.id != tok.id,
                            DatasetTokenization.status.in_(
                                [
                                    TokenizationStatus.QUEUED,
                                    TokenizationStatus.PROCESSING,
                                ]
                            ),
                        )
                        .count()
                    )
                    if still_busy == 0:
                        logger.warning(
                            f"Releasing dataset {dataset.id} from PROCESSING - "
                            f"its last tokenization ({tok.id}) is gone"
                        )
                        dataset.status = DatasetStatus.READY

                db.commit()
                cleaned_count += 1

                try:
                    emit_tokenization_status(
                        dataset_id=str(tok.dataset_id),
                        tokenization_id=tok.id,
                        status="error",
                        error_message=tok.error_message,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to emit WebSocket event for {tok.id}: {e}"
                    )

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stuck tokenization(s)")
            else:
                logger.info("No stuck tokenizations found")

            return {"cleaned": cleaned_count}

        except Exception as e:
            logger.error(
                f"Error in stuck tokenization cleanup: {e}", exc_info=True
            )
            raise
