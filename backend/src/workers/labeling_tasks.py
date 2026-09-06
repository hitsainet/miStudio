"""
Celery tasks for semantic labeling of SAE features.

These tasks run asynchronously to label features extracted from SAE models
without blocking the API. Labeling is independent from extraction, allowing
re-labeling without re-extraction.
"""

import logging
from typing import Dict, Any

from src.core.celery_app import celery_app
from src.services.labeling_service import LabelingService
from src.workers.base_task import DatabaseTask

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="label_features",
    max_retries=3,
    default_retry_delay=60,  # 1-minute back-off between retries
    autoretry_for=(ConnectionError, TimeoutError, OSError),
)
def label_features_task(
    self,
    labeling_job_id: str
) -> Dict[str, Any]:
    """
    Celery task for labeling features from an extraction job.

    This task:
    1. Fetches the labeling job record
    2. Updates labeling status to 'labeling'
    3. Calls LabelingService.label_features_for_extraction() for core logic
    4. Handles errors and updates status accordingly

    Args:
        labeling_job_id: ID of the labeling job to execute

    Returns:
        Dict with labeling statistics
    """
    logger.info(f"Starting labeling task for job {labeling_job_id}")

    with self.get_db() as db:
        try:
            # Pre-flight check: Verify labeling hasn't already completed
            from src.models.labeling_job import LabelingJob, LabelingStatus
            from datetime import datetime, timezone, timedelta

            labeling_job = db.query(LabelingJob).filter(
                LabelingJob.id == labeling_job_id
            ).first()

            if labeling_job:
                if labeling_job.status == LabelingStatus.COMPLETED.value:
                    logger.info(
                        f"Labeling {labeling_job.id} already completed at "
                        f"{labeling_job.completed_at}, skipping re-execution"
                    )
                    return labeling_job.statistics or {}

                if labeling_job.status == LabelingStatus.LABELING.value:
                    # Check if it's been running for too long (> 2 hours = likely stuck)
                    if labeling_job.updated_at:
                        time_since_update = datetime.now(timezone.utc) - labeling_job.updated_at
                        if time_since_update > timedelta(hours=2):
                            logger.warning(
                                f"Labeling {labeling_job.id} appears stuck "
                                f"(no update for {time_since_update}), allowing restart"
                            )
                        else:
                            logger.info(
                                f"Labeling {labeling_job.id} is already in progress "
                                f"(last update: {time_since_update} ago), skipping"
                            )
                            return {}

            labeling_service = LabelingService(db)

            # Core labeling logic is delegated to service
            statistics = labeling_service.label_features_for_extraction(labeling_job_id)

            logger.info(f"Labeling completed for job {labeling_job_id}")
            logger.info(f"Statistics: {statistics}")

            return statistics

        except LabelingService._LabelingCancelled:
            # A clean, user-initiated stop — not a failure. The job row is
            # already CANCELLED (that is what the loop noticed), so return
            # quietly and free the worker. Re-raising would mark the run failed
            # and log a spurious traceback for something the user asked for.
            #
            # THIS COMMENT WAS FALSE UNTIL MIS-E2E-058 was fixed. The service's
            # outer `except Exception` caught `_LabelingCancelled` first and set
            # status=FAILED before re-raising, so the row reaching here was
            # FAILED, not CANCELLED — and the comment asserting otherwise is
            # exactly why nobody looked. The service now handles the
            # cancellation explicitly, ahead of the generic handler.
            logger.info(
                f"Labeling job {labeling_job_id} stopped early: cancelled by user"
            )
            return {"cancelled": True, "labeling_job_id": labeling_job_id}

        except Exception as e:
            logger.error(
                f"Labeling task failed for job {labeling_job_id}: {e}",
                exc_info=True
            )
            # Service already handles status update on error
            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    # FULLY QUALIFIED on purpose. task_routes globs match the TASK NAME, not the
    # module path, so a short name like "label_features_trial" would match no
    # glob and land silently on the default `datasets` queue instead of
    # `processing`. This project has been bitten by that twice.
    name="src.workers.labeling_tasks.label_features_trial",
    max_retries=0,
)
def label_features_trial_task(self, labeling_job_id: str):
    """Run one prompt-template trial. Writes no Feature row.

    max_retries=0 deliberately: a trial is a measurement. Retrying it would
    silently spend the budget again and could interleave two runs over the same
    panel, which is the one thing panel identity exists to prevent.
    """
    from src.services.labeling_trial_service import (
        LabelingTrialService, TrialWroteToFeatures,
    )
    from src.models.labeling_trial_run import LabelingTrialRun
    from src.models.labeling_job import LabelingJob, LabelingStatus

    # `with`, not `db = self.get_db()`. get_db is a CONTEXT MANAGER, so the bare
    # call handed the service a _GeneratorContextManager instead of a Session and
    # every trial died before it started. It failed silently in two layers: the
    # type guard inside run_trial fired correctly, then this function's own error
    # handler called .query() on that object and `finally: db.close()` raised
    # AttributeError — masking the real error behind a cleanup failure.
    #
    # Every other task in this file and in circuit_record/circuit_capture/
    # cleanup_task_queue already uses the `with` form, including
    # label_features_task 90 lines above. This one was the outlier.
    with self.get_db() as db:
        try:
            service = LabelingTrialService(db)
            result = service.run_trial(labeling_job_id)
            logger.info("Trial %s complete: %s", labeling_job_id, result.get("stats"))
            return result
        except TrialWroteToFeatures:
            # Never swallow this one. It means the measurement path mutated the data
            # it was measuring, and every label in the extraction is now suspect.
            logger.critical(
                "TRIAL WROTE TO FEATURES for job %s — labels may be corrupted",
                labeling_job_id, exc_info=True,
            )
            raise
        except Exception as exc:
            logger.error("Trial %s failed: %s", labeling_job_id, exc, exc_info=True)
            try:
                job = db.query(LabelingJob).filter(
                    LabelingJob.id == labeling_job_id).first()
                if job:
                    job.status = LabelingStatus.FAILED.value
                    job.error_message = str(exc)[:500]
                    run = db.query(LabelingTrialRun).filter(
                        LabelingTrialRun.id == job.trial_run_id).first()
                    if run:
                        run.status = "failed"
                        run.error = str(exc)[:500]
                    db.commit()
            except Exception:
                logger.exception("could not record trial failure for %s", labeling_job_id)
            raise

