"""
Celery tasks for semantic labeling of SAE features.

These tasks run asynchronously to label features extracted from SAE models
without blocking the API. Labeling is independent from extraction, allowing
re-labeling without re-extraction.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from src.core.celery_app import celery_app
from src.core.cancellation import clear_cancel_request
from src.services.labeling_service import LabelingService
from src.services.labeling_sweep_service import LabelingSweepService
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

                # A CANCELLED JOB MUST NEVER RUN AGAIN, however the message
                # got back here.
                #
                # This guard handled COMPLETED and LABELING and nothing else, so
                # CANCELLED fell through every branch and the run started over
                # from feature 0. `acks_late` means a message stays unacked for
                # the whole run, and `task_reject_on_worker_lost` requeues it
                # whenever the worker goes away — a deploy, an OOM, a node
                # drain, or (until the sibling fix in `cancel_labeling_job`) the
                # cancel itself. Every one of those redelivered a cancelled job
                # to a fresh worker, which cheerfully resumed it.
                #
                # Observed: the 53k-feature L46 job was cancelled at 02:22 on
                # 2026-09-09, resurrected by the 10:57 deploy, and was still
                # running 13 hours later against the operator's explicit stop.
                #
                # The row is the authority, not the queue. Returning here also
                # ACKS the message, which is what finally takes it off the queue
                # for good.
                if labeling_job.status == LabelingStatus.CANCELLED.value:
                    logger.info(
                        f"Labeling {labeling_job.id} was cancelled at "
                        f"{labeling_job.completed_at}; refusing a redelivered "
                        f"message rather than restarting it"
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

            # A LEFTOVER CANCEL REQUEST MUST NOT OUTLIVE THE JOB THAT EARNED IT.
            #
            # `cancel_requested_at` is set by the endpoint and cleared by nobody
            # here, so a job that was cancelled and is then resumed reads its own
            # predecessor's request and abandons on the first guard tick — a
            # resume that reports "cancelled" without doing anything, forever.
            # The same omission once meant a cancelled model download could never
            # be downloaded again. Cleared at task START, before any work.
            clear_cancel_request("labeling", labeling_job_id)

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



@celery_app.task(
    bind=True,
    base=DatabaseTask,
    # FULLY QUALIFIED. `task_routes` globs match the TASK NAME, and a short name
    # silently lands on the default queue — which has cost this project a
    # debugging session before.
    name="labeling.resume_sweep_step",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError, OSError),
)
def resume_sweep_step(self, sweep_id: str) -> Dict[str, Any]:
    """Do ONE batch of a resume sweep, then re-enqueue for the next.

    NEVER A LOOP. 27 batches of 2000 at ~8 s/feature is ~59 GPU-hours against a
    10 h soft limit; a looping task would be killed part-way and, on an
    `acks_late` queue, strand its message for the full 12 h visibility timeout.
    One batch per task means each step gets a fresh limit and the row carries the
    progress.

    Idempotent at the boundary: the guard runs before any work, so a step
    delivered twice on a stopped or finished sweep does nothing.
    """
    with self.get_db() as db:
        service = LabelingSweepService(db)
        sweep = service.get(sweep_id)
        if sweep is None:
            # Consistent with the `labeling_sweep` cancel scope: a vanished row
            # is a stop signal, not an error.
            logger.info("Sweep %s no longer exists; stopping", sweep_id)
            return {"sweep_id": sweep_id, "status": "cancelled", "reason": "row deleted"}

        # A leftover request must not outlive the sweep that earned it — the
        # same trap that made a cancelled model download undownloadable.
        if sweep.batches_done == 0:
            clear_cancel_request("labeling_sweep", sweep_id)

        stop = service.should_stop(sweep)
        if stop is not None:
            status = "cancelled" if sweep.cancel_requested_at else "completed"
            service.finish(sweep, status, stop)
            return {"sweep_id": sweep_id, "status": status, "reason": stop}

        # R2: another batch job is already running on this extraction, so this
        # delivery is a duplicate (an `acks_late` redelivery, or a hand
        # re-trigger). Returning without enqueueing lets the live step carry the
        # sweep forward; enqueueing here would fork it into two chains.
        active = service.active_labeling_job(sweep.extraction_job_id)
        if active is not None:
            logger.info(
                "Sweep %s: labeling job %s is already running on %s; this step "
                "is a duplicate and is standing down",
                sweep_id, active.id, sweep.extraction_job_id,
            )
            return {
                "sweep_id": sweep_id,
                "status": "running",
                "reason": f"batch {active.id} already in flight",
            }

        feature_ids = service.next_batch_ids(sweep)
        if not feature_ids:
            # Nothing left is SUCCESS, not an error: the sweep did its job.
            service.finish(sweep, "completed", "no features left to label")
            return {"sweep_id": sweep_id, "status": "completed", "reason": "nothing left"}

        try:
            labeling_service = LabelingService(db)
            # Same builder the async start path uses, so a swept batch is
            # configured identically to a hand-started one. `start_labeling` is
            # async and cannot be called from here.
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            job = LabelingService.build_labeling_job_row(
                job_id=f"label_{sweep.extraction_job_id}_{timestamp}_{uuid.uuid4().hex[:6]}",
                extraction_job_id=sweep.extraction_job_id,
                config=sweep.config or {},
                total_features=len(feature_ids),
                panel_ids=feature_ids,
            )
            db.add(job)
            sweep.last_labeling_job_id = job.id
            db.commit()

            labeling_service.label_features_for_extraction(job.id)
            db.refresh(job)
            service.record_batch(sweep, job)
        except Exception as exc:  # noqa: BLE001 — recorded, then re-raised
            logger.error("Sweep %s batch failed: %s", sweep_id, exc, exc_info=True)
            service.finish(sweep, "failed", f"{type(exc).__name__}: {exc}")
            raise

        # RE-ENQUEUE, never recurse. The next batch is a new task with a new
        # soft limit.
        resume_sweep_step.delay(sweep_id)
        return {
            "sweep_id": sweep_id,
            "status": "running",
            "batches_done": sweep.batches_done,
            "features_labeled": sweep.features_labeled,
        }
