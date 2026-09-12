"""Periodic reaper for bulk labeling jobs that stopped without reporting.

`labeling_jobs` was the only long-running lifecycle in this system with no
janitor. Enhanced labeling has had one since MIS-E2E-092; bulk labeling did not,
and its absence is what turned the 409 lock into a trap: a job orphaned by a
worker restart sits QUEUED forever and 409s **every future labeling run on that
extraction**, naming a job id that only a manual DELETE can clear.

Threshold is 45 minutes, not the enhanced sibling's 10. Bulk labeling
legitimately runs for hours over tens of thousands of features; the 10-minute
figure is calibrated for a per-feature job that finishes in seconds.

Two additive guards, both from the shipped janitor pattern:

* `task_looks_alive` — the Celery task is genuinely running, so a quiet row is
  not evidence of death.
* `progress_stalled_seconds` — the work is ADVANCING. `None` means no evidence
  and must never be read as "stalled"; reaping on absence is how a healthy job
  gets killed.
"""

import logging
from datetime import datetime, timezone, timedelta

from src.core.celery_app import celery_app
from src.models.feature import Feature
from src.models.labeling_job import LabelingJob, LabelingStatus
from src.models.labeling_resume_sweep import LabelingResumeSweep
from src.models.labeling_trial_run import LabelingTrialRun
from src.workers.base_task import DatabaseTask
from src.workers.job_progress import progress_stalled_seconds
from src.workers.task_heartbeat import task_looks_alive

logger = logging.getLogger(__name__)

_STUCK_THRESHOLD_MINUTES = 45

#: A sweep is idle between batches only for as long as it takes to enqueue
#: the next step — milliseconds. But a BATCH runs up to ~4.4 h, and the row
#: is only touched when one finishes, so a healthy sweep can legitimately
#: look untouched for that long. Six hours leaves headroom above a full
#: batch without letting a genuinely wedged sweep sit for a day.
_SWEEP_STUCK_THRESHOLD_MINUTES = 360


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_labeling",
)
def cleanup_stuck_labeling_task(self):
    """Mark bulk labeling jobs stuck in QUEUED/LABELING as FAILED."""
    logger.info("Running stuck labeling cleanup task")

    with self.get_db() as db:
        try:
            threshold = datetime.now(timezone.utc) - timedelta(
                minutes=_STUCK_THRESHOLD_MINUTES)

            stuck = db.query(LabelingJob).filter(
                LabelingJob.status.in_([
                    LabelingStatus.QUEUED.value,
                    LabelingStatus.LABELING.value,
                ]),
                LabelingJob.updated_at < threshold,
            ).all()

            cleaned = 0
            for job in stuck:
                if job.celery_task_id and task_looks_alive(
                    job.celery_task_id, job,
                    started=str(getattr(job, "status", "")).lower() == "labeling",
                ):
                    logger.info(
                        "Labeling job %s has an active Celery task %s, skipping",
                        job.id, job.celery_task_id)
                    continue

                # IS THE WORK ADVANCING? A labeling run commits features_labeled
                # every batch, so a moving counter means the row's age is a lie.
                stalled = progress_stalled_seconds(
                    "labeling_job", job.id, getattr(job, "features_labeled", None))
                if stalled is not None and stalled < _STUCK_THRESHOLD_MINUTES * 60:
                    logger.info(
                        "%s advanced %.0fs ago; sparing it despite a stale row",
                        job.id, stalled)
                    continue

                # Capture BEFORE mutating: the message interpolates the status,
                # and assigning FAILED first makes every one of them read
                # "stuck in FAILED", discarding the only field that says what it
                # was actually stuck in.
                stuck_in = job.status
                stuck_minutes = int(
                    (datetime.now(timezone.utc) - job.updated_at).total_seconds() / 60)

                logger.warning(
                    "Marking stuck labeling job %s as FAILED (was %s, stuck %d min, "
                    "task_id: %s)", job.id, stuck_in, stuck_minutes,
                    job.celery_task_id or "None")

                job.status = LabelingStatus.FAILED.value
                job.error_message = (
                    f"Labeling job stuck in {stuck_in} for {stuck_minutes} minutes "
                    f"with no progress - the worker was restarted or the task was lost"
                )
                job.updated_at = datetime.now(timezone.utc)

                # RELEASE THE FEATURES THIS JOB CLAIMED.
                #
                # R4 HARDWARE FINDING, and it contradicts a claim made in
                # `_claim_features`' own docstring: "stale in_progress rows are
                # reclaimed by the existing cleanup_stuck_labeling sweeper".
                # They were not. This janitor marked the JOB failed and never
                # touched `features.label_status`, so a feature claimed by a
                # worker that died stayed `in_progress` FOREVER — excluded from
                # eligibility by design, and therefore invisible to every future
                # resume. Observed live: a pod restart mid-panel stranded 5 of 15.
                #
                # `failed`, not `pending`: an attempt genuinely happened and
                # `label_attempts` already counts it. Recording it as never
                # attempted would hand those features an extra free retry and
                # lose the only evidence that a worker died holding them.
                released = (
                    db.query(Feature)
                    .filter(
                        Feature.labeling_job_id == job.id,
                        Feature.label_status == "in_progress",
                    )
                    .update(
                        {
                            "label_status": "failed",
                            "label_error": (
                                "the worker was restarted while this feature was "
                                "being labeled; no verdict was produced"
                            ),
                            "label_error_at": datetime.now(timezone.utc),
                        },
                        synchronize_session=False,
                    )
                )
                if released:
                    logger.warning(
                        "Released %d features claimed by stuck job %s", released, job.id
                    )

                # A trial's RESULT row must be failed alongside its job, or the
                # panel stays locked against future trials by the in-flight check.
                if job.trial_run_id:
                    run = db.query(LabelingTrialRun).filter(
                        LabelingTrialRun.id == job.trial_run_id).first()
                    if run and run.status in ("queued", "running"):
                        run.status = "failed"
                        run.error = f"labeling job {job.id} was reaped as stuck"
                cleaned += 1

            if cleaned:
                db.commit()
            logger.info("Stuck labeling cleanup: %d job(s) marked FAILED", cleaned)
            return {"cleaned": cleaned, "scanned": len(stuck)}
        except Exception as exc:
            db.rollback()
            logger.error("Stuck labeling cleanup failed: %s", exc, exc_info=True)
            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_labeling_sweeps",
)
def cleanup_stuck_labeling_sweeps_task(self):
    """Reap resume sweeps that stopped advancing without reporting.

    R2 FINDING. A sweep step records its batch and THEN re-enqueues. A worker
    killed in that window leaves `status='running'` with no task queued — the
    sweep is finished as far as the system is concerned and finished for nobody
    else, because `create` refuses a second running sweep on the extraction. So
    the wedge is not merely untidy: it locks the extraction out of sweeping
    until someone edits the row by hand.

    The same shape as `cleanup_stuck_labeling`, and for the same reason: this is
    the only other long-running lifecycle here, and it had no janitor.

    A sweep whose batch is still running is NOT stuck. That is what the
    in-flight labeling-job check below is for — reaping on row age alone would
    kill a healthy sweep four hours into a batch.
    """
    logger.info("Running stuck labeling-sweep cleanup task")

    with self.get_db() as db:
        try:
            threshold = datetime.now(timezone.utc) - timedelta(
                minutes=_SWEEP_STUCK_THRESHOLD_MINUTES
            )
            stuck = (
                db.query(LabelingResumeSweep)
                .filter(
                    LabelingResumeSweep.status == "running",
                    LabelingResumeSweep.updated_at < threshold,
                )
                .all()
            )

            cleaned = 0
            for sweep in stuck:
                # A batch still in flight means the sweep is alive, however
                # quiet the row is. Absence of evidence is not evidence of death
                # — reaping on it is how a healthy job gets killed.
                active = (
                    db.query(LabelingJob)
                    .filter(
                        LabelingJob.extraction_job_id == sweep.extraction_job_id,
                        LabelingJob.status.in_(
                            [LabelingStatus.QUEUED.value, LabelingStatus.LABELING.value]
                        ),
                    )
                    .first()
                )
                if active is not None:
                    logger.info(
                        "Sweep %s still has batch %s in flight; sparing it",
                        sweep.id, active.id,
                    )
                    continue

                stuck_minutes = int(
                    (datetime.now(timezone.utc) - sweep.updated_at).total_seconds() / 60
                )
                logger.warning(
                    "Marking stuck sweep %s as FAILED (stuck %d min after %d batches)",
                    sweep.id, stuck_minutes, sweep.batches_done,
                )
                sweep.status = "failed"
                sweep.error_message = (
                    f"Sweep stopped advancing for {stuck_minutes} minutes after "
                    f"{sweep.batches_done} batches — the worker was restarted "
                    "between finishing a batch and queueing the next. The labels "
                    "it already wrote are unaffected; start a new sweep to continue."
                )
                sweep.completed_at = datetime.now(timezone.utc)
                sweep.updated_at = sweep.completed_at
                cleaned += 1

            if cleaned:
                db.commit()
                logger.warning("Reaped %d stuck labeling sweep(s)", cleaned)
            return {"cleaned": cleaned, "checked": len(stuck)}
        except Exception as exc:  # noqa: BLE001
            logger.error("Sweep cleanup failed: %s", exc, exc_info=True)
            db.rollback()
            raise
