"""
Periodic task to clean up stuck extraction jobs.

This task runs every 10 minutes and marks extraction jobs as FAILED if they've been
stuck for too long without updates.

A job is failed only when the clock says its row is stale AND the work is not
advancing. Both halves matter:

- a batch member waiting its turn is judged by whether its BATCH is advancing,
  not by its own age. Only the first member of a batch is dispatched, so a later
  member legitimately sits queued for hours with no task and no progress.
- a running job is judged by whether its progress counter moved, because the
  row's `updated_at` is rewritten every tick whether or not anything happened.

Thresholds:
- QUEUED batch members: 3 hours of BATCH silence (not row age)
- QUEUED jobs with a Celery task: 1 hour
- EXTRACTING jobs: 1 hour, and only if progress has not advanced
"""

import logging
from datetime import datetime, timezone, timedelta
from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.workers.websocket_emitter import emit_extraction_job_progress
from src.workers.job_progress import (
    batch_has_live_sibling,
    batch_last_activity,
    progress_stalled_seconds,
)

logger = logging.getLogger(__name__)

# Jobs waiting in a batch queue get a long grace period
BATCH_QUEUED_THRESHOLD_MINUTES = 180  # 3 hours
# Jobs with a Celery task that haven't started
QUEUED_THRESHOLD_MINUTES = 60  # 1 hour
# Jobs actively extracting that appear stuck
EXTRACTING_THRESHOLD_MINUTES = 60  # 1 hour


def _try_restart_batch(db, extraction) -> bool:
    """Restart a stalled batch chain once. Returns True if it was restarted.

    A queued member with no live sibling and a silent batch means the CHAIN
    failed, not the work: `_start_next_batch_job` never ran, or ran and lost
    its dispatch. Failing the member throws away a perfectly good queued job.

    Restarting is attempted once per sweep and the next sweep re-checks; if it
    does not take, the member is condemned then, with a message that says the
    chain stopped rather than blaming a worker.
    """
    try:
        from src.workers.nlp_analysis_tasks import _start_next_batch_job

        _start_next_batch_job(db, extraction)
        logger.warning(
            "Extraction %s: batch %s had stalled with queued work; restarted "
            "the chain rather than failing it",
            extraction.id, extraction.batch_id,
        )
        return True
    except Exception as exc:
        logger.warning(
            "Extraction %s: could not restart batch %s (%s); it will be failed",
            extraction.id, extraction.batch_id, exc,
        )
        return False


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_extractions"
)
def cleanup_stuck_extractions_task(self):
    """
    Clean up extraction jobs that have been stuck for too long.

    An extraction is considered stuck if:
    - QUEUED with no celery_task_id (batch waiting): no update in 3 hours
    - QUEUED with celery_task_id: no update in 1 hour AND task is not running
    - EXTRACTING: no update in 1 hour AND task is not running
    """
    logger.info("Running stuck extraction cleanup task")

    with self.get_db() as db:
        try:
            # Use the shortest threshold to find candidates, then apply per-status logic
            candidate_threshold = datetime.now(timezone.utc) - timedelta(minutes=QUEUED_THRESHOLD_MINUTES)

            candidate_extractions = db.query(ExtractionJob).filter(
                ExtractionJob.status.in_([
                    ExtractionStatus.QUEUED.value,
                    ExtractionStatus.EXTRACTING.value
                ]),
                ExtractionJob.updated_at < candidate_threshold
            ).all()

            cleaned_count = 0
            for extraction in candidate_extractions:
                age_minutes = (datetime.now(timezone.utc) - extraction.updated_at).total_seconds() / 60

                is_batch_queued = (
                    extraction.status == ExtractionStatus.QUEUED.value
                    and extraction.batch_id
                    and not extraction.celery_task_id
                )

                # A BATCH MEMBER WAITING ITS TURN IS NOT A STUCK JOB.
                #
                # Only the first member of a batch is dispatched; each successor
                # is started by `_start_next_batch_job` after the previous one's
                # NLP completes. So a queued member has no Celery task and no
                # progress BY DESIGN, for as long as its predecessors take.
                #
                # Judging it by its own age cannot work: one SAE extraction runs
                # ~169 minutes, so the third member of a three-job batch waits
                # ~5.6 hours against a 180-minute grace period and is
                # structurally guaranteed to be failed. That is what happened on
                # 2026-08-28 — reaped at 186 minutes, blamed on a crashed
                # worker, while job 1 had completed and job 2 was mid-run.
                #
                # Judge it by the BATCH instead. This self-scales to any batch
                # size, and a chain that genuinely dies still goes silent and is
                # still reclaimed.
                if is_batch_queued:
                    if batch_has_live_sibling(db, extraction.batch_id, extraction.id):
                        logger.debug(
                            "Extraction %s is waiting behind a live sibling in "
                            "batch %s, skipping", extraction.id, extraction.batch_id,
                        )
                        continue

                    last_activity = batch_last_activity(db, extraction.batch_id)
                    batch_idle_minutes = (
                        (datetime.now(timezone.utc) - last_activity).total_seconds() / 60
                        if last_activity is not None
                        else age_minutes
                    )
                    if batch_idle_minutes < BATCH_QUEUED_THRESHOLD_MINUTES:
                        logger.debug(
                            "Extraction %s: batch %s active %.0fmin ago "
                            "(threshold %dmin), skipping",
                            extraction.id, extraction.batch_id,
                            batch_idle_minutes, BATCH_QUEUED_THRESHOLD_MINUTES,
                        )
                        continue

                    # The chain has stopped. The queued WORK is fine — nothing
                    # dispatched it. Try to restart the chain once before
                    # destroying it; the next sweep re-checks.
                    if _try_restart_batch(db, extraction):
                        continue

                # Check if Celery task is actually running
                task_is_running = False
                if extraction.celery_task_id:

                    # MIS-E2E-092: see task_looks_alive — a dead worker and a
                    # queued task are both PENDING, so the old state check could
                    # never be false for a row carrying a task id.
                    from src.workers.task_heartbeat import task_looks_alive

                    task_is_running = task_looks_alive(
                        extraction.celery_task_id,
                        extraction,
                        started=str(getattr(extraction, "status", "")).lower()
                        in ("extracting", "running", "processing"),
                    )
                    if task_is_running:
                        logger.info(
                            f"Extraction {extraction.id} has an active Celery task "
                            f"{extraction.celery_task_id}, skipping cleanup"
                        )

                if not task_is_running:
                    # IS THE WORK ADVANCING? A quiet row and a stalled job are
                    # different things, and until now only the first was asked
                    # about. `update_extraction_status_sync` reassigns `status`
                    # and `updated_at` on every tick, so a task looping without
                    # advancing keeps its row fresh and looks healthy forever.
                    #
                    # Additive evidence: reap only when the clock says stale AND
                    # the counter has not moved. None means "no evidence" — a
                    # first sighting, or Redis unreachable — and must never be
                    # read as "stalled".
                    # getattr, not attribute access: a row without the field
                    # must degrade to "no evidence". A raise here is caught by
                    # the sweep's outer handler and the ENTIRE janitor does
                    # nothing — a supporting check must never be able to
                    # disable the thing it supports.
                    counter = (
                        getattr(extraction, "progress", None)
                        if not is_batch_queued
                        else None
                    )
                    stalled_seconds = progress_stalled_seconds(
                        "extraction", extraction.id, counter
                    )
                    if (
                        stalled_seconds is not None
                        and stalled_seconds < EXTRACTING_THRESHOLD_MINUTES * 60
                    ):
                        logger.info(
                            "Extraction %s advanced %.0fs ago (progress=%s), "
                            "sparing it despite a stale row",
                            extraction.id, stalled_seconds, counter,
                        )
                        continue

                    threshold_used = (
                        BATCH_QUEUED_THRESHOLD_MINUTES
                        if extraction.batch_id and not extraction.celery_task_id
                        else EXTRACTING_THRESHOLD_MINUTES
                    )
                    logger.warning(
                        f"Marking stuck extraction {extraction.id} as FAILED "
                        f"(status: {extraction.status}, age: {age_minutes:.0f}min, "
                        f"threshold: {threshold_used}min, "
                        f"task_id: {extraction.celery_task_id or 'None'})"
                    )

                    extraction.status = ExtractionStatus.FAILED.value
                    # SAY WHAT HAPPENED. One message for every case told a
                    # user whose batch was working normally that their worker
                    # had crashed.
                    if is_batch_queued:
                        position = (
                            f" (position {extraction.batch_position} of "
                            f"{extraction.batch_total})"
                            if extraction.batch_position
                            else ""
                        )
                        extraction.error_message = (
                            f"Batch stopped advancing{position} - no member of "
                            f"batch {extraction.batch_id} has made progress for "
                            f"{int(age_minutes)} minutes, and restarting the "
                            "chain did not help. This job never started."
                        )
                    else:
                        at_progress = (
                            f" at {extraction.progress * 100:.1f}%"
                            if extraction.progress is not None
                            else ""
                        )
                        extraction.error_message = (
                            f"Extraction stalled{at_progress} - no progress for "
                            f"more than {int(age_minutes)} minutes. This may "
                            "indicate a crashed worker or system issue."
                        )
                    extraction.completed_at = datetime.now(timezone.utc)
                    extraction.updated_at = datetime.now(timezone.utc)

                    db.commit()
                    cleaned_count += 1

                    # Emit WebSocket event to notify frontend
                    try:
                        emit_extraction_job_progress(
                            extraction_id=extraction.id,
                            training_id=extraction.training_id,
                            sae_id=extraction.external_sae_id,
                            status="failed",
                            message=extraction.error_message,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit WebSocket event for {extraction.id}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stuck extraction(s)")
            else:
                logger.info("No stuck extractions found")

            return {"cleaned": cleaned_count}

        except Exception as e:
            logger.error(f"Error in stuck extraction cleanup: {e}", exc_info=True)
            raise
