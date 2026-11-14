"""
Periodic task to clean up stuck NLP analysis passes.

WHY THIS EXISTS
---------------
Every other long-running status in this app has a janitor; ``nlp_status`` did
not. The ``cleanup_stuck_*`` family watches ``ExtractionJob.status``, so an NLP
pass whose worker dies leaves ``nlp_status='processing'`` forever — the row
claims to be working while nothing is. Observed 2026-07-26: a pod roll killed a
pass at 16,217/32,759 features and the row still read "processing" minutes
later, with no mechanism anywhere that would ever correct it.

That is the same failure shape as the queue starvation found the same day: not
a crash, just going quietly dark. Nothing surfaces it, so a user waits on work
that will never finish.

DETECTION
---------
There is deliberately NO Celery-task check here, unlike the sibling cleanups.
``ExtractionJob.celery_task_id`` belongs to the EXTRACTION task, not the NLP
task, and there is no ``nlp_celery_task_id`` column. Consulting it would ask
"is the extraction still running?" — a different question with a confidently
wrong answer.

Staleness alone is a strong signal here precisely because the NLP loop commits
once per feature (measured at ~1.4s per feature). A live pass touches the row
constantly, so silence for the threshold below means the pass is gone. The
threshold is set well above any plausible single-feature stall.

The direction of error is deliberate: ``updated_at`` is shared with non-NLP
writes to the row, so an unrelated write can make a dead pass look fresh. That
produces a FALSE NEGATIVE (cleanup delayed), never a false positive (a live
pass killed). Missing a stuck job is recoverable; failing a running one throws
away hours of work.

WHAT IT DOES NOT DO
-------------------
It does not touch ``ExtractionJob.status``. The extraction itself succeeded —
NLP is post-processing — and marking the extraction failed would hide a
perfectly good feature set.

It does not clear ``nlp_processed_count``. Analysis already written to the
database stays, and the resume path (``force_reprocess=false``) picks up from
there rather than redoing thousands of features.
"""

import logging
from datetime import datetime, timedelta, timezone

from src.core.celery_app import celery_app
from src.models.extraction_job import ExtractionJob
from src.workers.base_task import DatabaseTask
from src.workers.job_progress import progress_stalled_seconds

logger = logging.getLogger(__name__)

# Statuses that assert "work is in flight right now".
IN_FLIGHT_NLP_STATUSES = ("pending", "processing")

# A live pass writes every ~1.4s. Thirty minutes of silence is unambiguous, and
# leaves enormous headroom over any single-feature stall.
NLP_STALE_THRESHOLD_MINUTES = 30

# 'pending' is set by the extraction-completion path just before the NLP task
# is queued. If queueing then fails, the row sits pending forever with nothing
# scheduled — so it gets a shorter grace period than an interrupted run.
NLP_PENDING_THRESHOLD_MINUTES = 15


def _threshold_for(nlp_status: str) -> int:
    return (
        NLP_PENDING_THRESHOLD_MINUTES
        if nlp_status == "pending"
        else NLP_STALE_THRESHOLD_MINUTES
    )


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="cleanup_stuck_nlp",
)
def cleanup_stuck_nlp_task(self):
    """Mark abandoned NLP passes as failed so they stop claiming to run."""
    logger.info("Running stuck NLP cleanup task")

    with self.get_db() as db:
        try:
            now = datetime.now(timezone.utc)
            # Widest threshold selects candidates; per-status logic filters.
            candidate_cutoff = now - timedelta(
                minutes=min(NLP_STALE_THRESHOLD_MINUTES, NLP_PENDING_THRESHOLD_MINUTES)
            )

            candidates = db.query(ExtractionJob).filter(
                ExtractionJob.nlp_status.in_(IN_FLIGHT_NLP_STATUSES),
                ExtractionJob.updated_at < candidate_cutoff,
            ).all()

            cleaned_count = 0
            for job in candidates:
                updated_at = job.updated_at
                if updated_at is None:
                    # No timestamp to judge by — leave it rather than guess.
                    logger.debug("Extraction %s has no updated_at; skipping", job.id)
                    continue
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)

                age_minutes = (now - updated_at).total_seconds() / 60
                threshold = _threshold_for(job.nlp_status)
                if age_minutes < threshold:
                    logger.debug(
                        "NLP for %s is %.0fmin old (threshold %dmin); skipping",
                        job.id, age_minutes, threshold,
                    )
                    continue

                processed = job.nlp_processed_count or 0

                # IS THE WORK ADVANCING? `nlp_processed_count` is this
                # lifecycle's counter. Additive to the clock; None means no
                # evidence and never reads as "stalled". This janitor's
                # documented exemption from task-based liveness is untouched:
                # a progress counter is not a task identifier, and nothing here
                # consults one.
                _stalled = progress_stalled_seconds("nlp", job.id, processed)
                if _stalled is not None and _stalled < threshold * 60:
                    logger.info(
                        "NLP for %s advanced %.0fs ago; sparing it", job.id, _stalled
                    )
                    continue

                logger.warning(
                    "Marking stuck NLP analysis for %s as failed "
                    "(nlp_status=%s, age=%.0fmin, threshold=%dmin, processed=%d)",
                    job.id, job.nlp_status, age_minutes, threshold, processed,
                )

                job.nlp_status = "failed"
                job.nlp_error_message = (
                    f"NLP analysis stopped without finishing - no progress for "
                    f"{int(age_minutes)} minutes. "
                    f"{processed:,} features were analysed and are kept; "
                    "resuming continues from there rather than restarting."
                )
                job.updated_at = now
                # nlp_processed_count and nlp_progress are deliberately left
                # intact so the resume path can pick up where this stopped.

                db.commit()
                cleaned_count += 1

                # Tell the UI, or the card sits on a spinner until a refresh.
                try:
                    from src.workers.nlp_analysis_tasks import emit_nlp_analysis_progress

                    emit_nlp_analysis_progress(
                        extraction_job_id=job.id,
                        event="failed",
                        data={
                            "extraction_job_id": job.id,
                            "status": "failed",
                            "features_analyzed": processed,
                            "error": job.nlp_error_message,
                            "message": job.nlp_error_message,
                        },
                    )
                except Exception as emit_error:
                    logger.warning(
                        "Failed to emit NLP failure for %s: %s", job.id, emit_error
                    )

            if cleaned_count:
                logger.info("Cleaned up %d stuck NLP analysis pass(es)", cleaned_count)
            else:
                logger.info("No stuck NLP analyses found")

            return {"cleaned": cleaned_count}

        except Exception as e:
            logger.error("Error in stuck NLP cleanup: %s", e, exc_info=True)
            db.rollback()
            raise
