"""Run a labeling resume across many batches, one batch per task.

WHY THIS IS NOT A LOOP.

Finishing L46 is ~27 batches of 2000 at the measured ~8 s/feature — about 59
GPU-hours. Celery's soft limit here is 10 h and the hard limit 12 h, so a task
that looped over batches would be killed part-way through and, on an `acks_late`
queue, strand its message for the full 12 h visibility timeout. That shape has
already cost this project an outage once.

So `advance` does exactly ONE batch and returns whether another is wanted. The
task layer re-enqueues. Each step gets a fresh soft limit, the row survives a
worker restart, and an operator can stop between any two batches.

WHY PROGRESS IS NOT A BATCH COUNT.

A sweep whose every batch failed would report "27 of 27 complete" if it counted
batches. The counters here read what each batch's own job recorded — which is
truthful now that a failure writes `label_status='failed'` instead of a fake
label.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from ..core.cancellation import is_cancelled, request_cancel
from ..models.extraction_job import ExtractionJob
from ..models.labeling_job import LabelingJob, LabelingStatus
from ..models.labeling_resume_sweep import LabelingResumeSweep
from ..services import labeling_eligibility

logger = logging.getLogger(__name__)

#: A sweep cannot be started without a ceiling, and cannot be given a silly one.
#: 100 batches x 2000 x ~8 s is ~444 GPU-hours; beyond that an operator is
#: describing a different kind of job than this feature is for.
MAX_ALLOWED_BATCHES = 100


class SweepError(Exception):
    """A sweep cannot be started or advanced."""


class LabelingSweepService:
    def __init__(self, db: Session):
        self.db = db

    # ── start ────────────────────────────────────────────────────────────────

    def create(
        self,
        extraction_job_id: str,
        *,
        max_batches: int,
        config: Dict[str, Any],
        batch_size: int = 2000,
    ) -> LabelingResumeSweep:
        """Record the intent to sweep. Does NOT start work.

        `max_batches` is required by the signature and by the column. An
        open-ended sweep is a request to spend an unknown number of GPU-hours,
        and it must not be reachable by omitting a parameter — from the API, from
        an MCP tool, or from a later caller who did not read this docstring.
        """
        if max_batches < 1 or max_batches > MAX_ALLOWED_BATCHES:
            raise SweepError(
                f"max_batches must be 1..{MAX_ALLOWED_BATCHES}; got {max_batches}. "
                f"At ~8 s/feature, {MAX_ALLOWED_BATCHES} batches of {batch_size} "
                "is already several hundred GPU-hours."
            )
        if self.db.query(ExtractionJob).filter(
            ExtractionJob.id == extraction_job_id
        ).first() is None:
            raise SweepError(f"Extraction job {extraction_job_id} not found")

        # One running sweep per extraction. Two would compute overlapping
        # batches from the same `pending` rows and pay twice for one result;
        # claiming narrows that window but does not close it.
        existing = (
            self.db.query(LabelingResumeSweep)
            .filter(
                LabelingResumeSweep.extraction_job_id == extraction_job_id,
                LabelingResumeSweep.status == "running",
            )
            .first()
        )
        if existing is not None:
            raise SweepError(
                f"Sweep {existing.id} is already running on this extraction. "
                "Stop it before starting another."
            )

        sweep = LabelingResumeSweep(
            id=f"sweep_{uuid.uuid4().hex[:16]}",
            extraction_job_id=extraction_job_id,
            # FROZEN. A template edited mid-sweep would otherwise change the
            # judge between batches, and nothing afterwards could say which half
            # came from which.
            config=dict(config),
            status="running",
            max_batches=max_batches,
            batch_size=batch_size,
        )
        self.db.add(sweep)
        self.db.commit()
        self.db.refresh(sweep)
        return sweep

    # ── one step ─────────────────────────────────────────────────────────────

    def next_batch_ids(self, sweep: LabelingResumeSweep) -> list:
        """The next batch, under the SAME predicate the sweep was sized with.

        R3 FINDING, and the same defect the resume button had one frame up.

        The card computes `max_batches` from a coverage read that carries the
        template fingerprint and the judge, so the sweep is sized against the
        stale-inclusive backlog. This selected with neither, i.e. `pending` +
        `failed` only.

        After a template edit — which changes every fingerprint — the operator
        is quoted twenty batches and ~59 GPU-hours, confirms, and the sweep
        either terminates on the first empty batch having done nothing, or
        relabels a set twenty times smaller than the one it was sized for while
        the stale features it was booked to cover are never touched.

        The config is already frozen on the sweep row for exactly this reason:
        a sweep must keep running the run it was started as, even if someone
        edits the template underneath it.
        """
        config = sweep.config or {}
        rows = self.db.execute(
            labeling_eligibility.resume_batch_query(
                sweep.extraction_job_id,
                limit=sweep.batch_size,
                prompt_fingerprint=config.get("prompt_fingerprint"),
                judge_model=config.get("judge_model"),
            )
        )
        return [r[0] for r in rows.all()]

    def active_labeling_job(self, extraction_job_id: str) -> Optional[LabelingJob]:
        """A labeling job already running on this extraction, if any.

        R2 FINDING. `start_labeling` refuses to start a second job on an
        extraction that already has one QUEUED or LABELING. The sweep task builds
        its row through `build_labeling_job_row` and so BYPASSED that guard
        entirely — two deliveries of the same step (an `acks_late` redelivery, a
        hand re-trigger, a worker that lost its connection mid-ack) would each
        create a batch job on the same extraction.

        Per-feature claiming narrows the damage to one batch's worth of
        duplicated work; this closes it. The guard belongs here rather than
        inline in the task so it can be tested without a broker.
        """
        return (
            self.db.query(LabelingJob)
            .filter(
                LabelingJob.extraction_job_id == extraction_job_id,
                LabelingJob.status.in_(
                    [LabelingStatus.QUEUED.value, LabelingStatus.LABELING.value]
                ),
            )
            .first()
        )

    def should_stop(self, sweep: LabelingResumeSweep) -> Optional[str]:
        """Why this sweep must not take another batch, or None to continue.

        Checked BEFORE the work, every step. A stop that is only honoured
        between whole sweeps is not a stop.
        """
        # The registry's vocabulary, not a literal "cancelled": the scope is
        # where that word is written down, and a second spelling here could
        # disagree with what `request_cancel` actually wrote.
        if is_cancelled("labeling_sweep", sweep.status):
            return "cancelled by the operator"
        if sweep.status != "running":
            return f"sweep is {sweep.status}"
        # The TIMESTAMP is checked separately and deliberately. It separates
        # "the operator asked" from "the sweep stopped", and here that gap is a
        # whole batch — up to 4.4 hours — so a sweep finishing its current batch
        # must be distinguishable from one ignoring the request.
        if sweep.cancel_requested_at is not None:
            return "stop requested by the operator"
        if sweep.batches_done >= sweep.max_batches:
            return f"reached its ceiling of {sweep.max_batches} batches"
        return None

    def finish(self, sweep: LabelingResumeSweep, status: str, reason: str = "") -> None:
        sweep.status = status
        sweep.completed_at = datetime.now(timezone.utc)
        sweep.updated_at = sweep.completed_at
        if reason and status == "failed":
            sweep.error_message = reason[:1000]
        self.db.commit()
        logger.info("Sweep %s %s: %s", sweep.id, status, reason or "-")

    def record_batch(self, sweep: LabelingResumeSweep, job: LabelingJob) -> None:
        """Fold one finished batch's real outcome into the sweep.

        Reads the JOB's counters, not the batch size. A batch that failed every
        feature must move `features_failed`, never `features_labeled` — the
        distinction the whole adjudication arc exists to preserve.
        """
        stats = job.statistics or {}
        sweep.batches_done += 1
        sweep.features_labeled += int(stats.get("successfully_labeled", 0) or 0)
        sweep.features_failed += int(stats.get("failed_labels", 0) or 0)
        sweep.last_labeling_job_id = job.id
        sweep.updated_at = datetime.now(timezone.utc)
        self.db.commit()

    def request_stop(self, sweep_id: str) -> Optional[LabelingResumeSweep]:
        sweep = self.db.query(LabelingResumeSweep).filter(
            LabelingResumeSweep.id == sweep_id
        ).first()
        if sweep is None:
            return None
        # Only a RUNNING sweep can be stopped. Stamping a terminal row would
        # rewrite history and make a completed sweep look abandoned.
        if sweep.status == "running":
            # Through the registry, so this sweep stops the same way every other
            # lifecycle in the product does — and so the janitors and the
            # startup reconciliation see it.
            request_cancel("labeling_sweep", sweep.id, db=self.db)
            self.db.refresh(sweep)
        return sweep

    def get(self, sweep_id: str) -> Optional[LabelingResumeSweep]:
        return self.db.query(LabelingResumeSweep).filter(
            LabelingResumeSweep.id == sweep_id
        ).first()
