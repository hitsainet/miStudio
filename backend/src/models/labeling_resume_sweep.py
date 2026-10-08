"""A multi-batch labeling resume, driven one batch at a time.

WHY A ROW AND NOT A LOOP.

Finishing L46 is 27 batches of 2000 at the measured ~8 s/feature — about 59
GPU-hours. Celery's soft limit here is 10 h and the hard limit 12 h, so a task
that looped over batches would be killed part-way through and, because the queue
is `acks_late`, would strand its message for the full 12 h visibility timeout.
That exact shape has already cost this project one outage.

So a sweep is not a long task. It is a ROW plus a task that does ONE batch and
re-enqueues itself. Each step gets a fresh soft limit, the row survives worker
restarts, and an operator can stop it between any two batches.

The row is also the only honest place to record progress. Counting batches says
nothing: a sweep whose every batch failed would report "27 of 27 complete". The
counters here come from what each batch actually WROTE.
"""

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from ..core.database import Base


class LabelingResumeSweep(Base):
    """One operator-initiated run of "keep resuming until it is done"."""

    __tablename__ = "labeling_resume_sweeps"

    id = Column(String(64), primary_key=True)  # sweep_{uuid}

    extraction_job_id = Column(
        String(255),
        ForeignKey("extraction_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    #: The judge configuration every batch reuses, frozen at start.
    #:
    #: A sweep whose batches drift between models or templates produces labels
    #: that cannot be compared with each other, and nothing afterwards can say
    #: which half came from which. Frozen here rather than re-read per batch,
    #: because a template edited mid-sweep would silently change the judge.
    config = Column(JSONB, nullable=False, default=dict)

    #: running | completed | failed | cancelled
    status = Column(String(16), nullable=False, default="running", index=True)

    batches_done = Column(Integer, nullable=False, default=0)
    #: Counted from what each batch WROTE, never from batch count — a sweep whose
    #: batches all fail must not report itself complete.
    features_labeled = Column(Integer, nullable=False, default=0)
    features_failed = Column(Integer, nullable=False, default=0)

    #: REQUIRED, with no unbounded default anywhere that constructs one.
    #: An open-ended sweep is a request to spend an unknown number of GPU-hours,
    #: and neither an operator nor an agent should start one by omitting a
    #: parameter.
    max_batches = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False, default=2000)

    #: Cooperative stop, written by `request_cancel("labeling_sweep", id)`.
    #:
    #: NAMED FOR THE FRAMEWORK, not invented beside it. The first version of this
    #: table carried a `stop_requested_at` of its own while `core.cancellation`
    #: already provided the mechanism — a registered scope wired to nothing, and
    #: a second way to express the same intent that could disagree with the
    #: first. `test_cancel_registry_completeness` caught all four halves of that:
    #: no route called `request_cancel`, nothing polled the scope, no test named
    #: it, and the declared `progress_field` was not a column here.
    #:
    #: The pool is `--pool=solo -c 1`, so `revoke(terminate=True)` signals a pool
    #: child that does not exist and a busy solo worker never reads the control
    #: queue. Stopping has to be something the task itself checks, and it does,
    #: before every batch.
    cancel_requested_at = Column(DateTime(timezone=True), nullable=True)

    #: 0.0-1.0, so the registry's `progress_field` names a real column.
    #: `record_progress` refuses to move a terminal row, which is what stops an
    #: in-flight write from overwriting a CANCELLED status seconds later.
    progress = Column(Float, nullable=False, default=0.0)

    #: The batch currently in flight, for an operator following along.
    last_labeling_job_id = Column(String(255), nullable=True)
    error_message = Column(String(1000), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<LabelingResumeSweep(id={self.id}, status={self.status}, "
            f"batches={self.batches_done}/{self.max_batches}, "
            f"labeled={self.features_labeled})>"
        )
