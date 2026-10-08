"""A GPU lease: which job holds a card (multi-GPU Phase 3, decision D6).

Phase 3 runs one job per card at once, and an Auto job's card is chosen when a
worker is free to run it (decision 5, 2026-09-14). NVML free memory alone cannot
keep two jobs off one card: a job that has just been placed has not allocated yet,
so a second placement a moment later sees the card as free. A lease records the
choice the moment it is made.

One row per card, keyed by the card's UUID, so the primary key itself refuses a
second holder. A job split across cards holds one row per card, taken together
in one transaction or not at all. A lease expires unless its holder renews it,
so a worker that dies without releasing cannot hold a card for ever.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from ..core.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GpuLease(Base):
    """The job holding one GPU, until it releases the card or stops renewing."""

    __tablename__ = "gpu_leases"

    #: The card, as NVML reports its UUID (``GPU-…``).
    gpu_uuid = Column(String(64), primary_key=True)
    #: Who holds it: a stable job identity such as ``training:train_ab12``.
    holder = Column(String(128), nullable=False)
    task_id = Column(String(155), nullable=True)
    acquired_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    heartbeat_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    #: After this the card is free again, whoever the row names.
    expires_at = Column(DateTime(timezone=True), nullable=False)
    #: Host RAM (MB) the holder will allocate beside its load and has NOT yet —
    #: a cached-activation training's pinned rolling buffer. Zeroed once
    #: allocated, when MemAvailable already reflects it. NULL: a row from before
    #: this column, which counts its kind's reserve (review round 1, R1-6).
    host_reserve_mb = Column(Integer, nullable=True)
