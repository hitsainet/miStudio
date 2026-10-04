"""Take, renew and release GPU leases (multi-GPU Phase 3, decision D6).

A lease says which job holds a card. It is taken when a job's card is chosen and
released when the job reaches any terminal state; a holder that stops renewing
loses it at ``expires_at``, so a worker killed mid-job cannot hold a card for
ever. Leases are per application: miLLM keeps none here, and live NVML free
memory stays the only truth shared across applications.

A job split across cards takes every one of its cards in ONE transaction or none
of them. Taking them one at a time would let two split jobs each take one card
and wait for the other's, for ever.

Every function takes a sync SQLAlchemy session (a worker's
``get_sync_db()``) and commits or rolls back itself, so a caller cannot leave a
half-taken lease in an open transaction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.gpu_lease import GpuLease

logger = logging.getLogger(__name__)

#: How long a lease lasts without a renewal. A running job renews far more often
#: (its progress heartbeat); a worker that died mid-job frees its card after this.
LEASE_TTL_SECONDS = 600


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuids(uuids: Iterable[str]) -> list[str]:
    # SORTED, so two transactions taking overlapping cards lock rows in the same
    # order and cannot deadlock each other.
    return sorted({str(uuid) for uuid in uuids if uuid})


def acquire(
    db: Session,
    uuids: Iterable[str],
    holder: str,
    *,
    task_id: Optional[str] = None,
    ttl_seconds: int = LEASE_TTL_SECONDS,
    now: Optional[datetime] = None,
    host_reserve_mb: Optional[int] = None,
) -> bool:
    """Take every card in ``uuids`` for ``holder``, or none of them.

    A card whose lease has expired is free. A card ``holder`` already holds is
    renewed rather than refused, so a retry of the same job is not blocked by
    its own lease.

    ``host_reserve_mb`` is the host RAM the holder will allocate beside its load
    and has not yet (see :func:`host_reserves`); recorded on a NEW row only, so
    a renewal-by-acquire never re-reserves memory the holder already allocated.

    Returns:
        True when ``holder`` holds every card afterwards; False when any card is
        held by someone else (nothing is taken in that case).
    """
    cards = _uuids(uuids)
    if not cards:
        raise ValueError("a lease needs at least one GPU")
    now = now or _utcnow()
    expires = now + timedelta(seconds=ttl_seconds)
    try:
        db.execute(delete(GpuLease).where(GpuLease.gpu_uuid.in_(cards), GpuLease.expires_at <= now))
        existing = {
            row.gpu_uuid: row
            for row in db.execute(
                select(GpuLease).where(GpuLease.gpu_uuid.in_(cards)).with_for_update()
            ).scalars()
        }
        taken = {uuid: row.holder for uuid, row in existing.items() if row.holder != holder}
        if taken:
            db.rollback()
            logger.info("GPU lease for %s refused: held by %s", holder, taken)
            return False
        for uuid in cards:
            if uuid in existing:
                existing[uuid].heartbeat_at = now
                existing[uuid].expires_at = expires
                existing[uuid].task_id = task_id or existing[uuid].task_id
            else:
                db.add(GpuLease(
                    gpu_uuid=uuid, holder=holder, task_id=task_id,
                    acquired_at=now, heartbeat_at=now, expires_at=expires,
                    host_reserve_mb=host_reserve_mb,
                ))
        db.commit()
    except IntegrityError:
        # Another transaction inserted one of these cards between our read and
        # our insert: the primary key refused the second holder. Nothing of
        # ours was committed.
        db.rollback()
        logger.info("GPU lease for %s refused: a card was taken concurrently", holder)
        return False
    logger.info("GPU lease taken by %s on %s until %s", holder, cards, expires.isoformat())
    return True


def renew(
    db: Session,
    holder: str,
    *,
    ttl_seconds: int = LEASE_TTL_SECONDS,
    now: Optional[datetime] = None,
) -> int:
    """Push back the expiry of every live lease ``holder`` has. Returns how many were renewed.

    An already-expired lease is NOT revived: once it has lapsed another job may
    have taken the card, and a renewal must never take it back.
    """
    now = now or _utcnow()
    result = db.execute(
        update(GpuLease)
        .where(GpuLease.holder == holder, GpuLease.expires_at > now)
        .values(heartbeat_at=now, expires_at=now + timedelta(seconds=ttl_seconds))
    )
    db.commit()
    return int(result.rowcount or 0)


def release(db: Session, holder: str, uuids: Optional[Iterable[str]] = None) -> int:
    """Release ``holder``'s leases (all of them, or only ``uuids``). Returns how many.

    Only ``holder``'s own rows: a job releasing late — after its lease expired
    and another job took the card — must not free the other job's lease.
    """
    statement = delete(GpuLease).where(GpuLease.holder == holder)
    if uuids is not None:
        statement = statement.where(GpuLease.gpu_uuid.in_(_uuids(uuids)))
    result = db.execute(statement)
    db.commit()
    released = int(result.rowcount or 0)
    if released:
        logger.info("GPU lease released by %s (%d card(s))", holder, released)
    return released


def release_stale_for_task(
    db: Session,
    task_id: Optional[str],
    *,
    stale_after_s: float,
    now: Optional[datetime] = None,
) -> int:
    """Release the leases of a job a janitor has reaped — only those its holder stopped renewing.

    Keyed by the Celery task id, which a job keeps across hand-offs. A lease whose
    heartbeat is fresher than ``stale_after_s`` is left alone: janitors have
    reaped live jobs before, and freeing a running job's card would put a second
    job on it.
    """
    if not task_id:
        return 0
    now = now or _utcnow()
    result = db.execute(
        delete(GpuLease).where(
            GpuLease.task_id == task_id,
            GpuLease.heartbeat_at <= now - timedelta(seconds=stale_after_s),
        )
    )
    db.commit()
    released = int(result.rowcount or 0)
    if released:
        logger.info("GPU lease of reaped task %s released (%d card(s))", task_id, released)
    return released


def live_leases(db: Session, now: Optional[datetime] = None) -> dict[str, str]:
    """Every card with a live lease, mapped to its holder. Expired leases are left out."""
    now = now or _utcnow()
    rows = db.execute(select(GpuLease).where(GpuLease.expires_at > now)).scalars()
    return {row.gpu_uuid: row.holder for row in rows}


def host_reserves(db: Session, now: Optional[datetime] = None) -> dict[str, Optional[int]]:
    """Each live holder's recorded host RAM reserve not yet allocated, in MB — once per holder.

    None for a holder none of whose rows recorded one (a row from before the
    column): the caller then counts its kind's reserve.
    """
    now = now or _utcnow()
    reserves: dict[str, Optional[int]] = {}
    for row in db.execute(select(GpuLease).where(GpuLease.expires_at > now)).scalars():
        value = row.host_reserve_mb
        known = reserves.get(row.holder)
        reserves[row.holder] = value if known is None else (known if value is None else max(known, value))
    return reserves


def live_holders_for_task(
    db: Session,
    task_id: Optional[str],
    *,
    fresh_after_s: float,
    now: Optional[datetime] = None,
) -> set[str]:
    """Holders of live leases for this Celery task id whose heartbeat is fresh.

    Each is an execution still running the task. A lease its holder stopped
    renewing (older heartbeat than ``fresh_after_s``) belongs to an execution
    that died, and is not counted.
    """
    if not task_id:
        return set()
    now = now or _utcnow()
    rows = db.execute(
        select(GpuLease.holder).where(
            GpuLease.task_id == task_id,
            GpuLease.expires_at > now,
            GpuLease.heartbeat_at > now - timedelta(seconds=fresh_after_s),
        )
    ).scalars()
    return set(rows)


def set_host_reserve(db: Session, holder: str, reserve_mb: int) -> int:
    """Set ``holder``'s recorded host reserve on all its rows (0 once allocated). Returns how many."""
    result = db.execute(
        update(GpuLease).where(GpuLease.holder == holder).values(host_reserve_mb=int(reserve_mb))
    )
    db.commit()
    return int(result.rowcount or 0)
