"""Where a worker runs the job it just took, or where it sends it (multi-GPU Phase 3).

Every card gets its own solo worker. A job that names a GPU is queued on that
card's queue; an Auto or ``"all"`` job is queued on one shared queue that every
card's worker consumes. The card is therefore chosen when a worker is FREE to run
the job, against live free memory (decision 5, 2026-09-14) — not when it is
submitted, which would leave a job waiting for a busy card while another sat idle.

The worker that takes a job calls :func:`decide_claim` and gets one answer:

* :class:`RunHere` — take the leases on these cards and run.
* :class:`SendTo` — the job belongs on another card's queue: a named card, or a
  more-free idle card (decision 1 still holds: Auto takes the most free card that
  fits, so the first free worker does not simply keep a job another card fits
  better).
* :class:`WaitForCard` — the job could run, but the cards it needs are held by
  other jobs' leases; put it back and try again later.
* :class:`Refuse` — nothing on this node could ever run it as asked.

The decision is a pure function of the inventory and the leases, so every rule
is testable without a GPU. Taking the leases is a separate, transactional step
(:mod:`services.gpu_leases`); a lease lost to a race sends the job back through
this decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Union

from .gpu_placement import (
    AUTO,
    SHARD_RESERVE_MB,
    GpuCard,
    GpuPlacementError,
    GpuRequest,
    find_card,
    is_all,
    is_auto,
    normalise_uuid,
    resolve_card,
    resolve_cards,
)

#: The queue every card's worker consumes: Auto and "all" jobs wait here.
AUTO_QUEUE = "gpu.auto"


def queue_for(uuid: str) -> str:
    """The queue of one card. ONE spelling everywhere: a queue name that differs
    in case from the one a worker consumes would be created silently by Celery
    and its messages held until the visibility timeout."""
    return f"gpu.{normalise_uuid(uuid)}"


@dataclass(frozen=True)
class RunHere:
    """Run on these cards (one card, or a split), most free first as ``resolve_cards`` chose them.

    Not the order a split fills them: transformers fills a split in CUDA index order.
    """

    cards: tuple[GpuCard, ...]


@dataclass(frozen=True)
class SendTo:
    """Re-queue the job on another card's queue."""

    queue: str
    reason: str


@dataclass(frozen=True)
class WaitForCard:
    """The cards the job needs are leased by other jobs; try again later."""

    reason: str


@dataclass(frozen=True)
class Refuse:
    """No card, or set of cards, on this node can run the job as asked."""

    error: GpuPlacementError


Claim = Union[RunHere, SendTo, WaitForCard, Refuse]


def decide_claim(
    *,
    worker_uuid: str,
    requested: GpuRequest,
    required_mb: Optional[float],
    allow_shard: bool,
    cards: Iterable[GpuCard],
    leases: Mapping[str, str],
    holder: str,
) -> Claim:
    """What the worker for ``worker_uuid`` should do with a job it has just taken.

    Args:
        worker_uuid: The card this worker runs jobs on.
        requested: The job's ``gpu_request`` — ``"auto"``, ``"all"`` or a UUID.
        required_mb: The job's estimated need, when known.
        allow_shard: Whether the job's code can run split across cards.
        cards: The live inventory.
        leases: Live leases, card UUID -> holder (see ``gpu_leases.live_leases``).
        holder: This job's lease identity. A card it already holds is free to it.
    """
    cards = list(cards)
    held_by_others = {
        normalise_uuid(uuid) for uuid, who in leases.items() if who != holder
    }

    def idle(card: GpuCard) -> bool:
        return normalise_uuid(card.uuid) not in held_by_others

    mine = next((c for c in cards if normalise_uuid(c.uuid) == normalise_uuid(worker_uuid)), None)
    if mine is None:
        return Refuse(GpuPlacementError(
            f"This worker's GPU {worker_uuid} is not visible; it cannot run jobs.",
            requested=requested, required_mb=required_mb, cards=cards,
        ))

    if is_all(requested):
        if not allow_shard:
            return Refuse(GpuPlacementError(
                "This job cannot run split across GPUs. Choose Auto or one GPU.",
                requested=requested, required_mb=required_mb, cards=cards,
            ))
        busy = [c for c in cards if not idle(c)]
        if busy:
            return WaitForCard(f"waiting for every GPU; {len(busy)} held by other jobs")
        try:
            return RunHere(resolve_cards(requested, required_mb=required_mb, cards=cards, allow_shard=True))
        except GpuPlacementError as exc:
            return Refuse(exc)

    if not is_auto(requested):
        try:
            named = find_card(requested, cards)
        except GpuPlacementError as exc:
            return Refuse(exc)
        if normalise_uuid(named.uuid) != normalise_uuid(mine.uuid):
            return SendTo(queue_for(named.uuid), f"the job names {named.describe()}")
        if not idle(named):
            return WaitForCard(f"{named.describe()} is held by another job")
        try:
            return RunHere((resolve_card(requested, required_mb=required_mb, cards=cards),))
        except GpuPlacementError as exc:
            return Refuse(exc)

    # Auto: decide among the cards no other job holds.
    idle_cards = [c for c in cards if idle(c)]
    if not idle_cards:
        return WaitForCard("every GPU is held by another job")

    fitting = [c for c in idle_cards if required_mb is None or c.free_mb >= required_mb]
    if fitting:
        best = max(fitting, key=lambda c: (c.free_mb, -c.index))
        if normalise_uuid(best.uuid) == normalise_uuid(mine.uuid):
            return RunHere((mine,))
        return SendTo(queue_for(best.uuid), f"{best.describe()} is idle and has the most free memory")

    if allow_shard and required_mb is not None and len(idle_cards) >= 2:
        try:
            split = resolve_cards(AUTO, required_mb=required_mb, cards=idle_cards, allow_shard=True)
        except GpuPlacementError:
            split = ()
        if len(split) > 1:
            if any(normalise_uuid(c.uuid) == normalise_uuid(mine.uuid) for c in split):
                return RunHere(split)
            return SendTo(queue_for(split[0].uuid), "a split over idle GPUs that excludes this one")

    # NEVER WAIT FOR WHAT CAN NEVER COME. Waiting is right only if releasing the
    # busy cards could make room; a job larger than any card (or, split, larger
    # than every card together) would otherwise be re-queued for ever.
    largest_single = max(c.total_mb for c in cards)
    split_capacity = sum(max(c.total_mb - SHARD_RESERVE_MB, 0) for c in cards) if allow_shard and len(cards) >= 2 else 0
    could_ever_fit = required_mb is None or required_mb <= max(largest_single, split_capacity)

    if could_ever_fit and len(idle_cards) < len(cards):
        # Some cards are busy: the job may fit once they are released.
        return WaitForCard(f"no idle GPU can hold ~{required_mb:,.0f} MB; waiting for a GPU to be released")

    try:
        # Every card is idle and nothing fits, or nothing ever could: this is the
        # Phase 1/2 refusal, with its figures.
        resolve_cards(AUTO, required_mb=required_mb, cards=cards, allow_shard=allow_shard)
    except GpuPlacementError as exc:
        return Refuse(exc)
    # resolve_cards found a placement the checks above did not (a card's memory
    # counted as busy became free meanwhile). Never run on a placement this
    # function did not decide: go round again.
    return WaitForCard("placement changed while deciding; trying again")
