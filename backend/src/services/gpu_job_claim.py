"""Claim a GPU before a job places on it (multi-GPU Phase 3).

``place_job`` asks :func:`claim_cards` for its cards. With no claim in progress
(single-worker mode, the API process, tests) that is exactly Phase 2's
``resolve_cards``. Inside a job the claim wrapper started (``workers/gpu_job.py``)
it is a claim:

1. read the live inventory and the live leases;
2. ``gpu_claim.decide_claim`` with the job's EXACT size, known only here;
3. RunHere: take the leases (every card of a split, all or none), start renewing
   them, and return THOSE cards — placement never chooses among any others;
   SendTo / WaitForCard: hand the job back (a :class:`JobHandoff`, which the
   wrapper turns into a re-dispatch) or, for a job that cannot be re-dispatched
   part-way, wait here; Refuse: raise the placement error the job already
   records as its failure.

THE HOLDER IS ONE EXECUTION, not one job. ``make_holder`` mixes the Celery task
id with a nonce minted per execution, so a redelivered duplicate of a job, a
retry, or a hand-off that raced its own original can neither share another
execution's lease nor release it. ``gpu_leases.release`` deletes only the
holder's own rows.

TWO WAYS TO NOT RUN NOW.

* Hand off (card workers, whole-GPU tasks). The job has done nothing that
  touches a GPU, so the wrapper re-publishes it with the same task id and acks
  this delivery. A job that must wait is PARKED in Redis
  (``services/gpu_worker_queues.py``) and republished when due: the solo worker
  is free at once, and nothing sits unacked in a worker's memory, which is what a
  Celery ``countdown`` does on the Redis transport — and a worker killed while
  holding one strands it for the 12-hour visibility timeout.
* Wait in place (tasks whose GPU step comes after work that must not be
  repeated — a model download's inspection load, a local labeling judge, a
  steering generation). Polled on a TIME interval, bounded by the task's own
  ``wait_timeout_s``, then refused with a message saying so.

Plan: ``0xcc/plans/Multi-GPU-Plan.md``, Phase 3.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
import time
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from . import gpu_leases
from .gpu_claim import RunHere, Refuse, SendTo, WaitForCard, decide_claim, queue_for
from .gpu_placement import (
    GpuCard,
    GpuPlacementError,
    GpuRequest,
    find_card,
    is_all,
    is_auto,
    list_cards,
    normalise_uuid,
    resolve_cards,
)

logger = logging.getLogger(__name__)

#: How often a running job pushes back its leases' expiry. On a TIMER, never on
#: a count of steps or batches: a training step can be milliseconds and a
#: circuit attribution batch minutes. Ten renewals fit in one
#: ``gpu_leases.LEASE_TTL_SECONDS``.
RENEW_INTERVAL_S = 60.0

#: A lease whose heartbeat is older than this belongs to a holder that has
#: stopped renewing. Only such a lease is released by a janitor: a janitor that
#: falsely reaps a LIVE job (it has happened — see long-phases-need-a-db-heartbeat)
#: must not also hand its card to another job while it still runs there.
STALE_HEARTBEAT_S = 3 * RENEW_INTERVAL_S

#: How long a handed-off job waits before it is offered to a worker again.
HANDOFF_WAIT_S = 30.0

#: A job waiting in place re-reads the leases this often.
IN_PLACE_POLL_S = 10.0

#: The default bound on waiting in place.
DEFAULT_IN_PLACE_TIMEOUT_S = 30 * 60.0

#: Leases lost to a concurrent claim before the job waits instead of retrying.
MAX_ACQUIRE_RACES = 3

#: How long a finished job's memory release may keep its cards leased. Past it
#: the lease is released anyway, with a warning: a stalled release (a read
#: thread blocked on disk) must not hold a card for ever, and the lease TTL is
#: no better an answer. Well inside ``gpu_leases.LEASE_TTL_SECONDS``, and the
#: renewal keeps running while the release does.
RELEASE_MEMORY_TIMEOUT_S = 120.0

UNCLAIMED_MESSAGE = (
    "A GPU job was placed without a GPU lease claim. In per-card mode every task "
    "that places on a GPU must be wrapped by workers.gpu_job.gpu_job, or two jobs "
    "can land on one card. This is a wiring defect, not a busy GPU."
)


class JobHandoff(BaseException):
    """This worker will not run the job now: re-queue it on ``queue`` after ``delay_s``.

    A BaseException for the reason ``core.cancellation.OperatorCancelled`` is:
    tasks here wrap their bodies in ``except Exception`` handlers that write
    FAILED, and J-lens's ``owns_its_failure`` records any exception. A hand-off
    is neither a failure nor an error.
    """

    def __init__(self, queue: str, reason: str, delay_s: float = 0.0) -> None:
        self.queue = queue
        self.reason = reason
        self.delay_s = float(delay_s)
        super().__init__(f"{reason} -> {queue}" + (f" in {self.delay_s:.0f} s" if self.delay_s else ""))


class DuplicateExecution(BaseException):
    """Another execution of THIS task id holds a live, renewed lease: this delivery is a copy.

    Every task here acks late, and kombu's Redis transport restores an unacked
    message older than ``visibility_timeout`` (12 h) to its queue whether or not
    its consumer is still alive. With one GPU worker the copy waited behind the
    original; with a worker per card the other card's worker takes it at once and
    would run the same job concurrently. The wrapper acks and drops it.
    A BaseException for the reason :class:`JobHandoff` is.
    """

    def __init__(self, task_id: str, holders: Iterable[str]) -> None:
        self.task_id = task_id
        self.holders = sorted(holders)
        super().__init__(f"task {task_id} is still running as {self.holders}")


class GpuLeaseLost(GpuPlacementError):
    """GPU work stopped because its job can no longer prove it holds its GPU lease.

    For work with no cooperative cancellation check of its own — a logit lens over
    every feature of an SAE, in a Celery task or in the API process — raised at its
    next batch by :func:`raise_if_lease_lost` (Phase 3 review round 2). A
    ``GpuPlacementError``, so a caller that fails a job with a placement's message
    rather than skipping the step (the Neuronpedia export) fails it with this one.
    """


def make_holder(kind: str, task_id: Optional[str]) -> str:
    """A lease holder for ONE execution of a job. See the module docstring."""
    nonce = _uuid.uuid4().hex[:8]
    return f"{kind[:40]}:{(task_id or 'no-task')[:70]}:{nonce}"


def host_available_mb() -> float:
    """Host memory the kernel can hand out now (psutil ``available``: free + reclaimable cache)."""
    import psutil

    return psutil.virtual_memory().available / 2**20


#: Host memory a job of a KIND holds beyond its model load, keyed by the lease
#: holder's kind (``make_holder``). A cached-activation SAE training prepares its
#: next rolling buffer in PINNED host memory: about 24 GiB for three layers, since
#: torch rounds pinned blocks up to a power of two (Phase 4 buffer review,
#: `14acc05a`). Pinned memory cannot be reclaimed, and the worker containers have
#: no cgroup limit, so the host's MemAvailable is the only bound — and a worker is
#: the first OOM-kill victim when it runs out.
HOST_RESERVE_MB_BY_KIND = {"training": 24 * 1024}

#: Lease holder kinds that are not jobs and load nothing: the GPU supervisor's
#: "card unavailable" lease on a card whose worker is down
#: (``workers.gpu_supervisor.CardAvailability.HOLDER_PREFIX``; a test pins the two
#: equal). The host RAM guard does not count them as another job loading.
NOT_A_JOB_HOLDER_KINDS = frozenset({"gpu-worker-down"})


def holder_kind(holder: str) -> str:
    return holder.split(":", 1)[0]


def host_ram_refusal(
    required_mb: Optional[float],
    *,
    others_hold_leases: bool,
    available_mb: float,
    other_reserves_mb: float = 0.0,
    own_reserve_mb: float = 0.0,
) -> Optional[str]:
    """Why a job must not start loading beside another job for host RAM, or None.

    THE ESTIMATE. This job's host peak is the larger of ``required_mb`` — the load
    preflight's own figure (``resource_config``: weights at load precision,
    ``_BYTES_PER_PARAM``, plus ``_ACTIVATION_HEADROOM_GB``) — and its kind's host
    reserve. transformers memory-maps safetensors and materialises each tensor on
    its device-map target (the Phase 2 note on decision 3), so a GPU-only load's
    host peak is at most about one copy of the weights; the figure overstates an
    FP16 load by the headroom, and can understate a bitsandbytes load whose
    checkpoint is FP16 (an open risk in the plan).

    WHAT IS AVAILABLE. psutil's ``available`` is the kernel's MemAvailable, which
    already excludes memory other processes hold — a training's pinned buffer
    included, once allocated. A training on another card may not have allocated
    it YET, so every other live holder's kind reserve is subtracted as well. A
    training that already holds its buffer is therefore counted twice; on the
    124 GB node that costs about 24 GB of headroom, which is the conservative
    side to be wrong on. ``ResourceConfig.RAM_SAFETY_MARGIN`` of what is available
    is kept free on top.

    Only a SECOND concurrent job is refused: with no other job holding a lease
    nothing else is loading, and the first load was never guarded.
    """
    need = max(required_mb or 0.0, own_reserve_mb or 0.0)
    if need <= 0 or not others_hold_leases:
        return None
    from .resource_config import ResourceConfig

    usable = available_mb * (1 - ResourceConfig.RAM_SAFETY_MARGIN) - other_reserves_mb
    if usable >= need:
        return None
    return (
        f"host RAM: starting beside another job needs ~{need:,.0f} MB; {available_mb:,.0f} MB is "
        f"available, {ResourceConfig.RAM_SAFETY_MARGIN:.0%} of it kept free and "
        f"{other_reserves_mb:,.0f} MB reserved for the jobs already running"
    )


#: The ``OperatorCancelled.reason`` of a job stopped because it can no longer
#: prove it holds its GPU lease. Distinct from an operator's "cancelled": the
#: row is not terminal yet, so the stop is recorded as a FAILURE with the detail.
#: Defined once, in ``core.cancellation``, which records it.
from ..core.cancellation import GPU_LEASE_LOST as LEASE_LOST_REASON  # noqa: E402


class LeaseRenewer:
    """Renews one holder's leases on a timer thread until stopped.

    A LEASE THAT LAPSES IS TAKEN BACK OR GIVEN UP, NEVER JUST LOGGED. Renewal
    never revives an expired row (``gpu_leases.renew``). When a renewal finds
    the lease gone — a database outage, a stalled process, a janitor that freed
    a stale heartbeat — the renewer asks for the SAME cards again with
    ``gpu_leases.acquire``, which succeeds only if no other job holds any of
    them. If another job does, or if no renewal has succeeded for a whole TTL
    (by then the lease has expired whatever the database says, and any job that
    can reach it may take the card), the lease is LOST: ``lost`` carries the
    reason, and the job stops at its next cancellation check
    (``core.cancellation.CancelCheck``, training's status check) with
    :data:`LEASE_LOST_REASON`.

    Why stop rather than keep running: a job on a card it no longer holds can
    share it with a job that was placed there by the rules — two models on one
    card, an out-of-memory in whichever allocates next. Why re-acquire first: a
    lapse nobody exploited costs nothing, and ``acquire`` is all-or-none under
    row locks, so taking the cards back can never take one another job holds.
    """

    def __init__(
        self,
        holder: str,
        session: Callable[[], Any],
        interval_s: float = RENEW_INTERVAL_S,
        on_renew: Optional[Callable[[int], None]] = None,
        *,
        uuids: Iterable[str] = (),
        task_id: Optional[str] = None,
        ttl_s: float = gpu_leases.LEASE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        reserve_mb: Optional[Callable[[], Optional[int]]] = None,
    ) -> None:
        self.holder = holder
        #: The host RAM reserve the job has NOT yet allocated, for a take-back's new rows
        #: (review round 2). None records no reserve, as before.
        self._reserve_mb = reserve_mb
        self._session = session
        self.interval_s = interval_s
        self._on_renew = on_renew
        self.uuids = tuple(uuids)
        self.task_id = task_id
        self._ttl_s = float(ttl_s)
        self._clock = clock
        self._last_renewed = clock()
        #: Why this holder can no longer prove it holds its cards, or None.
        self.lost: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"gpu-lease-renew-{holder[:24]}"
        )

    def start(self) -> None:
        self._thread.start()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _lose(self, reason: str) -> None:
        if self.lost is None:
            self.lost = reason
            logger.error("GPU lease of %s LOST: %s. The job stops at its next check.", self.holder, reason)

    def tick(self) -> int:
        """One renewal. Returns how many leases were renewed; -1 when the database could not be reached."""
        if self.lost is not None:
            return 0
        renewed = -1
        try:
            with self._session() as db:
                renewed = gpu_leases.renew(db, self.holder)
                if renewed == 0 and self.uuids:
                    # A STOPPED RENEWER NEVER TAKES A CARD BACK (Phase 3 review round 2).
                    # `ClaimContext.close()` stops the renewer with a BOUNDED join and then
                    # deletes the holder's rows; a renewal still inside the database when
                    # the join gives up finds them gone and reads the release as a lapse.
                    # Taking the card back then leases it to a finished job for a whole TTL.
                    if self._stop.is_set():
                        return 0
                    reserve = self._reserve_mb() if self._reserve_mb is not None else None
                    if gpu_leases.acquire(db, self.uuids, self.holder, task_id=self.task_id,
                                          host_reserve_mb=reserve):
                        # The stop may have landed after the check and before the commit:
                        # `close()` sets it before it deletes, so a take-back that commits
                        # after the stop either is deleted by close() or is given up here.
                        if self._stop.is_set():
                            gpu_leases.release(db, self.holder)
                            logger.info("GPU lease of %s was taken back as its job ended; given up again",
                                        self.holder)
                            return 0
                        logger.warning(
                            "GPU lease of %s had lapsed and was taken back on %s; no other job "
                            "held the card(s) meanwhile", self.holder, list(self.uuids),
                        )
                        renewed = len(self.uuids)
                    else:
                        self._lose(
                            f"its lease on {list(self.uuids)} lapsed and another job now holds "
                            "the card; it cannot keep running there"
                        )
        except Exception:  # noqa: BLE001 - a failed renewal must not kill the job by itself
            logger.exception("Could not renew the GPU lease of %s", self.holder)
        if renewed > 0:
            self._last_renewed = self._clock()
        elif self.lost is None and self._clock() - self._last_renewed >= self._ttl_s:
            self._lose(
                f"its lease on {list(self.uuids)} could not be renewed for "
                f"{self._clock() - self._last_renewed:.0f} s, past its {self._ttl_s:.0f} s lifetime, "
                "so another job may have been placed on the card"
            )
        return renewed

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            renewed = self.tick()
            if self._on_renew is not None:
                try:
                    self._on_renew(renewed)
                except Exception:  # noqa: BLE001 - a callback must not end the renewals
                    logger.exception("GPU lease renewal callback failed for %s", self.holder)
            if self.lost is not None:
                return

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)


def _sync_session():
    from ..core.database import get_sync_db

    return get_sync_db()


@dataclass
class ClaimContext:
    """One execution's claim: who it is, where it may run, what it holds."""

    holder: str
    task_id: Optional[str] = None
    #: The card this worker consumes jobs for (``MISTUDIO_WORKER_GPU_UUID``);
    #: None for the general worker and the API process.
    worker_uuid: Optional[str] = None
    #: True: SendTo/WaitForCard raise JobHandoff. False: wait here.
    handoff: bool = True
    #: The broker RESTORED this delivery (kombu sets ``delivery_info['redelivered']``
    #: on a message put back after the visibility timeout or a lost consumer). Only
    #: such a delivery can be a copy of an execution still running: a retry or a
    #: hand-off is a fresh publish, and may legitimately start while the attempt
    #: before it is still releasing its lease. See :meth:`_refuse_duplicate`.
    redelivered: bool = False
    wait_timeout_s: float = DEFAULT_IN_PLACE_TIMEOUT_S
    #: Defaults resolve module globals AT CALL TIME, so a test can substitute
    #: the session, the inventory and host memory for every claim a wrapped
    #: task makes.
    session: Optional[Callable[[], Any]] = None
    inventory: Optional[Callable[[], list]] = None
    available_mb: Optional[Callable[[], float]] = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    renew_interval_s: Optional[float] = None
    on_renew: Optional[Callable[[int], None]] = None
    #: Frees the job's memory on the cards it holds (given their UUIDs), run by
    #: :meth:`close` BEFORE the leases go. See :meth:`_release_memory`.
    release_memory: Optional[Callable[[tuple], None]] = None
    release_timeout_s: float = RELEASE_MEMORY_TIMEOUT_S
    held: tuple = ()
    #: The host RAM reserve this execution recorded on its lease rows and has NOT yet
    #: allocated (:func:`host_reserve_allocated` zeroes it). A lease the renewer takes
    #: back is a new row and records this, not NULL — which the guard would read as a
    #: row from before the column and count the kind's whole reserve again.
    host_reserve_mb: Optional[int] = None
    _renewer: Optional[LeaseRenewer] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = lambda: _sync_session()
        if self.inventory is None:
            self.inventory = lambda: list_cards()
        if self.available_mb is None:
            self.available_mb = lambda: host_available_mb()
        if self.renew_interval_s is None:
            self.renew_interval_s = RENEW_INTERVAL_S

    # ── deciding ─────────────────────────────────────────────────────────

    def _others(self, leases: dict) -> set:
        return {normalise_uuid(uuid) for uuid, who in leases.items() if who != self.holder}

    def _return_queue(self, requested: GpuRequest) -> str:
        from .gpu_dispatch import gpu_queue_for

        return gpu_queue_for(requested)

    def decide(self, requested, required_mb, allow_shard, inventory, leases):
        """What this execution should do, as a gpu_claim decision. Pure over its inputs."""
        if not inventory:
            return Refuse(GpuPlacementError(
                "No GPU is visible to this process.", requested=requested,
                required_mb=required_mb, cards=inventory,
            ))
        if self.handoff:
            if self.worker_uuid is None:
                # The general worker took a whole-GPU job (a message queued on a
                # pre-Phase-3 queue, or a site that bypassed dispatch): it has no
                # card of its own, so it forwards the job to where one will take it.
                return SendTo(self._return_queue(requested), "this worker has no GPU of its own")
            return decide_claim(
                worker_uuid=self.worker_uuid, requested=requested, required_mb=required_mb,
                allow_shard=allow_shard, cards=inventory, leases=leases, holder=self.holder,
            )
        return self._decide_in_place(requested, required_mb, allow_shard, inventory, leases)

    def _decide_in_place(self, requested, required_mb, allow_shard, inventory, leases):
        """A decision that never hands off: this process can run on any card it leases."""
        busy = self._others(leases)
        if not (is_auto(requested) or is_all(requested)):
            try:
                start = find_card(requested, inventory).uuid
            except GpuPlacementError as exc:
                return Refuse(exc)
        elif self.worker_uuid and any(
            normalise_uuid(card.uuid) == normalise_uuid(self.worker_uuid) for card in inventory
        ):
            start = self.worker_uuid
        else:
            pool = [card for card in inventory if normalise_uuid(card.uuid) not in busy] or inventory
            start = max(pool, key=lambda card: (card.free_mb, -card.index)).uuid
        tried = set()
        for _ in range(len(inventory) + 1):
            decision = decide_claim(
                worker_uuid=start, requested=requested, required_mb=required_mb,
                allow_shard=allow_shard, cards=inventory, leases=leases, holder=self.holder,
            )
            if not isinstance(decision, SendTo):
                return decision
            # "Another card is better": from here, that card is simply the one to lease.
            tried.add(normalise_uuid(start))
            target = next((card.uuid for card in inventory if queue_for(card.uuid) == decision.queue), None)
            if target is None or normalise_uuid(target) in tried:
                break
            start = target
        return WaitForCard("no settled choice of GPU yet")

    # ── claiming ─────────────────────────────────────────────────────────

    def precheck(self, requested: GpuRequest, can_split: bool = True) -> None:
        """At task start, before the job does anything: hand it on if this worker should not take it.

        No size is known yet, so only what holds for any size is acted on: a job
        that names another card, a worker whose card another job holds, a busy
        node. RunHere and Refuse fall through — the placement decides with the
        job's size, and a refusal is recorded by the job's own failure handling.

        ``can_split`` is the job's own answer. With True for every job, an ``"all"``
        request for a job that can never split was PARKED whenever a card was
        busy, and refused only once every card was free. With the job's answer it
        is a Refuse here, which falls through to the placement's refusal.
        """
        # A REDELIVERED COPY OF A JOB STILL RUNNING is dropped before it does
        # anything (review round 1, R1-7). See DuplicateExecution.
        self._refuse_duplicate()
        inventory = list(self.inventory())
        with self.session() as db:
            leases = gpu_leases.live_leases(db)
        decision = self.decide(requested, None, bool(can_split), inventory, leases)
        if isinstance(decision, SendTo):
            raise JobHandoff(decision.queue, decision.reason)
        if isinstance(decision, WaitForCard):
            raise JobHandoff(self._return_queue(requested), decision.reason, delay_s=HANDOFF_WAIT_S)

    def claim(
        self,
        requested: GpuRequest,
        *,
        required_mb: Optional[float] = None,
        cards: Optional[Iterable[GpuCard]] = None,
        allow_shard: bool = False,
    ) -> tuple:
        """Lease the cards this job runs on and return them. See the module docstring."""
        if self.held:
            return self._within_held(requested, required_mb, cards, allow_shard)
        # A task that waits in place is never prechecked: its copy is caught here.
        self._refuse_duplicate()
        deadline = self.clock() + self.wait_timeout_s
        races = 0
        while True:
            inventory = list(cards) if cards is not None else list(self.inventory())
            with self.session() as db:
                leases = gpu_leases.live_leases(db)
            decision = self.decide(requested, required_mb, allow_shard, inventory, leases)

            if isinstance(decision, RunHere):
                # JOBS only: the GPU supervisor's "card unavailable" lease on a card
                # whose worker is down (gpu_supervisor.CardAvailability) loads nothing.
                others = {who for who in leases.values()
                          if who != self.holder and holder_kind(who) not in NOT_A_JOB_HOLDER_KINDS}
                ram = host_ram_refusal(
                    required_mb, others_hold_leases=bool(others),
                    available_mb=self.available_mb(),
                    # Once per job, however many cards a split holds — and only
                    # what each has NOT yet allocated (review round 1, R1-6).
                    other_reserves_mb=self._other_reserves_mb(others),
                    own_reserve_mb=HOST_RESERVE_MB_BY_KIND.get(holder_kind(self.holder), 0),
                )
                if ram is not None:
                    decision = WaitForCard(ram)
                else:
                    uuids = [card.uuid for card in decision.cards]
                    if self._take(uuids):
                        logger.info(
                            "GPU job %s leased %s (requested %r, ~%s MB)", self.holder,
                            " + ".join(card.describe() for card in decision.cards), requested,
                            "unknown" if required_mb is None else f"{required_mb:,.0f}",
                        )
                        return tuple(decision.cards)
                    races += 1
                    if races < MAX_ACQUIRE_RACES:
                        continue
                    decision = WaitForCard("a card was leased by another job while this one claimed it")

            if isinstance(decision, Refuse):
                raise decision.error
            if isinstance(decision, SendTo):
                raise JobHandoff(decision.queue, decision.reason)
            self._wait_or_hand_off(requested, required_mb, inventory, decision.reason, deadline)

    def _refuse_duplicate(self) -> None:
        """Raise :class:`DuplicateExecution` when another execution of this task id is still running.

        Called only before this execution holds anything, so every holder found
        is another execution. A lease whose heartbeat went stale belongs to an
        execution that died, and a redelivery after a crash still runs.
        """
        # ONLY A RESTORED DELIVERY. A retry (`self.retry`, `autoretry_for`) is published
        # while the failing attempt still holds its lease — for up to its memory
        # release — and must not be mistaken for a copy and dropped.
        if not self.task_id or not self.redelivered:
            return
        with self.session() as db:
            running = gpu_leases.live_holders_for_task(db, self.task_id, fresh_after_s=STALE_HEARTBEAT_S)
        if running:
            raise DuplicateExecution(self.task_id, running)

    def claim_exact(self, cards: Iterable[GpuCard], requested: GpuRequest = None) -> None:
        """Lease exactly these cards — a model a previous task left loaded on them."""
        cards = list(cards)
        uuids = [card.uuid for card in cards]
        if self.held:
            if {normalise_uuid(u) for u in uuids} <= {normalise_uuid(u) for u in self.held}:
                return
            raise GpuPlacementError(
                f"This job already holds {list(self.held)} and cannot also use {uuids}.",
                requested=requested, cards=cards,
            )
        self._refuse_duplicate()
        deadline = self.clock() + self.wait_timeout_s
        while True:
            with self.session() as db:
                leases = gpu_leases.live_leases(db)
            busy = self._others(leases) & {normalise_uuid(u) for u in uuids}
            if not busy and self._take(uuids):
                return
            reason = (f"{sorted(busy)} held by another job" if busy
                      else "a card was leased by another job while this one claimed it")
            self._wait_or_hand_off(requested, None, cards, reason, deadline)

    def _other_reserves_mb(self, others: set) -> float:
        """Host RAM the other live jobs reserved and have NOT yet allocated — once per job.

        A job records its kind's reserve on its lease rows and zeroes it once
        allocated (:func:`host_reserve_allocated`): from then on MemAvailable
        already excludes it, and subtracting it again counted it twice. A row
        from before the column counts its kind's reserve, as before.
        """
        if not others:
            return 0.0
        with self.session() as db:
            recorded = gpu_leases.host_reserves(db)
        return float(sum(
            HOST_RESERVE_MB_BY_KIND.get(holder_kind(who), 0) if recorded.get(who) is None else recorded[who]
            for who in others
        ))

    def _take(self, uuids: list) -> bool:
        reserve = HOST_RESERVE_MB_BY_KIND.get(holder_kind(self.holder), 0)
        with self.session() as db:
            taken = gpu_leases.acquire(
                db, uuids, self.holder, task_id=self.task_id, host_reserve_mb=reserve,
            )
        if taken:
            self.held = tuple(uuids)
            self.host_reserve_mb = reserve
            self._start_renewal()
            self._waiting(False)
        return taken

    def _waiting(self, waiting: bool) -> None:
        """Tell the janitors this job is (or is no longer) waiting for a GPU. Per-card mode only."""
        from .gpu_dispatch import per_card_workers

        if not self.task_id or not per_card_workers():
            return
        from . import gpu_worker_queues

        if waiting:
            gpu_worker_queues.mark_waiting(self.task_id)
        else:
            gpu_worker_queues.clear_waiting(self.task_id)

    def _wait_or_hand_off(self, requested, required_mb, inventory, reason, deadline) -> None:
        if self.handoff:
            raise JobHandoff(self._return_queue(requested), reason, delay_s=HANDOFF_WAIT_S)
        if self.clock() >= deadline:
            raise GpuPlacementError(
                f"Waited {self.wait_timeout_s / 60:.0f} min for a GPU and none became free "
                f"({reason}). Start the job again when a GPU is free.",
                requested=requested, required_mb=required_mb, cards=list(inventory or ()),
            )
        logger.info("GPU job %s waiting for a GPU: %s", self.holder, reason)
        # A job waiting in place sends no other heartbeat; without this mark a
        # janitor reaps it after ten minutes of waiting (cleanup_orphaned_tasks).
        self._waiting(True)
        self.sleep(IN_PLACE_POLL_S)

    def _within_held(self, requested, required_mb, cards, allow_shard) -> tuple:
        """A SECOND placement in one execution chooses only among the cards it already holds."""
        inventory = list(cards) if cards is not None else list(self.inventory())
        held = {normalise_uuid(uuid) for uuid in self.held}
        mine = [card for card in inventory if normalise_uuid(card.uuid) in held]
        if len(mine) != len(held):
            raise GpuPlacementError(
                f"A GPU this job leased ({sorted(held)}) is no longer visible.",
                requested=requested, required_mb=required_mb, cards=inventory,
            )
        return resolve_cards(requested, required_mb=required_mb, cards=mine, allow_shard=allow_shard)

    # ── lifetime ─────────────────────────────────────────────────────────

    def _start_renewal(self) -> None:
        if self._renewer is None:
            self._renewer = LeaseRenewer(
                self.holder, self.session, interval_s=self.renew_interval_s, on_renew=self.on_renew,
                uuids=self.held, task_id=self.task_id, reserve_mb=lambda: self.host_reserve_mb,
            )
            self._renewer.start()

    def lease_lost(self) -> Optional[str]:
        """Why this execution can no longer prove it holds its cards, or None.

        Lost when its renewer gave the lease up (see :class:`LeaseRenewer`), or
        when the renewer thread is gone while the job still holds cards —
        nothing is renewing, so the lease will lapse unseen.
        """
        renewer = self._renewer
        if renewer is None or not self.held:
            return None
        if renewer.lost is not None:
            return renewer.lost
        if not renewer.stopped and not renewer.is_alive():
            return f"the renewal of its lease on {list(self.held)} stopped while the job still runs"
        return None

    def _release_memory(self, held: tuple) -> None:
        """Free this job's memory on its cards BEFORE they can be offered to another job.

        A lease released while a finished job's buffer and cached blocks are
        still allocated hands the next placement a card that reads fuller than
        it is: an Auto job goes elsewhere or waits, and a job NAMING the card is
        refused outright. So the release runs first, on a thread bounded by
        ``release_timeout_s``; past that the lease is released anyway, with a
        warning. A release that raises is logged. Never raises, never skips the
        lease release that follows it.
        """
        # EVEN WITH NOTHING HELD: a job that hands off or fails before placing
        # still frees the idle copies this process keeps (a J-lens readout's kept
        # model, the logit-lens cache — 3dac5fb8), which sit on a card no lease
        # covers. The release itself decides what needs held cards.
        release = self.release_memory
        if release is None:
            return
        done = threading.Event()

        def run() -> None:
            try:
                release(held)
            except BaseException:  # noqa: BLE001 - the lease release must follow regardless
                logger.exception("GPU job %s could not release its memory on %s", self.holder, list(held))
            finally:
                done.set()

        try:
            threading.Thread(target=run, daemon=True, name=f"gpu-release-{self.holder[:24]}").start()
            if not done.wait(self.release_timeout_s):
                logger.warning(
                    "GPU job %s had not released its memory on %s after %.0f s; releasing its "
                    "lease anyway, so the next job may find the card fuller than it will be",
                    self.holder, list(held), self.release_timeout_s,
                )
        except Exception:  # noqa: BLE001 - e.g. a thread that could not start
            logger.exception("GPU job %s: memory release did not run", self.holder)

    def close(self) -> None:
        """Release the job's memory, stop renewing and release every lease it holds. Never raises."""
        try:
            self._release_memory(self.held)
        finally:
            if self._renewer is not None:
                self._renewer.stop()
                self._renewer = None
            try:
                with self.session() as db:
                    gpu_leases.release(db, self.holder)
            except Exception:  # noqa: BLE001 - the lease expires at its TTL anyway
                logger.exception("Could not release the GPU leases of %s; they expire at their TTL", self.holder)
            self.held = ()


#: The claim of the job running in this process. A module global rather than a
#: contextvar: every worker here is solo, so there is exactly one job per
#: process, and code the job runs in a thread pool (which does not copy
#: contextvars) must still find it.
_ACTIVE: list = []
_ACTIVE_LOCK = threading.Lock()

#: The claim of GPU work running in THIS asyncio task of the API process
#: (``logit_lens_service.LogitLensLease``). The API process serves many requests at
#: once, so it never uses the process-wide claim above; a context variable follows
#: the request (and the threads ``asyncio.to_thread`` starts for it). Without it a
#: lease the API process lost was never read by anything (review round 2).
_TASK_CLAIM: contextvars.ContextVar = contextvars.ContextVar("gpu_task_claim", default=None)


def current_claim() -> Optional[ClaimContext]:
    with _ACTIVE_LOCK:
        return _ACTIVE[-1] if _ACTIVE else None


def lease_lost_reason() -> Optional[str]:
    """Why the job running in this process can no longer prove it holds its GPU lease, or None.

    Read by the cooperative cancellation checks (``core.cancellation.CancelCheck``
    and training's status check): a job that has lost its lease stops there. None
    outside a claim — the API process, single mode, a job before it places.
    """
    ctx = current_claim()
    if ctx is None:
        ctx = _TASK_CLAIM.get()
    return ctx.lease_lost() if ctx is not None else None


def raise_if_lease_lost(doing: str) -> None:
    """Raise :class:`GpuLeaseLost` when the running GPU work has lost its lease. For a loop's
    boundary; a no-op outside a claim (single mode, the API process without a lens lease)."""
    lost = lease_lost_reason()
    if lost:
        raise GpuLeaseLost(f"Stopped while {doing}: {lost}. Run it again.")


def host_reserve_allocated() -> None:
    """The running job has now allocated the host memory its kind reserves (review round 1, R1-6).

    Called by a cached-activation training once its rolling buffer and pinned
    pool exist: MemAvailable already excludes that memory from then on, so
    another job's host RAM guard must stop subtracting the reserve on top of it.
    Zeroes the reserve on this execution's lease rows. A no-op outside a claim
    (single mode, the API process). Never raises: the worst outcome of a failed
    write is the old, conservative double count.
    """
    ctx = current_claim()
    if ctx is None or not ctx.held:
        return
    # Before the write: a take-back racing it records the truth.
    ctx.host_reserve_mb = 0
    try:
        with ctx.session() as db:
            gpu_leases.set_host_reserve(db, ctx.holder, 0)
    except Exception:  # noqa: BLE001 - the guard stays conservative
        logger.warning("Could not record that %s allocated its host reserve", ctx.holder, exc_info=True)


@contextlib.contextmanager
def claiming(ctx: ClaimContext):
    """Make ``ctx`` this process's claim for the duration; release its leases on EVERY exit."""
    with _ACTIVE_LOCK:
        _ACTIVE.append(ctx)
    try:
        yield ctx
    finally:
        with _ACTIVE_LOCK:
            if ctx in _ACTIVE:
                _ACTIVE.remove(ctx)
        ctx.close()


def claim_cards(
    requested: GpuRequest,
    *,
    required_mb: Optional[float] = None,
    cards: Optional[Iterable[GpuCard]] = None,
    allow_shard: bool = False,
) -> tuple:
    """The cards ``place_job`` places on: leased by the running job, or resolved as in Phase 2."""
    from .gpu_dispatch import per_card_workers

    ctx = current_claim()
    if ctx is None:
        if per_card_workers():
            raise GpuPlacementError(UNCLAIMED_MESSAGE, requested=requested, required_mb=required_mb)
        return resolve_cards(requested, required_mb=required_mb, cards=cards, allow_shard=allow_shard)
    return ctx.claim(requested, required_mb=required_mb, cards=cards, allow_shard=allow_shard)


def cards_held_by_other_jobs(cards: Iterable[GpuCard]) -> list:
    """The cards among ``cards`` another execution holds a live lease on. Reads only.

    Empty with no claim in progress (single mode, the API process): there are no
    leases to consult. For a caller that has a cheaper answer than waiting for a
    card — a kept model that can be released and placed again.
    """
    ctx = current_claim()
    if ctx is None:
        return []
    with ctx.session() as db:
        others = ctx._others(gpu_leases.live_leases(db))
    return [card for card in cards if normalise_uuid(card.uuid) in others]


def claim_exact_cards(cards: Iterable[GpuCard], requested: GpuRequest = None) -> None:
    """Lease exactly ``cards`` for the running job (a resident model). No-op in single mode."""
    from .gpu_dispatch import per_card_workers

    ctx = current_claim()
    if ctx is None:
        if per_card_workers():
            raise GpuPlacementError(UNCLAIMED_MESSAGE, requested=requested)
        return
    ctx.claim_exact(cards, requested)
