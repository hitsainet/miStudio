"""One circuit GPU task per GPU, not one on the whole node (multi-GPU Phase 3).

Before Phase 3 every circuit GPU endpoint (capture, attribution, validation,
faithfulness, calibration, the steered transcript recorder) refused a new run
while ANY circuit GPU run was pending or running — "one GPU circuit task at a
time on the single 3090" — under one global advisory lock. With a worker per
card, two circuit runs on different cards can run at once; on the same card they
may not. In per-card mode ``CircuitCaptureService.assert_no_active_gpu_run``
calls :func:`assert_card_free_for_circuit` instead.

THE LOCK KEY IS PER CARD. The check-then-mark transaction takes
``pg_advisory_xact_lock`` on the key of every card the request could use, in
sorted order (so two transactions cannot deadlock): a named card's key, or every
card's for Auto and ``"all"``. Two submissions naming different cards therefore
do not wait on each other; anything that could land on the same card does.

WHAT A RUN IS BOUND TO. Captures and recordings store their request and, once
placed, their card(s); attribution, validation, faithfulness and calibration
carry the request in the task message only, so their card is unknown here. The
rule is built so an unknown binding can only make the check stricter:

* ``"all"`` needs every card: refused while any circuit run is active;
* a run bound to ``"all"`` blocks everything;
* a named card is refused while a run is known to be on it or to name it;
* every request is refused once there are as many active circuit runs as cards —
  each card may already have one, wherever the unknown ones landed.

The GPU leases remain what actually keeps two jobs off one card at run time;
this guard only keeps circuit runs from queueing more deeply than the cards can
take them, as its single-GPU predecessor did.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional

from .gpu_placement import GpuCard, GpuRequest, is_all, is_auto, list_cards, normalise_uuid

#: The pre-Phase-3 global key, kept for a node with no visible GPU.
GLOBAL_LOCK_KEY = 0x1C1C_C0DE


def lock_key(card_uuid: str) -> int:
    """A stable bigint advisory-lock key for one card (60 bits of a SHA-256)."""
    digest = hashlib.sha256(f"mistudio-circuit-gpu:{normalise_uuid(card_uuid)}".encode()).hexdigest()
    return int(digest[:15], 16)


def keys_for(gpu_request: GpuRequest, cards: Iterable[GpuCard]) -> list:
    """The lock keys a circuit request takes, sorted."""
    cards = list(cards)
    if not cards:
        return [GLOBAL_LOCK_KEY]
    if not (is_auto(gpu_request) or is_all(gpu_request)):
        return [lock_key(str(gpu_request))]
    return sorted(lock_key(card.uuid) for card in cards)


@dataclass(frozen=True)
class ActiveRun:
    """A pending or running circuit GPU run, and what is known of its card."""

    description: str
    #: Card UUIDs it is placed on or names; empty when unknown.
    cards: frozenset = frozenset()
    #: It asked for every card.
    whole_node: bool = False


def binding(gpu_request, gpu_uuid=None, gpu_uuids=None) -> tuple:
    """(cards, whole_node) from a run row's GPU columns."""
    if gpu_uuids:
        return frozenset(normalise_uuid(u) for u in gpu_uuids), False
    if gpu_uuid:
        return frozenset({normalise_uuid(gpu_uuid)}), False
    if is_all(gpu_request):
        return frozenset(), True
    if gpu_request and not is_auto(gpu_request):
        return frozenset({normalise_uuid(gpu_request)}), False
    return frozenset(), False


def conflict(gpu_request: GpuRequest, active: list, card_count: int) -> Optional[str]:
    """Why a new circuit run with ``gpu_request`` must wait, or None. Pure."""
    if not active:
        return None
    if card_count <= 0 or is_all(gpu_request):
        return f"{active[0].description} — a circuit run that needs every GPU waits until none is active"
    for run in active:
        if run.whole_node:
            return f"{run.description} across every GPU — wait or cancel it first"
    if not is_auto(gpu_request):
        wanted = normalise_uuid(str(gpu_request))
        for run in active:
            if wanted in run.cards:
                return f"{run.description} on that GPU — one circuit task per GPU; wait, cancel it, or choose another GPU"
    if len(active) >= card_count:
        return (f"{len(active)} circuit GPU task(s) are active on {card_count} GPU(s) — one per GPU; "
                f"wait or cancel one first ({active[0].description})")
    return None


def active_circuit_runs(db) -> list:
    """Every pending or running circuit GPU run, with its known card binding."""
    from ..models.circuit import Circuit
    from ..models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
    from ..models.steering_record_run import SteeringRecordRun

    runs = []
    for row in db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.status.in_(("pending", "estimating", "running"))).all():
        cards, whole = binding(row.gpu_request, getattr(row, "gpu_uuid", None), getattr(row, "gpu_uuids", None))
        runs.append(ActiveRun(f"Capture {row.id} is already {row.status}", cards, whole))
    for row in db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.attribution_status.in_(("pending", "running"))).all():
        runs.append(ActiveRun(f"Attribution pass on {row.id} is {row.attribution_status}"))
    for row in db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.validation_status.in_(("pending", "running"))).all():
        runs.append(ActiveRun(f"Validation pass on {row.id} is {row.validation_status}"))
    for row in db.query(Circuit).filter(Circuit.faithfulness_status.in_(("pending", "running"))).all():
        runs.append(ActiveRun(f"Faithfulness pass on {row.id} is {row.faithfulness_status}"))
    for row in db.query(Circuit).filter(Circuit.calibration_status.in_(("pending", "running"))).all():
        runs.append(ActiveRun(f"Calibration pass on {row.id} is {row.calibration_status}"))
    for row in db.query(SteeringRecordRun).filter(SteeringRecordRun.status.in_(("pending", "running"))).all():
        cards, whole = binding(row.gpu_request, getattr(row, "gpu_uuid", None), getattr(row, "gpu_uuids", None))
        runs.append(ActiveRun(f"Steering record {row.id} is {row.status}", cards, whole))
    return runs


def assert_card_free_for_circuit(db, gpu_request: GpuRequest, cards: Optional[Iterable[GpuCard]] = None) -> None:
    """Take the per-card lock(s) for ``gpu_request`` and refuse when its card(s) are taken.

    Raises:
        CaptureConflictError: with the reason, as the single-GPU guard did.
    """
    from sqlalchemy import text

    from .circuit_capture_service import CaptureConflictError

    cards = list_cards() if cards is None else list(cards)
    for key in keys_for(gpu_request, cards):
        db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": key})
    reason = conflict(gpu_request, active_circuit_runs(db), len(cards))
    if reason is not None:
        raise CaptureConflictError(reason)
