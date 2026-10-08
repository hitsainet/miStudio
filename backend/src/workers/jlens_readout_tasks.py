"""
Readout as a background task.

WHY THIS EXISTS, measured rather than assumed. The readout was bound
synchronously and 502'd at the ingress twice on a real model:

    POST /jlens/readout (gemma-2-2b-it, CPU)  ->  502 after 64.9s
    POST /jlens/readout (retry, 2 layers)     ->  502 after 54.0s

nginx gives up at 60s. The work itself is fine — a J-space readout needs the
whole model resident for its forward pass, and loading gemma-2-2b takes about a
minute on CPU. Raising the proxy timeout would be a bandaid: readout cost is
O(positions x layers x top_n) on top of the load, so no fixed timeout bounds it.

So the readout follows the pattern every other model-bound operation in this
codebase already uses — steering, extraction, calibration all queue and poll.
The API returns a task id immediately and the worker does the load.

THE WORKER IS THE RIGHT PLACE TO LOAD, and the API is not. A Celery worker is
a separate process: a model loaded inside the API process cannot help the
worker, and one loaded inside the worker cannot help a synchronous API handler.

ON THE CARD THE CALLER CHOSE (0xcc/plans/Multi-GPU-Plan.md, Phase 1). Capture
used to take "any resident copy, else the CPU". It now runs on the card
`gpu_request` names, or on the one Auto picks against the memory free when the
task starts, and the card is written to the task's row before the load. The
readout maths still runs on READOUT_DEVICE.

LOADED OR UNLOADED BETWEEN READOUTS IS THE CALLER'S CHOICE (`unload_after`,
user decision 2026-09-13). A copy that is freed makes the next readout pay a
full load; a copy that is kept sits on a card miLLM may serve from until the
next job placed in this worker frees it (`gpu_placement.place_job` runs the
registry's idle release before reading the cards) — the defect the fit task
records, where an LFM2 fit left 4.0 GB resident at 0% utilisation. The default keeps it; the panel's
"Unload model after each readout" frees it. A readout that FAILS always frees
it, so a failure never strands a model on a card.

A KEPT COPY IS REUSED, NOT JOINED. Auto judges the cards by the memory free now,
and the card holding the copy has less of it, so a second Auto readout would
otherwise load another copy on the other card. `plan_readout` decides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..core.celery_app import celery_app
from ..services.gpu_placement import AUTO, is_all, is_auto, normalise_uuid
from . import jlens_progress
from .gpu_job import gpu_job
from .task_heartbeat import beat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadoutPlan:
    """What a readout does about a copy of its model already loaded in this worker."""

    #: Run on the resident copy's card; no placement is made.
    reuse: bool
    #: Free the resident copy BEFORE placing, so two GPU copies never coexist
    #: and the placement sees the memory that copy held.
    release_first: bool


def plan_readout(
    resident_device: Optional[str],
    resident_card: Any,
    gpu_request: Optional[str],
    cards: Optional[Sequence[Any]] = None,
) -> ReadoutPlan:
    """Reuse a resident copy, or place afresh — and whether to free it first.

    `resident_device` is the registry's device spec ("cpu", "cuda:1",
    "cuda:0+cuda:1"). `resident_card` is the card that copy is on, or for a
    SPLIT copy the tuple of its cards, one per device; None (or a None among
    them) when NVML cannot name a card. `cards` is the node's inventory, needed
    only to judge "all".

    * Nothing resident, or a CPU copy: place. A CPU copy holds no card, and the
      single-entry cache replaces it when the placed copy loads.
    * A GPU copy on a card that cannot be named: free it, then place.
    * Auto: reuse the copy, split or not.
    * A named card: reuse only a single-card copy on THAT card. A split is not
      that card even when the card is one of its cards — a named card is
      honoured or refused, never swapped for another choice.
    * "all": reuse only a copy that already occupies every card the node has.
    * Anything else: free the copy, then place.
    """
    from ..services.jlens_model_registry import parse_device_spec

    devices = parse_device_spec(resident_device)
    if not devices or not all(_on_gpu(name) for name in devices):
        return ReadoutPlan(reuse=False, release_first=False)
    held = tuple(resident_card) if isinstance(resident_card, (tuple, list)) else (resident_card,)
    if len(held) != len(devices) or any(card is None for card in held):
        return ReadoutPlan(reuse=False, release_first=True)

    reuse = ReadoutPlan(reuse=True, release_first=False)
    release = ReadoutPlan(reuse=False, release_first=True)
    held_uuids = {normalise_uuid(card.uuid) for card in held}
    if is_auto(gpu_request):
        return reuse
    if is_all(gpu_request):
        every = {normalise_uuid(card.uuid) for card in (cards or ())}
        return reuse if every and held_uuids == every else release
    if len(held) == 1 and held_uuids == {normalise_uuid(gpu_request)}:
        return reuse
    return release


def _on_gpu(name: str) -> bool:
    import torch

    try:
        return torch.device(name).type == "cuda"
    except (RuntimeError, TypeError, ValueError):
        return False


@celery_app.task(
    name="src.workers.jlens_readout_tasks.compute_readout",
    bind=True,
    max_retries=0,
)
@gpu_job("jlens_readout")
@jlens_progress.owns_its_failure
def compute_readout(
    self,
    model_id: str,
    prompt: str,
    types: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    top_n: int = 8,
    artifact_id: Optional[str] = None,
    gpu_request: str = AUTO,
    unload_after: bool = False,
) -> Dict[str, Any]:
    """Produce a wire-format readout. Returns the serialised meta + tokens.

    `gpu_request` is what the endpoint resolved: "auto", or the UUID of the card
    the caller named. A card that cannot be used fails the task with the reason.

    `unload_after` False (the default) leaves the model loaded on its card for
    the next readout; True frees the card when this readout finishes. A readout
    that fails frees the card either way.

    `max_retries=0`: a readout that failed for a real reason (model missing,
    artifact unvalidated, request too large) fails the same way on a retry, and
    retrying a minute-long model load on a shared box makes things worse.
    """
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services import jlens_model_registry as registry

    requested = types or ["LOGIT_LENS"]

    # EVERY NAME EXISTS BEFORE ANYTHING CAN RAISE, so the `finally` can always
    # run — and the load is inside it, because a CUDA OOM part-way through
    # `from_pretrained` leaves allocations on the card.
    placement = None
    loaded = None
    keep_loaded = False
    try:
        with get_sync_db() as db:
            record = db.query(Model).filter(Model.id == model_id).first()
            if record is None:
                raise ValueError(f"No model with id {model_id!r}")

            self.update_state(state="PROGRESS", meta=beat({"stage": "loading_model"}))
            jlens_progress.update_row(self.request.id, status="running", progress=1.0)
            placement = _place_readout(self.request.id, record, gpu_request)
            try:
                loaded = registry.load_for_readout(record, placement=placement)
            except registry.ModelNotAvailable as exc:
                # Surfaced as the task's failure message rather than a retry: the
                # model is not in a state a readout can use and waiting will not
                # change that.
                raise ValueError(str(exc)) from exc

        result = _read_out(
            self,
            loaded=loaded,
            prompt=prompt,
            requested=requested,
            layers=layers,
            top_n=top_n,
            artifact_id=artifact_id,
        )
        # Only a readout that SUCCEEDED may leave its model on the card.
        keep_loaded = not unload_after
    finally:
        loaded = None  # noqa: F841 - the assignment IS the release
        if not keep_loaded:
            jlens_progress.release_card(placement)

    jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
    return result


def _place_readout(task_id: Optional[str], record: Any, gpu_request: Optional[str]):
    """The card(s) this readout runs on: a resident copy's, or a fresh placement.

    SPLIT-SAFE, SO IT MAY SPLIT (`allow_shard=True`). Capture hooks copy every
    residual to READOUT_DEVICE as it is produced, the final norm is borrowed on
    its own card, input ids go to the embedding's card, and the release empties
    every card the copy holds.
    """
    import torch

    from ..services import jlens_model_registry as registry
    from ..services.gpu_placement import card_for_device, list_cards

    resident = registry.resident_device_for(record)
    devices = registry.parse_device_spec(resident)
    held = ()
    inventory = None
    if devices and all(_on_gpu(name) for name in devices):
        # ONE inventory read for every card of the copy and for "all".
        inventory = list_cards()
        held = tuple(card_for_device(name, cards=inventory) for name in devices)
    plan = plan_readout(resident, held or None, gpu_request, cards=inventory)
    if plan.reuse:
        return jlens_progress.reuse_card(
            task_id,
            held if len(held) > 1 else held[0],
            tuple(torch.device(name) for name in devices) if len(held) > 1 else torch.device(devices[0]),
            gpu_request,
        )
    if plan.release_first:
        logger.info(
            "Freeing the copy of %s on %s before placing readout %s (requested %r)",
            getattr(record, "id", "?"),
            resident,
            task_id,
            gpu_request,
        )
        registry.clear_cache()
    return jlens_progress.place_on_card(
        task_id,
        gpu_request,
        required_mb=registry.estimate_weights_mb(record),
        allow_shard=True,
    )


def _read_out(
    self,
    loaded,
    prompt: str,
    requested: List[str],
    layers: Optional[List[int]],
    top_n: int,
    artifact_id: Optional[str],
) -> Dict[str, Any]:
    """The readout itself, split out so the task can guarantee the release.

    The `ReadoutService` built here holds `model=loaded.model`. Its frame is gone
    by the time the caller's `finally` runs on success; on a failure the
    traceback keeps it, and `release_card` clears those frames.
    """
    from ..services.jlens_readout_service import IdentityTransport, ReadoutService
    from ..schemas.jlens import LensMetaMessage, LensTokenMessage

    transports = []
    for lens_type in requested:
        if lens_type == "LOGIT_LENS":
            transports.append(IdentityTransport())
            continue
        # Imported here so the API's validation logic stays the single
        # definition of what a serviceable artifact is.
        from ..api.v1.endpoints.jlens import _jacobian_transport

        transports.append(_jacobian_transport(loaded, artifact_id))

    self.update_state(state="PROGRESS", meta=beat({"stage": "reading_out"}))
    service = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )

    meta = None
    tokens = []
    for message in service.stream(prompt, transports, layers=layers, top_n=top_n):
        if isinstance(message, LensMetaMessage):
            meta = message.model_dump()
        elif isinstance(message, LensTokenMessage):
            tokens.append(message.model_dump())

    if meta is None:
        # Never returned as an empty success: an empty readout is
        # indistinguishable from a real one with no content.
        raise ValueError("readout produced no meta message")

    return {"meta": meta, "tokens": tokens}
