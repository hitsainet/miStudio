"""Which GPU a job runs on — the one module in miStudio that chooses a GPU.

The node gained a second card on 2026-09-13: an RTX 3080 Ti took NVML index 0
and the RTX 3090 moved to index 1. Every GPU job except activation extraction
had been written for one card — bare ``"cuda"``, ``mem_get_info(0)``, a steering
worker pinned with ``CUDA_VISIBLE_DEVICES="0"`` — so all of it quietly moved to
the smaller card. This module replaces those choices.

* A card's identity is its UUID. An NVML index is only a position, and adding a
  card changed it. Job rows store the UUID.
* A job asks for ``"auto"`` (the default), an index, or a UUID. Auto picks the
  card with the most free memory that fits. An explicit card without room is
  refused with the figures — never swapped for another card.
* Free memory is read live from NVML, which counts every process on the card,
  miLLM's included. Nothing is reserved for anyone.
* NVML needs no CUDA context, so the API process can resolve a request without
  holding memory on every card. A worker turns the chosen UUID into a torch
  device with :func:`torch_device`, which matches by UUID and so stays right
  under ``CUDA_VISIBLE_DEVICES`` or a renumbering.

Plan: ``0xcc/plans/Multi-GPU-Plan.md``, Phase 1.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

AUTO = "auto"

#: Split one model across every visible card. An explicit choice, like naming a
#: card: honoured by a job that can run split, refused by one that cannot.
ALL = "all"

#: Memory kept free on each card of a split, beyond the model's share there: the
#: CUDA context, the activations of the layers that land on that card, and
#: allocator slack. A split fills every card it uses; a single-card job keeps
#: its whole headroom on one device, so a split needs a reserve per card.
SHARD_RESERVE_MB = 1024

#: What a caller may ask for: ``None``/``"auto"``, ``"all"``, an NVML index, or a UUID.
GpuRequest = Union[None, int, str]


@dataclass(frozen=True)
class GpuCard:
    """One GPU as NVML reports it at the moment of the query."""

    index: int
    uuid: str
    name: str
    total_mb: int
    free_mb: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        return f"GPU {self.index} ({self.name}, {self.free_mb:,} of {self.total_mb:,} MB free)"


class GpuPlacementError(RuntimeError):
    """No card can take the job as asked. Carries the figures for the caller."""

    def __init__(
        self,
        message: str,
        *,
        requested: GpuRequest = None,
        required_mb: Optional[float] = None,
        cards: Iterable[GpuCard] = (),
    ) -> None:
        super().__init__(message)
        self.requested = requested
        self.required_mb = required_mb
        self.cards = list(cards)

    def details(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "required_mb": None if self.required_mb is None else int(self.required_mb),
            "cards": [card.to_dict() for card in self.cards],
        }


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def normalise_uuid(value: Any) -> str:
    """``GPU-xxxxxxxx-…`` in lower case, whether or not the source had the prefix.

    NVML prints the prefix; ``torch.cuda.get_device_properties(i).uuid`` does not.
    """
    text = _text(value).strip().lower()
    return text if text.startswith("gpu-") else f"gpu-{text}"


def list_cards() -> list[GpuCard]:
    """Every GPU visible to this process, in NVML (PCI bus) order.

    Empty when NVML is not installed or finds no driver — a CPU-only
    development machine is a normal state, not an error.
    """
    try:
        import pynvml
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - no driver is a normal state
        logger.info("No GPU visible to NVML: %s", exc)
        return []
    try:
        cards = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            cards.append(
                GpuCard(
                    index=index,
                    uuid=_text(pynvml.nvmlDeviceGetUUID(handle)),
                    name=_text(pynvml.nvmlDeviceGetName(handle)),
                    total_mb=int(memory.total // (1024 * 1024)),
                    free_mb=int(memory.free // (1024 * 1024)),
                )
            )
        return cards
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown is best effort
            pass


def is_auto(requested: GpuRequest) -> bool:
    return requested is None or (
        isinstance(requested, str) and requested.strip().lower() in ("", AUTO)
    )


def is_all(requested: GpuRequest) -> bool:
    return isinstance(requested, str) and requested.strip().lower() == ALL


def _summary(cards: Iterable[GpuCard]) -> str:
    return "; ".join(card.describe() for card in cards) or "no GPUs"


def find_card(requested: Union[int, str], cards: Iterable[GpuCard]) -> GpuCard:
    """The card an explicit request names, by NVML index or UUID."""
    cards = list(cards)
    if isinstance(requested, bool):
        raise GpuPlacementError(f"Not a GPU: {requested!r}", requested=requested, cards=cards)
    if isinstance(requested, int) or (isinstance(requested, str) and requested.strip().isdigit()):
        index = int(requested)
        match = next((card for card in cards if card.index == index), None)
    else:
        wanted = normalise_uuid(requested)
        match = next((card for card in cards if normalise_uuid(card.uuid) == wanted), None)
    if match is None:
        raise GpuPlacementError(
            f"No GPU {requested!r} on this node. Available: {_summary(cards)}.",
            requested=requested,
            cards=cards,
        )
    return match


def resolve_card(
    requested: GpuRequest = AUTO,
    required_mb: Optional[float] = None,
    cards: Optional[Iterable[GpuCard]] = None,
) -> GpuCard:
    """The card a job should run on.

    Args:
        requested: ``None``/``"auto"``, an NVML index, or a UUID.
        required_mb: The job's estimated need, when known. Without it Auto
            simply takes the card with the most free memory.
        cards: The inventory to choose from; read live from NVML when omitted.

    Raises:
        GpuPlacementError: no GPU is visible, the named card does not exist,
            the named card lacks ``required_mb``, or no single card has it.
    """
    if is_all(requested):
        # A one-card resolver asked to split. Without this, "all" was looked up
        # as a card UUID and refused as "No GPU 'all' on this node" — by a job
        # (the logit lens) whose API had accepted it.
        raise GpuPlacementError(
            "This job cannot run split across GPUs. Choose Auto or one GPU.",
            requested=requested, required_mb=required_mb,
            cards=() if cards is None else cards,
        )
    cards = list_cards() if cards is None else list(cards)
    if not cards:
        raise GpuPlacementError(
            "No GPU is visible to this process.", requested=requested,
            required_mb=required_mb, cards=cards,
        )

    if is_auto(requested):
        fitting = [card for card in cards if required_mb is None or card.free_mb >= required_mb]
        if not fitting:
            raise GpuPlacementError(
                f"No single GPU has the ~{required_mb:,.0f} MB this job needs. "
                f"{_summary(cards)}.",
                requested=requested, required_mb=required_mb, cards=cards,
            )
        # Most free memory wins; the lower index breaks a tie so the choice is
        # deterministic.
        return max(fitting, key=lambda card: (card.free_mb, -card.index))

    card = find_card(requested, cards)
    if required_mb is not None and card.free_mb < required_mb:
        raise GpuPlacementError(
            f"{card.describe()} cannot take this job: it needs ~{required_mb:,.0f} MB. "
            "Choose another GPU or Auto.",
            requested=requested, required_mb=required_mb, cards=cards,
        )
    return card


def _shard_budget_mb(card: GpuCard) -> int:
    """What one card can hold of a split model: its free memory less the per-card reserve."""
    return max(card.free_mb - SHARD_RESERVE_MB, 0)


def _most_free_first(cards: Iterable[GpuCard]) -> list[GpuCard]:
    return sorted(cards, key=lambda card: (-card.free_mb, card.index))


def resolve_cards(
    requested: GpuRequest = AUTO,
    required_mb: Optional[float] = None,
    cards: Optional[Iterable[GpuCard]] = None,
    allow_shard: bool = False,
) -> tuple[GpuCard, ...]:
    """The card a job runs on, or the cards, most free first, when its model must be split.

    Most free first is the order recorded (``gpu_uuids``), NOT the order the
    model fills them: transformers fills a split in CUDA index order, so the
    embedding and the first layers sit on the lowest-index card whichever card
    comes first here (``ml/split_load.py``).

    ONE CARD WHENEVER ONE CARD FITS. A split pays for a cross-card copy at every
    boundary in every forward pass, so it is the fallback, never the preference
    (plan decision D4). Only a job that can run split passes ``allow_shard``; for
    every other job the answer is exactly :func:`resolve_card`'s.

    * ``"all"``: every card, most free first. Refused for a job that cannot run
      split, and when all the cards together lack ``required_mb``.
    * Auto: the most-free card that fits. Failing that, and only with
      ``allow_shard``, the FEWEST cards, most free first, whose budgets (free
      memory less :data:`SHARD_RESERVE_MB` each) cover ``required_mb``. A card
      the split does not need is left free for other work.
    * A named card: :func:`resolve_card` — one card, honoured or refused.

    Raises:
        GpuPlacementError: as :func:`resolve_card`; also when no set of cards
            covers the job, or ``"all"`` is asked of a job that cannot split.
    """
    cards = list_cards() if cards is None else list(cards)

    if is_all(requested):
        if not allow_shard:
            raise GpuPlacementError(
                "This job cannot run split across GPUs. Choose Auto or one GPU.",
                requested=requested, required_mb=required_mb, cards=cards,
            )
        if not cards:
            raise GpuPlacementError(
                "No GPU is visible to this process.", requested=requested,
                required_mb=required_mb, cards=cards,
            )
        ordered = tuple(_most_free_first(cards))
        budget = sum(_shard_budget_mb(card) for card in ordered)
        if required_mb is not None and budget < required_mb:
            raise GpuPlacementError(
                f"All {len(ordered)} GPUs together can hold ~{budget:,} MB of a split model "
                f"({SHARD_RESERVE_MB:,} MB kept free on each); this job needs "
                f"~{required_mb:,.0f} MB. {_summary(cards)}.",
                requested=requested, required_mb=required_mb, cards=cards,
            )
        return ordered

    if not (allow_shard and is_auto(requested)):
        return (resolve_card(requested, required_mb=required_mb, cards=cards),)

    try:
        return (resolve_card(requested, required_mb=required_mb, cards=cards),)
    except GpuPlacementError:
        # Without a size there is nothing to split against, and with fewer
        # than two cards there is nothing to split across.
        if required_mb is None or len(cards) < 2:
            raise

    chosen: list[GpuCard] = []
    budget = 0
    for card in _most_free_first(cards):
        chosen.append(card)
        budget += _shard_budget_mb(card)
        if len(chosen) >= 2 and budget >= required_mb:
            return tuple(chosen)
    raise GpuPlacementError(
        f"No GPU, and no set of GPUs splitting the model between them, has the "
        f"~{required_mb:,.0f} MB this job needs ({SHARD_RESERVE_MB:,} MB kept free on each "
        f"card of a split). {_summary(cards)}.",
        requested=requested, required_mb=required_mb, cards=cards,
    )


def resolve_request(requested: GpuRequest, cards: Optional[Iterable[GpuCard]] = None) -> str:
    """What a job records when it is submitted: ``"auto"``, or the UUID of the named card.

    An index becomes a UUID here, while it still means the card the user was
    shown; a queued job may start after a card is added and the indices shift.
    Free memory is NOT checked at submit — a queued job is judged against the
    memory free when it starts.

    Raises:
        GpuPlacementError: the named card does not exist.
    """
    if is_auto(requested):
        return AUTO
    if is_all(requested):
        # The cards themselves are chosen when the job starts, from the cards
        # there are then — like Auto, and unlike a named card.
        return ALL
    cards = list_cards() if cards is None else list(cards)
    return find_card(requested, cards).uuid


@dataclass(frozen=True)
class Placement:
    """Where a started job runs.

    ``card`` and ``device`` are the job's first card and its torch device (None
    and the CPU on a machine with no CUDA). A SPLIT also carries every card,
    most free first (transformers fills them in CUDA index order), their torch
    devices, and each card's memory budget by torch
    index. A single-card placement leaves those empty, so code written for one
    card reads it exactly as before.
    """

    card: Optional[GpuCard]
    device: "torch.device"
    cards: tuple = ()
    devices: tuple = ()
    max_memory_mb: Optional[dict] = None

    @property
    def is_shard(self) -> bool:
        return len(self.cards) > 1

    @property
    def all_cards(self) -> tuple:
        """Every card the job may use: the split's cards, or its one card."""
        if self.cards:
            return tuple(self.cards)
        return () if self.card is None else (self.card,)

    @property
    def all_devices(self) -> tuple:
        return tuple(self.devices) if self.devices else (self.device,)

    @property
    def uuid(self) -> Optional[str]:
        return None if self.card is None else self.card.uuid

    @property
    def uuids(self) -> list[str]:
        return [card.uuid for card in self.all_cards]

    @property
    def device_map(self) -> str:
        """``device_map`` for ``from_pretrained``: a sequential split, or the one device.

        ``"sequential"``, NOT ``"auto"``. Given any other value, transformers
        (5.15.1, ``integrations/accelerate._get_device_map``) passes ``max_memory``
        through accelerate's ``get_balanced_memory``, which caps every card but
        the last at roughly the model's size divided by the number of cards —
        so a model the per-card budgets can hold gets layers mapped to disk,
        which the loader then refuses. Sequential fills the cards in index order
        up to each card's budget.
        """
        return "sequential" if self.is_shard else str(self.device)

    @property
    def max_memory(self) -> Optional[dict]:
        """``max_memory`` for ``from_pretrained``: each card's budget and NO ``"cpu"`` key.

        A transformers load in miStudio runs on GPUs only (operator decision 3).
        Without a CPU entry accelerate cannot offload a layer to host RAM, but it
        can still map one to "disk", so the loader checks the result and refuses.
        None for a single-card placement, whose ``device_map`` names its device.
        """
        if not self.is_shard:
            return None
        return {index: f"{mb}MiB" for index, mb in self.max_memory_mb.items()}

    def gpu_columns(self) -> dict[str, Any]:
        """What a job row records: ``gpu_uuid`` (the first card) and ``gpu_uuids`` (a split's cards)."""
        return {"gpu_uuid": self.uuid, "gpu_uuids": self.uuids if self.is_shard else None}

    def describe(self) -> str:
        if self.card is None:
            return "CPU (no GPU visible)"
        if self.is_shard:
            return "split across " + " + ".join(card.describe() for card in self.cards)
        return self.card.describe()


#: Callables that free GPU memory THIS process holds only as an idle cache. Run
#: by :func:`place_job` before it reads the inventory. See
#: :func:`register_idle_release`.
_IDLE_RELEASERS: list = []


def register_idle_release(release) -> None:
    """Register a callable that frees GPU memory an idle cache holds in this process.

    WHY PLACEMENT RUNS THEM. A J-lens readout may leave its model on a card
    (`unload_after=False`, user decision 2026-09-13) in the worker that also
    runs every other `extraction`-queue job: fits, probes, band reports,
    interventions, activation extractions, circuit runs. Placement reads free
    memory from NVML, which counts that idle copy, so Auto judged the card
    holding it as fuller and chose the other one — for a model that only fits
    on the card holding the copy, an OOM on the 12 GB card — and a job that
    never touches the J-lens cache ran for hours beside the copy. The worker is
    solo, so nothing can use the copy while another job runs.

    Registering twice is a no-op. A releaser must be cheap when it holds nothing.
    """
    if release not in _IDLE_RELEASERS:
        _IDLE_RELEASERS.append(release)


def release_idle_gpu_memory() -> None:
    """Run every registered idle-cache release. Never raises: placement is the point."""
    for release in list(_IDLE_RELEASERS):
        try:
            release()
        except Exception as exc:  # noqa: BLE001 - a failed release must not fail the job
            logger.warning("Idle GPU cache release %r failed: %s", release, exc)


def empty_cache_on_cards(uuids: Iterable[str]) -> None:
    """Return cached, unused CUDA blocks to the driver on each of these cards, by UUID.

    ``torch.cuda.empty_cache()`` without a device empties only the CURRENT
    device, and a job that split or was placed elsewhere leaves blocks on cards
    that are not current. A card this process cannot see is skipped. The release
    itself is ``ml.model_devices.empty_cache_on``; this only resolves the cards.
    """
    import torch

    if not torch.cuda.is_available():
        return
    from ..ml.model_devices import empty_cache_on

    devices = []
    for uuid in uuids:
        try:
            devices.append(torch_device(uuid))
        except GpuPlacementError:
            continue
    empty_cache_on(devices)


def place_job(
    requested: GpuRequest = AUTO,
    required_mb: Optional[float] = None,
    cards: Optional[Iterable[GpuCard]] = None,
    allow_shard: bool = False,
) -> Placement:
    """For a worker starting a job: choose the card(s) against live free memory and make one current.

    The device is made CURRENT because code the job calls may still ask CUDA for
    "the current device" — bitsandbytes, and any ``torch.cuda`` call without a
    device argument — and would otherwise land on CUDA index 0 whatever card
    was chosen. A split makes its first card current.

    ``allow_shard`` is passed only by a job whose model code runs split: inputs
    on the embedding's device, hooked modules on their layer's device, memory
    accounted on every card. Every other job keeps one card, so a split can
    never reach code that would break on one. See :func:`resolve_cards`.

    With no CUDA, Auto returns the CPU, as the jobs did before; a named card or
    ``"all"`` is refused.

    Raises:
        GpuPlacementError: see :func:`resolve_cards` and :func:`torch_device`.
    """
    import torch

    if not torch.cuda.is_available():
        if is_auto(requested):
            return Placement(card=None, device=torch.device("cpu"))
        raise GpuPlacementError(
            f"CUDA is not available here, so GPU {requested!r} cannot be used.",
            requested=requested,
        )
    # BEFORE the inventory is read, so the choice sees the memory an idle cache
    # in this process was holding. See `register_idle_release`.
    release_idle_gpu_memory()
    # PHASE 3: inside a job, the cards the job has LEASED (claimed against the
    # live leases, all or none); otherwise exactly `resolve_cards`.
    from .gpu_job_claim import claim_cards

    chosen = claim_cards(requested, required_mb=required_mb, cards=cards, allow_shard=allow_shard)
    devices = tuple(torch_device(card) for card in chosen)
    torch.cuda.set_device(devices[0])
    if len(chosen) == 1:
        logger.info("Job placed on %s as %s (requested %r)", chosen[0].describe(), devices[0], requested)
        return Placement(card=chosen[0], device=devices[0])
    placement = Placement(
        card=chosen[0],
        device=devices[0],
        cards=chosen,
        devices=devices,
        # By TORCH index, which is what `max_memory` keys name; the NVML index
        # can differ under CUDA_VISIBLE_DEVICES.
        max_memory_mb={device.index: _shard_budget_mb(card) for card, device in zip(chosen, devices)},
    )
    logger.info("Job %s (requested %r)", placement.describe(), requested)
    return placement


def torch_device(card: Union[GpuCard, str]) -> "torch.device":
    """The torch device for a card in THIS process, matched by UUID.

    CUDA indices depend on ``CUDA_VISIBLE_DEVICES`` and on the order the driver
    enumerates cards; the UUID does not.
    """
    import torch

    uuid = card.uuid if isinstance(card, GpuCard) else card
    if not torch.cuda.is_available():
        raise GpuPlacementError(f"CUDA is not available here, so GPU {uuid} cannot be used.", requested=uuid)
    wanted = normalise_uuid(uuid)
    for index in range(torch.cuda.device_count()):
        if normalise_uuid(torch.cuda.get_device_properties(index).uuid) == wanted:
            return torch.device("cuda", index)
    raise GpuPlacementError(f"GPU {uuid} is not visible to this process.", requested=uuid)


def card_for_device(
    device: Any, cards: Optional[Iterable[GpuCard]] = None
) -> Optional[GpuCard]:
    """The card a torch device in THIS process is — :func:`torch_device` in reverse.

    For a job that finds a model ALREADY on a GPU and must say which card that
    is: the device string a cache recorded ("cuda:1") is only a position in this
    process, so it is matched to NVML's inventory by UUID, like everything else
    here.

    None for a CPU device, when CUDA is unavailable, or when NVML does not list
    the card — the caller then has no card to name and must place the job.
    """
    import torch

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    wanted = normalise_uuid(torch.cuda.get_device_properties(index).uuid)
    cards = list_cards() if cards is None else list(cards)
    return next((card for card in cards if normalise_uuid(card.uuid) == wanted), None)


def make_current(device: Any) -> None:
    """Make a GPU device CURRENT, as :func:`place_job` does for a card it chose.

    For a job that runs on a card chosen EARLIER — a model left loaded by a
    previous job — rather than by a placement now. Code that asks CUDA for "the
    current device" would otherwise land wherever the last job left it. A CPU
    device, or a process without CUDA, has nothing to make current.
    """
    import torch

    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
