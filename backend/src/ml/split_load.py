"""Map a split load the way transformers will, before a single weight is read.

A split placement (``services.gpu_placement``) gives each card a budget: its free
memory less ``SHARD_RESERVE_MB``. It accepts the split when those budgets cover
the job. transformers 5.15.1 then maps MODULES onto the cards with its copy of
accelerate's ``infer_auto_device_map`` (``integrations/accelerate.py``), which
does not honour the budgets as given:

1. it fills the cards in CUDA INDEX order, whatever order the placement chose
   them in (``get_max_memory`` sorts the integer keys);
2. it holds back room for the model's largest layer on the LOWEST-index card
   (``main_devices = [gpus[0], "cpu"]``). That room is for putting back a layer
   offloaded to the CPU — and a miStudio split has no ``"cpu"`` key, so nothing
   is ever put back and the room is simply lost;
3. for bitsandbytes — requested at load, or already in the checkpoint — it fills
   each card only to 90% (``adjust_max_memory``);
4. it strands the tail of a card when the next unsplittable block does not fit.

What spills goes past the last card to ``"disk"``. So a split whose budgets held
the model was refused — by the loader's post-load check after ``from_pretrained``
had read every weight, or by bitsandbytes with a message recommending CPU
offload. On the node the lowest index is the 12 GB 3080 Ti, which makes (2) the
largest share of its budget.

:func:`plan_split_load` builds the model on the meta device (no memory, no CUDA)
with the quantizer ``from_pretrained`` would pick, computes transformers' own
map, and:

* returns the hold-back of (2) to the lowest-index card, checked against every
  card's real limit — accelerate re-measures its "largest layer" as modules are
  placed, so returning the whole initial figure can let that card take more than
  its budget, and the return is shrunk until it does not. A map with the room
  returned is never preferred to transformers' own when only the latter fits;
* refuses, with the map's figures, a split that spills either way — before
  anything is downloaded or loaded.

(3) and (4) are kept: the 10% is bitsandbytes' quantization workspace, and an
unsplittable block cannot be divided.

WORK BESIDE A LAYER (``extra_mb_by_layer``, review round 2). Steering, the
steering core (calibration, the recorder) and the circuit passes put each
layer's SAE on the card that layer is mapped to (plan decision D5). The budget
keeps only ``SHARD_RESERVE_MB`` free on each card, and a 12B-class SAE is
1.2-2.6 GB, so a layer mapped to a card the model had filled took the reserve
and the job ran out of memory after loading. Measured on OLMo-2-1124-13B at
FP16 over the node's cards (11,000 + 23,000 MB free): the map leaves 525 MiB on
the 3080 Ti, where layer 13 and its SAE go. Given each layer's extra MiB, the
plan charges it to the card its layer maps to, takes any excess off that card's
budget and maps again, until every card holds its share of the model AND the
work beside its layers, or refuses.

None means "not verifiable here" (a class that will not build on meta, a
quantizer whose package is missing, no decoder layers to charge an allowance
to) and never refuses: the load maps the split itself, and the loader's
post-load check still refuses anything off the GPUs.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

#: How many times the returned hold-back is shrunk after it let the lowest-index
#: card take more than its limit, before settling for transformers' own map.
RECLAIM_ATTEMPTS = 4

#: How many times a card's budget is cut for the work beside its layers and the
#: split mapped again, before a split that still overcommits is refused. Each cut
#: moves at least the excess off its card, so a pass per card is the usual need.
ALLOWANCE_PASSES = 16


class SplitDoesNotFit(RuntimeError):
    """transformers would map part of the model off the GPUs of its split."""

    def __init__(self, message: str, *, mapped_mb: dict, limit_mb: dict) -> None:
        super().__init__(message)
        self.mapped_mb = mapped_mb
        self.limit_mb = limit_mb


@dataclass(frozen=True)
class SplitLoadPlan:
    """What a split load passes to ``from_pretrained`` and what transformers maps from it."""

    #: ``max_memory`` for ``device_map="sequential"``, by torch index, no "cpu" key.
    max_memory: dict
    #: The module map transformers computes from ``max_memory``.
    device_map: dict
    #: MiB of that map on each card, by torch index.
    mapped_mb: dict
    #: MiB of the hold-back returned to the lowest-index card.
    reclaimed_mb: int
    #: MiB of ``extra_mb_by_layer`` charged to each card its layers map to, by torch index.
    extra_mb: dict = field(default_factory=dict)
    #: MiB of ``working_mb_by_layer`` kept on each card: the largest of its layers', by torch index.
    working_mb: dict = field(default_factory=dict)


def _version(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # noqa: BLE001 - a version is only named in a message
        return "unknown"


def _to_bytes(value: Any) -> int:
    if isinstance(value, str):
        from accelerate.utils import convert_file_size_to_int

        return int(convert_file_size_to_int(value))
    return int(value)


def _quantizer(config: Any, quantization_config: Any):
    """The quantizer ``from_pretrained`` would use, picked as ``get_hf_quantizer`` picks it.

    A checkpoint that ships its own quantization (a ``-bnb-4bit`` repo) gets its
    quantizer from its config even when the load passes none, and bitsandbytes
    then fills each card to 90% just the same. Without the environment checks
    ``get_hf_quantizer`` also runs: those refuse a map off the GPU, which is what
    this module reports, with figures.
    """
    from transformers.quantizers.auto import AutoHfQuantizer

    from_checkpoint = getattr(config, "quantization_config", None)
    if from_checkpoint is None:
        text_config = getattr(config, "get_text_config", None)
        text = text_config(decoder=True) if callable(text_config) else None
        from_checkpoint = getattr(text, "quantization_config", None)
    if from_checkpoint is not None and AutoHfQuantizer.supports_quant_method(from_checkpoint):
        merged = AutoHfQuantizer.merge_quantization_configs(from_checkpoint, quantization_config)
        return AutoHfQuantizer.from_config(merged, pre_quantized=True)
    if quantization_config is not None:
        return AutoHfQuantizer.from_config(quantization_config, pre_quantized=False)
    return None


def _skeleton(config: Any, dtype: torch.dtype, quantization_config: Any, trust_remote_code: bool):
    """The model ``from_pretrained`` would build, on the meta device, with the quantizer it would use."""
    from transformers import AutoModelForCausalLM

    config = copy.deepcopy(config)
    quantizer = _quantizer(config, quantization_config)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, trust_remote_code=trust_remote_code)
    if quantizer is not None:
        quantizer.preprocess_model(
            model=model, dtype=dtype, device_map="sequential", checkpoint_files=None, use_kernels=False
        )
    return model, quantizer


def _infer_map(model: Any, max_memory: dict, quantizer: Any) -> dict:
    """transformers' ``_get_device_map`` for "sequential", without its quantizer validation.

    The validation only raises on a map that leaves the GPU — the case this
    module reports itself, with the figures that message lacks.
    """
    from transformers.integrations.accelerate import get_max_memory, infer_auto_device_map

    inferred = get_max_memory(dict(max_memory))
    if quantizer is not None:
        inferred = quantizer.adjust_max_memory(inferred)
    return infer_auto_device_map(
        model, max_memory=inferred, no_split_module_classes=model._no_split_modules, hf_quantizer=quantizer
    )


def _largest_layer_bytes(model: Any, sizes: Mapping[str, int]) -> int:
    """The hold-back accelerate takes off the lowest-index card when it starts filling."""
    from accelerate.utils.modeling import get_max_layer_size

    top = (
        list(model.named_parameters(recurse=False))
        + list(model.named_children())
        + list(model.named_buffers(recurse=False))
    )
    size, _ = get_max_layer_size(top, sizes, list(model._no_split_modules or []))
    return int(size)


def _largest_block_bytes(model: Any, sizes: Mapping[str, int]) -> int:
    no_split = set(model._no_split_modules or [])
    blocks = [sizes.get(name, 0) for name, module in model.named_modules() if type(module).__name__ in no_split]
    return int(max(blocks, default=0))


def _is_gpu(device: Any) -> bool:
    return isinstance(device, int) and not isinstance(device, bool)


def _label(device: Any) -> str:
    return f"cuda:{device}" if _is_gpu(device) else str(device)


def _layer_names(model: Any) -> Optional[list]:
    """Each decoder layer's module name, in layer order; None when no layers are found.

    Found with ``discover_transformer_structure``, the discovery the jobs run on the
    LOADED model to read a layer's card (``layer_device``), so an allowance for
    layer N is charged to the module those jobs call layer N.
    """
    from .layer_discovery import discover_transformer_structure

    structure = discover_transformer_structure(model)
    names = {id(module): name for name, module in model.named_modules()}
    layers = [names.get(id(layer)) for layer in structure.layers_module]
    if not layers or any(name is None for name in layers):
        return None
    return layers


def _owner(device_map: Mapping[str, Any], name: str) -> Optional[str]:
    """The device-map key that places module ``name``: its own entry or its nearest ancestor's.

    A module the map divided has no entry of its own; it is placed by its first
    mapped child, as ``module_device`` reads its first parameter.
    """
    owner = None
    for key in device_map:
        if key == "" or key == name or name.startswith(key + "."):
            if owner is None or len(key) > len(owner):
                owner = key
    if owner is not None:
        return owner
    return next((key for key in device_map if key.startswith(name + ".")), None)


def _device_of(device_map: Mapping[str, Any], name: str) -> Any:
    """The device a device map gives module ``name`` (see :func:`_owner`)."""
    key = _owner(device_map, name)
    return None if key is None else device_map[key]


def _model_bytes_to_move(
    device_map: Mapping[str, Any],
    sizes: Mapping[str, int],
    index: int,
    limit: float,
    extra_by_key: Mapping[str, int],
    work_by_key: Mapping[str, int],
    unowned: int = 0,
) -> int:
    """The model bytes that must leave card ``index`` for it to hold its share beside its layers' work.

    accelerate fills a card with a RUN of modules in the map's order, so a smaller
    budget takes modules off the END of that run. A module that leaves takes the
    work beside its layers with it. Cutting the budget by the whole excess (review
    round 2) therefore moved the SAEs' bytes of MODEL off the card as well: on
    OLMo-2-13B with an SAE beside layers 11-13 it moved seven layers where two had
    to go, the last card could not take them, and a split that fits was refused.

    Removes the card's trailing modules until the rest, beside the work of the
    layers left on it, is within ``limit``, and returns the model bytes removed.
    """
    held = [key for key, device in device_map.items() if _is_gpu(device) and device == index]

    def charge(keys: list) -> int:
        return (
            sum(sizes.get(key, 0) + extra_by_key.get(key, 0) for key in keys)
            + max((work_by_key.get(key, 0) for key in keys), default=0)
            + unowned
        )

    moved = 0
    while held and charge(held) > limit:
        moved += sizes.get(held.pop(), 0)
    return moved


def plan_split_load(
    config: Any,
    *,
    max_memory: Mapping[Any, Any],
    dtype: torch.dtype,
    quantization_config: Any = None,
    trust_remote_code: bool = False,
    model_name: str = "model",
    extra_mb_by_layer: Optional[Mapping[int, float]] = None,
    working_mb_by_layer: Optional[Mapping[int, float]] = None,
) -> Optional[SplitLoadPlan]:
    """The ``max_memory`` a split load should pass, verified against transformers' own map.

    Args:
        config: The checkpoint's config, as ``from_pretrained`` will read it.
        max_memory: The split's per-card budget by torch index (``Placement.max_memory``).
        dtype: The dtype the load passes.
        quantization_config: The bitsandbytes config the load passes, if any.
        extra_mb_by_layer: MiB the job HOLDS beside each decoder layer for the
            whole job, on that layer's card, by layer index (its SAE). Every
            layer's is charged to the card the layer maps to.
        working_mb_by_layer: MiB the job allocates beside a layer only while it
            works on that layer, one layer at a time (an SAE encode's codes). A
            card keeps room for the LARGEST of its layers', not their sum.

        A card that cannot hold its share of the model beside both gives up the
        modules at the end of its run, and the split is mapped again. None or
        empty: nothing extra.

    Returns:
        The plan, or None when the map cannot be computed here.

    Raises:
        SplitDoesNotFit: part of the model would map to the CPU or disk, or a
            card cannot hold its share of the model beside ``extra_mb_by_layer``.
    """
    def unverifiable(exc: BaseException) -> None:
        logger.warning(
            "Could not map the split of %s before loading (%s: %s). The load maps it itself "
            "and is checked after.", model_name, type(exc).__name__, exc,
        )

    def planner_broken(exc: BaseException) -> None:
        # NOT a checkpoint this planner cannot map: its own calls into transformers'
        # and accelerate's private internals failed, which a library upgrade does to
        # EVERY split load at once. Still never a refusal, but never a quiet one.
        logger.error(
            "The split planner cannot run against transformers %s / accelerate %s (%s: %s). "
            "Every split load is now unmapped before loading: a split whose budgets hold the "
            "model can spill to disk and be refused only after its weights load. "
            "ml/split_load.py needs updating for these versions.",
            _version("transformers"), _version("accelerate"), type(exc).__name__, exc,
        )

    try:
        from accelerate.utils import convert_file_size_to_int  # noqa: F401
        from accelerate.utils.modeling import get_max_layer_size  # noqa: F401
        from transformers.integrations.accelerate import (  # noqa: F401
            compute_module_sizes,
            get_max_memory,
            infer_auto_device_map,
        )
        from transformers.quantizers.auto import AutoHfQuantizer  # noqa: F401
    except ImportError as exc:
        planner_broken(exc)
        return None

    budgets = {int(index): _to_bytes(value) for index, value in max_memory.items()}
    if not budgets:
        return None
    extra = {int(layer): float(mb) for layer, mb in (extra_mb_by_layer or {}).items() if mb and float(mb) > 0}
    work = {int(layer): float(mb) for layer, mb in (working_mb_by_layer or {}).items() if mb and float(mb) > 0}

    try:
        model, quantizer = _skeleton(config, dtype, quantization_config, trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - unverifiable here; the load maps and checks it
        unverifiable(exc)
        return None
    try:
        sizes, _ = compute_module_sizes(model, quantizer, only_modules=False)
        hold_back = _largest_layer_bytes(model, sizes)
        block = _largest_block_bytes(model, sizes)
        fill = 1.0 if quantizer is None else quantizer.adjust_max_memory({0: MIB})[0] / MIB
    except Exception as exc:  # noqa: BLE001 - the model built; the internals did not run
        planner_broken(exc)
        return None
    # Which module each layer is: the jobs' own layer discovery, not a library
    # internal, so a model it finds no layers in is unverifiable here, like one that
    # will not build on meta - never a broken planner.
    try:
        layer_names = _layer_names(model) if (extra or work) else None
    except Exception as exc:  # noqa: BLE001 - no layers to charge an allowance to; the load maps it
        unverifiable(exc)
        return None
    if extra or work:
        if layer_names is None:
            unverifiable(LookupError("no decoder layers to charge the per-layer allowance to"))
            return None
        outside = sorted(layer for layer in {*extra, *work} if not 0 <= layer < len(layer_names))
        if outside:
            logger.warning(
                "%s has %d decoder layers; no allowance is kept for layer(s) %s.",
                model_name, len(layer_names), outside,
            )
            extra = {layer: mb for layer, mb in extra.items() if layer not in outside}
            work = {layer: mb for layer, mb in work.items() if layer not in outside}
    charged = bool(extra or work)

    # What accelerate may put on each card: the budget at bitsandbytes' fill.
    limits = {index: budget * fill for index, budget in budgets.items()}
    lowest = min(budgets)
    reclaim = hold_back
    attempts = 0
    # Budget taken off each card for the work beside the layers it holds.
    cut = {index: 0 for index in budgets}
    passes = 0
    while True:
        passed = {index: max(budgets[index] - cut[index], 0) for index in budgets}
        passed[lowest] = passed[lowest] + math.floor(reclaim / fill)
        as_mib = {index: f"{passed[index] // MIB}MiB" for index in sorted(passed)}
        try:
            device_map = _infer_map(model, as_mib, quantizer)
        except Exception as exc:  # noqa: BLE001 - from_pretrained runs this same inference
            planner_broken(exc)
            return None
        mapped: dict = {}
        for name, device in device_map.items():
            mapped[device] = mapped.get(device, 0) + sizes.get(name, 0)
        over = max((mapped.get(index, 0) - limit for index, limit in limits.items()), default=0)
        spills = any(not _is_gpu(device) for device in device_map.values())
        if reclaim > 0 and over > 0:
            attempts += 1
            # Smaller by what overflowed and one unsplittable block, so at least
            # one whole block leaves the card; after RECLAIM_ATTEMPTS, none.
            reclaim = 0 if attempts >= RECLAIM_ATTEMPTS else max(reclaim - over - block, 0)
            continue
        if reclaim > 0 and spills:
            # Never refuse a split transformers' own map would load.
            reclaim = 0
            continue
        # What each card holds beside its layers (summed) and works on beside
        # them (the largest), by device, and by the map key each layer belongs to.
        beside: dict = {}
        working: dict = {}
        extra_by_key: dict = {}
        work_by_key: dict = {}
        unowned: dict = {}
        for allowance, by_key, into, combine in ((extra, extra_by_key, beside, lambda a, b: a + b),
                                                 (work, work_by_key, working, max)):
            for layer, mb in allowance.items():
                key = _owner(device_map, layer_names[layer])
                device = lowest if key is None else device_map[key]
                amount = math.ceil(mb * MIB)
                into[device] = combine(into.get(device, 0), amount)
                if key is None:
                    unowned[device] = unowned.get(device, 0) + amount
                else:
                    by_key[key] = combine(by_key.get(key, 0), amount)
        # ONLY A SPLIT WITH WORK BESIDE ITS LAYERS is re-mapped here. A model that
        # overcommits a card on its own is the hold-back's to correct (above), and
        # a split with no allowance maps exactly as it did before this existed.
        excess = {} if not charged else {
            index: mapped.get(index, 0) + beside.get(index, 0) + working.get(index, 0) - limit
            for index, limit in limits.items()
            if mapped.get(index, 0) + beside.get(index, 0) + working.get(index, 0) > limit
        }
        if excess and not spills and passes < ALLOWANCE_PASSES:
            passes += 1
            # THE LOWEST OVERCOMMITTED CARD, BY ONLY WHAT MUST LEAVE IT. Every
            # module it gives up lands on a higher card, so a higher card's
            # excess is judged only after the cards below it have settled; and
            # the modules leaving take their layers' work with them, so the cut
            # is their model bytes, never the whole excess. At bitsandbytes'
            # fill, and at least a MiB, so every pass moves the map.
            index = min(excess)
            moved = _model_bytes_to_move(
                device_map, sizes, index, limits[index], extra_by_key, work_by_key, unowned.get(index, 0)
            )
            cut[index] += max(math.ceil(moved / fill), MIB)
            continue
        break

    mapped_mb = {_label(device): math.ceil(size / MIB) for device, size in mapped.items()}
    limit_mb = {_label(index): math.floor(limit / MIB) for index, limit in limits.items()}
    beside_mb = {index: math.ceil(size / MIB) for index, size in sorted(beside.items(), key=lambda kv: str(kv[0]))}
    working_mb = {index: math.ceil(size / MIB) for index, size in sorted(working.items(), key=lambda kv: str(kv[0]))}
    alongside = (
        f" beside {sum(extra.values()):,.0f} MiB its job puts on its layers' cards "
        f"({', '.join(f'{_label(i)} {mb:,}' for i, mb in beside_mb.items())} MiB)"
        if extra else ""
    ) + (
        f"{' and' if extra else ' beside'} the up to {max(work.values()):,.0f} MiB it works with beside a layer "
        f"({', '.join(f'{_label(i)} {mb:,}' for i, mb in working_mb.items())} MiB)"
        if work else ""
    )
    off_gpu = {name: device for name, device in device_map.items() if not _is_gpu(device)}
    if off_gpu:
        first_name, first_device = next(iter(off_gpu.items()))
        figures = ", ".join(
            f"{label} {mapped_mb[label]:,} of {limit_mb[label]:,} MiB" if label in limit_mb
            else f"{label} {mapped_mb[label]:,} MiB"
            for label in sorted(mapped_mb, key=lambda label: (not label.startswith("cuda:"), label))
        )
        raise SplitDoesNotFit(
            f"{model_name} does not fit on the GPUs it was split across{alongside}: transformers would map "
            f"{len(off_gpu)} module(s) to {'/'.join(sorted({str(d) for d in off_gpu.values()}))} "
            f"(for example {first_name} on {first_device}). Mapped per device: {figures}. "
            f"Nothing was loaded. miStudio runs models on GPUs only; free memory on those cards "
            f"or choose a smaller quantization.",
            mapped_mb=mapped_mb,
            limit_mb=limit_mb,
        )
    if over > 0:
        # Unreachable while accelerate honours its own limits; never hand on a
        # map that takes a card's reserve.
        raise SplitDoesNotFit(
            f"{model_name}: transformers' map exceeds a card's budget ({mapped_mb} MiB mapped, "
            f"limits {limit_mb} MiB). Nothing was loaded.",
            mapped_mb=mapped_mb,
            limit_mb=limit_mb,
        )
    if excess:
        raise SplitDoesNotFit(
            f"{model_name} does not fit on the GPUs it was split across{alongside}: after "
            f"{passes} re-maps a card still cannot hold its share of the model beside that work "
            f"({mapped_mb} MiB of model mapped, limits {limit_mb} MiB). Nothing was loaded.",
            mapped_mb=mapped_mb,
            limit_mb=limit_mb,
        )
    plan = SplitLoadPlan(
        max_memory=as_mib,
        device_map=dict(device_map),
        mapped_mb={index: math.ceil(mapped.get(index, 0) / MIB) for index in sorted(budgets)},
        reclaimed_mb=reclaim // MIB,
        extra_mb={index: mb for index, mb in beside_mb.items() if _is_gpu(index)},
        working_mb={index: mb for index, mb in working_mb.items() if _is_gpu(index)},
    )
    logger.info(
        "Split of %s mapped before loading: %s MiB per card (limits %s MiB)%s%s; %s MiB of accelerate's "
        "hold-back returned to cuda:%d.", model_name, plan.mapped_mb, limit_mb,
        f", {plan.extra_mb} MiB beside its layers" if extra else "",
        f", {plan.working_mb} MiB of working room beside them" if work else "", plan.reclaimed_mb, lowest,
    )
    unused = [index for index, mb in plan.mapped_mb.items() if mb == 0]
    if unused:
        # NOT A REFUSAL: the model and its work fit on the other cards. But the
        # placement chose, recorded and holds these cards, and nothing of this job
        # will run on them while it does.
        logger.warning(
            "Split of %s puts none of the model on %s: the layers it could hold do not fit there beside "
            "the work of those layers. The job still holds %s.",
            model_name, ", ".join(_label(index) for index in unused), "it" if len(unused) == 1 else "them",
        )
    return plan
