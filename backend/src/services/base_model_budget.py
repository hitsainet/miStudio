"""How big a job's base model is, and how much of a split it may take beside an SAE.

Phase 2 of ``0xcc/plans/Multi-GPU-Plan.md`` lets a job load a model that fits no
single card across several. Two SAE jobs do that: on-the-fly SAE training and
SAE feature extraction. Both keep their SAE (and, for training, its optimizer and
batches) on ONE card, the placement's first — and ``Placement.max_memory`` hands
every card of the split to the model, less a 1 GB reserve each. accelerate fills
the SAE's card up to that budget, so the model would take the memory the SAE was
placed there to use and the first batch would OOM on a card the placement said
had room. :func:`budget_beside_sae` takes the SAE's share off that card's budget
before the model is loaded.

:func:`base_model_mb` sizes the model for the placement itself. ``place_job``
can only choose a split when it is given the job's size: without one, Auto
simply takes the freest card and the load then fails on it.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .gpu_placement import GpuPlacementError, Placement

#: Memory an inference SAE's card keeps beyond the SAE's weights: the per-batch
#: feature tensor ``[batch, seq, latent]`` and the top-k vectorisation over it.
#: The same allowance the loader's preflight keeps above a model's weights.
SAE_WORKING_RESERVE_MB = 2048

_MIB = 1024 * 1024


#: Formats whose row under-counts its parameters. A model row's ``params_count`` is
#: counted off the model the download loaded AT THE ROW'S QUANTIZATION, and a
#: quantized bitsandbytes 4-bit weight packs two values per byte, so it reports half
#: its elements (``Params4bit`` of a 64x128 weight: numel 4,096, measured). Sized at
#: 0.5 bytes a parameter, a Q4 model came out at about a quarter of its weights.
PACKED_FORMATS = frozenset({"Q4", "Q2"})


#: The routed-expert count, as each implementation spells it (Mixtral/granite, Qwen-MoE,
#: DeepSeek). ``model_loader.SHAPE_FIELDS`` records them when present. ERNIE's config
#: answers ``num_experts`` for its ``moe_num_experts``, so it needs no spelling of its own.
_ROUTED_EXPERT_FIELDS = ("num_local_experts", "num_experts", "n_routed_experts")

#: The always-on shared experts' COUNT, each at the routed experts' width (DeepSeek, ERNIE).
_SHARED_EXPERT_COUNT_FIELDS = ("n_shared_experts", "moe_num_shared_experts")

#: The shared MLP's total WIDTH, one of them (Qwen-MoE, granite's shared MLP).
_SHARED_EXPERT_WIDTH_FIELDS = ("shared_expert_intermediate_size", "shared_intermediate_size")


def _positive_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _expert_params_beyond_one_mlp(architecture_config: dict) -> int:
    """What a mixture-of-experts layer holds beyond the ONE dense MLP the config estimate counts.

    ``estimate_parameter_count`` counts ``3 * hidden * intermediate`` a layer. An MoE
    layer holds that per routed expert (at ``moe_intermediate_size`` when the config
    names one), plus any always-on shared experts. Without this a Q4 Mixtral-8x7B was
    described at about a sixth of its weights, below even its packed count.
    """
    routed = next(
        (n for n in (_positive_int(architecture_config.get(f)) for f in _ROUTED_EXPERT_FIELDS) if n), None
    )
    hidden = _positive_int(architecture_config.get("hidden_size"))
    layers = _positive_int(architecture_config.get("num_hidden_layers"))
    if not routed or routed <= 1 or not hidden or not layers:
        return 0
    intermediate = _positive_int(architecture_config.get("intermediate_size")) or 4 * hidden
    expert_width = _positive_int(architecture_config.get("moe_intermediate_size")) or intermediate
    shared_width = next(
        (n for n in (_positive_int(architecture_config.get(f)) for f in _SHARED_EXPERT_WIDTH_FIELDS) if n), None
    )
    shared = next(
        (n for n in (_positive_int(architecture_config.get(f)) for f in _SHARED_EXPERT_COUNT_FIELDS) if n), None
    ) or (1 if shared_width else 0)
    moe_mlp = 3 * hidden * (routed * expert_width + shared * (shared_width or expert_width))
    return layers * (moe_mlp - 3 * hidden * intermediate)


def _params_from_architecture(architecture_config: Any) -> Optional[int]:
    """The parameter count a row's ``architecture_config`` implies; None when it cannot say."""
    if not isinstance(architecture_config, dict) or not architecture_config:
        return None
    from types import SimpleNamespace

    from ..ml.model_loader import estimate_parameter_count

    try:
        dense = estimate_parameter_count(SimpleNamespace(**architecture_config))
    except Exception:  # noqa: BLE001 - an unreadable description leaves the row's own count
        return None
    if not dense:
        return dense
    return dense + _expert_params_beyond_one_mlp(architecture_config)


def params_for_sizing(params_count: Any, quantization: Any, architecture_config: Any = None) -> Optional[int]:
    """The parameter count to size a model row's weights with; None when the row has none.

    A Q4 or Q2 row's count is the PACKED count (see :data:`PACKED_FORMATS`), so it
    is sized from the row's ``architecture_config``, the config arithmetic the
    download itself sizes with, and never below the row's own count. Every other
    format, and a packed row without a usable description, keeps its own count.

    Every placement sizer that reads ``params_count`` goes through this, so a fix to
    the packed count cannot reach one job and miss its siblings.
    """
    if isinstance(params_count, bool) or not isinstance(params_count, int) or params_count <= 0:
        return None
    fmt = str(getattr(quantization, "value", quantization) or "").upper()
    if fmt in PACKED_FORMATS:
        described = _params_from_architecture(architecture_config)
        if described:
            return max(params_count, described)
    return params_count


def base_model_mb(
    params_count: Any, quantization: Any, architecture_config: Any = None
) -> Optional[float]:
    """A base model's estimated footprint in MB at its quantization; None when unknown.

    ``estimate_model_memory``'s figure: the weights plus 20%. Unknown when the
    row has no parameter count or names no format the loader knows — the caller
    then places without a size, exactly as before Phase 2.

    The count is :func:`params_for_sizing`'s: a Q4 or Q2 row is sized from its
    architecture, not its packed count.
    """
    params_count = params_for_sizing(params_count, quantization, architecture_config)
    if params_count is None:
        return None
    from ..ml.model_loader import QuantizationFormat, estimate_model_memory

    try:
        fmt = QuantizationFormat(getattr(quantization, "value", quantization))
    except ValueError:
        return None
    return estimate_model_memory(params_count, fmt) / _MIB


def inference_sae_card_mb(hidden_dim: Any, latent_dim: Any) -> float:
    """What an SAE used for inference needs on its card: fp32 weights plus working memory.

    Weights are the encoder and decoder matrices and both biases. Dimensions a
    row does not record count as zero, leaving the working reserve.
    """
    hidden = hidden_dim if isinstance(hidden_dim, int) and hidden_dim > 0 else 0
    latent = latent_dim if isinstance(latent_dim, int) and latent_dim > 0 else 0
    weights_mb = (2 * hidden * latent + hidden + latent) * 4 / _MIB
    return weights_mb + SAE_WORKING_RESERVE_MB


def sae_weights_mb(hidden_dim: Any, latent_dim: Any, bytes_per_value: float = 4) -> float:
    """An SAE's weights in MB: its encoder and decoder matrices and both biases.

    ``bytes_per_value`` is the precision the job holds it at: 4 for the fp32 SAEs
    ``circuit_capture_service._load_sae_sync`` builds for the steering core and the
    circuit passes, 2 for steering's ``.half()`` copies. Dimensions a row does not
    record count as zero.
    """
    hidden = hidden_dim if isinstance(hidden_dim, int) and hidden_dim > 0 else 0
    latent = latent_dim if isinstance(latent_dim, int) and latent_dim > 0 else 0
    return (2 * hidden * latent + hidden + latent) * bytes_per_value / _MIB


def sae_decoder_mb(hidden_dim: Any, latent_dim: Any, bytes_per_value: float = 4) -> float:
    """An SAE's decoder matrix in MB: what the steering core keeps on a layer's card.

    The core resolves each layer's decoder from an SAE loaded on the CPU and moves
    only that matrix to the card (``steering_core._load_wdec_by_layer``).
    """
    hidden = hidden_dim if isinstance(hidden_dim, int) and hidden_dim > 0 else 0
    latent = latent_dim if isinstance(latent_dim, int) and latent_dim > 0 else 0
    return hidden * latent * bytes_per_value / _MIB


#: An SAE encode's working memory beside its layer, in units of ``tokens x d_sae``
#: fp32 values. MEASURED on the real per-layer capture path (review round 3,
#: scratchpad p2-r3/probe_capture_working.py, torch MemTracker): the peak of
#: `_encode_layer` + decode + the event mask and its nonzero is 2.0 units for a
#: standard or top-k SAE and 3.0 for JumpReLU (pre-activations, threshold mask and
#: codes live together). One more unit: capture's per-layer loop still holds the
#: previous layer's codes while the next layer encodes, on the same card. And one
#: for the events both layers' masks select (int64 indices and fp32 values, 20 bytes
#: an event): enough for SAEs firing on up to 10% of features. A denser SAE can
#: take more than this.
SAE_ENCODE_WORKING_UNITS = 5

#: What an attribution pass keeps beside EACH hooked layer until its backward is
#: done, in the same units. MEASURED on the real passthrough hook + attribute_prompt
#: (scratchpad p2-r3/probe_attr_working.py): 3.1-4.4 units a layer for standard and
#: top-k SAEs, up to 7.9 for JumpReLU (codes, their retained gradient, the graph's
#: saved pre-activations and masks). Every hooked layer holds it at once.
SAE_BACKWARD_UNITS = 8


def sae_encode_working_mb(latent_dim: Any, tokens: int) -> float:
    """MB an SAE encode of ``tokens`` needs beside its layer while it runs (see :data:`SAE_ENCODE_WORKING_UNITS`)."""
    latent = latent_dim if isinstance(latent_dim, int) and latent_dim > 0 else 0
    return SAE_ENCODE_WORKING_UNITS * max(int(tokens), 0) * latent * 4 / _MIB


def sae_backward_retained_mb(latent_dim: Any, tokens: int) -> float:
    """MB an attribution pass keeps beside each hooked layer until its backward (see :data:`SAE_BACKWARD_UNITS`)."""
    latent = latent_dim if isinstance(latent_dim, int) and latent_dim > 0 else 0
    return SAE_BACKWARD_UNITS * max(int(tokens), 0) * latent * 4 / _MIB


def budget_beside_sae(placement: Placement, sae_card_mb: float) -> Optional[dict]:
    """``max_memory`` for the base model of a job whose SAE works on ``placement.device``.

    None for a single-card placement, whose ``device_map`` names its device. For
    a split, ``placement.max_memory`` with ``sae_card_mb`` taken off the SAE's
    card, keyed by torch index and with no ``"cpu"`` key.

    Raises:
        GpuPlacementError: the SAE's card cannot hold the SAE at all, so no part
            of the model could go there either.
    """
    if not placement.is_shard:
        return None
    budgets = dict(placement.max_memory_mb)
    index = placement.device.index
    remaining = budgets[index] - int(math.ceil(sae_card_mb))
    if remaining <= 0:
        raise GpuPlacementError(
            f"{placement.card.describe()} is the SAE's card in this split and cannot hold the "
            f"SAE's ~{sae_card_mb:,.0f} MB beside its share of the model "
            f"({budgets[index]:,} MB budget). Free memory on it or choose a GPU.",
            required_mb=sae_card_mb,
            cards=placement.cards,
        )
    budgets[index] = remaining
    return {torch_index: f"{mb}MiB" for torch_index, mb in budgets.items()}
