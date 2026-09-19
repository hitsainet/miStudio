"""Where a loaded model's tensors are: one card, or several when the model is split.

Phase 2 of ``0xcc/plans/Multi-GPU-Plan.md`` loads a model that fits no single
card across several (``Placement.is_shard``). Code written for one card asks
``model.device`` or ``next(model.parameters()).device`` and moves everything
there. On a split model that is only the device of the FIRST parameter: input
ids may land right by luck (it is usually the embedding), but an SAE, a steering
vector or a lens tensor put there sits on the wrong card for every layer that
landed on another one, and the forward pass raises a device mismatch.

These helpers answer the questions a split makes real. They read devices from
the tensors themselves, never from a device name, so nothing here chooses a
card — ``services.gpu_placement`` does that.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Iterable, Optional

import torch

logger = logging.getLogger(__name__)

#: ``hf_device_map`` values that are not a GPU. A transformers load in miStudio
#: runs on GPUs only (operator decision 3, 2026-09-13), so any of these means
#: the load did not do what it was asked.
OFF_GPU_TARGETS = frozenset({"cpu", "disk", "meta"})


def module_device(module: torch.nn.Module) -> Optional[torch.device]:
    """The device of a module's first parameter or buffer; None for a module with neither."""
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        return tensor.device
    return None


def input_device(model: Any) -> torch.device:
    """Where input ids and attention masks go: the input embedding's device.

    Not ``model.device``, which transformers defines as the first parameter's
    device. On a split model those differ whenever the first registered
    parameter is not the embedding — and the mismatch surfaces as an error
    inside the embedding lookup, far from the ``.to(model.device)`` that caused it.
    """
    getter = getattr(model, "get_input_embeddings", None)
    embeddings = getter() if callable(getter) else None
    if isinstance(embeddings, torch.nn.Module):
        device = module_device(embeddings)
        if device is not None and device.type != "meta":
            return device
    device = module_device(model)
    if device is None:
        raise ValueError("The model holds no tensors, so there is no device for its inputs")
    return device


def model_devices(model: torch.nn.Module) -> list[torch.device]:
    """Every device the model holds tensors on: GPUs by index first, then anything else."""
    found: dict[str, torch.device] = {}
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        found.setdefault(str(tensor.device), tensor.device)
    return sorted(
        found.values(),
        key=lambda device: (device.type != "cuda", device.index if device.index is not None else -1, device.type),
    )


def cuda_devices(model: torch.nn.Module) -> list[torch.device]:
    """The GPUs a model holds tensors on — every card whose memory a cleanup must release."""
    return [device for device in model_devices(model) if device.type == "cuda"]


def off_gpu_modules(model: Any) -> dict[str, str]:
    """Modules accelerate mapped somewhere other than a GPU, by module name.

    Read from ``hf_device_map``: a module offloaded to "disk" holds ``meta``
    tensors, which a scan of the parameters would report as a device rather than
    as the offload it is.
    """
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return {}
    return {name: str(target) for name, target in device_map.items() if str(target) in OFF_GPU_TARGETS}


def detach_dispatch_hooks(model: torch.nn.Module) -> None:
    """Remove accelerate's dispatch hooks, so a split model can be moved or freed.

    A model loaded with ``device_map="auto"`` carries ``AlignDevicesHook``s that
    move tensors between cards inside ``forward``. ``model.to(...)`` on such a
    model fights those hooks, and the hooks keep references a cleanup expects to
    have dropped. A model with no hooks is left as it is.
    """
    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:  # pragma: no cover - accelerate ships with transformers here
        return
    remove_hook_from_module(model, recurse=True)


def empty_cache_on(devices: Iterable[torch.device]) -> None:
    """Return cached allocator memory on each GPU in ``devices``; a CPU device has none."""
    for device in devices:
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
