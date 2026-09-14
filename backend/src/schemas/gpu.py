"""The ``gpu`` field every GPU job request carries.

``"auto"`` (the default) lets the worker pick the card with the most free memory
when the job starts. A GPU UUID, as ``GET /api/v1/system/gpu-list`` reports it,
names one card; an NVML index is accepted too and is turned into that card's
UUID when the job is submitted, because an index names a different card once
another is added. Plan: ``0xcc/plans/Multi-GPU-Plan.md``, Phase 1.
"""

from typing import Annotated

from pydantic import StringConstraints

#: Equal to ``services.gpu_placement.AUTO`` (a test pins it). Not imported from
#: there: importing ``src.services.gpu_placement`` runs ``services/__init__``,
#: which imports ``training_service``, which imports ``schemas.training`` — so a
#: request schema that uses ``GpuRequestStr`` would import itself half-built.
AUTO = "auto"

#: Equal to ``services.gpu_placement.ALL`` (a test pins it): split the model across every card.
ALL = "all"

#: ``auto``, ``all``, an NVML index, or a GPU UUID with or without NVML's ``GPU-`` prefix.
GPU_REQUEST_PATTERN = (
    r"^(auto|all|[0-9]{1,2}|([Gg][Pp][Uu]-)?[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})$"
)

GPU_REQUEST_DESCRIPTION = (
    '"auto" to run on the GPU with the most free memory when the job starts (a model '
    'no single GPU can hold is split across GPUs where the job supports it), "all" to '
    "split the model across every GPU, or a GPU UUID (or index) from "
    "/api/v1/system/gpu-list to run on that card. A named card, or a split, that "
    "cannot take the job is refused, never swapped for another choice."
)

GpuRequestStr = Annotated[
    str, StringConstraints(strip_whitespace=True, max_length=64, pattern=GPU_REQUEST_PATTERN)
]

__all__ = ["ALL", "AUTO", "GPU_REQUEST_DESCRIPTION", "GPU_REQUEST_PATTERN", "GpuRequestStr"]
