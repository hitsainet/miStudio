"""Peak GPU memory per stage of a job, and the card it was measured on (MCP gap G11, 2026-10-21).

A probe training run recorded each stage's wall time and the model's dtype, and nothing about
memory — so neither an agent nor the operator could size the next run, or say how much of a card
a probe run takes, without watching `nvidia-smi` during it.

`torch.cuda.max_memory_allocated` is a high-water mark since the last
`reset_peak_memory_stats`; reading it at a stage's end without a reset at its start would report
the PEAK SO FAR for every stage, so later stages would inherit earlier ones' numbers. The reset
per stage is therefore the load-bearing line, and `tests/unit/test_gpu_stage_memory.py` fails
without it.

What it records is PyTorch's allocator view of THIS process: tensors allocated (and the cache
reserved) by the job. The CUDA context and other processes on the card are not in it; the card's
total use is `get_gpu_status`'s business.

On a model that is not on a CUDA device (a CPU host, the pipeline test) the record says
`measured: false` with the reason, and each stage still appears — with `null` peaks, never zeros:
an unmeasured value written as 0 would read as "this stage used no memory".
"""

from __future__ import annotations

from typing import Any, Optional

MIB = 1024 * 1024


def _cuda() -> Any:
    import torch

    return torch.cuda


def cuda_device_of(model: Any) -> Optional[Any]:
    """The CUDA device a model's first parameter sits on, or None (CPU, meta, no parameters)."""
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return None
    return device if getattr(device, "type", None) == "cuda" else None


class StagePeakMemory:
    """Reset the peak at each stage's start; read it at the stage's end.

    `start(name)` closes any stage still open (so a stage with no explicit end is still recorded)
    and resets the peak; `stop(name)` records it. `record()` is the JSON-able dict a job stores.
    """

    def __init__(self, device: Optional[Any], gpu_uuid: Optional[str] = None,
                 cuda: Any = None) -> None:
        self._cuda = cuda if cuda is not None else _cuda()
        self._device = device
        self._open: Optional[str] = None
        measured = device is not None
        card_uuid = gpu_uuid
        if measured:
            try:
                props = self._cuda.get_device_properties(device)
                torch_uuid = getattr(props, "uuid", None)
                if torch_uuid is not None:
                    text = str(torch_uuid).lower()
                    card_uuid = text if text.startswith("gpu-") else f"GPU-{text}"
            except Exception:  # noqa: BLE001 - the row's own UUID still stands
                pass
        self._record: dict[str, Any] = {
            "measured": measured,
            "gpu_uuid": card_uuid,
            "device": str(device) if device is not None else None,
            "unit": "MiB",
            "source": "torch.cuda.max_memory_allocated / max_memory_reserved, reset per stage",
            "stages": {},
            "peak_allocated_mib": None,
            "peak_reserved_mib": None,
        }
        if not measured:
            self._record["reason"] = (
                "the model is not on a CUDA device, so there is no GPU memory to measure"
            )

    def start(self, name: str) -> None:
        if self._open is not None:
            self.stop(self._open)
        self._open = name
        if self._device is not None:
            self._cuda.reset_peak_memory_stats(self._device)

    def stop(self, name: str) -> None:
        if self._open != name:
            return  # a stop with no matching start records nothing rather than a stale peak
        self._open = None
        entry: dict[str, Optional[float]] = {"peak_allocated_mib": None, "peak_reserved_mib": None}
        if self._device is not None:
            allocated = round(self._cuda.max_memory_allocated(self._device) / MIB, 1)
            reserved = round(self._cuda.max_memory_reserved(self._device) / MIB, 1)
            entry = {"peak_allocated_mib": allocated, "peak_reserved_mib": reserved}
            for key, value in (("peak_allocated_mib", allocated), ("peak_reserved_mib", reserved)):
                current = self._record[key]
                self._record[key] = value if current is None else max(current, value)
        self._record["stages"][name] = entry

    def record(self) -> dict[str, Any]:
        return {**self._record, "stages": {k: dict(v) for k, v in self._record["stages"].items()}}
