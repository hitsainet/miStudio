"""How much host memory each GPU worker already holds, so a forecast can credit it (ENH-006 G14).

A probe run executes inside a GPU worker, and that worker's heap is reused by its next job. On
2026-11-04 an idle worker held 19.2 GiB; the specialist run then ran in the same process and the
worker grew only to 31.5 GiB against a 30.9 GiB forecast. The forecast, though, measured host
`MemAvailable`, which excludes what the worker holds, and so counted that memory as unavailable:
the general run was refused at 68.95 GiB needed against 54.79 usable.

What a run needs from the host beyond the worker it runs in is ``needed - held``. ``held`` is the
worker's ANONYMOUS memory (``RssAnon``), never its whole RSS: file-backed pages (the run's
memory-mapped token files) are page cache the kernel can drop, which `MemAvailable` already counts
as available, so crediting them would count them twice.

Two readers: the run's own check, inside the worker (:func:`anon_bytes` of itself), and the
request-time preflight in the API container, which cannot see another container's processes. For
that one the GPU supervisor publishes each worker's figure to a small file on the shared ``/data``
volume (:class:`WorkerMemoryPublisher`), and :func:`held_bytes` reads it. A figure older than
:data:`FRESH_S` is ignored, and a request that names no card credits the LEAST any worker holds,
since any of them may run it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..core import legacy_names

logger = logging.getLogger(__name__)

#: 042 FR-042.8: the new name; the old MISTUDIO_GPU_WORKER_MEMORY_DIR is retired since ENH-009 P7 (never read).
MEMORY_DIR_ENV = "MIWORKBENCH_GPU_WORKER_MEMORY_DIR"
DEFAULT_MEMORY_DIR = "/data/.gpu-worker-memory"
#: A published figure older than this is not trusted (the supervisor publishes every few seconds).
FRESH_S = 60.0
#: How often the supervisor rewrites the figures.
PUBLISH_EVERY_S = 10.0
#: File name for the worker of a machine with no card.
NO_CARD = "none"


def memory_dir(environ: Optional[dict] = None) -> str:
    return legacy_names.env_from(environ, MEMORY_DIR_ENV) or DEFAULT_MEMORY_DIR


def anon_bytes(pid: Optional[int] = None, *, proc: str = "/proc") -> Optional[int]:
    """A process's anonymous resident memory (``RssAnon``); None where /proc does not give it."""
    path = f"{proc}/{'self' if pid is None else pid}/status"
    try:
        with open(path, encoding="ascii") as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def entry_path(directory: str | os.PathLike, card: Optional[str]) -> Path:
    return Path(directory) / f"{card or NO_CARD}.json"


class WorkerMemoryPublisher:
    """Supervisor side: write each running worker's held memory; remove a stopped worker's entry."""

    def __init__(
        self,
        directory: str,
        *,
        read: Callable[[int], Optional[int]] = anon_bytes,
        clock: Callable[[], float] = time.time,
        every_s: float = PUBLISH_EVERY_S,
    ):
        self.directory, self.read, self.clock, self.every_s = directory, read, clock, every_s
        self._last: Optional[float] = None

    def __call__(self, statuses: Iterable[dict]) -> None:
        now = self.clock()
        if self._last is not None and now - self._last < self.every_s:
            return
        self._last = now
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        for status in statuses:
            path = entry_path(self.directory, status.get("card"))
            pid = status.get("pid")
            held = self.read(pid) if status.get("state") == "running" and pid else None
            if held is None:
                path.unlink(missing_ok=True)
                continue
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"card": status.get("card"), "pid": pid, "anon_bytes": held, "at": now})
            )
            tmp.replace(path)  # a reader never sees half a file


def held_bytes(directory: Optional[str], gpu: Optional[str], *, now: Optional[float] = None) -> int:
    """Memory the worker that will run a job already holds and will reuse. 0 when unknown.

    ``gpu`` is a card UUID for a job that names its card; anything else ("auto", an index, None)
    credits the least any fresh worker holds, because the job may land on any of them.
    """
    if not directory:
        return 0
    current = time.time() if now is None else now
    figures: dict = {}
    try:
        paths = list(Path(directory).glob("*.json"))
    except OSError:
        return 0
    for path in paths:
        try:
            entry = json.loads(path.read_text())
            at, held = float(entry["at"]), int(entry["anon_bytes"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if 0 <= current - at <= FRESH_S and held >= 0:
            figures[entry.get("card") or NO_CARD] = held
    if not figures:
        return 0
    if gpu and gpu.startswith("GPU-"):
        return figures.get(gpu, 0)
    return min(figures.values())
