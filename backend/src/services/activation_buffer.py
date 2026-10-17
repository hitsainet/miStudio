"""Serve SAE training batches from the WHOLE of every cached extraction.

WHY THIS EXISTS. Training used to load one random subsample of token
positions, sized to what fits in GPU memory, and draw every batch from it with
replacement for the entire run. For LFM2.5-1.2B at 2,048 dims on a 24 GB card
that is ~2.4M tokens per layer (single-layer run) or ~0.3M (five layers at once)
— so a 20M-token extraction was ~90% never read, and the default 50,000 x 2,048
steps drew 102M samples from 2.4M tokens: every token seen ~43 times while most
of the corpus sat on disk. (When the pool did not fit on the GPU, the CPU path
instead loaded EVERYTHING into pinned RAM as float32, which a large extraction
cannot survive.)

WHAT IT DOES INSTEAD. A buffer the same size as before is filled with whole
blocks drawn from every source, tokens are shuffled across the buffer with one
permutation shared by every layer, and batches are served WITHOUT replacement.
When the buffer is spent it is refilled from blocks not yet used. A source's
blocks are only reshuffled once all of them have been served, so over a run
every trainable token is used before any is repeated.

WHOLE BLOCKS, NOT RANDOM POSITIONS. A buffer of 2.4M positions scattered across
20M would touch nearly every block of an 84 GB file per refill. Whole blocks
keep each refill to about the buffer's own size in reads; mixing across
~1,000+ blocks per buffer supplies the decorrelation.

WHY REFILLS ARE READ IN PARALLEL AND AHEAD OF TIME (multi-GPU Phase 4, 2026-09-14).
Measured on mcs-lnxhost02: a 3-layer JumpReLU run (batch 2,048, five
extractions, gpu_rolling on the 3090) stalled 31-33 s at every refill reading
~657k tokens x 3 layers (~15 GB) against ~41 s of training on that buffer — 44%
of wall time spent waiting on the disk. Random 8 MB rows from the extraction
read at 437-457 MB/s one at a time, 1,020 MB/s with a thread per layer and
1,560 MB/s with 12 threads, because numpy releases the GIL during the copy. So:

* rows are read by a bounded thread pool (``ACTIVATION_READ_THREADS``), each
  written straight to its shuffled position, so no unshuffled copy is ever held;
* the NEXT buffer is prepared on a background thread as soon as the current one
  is served, into host memory allocated once (pinned when the buffer lives on
  the GPU), so a refill is one copy into the existing storage;
* both preserve exactly which rows are chosen and the order tokens are served
  in — the selection and the permutation are drawn in the same order from the
  same generators, whichever thread draws them.

The prepared copy costs a buffer's worth of host RAM. When the caller's RAM
budget cannot hold it, refills stay synchronous (still parallel) and say why.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

Key = Tuple[int, str]

STORAGE_MODES = ("gpu_all", "gpu_rolling", "cpu_all", "cpu_rolling")

READ_THREADS_ENV = "ACTIVATION_READ_THREADS"
PREFETCH_ENV = "ACTIVATION_PREFETCH"
#: 12 threads read random extraction rows at 1,560 MB/s on the node's NVMe
#: against 457 MB/s sequentially (2026-09-14); past that the disk, not the CPU,
#: is the limit.
DEFAULT_READ_THREADS = 12

#: One row's read: (source index, row, positions in the row, lo, hi) where
#: [lo, hi) is where the row's tokens sit in the unshuffled concatenation.
RowRead = Tuple[int, int, Union[slice, np.ndarray], int, int]


def plan_activation_storage(
    total_tokens: int,
    gpu_capacity_tokens: int,
    min_useful_tokens: int,
    ram_capacity_tokens: int,
) -> Tuple[str, int]:
    """Where the training pool lives, and how many tokens it holds at once.

    Returns ``(mode, capacity)``:
      * ``gpu_all``      — everything fits on the GPU; load it once.
      * ``gpu_rolling``  — the GPU holds a useful buffer but not the pool; cycle.
      * ``cpu_all``      — the GPU cannot hold a useful buffer; the pool fits in
                           pinned RAM.
      * ``cpu_rolling``  — neither holds the pool; cycle a RAM-sized buffer.

    The GPU is preferred whenever it can hold ``min_useful_tokens``, matching the
    previous rule, so small runs behave exactly as before.
    """
    total_tokens = max(0, int(total_tokens))
    gpu_capacity_tokens = max(0, int(gpu_capacity_tokens))
    ram_capacity_tokens = max(0, int(ram_capacity_tokens))
    if gpu_capacity_tokens >= min_useful_tokens:
        if total_tokens <= gpu_capacity_tokens:
            return "gpu_all", total_tokens
        return "gpu_rolling", gpu_capacity_tokens
    if total_tokens <= ram_capacity_tokens:
        return "cpu_all", total_tokens
    if ram_capacity_tokens >= min_useful_tokens:
        return "cpu_rolling", ram_capacity_tokens
    raise ValueError(
        f"Cannot hold a useful training buffer: {min_useful_tokens:,} tokens needed, "
        f"GPU holds {gpu_capacity_tokens:,} and RAM {ram_capacity_tokens:,}. Reduce the "
        f"batch size or the number of layers trained together."
    )


def read_thread_count(
    env: Optional[Mapping[str, str]] = None, cpu_count: Optional[int] = None
) -> int:
    """Threads reading extraction rows: ``ACTIVATION_READ_THREADS``, capped by the CPUs."""
    env = os.environ if env is None else env
    raw = env.get(READ_THREADS_ENV)
    wanted = DEFAULT_READ_THREADS
    if raw is not None and str(raw).strip() != "":
        try:
            wanted = int(raw)
        except ValueError:
            wanted = 0
        if wanted < 1:
            logger.warning(
                "%s=%r is not a positive integer; using %d read threads",
                READ_THREADS_ENV, raw, DEFAULT_READ_THREADS,
            )
            wanted = DEFAULT_READ_THREADS
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, min(wanted, int(cpus)))


def prefetch_requested(env: Optional[Mapping[str, str]] = None) -> bool:
    """``ACTIVATION_PREFETCH`` switches the background refill off (``0``/``false``/``off``/``no``)."""
    env = os.environ if env is None else env
    raw = str(env.get(PREFETCH_ENV, "1")).strip().lower()
    return raw not in ("0", "false", "off", "no")


def pinned_block_bytes(nbytes: int) -> int:
    """What PyTorch really pins for a request of ``nbytes``: the next power of two.

    The CUDA caching host allocator rounds every pinned allocation up (torch
    2.9.1, ``ATen/core/CachingHostAllocator.h``, ``allocate``: ``roundSize =
    PowerOf2Ceil(size)``). One layer of the 3-layer LFM2.5 buffer measured on
    2026-09-14 is 657,408 x 2,048 x 4 bytes = 5.02 GiB, which pins an 8 GiB
    block, so the three layers pin 24 GiB, not the 15 GiB a byte count predicts.
    """
    nbytes = int(nbytes)
    return 0 if nbytes <= 0 else 1 << (nbytes - 1).bit_length()


def pin_within_budget(
    tensor: torch.Tensor, allowed_bytes: Optional[int], what: str
) -> Tuple[torch.Tensor, bool]:
    """``tensor`` copied into pinned memory when PyTorch's ROUNDED block fits ``allowed_bytes``.

    Returns ``(tensor, pinned)``. The tensor comes back unchanged — pageable — when
    the block :func:`pinned_block_bytes` would pin exceeds the allowance, or when
    pinning fails. The fixed pool (``cpu_all``) pinned every layer unconditionally,
    so a layer sized to the storage plan's per-layer RAM allowance could lock up to
    TWICE it, unswappable, for every layer — the defect review round 1 fixed for the
    rolling buffer's host memory and left in its sibling (review round 2).
    """
    nbytes = tensor.numel() * tensor.element_size()
    block = pinned_block_bytes(nbytes)
    if allowed_bytes is not None and block > int(allowed_bytes):
        logger.warning(
            "Not pinning the %s: %.2f GiB pins a %.2f GiB block, over the %.2f GiB host "
            "allowance per layer; using pageable memory",
            what, nbytes / 1024**3, block / 1024**3, int(allowed_bytes) / 1024**3,
        )
        return tensor, False
    try:
        return tensor.pin_memory(), True
    except RuntimeError as exc:
        logger.warning(
            "Could not pin %.2f GiB for the %s (%s); using pageable memory",
            nbytes / 1024**3, what, exc,
        )
        return tensor, False


def plan_prefetch(
    requested: bool,
    host_ram_tokens: Optional[int],
    buffer_tokens: int,
    storage_on_host: bool,
    bytes_per_token: Optional[int] = None,
) -> Tuple[bool, str]:
    """Whether the next buffer may be prepared ahead, and why (per layer, in tokens).

    The prepared copy is a second buffer's worth of host memory. When the buffer
    itself already lives in host RAM (``cpu_rolling``) that is TWO buffers
    against the same budget, and the storage plan already sized the buffer to
    the whole budget — so it cannot fit, and refills stay synchronous.

    When the copy is PINNED (the buffer lives on the GPU), pass
    ``bytes_per_token``: the copy costs the rounded block
    (:func:`pinned_block_bytes`), up to twice its size, and pinned memory can be
    neither swapped nor reclaimed. Budgeting the unrounded size approved 24 GiB
    of locked RAM against a 15 GiB allowance.

    No budget means no prefetch: holding an unbounded second copy is the one
    failure here that takes the worker down with it.
    """
    if not requested:
        return False, f"switched off ({PREFETCH_ENV} or prefetch=False)"
    if host_ram_tokens is None:
        return False, "no host RAM budget was given, so a second copy cannot be bounded"
    if storage_on_host:
        needed, what = int(buffer_tokens) * 2, "buffer + copy, both in RAM"
    elif bytes_per_token:
        per_token = int(bytes_per_token)
        needed = -(-pinned_block_bytes(int(buffer_tokens) * per_token) // per_token)
        what = "one pinned buffer, rounded up to a power of two"
    else:
        needed, what = int(buffer_tokens), "one buffer"
    if needed > int(host_ram_tokens):
        return False, (
            f"host RAM holds {int(host_ram_tokens):,} tokens per layer and a prepared copy "
            f"needs {needed:,} ({what})"
        )
    return True, (
        f"a prepared copy needs {needed:,} of {int(host_ram_tokens):,} host tokens per layer ({what})"
    )


def refill_capacity(sources: Sequence["BufferSource"], quotas: Sequence[int]) -> int:
    """The most tokens one refill can select, so the buffer is allocated ONCE.

    :meth:`RowCycle.take` always takes at least one whole row, so a source whose
    quota is smaller than its largest row can exceed the quota by up to that
    row. Storage sized to the quotas alone was reallocated at the first refill
    that drew such a row — on the GPU, while the training loop still held views
    of the old buffer, which is the refill OOM that c955dfc0 set out to end.
    """
    total = 0
    for src, quota in zip(sources, quotas, strict=True):
        quota = int(quota)
        if quota <= 0 or src.row_counts.size == 0:
            continue
        total += max(quota, int(src.row_counts.max()))
    return total


def _empty_pinned_host_cache(device: Optional[torch.device] = None) -> None:
    """Hand cached pinned host blocks back to the OS, from ``device``'s CUDA context.

    Freeing a pinned tensor returns its block to PyTorch's host cache, which
    keeps it locked for reuse; ``torch.cuda.empty_cache()`` releases DEVICE
    memory only. So a finished training's 24 GiB of pinned staging stayed
    locked inside the solo worker, under whatever job it ran next. torch 2.9.1
    exposes the host release only as the private ``torch._C._host_emptyCache``.
    It releases every unused cached pinned block in the process; in a solo
    worker that is this training's.

    PASS THE JOB'S CARD. The release runs against the calling thread's CURRENT
    device, and a thread that never selected a card has device 0 current: there
    it CREATES a CUDA context. Measured on the node (2026-09-14): 253 MiB appeared
    on cuda:0 when the claim's release thread ran this after a training on
    cuda:1 — memory on a card no lease of that job covered. ``None`` keeps the
    calling thread's current device, which is right only on the thread that
    placed the job.
    """
    release = getattr(torch._C, "_host_emptyCache", None)
    if release is None:
        logger.warning("This torch has no _host_emptyCache; cached pinned memory stays locked")
        return
    try:
        if device is not None and device.type == "cuda":
            with torch.cuda.device(device):
                release()
        else:
            release()
    except Exception as exc:  # noqa: BLE001 - shutdown must finish
        logger.warning("Could not release cached pinned host memory: %r", exc)


def _cuda_device_of(*devices: torch.device) -> Optional[torch.device]:
    """The first CUDA device among ``devices``: the card a buffer's memory belongs to."""
    return next((device for device in devices if device is not None and device.type == "cuda"), None)


def plan_row_reads(source: int, flat_sorted: np.ndarray, seq_len: int, offset: int) -> List[RowRead]:
    """One read per row for ascending flat positions, in concatenation order.

    The same grouping ``activation_mask.gather_tokens`` uses, so the rows come
    out in exactly the order that function returns them. A row whose positions
    are one unbroken run is read as a slice — one copy instead of a fancy take.
    """
    flat_sorted = np.asarray(flat_sorted, dtype=np.int64)
    if flat_sorted.size == 0:
        return []
    samples = flat_sorted // seq_len
    tokens = flat_sorted % seq_len
    boundaries = np.flatnonzero(np.diff(samples)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [flat_sorted.size]))
    reads: List[RowRead] = []
    for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
        toks = tokens[a:b]
        first = int(toks[0])
        if toks.size == 1 or bool((np.diff(toks) == 1).all()):
            positions: Union[slice, np.ndarray] = slice(first, first + int(toks.size))
        else:
            positions = toks
        reads.append((source, int(samples[a]), positions, offset + a, offset + b))
    return reads


@dataclass
class BufferSource:
    """One extraction: a file per (layer, hook) and which positions are trainable.

    ``valid_flat`` is the flat ``(row * seq_len + pos)`` index of every trainable
    position — padding and held-out documents already removed — or ``None`` when
    every position is trainable.
    """

    label: str
    files: Dict[Key, Path]
    num_rows: int
    seq_len: int
    valid_flat: Optional[np.ndarray] = None
    rows: np.ndarray = field(init=False, repr=False)
    row_counts: np.ndarray = field(init=False, repr=False)
    _starts: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.valid_flat is None:
            self.rows = np.arange(self.num_rows, dtype=np.int64)
            self.row_counts = np.full(self.num_rows, self.seq_len, dtype=np.int64)
            return
        flat = np.sort(np.asarray(self.valid_flat, dtype=np.int64))
        self.valid_flat = flat
        self.rows, self._starts, self.row_counts = np.unique(
            flat // self.seq_len, return_index=True, return_counts=True
        )

    @property
    def total_tokens(self) -> int:
        return int(self.row_counts.sum())

    def flat_positions(self, chosen: np.ndarray) -> np.ndarray:
        """Flat indices of the trainable tokens in the chosen rows (indices into ``rows``)."""
        chosen = np.asarray(chosen, dtype=np.int64)
        if chosen.size == 0:
            return np.empty(0, dtype=np.int64)
        if self.valid_flat is None:
            starts = self.rows[chosen] * self.seq_len
            return (starts[:, None] + np.arange(self.seq_len, dtype=np.int64)[None, :]).reshape(-1)
        return np.concatenate([
            self.valid_flat[self._starts[i]: self._starts[i] + self.row_counts[i]]
            for i in chosen
        ])


class RowCycle:
    """A source's rows in shuffled order, served without repetition.

    Rows are reshuffled only after every row has been served. When that happens
    part-way through a refill, rows already taken for the refill are moved to the
    end of the new order, so no row appears twice in one buffer.
    """

    def __init__(self, n_rows: int, rng: np.random.Generator) -> None:
        self.n = int(n_rows)
        self._rng = rng
        self.order = rng.permutation(self.n)
        self.cursor = 0
        self.epochs_completed = 0

    def take(self, counts: np.ndarray, quota: int) -> np.ndarray:
        """Rows whose token counts sum to at most ``quota`` (at least one row)."""
        if self.n == 0 or quota <= 0:
            return np.empty(0, dtype=np.int64)
        chosen: List[int] = []
        got = 0
        while len(chosen) < self.n:
            if self.cursor == self.n:
                new = self._rng.permutation(self.n)
                if chosen:
                    already = np.isin(new, chosen)
                    new = np.concatenate([new[~already], new[already]])
                self.order, self.cursor = new, 0
                self.epochs_completed += 1
            row = int(self.order[self.cursor])
            if chosen and got + int(counts[row]) > quota:
                break
            chosen.append(row)
            got += int(counts[row])
            self.cursor += 1
            if got >= quota:
                break
        return np.asarray(chosen, dtype=np.int64)


class _Stopped(Exception):
    """A read abandoned because the buffer is closing."""


@dataclass
class _Selection:
    """Everything a refill needs except the bytes: which rows, and where each token goes."""

    total: int
    tokens_per_source: List[int]
    epochs: List[int]
    reads: List[RowRead]
    #: ``inverse[i]`` is the buffer row of the i-th token of the unshuffled
    #: concatenation — the inverse of the served permutation.
    inverse: np.ndarray
    select_s: float
    shuffle_s: float
    gather_s: float = 0.0
    #: Every row has been read into the prepared copy (prefetch only). A selection
    #: whose read failed stays current with this False, and is read again.
    gathered: bool = False
    #: The row cycles' and the permutation generator's state just BEFORE this
    #: selection was drawn. Selecting again from it reproduces this selection
    #: exactly; see :meth:`RollingActivationBuffer.state_dict`.
    snapshot: Optional[dict] = None


class RollingActivationBuffer:
    """Cycle every source's tokens through a fixed-size training buffer.

    ``tensors`` is updated IN PLACE on every refill, so a caller holding the dict
    (the training loop's ``cached_activations``) always sees the current buffer.

    Call :meth:`close` when training ends, however it ends: it stops the
    background reads and joins their threads.
    """

    def __init__(
        self,
        sources: Sequence[BufferSource],
        keys: Sequence[Key],
        quotas: Sequence[int],
        *,
        seed: int,
        storage_device: torch.device,
        train_device: torch.device,
        pin_memory: bool = False,
        host_ram_tokens: Optional[int] = None,
        prefetch: Optional[bool] = None,
        read_threads: Optional[int] = None,
    ) -> None:
        if len(quotas) != len(sources):
            raise ValueError(f"{len(quotas)} quotas for {len(sources)} sources")
        if sum(int(q) for q in quotas) <= 0:
            raise ValueError("every quota is zero; the buffer would hold nothing")
        for src in sources:
            missing = [k for k in keys if k not in src.files]
            if missing:
                raise ValueError(f"source {src.label} has no file for {missing}")
        self.sources = list(sources)
        self.keys = list(keys)
        self.quotas = [int(q) for q in quotas]
        self.storage_device = storage_device
        self.train_device = train_device
        self.pin_memory = pin_memory
        self._cycles = [
            RowCycle(len(src.rows), np.random.default_rng([int(seed), i]))
            for i, src in enumerate(self.sources)
        ]
        self._torch_gen = torch.Generator().manual_seed(int(seed))

        # Opened up front: the read threads share this dict and must never race
        # to insert into it.
        self._mmaps: Dict[Tuple[int, Key], np.ndarray] = {}
        dims = set()
        for i, src in enumerate(self.sources):
            for key in self.keys:
                mm = np.load(src.files[key], mmap_mode="r")
                if mm.ndim != 3 or mm.shape[1] != src.seq_len:
                    raise ValueError(
                        f"{src.files[key]} is shaped {mm.shape}; expected (rows, {src.seq_len}, d)"
                    )
                self._mmaps[(i, key)] = mm
                dims.add(int(mm.shape[2]))
        if len(dims) != 1:
            raise ValueError(f"sources disagree on the activation width: {sorted(dims)}")
        self.hidden_dim = dims.pop()

        #: Each key's buffer memory, allocated once and refilled in place;
        #: ``tensors[key]`` is a view of its first ``size`` rows.
        self._storage: Dict[Key, Optional[torch.Tensor]] = {}
        self.tensors: Dict[Key, Optional[torch.Tensor]] = {}
        self.size = 0
        self._pos = 0
        self.refills = 0
        self.tokens_loaded = [0] * len(self.sources)
        self.tokens_dropped = 0
        self.last_refill_timings: Dict[str, object] = {}
        #: Source passes completed as of the buffer being SERVED — the cycles
        #: themselves run a buffer ahead while the next one is prepared.
        self._served_epochs = [0] * len(self.sources)

        self.read_threads = int(read_threads) if read_threads else read_thread_count()
        self._stop = threading.Event()
        self._closed = False
        #: The selection being made current: set when it is selected, cleared only
        #: when a refill SERVES it, so a refill that fails is retried on the same rows.
        self._next: Optional[_Selection] = None
        self._pool = ThreadPoolExecutor(
            max_workers=self.read_threads, thread_name_prefix="activation-read"
        )
        #: A transfer of a batch out of host storage that may still be queued on
        #: the GPU; a refill waits for it before overwriting those rows.
        self._pending_transfer = None

        #: Host RAM allowance per layer, in tokens: bounds the prepared copy and
        #: whether host memory may be pinned (pinned blocks are rounded up).
        self.host_ram_tokens = None if host_ram_tokens is None else int(host_ram_tokens)
        #: Pinned host bytes this buffer allocated (rounded blocks); close()
        #: releases PyTorch's pinned cache when non-zero.
        self._pinned_bytes = 0
        #: Every allocation is sized to this, so no refill ever reallocates.
        self.capacity = refill_capacity(self.sources, self.quotas)
        capacity = self.capacity
        self.prefetch, self.prefetch_reason = plan_prefetch(
            prefetch_requested() if prefetch is None else bool(prefetch),
            self.host_ram_tokens,
            capacity,
            storage_on_host=storage_device.type == "cpu",
            bytes_per_token=self.hidden_dim * 4 if self._staging_pinned() else None,
        )
        self._staging: Dict[Key, torch.Tensor] = {}
        self._prefetcher: Optional[ThreadPoolExecutor] = None
        self._pending: Optional[Future] = None
        if self.prefetch:
            self._prefetcher = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="activation-prefetch"
            )
            self._allocate_staging(capacity)
        logger.info(
            "Activation buffer: %s tokens per layer x %d layers (quotas %s), %d read threads, "
            "prefetch %s — %s; pinned host memory %.2f GiB",
            f"{capacity:,}", len(self.keys), f"{sum(self.quotas):,}", self.read_threads,
            "ON" if self.prefetch else "OFF (refills are synchronous)", self.prefetch_reason,
            self._pinned_bytes / 1024**3,
        )
        try:
            self.refill()
        except BaseException:
            self.close()
            raise

    # ── memory ────────────────────────────────────────────────────────────

    def _mmap(self, i: int, key: Key) -> np.ndarray:
        return self._mmaps[(i, key)]

    def _staging_pinned(self) -> bool:
        """The prepared copy is pinned when the buffer lives on the GPU: pageable
        memory is copied through a pinned bounce buffer, which would add a second
        host copy to every refill's ~15 GB transfer."""
        return self.storage_device.type == "cuda"

    def _allocate_host(self, rows: int, *, pin: bool, what: str) -> torch.Tensor:
        """Host memory for ``rows`` activations: pinned when asked AND affordable.

        A pinned block is rounded up to a power of two (:func:`pinned_block_bytes`)
        and cannot be swapped or reclaimed. cpu_rolling's storage is sized to the
        whole host budget, so pinning it rounded could lock up to ALL of the
        memory the plan measured as available — it is kept pageable instead,
        which only makes each batch's copy to the GPU synchronous. And when
        pinning fails outright, pageable memory is used rather than failing the
        training.
        """
        shape = (int(rows), self.hidden_dim)
        nbytes = shape[0] * shape[1] * 4
        if pin and self.host_ram_tokens is not None:
            allowed = self.host_ram_tokens * self.hidden_dim * 4
            if pinned_block_bytes(nbytes) > allowed:
                logger.warning(
                    "Not pinning the %s: %.2f GiB pins a %.2f GiB block, over the %.2f GiB "
                    "host allowance per layer; using pageable memory",
                    what, nbytes / 1024**3, pinned_block_bytes(nbytes) / 1024**3, allowed / 1024**3,
                )
                pin = False
        if pin:
            try:
                tensor = torch.empty(shape, dtype=torch.float32, pin_memory=True)
            except RuntimeError as exc:
                logger.warning("Could not pin %.2f GiB for the %s (%s); using pageable memory",
                               nbytes / 1024**3, what, exc)
            else:
                self._pinned_bytes += pinned_block_bytes(nbytes)
                return tensor
        return torch.empty(shape, dtype=torch.float32)

    def _allocate_staging(self, capacity: int) -> None:
        """Host memory for the prepared buffer, allocated once."""
        for key in self.keys:
            self._staging.pop(key, None)
            self._staging[key] = self._allocate_host(
                capacity, pin=self._staging_pinned(), what="prepared copy"
            )

    def _storage_for(self, key: Key, total: int) -> torch.Tensor:
        storage = self._storage.get(key)
        if storage is None or storage.shape[0] < total:
            # First refill. (A larger refill cannot happen: every allocation is
            # sized to refill_capacity. The branch stays as a release-first
            # fallback, not as a path any refill is expected to take.)
            self.tensors[key] = None
            self._storage[key] = None
            del storage
            if self.storage_device.type == "cuda":
                torch.cuda.empty_cache()
            rows = max(total, self.capacity)
            if self.storage_device.type == "cpu":
                storage = self._allocate_host(rows, pin=self.pin_memory, what="activation buffer")
            else:
                storage = torch.empty(
                    (rows, self.hidden_dim), dtype=torch.float32, device=self.storage_device
                )
            self._storage[key] = storage
        return storage

    def _await_pending_transfer(self) -> None:
        """Wait for the GPU to finish copying the last batch out of host storage.

        A batch served from pinned host storage is copied with
        ``non_blocking=True``, so the copy may still be queued when the next draw
        triggers a refill — and a refill writes new rows into that same memory.
        Without this the GPU could read half-overwritten rows, silently.
        """
        marker, self._pending_transfer = self._pending_transfer, None
        if marker is not None:
            marker.synchronize()

    @staticmethod
    def _transfer_marker(device: torch.device):
        if device.type != "cuda":
            return None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        return event

    # ── selection and reads ───────────────────────────────────────────────

    def epochs_completed(self) -> List[int]:
        return list(self._served_epochs)

    def _selection_state(self) -> dict:
        """Everything :meth:`_select` draws from, as plain values and CPU tensors."""
        import json

        return {
            "torch_generator": self._torch_gen.get_state(),
            "cycles": [
                {
                    "order": torch.from_numpy(np.asarray(cycle.order, dtype=np.int64).copy()),
                    "cursor": int(cycle.cursor),
                    "epochs_completed": int(cycle.epochs_completed),
                    # A PCG64 state holds 128-bit integers; JSON keeps them exact.
                    "rng": json.dumps(cycle._rng.bit_generator.state),
                }
                for cycle in self._cycles
            ],
        }

    def _select(self) -> _Selection:
        """Choose the next buffer's rows and its permutation.

        Always called in refill order and never concurrently, so the generators
        are consumed in the same sequence with prefetch on or off.
        """
        started = time.monotonic()
        snapshot = self._selection_state()
        per_source: List[np.ndarray] = []
        for src, cycle, quota in zip(self.sources, self._cycles, self.quotas, strict=True):
            chosen = cycle.take(src.row_counts, quota)
            per_source.append(np.sort(src.flat_positions(chosen)))
        total = int(sum(f.size for f in per_source))
        if total == 0:
            raise ValueError("a refill selected no tokens")
        reads: List[RowRead] = []
        offset = 0
        for i, flat in enumerate(per_source):
            reads.extend(plan_row_reads(i, flat, self.sources[i].seq_len, offset))
            offset += int(flat.size)
        selected = time.monotonic()

        order = torch.randperm(total, generator=self._torch_gen).numpy()
        inverse = np.empty(total, dtype=np.int64)
        inverse[order] = np.arange(total, dtype=np.int64)
        return _Selection(
            total=total,
            tokens_per_source=[int(f.size) for f in per_source],
            epochs=[c.epochs_completed for c in self._cycles],
            reads=reads,
            inverse=inverse,
            select_s=selected - started,
            shuffle_s=time.monotonic() - selected,
            snapshot=snapshot,
        )

    def _read_row(
        self,
        key: Key,
        dest: np.ndarray,
        inverse: np.ndarray,
        read: RowRead,
        abort: threading.Event,
    ) -> None:
        # Checked per row, so closing (or a failed row) never waits for more than
        # the rows already being copied (a few MB each), not for the rest of a
        # 15 GB refill.
        if self._stop.is_set() or abort.is_set():
            raise _Stopped()
        i, row, positions, lo, hi = read
        dest[inverse[lo:hi]] = self._mmaps[(i, key)][row, positions]

    def _read_guarded(
        self,
        key: Key,
        dest: np.ndarray,
        inverse: np.ndarray,
        read: RowRead,
        abort: threading.Event,
    ) -> None:
        try:
            self._read_row(key, dest, inverse, read, abort)
        except BaseException:
            # Set by the failing row itself, before its thread takes another
            # task. Set by the waiting thread instead, a single read thread got
            # through 49 more rows first (measured by the test that pins this).
            abort.set()
            raise

    def _gather(self, selection: _Selection, dests: Dict[Key, np.ndarray]) -> float:
        """Read every row for the given keys into ``dests``, shuffled. Returns seconds."""
        started = time.monotonic()
        abort = threading.Event()
        futures = [
            self._pool.submit(self._read_guarded, key, dest, selection.inverse, read, abort)
            for read in selection.reads
            for key, dest in dests.items()
        ]
        # After a failure the rows not yet started skip themselves, and the rows
        # still being copied finish BEFORE this returns — the memory they write
        # is refilled or served next. Not Future.cancel(): wait() on a future
        # cancelled that way returns only once a worker dequeues it, and never
        # if the pool is shut down with cancel_futures (a mutation control hung
        # the suite on exactly that).
        wait(futures)
        errors = [f.exception() for f in futures if f.exception() is not None]
        real = [e for e in errors if not isinstance(e, _Stopped)]
        if real:
            raise real[0]
        if errors:
            raise errors[0]
        return time.monotonic() - started

    def _prepare_next(self) -> _Selection:
        """Select the next buffer (unless a failed refill left one) and read it into staging.

        The selection is recorded in ``_next`` BEFORE its rows are read. Rows are
        taken from the cycles when they are selected, so a read that fails must
        leave that selection for the next refill to read again — a new selection
        would skip the failed one's rows for the whole pass. Runs on the prefetch
        thread, or on the training thread when a refill reads the buffer itself;
        never both at once, because a refill looks at ``_next`` only after the
        background preparation has finished.
        """
        if self._next is None:
            self._next = self._select()
        selection = self._next
        if selection.total > next(iter(self._staging.values())).shape[0]:
            self._allocate_staging(selection.total)
        selection.gather_s = self._gather(
            selection, {key: self._staging[key].numpy() for key in self.keys}
        )
        selection.gathered = True
        return selection

    def _retire_served(self) -> None:
        """The served buffer is spent: count its unserved tail, and serve nothing more from it.

        Called just before a refill writes into the served memory. If that refill
        then fails part-way, a draw small enough to fit the old buffer would serve
        half-overwritten rows, or one layer's new rows beside another's old ones;
        at size zero it refills instead. The tail is counted HERE, once, so a
        retried refill cannot count it again.
        """
        if self.size:
            self.tokens_dropped += self.size - self._pos
        self.size = self._pos = 0

    # ── refill ────────────────────────────────────────────────────────────

    def refill(self) -> None:
        """Make the next buffer current.

        With prefetch this waits for the background preparation (normally
        already finished) and copies it into the storage.

        A REFILL THAT FAILS IS RETRIED, NEVER SKIPPED. Its selection stays in
        ``_next`` whatever failed — a read, the synchronize, the copy — so the
        next refill serves exactly those rows (the training loop's OOM handler
        retries steps). A read that failed in the background is read again here,
        on the training thread: a failure that persists is raised HERE, and one
        that has passed costs a synchronous read. (Until review round 2 a failure
        after the prepared buffer was taken skipped it, a synchronous read failure
        re-selected, and a background read failure was sticky.)
        """
        if self._closed:
            raise RuntimeError("the activation buffer is closed")
        started = time.monotonic()

        wait_s = copy_s = 0.0
        prefetched = False
        if self._pending is not None:
            pending, self._pending = self._pending, None
            waited = time.monotonic()
            failure = pending.exception()
            wait_s = time.monotonic() - waited
            if failure is None:
                prefetched = True
            else:
                # repr, not the exception: a log record keeps its args, and this
                # traceback's frames hold numpy views of the prepared copy.
                logger.warning(
                    "Preparing activation buffer #%d in the background failed (%s); "
                    "reading it again on the training thread", self.refills + 1, repr(failure),
                )
            pending = failure = None

        if self.prefetch:
            if self._next is None or not self._next.gathered:
                self._prepare_next()
            selection = self._next
            copying = time.monotonic()
            self._await_pending_transfer()
            self._retire_served()
            for key in self.keys:
                storage = self._storage_for(key, selection.total)
                storage[:selection.total].copy_(self._staging[key][:selection.total])
                self.tensors[key] = storage[:selection.total]
            copy_s = time.monotonic() - copying
        else:
            if self._next is None:
                self._next = self._select()
            selection = self._next
            selection.gather_s = 0.0
            self._await_pending_transfer()
            self._retire_served()
            for key in self.keys:
                storage = self._storage_for(key, selection.total)
                if storage.device.type == "cpu":
                    # The buffer is spent and batches are copies, so the rows
                    # can be written straight into the served memory.
                    selection.gather_s += self._gather(selection, {key: storage.numpy()})
                else:
                    # One layer at a time, so a synchronous refill never holds
                    # more than one layer's rows in host memory.
                    host = np.empty((selection.total, self.hidden_dim), dtype=np.float32)
                    selection.gather_s += self._gather(selection, {key: host})
                    copying = time.monotonic()
                    storage[:selection.total].copy_(torch.from_numpy(host))
                    copy_s += time.monotonic() - copying
                    del host
                self.tensors[key] = storage[:selection.total]

        self._next = None
        for i, n in enumerate(selection.tokens_per_source):
            self.tokens_loaded[i] += n
        self.size, self._pos = selection.total, 0
        self.refills += 1
        self._served_epochs = list(selection.epochs)
        #: What state_dict() saves: the generators as they stood before THIS
        #: buffer was drawn (the cycles themselves may already be a buffer ahead).
        self._served_snapshot = selection.snapshot
        if self.prefetch:
            self._pending = self._prefetcher.submit(self._prepare_next)

        stall_s = time.monotonic() - started
        self.last_refill_timings = {
            "stall_s": stall_s, "wait_s": wait_s, "copy_s": copy_s,
            "select_s": selection.select_s, "shuffle_s": selection.shuffle_s,
            "gather_s": selection.gather_s, "prefetched": prefetched,
        }
        logger.info(
            "Activation buffer refill #%d: %s tokens [%s] in %.1fs "
            "(wait %.1fs, copy %.1fs; %s: select %.2fs, shuffle %.2fs, gather %.1fs, "
            "%d read threads); source passes completed %s",
            self.refills, f"{selection.total:,}",
            ", ".join(
                f"{s.label}={n:,}"
                for s, n in zip(self.sources, selection.tokens_per_source, strict=True)
            ),
            stall_s, wait_s, copy_s,
            "prepared in background" if prefetched else "prepared on this thread",
            selection.select_s, selection.shuffle_s, selection.gather_s,
            self.read_threads, self.epochs_completed(),
        )

    def next_batch(self, batch_size: int) -> Dict[Key, torch.Tensor]:
        """The next ``batch_size`` tokens for every key, never repeating within a buffer."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._pos + batch_size > self.size:
            self.refill()
            if batch_size > self.size:
                raise ValueError(
                    f"batch_size {batch_size:,} exceeds the buffer ({self.size:,} tokens)"
                )
        lo, hi = self._pos, self._pos + batch_size
        out: Dict[Key, torch.Tensor] = {}
        transferred = False
        for key in self.keys:
            batch = self.tensors[key][lo:hi]
            if batch.device != self.train_device:
                batch = batch.to(self.train_device, non_blocking=True)
                transferred = True
            else:
                # A COPY, NOT A VIEW. A slice keeps the WHOLE buffer alive, and
                # the training loop still holds the previous step's batch when
                # the next step's draw triggers a refill — so "release before
                # loading" released nothing. On 2026-09-14 a 3-layer JumpReLU run
                # on the 3090 (gpu_rolling, ~15 GB of buffer) died at step 318,
                # the first refill: "Tried to allocate 4.97 GiB ... 18.13 GiB
                # allocated by PyTorch". Halving the batch cannot fix that, so
                # the OOM retries failed too. A copy is batch_size rows per layer.
                batch = batch.clone()
            out[key] = batch
        # Advanced only once every layer's copy exists. A copy can fail — a CUDA
        # OOM on the clone or the transfer, which the training loop's handler
        # retries with half the batch — and a position moved first skipped those
        # tokens for the whole pass without counting them (review round 2).
        self._pos = hi
        if transferred:
            self._pending_transfer = self._transfer_marker(self.train_device)
        return out

    # ── shutdown ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop background reads, join every thread, and release the buffer's memory.

        Idempotent. Rows already being copied finish (a few MB each); nothing
        waits for the rest of a refill, so a cancelled training stops promptly.

        The storage, the served views and the prepared copy are dropped HERE,
        not whenever the buffer object happens to die, and PyTorch's pinned host
        cache is emptied afterwards: freeing a pinned tensor only returns its
        block to that cache, where it stays locked under the worker's next job.
        Memory another object still references is not freed — nothing is
        released out from under a live view.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        pending, self._pending = self._pending, None
        if self._prefetcher is not None:
            self._prefetcher.shutdown(wait=True)
        # No cancel_futures: queued reads see the stop flag and return at once,
        # while a future cancelled by shutdown never wakes a wait() on it.
        self._pool.shutdown(wait=True)
        if pending is not None and pending.done() and not pending.cancelled():
            exc = pending.exception()
            if exc is not None and not isinstance(exc, _Stopped):
                logger.warning(
                    "The activation buffer being prepared when training ended failed: %s",
                    repr(exc),
                )
            exc = None
        # THE FUTURE GOES BEFORE THE CACHE IS EMPTIED. A preparation stopped by this
        # close (the normal case within seconds of a refill) or failed carries a
        # traceback whose frames hold numpy views of the prepared copy — a view
        # holds the STORAGE, not the tensor object — so emptying the cache while
        # this local still held it released nothing (review round 2).
        pending = None
        self._next = None
        self._staging.clear()
        self._storage.clear()
        self.tensors.clear()
        self.size = self._pos = 0
        if self._pinned_bytes:
            # Those frames, and a failed read's, sit in reference cycles with their
            # exceptions: only a collection frees what they hold.
            gc.collect()
            # From the training's own card: close() also runs on the claim's
            # release thread, whose current device is cuda:0.
            _empty_pinned_host_cache(device=_cuda_device_of(self.storage_device, self.train_device))

    # ── resume (tracker item 4) ───────────────────────────────────────────

    STATE_KIND = "rolling_activation_buffer"
    STATE_VERSION = 1

    def _identity(self) -> dict:
        return {
            "labels": [src.label for src in self.sources],
            "keys": [[int(layer), str(hook)] for layer, hook in self.keys],
            "quotas": [int(q) for q in self.quotas],
        }

    def state_dict(self) -> dict:
        """Where this buffer stands in the pool: JSON-style values and CPU tensors only.

        What is saved is the generators' state from BEFORE the current buffer was
        drawn, and the position within that buffer. Not the generators as they
        are now: the background preparation may already have drawn the NEXT
        buffer from them, and a resumed run must draw it again, not skip it.
        """
        snapshot = getattr(self, "_served_snapshot", None)
        if snapshot is None or self.size == 0:
            raise RuntimeError("the activation buffer has not served a buffer to record a position in")
        return {
            "kind": self.STATE_KIND,
            "version": self.STATE_VERSION,
            **self._identity(),
            "selection_state": snapshot,
            "position": int(self._pos),
            "size": int(self.size),
            "refills": int(self.refills),
            "tokens_loaded": [int(n) for n in self.tokens_loaded],
            "tokens_dropped": int(self.tokens_dropped),
        }

    def load_state_dict(self, state: dict) -> None:
        """Serve, from the next batch on, exactly what the saved buffer would have served.

        The buffer must be built over the same sources, keys and quotas: another
        mixture would select other rows, and that is refused rather than served.
        The saved buffer is drawn again from the restored generators and read
        from disk, and the preparation of the one after it starts as usual.
        """
        if self._closed:
            raise RuntimeError("the activation buffer is closed")
        if state.get("kind") != self.STATE_KIND or int(state.get("version", 0)) != self.STATE_VERSION:
            raise ValueError(f"not a {self.STATE_KIND} v{self.STATE_VERSION} state: {state.get('kind')!r}")
        for name, value in self._identity().items():
            if state.get(name) != value:
                raise ValueError(
                    f"the saved buffer's {name} {state.get(name)} differ from this buffer's {value}; "
                    "resuming would serve a different mixture"
                )
        # The preparation in flight drew from the generators about to be replaced.
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.exception()  # waits for it; whatever it read is discarded
            pending = None
        self._next = None
        self._restore_selection_state(state["selection_state"])
        self.refill()
        if self.size != int(state["size"]):
            raise ValueError(
                f"the restored buffer holds {self.size:,} tokens and the saved one held "
                f"{int(state['size']):,}; the extraction is not the one the run trained on"
            )
        self._pos = int(state["position"])
        self.refills = int(state["refills"])
        self.tokens_loaded = [int(n) for n in state["tokens_loaded"]]
        self.tokens_dropped = int(state["tokens_dropped"])

    def load_state_dict_after_replan(self, state: dict) -> int:
        """Continue a saved pass under DIFFERENT quotas. Returns the tokens skipped.

        For a resume whose saved storage plan no longer fits this process
        (services/activation_plan.py, review round 1 R1D-1). :meth:`load_state_dict`
        refuses other quotas, because re-drawing the served buffer under them would
        select other rows. This re-draws the served buffer under its OWN saved quotas
        — the selection only; nothing is read — which moves the row cycles and the
        permutation generator to exactly where they stood after it, and fills the
        next buffer under this buffer's quotas from there.

        WHY NO ROW REPEATS. A row is served only when ``RowCycle.take`` moves the
        cycle's cursor past it; the cursor walks one permutation of the source's
        rows and reshuffles only once every row has been taken. A quota decides how
        FAR one take moves the cursor, never which rows remain or in what order. So
        a pass whose quotas change between two refills still takes each row at most
        once. What this costs is the interrupted buffer's unserved tail: its rows
        were taken, so its unserved tokens wait for the source's next pass. They are
        counted in ``tokens_dropped`` as any spent buffer's tail is. The held-out
        split is untouched: its positions were removed from the sources before this
        buffer was built.
        """
        if self._closed:
            raise RuntimeError("the activation buffer is closed")
        if state.get("kind") != self.STATE_KIND or int(state.get("version", 0)) != self.STATE_VERSION:
            raise ValueError(f"not a {self.STATE_KIND} v{self.STATE_VERSION} state: {state.get('kind')!r}")
        identity = self._identity()
        for name in ("labels", "keys"):
            if state.get(name) != identity[name]:
                raise ValueError(
                    f"the saved buffer's {name} {state.get(name)} differ from this buffer's {identity[name]}"
                )
        saved_quotas = [int(q) for q in state["quotas"]]
        if len(saved_quotas) != len(self.sources):
            raise ValueError(f"the saved buffer has {len(saved_quotas)} quotas for {len(self.sources)} sources")
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.exception()  # waits for it; whatever it read is discarded
            pending = None
        self._next = None
        self._restore_selection_state(state["selection_state"])
        quotas, self.quotas = self.quotas, saved_quotas
        try:
            served_total = self._select().total
        finally:
            self.quotas = quotas
        if served_total != int(state["size"]):
            raise ValueError(
                f"re-drawing the saved buffer selected {served_total:,} tokens and the saved one held "
                f"{int(state['size']):,}; the extraction is not the one the run trained on"
            )
        skipped = served_total - int(state["position"])
        self.size = self._pos = 0
        self.refills = int(state["refills"])
        self.tokens_loaded = [int(n) for n in state["tokens_loaded"]]
        self.tokens_dropped = int(state["tokens_dropped"]) + skipped
        self.refill()
        return skipped

    def _restore_selection_state(self, snapshot: dict) -> None:
        import json

        cycles = snapshot["cycles"]
        if len(cycles) != len(self._cycles):
            raise ValueError(f"the saved state has {len(cycles)} sources; this buffer has {len(self._cycles)}")
        for cycle, saved in zip(self._cycles, cycles, strict=True):
            if int(saved["order"].numel()) != cycle.n:
                raise ValueError(
                    f"a source has {cycle.n:,} rows and its saved order has {int(saved['order'].numel()):,}"
                )
        self._torch_gen.set_state(snapshot["torch_generator"])
        for cycle, saved in zip(self._cycles, cycles, strict=True):
            cycle.order = saved["order"].numpy().astype(np.int64).copy()
            cycle.cursor = int(saved["cursor"])
            cycle.epochs_completed = int(saved["epochs_completed"])
            cycle._rng.bit_generator.state = json.loads(saved["rng"])


class FixedPoolSampler:
    """Batches drawn uniformly WITH replacement from a pool that fits in memory.

    The ``gpu_all`` and ``cpu_all`` storage modes. Draws come from this object's
    OWN generator rather than torch's global one, so nothing else that consumes
    random numbers (a resample, a calibration) moves the data stream, and
    ``state_dict()`` is the whole of its position — the same resume contract the
    rolling buffer and the on-the-fly source keep.
    """

    STATE_KIND = "fixed_activation_pool"
    STATE_VERSION = 1

    def __init__(self, tensors: Mapping[Key, torch.Tensor], keys: Sequence[Key], *, seed: int) -> None:
        self.tensors = tensors
        self.keys = list(keys)
        sizes = {int(tensors[key].shape[0]) for key in self.keys}
        if len(sizes) != 1:
            raise ValueError(f"layers disagree on the pool size: {sorted(sizes)}")
        self.num_samples = sizes.pop()
        if self.num_samples <= 0:
            raise ValueError("the activation pool is empty")
        self._generator = torch.Generator().manual_seed(int(seed))
        self.batches_drawn = 0

    def next_batch(self, batch_size: int) -> Dict[Key, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = torch.randint(0, self.num_samples, (int(batch_size),), generator=self._generator)
        self.batches_drawn += 1
        return {key: self.tensors[key][indices.to(self.tensors[key].device)] for key in self.keys}

    def state_dict(self) -> dict:
        return {
            "kind": self.STATE_KIND,
            "version": self.STATE_VERSION,
            "num_samples": int(self.num_samples),
            "keys": [[int(layer), str(hook)] for layer, hook in self.keys],
            "generator": self._generator.get_state(),
            "batches_drawn": int(self.batches_drawn),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != self.STATE_KIND or int(state.get("version", 0)) != self.STATE_VERSION:
            raise ValueError(f"not a {self.STATE_KIND} v{self.STATE_VERSION} state: {state.get('kind')!r}")
        if int(state["num_samples"]) != self.num_samples:
            raise ValueError(
                f"the saved pool held {int(state['num_samples']):,} tokens and this one holds "
                f"{self.num_samples:,}; the pool is not the one the run trained on"
            )
        if state["keys"] != [[int(layer), str(hook)] for layer, hook in self.keys]:
            raise ValueError(f"the saved pool is for layers {state['keys']}")
        self._generator.set_state(state["generator"])
        self.batches_drawn = int(state["batches_drawn"])

    def close(self) -> None:
        """Nothing to release: the tensors belong to the caller."""
