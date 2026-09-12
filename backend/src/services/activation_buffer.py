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
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import activation_mask

logger = logging.getLogger(__name__)

Key = Tuple[int, str]

STORAGE_MODES = ("gpu_all", "gpu_rolling", "cpu_all", "cpu_rolling")


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


class RollingActivationBuffer:
    """Cycle every source's tokens through a fixed-size training buffer.

    ``tensors`` is updated IN PLACE on every refill, so a caller holding the dict
    (the training loop's ``cached_activations``) always sees the current buffer.
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
        self._mmaps: Dict[Tuple[int, Key], np.ndarray] = {}
        self.tensors: Dict[Key, Optional[torch.Tensor]] = {}
        self.size = 0
        self._pos = 0
        self.refills = 0
        self.tokens_loaded = [0] * len(self.sources)
        self.tokens_dropped = 0
        self.refill()

    def _mmap(self, i: int, key: Key) -> np.ndarray:
        if (i, key) not in self._mmaps:
            self._mmaps[(i, key)] = np.load(self.sources[i].files[key], mmap_mode="r")
        return self._mmaps[(i, key)]

    def epochs_completed(self) -> List[int]:
        return [c.epochs_completed for c in self._cycles]

    def refill(self) -> None:
        started = time.monotonic()
        if self.size:
            self.tokens_dropped += self.size - self._pos

        per_source: List[np.ndarray] = []
        for i, (src, cycle, quota) in enumerate(zip(self.sources, self._cycles, self.quotas)):
            chosen = cycle.take(src.row_counts, quota)
            per_source.append(np.sort(src.flat_positions(chosen)))
        total = int(sum(f.size for f in per_source))
        if total == 0:
            raise ValueError("a refill selected no tokens")

        # RELEASE BEFORE LOADING. The buffer is sized to the free memory; holding
        # the previous one while the next is built would need twice that.
        for key in self.keys:
            self.tensors[key] = None
        if self.storage_device.type == "cuda":
            torch.cuda.empty_cache()

        order = torch.randperm(total, generator=self._torch_gen)
        for key in self.keys:
            parts = [
                activation_mask.gather_tokens(self._mmap(i, key), flat)
                for i, flat in enumerate(per_source) if flat.size
            ]
            host = torch.from_numpy(parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0))
            host = host[order]
            if self.storage_device.type == "cuda":
                self.tensors[key] = host.to(self.storage_device)
            else:
                self.tensors[key] = host.pin_memory() if self.pin_memory else host
            del host, parts

        for i, flat in enumerate(per_source):
            self.tokens_loaded[i] += int(flat.size)
        self.size, self._pos = total, 0
        self.refills += 1
        logger.info(
            "Activation buffer refill #%d: %s tokens [%s] in %.1fs; source passes completed %s",
            self.refills, f"{total:,}",
            ", ".join(f"{s.label}={f.size:,}" for s, f in zip(self.sources, per_source)),
            time.monotonic() - started, self.epochs_completed(),
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
        self._pos = hi
        out: Dict[Key, torch.Tensor] = {}
        for key in self.keys:
            batch = self.tensors[key][lo:hi]
            if batch.device != self.train_device:
                batch = batch.to(self.train_device, non_blocking=True)
            out[key] = batch
        return out
