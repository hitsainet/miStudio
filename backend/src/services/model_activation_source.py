"""Train an SAE on activations the base model produces while training runs (the on-the-fly path).

WHY THIS EXISTS (SAE training remediation item 3, 2026-09-15). A training with no
extractions extracted activations inside the step loop, and did it badly:

* it crashed at step 0 — the held-out evaluation read ``holdout_activations``,
  which only the cached branch assigned (UnboundLocalError, run FAILED);
* every step drew ``batch_size`` random ROWS with replacement and forwarded them
  all at full length, then kept ``batch_size`` random tokens — so one batch came
  from a handful of documents, rows repeated before others were read, and a
  2,048-token window forwarded ~4M positions to keep 4,096;
* each layer drew its own random positions, so layer 11's SAE and layer 12's saw
  different tokens;
* ``dataset_weights`` was ignored with a warning, datasets were concatenated and
  sampled by row count, and there was no held-out split, no b_dec mean
  initialisation, and no record of how much of a corpus had been read.

WHAT IT DOES INSTEAD — the rolling buffer's contract, with the model as the reader.

* A buffer is filled by running the model over WHOLE tokenized rows (blocks),
  taken without replacement per source by a seeded ``RowCycle``: no row repeats
  until its source is exhausted, and the source then reshuffles.
* Each source contributes a per-refill quota of REAL tokens
  (``dataset_mixture.allocate_tokens`` over the training's ``dataset_weights``,
  positional over its ``dataset_ids``). Padding is dropped by the tokenization's
  own attention mask (``activation_mask.select_real_tokens``).
* Every layer's tokens are written through ONE permutation, so a batch holds the
  same positions for every SAE, drawn from many blocks.
* The forward is micro-batched (``micro_batch_rows`` rows at a time), and the
  buffer lives on the device and at the size the training task measured.
* ``state_dict``/``load_state_dict`` implement the buffer-state contract: the
  state records how the SERVED buffer was selected (the row cycles and the
  permutation generator as they were before its refill) and how far it was
  served. Loading re-selects the same rows, re-runs the model over them with the
  same micro-batching, and resumes at the same position. The checkpoint holds each
  source's row order (int64 per training row, ~60 MB for a 7.5M-block corpus)
  instead of the buffer itself, which is gigabytes.
"""

from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import activation_mask, dataset_mixture
from .activation_buffer import RowCycle
from .holdout_evaluation import holdout_row_order

logger = logging.getLogger(__name__)

Key = Tuple[int, str]
#: ``capture(padded_input_ids, attention_masks) -> {key: (rows, seq, d) activations}``.
#: When it is also a context manager the source enters it around each refill, so
#: hooks are registered once per refill rather than once per forward.
Capture = Callable[[List[List[int]], List[List[int]]], Dict[Key, torch.Tensor]]

STATE_KIND = "model_activation_source"
STATE_VERSION = 1
#: Rows sampled per source to estimate its real (non-pad) token count.
#: RECORDED DEFECT (review R1-B F2): on rows of mixed length (10% full, 90% a single
#: token) this estimate ranges 0.39x-1.42x of the truth over 20 seeds, and an
#: overshoot makes a quota the source cannot fill (now warned about in ``refill``).
#: 1,024 rows keeps it within ~10%. Not changed here: it feeds quota planning, which
#: the integrator's storage-plan fix for R1D-1/R1D-2 owns. The strict xfail
#: ``test_the_real_token_estimate_is_close_on_mixed_length_rows`` flips when it is.
REAL_TOKEN_SAMPLE_ROWS = 64


def _as_list(values) -> List[int]:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def read_row(dataset: Any, row: int) -> Tuple[List[int], Optional[List[int]]]:
    """``(input_ids, attention_mask or None)`` of one tokenized row."""
    item = dataset[int(row)]
    ids = _as_list(item["input_ids"])
    mask = item.get("attention_mask") if hasattr(item, "get") else None
    return ids, _as_list(mask)


@dataclass
class TokenRowSource:
    """One tokenized dataset: the rows training may read.

    ``rows`` are the dataset's row numbers left for training (held-out rows
    removed), in any order; ``max_row_tokens`` is the widest a row can be (the
    tokenization's ``max_length``), which bounds how far one refill can exceed
    its quota.
    """

    label: str
    dataset: Any
    rows: np.ndarray
    max_row_tokens: int

    def __post_init__(self) -> None:
        self.rows = np.asarray(self.rows, dtype=np.int64)
        self.max_row_tokens = int(self.max_row_tokens)


@dataclass
class _PlannedRow:
    source: int
    row: int
    ids: List[int]
    mask: List[int]
    real: int


class _RowReader:
    """The token counts ``RowCycle.take`` asks for, read from the dataset on demand.

    A corpus of 7.5M blocks cannot have its attention masks summed up front, and
    one refill needs only the rows it takes (plus the one that did not fit).
    Each read row is kept, so the refill forwards exactly the ids it counted.
    """

    def __init__(self, source: TokenRowSource, index: int) -> None:
        self.source = source
        self.index = index
        self.cache: Dict[int, _PlannedRow] = {}
        self.missing_masks = 0

    def entry(self, i: int) -> _PlannedRow:
        i = int(i)
        hit = self.cache.get(i)
        if hit is None:
            row = int(self.source.rows[i])
            ids, mask = read_row(self.source.dataset, row)
            # THE SAME builder the forward uses, so a count here is exactly the
            # number of real positions the mask will select there.
            _, masks, lengths, _, missing = activation_mask.build_padded_batch(
                [ids], [mask], len(ids), 0
            )
            self.missing_masks += missing
            hit = _PlannedRow(self.index, row, ids, masks[0], int(lengths[0]))
            self.cache[i] = hit
        return hit

    def __getitem__(self, i: int) -> int:
        return self.entry(i).real


def estimate_real_tokens(source: TokenRowSource, seed: int, index: int) -> int:
    """A source's trainable real tokens, from the attention masks of a seeded sample of rows."""
    n = int(source.rows.size)
    if n == 0:
        return 0
    rng = np.random.default_rng([int(seed), 0x5A4D, int(index)])
    sample = rng.choice(n, size=min(n, REAL_TOKEN_SAMPLE_ROWS), replace=False)
    reader = _RowReader(source, index)
    mean = float(np.mean([reader[i] for i in sample]))
    return int(round(mean * n))


def _cycle_state(cycle: RowCycle) -> Dict[str, Any]:
    return {
        "order": torch.from_numpy(np.array(cycle.order, dtype=np.int64, copy=True)),
        "cursor": int(cycle.cursor),
        "epochs_completed": int(cycle.epochs_completed),
        "rng": cycle._rng.bit_generator.state,
    }


def _restore_cycle(cycle: RowCycle, state: Dict[str, Any]) -> None:
    order = state["order"]
    order = order.numpy() if isinstance(order, torch.Tensor) else np.asarray(order)
    if order.size != cycle.n:
        raise ValueError(f"a row cycle of {cycle.n} rows cannot load a state of {order.size}")
    cycle.order = np.array(order, dtype=np.int64, copy=True)
    cycle.cursor = int(state["cursor"])
    cycle.epochs_completed = int(state["epochs_completed"])
    cycle._rng.bit_generator.state = state["rng"]


def _rows_digest(rows: np.ndarray) -> int:
    return zlib.crc32(np.ascontiguousarray(rows, dtype=np.int64).tobytes())


class ModelActivationSource:
    """A rolling training buffer filled by running the base model over tokenized rows.

    Serves ``next_batch`` exactly as ``RollingActivationBuffer`` does: batches
    without replacement, copies rather than views, a refill when the buffer is
    spent. ``tensors`` is updated in place, so the training loop's
    ``cached_activations`` always names the current buffer.
    """

    def __init__(
        self,
        sources: Sequence[TokenRowSource],
        keys: Sequence[Key],
        quotas: Sequence[int],
        *,
        capture: Capture,
        hidden_dim: int,
        seed: int,
        storage_device: torch.device,
        train_device: torch.device,
        micro_batch_rows: int,
        pad_token_id: int = 0,
        prefill: bool = True,
    ) -> None:
        if len(quotas) != len(sources):
            raise ValueError(f"{len(quotas)} quotas for {len(sources)} sources")
        if sum(int(q) for q in quotas) <= 0:
            raise ValueError("every quota is zero; the buffer would hold nothing")
        self.sources = list(sources)
        self.keys = [tuple(k) for k in keys]
        self.quotas = [int(q) for q in quotas]
        self.hidden_dim = int(hidden_dim)
        self.capture = capture
        self.storage_device = torch.device(storage_device)
        self.train_device = torch.device(train_device)
        self.micro_batch_rows = max(1, int(micro_batch_rows))
        self.pad_token_id = int(pad_token_id)
        self._cycles = [
            RowCycle(int(src.rows.size), np.random.default_rng([int(seed), i]))
            for i, src in enumerate(self.sources)
        ]
        self._torch_gen = torch.Generator().manual_seed(int(seed))
        #: A refill takes at least one whole row per source with a quota, so it
        #: can exceed a quota by up to that source's widest row.
        self.capacity = sum(
            max(q, src.max_row_tokens) for q, src in zip(self.quotas, self.sources) if q > 0
        )
        self._storage: Dict[Key, Optional[torch.Tensor]] = {}
        self.tensors: Dict[Key, torch.Tensor] = {}
        self.size = 0
        self._pos = 0
        self.refills = 0
        self.tokens_loaded = [0] * len(self.sources)
        self.rows_loaded = [0] * len(self.sources)
        self.tokens_dropped = 0
        self._served_epochs = [0] * len(self.sources)
        self.last_refill_timings: Dict[str, float] = {}
        #: How the served buffer was selected: the cycles and the permutation
        #: generator as they stood before its refill. None until the first refill.
        self._served_selection: Optional[Dict[str, Any]] = None
        #: Sources already named for falling short of their quota (warned once each).
        self._warned_short: set = set()
        self._closed = False
        if prefill:
            self.refill()

    # ── selection ─────────────────────────────────────────────────────────

    def _selection_state(self) -> Dict[str, Any]:
        return {
            "cycles": [_cycle_state(c) for c in self._cycles],
            "torch_generator": self._torch_gen.get_state(),
        }

    def _restore_selection(self, state: Dict[str, Any]) -> None:
        if len(state["cycles"]) != len(self._cycles):
            raise ValueError("the saved selection is for a different number of sources")
        for cycle, saved in zip(self._cycles, state["cycles"], strict=True):
            _restore_cycle(cycle, saved)
        self._torch_gen.set_state(state["torch_generator"])

    def epochs_completed(self) -> List[int]:
        """Passes completed over each source, as of the buffer being served."""
        return list(self._served_epochs)

    def _plan(self) -> Tuple[List[_PlannedRow], List[int], List[int], int]:
        planned: List[_PlannedRow] = []
        tokens = [0] * len(self.sources)
        rows = [0] * len(self.sources)
        missing = 0
        for i, (src, cycle, quota) in enumerate(zip(self.sources, self._cycles, self.quotas, strict=True)):
            reader = _RowReader(src, i)
            chosen = cycle.take(reader, quota)
            for c in chosen.tolist():
                entry = reader.entry(c)
                planned.append(entry)
                tokens[i] += entry.real
                rows[i] += 1
            missing += reader.missing_masks
        return planned, tokens, rows, missing

    # ── memory ────────────────────────────────────────────────────────────

    def _storage_for(self, key: Key, total: int) -> torch.Tensor:
        storage = self._storage.get(key)
        if storage is None or storage.shape[0] < total:
            if storage is not None:
                logger.warning(
                    "On-the-fly buffer: a refill selected %s tokens, more than the %s allocated "
                    "(a row wider than its tokenization's max_length); reallocating",
                    f"{total:,}", f"{storage.shape[0]:,}",
                )
            self.tensors.pop(key, None)
            self._storage[key] = None
            del storage
            rows = max(int(total), self.capacity)
            self._storage[key] = torch.empty(
                (rows, self.hidden_dim), dtype=torch.float32, device=self.storage_device
            )
        return self._storage[key]

    # ── refill ────────────────────────────────────────────────────────────

    def refill(self) -> None:
        """Select the next rows, run the model over them, and serve the result shuffled."""
        if self._closed:
            raise RuntimeError("the on-the-fly activation source is closed")
        started = time.monotonic()
        before = self._selection_state()
        planned, tokens, rows, missing = self._plan()
        total = int(sum(tokens))
        if total == 0:
            self._restore_selection(before)
            raise ValueError(
                "an on-the-fly refill selected no real tokens: every row it took is padding"
            )
        selected = time.monotonic()
        order = torch.randperm(total, generator=self._torch_gen)
        inverse = torch.empty(total, dtype=torch.int64)
        inverse[order] = torch.arange(total, dtype=torch.int64)
        epochs = [c.epochs_completed for c in self._cycles]

        # The served buffer is spent: count its unserved tail once.
        if self.size:
            self.tokens_dropped += self.size - self._pos
        self.size = self._pos = 0
        try:
            storages = {key: self._storage_for(key, total) for key in self.keys}
            index = inverse.to(self.storage_device)
            forward_s = self._fill(planned, storages, index)
        except BaseException:
            # A REFILL THAT FAILS IS RETRIED, NEVER SKIPPED (the rolling buffer's
            # rule). The cycles have already moved past the rows this refill took;
            # left there, a retry after an OOM in the forward — which the training
            # loop makes — would select new rows and skip these for the whole
            # pass. The served buffer stays retired (size 0), so nothing
            # half-written is served.
            self._restore_selection(before)
            raise

        for key in self.keys:
            self.tensors[key] = storages[key][:total]
        for i in range(len(self.sources)):
            self.tokens_loaded[i] += tokens[i]
            self.rows_loaded[i] += rows[i]
        self.size, self._pos = total, 0
        self.refills += 1
        self._served_epochs = epochs
        self._served_selection = before
        # A SOURCE THAT CANNOT FILL ITS QUOTA IS NAMED, ONCE (review R1-B F2). Its
        # take read every row it has and still came up short: it is re-read whole on
        # every refill and the buffer's mixture falls short of what was asked. That
        # happens whenever its real-token estimate overshoots, and was silent.
        for i, (src, quota, cycle) in enumerate(zip(self.sources, self.quotas, self._cycles, strict=True)):
            if quota > 0 and rows[i] >= cycle.n and tokens[i] < quota and i not in self._warned_short:
                self._warned_short.add(i)
                logger.warning(
                    "On-the-fly buffer: %s holds only %s real training tokens against its per-refill "
                    "quota of %s, so it is read whole on every refill (a pass per buffer) and supplies "
                    "%.1f%% of the buffer instead of %.1f%%",
                    src.label, f"{tokens[i]:,}", f"{quota:,}",
                    tokens[i] / total * 100, quota / max(1, sum(self.quotas)) * 100,
                )
        self.last_refill_timings = {
            "stall_s": time.monotonic() - started,
            "select_s": selected - started,
            "forward_s": forward_s,
        }
        logger.info(
            "On-the-fly buffer refill #%d: %s real tokens from %d rows [%s] in %.1fs "
            "(select %.2fs, forward %.1fs, %d rows per forward); source passes completed %s%s",
            self.refills, f"{total:,}", sum(rows),
            ", ".join(
                f"{s.label}={t:,} ({r} rows)"
                for s, t, r in zip(self.sources, tokens, rows, strict=True)
            ),
            self.last_refill_timings["stall_s"], selected - started, forward_s,
            self.micro_batch_rows, self.epochs_completed(),
            f"; {missing} rows had no attention_mask and were read as all real" if missing else "",
        )

    def _fill(
        self, planned: List[_PlannedRow], storages: Dict[Key, torch.Tensor], index: torch.Tensor
    ) -> float:
        """Forward the planned rows in micro-batches; write each token to its shuffled slot."""
        started = time.monotonic()
        # All-padding rows are consumed (the cycle has served them) but have
        # nothing to forward.
        forwarded = [p for p in planned if p.real > 0]
        offset = 0
        session = self.capture if hasattr(self.capture, "__enter__") else None
        if session is not None:
            session.__enter__()
        try:
            for lo in range(0, len(forwarded), self.micro_batch_rows):
                offset = self._forward_rows(forwarded[lo:lo + self.micro_batch_rows], storages, index, offset)
        finally:
            if session is not None:
                session.__exit__(None, None, None)
        if offset != index.shape[0]:
            raise RuntimeError(
                f"the forward produced {offset:,} real tokens for a refill planned at {index.shape[0]:,}"
            )
        return time.monotonic() - started

    def _forward_rows(
        self, rows: List[_PlannedRow], storages: Dict[Key, torch.Tensor], index: torch.Tensor, offset: int
    ) -> int:
        padded, masks, lengths, _, _ = activation_mask.build_padded_batch(
            [r.ids for r in rows], [r.mask for r in rows],
            max(len(r.ids) for r in rows), self.pad_token_id,
        )
        n_real = int(sum(lengths))
        captured = self.capture(padded, masks)
        for key in self.keys:
            if key not in captured:
                raise RuntimeError(f"the capture returned no activations for {key}")
            # PADDING IS NOT TRAINING DATA: selected by the batch's own mask.
            real = activation_mask.select_real_tokens(captured[key], masks)
            if real.shape[0] != n_real:
                raise RuntimeError(
                    f"{key}: {real.shape[0]} real tokens captured for {n_real} planned"
                )
            slots = index[offset:offset + n_real]
            storages[key][slots] = real.to(device=self.storage_device, dtype=torch.float32)
        return offset + n_real

    # ── serving ───────────────────────────────────────────────────────────

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
        for key in self.keys:
            batch = self.tensors[key][lo:hi]
            # A copy, never a view: a view pins the whole buffer across the next
            # refill (see RollingActivationBuffer.next_batch).
            batch = batch.to(self.train_device) if batch.device != self.train_device else batch.clone()
            out[key] = batch
        self._pos = hi
        return out

    # ── buffer-state contract ─────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        """Everything needed to serve, after :meth:`load_state_dict`, the batches this source would serve next.

        JSON-serialisable values and CPU tensors only (saved with ``torch.save``,
        loaded with ``weights_only=True``). The buffer's contents are NOT saved:
        loading recomputes them from the recorded selection.
        """
        return {
            "kind": STATE_KIND,
            "version": STATE_VERSION,
            "labels": [s.label for s in self.sources],
            "rows_digest": [_rows_digest(s.rows) for s in self.sources],
            "keys": [[int(k[0]), str(k[1])] for k in self.keys],
            "quotas": list(self.quotas),
            "micro_batch_rows": int(self.micro_batch_rows),
            "selection": self._served_selection,
            "position": int(self._pos),
            "size": int(self.size),
            "refills": int(self.refills),
            "tokens_loaded": list(self.tokens_loaded),
            "rows_loaded": list(self.rows_loaded),
            "tokens_dropped": int(self.tokens_dropped),
            "served_epochs": list(self._served_epochs),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Resume from :meth:`state_dict`: re-select the served buffer, re-run the model, seek.

        Refuses a state recorded for other sources, rows, layers or quotas —
        replaying it would serve different tokens while claiming to continue.
        Adopts the saved micro-batching: the same rows forwarded in different
        groups can differ in the last bits of a float.
        """
        if self._closed:
            raise RuntimeError("the on-the-fly activation source is closed")
        if state.get("kind") != STATE_KIND or int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError(f"not a {STATE_KIND} v{STATE_VERSION} state: {state.get('kind')!r}")
        checks = {
            "labels": [s.label for s in self.sources],
            "rows_digest": [_rows_digest(s.rows) for s in self.sources],
            "keys": [[int(k[0]), str(k[1])] for k in self.keys],
            "quotas": list(self.quotas),
        }
        for name, mine in checks.items():
            if list(state[name]) != mine:
                raise ValueError(
                    f"the saved buffer state has {name}={state[name]!r}, this source has {mine!r}"
                )
        self.micro_batch_rows = max(1, int(state["micro_batch_rows"]))
        self.size = self._pos = 0
        selection = state["selection"]
        if selection is not None:
            self._restore_selection(selection)
            self.refill()
            if self.size != int(state["size"]):
                raise RuntimeError(
                    f"replaying the saved selection produced {self.size:,} tokens, not "
                    f"{int(state['size']):,}: the tokenized data changed"
                )
        self._pos = int(state["position"])
        self.refills = int(state["refills"])
        self.tokens_loaded = [int(v) for v in state["tokens_loaded"]]
        self.rows_loaded = [int(v) for v in state["rows_loaded"]]
        self.tokens_dropped = int(state["tokens_dropped"])
        self._served_epochs = [int(v) for v in state["served_epochs"]]

    def load_state_dict_after_replan(self, state: Dict[str, Any]) -> int:
        """Continue a saved pass under DIFFERENT quotas. Returns the tokens skipped.

        For a resume whose saved storage plan no longer fits (services/activation_plan.py,
        review round 1 R1D-2). The served buffer's rows are planned again under its OWN
        saved quotas and its permutation drawn again — nothing is forwarded — so the row
        cycles and the permutation generator stand exactly where they stood after it; the
        next buffer is then filled under this source's quotas.

        No row repeats for the reason given in
        ``RollingActivationBuffer.load_state_dict_after_replan``: a quota decides how far
        ``RowCycle.take`` moves through a pass, never which rows are left in it. The
        interrupted buffer's unserved tail is skipped until the source's next pass and
        counted in ``tokens_dropped``. The held-out rows were removed from ``rows`` before
        the source was built.
        """
        if self._closed:
            raise RuntimeError("the on-the-fly activation source is closed")
        if state.get("kind") != STATE_KIND or int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError(f"not a {STATE_KIND} v{STATE_VERSION} state: {state.get('kind')!r}")
        checks = {
            "labels": [s.label for s in self.sources],
            "rows_digest": [_rows_digest(s.rows) for s in self.sources],
            "keys": [[int(k[0]), str(k[1])] for k in self.keys],
        }
        for name, mine in checks.items():
            if list(state[name]) != mine:
                raise ValueError(
                    f"the saved buffer state has {name}={state[name]!r}, this source has {mine!r}"
                )
        saved_quotas = [int(q) for q in state["quotas"]]
        if len(saved_quotas) != len(self.sources):
            raise ValueError(f"the saved state has {len(saved_quotas)} quotas for {len(self.sources)} sources")
        self.micro_batch_rows = max(1, int(state["micro_batch_rows"]))
        self.size = self._pos = 0
        skipped = 0
        if state["selection"] is not None:
            self._restore_selection(state["selection"])
            quotas, self.quotas = self.quotas, saved_quotas
            try:
                served_tokens = self._plan()[1]
            finally:
                self.quotas = quotas
            served_total = int(sum(served_tokens))
            if served_total != int(state["size"]):
                raise RuntimeError(
                    f"planning the saved selection again produced {served_total:,} tokens, not "
                    f"{int(state['size']):,}: the tokenized data changed"
                )
            # The served buffer's permutation: refill() draws exactly this after its plan.
            torch.randperm(served_total, generator=self._torch_gen)
            skipped = served_total - int(state["position"])
        self.refills = int(state["refills"])
        self.tokens_loaded = [int(v) for v in state["tokens_loaded"]]
        self.rows_loaded = [int(v) for v in state["rows_loaded"]]
        self.tokens_dropped = int(state["tokens_dropped"]) + skipped
        self.refill()
        return skipped

    # ── shutdown ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the buffer, and the capture (which holds the base model). Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.tensors.clear()
        self._storage.clear()
        self.size = self._pos = 0
        self.capture = None


def collect_holdout_activations(
    sources: Sequence[Tuple[str, Any, np.ndarray]],
    keys: Sequence[Key],
    quotas: Sequence[int],
    *,
    capture: Capture,
    seed: int,
    micro_batch_rows: int,
    pad_token_id: int = 0,
    weights: Optional[Sequence[float]] = None,
) -> Dict[Key, torch.Tensor]:
    """Run the model over held-out rows once: ``{key: [tokens, d] on the CPU}``.

    ``sources`` are ``(label, dataset, held_rows)``. Each source's rows are taken
    in ``holdout_evaluation.holdout_row_order`` (seeded, never lowest-first) until
    its quota of real tokens is met, and the last row is cut to fit. Every key is
    evaluated on the same positions.

    A SOURCE THAT RUNS DRY HANDS THE REST ON (review R1-B F3), as
    ``holdout_evaluation.holdout_quotas`` promises on the cached path. A quota is
    sized from an estimate, and a padding-heavy source can hold fewer real
    held-out tokens than its share; its shortfall is then re-allocated, by
    ``weights`` (equal when None), to the sources with held-out rows left, and a
    warning names it. Before, the shortfall was simply not evaluated.
    """
    keys = [tuple(k) for k in keys]
    n = len(sources)
    quotas = [max(0, int(q)) for q in quotas]
    if len(quotas) != n:
        raise ValueError(f"{len(quotas)} held-out quotas for {n} sources")
    step = max(1, int(micro_batch_rows))
    goal = sum(quotas)
    targets = list(quotas)
    readers, orders = [], []
    cursor, got, rows_used = [0] * n, [0] * n, [0] * n
    exhausted = [False] * n
    parts: List[Dict[Key, List[torch.Tensor]]] = [{key: [] for key in keys} for _ in range(n)]
    widest = 1
    for index, (label, dataset, held_rows) in enumerate(sources):
        held_rows = np.asarray(held_rows, dtype=np.int64)
        readers.append(_RowReader(TokenRowSource(label, dataset, held_rows, 0), index))
        orders.append(holdout_row_order(held_rows.size, seed, index))
        exhausted[index] = held_rows.size == 0

    def forward(chosen: List[_PlannedRow], into: Dict[Key, List[torch.Tensor]]) -> None:
        for lo in range(0, len(chosen), step):
            group = chosen[lo:lo + step]
            padded, masks, _, _, _ = activation_mask.build_padded_batch(
                [r.ids for r in group], [r.mask for r in group],
                max(len(r.ids) for r in group), int(pad_token_id),
            )
            captured = capture(padded, masks)
            for key in keys:
                if key not in captured:
                    raise RuntimeError(f"the capture returned no held-out activations for {key}")
                into[key].append(
                    activation_mask.select_real_tokens(captured[key], masks).to("cpu", torch.float32)
                )

    session = capture if hasattr(capture, "__enter__") else None
    if session is not None:
        session.__enter__()
    try:
        # Each round reads every source up to its target; a round that leaves a
        # deficit exhausts at least one source, so there are at most n + 1 rounds.
        for _ in range(n + 1):
            for index in range(n):
                if exhausted[index] or got[index] >= targets[index]:
                    continue
                reader, order = readers[index], orders[index]
                chosen: List[_PlannedRow] = []
                while got[index] < targets[index] and cursor[index] < order.size:
                    entry = reader.entry(order[cursor[index]])
                    reader.cache.clear()
                    cursor[index] += 1
                    if entry.real == 0:
                        continue
                    widest = max(widest, len(entry.ids))
                    chosen.append(entry)
                    got[index] += entry.real
                if got[index] < targets[index]:
                    exhausted[index] = True
                rows_used[index] += len(chosen)
                forward(chosen, parts[index])
            deficit = goal - sum(min(g, t) for g, t in zip(got, targets))
            open_sources = [i for i in range(n) if not exhausted[i]]
            if deficit <= 0 or not open_sources:
                break
            shares = [
                (float(weights[i]) if weights is not None else 1.0) if i in open_sources else 0.0
                for i in range(n)
            ]
            if sum(shares) <= 0:
                break
            room = [
                (orders[i].size - cursor[i]) * widest + max(0, got[i] - targets[i]) if i in open_sources else 0
                for i in range(n)
            ]
            extra = dataset_mixture.allocate_tokens(room, deficit, shares)
            if sum(extra) == 0:
                break
            targets = [t + e for t, e in zip(targets, extra)]
    finally:
        if session is not None:
            session.__exit__(None, None, None)

    out: Dict[Key, List[torch.Tensor]] = {key: [] for key in keys}
    for index, (label, _, _) in enumerate(sources):
        taken = min(got[index], targets[index])
        logger.info(
            "Held-out evaluation set: %s real tokens from %d held-out rows of %s (quota %s)",
            f"{taken:,}", rows_used[index], label, f"{targets[index]:,}",
        )
        if got[index] < quotas[index]:
            logger.warning(
                "Held-out evaluation: %s holds only %s real held-out tokens against its quota of %s; "
                "the sources with held-out rows left supply the rest",
                label, f"{got[index]:,}", f"{quotas[index]:,}",
            )
        for key in keys:
            if parts[index][key] and taken > 0:
                out[key].append(torch.cat(parts[index][key])[:taken])
    return {key: torch.cat(p).contiguous() for key, p in out.items() if p}
