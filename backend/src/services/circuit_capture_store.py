"""
Circuit capture store IO (Feature 016, BR-005/BR-023 — FTDD §1).

Columnar on-disk store for sparse above-threshold SAE activation events with
FIRST-CLASS token positions (the Tier-2.5 join key), per-(doc, token) SAE
reconstruction-error norms, and an optional attention top-k sidecar.

Layout under /data/circuit_captures/{cap_<hex12>}/:
  manifest.json      (written by the capture service, not here)
  layer_{L}.events   sorted structured array: (doc_id u32, token_pos u16,
                     feature_idx u16, act f16) — sorted by (feature_idx,
                     doc_id, token_pos) so per-feature ranges are contiguous
  layer_{L}.index    npz {feature_ids u32[], starts u64[], ends u64[]}
  layer_{L}.errnorm  sorted structured array: (doc_id u32, token_pos u16,
                     norm f16) — dense over captured tokens, scalar per token
  attn_l{L}.topk     OPTIONAL structured array: (doc_id u32, t_q u16,
                     head u16, t_k u16, mass f16)

Pure numpy — no torch/DB dependencies; exhaustively unit-testable with
synthetic stores (planted edges feed the stats-service tests).
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

U16_MAX = 65_536  # token positions are u16 (docs ≤512 tokens) — asserted at write

EVENT_DTYPE = np.dtype([
    ("doc_id", np.uint32),
    ("token_pos", np.uint16),
    # u32: wide SAEs (Gemma Scope 16k–131k latents) exceed u16 (R1 CR#2 —
    # a 65k+ feature index used to abort the whole capture). The merge-join
    # keys pack only doc_id<<16|token_pos, so feature width is independent.
    ("feature_idx", np.uint32),
    ("act", np.float16),
])

ERRNORM_DTYPE = np.dtype([
    ("doc_id", np.uint32),
    ("token_pos", np.uint16),
    ("norm", np.float16),
])

ATTN_DTYPE = np.dtype([
    ("doc_id", np.uint32),
    ("t_q", np.uint16),
    ("head", np.uint16),
    ("t_k", np.uint16),
    ("mass", np.float16),
])


def _events_path(store_dir: Path, layer: int) -> Path:
    return store_dir / f"layer_{layer}.events"


def _index_path(store_dir: Path, layer: int) -> Path:
    return store_dir / f"layer_{layer}.index"


def _errnorm_path(store_dir: Path, layer: int) -> Path:
    return store_dir / f"layer_{layer}.errnorm"


def _attn_path(store_dir: Path, layer: int) -> Path:
    return store_dir / f"attn_l{layer}.topk"


class _BufferedWriter:
    """Append-in-batches, sort-on-finalize base for the three artifact kinds."""

    dtype: np.dtype = EVENT_DTYPE
    sort_keys: Tuple[str, ...] = ()

    def __init__(self, path: Path):
        self.path = Path(path)
        self._chunks: List[np.ndarray] = []
        self._finalized = False

    def _append(self, arr: np.ndarray) -> None:
        if self._finalized:
            raise RuntimeError(f"{type(self).__name__} already finalized")
        if arr.dtype != self.dtype:
            raise TypeError(f"expected dtype {self.dtype}, got {arr.dtype}")
        if len(arr):
            self._chunks.append(np.asarray(arr).copy())

    @property
    def count(self) -> int:
        return sum(len(c) for c in self._chunks)

    def finalize(self) -> int:
        """Sort, write, return event count. Idempotent-guarded."""
        if self._finalized:
            raise RuntimeError(f"{type(self).__name__} already finalized")
        self._finalized = True
        data = (np.concatenate(self._chunks) if self._chunks
                else np.empty(0, dtype=self.dtype))
        if len(data) and self.sort_keys:
            data = data[np.argsort(data, order=list(self.sort_keys), kind="stable")]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            np.save(f, data, allow_pickle=False)
        return len(data)


class EventWriter(_BufferedWriter):
    """Above-threshold activation events for one layer."""

    dtype = EVENT_DTYPE
    # feature-major so the per-feature index maps to contiguous ranges;
    # (doc, pos) within feature so per-feature streams merge-join directly.
    sort_keys = ("feature_idx", "doc_id", "token_pos")

    def append(self, doc_ids: np.ndarray, token_pos: np.ndarray,
               feature_idx: np.ndarray, acts: np.ndarray) -> None:
        feature_idx = np.asarray(feature_idx)
        token_pos = np.asarray(token_pos)
        # token_pos is u16 (docs are ≤512 tokens); feature_idx is u32 (wide SAEs).
        if len(token_pos) and int(token_pos.max()) >= U16_MAX:
            raise ValueError(f"token_pos >= {U16_MAX} does not fit u16")
        arr = np.empty(len(feature_idx), dtype=EVENT_DTYPE)
        arr["doc_id"] = np.asarray(doc_ids, dtype=np.uint32)
        arr["token_pos"] = token_pos.astype(np.uint16)
        arr["feature_idx"] = feature_idx.astype(np.uint32)
        arr["act"] = np.asarray(acts, dtype=np.float16)
        self._append(arr)

    def finalize(self) -> int:
        n = super().finalize()
        # Companion index: feature_idx → [start, end) into the sorted events.
        events = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if len(events):
            fids = np.asarray(events["feature_idx"])
            ids, starts = np.unique(fids, return_index=True)
            ends = np.append(starts[1:], len(fids))
        else:
            ids = np.empty(0, dtype=np.uint32)
            starts = ends = np.empty(0, dtype=np.uint64)
        # File handle, not a bare path — np.savez appends ".npz" to paths.
        with open(_index_path(self.path.parent, _layer_of(self.path)), "wb") as f:
            np.savez(f,
                     feature_ids=ids.astype(np.uint32),
                     starts=starts.astype(np.uint64),
                     ends=ends.astype(np.uint64))
        return n


def _layer_of(events_path: Path) -> int:
    # layer_{L}.events → L
    return int(events_path.stem.split("_")[1])


class EventReader:
    """Memmap-backed reader over one layer's sorted events."""

    def __init__(self, store_dir: Path, layer: int):
        self.store_dir = Path(store_dir)
        self.layer = layer
        self.events = np.load(_events_path(self.store_dir, layer),
                              mmap_mode="r", allow_pickle=False)
        idx = np.load(_index_path(self.store_dir, layer), allow_pickle=False)
        self._starts = dict(zip(idx["feature_ids"].tolist(),
                                zip(idx["starts"].tolist(), idx["ends"].tolist())))

    def __len__(self) -> int:
        return len(self.events)

    @property
    def feature_ids(self) -> List[int]:
        return sorted(self._starts.keys())

    def feature_events(self, feature_idx: int) -> np.ndarray:
        """All events for one feature, sorted by (doc_id, token_pos)."""
        span = self._starts.get(int(feature_idx))
        if span is None:
            return np.empty(0, dtype=EVENT_DTYPE)
        return self.events[span[0]:span[1]]

    def feature_token_keys(self, feature_idx: int) -> np.ndarray:
        """(doc_id << 16 | token_pos) u64 keys — the merge-join currency."""
        ev = self.feature_events(feature_idx)
        return (ev["doc_id"].astype(np.uint64) << np.uint64(16)) | \
            ev["token_pos"].astype(np.uint64)

    def doc_ids(self) -> np.ndarray:
        return np.unique(self.events["doc_id"])

    def feature_activation_mass(self) -> Dict[int, float]:
        """{feature_idx: sum of captured activations} in ONE vectorized pass
        over the whole events array — replaces per-feature Python loops over
        16k–131k features (017 R2 B-6/B-7)."""
        if len(self.events) == 0:
            return {}
        fids = np.asarray(self.events["feature_idx"], dtype=np.int64)
        acts = np.asarray(self.events["act"], dtype=np.float64)
        totals = np.bincount(fids, weights=acts)
        nz = np.nonzero(totals)[0]
        return {int(i): float(totals[i]) for i in nz}


class ErrNormWriter(_BufferedWriter):
    """Per-(doc, token) SAE reconstruction-error norms (scalar, dense)."""

    dtype = ERRNORM_DTYPE
    sort_keys = ("doc_id", "token_pos")

    def append(self, doc_ids: np.ndarray, token_pos: np.ndarray,
               norms: np.ndarray) -> None:
        token_pos = np.asarray(token_pos)
        if len(token_pos) and int(token_pos.max()) >= U16_MAX:
            raise ValueError(f"token_pos >= {U16_MAX} does not fit u16")
        arr = np.empty(len(token_pos), dtype=ERRNORM_DTYPE)
        arr["doc_id"] = np.asarray(doc_ids, dtype=np.uint32)
        arr["token_pos"] = token_pos.astype(np.uint16)
        arr["norm"] = np.asarray(norms, dtype=np.float16)
        self._append(arr)


class ErrNormReader:
    def __init__(self, store_dir: Path, layer: int):
        self.data = np.load(_errnorm_path(Path(store_dir), layer),
                            mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        return len(self.data)

    def doc_norms(self, doc_id: int) -> np.ndarray:
        d = self.data["doc_id"]
        lo = np.searchsorted(d, doc_id, side="left")
        hi = np.searchsorted(d, doc_id, side="right")
        return self.data[lo:hi]


class AttnTopKWriter(_BufferedWriter):
    """Optional attention sidecar: top-k key positions per query token."""

    dtype = ATTN_DTYPE
    sort_keys = ("doc_id", "t_q", "head")

    def append(self, doc_ids: np.ndarray, t_q: np.ndarray, heads: np.ndarray,
               t_k: np.ndarray, mass: np.ndarray) -> None:
        for name, col in (("t_q", t_q), ("t_k", t_k), ("head", heads)):
            col = np.asarray(col)
            if len(col) and int(col.max()) >= U16_MAX:
                raise ValueError(f"{name} >= {U16_MAX} does not fit u16")
        arr = np.empty(len(np.asarray(t_q)), dtype=ATTN_DTYPE)
        arr["doc_id"] = np.asarray(doc_ids, dtype=np.uint32)
        arr["t_q"] = np.asarray(t_q, dtype=np.uint16)
        arr["head"] = np.asarray(heads, dtype=np.uint16)
        arr["t_k"] = np.asarray(t_k, dtype=np.uint16)
        arr["mass"] = np.asarray(mass, dtype=np.float16)
        self._append(arr)


class AttnTopKReader:
    def __init__(self, store_dir: Path, layer: int):
        self.data = np.load(_attn_path(Path(store_dir), layer),
                            mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        return len(self.data)

    def query_rows(self, doc_id: int, t_q: int) -> np.ndarray:
        mask = (self.data["doc_id"] == doc_id) & (self.data["t_q"] == t_q)
        return self.data[mask]


def open_writers(store_dir: Path, layer: int, *, attention: bool = False):
    """Convenience: (EventWriter, ErrNormWriter, AttnTopKWriter|None)."""
    store_dir = Path(store_dir)
    ev = EventWriter(_events_path(store_dir, layer))
    en = ErrNormWriter(_errnorm_path(store_dir, layer))
    at = AttnTopKWriter(_attn_path(store_dir, layer)) if attention else None
    return ev, en, at


def layer_files_exist(store_dir: Path, layer: int) -> bool:
    return (_events_path(Path(store_dir), layer).exists()
            and _index_path(Path(store_dir), layer).exists())
