"""
Circuit statistics service (Feature 016, BR-015 — normative math CIRCUITS-002 A.3).

Pure functions over capture-store readers / key arrays — no GPU, no DB —
so every statistical property is pinned by synthetic-store unit tests.

Discipline (all recorded in the run report):
- Association = PMI (log lift), never raw counts; Spearman secondary.
- Minimum support n_ud >= s_min before ranking.
- Null = WITHIN-DOCUMENT CIRCULAR SHIFT per unit (marginal rates and
  within-doc burstiness preserved EXACTLY). Naive whole-corpus permutation
  is not implemented and not reachable from config — it inflates
  significance and is forbidden (pinned).
- Multiple comparisons: Benjamini–Hochberg FDR.
- Held-out replication runs the SAME pipeline on the capture-time held-out
  partition (never re-split at mine time).

Unit activation keys are u64 `(doc_id << 16) | token_pos` (the store's
merge-join currency). Lag-0 limitation: all co-activation is same-position;
disclosed wherever results appear (IDL-35).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_S_MIN = 20
DEFAULT_NULL_SHUFFLES = 100
DEFAULT_NULL_PERCENTILE = 99.0
DEFAULT_FDR_Q = 0.05


def doc_of(keys: np.ndarray) -> np.ndarray:
    return (keys >> np.uint64(16)).astype(np.uint32)


def pos_of(keys: np.ndarray) -> np.ndarray:
    return (keys & np.uint64(0xFFFF)).astype(np.uint16)


def make_keys(doc_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return (np.asarray(doc_ids, dtype=np.uint64) << np.uint64(16)) | \
        np.asarray(positions, dtype=np.uint64)


def supernode_keys(member_key_sets: Sequence[np.ndarray]) -> np.ndarray:
    """A_C(t) = max over members, binarized at the capture θ — a supernode
    fires wherever ANY member fired (events are already θ-thresholded, so
    max > θ ⇔ any member event). 'max' is the recorded definition; 'mean'
    is not derivable from thresholded events and is documented as the
    non-default alternative requiring re-capture."""
    if not member_key_sets:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.concatenate([np.asarray(k, dtype=np.uint64)
                                     for k in member_key_sets]))


def restrict_to_docs(keys: np.ndarray, doc_ids: np.ndarray) -> np.ndarray:
    """Restrict a unit's keys to a document partition (discovery vs held-out)."""
    return keys[np.isin(doc_of(keys), np.asarray(doc_ids, dtype=np.uint32))]


@dataclass
class PairStats:
    n_up: int
    n_down: int
    n_ud: int
    n_tokens: int  # N — total captured token count in the partition
    pmi: float
    lift: float
    spearman: Optional[float]


def pair_stats(up_keys: np.ndarray, down_keys: np.ndarray, n_tokens: int,
               up_acts: Optional[np.ndarray] = None,
               down_acts: Optional[np.ndarray] = None) -> PairStats:
    """Lag-0 association between two binarized units.

    up_acts/down_acts (aligned with the key arrays) enable the secondary
    Spearman over the SHARED positions' activation magnitudes.
    """
    n_u, n_d = len(up_keys), len(down_keys)
    shared, up_idx, down_idx = np.intersect1d(
        up_keys, down_keys, assume_unique=True, return_indices=True)
    n_ud = len(shared)
    if n_ud == 0 or n_u == 0 or n_d == 0 or n_tokens == 0:
        pmi = float("-inf") if n_tokens else 0.0
        lift = 0.0
    else:
        lift = n_ud * n_tokens / (n_u * n_d)
        pmi = float(np.log(lift))
    spearman = None
    if up_acts is not None and down_acts is not None and n_ud >= 5:
        spearman = _spearman(np.asarray(up_acts, dtype=np.float64)[up_idx],
                             np.asarray(down_acts, dtype=np.float64)[down_idx])
    return PairStats(n_up=n_u, n_down=n_d, n_ud=n_ud, n_tokens=n_tokens,
                     pmi=pmi, lift=float(lift), spearman=spearman)


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    def _rank(x):
        order = np.argsort(x, kind="stable")
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x), dtype=np.float64)
        # average ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        return sums[inv] / counts[inv]
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def circular_shift_keys(keys: np.ndarray, doc_lengths: Dict[int, int],
                        rng: np.random.Generator) -> np.ndarray:
    """One null draw: per-document circular shift of a unit's positions.

    Preserves the per-document event count EXACTLY (marginal rates) and the
    within-document gap structure (burstiness) — the properties the null
    must hold fixed (RSK-011; pinned by unit test)."""
    if len(keys) == 0:
        return keys
    docs = doc_of(keys)
    pos = pos_of(keys).astype(np.uint32)
    out_pos = pos.copy()
    for d in np.unique(docs):
        length = int(doc_lengths.get(int(d), 0))
        if length <= 1:
            continue
        mask = docs == d
        shift = int(rng.integers(1, length))  # never the identity shift
        out_pos[mask] = (pos[mask] + shift) % length
    return np.unique(make_keys(docs, out_pos))


@dataclass
class NullResult:
    observed_n_ud: int
    null_n_ud: np.ndarray          # length K
    null_percentile: float         # where observed sits in the null (0-100)
    p_value: float                 # empirical, add-one smoothed
    threshold_n_ud: float          # the configured percentile of the null


def null_test(up_keys: np.ndarray, down_keys: np.ndarray,
              doc_lengths: Dict[int, int], *, k_shuffles: int = DEFAULT_NULL_SHUFFLES,
              percentile: float = DEFAULT_NULL_PERCENTILE,
              seed: int = 0) -> NullResult:
    """Empirical null for one pair via K circular-shift draws of the UP unit.

    With margins (n_u, n_d, N) fixed under the shift, PMI is monotone in
    n_ud — so the null is computed on n_ud and applies to PMI rank
    identically."""
    rng = np.random.default_rng(seed)
    down_sorted = np.sort(np.asarray(down_keys, dtype=np.uint64))
    observed = _intersect_count(np.asarray(up_keys, dtype=np.uint64), down_sorted)
    null_counts = np.empty(k_shuffles, dtype=np.int64)
    for k in range(k_shuffles):
        shifted = circular_shift_keys(up_keys, doc_lengths, rng)
        null_counts[k] = _intersect_count(shifted, down_sorted)
    pct = float((null_counts < observed).mean() * 100.0)
    p = float((1 + int((null_counts >= observed).sum())) / (k_shuffles + 1))
    return NullResult(
        observed_n_ud=int(observed),
        null_n_ud=null_counts,
        null_percentile=pct,
        p_value=p,
        threshold_n_ud=float(np.percentile(null_counts, percentile)),
    )


def _intersect_count(a: np.ndarray, b_sorted: np.ndarray) -> int:
    if len(a) == 0 or len(b_sorted) == 0:
        return 0
    idx = np.searchsorted(b_sorted, a)
    idx[idx == len(b_sorted)] = len(b_sorted) - 1
    return int((b_sorted[idx] == a).sum())


def pooled_null_pvalues(null_results: Sequence[NullResult]) -> np.ndarray:
    """Pooled standardized empirical p-values across pairs.

    Per-pair empirical p-values floor at 1/(K+1); with K=100 shuffles and
    m~2000 pairs, BH's rank-1 threshold (q/m ≈ 2.5e-5) is unreachable and
    the FDR stage would reject EVERYTHING. The standard remedy (pooled
    empirical null): standardize each pair's observed count against its own
    null (z = (obs − μ_null)/σ_null), pool ALL pairs' standardized null
    draws into one reference distribution, and compute each pair's p there —
    resolution improves to 1/(K·m + 1). Assumes post-support pairs' nulls
    are exchangeable after standardization (disclosed in the run report).
    """
    if not null_results:
        return np.zeros(0, dtype=np.float64)
    z_obs = np.empty(len(null_results))
    pool: List[np.ndarray] = []
    for i, nr in enumerate(null_results):
        mu = float(nr.null_n_ud.mean())
        sd = float(nr.null_n_ud.std())
        if sd == 0:
            z_obs[i] = np.inf if nr.observed_n_ud > mu else 0.0
            pool.append(np.zeros(len(nr.null_n_ud)))
        else:
            z_obs[i] = (nr.observed_n_ud - mu) / sd
            pool.append((nr.null_n_ud - mu) / sd)
    pooled = np.concatenate(pool)
    n = len(pooled)
    return np.array([(1 + int((pooled >= z).sum())) / (n + 1) for z in z_obs])


def bh_fdr(p_values: Sequence[float], q: float = DEFAULT_FDR_Q) -> np.ndarray:
    """Benjamini–Hochberg: boolean keep-mask over the pair list."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    passing = np.nonzero(ranked <= thresholds)[0]
    keep = np.zeros(m, dtype=bool)
    if len(passing):
        keep[order[: passing.max() + 1]] = True
    return keep


@dataclass
class ReplicationResult:
    tested: int
    replicated: int

    @property
    def rate(self) -> Optional[float]:
        return self.replicated / self.tested if self.tested else None


def heldout_replication(pairs: List[Tuple[np.ndarray, np.ndarray]],
                        heldout_docs: np.ndarray,
                        doc_lengths: Dict[int, int], *,
                        k_shuffles: int = DEFAULT_NULL_SHUFFLES,
                        percentile: float = DEFAULT_NULL_PERCENTILE,
                        seed: int = 1) -> Tuple[ReplicationResult, List[bool]]:
    """Re-test surfaced candidates on the held-out partition with its OWN
    null. Replicated = held-out n_ud exceeds the held-out null threshold.
    Uses the capture-time split — callers pass `heldout_docs` from the
    manifest, never a re-split."""
    flags: List[bool] = []
    heldout_docs = np.asarray(heldout_docs, dtype=np.uint32)
    for i, (up_keys, down_keys) in enumerate(pairs):
        up_h = restrict_to_docs(up_keys, heldout_docs)
        down_h = restrict_to_docs(down_keys, heldout_docs)
        if len(up_h) == 0 or len(down_h) == 0:
            flags.append(False)
            continue
        null = null_test(up_h, down_h, doc_lengths,
                         k_shuffles=k_shuffles, percentile=percentile,
                         seed=seed + i)
        flags.append(null.observed_n_ud > null.threshold_n_ud)
    return ReplicationResult(tested=len(flags), replicated=sum(flags)), flags
