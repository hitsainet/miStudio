"""Statistical-soundness pins for circuit mining (016 Tasks 3.1, A.3 normative):
planted edges surface, high-base-rate pairs do NOT, circular-shift null
preserves marginals EXACTLY, whole-corpus permutation is not reachable,
BH-FDR, held-out replication."""

import inspect

import numpy as np
import pytest

from src.services.circuit_stats_service import (
    bh_fdr,
    pooled_null_pvalues,
    circular_shift_keys,
    doc_of,
    heldout_replication,
    make_keys,
    null_test,
    pair_stats,
    pos_of,
    restrict_to_docs,
    supernode_keys,
)


def _synthetic_corpus(n_docs=50, doc_len=128, seed=7):
    """Planted world: feature A drives feature B (co-fire 80% of A's tokens);
    features X and Y are independent but HIGH base-rate (30% of all tokens)."""
    rng = np.random.default_rng(seed)
    doc_lengths = {d: doc_len for d in range(n_docs)}
    a_keys, b_keys, x_keys, y_keys = [], [], [], []
    for d in range(n_docs):
        a_pos = rng.choice(doc_len, size=6, replace=False)
        co = a_pos[: int(len(a_pos) * 0.8)]
        b_extra = rng.choice(doc_len, size=3, replace=False)
        x_pos = rng.choice(doc_len, size=int(doc_len * 0.3), replace=False)
        y_pos = rng.choice(doc_len, size=int(doc_len * 0.3), replace=False)
        a_keys.append(make_keys(np.full(len(a_pos), d), a_pos))
        b_keys.append(make_keys(np.full(len(co) + 3, d),
                                np.concatenate([co, b_extra])))
        x_keys.append(make_keys(np.full(len(x_pos), d), x_pos))
        y_keys.append(make_keys(np.full(len(y_pos), d), y_pos))
    cat = lambda ks: np.unique(np.concatenate(ks))
    return (cat(a_keys), cat(b_keys), cat(x_keys), cat(y_keys),
            doc_lengths, n_docs * doc_len)


class TestPlantedWorld:
    def test_planted_edge_beats_null(self):
        a, b, x, y, doc_lengths, n = _synthetic_corpus()
        planted = null_test(a, b, doc_lengths, seed=3)
        assert planted.observed_n_ud > planted.threshold_n_ud
        assert planted.p_value < 0.02
        assert pair_stats(a, b, n).pmi > 1.0

    def test_high_base_rate_pair_does_not_surface(self):
        a, b, x, y, doc_lengths, n = _synthetic_corpus()
        base = null_test(x, y, doc_lengths, seed=4)
        # Independent-but-frequent: large n_ud, but NOT beyond its own null.
        assert base.observed_n_ud > 100  # raw counts would rank it top
        assert base.p_value > 0.05
        assert abs(pair_stats(x, y, n).pmi) < 0.5  # PMI ≈ 0 for independence


class TestNullConstruction:
    def test_circular_shift_preserves_marginals_exactly(self):
        a, *_ , doc_lengths, _ = _synthetic_corpus()
        rng = np.random.default_rng(0)
        shifted = circular_shift_keys(a, doc_lengths, rng)
        assert len(shifted) == len(a)  # bijective per doc — nothing collides
        orig_counts = {int(d): int((doc_of(a) == d).sum())
                       for d in np.unique(doc_of(a))}
        new_counts = {int(d): int((doc_of(shifted) == d).sum())
                      for d in np.unique(doc_of(shifted))}
        assert new_counts == orig_counts
        # Positions stay in-bounds.
        assert int(pos_of(shifted).max()) < 128

    def test_shift_preserves_within_doc_gap_structure(self):
        doc_lengths = {0: 100}
        keys = make_keys(np.zeros(3, dtype=np.uint32), np.array([10, 11, 50]))
        rng = np.random.default_rng(5)
        shifted = np.sort(pos_of(circular_shift_keys(keys, doc_lengths, rng)).astype(int))
        gaps = sorted(((b - a) % 100 for a, b in
                       zip(shifted, np.roll(shifted, -1))))
        assert gaps == [1, 39, 60]  # circular gap multiset invariant

    def test_whole_corpus_permutation_not_reachable(self):
        # The forbidden naive null must not be selectable: no method/mode
        # parameter exists on the null API (pinned per FTDD §2).
        params = inspect.signature(null_test).parameters
        assert "method" not in params and "mode" not in params


class TestFDR:
    def test_bh_keeps_small_ps_only(self):
        p = [0.001, 0.002, 0.003, 0.5, 0.8, 0.9]
        keep = bh_fdr(p, q=0.05)
        assert keep.tolist() == [True, True, True, False, False, False]

    def test_bh_empty(self):
        assert bh_fdr([]).tolist() == []

    def test_bh_none_pass(self):
        assert bh_fdr([0.5, 0.9], q=0.05).sum() == 0


class TestHeldout:
    def test_planted_edge_replicates_and_base_rate_does_not(self):
        a, b, x, y, doc_lengths, _ = _synthetic_corpus()
        heldout = np.arange(40, 50, dtype=np.uint32)  # capture-time split
        result, flags = heldout_replication(
            [(a, b), (x, y)], heldout, doc_lengths, seed=11)
        assert result.tested == 2
        assert flags[0] is True and flags[1] is False
        assert result.rate == 0.5

    def test_restrict_to_docs(self):
        a, *_ = _synthetic_corpus()
        sub = restrict_to_docs(a, np.array([0, 1]))
        assert set(np.unique(doc_of(sub)).tolist()) <= {0, 1}


class TestSupernode:
    def test_union_semantics(self):
        m1 = make_keys(np.array([0, 0]), np.array([1, 2]))
        m2 = make_keys(np.array([0, 1]), np.array([2, 3]))
        keys = supernode_keys([m1, m2])
        assert len(keys) == 3  # (0,1),(0,2),(1,3) — max ⇔ any-member

    def test_empty(self):
        assert len(supernode_keys([])) == 0


class TestPooledNull:
    def test_pooled_p_beats_the_per_pair_floor(self):
        """The reason pooling exists: per-pair empirical p floors at 1/(K+1),
        which BH over m pairs can never clear. Pooled resolution = 1/(K·m+1)."""
        a, b, x, y, doc_lengths, _ = _synthetic_corpus()
        results = [null_test(a, b, doc_lengths, k_shuffles=50, seed=3),
                   null_test(x, y, doc_lengths, k_shuffles=50, seed=4)]
        pooled = pooled_null_pvalues(results)
        per_pair_floor = 1 / 51
        assert pooled[0] < per_pair_floor          # planted: finer than the floor
        assert pooled[1] > 0.05                     # independent: still rejected
        keep = bh_fdr(pooled, q=0.05)
        assert keep.tolist() == [True, False]

    def test_pooled_empty(self):
        assert pooled_null_pvalues([]).tolist() == []


class TestPooledNullEdges:
    """R1 Test-P3: the FDR-making fix must not silently return 0 or all."""

    def test_zero_variance_null(self):
        # A pair whose null never varies (sd==0): observed>mu → z=inf (kept),
        # observed==mu → z=0. Must not raise / NaN.
        from src.services.circuit_stats_service import NullResult
        nr_hit = NullResult(observed_n_ud=10, null_n_ud=np.zeros(20),
                            null_percentile=100.0, p_value=1/21, threshold_n_ud=0.0)
        nr_flat = NullResult(observed_n_ud=0, null_n_ud=np.zeros(20),
                             null_percentile=0.0, p_value=1.0, threshold_n_ud=0.0)
        p = pooled_null_pvalues([nr_hit, nr_flat])
        assert 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0
        assert p[0] < p[1]  # the hit is more significant than the flat pair

    def test_single_pair_pool(self):
        from src.services.circuit_stats_service import NullResult
        nr = NullResult(observed_n_ud=50, null_n_ud=np.arange(20).astype(float),
                        null_percentile=100.0, p_value=1/21, threshold_n_ud=19.0)
        p = pooled_null_pvalues([nr])
        assert len(p) == 1 and 0.0 <= p[0] <= 1.0
