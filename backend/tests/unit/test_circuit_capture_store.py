"""Round-trip pins for the circuit capture store (016 Task 1.1, FTDD §1):
events+index, errnorm, and attention sidecar; u16 bounds; sort order;
merge-join key currency."""

import numpy as np
import pytest

from src.services.circuit_capture_store import (
    ATTN_DTYPE,
    ERRNORM_DTYPE,
    EVENT_DTYPE,
    AttnTopKReader,
    AttnTopKWriter,
    ErrNormReader,
    ErrNormWriter,
    EventReader,
    EventWriter,
    layer_files_exist,
    open_writers,
)


@pytest.fixture
def store(tmp_path):
    return tmp_path


class TestEventRoundTrip:
    def test_round_trip_sorted_and_indexed(self, store):
        ev, en, at = open_writers(store, 13)
        # Append DELIBERATELY out of order across two batches.
        ev.append(np.array([2, 1]), np.array([5, 3]),
                  np.array([7, 7]), np.array([0.5, 0.25]))
        ev.append(np.array([1, 3]), np.array([0, 9]),
                  np.array([2, 7]), np.array([1.5, 0.75]))
        n = ev.finalize()
        assert n == 4
        assert layer_files_exist(store, 13)

        r = EventReader(store, 13)
        assert len(r) == 4
        assert r.feature_ids == [2, 7]
        f7 = r.feature_events(7)
        # Sorted by (doc_id, token_pos) within the feature.
        assert f7["doc_id"].tolist() == [1, 2, 3]
        assert f7["token_pos"].tolist() == [3, 5, 9]
        assert f7["act"].astype(float).tolist() == [0.25, 0.5, 0.75]
        assert len(r.feature_events(999)) == 0

    def test_merge_join_keys(self, store):
        ev, _, _ = open_writers(store, 13)
        ev.append(np.array([1, 1]), np.array([3, 4]),
                  np.array([5, 5]), np.array([1.0, 1.0]))
        ev.finalize()
        keys = EventReader(store, 13).feature_token_keys(5)
        assert keys.tolist() == [(1 << 16) | 3, (1 << 16) | 4]

    def test_wide_feature_idx_is_u32(self, store):
        # Gemma Scope SAEs have 16k–131k latents — feature_idx must not be
        # capped at u16 (R1 CR#2). token_pos stays u16 (docs ≤512 tokens).
        ev, _, _ = open_writers(store, 13)
        ev.append(np.array([0, 0]), np.array([1, 2]),
                  np.array([70_000, 131_071]), np.array([1.0, 2.0]))
        ev.finalize()
        r = EventReader(store, 13)
        assert set(r.feature_ids) == {70_000, 131_071}
        assert r.feature_events(131_071)["act"].astype(float).tolist() == [2.0]

    def test_token_pos_u16_bound_asserted(self, store):
        ev, _, _ = open_writers(store, 13)
        with pytest.raises(ValueError, match="token_pos"):
            ev.append(np.array([0]), np.array([70_000]),
                      np.array([1]), np.array([1.0]))

    def test_empty_store_valid(self, store):
        ev, _, _ = open_writers(store, 14)
        assert ev.finalize() == 0
        r = EventReader(store, 14)
        assert len(r) == 0 and r.feature_ids == []

    def test_double_finalize_rejected(self, store):
        ev, _, _ = open_writers(store, 13)
        ev.finalize()
        with pytest.raises(RuntimeError):
            ev.finalize()


class TestErrNormRoundTrip:
    def test_round_trip(self, store):
        _, en, _ = open_writers(store, 13)
        en.append(np.array([2, 1, 1]), np.array([0, 1, 0]),
                  np.array([0.5, 0.125, 0.25]))
        en.finalize()
        r = ErrNormReader(store, 13)
        assert len(r) == 3
        d1 = r.doc_norms(1)
        assert d1["token_pos"].tolist() == [0, 1]  # sorted within doc
        assert d1["norm"].astype(float).tolist() == [0.25, 0.125]
        assert len(r.doc_norms(99)) == 0


class TestAttnSidecar:
    def test_round_trip(self, store):
        _, _, at = open_writers(store, 12, attention=True)
        at.append(np.array([1, 1]), np.array([5, 5]), np.array([0, 1]),
                  np.array([2, 3]), np.array([0.5, 0.25]))
        at.finalize()
        r = AttnTopKReader(store, 12)
        rows = r.query_rows(1, 5)
        assert len(rows) == 2
        assert rows["t_k"].tolist() == [2, 3]

    def test_disabled_by_default(self, store):
        _, _, at = open_writers(store, 12)
        assert at is None


class TestSizeGuardrail:
    """R2 T2: the store-size abort math (R1 QA-P1) is now unit-testable
    without a GPU."""

    def test_exceeds_ceiling(self):
        from src.services.circuit_capture_service import (
            EVENT_BYTES, STORE_SIZE_MULTIPLIER, exceeds_size_ceiling,
            size_ceiling_bytes)
        est = 10_000_000  # big enough estimate that the 64MB floor doesn't apply
        ceiling = size_ceiling_bytes(est)
        assert ceiling == est * EVENT_BYTES * STORE_SIZE_MULTIPLIER
        under = int(ceiling / EVENT_BYTES) - 1
        over = int(ceiling / EVENT_BYTES) + 1000
        assert not exceeds_size_ceiling(under, est)
        assert exceeds_size_ceiling(over, est)

    def test_floor_protects_small_captures(self):
        from src.services.circuit_capture_service import (
            EVENT_BYTES, exceeds_size_ceiling, size_ceiling_bytes)
        # Tiny estimate → 64 MB floor, not 5×tiny.
        assert size_ceiling_bytes(10) == 64 * 2**20
        # A modest real capture under the floor is NOT aborted.
        assert not exceeds_size_ceiling(1_000_000, 10)


class TestActivationMass:
    """R2 B-6/B-7: vectorized per-feature activation mass (replaces 131k-wide
    Python loops)."""

    def test_mass_matches_per_feature_sum(self, store):
        ev, _, _ = open_writers(store, 13)
        ev.append(np.array([0, 0, 1]), np.array([1, 2, 3]),
                  np.array([5, 5, 7]), np.array([1.0, 2.0, 4.0]))
        ev.finalize()
        r = EventReader(store, 13)
        mass = r.feature_activation_mass()
        assert mass[5] == pytest.approx(3.0)   # 1.0 + 2.0
        assert mass[7] == pytest.approx(4.0)
        assert 999 not in mass

    def test_empty_store(self, store):
        ev, _, _ = open_writers(store, 13)
        ev.finalize()
        assert EventReader(store, 13).feature_activation_mass() == {}
