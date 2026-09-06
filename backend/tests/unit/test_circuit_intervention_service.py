"""GPU-free pins for the intervention service: edge selection, σ_d from store,
write-back annotation (017 Phase 3)."""

import numpy as np
import pytest

from src.services.circuit_capture_store import open_writers, EventReader
from src.services.circuit_intervention_service import (
    CircuitInterventionService,
    InterventionConfigError,
    sigma_d_from_store,
)


class TestScope:
    def test_valid(self):
        s = CircuitInterventionService.create_scope({"ordering": "attr", "k": 5})
        assert s["ordering"] == "attr" and s["k"] == 5

    def test_bad_ordering(self):
        with pytest.raises(InterventionConfigError):
            CircuitInterventionService.create_scope({"ordering": "bogus"})

    def test_bad_k(self):
        with pytest.raises(InterventionConfigError):
            CircuitInterventionService.create_scope({"k": 0})


class TestTopKSelection:
    def _cands(self):
        return [
            {"up": {"layer": 13, "feature_idx": 1}, "down": {}, "orderings": {"coact_rank": 2, "attr_rank": 0}},
            {"up": {"layer": 13, "feature_idx": 2}, "down": {}, "orderings": {"coact_rank": 0, "attr_rank": 2}},
            {"up": {"layer": 13, "feature_idx": 3}, "down": {}, "orderings": {"coact_rank": 1, "attr_rank": 1}},
        ]

    def test_coact_ordering(self):
        top = CircuitInterventionService.top_k_edges(self._cands(), "coact", 2)
        assert [c["up"]["feature_idx"] for c in top] == [2, 3]  # coact_rank 0,1

    def test_attr_ordering(self):
        top = CircuitInterventionService.top_k_edges(self._cands(), "attr", 2)
        assert [c["up"]["feature_idx"] for c in top] == [1, 3]  # attr_rank 0,1

    def test_missing_rank_sorts_last(self):
        cands = [{"up": {"feature_idx": 9}, "down": {}, "orderings": {}}]
        top = CircuitInterventionService.top_k_edges(cands, "attr", 5)
        assert len(top) == 1


class TestSigmaD:
    def test_sd_from_store(self, tmp_path):
        ev, en, _ = open_writers(tmp_path, 14)
        ev.append(np.zeros(4, dtype=np.uint32), np.arange(4),
                  np.full(4, 2), np.array([1.0, 2.0, 3.0, 4.0]))
        ev.finalize()
        r = EventReader(tmp_path, 14)
        sd = sigma_d_from_store(r, 2)
        assert sd == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))

    def test_lone_firing_returns_one(self, tmp_path):
        ev, en, _ = open_writers(tmp_path, 14)
        ev.append(np.zeros(1, dtype=np.uint32), np.array([0]),
                  np.array([2]), np.array([1.0]))
        ev.finalize()
        assert sigma_d_from_store(EventReader(tmp_path, 14), 2) == 1.0


class TestWriteBack:
    def test_annotates_candidates_with_validation(self):
        class _Run:
            candidates = [
                {"up": {"layer": 13, "feature_idx": 1},
                 "down": {"layer": 14, "feature_idx": 2}},
                {"up": {"layer": 13, "feature_idx": 9},
                 "down": {"layer": 14, "feature_idx": 8}},
            ]
        run = _Run()
        results = [{
            "up": {"layer": 13, "feature_idx": 1},
            "down": {"layer": 14, "feature_idx": 2},
            "effect_size": 2.5, "rung": 2, "tested_and_failed": False,
            "verdict": {"passed": True, "reason": "ok"}}]
        CircuitInterventionService._write_back(run, results, "attr", "vman_x")
        c0 = run.candidates[0]
        assert c0["validation"]["passed"] is True
        assert c0["validation"]["manifest_id"] == "vman_x"
        assert c0["validated_rung"] == 2
        # untouched candidate has no validation
        assert "validation" not in run.candidates[1]

    def test_failed_records_history_not_demotion(self):
        class _Run:
            candidates = [{"up": {"layer": 13, "feature_idx": 1},
                           "down": {"layer": 14, "feature_idx": 2}}]
        run = _Run()
        results = [{
            "up": {"layer": 13, "feature_idx": 1},
            "down": {"layer": 14, "feature_idx": 2},
            "effect_size": 0.1, "rung": None, "tested_and_failed": True,
            "verdict": {"passed": False, "reason": "below null"}}]
        CircuitInterventionService._write_back(run, results, "coact", "vman_y")
        c = run.candidates[0]
        assert c["validation"]["passed"] is False
        assert "validated_rung" not in c  # never demotes
        assert c["tested_and_failed_history"][0]["reason"] == "below null"
