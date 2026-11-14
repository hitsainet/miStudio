"""Hazard-v2 matrix (015 Task 3.2, BR-024): validated-edge quantification
(primary), weight-prior heuristic (fallback, labeled), sign → compounding vs
cancellation, and the copy discipline that heuristic warnings never claim
causal (IDL-35)."""

import pytest

torch = pytest.importorskip("torch")

from src.services.steering_hazards import (
    Hazard,
    detect_hazards,
    weight_prior,
)


def _feat(layer, idx, strength=50.0):
    return {"layer": layer, "feature_idx": idx, "strength": strength}


def _edge(ul, ui, dl, di, rung, es):
    return {"up": {"layer": ul, "feature_idx": ui},
            "down": {"layer": dl, "feature_idx": di},
            "rung": rung, "effect_size": es}


class TestValidatedEdgePrimary:
    def test_rung2_positive_same_sign_is_compounding_quantified(self):
        h = detect_hazards(
            [_feat(13, 1, 50), _feat(14, 2, 50)],
            circuit_edges=[_edge(13, 1, 14, 2, 2, 0.8)])
        assert len(h) == 1
        assert h[0].type == "compounding"
        assert h[0].quantified_effect == pytest.approx(0.8)
        assert h[0].evidence.startswith("validated:ES=")
        assert h[0].rung == 2

    def test_rung2_negative_edge_flips_to_cancellation(self):
        h = detect_hazards(
            [_feat(13, 1, 50), _feat(14, 2, 50)],
            circuit_edges=[_edge(13, 1, 14, 2, 2, -0.7)])
        assert h[0].type == "cancellation"

    def test_opposite_steering_signs_with_positive_edge_cancels(self):
        h = detect_hazards(
            [_feat(13, 1, 50), _feat(14, 2, -50)],
            circuit_edges=[_edge(13, 1, 14, 2, 2, 0.8)])
        assert h[0].type == "cancellation"

    def test_rung1_edge_does_not_quantify(self):
        # rung < 2 → no validated evidence; falls through to (absent) heuristic
        h = detect_hazards(
            [_feat(13, 1), _feat(14, 2)],
            circuit_edges=[_edge(13, 1, 14, 2, 1, None)])
        assert h == []


class TestWeightPriorFallback:
    def test_high_prior_warns_labeled_heuristic(self):
        d_model, d_sae = 8, 16
        # make feature 1's decoder direction ≈ feature 2's encoder direction
        v = torch.randn(d_model)
        dec = torch.randn(d_model, d_sae); dec[:, 1] = v
        enc = torch.randn(d_sae, d_model); enc[2, :] = v
        h = detect_hazards(
            [_feat(13, 1, 50), _feat(14, 2, 50)],
            decoders={13: dec}, encoders={14: enc}, prior_threshold=0.5)
        assert len(h) == 1
        assert h[0].evidence.startswith("heuristic:weight_prior=")
        assert h[0].rung == 0
        assert h[0].quantified_effect is None

    def test_low_prior_no_warning(self):
        d_model, d_sae = 8, 16
        dec = torch.zeros(d_model, d_sae); dec[0, 1] = 1.0
        enc = torch.zeros(d_sae, d_model); enc[2, 1] = 1.0  # orthogonal to dec[:,1]
        h = detect_hazards(
            [_feat(13, 1), _feat(14, 2)],
            decoders={13: dec}, encoders={14: enc}, prior_threshold=0.5)
        assert h == []

    def test_validated_edge_takes_priority_over_prior(self):
        d_model, d_sae = 8, 16
        v = torch.randn(d_model)
        dec = torch.randn(d_model, d_sae); dec[:, 1] = v
        enc = torch.randn(d_sae, d_model); enc[2, :] = v
        h = detect_hazards(
            [_feat(13, 1), _feat(14, 2)],
            circuit_edges=[_edge(13, 1, 14, 2, 2, 0.9)],
            decoders={13: dec}, encoders={14: enc})
        assert len(h) == 1 and h[0].evidence.startswith("validated:")


class TestStructure:
    def test_only_upstream_to_downstream_pairs(self):
        # same layer / downstream→upstream never form a hazard
        h = detect_hazards([_feat(14, 2), _feat(13, 1)],
                           circuit_edges=[_edge(13, 1, 14, 2, 2, 0.8)])
        assert len(h) == 1 and h[0].up["layer"] == 13

    def test_weight_prior_orientation(self):
        # a feature's decoder vs its OWN encoder (tied-ish) → cos near 1
        d_model, d_sae = 8, 16
        v = torch.randn(d_model)
        dec = torch.zeros(d_model, d_sae); dec[:, 3] = v
        enc = torch.zeros(d_sae, d_model); enc[3, :] = v
        assert weight_prior(dec, 3, enc, 3) == pytest.approx(1.0, abs=1e-5)


class TestCopyDiscipline:
    def test_heuristic_evidence_never_says_causal(self):
        d_model, d_sae = 8, 16
        v = torch.randn(d_model)
        dec = torch.randn(d_model, d_sae); dec[:, 1] = v
        enc = torch.randn(d_sae, d_model); enc[2, :] = v
        h = detect_hazards([_feat(13, 1), _feat(14, 2)],
                           decoders={13: dec}, encoders={14: enc})
        for hz in h:
            assert "causal" not in hz.evidence.lower()
            assert "heuristic" in hz.evidence


class TestR1Fixes:
    def test_cluster_ref_edges_skipped(self):
        # R1 QA-2: an edge with feature_idx=None (cluster_ref) must be skipped,
        # not crash / silently mis-key.
        h = detect_hazards(
            [_feat(13, 1), _feat(14, 2)],
            circuit_edges=[{"up": {"layer": 13, "feature_idx": None},
                            "down": {"layer": 14, "feature_idx": None},
                            "rung": 2, "effect_size": 0.9}])
        assert h == []  # no feature-level edge matched; no crash

    def test_cluster_ref_members_skipped(self):
        # a steered member with no feature_idx is filtered out
        h = detect_hazards(
            [{"layer": 13, "strength": 50}, _feat(14, 2)],  # no feature_idx on first
            circuit_edges=[_edge(13, 1, 14, 2, 2, 0.9)])
        assert h == []

    def test_out_of_range_prior_index_no_crash(self):
        import torch
        # R1 #5: weight_prior with an out-of-range index → 0.0, not IndexError
        dec = torch.randn(8, 4)   # only 4 features
        enc = torch.randn(4, 8)
        assert weight_prior(dec, 99, enc, 0) == 0.0
        assert weight_prior(dec, 0, enc, 99) == 0.0

    def test_duplicate_members_dedup(self):
        # R1 #6: the same (up,down) pair from duplicate members warns ONCE
        h = detect_hazards(
            [_feat(13, 1, 50), _feat(13, 1, 50), _feat(14, 2, 50)],
            circuit_edges=[_edge(13, 1, 14, 2, 2, 0.8)])
        assert len(h) == 1
