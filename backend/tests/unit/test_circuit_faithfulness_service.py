"""Faithfulness pins (017 Phase 4): member expansion, score assembly,
necessity-only mode, manifest payload records the metric identity."""

import pytest

from src.services.circuit_faithfulness_service import (
    CircuitFaithfulnessService,
    FaithfulnessConfigError,
    expand_members,
    scores_from_behaviors,
)


class TestConfig:
    def test_bad_mode(self):
        with pytest.raises(FaithfulnessConfigError):
            CircuitFaithfulnessService.create_config({"mode": "bogus"})

    def test_defaults(self):
        c = CircuitFaithfulnessService.create_config({})
        assert c["mode"] == "both"
        assert c["metric_id"] == "compare_output_shift/v1"


class TestMemberExpansion:
    def test_feature_and_cluster_members(self):
        members = [
            {"layer": 13, "feature": {"feature_idx": 5}},
            {"layer": 14, "member_kind": "cluster_ref", "cluster_profile_id": "clp"},
        ]
        by_layer = expand_members(members, lambda pid: [1, 2, 3])
        assert by_layer[13] == [5]
        assert by_layer[14] == [1, 2, 3]


class TestScores:
    def test_necessity_and_sufficiency(self):
        s = scores_from_behaviors(1.0, 0.2, 0.0, 0.9, mode="both")
        assert s["necessity"] == 0.8
        assert s["sufficiency"] == 0.9

    def test_necessity_only_marks_sufficiency_untested(self):
        s = scores_from_behaviors(1.0, 0.2, 0.0, None, mode="necessity")
        assert s["necessity"] == 0.8
        assert s["sufficiency"] is None
        assert "untested" in s["sufficiency_status"]


class TestManifest:
    def test_records_metric_and_proxy(self):
        cfg = CircuitFaithfulnessService.create_config({"ablate_all_n": 512})
        p = CircuitFaithfulnessService.build_manifest_payload(
            "crc_1", cfg, {"necessity": 0.8},
            {"b_clean": 1.0, "b_ablate_m": 0.2})
        assert p["metric_id"] == "compare_output_shift/v1"  # always recorded
        assert p["ablate_all_proxy"]["per_layer_top_n"] == 512
        assert p["circuit_id"] == "crc_1"

    def test_manifest_from_run_carries_scores_and_metric_definition(self):
        """The manifest a run would persist carries necessity/sufficiency AND
        the exact behavior-metric definition (the number is never trusted
        blind) — pinning the run's assembly without a GPU."""
        cfg = CircuitFaithfulnessService.create_config({"mode": "both"})
        behaviors = {"b_clean": 1.0, "b_ablate_m": 0.2,
                     "b_ablate_all": 0.0, "b_ablate_nonmembers": 0.9}
        scores = scores_from_behaviors(
            behaviors["b_clean"], behaviors["b_ablate_m"],
            behaviors["b_ablate_all"], behaviors["b_ablate_nonmembers"],
            mode="both")
        p = CircuitFaithfulnessService.build_manifest_payload(
            "crc_9", cfg, scores, behaviors,
            provenance={"down_layer": 14, "n_prompts_used": 12})
        assert p["scores"]["necessity"] == 0.8
        assert p["scores"]["sufficiency"] == 0.9
        assert p["metric_id"] == "compare_output_shift/v1"
        # metric definition is spelled out (honesty: what B actually is)
        assert "downstream-most members" in p["metric_definition"]
        # self-contained keys the manifest validator requires
        assert {"intervention", "config", "seeds"} <= set(p)
        assert p["provenance"]["down_layer"] == 14

    def test_manifest_validates_as_self_contained(self):
        """The faithfulness payload passes the manifest self-containment gate
        (no filesystem paths, required keys present)."""
        from src.services.manifest_service import validate_payload
        cfg = CircuitFaithfulnessService.create_config({})
        p = CircuitFaithfulnessService.build_manifest_payload(
            "crc_1", cfg, {"necessity": 0.5, "sufficiency": None},
            {"b_clean": 1.0, "b_ablate_m": 0.5, "b_ablate_all": 0.0,
             "b_ablate_nonmembers": None})
        validate_payload("faithfulness", p)  # raises if not self-contained


class TestMemberExpansionWithResolver:
    def test_cluster_ref_expands_via_profile_resolver(self):
        """A realistic resolver (queries a ClusterProfile's members) turns a
        cluster_ref member into its feature indices, keyed by layer."""
        profiles = {
            "clp_a": [{"feature_idx": 11}, {"feature_idx": 22}],
            "clp_b": [{"feature_idx": 7}],
        }

        def resolve(pid):
            return [m["feature_idx"] for m in profiles.get(pid, [])]

        members = [
            {"layer": 5, "member_kind": "cluster_ref",
             "cluster_profile_id": "clp_a"},
            {"layer": 9, "feature": {"feature_idx": 3}},
            {"layer": 9, "member_kind": "cluster_ref",
             "cluster_profile_id": "clp_b"},
        ]
        by_layer = expand_members(members, resolve)
        assert by_layer[5] == [11, 22]
        assert sorted(by_layer[9]) == [3, 7]
