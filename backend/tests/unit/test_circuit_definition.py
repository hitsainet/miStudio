"""
Pins for mistudio.circuit-definition/v1 (018 Task 2.1/2.2, IDL-33):
structure validators, per-layer caps, edge-endpoint integrity, round-trip
fidelity, projection slicing validity + markers.
"""

import json

import pytest

from src.schemas.circuit_definition import (
    CIRCUIT_KIND,
    CircuitBudget,
    CircuitDefinitionV1,
    CircuitEdge,
    CircuitMember,
    CircuitNodeRef,
    EdgeAttribution,
    EdgeCoactivation,
    to_layer_slice,
)
from src.schemas.cluster_profile import (
    DefinitionSAERef,
    ProfileBudget,
    ProfileMember,
)
from src.schemas.evidence_ladder import EvidenceRung


def _member(layer, idx, **kw):
    return CircuitMember(
        layer=layer,
        feature=ProfileMember(feature_idx=idx, strength=kw.pop("strength", 0.5), **kw),
    )


def _defn(**overrides):
    base = dict(
        name="Support voice circuit",
        narrative="L13 condolences drives L14 reassurance.",
        saes=[
            DefinitionSAERef(mistudio_sae_id="sae_l13", layer=13, n_features=8192),
            DefinitionSAERef(mistudio_sae_id="sae_l14", layer=14, n_features=8192),
        ],
        members=[_member(13, 712, label="condolences"), _member(14, 231, label="reassurance")],
        edges=[
            CircuitEdge(
                up=CircuitNodeRef(layer=13, feature_idx=712),
                down=CircuitNodeRef(layer=14, feature_idx=231),
                rung=EvidenceRung.ATTRIBUTION_SUPPORTED,
                coactivation=EdgeCoactivation(pmi=2.1, support=44, null_percentile=99.4),
                attribution=EdgeAttribution(score=0.31, sign_consistency=0.9, method="raw"),
                weight_prior=0.12,
            )
        ],
        budget=CircuitBudget(layers={"13": ProfileBudget(B=1.0), "14": ProfileBudget(B=2.4)}),
    )
    base.update(overrides)
    return CircuitDefinitionV1(**base)


class TestStructure:
    def test_valid_definition_builds(self):
        d = _defn()
        assert d.kind == CIRCUIT_KIND
        assert d.displayed_rung() == EvidenceRung.ATTRIBUTION_SUPPORTED

    def test_sae_ref_requires_layer(self):
        with pytest.raises(Exception, match="carry its layer"):
            _defn(saes=[DefinitionSAERef(mistudio_sae_id="x")])

    def test_member_layer_needs_sae_ref(self):
        with pytest.raises(Exception, match="no SAE ref"):
            _defn(members=[_member(10, 5), _member(13, 712), _member(14, 231)])

    def test_per_layer_cap_not_total(self):
        # 20 at L13 + 20 at L14 = 40 total is LEGAL (BR-025)…
        members = [_member(13, i) for i in range(20)] + [_member(14, i) for i in range(20)]
        d = _defn(members=members, edges=[])
        assert len(d.members) == 40
        # …but 21 at one layer is not.
        with pytest.raises(Exception, match="per-layer member cap"):
            _defn(members=[_member(13, i) for i in range(21)], edges=[])

    def test_cluster_ref_expansion_counts_toward_layer_cap(self):
        cluster = CircuitMember(
            layer=13, member_kind="cluster_ref", cluster_profile_id="clp_x",
            cluster_name="fear",
            expanded_members=[ProfileMember(feature_idx=i, strength=0.2) for i in range(15)],
        )
        with pytest.raises(Exception, match="per-layer member cap"):
            _defn(members=[cluster] + [_member(13, 100 + i) for i in range(6)], edges=[])

    def test_edge_must_ascend_layers(self):
        with pytest.raises(Exception, match="lower layer to a higher layer"):
            CircuitEdge(up=CircuitNodeRef(layer=14, feature_idx=1),
                        down=CircuitNodeRef(layer=13, feature_idx=2))

    def test_edge_endpoints_must_be_members(self):
        with pytest.raises(Exception, match="does not reference a circuit member"):
            _defn(edges=[CircuitEdge(up=CircuitNodeRef(layer=13, feature_idx=9999),
                                     down=CircuitNodeRef(layer=14, feature_idx=231))])

    def test_edge_can_target_cluster_expanded_feature(self):
        cluster = CircuitMember(
            layer=13, member_kind="cluster_ref", cluster_profile_id="clp_x",
            cluster_name="fear",
            expanded_members=[ProfileMember(feature_idx=712, strength=0.2)],
        )
        d = _defn(members=[cluster, _member(14, 231)])
        assert d.edges[0].up.feature_idx == 712

    def test_cluster_ref_requires_expansion_snapshot(self):
        with pytest.raises(Exception, match="expanded_members"):
            CircuitMember(layer=13, member_kind="cluster_ref", cluster_profile_id="clp_x")


class TestRoundTrip:
    def test_json_round_trip_semantic_equality(self):
        d = _defn()
        restored = CircuitDefinitionV1.model_validate(json.loads(d.model_dump_json()))
        assert restored.model_dump() == d.model_dump()
        assert restored.edges[0].rung == EvidenceRung.ATTRIBUTION_SUPPORTED
        assert restored.edges[0].tested_and_failed == []

    def test_rung_history_survives_round_trip(self):
        d = _defn()
        d.edges[0].tested_and_failed = [EvidenceRung.CAUSALLY_VALIDATED]
        restored = CircuitDefinitionV1.model_validate(d.model_dump())
        assert restored.edges[0].tested_and_failed == [EvidenceRung.CAUSALLY_VALIDATED]

    def test_position_fields_nullable_and_preserved(self):
        d = _defn()
        assert d.edges[0].position is None  # Tier-2.5-ready, empty today
        d.edges[0].position = {"mediating_heads": [3, 7]}
        restored = CircuitDefinitionV1.model_validate(d.model_dump())
        assert restored.edges[0].position.mediating_heads == [3, 7]


class TestProjection:
    def test_slice_is_valid_v1_with_markers(self):
        d = _defn()
        s = to_layer_slice(d, 14)
        # Valid v1 by construction (model built); markers display-only.
        assert s.kind == "mistudio.cluster-definition"
        assert s.members[0].feature_idx == 231
        assert s.sae.layer == 14
        assert "partial_rendering=true" in (s.provenance.source_note or "")
        assert "parent_rung=1" in s.provenance.source_note
        assert "[L14 slice]" in s.name

    def test_slice_budget_carries_global_intensity(self):
        d = _defn()
        d.budget.intensity = 1.5
        s = to_layer_slice(d, 13)
        assert s.budget.B == 1.0 and s.budget.intensity == 1.5

    def test_slice_expands_cluster_members(self):
        cluster = CircuitMember(
            layer=13, member_kind="cluster_ref", cluster_profile_id="clp_x",
            cluster_name="fear",
            expanded_members=[ProfileMember(feature_idx=1, strength=0.2),
                              ProfileMember(feature_idx=2, strength=0.3)],
        )
        d = _defn(members=[cluster, _member(14, 231)], edges=[])
        s = to_layer_slice(d, 13)
        assert [m.feature_idx for m in s.members] == [1, 2]

    def test_slice_missing_layer_raises(self):
        with pytest.raises(ValueError, match="no SAE ref for layer"):
            to_layer_slice(_defn(), 10)
