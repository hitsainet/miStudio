"""
Contract tests for POST /steering/cluster-allocation (Feature 013).

The endpoint is CPU-only glue: resolve the SAE, load the decoder via the shared
resolver (degrading to the approximate G=1 path on failure), delegate to the
pure math core, map ValueError → 400. These tests exercise the glue with the
service layer mocked — the math itself is covered in test_cluster_allocation.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from src.api.v1.endpoints.steering import compute_cluster_strength_allocation
from src.schemas.steering import ClusterAllocationMember, ClusterAllocationRequest
from src.services import steering_hazards as steering_hazards_mod
from fastapi import HTTPException


def _mk_sae(n_features=100, status="ready", local_path="saes/x"):
    sae = MagicMock()
    sae.status = status
    sae.local_path = local_path
    sae.n_features = n_features
    sae.layer = 12
    sae.d_model = 16
    sae.architecture = "jumprelu"
    return sae


def _req(idxs=(0, 1), layer=12, **kw):
    return ClusterAllocationRequest(
        sae_id="sae_x",
        members=[
            ClusterAllocationMember(feature_idx=i, layer=layer, similarity=0.8, activation_frequency=0.2)
            for i in idxs
        ],
        **kw,
    )


@pytest.mark.asyncio
async def test_404_when_sae_missing():
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=AsyncMock(return_value=None)
    ):
        with pytest.raises(HTTPException) as e:
            await compute_cluster_strength_allocation(_req(), db=MagicMock())
        assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_400_when_index_out_of_bounds():
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_mk_sae(n_features=5)),
    ):
        with pytest.raises(HTTPException) as e:
            await compute_cluster_strength_allocation(_req(idxs=(0, 99)), db=MagicMock())
        assert e.value.status_code == 400
        assert "out of bounds" in str(e.value.detail)


@pytest.mark.asyncio
async def test_422_mixed_layers_single_sae_is_layer_mismatch():
    """Feature 015: mixed layers is now a MULTI-LAYER request. With no per-member
    sae_id both members route to the request-level SAE (layer 12), so the layer-13
    member mismatches its SAE → 422 listing offenders (was 400 pre-015)."""
    req = ClusterAllocationRequest(
        sae_id="sae_x",
        members=[
            ClusterAllocationMember(feature_idx=0, layer=12),
            ClusterAllocationMember(feature_idx=1, layer=13),
        ],
    )
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_mk_sae()),  # always layer 12
    ), patch(
        "src.api.v1.endpoints.steering.get_steering_service"
    ) as svc:
        svc.return_value.load_sae = AsyncMock(side_effect=RuntimeError("skip decoder"))
        with pytest.raises(HTTPException) as e:
            await compute_cluster_strength_allocation(req, db=MagicMock())
        assert e.value.status_code == 422
        assert e.value.detail["code"] == "sae_layer_mismatch"
        offenders = e.value.detail["offenders"]
        assert any(o["feature_idx"] == 1 and o["layer"] == 13 for o in offenders)


@pytest.mark.asyncio
async def test_decoder_failure_degrades_to_approximate():
    """Loader errors must not 500 — the allocation degrades to G=1/approximate."""
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_mk_sae()),
    ), patch(
        "src.api.v1.endpoints.steering.load_sae_weights_cpu",
        side_effect=RuntimeError("boom"),
    ), patch(
        "src.api.v1.endpoints.steering.settings"
    ) as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        st.steering_cluster_constants_json = "{}"
        resp = await compute_cluster_strength_allocation(_req(), db=MagicMock())
    assert resp.approximate is True
    assert resp.G == 1.0
    assert "approximate" in resp.flags
    assert resp.B == pytest.approx(resp.B_dir, abs=1e-6)


@pytest.mark.asyncio
async def test_happy_path_with_decoder_and_flags():
    sae = _mk_sae()
    dec = torch.zeros(16, 100)
    dec[0, 0] = 1.0
    dec[0, 1] = 1.0  # identical directions → G = 1
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=AsyncMock(return_value=sae)
    ), patch(
        # single-layer now loads the decoder on CPU via load_sae_weights_cpu
        # (R2 F5 — no GPU load on the read-only endpoint).
        "src.api.v1.endpoints.steering.load_sae_weights_cpu",
        return_value=(dec, torch.zeros(100, 16)),
    ), patch(
        "src.api.v1.endpoints.steering.settings"
    ) as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        st.steering_cluster_constants_json = "{}"
        resp = await compute_cluster_strength_allocation(
            _req(group_cohesion=0.3), db=MagicMock()
        )
    assert resp.approximate is False
    assert resp.G == pytest.approx(1.0, abs=1e-3)
    assert resp.B == pytest.approx(resp.B_dir, abs=1e-2)
    assert "low_cohesion" in resp.flags  # cohesion 0.3 < gate 0.5
    assert resp.formula_id == "freq-budget/sim-alloc@1"
    assert len(resp.strengths) == 2


@pytest.mark.asyncio
async def test_400_when_sae_not_ready():
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_mk_sae(status="downloading")),
    ):
        with pytest.raises(HTTPException) as e:
            await compute_cluster_strength_allocation(_req(), db=MagicMock())
        assert e.value.status_code == 400


# ── Feature 015: multi-layer branch ─────────────────────────────────────────

def _mk_sae_layer(layer, sae_id, n_features=100):
    sae = MagicMock()
    sae.status = "ready"
    sae.local_path = f"saes/{sae_id}"
    sae.n_features = n_features
    sae.layer = layer
    sae.d_model = 16
    sae.architecture = "jumprelu"
    return sae


@pytest.mark.asyncio
async def test_multi_layer_returns_per_layer_response_and_hazards():
    """Two layers with matching per-member SAEs → MultiLayerAllocationResponse
    with a layers map, request-order strengths, and (approximate) no crash."""
    from src.schemas.steering import MultiLayerAllocationResponse

    req = ClusterAllocationRequest(
        sae_id="sae_A",
        members=[
            ClusterAllocationMember(feature_idx=1, layer=13, similarity=0.8,
                                    activation_frequency=0.2, sae_id="sae_A"),
            ClusterAllocationMember(feature_idx=2, layer=14, similarity=0.8,
                                    activation_frequency=0.2, sae_id="sae_B"),
        ],
    )
    saes = {"sae_A": _mk_sae_layer(13, "sae_A"), "sae_B": _mk_sae_layer(14, "sae_B")}

    async def get_sae(db, sid):
        return saes[sid]

    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=get_sae
    ), patch(
        "src.api.v1.endpoints.steering.get_steering_service"
    ) as svc, patch(
        "src.api.v1.endpoints.steering.settings"
    ) as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=False))
        st.steering_cluster_constants_json = "{}"
        st.steering_hazard_prior_threshold = 0.5
        # Decoder loading skipped (path doesn't exist) → approximate G=1.
        resp = await compute_cluster_strength_allocation(req, db=MagicMock())

    assert isinstance(resp, MultiLayerAllocationResponse)
    assert resp.formula_id == "freq-budget/sim-alloc/per-layer@1"
    assert set(resp.layers) == {"13", "14"}
    assert resp.layers["13"].sae_id == "sae_A"
    assert resp.layers["14"].sae_id == "sae_B"
    assert len(resp.strengths) == 2
    # No decoder/encoder loaded → no heuristic hazards.
    assert resp.hazards == []


@pytest.mark.asyncio
async def test_multi_layer_422_when_member_sae_wrong_layer():
    """A member on layer 14 routed to a layer-13 SAE → 422 listing offenders."""
    req = ClusterAllocationRequest(
        sae_id="sae_A",
        members=[
            ClusterAllocationMember(feature_idx=1, layer=13, sae_id="sae_A"),
            ClusterAllocationMember(feature_idx=2, layer=14, sae_id="sae_A"),  # wrong
        ],
    )
    # sae_A is layer 13; the layer-14 member mismatches.
    async def get_sae(db, sid):
        return _mk_sae_layer(13, "sae_A")

    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=get_sae
    ), patch(
        "src.api.v1.endpoints.steering.get_steering_service"
    ) as svc, patch(
        "src.api.v1.endpoints.steering.settings"
    ) as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=False))
        st.steering_cluster_constants_json = "{}"
        st.steering_hazard_prior_threshold = 0.5
        with pytest.raises(HTTPException) as e:
            await compute_cluster_strength_allocation(req, db=MagicMock())
    assert e.value.status_code == 422
    assert e.value.detail["code"] == "sae_layer_mismatch"
    assert any(o["feature_idx"] == 2 for o in e.value.detail["offenders"])


@pytest.mark.asyncio
async def test_multi_layer_validated_hazard_from_circuit_edges():
    """R2 F9 — the ARC PAYOFF seam: a circuit_id whose stored edges carry a
    rung-2 validated edge (with effect_size) produces a QUANTIFIED hazard,
    not just the heuristic. This is 017's validated ES reaching 015's steering."""
    from src.schemas.steering import MultiLayerAllocationResponse

    req = ClusterAllocationRequest(
        sae_id="sae_A", circuit_id="crc_1",
        members=[
            ClusterAllocationMember(feature_idx=1, layer=13, similarity=0.8,
                                    activation_frequency=0.2, sae_id="sae_A"),
            ClusterAllocationMember(feature_idx=2, layer=14, similarity=0.8,
                                    activation_frequency=0.2, sae_id="sae_B"),
        ],
    )
    saes = {"sae_A": _mk_sae_layer(13, "sae_A"), "sae_B": _mk_sae_layer(14, "sae_B")}

    async def get_sae(db, sid):
        return saes[sid]

    # the circuit's stored edges: a rung-2 validated edge 13:1 → 14:2, ES=0.8
    validated_edges = [{
        "up": {"layer": 13, "feature_idx": 1},
        "down": {"layer": 14, "feature_idx": 2},
        "rung": 2, "effect_size": 0.8,
    }]

    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=get_sae
    ), patch(
        "src.api.v1.endpoints.steering.get_steering_service"
    ), patch(
        "src.api.v1.endpoints.steering._load_circuit_edges",
        new=AsyncMock(return_value=validated_edges),
    ), patch(
        "src.api.v1.endpoints.steering.settings"
    ) as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=False))
        st.steering_cluster_constants_json = "{}"
        st.steering_hazard_prior_threshold = 0.5
        resp = await compute_cluster_strength_allocation(req, db=MagicMock())

    assert isinstance(resp, MultiLayerAllocationResponse)
    assert len(resp.hazards) == 1
    h = resp.hazards[0]
    assert h.evidence.startswith("validated:ES=")   # quantified, not heuristic
    assert h.rung == 2
    assert h.quantified_effect == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_a_CLUSTER_LEVEL_circuit_edge_reaches_the_hazard_analysis():
    """BR-016 wiring: cluster edges are expanded before hazards are detected.

    A capability is not shipped until a test fails when its wiring is removed.
    This drives the real endpoint with a real cluster-level edge and a real
    profile, and requires the resolved feature pair to arrive at
    `detect_hazards`. Deleting the `expand_cluster_edges` call turns it red.

    Asserted on the ARGUMENT, not on "was called": a reachability test that only
    checks arrival passes against a call handing over the unexpanded edge, which
    is exactly the defect.
    """
    from src.schemas.steering import MultiLayerAllocationResponse

    req = ClusterAllocationRequest(
        sae_id="sae_A",
        circuit_id="crc_test",
        members=[
            ClusterAllocationMember(feature_idx=1, layer=13, similarity=0.8,
                                    activation_frequency=0.2, sae_id="sae_A"),
            ClusterAllocationMember(feature_idx=2, layer=14, similarity=0.8,
                                    activation_frequency=0.2, sae_id="sae_B"),
        ],
    )
    saes = {"sae_A": _mk_sae_layer(13, "sae_A"), "sae_B": _mk_sae_layer(14, "sae_B")}

    async def get_sae(db, sid):
        return saes[sid]

    cluster_edge = {
        "up": {"layer": 13, "cluster_profile_id": "cp_up", "feature_idx": None},
        "down": {"layer": 14, "cluster_profile_id": "cp_down", "feature_idx": None},
        "rung": 2,
        "effect_size": 0.9,
    }

    async def load_edges(db, cid):
        return [dict(cluster_edge)]

    async def resolve(db, ids):
        return {"cp_up": [1], "cp_down": [2]}

    seen = {}
    real_detect = steering_hazards_mod.detect_hazards

    def spy(steered, circuit_edges=None, **kw):
        seen["edges"] = circuit_edges
        return real_detect(steered, circuit_edges=circuit_edges, **kw)

    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=get_sae
    ), patch(
        "src.api.v1.endpoints.steering._load_circuit_edges", new=load_edges
    ), patch(
        "src.api.v1.endpoints.steering._resolve_cluster_members", new=resolve
    ), patch(
        "src.api.v1.endpoints.steering.get_steering_service"
    ), patch.object(
        steering_hazards_mod, "detect_hazards", new=spy
    ), patch(
        "src.api.v1.endpoints.steering.settings"
    ) as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=False))
        st.steering_cluster_constants_json = "{}"
        st.steering_hazard_prior_threshold = 0.5
        resp = await compute_cluster_strength_allocation(req, db=MagicMock())

    assert isinstance(resp, MultiLayerAllocationResponse)
    delivered = seen.get("edges") or []
    assert delivered, "no circuit edges reached detect_hazards at all"
    assert all(
        e["up"].get("feature_idx") is not None
        and e["down"].get("feature_idx") is not None
        for e in delivered
    ), (
        "a cluster-level edge arrived at detect_hazards unexpanded — its key "
        f"can never match the feature-level steered list: {delivered}"
    )
    assert (delivered[0]["up"]["feature_idx"], delivered[0]["down"]["feature_idx"]) == (1, 2)
    assert delivered[0]["rung"] == 2 and delivered[0]["effect_size"] == 0.9
    # And the warning is actually produced, which is the point of the expansion.
    assert resp.hazards, "the validated cluster edge produced no hazard"


@pytest.mark.asyncio
async def test_an_UNRESOLVABLE_cluster_edge_is_reported_not_silently_dropped():
    """"No hazards" must not be reportable when nothing could be checked."""
    req = ClusterAllocationRequest(
        sae_id="sae_A",
        circuit_id="crc_test",
        members=[
            ClusterAllocationMember(feature_idx=1, layer=13, sae_id="sae_A"),
            ClusterAllocationMember(feature_idx=2, layer=14, sae_id="sae_B"),
        ],
    )
    saes = {"sae_A": _mk_sae_layer(13, "sae_A"), "sae_B": _mk_sae_layer(14, "sae_B")}

    async def get_sae(db, sid):
        return saes[sid]

    async def load_edges(db, cid):
        return [{
            "up": {"layer": 13, "cluster_profile_id": "cp_gone", "feature_idx": None},
            "down": {"layer": 14, "cluster_profile_id": "cp_gone2", "feature_idx": None},
            "rung": 2, "effect_size": 0.9,
        }]

    async def resolve(db, ids):
        return {}          # both profiles deleted

    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=get_sae
    ), patch(
        "src.api.v1.endpoints.steering._load_circuit_edges", new=load_edges
    ), patch(
        "src.api.v1.endpoints.steering._resolve_cluster_members", new=resolve
    ), patch(
        "src.api.v1.endpoints.steering.get_steering_service"
    ), patch(
        "src.api.v1.endpoints.steering.settings"
    ) as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=False))
        st.steering_cluster_constants_json = "{}"
        st.steering_hazard_prior_threshold = 0.5
        resp = await compute_cluster_strength_allocation(req, db=MagicMock())

    assert resp.hazards == []
    assert resp.unchecked_edges, (
        "an edge that could not be checked was dropped silently — an empty "
        "hazard list then reads as 'safe' when it means 'not analysed'"
    )
    assert "could not be checked" in resp.unchecked_edges[0]["reason"]
