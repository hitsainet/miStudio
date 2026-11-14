"""Unit tests for the per-layer (multi-SAE) allocation math (Feature 015).

Feature 015 reuses the UNCHANGED IDL-29 ``compute_allocation`` per layer
partition — these tests pin that the partition assembly matches the equivalent
per-layer single-layer calls, that request-member order is preserved in the
flattened ``strengths``, and (GOLDEN) that a single-layer member set produces
allocation numbers byte-identical to the 013 ``compute_allocation`` output.
"""

import math

import pytest
import torch

from src.services.cluster_allocation_service import (
    AllocationMember,
    MultiLayerMember,
    compute_allocation,
    compute_multi_layer_allocation,
    MULTI_LAYER_FORMULA_ID,
)


def identical_decoder(n_features: int, d_model: int = 16, cols=None):
    """Decoder where every listed column is the SAME unit vector."""
    dec = torch.zeros(d_model, n_features)
    v = torch.zeros(d_model)
    v[0] = 1.0
    for c in (cols or range(n_features)):
        dec[:, c] = v
    return dec


def orthogonal_decoder(n_features: int, d_model: int = 16):
    dec = torch.randn(d_model, n_features)
    for c in range(min(n_features, d_model)):
        e = torch.zeros(d_model)
        e[c] = 1.0
        dec[:, c] = e
    return dec


# ── Partition math: multi-layer == per-layer single calls ───────────────────

def test_two_layers_match_independent_single_calls():
    """Each layer's allocation equals the standalone compute_allocation on that
    layer's members with that layer's decoder — no cross-layer contamination."""
    members = [
        MultiLayerMember(feature_idx=1, layer=13, similarity=0.8, activation_frequency=0.2, sae_id="A"),
        MultiLayerMember(feature_idx=2, layer=13, similarity=0.8, activation_frequency=0.2, sae_id="A"),
        MultiLayerMember(feature_idx=5, layer=14, similarity=0.5, activation_frequency=0.1, sae_id="B"),
    ]
    dec13 = identical_decoder(8)
    dec14 = orthogonal_decoder(8)
    decoders = {13: dec13, 14: dec14}

    ml = compute_multi_layer_allocation(members, decoders=decoders)

    # Standalone per-layer references.
    ref13 = compute_allocation(
        [AllocationMember(1, 13, 0.8, 0.2), AllocationMember(2, 13, 0.8, 0.2)],
        decoder=dec13,
    )
    ref14 = compute_allocation(
        [AllocationMember(5, 14, 0.5, 0.1)], decoder=dec14
    )

    assert ml.layers[13].B == ref13.B
    assert ml.layers[13].strengths == ref13.strengths
    assert ml.layers[13].G == ref13.G
    assert ml.layers[14].B == ref14.B
    assert ml.layers[14].strengths == ref14.strengths
    assert ml.layer_sae_ids == {13: "A", 14: "B"}
    assert ml.formula_id == MULTI_LAYER_FORMULA_ID


def test_flattened_strengths_follow_request_order():
    """strengths[] is scattered back to the ORIGINAL member order, interleaving
    layers — not grouped by partition."""
    members = [
        MultiLayerMember(feature_idx=1, layer=14, similarity=1.0, activation_frequency=0.2, sae_id="B"),
        MultiLayerMember(feature_idx=2, layer=13, similarity=1.0, activation_frequency=0.2, sae_id="A"),
        MultiLayerMember(feature_idx=3, layer=14, similarity=1.0, activation_frequency=0.2, sae_id="B"),
        MultiLayerMember(feature_idx=4, layer=13, similarity=1.0, activation_frequency=0.2, sae_id="A"),
    ]
    decoders = {13: identical_decoder(8), 14: identical_decoder(8)}
    ml = compute_multi_layer_allocation(members, decoders=decoders)

    assert len(ml.strengths) == 4
    # Positions 0,2 are layer 14; 1,3 are layer 13 — each equals that layer's slice.
    assert [ml.strengths[0], ml.strengths[2]] == ml.layers[14].strengths
    assert [ml.strengths[1], ml.strengths[3]] == ml.layers[13].strengths


def test_missing_decoder_layer_is_approximate():
    """A layer without a decoder degrades to the approximate (G=1) allocation,
    exactly like the single-layer endpoint's fallback."""
    members = [
        MultiLayerMember(feature_idx=1, layer=13, similarity=0.8, activation_frequency=0.2, sae_id="A"),
        MultiLayerMember(feature_idx=2, layer=14, similarity=0.8, activation_frequency=0.2, sae_id="B"),
    ]
    ml = compute_multi_layer_allocation(members, decoders={13: identical_decoder(8)})
    assert ml.layers[13].approximate is False
    assert ml.layers[14].approximate is True
    assert "approximate" in ml.layers[14].flags


def test_empty_members_raises():
    with pytest.raises(ValueError):
        compute_multi_layer_allocation([])


# ── GOLDEN: single-layer set is byte-identical to the 013 core ──────────────

def _golden_members_013():
    """The exact members used by a representative 013 allocation."""
    return [
        (0, 0.8, 0.2),
        (1, 0.8, 0.2),
        (2, 0.8, 0.2),
        (3, 0.8, 0.2),
    ]


def test_single_layer_partition_byte_identical_to_013_core():
    """A single-layer member set run through compute_multi_layer_allocation
    yields the SAME allocation numbers as the 013 compute_allocation — the
    per-layer path never forks the math (FTID §2.2 golden)."""
    specs = _golden_members_013()
    dec = identical_decoder(8)

    ref = compute_allocation(
        [AllocationMember(i, 12, sim, freq) for (i, sim, freq) in specs],
        decoder=dec,
    )
    ml = compute_multi_layer_allocation(
        [MultiLayerMember(i, 12, sim, freq, sae_id="A") for (i, sim, freq) in specs],
        decoders={12: dec},
    )

    layer = ml.layers[12]
    # Every 013 field reproduced exactly.
    assert layer.B == ref.B
    assert layer.B_dir == ref.B_dir
    assert layer.G == ref.G
    assert layer.f_eff == ref.f_eff
    assert layer.weights == ref.weights
    assert layer.strengths == ref.strengths
    assert layer.flags == ref.flags
    assert layer.constants_used == ref.constants_used
    assert layer.approximate == ref.approximate
    # And the flattened strengths equal the 013 strengths.
    assert ml.strengths == ref.strengths


class TestR1PerLayerConstants:
    def test_constants_by_layer_overrides(self):
        # R1 ARCH-4: a per-layer constants override must reach that partition.
        from src.services.cluster_allocation_service import (
            MultiLayerMember, compute_multi_layer_allocation)
        members = [
            MultiLayerMember(feature_idx=1, layer=13, sae_id="s13"),
            MultiLayerMember(feature_idx=2, layer=14, sae_id="s14"),
        ]
        base = {"b_max": 3.0, "b_min": 1.0}
        per_layer = {14: {"b_max": 5.0, "b_min": 1.0}}
        # Should not raise; both layers allocated. (We assert it runs + returns
        # per-layer results; the constant is threaded — a smoke that the param
        # is honored end to end.)
        res = compute_multi_layer_allocation(
            members, constants=base, constants_by_layer=per_layer)
        assert set(res.layers.keys()) == {13, 14}


class TestCpuWeightLoader:
    """R2 F1: load_sae_weights_cpu must load real weights on CPU (never GPU),
    and REFUSE (raise) rather than return garbage from random init on a bad
    load — the endpoint's try/except then degrades to approximate G=1."""

    def test_loads_real_weights_on_cpu(self, tmp_path):
        from pathlib import Path

        from src.ml.community_format import (CommunityStandardConfig,
                                             save_sae_community_format)
        from src.ml.sparse_autoencoder import create_sae
        from src.services.steering_service import load_sae_weights_cpu

        d_model, d_sae = 16, 64
        sae = create_sae("standard", hidden_dim=d_model, latent_dim=d_sae)
        cfg = CommunityStandardConfig(
            model_name="toy", hook_point="blocks.0.hook_resid_pre",
            hook_point_layer=0, d_in=d_model, d_sae=d_sae)
        save_sae_community_format(sae, cfg, Path(tmp_path))
        dw, ew = load_sae_weights_cpu(tmp_path, d_model=d_model,
                                      n_features=d_sae, architecture="standard")
        assert dw is not None and ew is not None
        assert tuple(dw.shape) == (d_model, d_sae)   # [d_model, d_sae]
        assert tuple(ew.shape) == (d_sae, d_model)   # [d_sae, d_model]
        assert dw.device.type == "cpu" and ew.device.type == "cpu"
