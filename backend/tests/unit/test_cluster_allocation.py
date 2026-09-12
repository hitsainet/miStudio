"""
Unit tests for the cluster strength allocation math (Feature 013, IDL-29).

Covers the FTDD §2 formula steps, the three sanity anchors that ground the
design (identical ⇒ B=B_dir; orthogonal ⇒ B=B_dir·√N; N=1 ⇒ solo law), and
every row of the FTDD §3 edge-case table that the pure core owns.
"""

import math

import pytest
import torch

from src.services.cluster_allocation_service import (
    AllocationMember,
    DEFAULT_B_DIR,
    DEFAULT_CONSTANTS,
    G_FLOOR,
    compute_allocation,
    rebalance,
    resolve_constants,
)


def members(*specs):
    """specs: (idx, sim, freq[, sign]) tuples on layer 12."""
    out = []
    for s in specs:
        idx, sim, freq = s[0], s[1], s[2]
        sign = s[3] if len(s) > 3 else 1
        out.append(AllocationMember(feature_idx=idx, layer=12, similarity=sim,
                                    activation_frequency=freq, sign=sign))
    return out


def identical_decoder(n_features: int, d_model: int = 16, cols=None):
    """Decoder where every listed column is the SAME unit vector."""
    dec = torch.randn(d_model, n_features)
    v = torch.zeros(d_model)
    v[0] = 1.0
    for c in (cols or range(n_features)):
        dec[:, c] = v
    return dec


def orthogonal_decoder(n_features: int, d_model: int = 16):
    """Decoder whose first columns are orthonormal basis vectors."""
    dec = torch.randn(d_model, n_features)
    for c in range(min(n_features, d_model)):
        e = torch.zeros(d_model)
        e[c] = 1.0
        dec[:, c] = e
    return dec


# ── Sanity anchors ──────────────────────────────────────────────────────────

def test_identical_members_budget_equals_b_dir():
    """Perfectly aligned members act as one direction: B = B_dir, split equally."""
    ms = members((0, 0.8, 0.2), (1, 0.8, 0.2), (2, 0.8, 0.2), (3, 0.8, 0.2))
    dec = identical_decoder(8)
    r = compute_allocation(ms, decoder=dec)
    expected_b_dir = 2.9 - 2.6 * 0.2  # 2.38
    assert r.B_dir == pytest.approx(expected_b_dir, abs=1e-3)
    assert r.G == pytest.approx(1.0, abs=1e-3)
    assert r.B == pytest.approx(expected_b_dir, abs=1e-2)
    # Equal sims ⇒ equal strengths (BR-004 letter)
    assert len(set(r.strengths)) == 1
    assert sum(r.strengths) == pytest.approx(r.B, abs=0.06)  # 0.1 grain tolerance


def test_orthogonal_members_budget_default_gamma0():
    """Fitted default γ=0 (2026-07-16 validation): B = B_dir regardless of G —
    members saturate on Σ|s| independent of decoder-direction cancellation."""
    n = 4
    ms = members(*[(i, 0.5, 0.1) for i in range(n)])
    dec = orthogonal_decoder(8)
    r = compute_allocation(ms, decoder=dec)
    assert r.G == pytest.approx(1.0 / math.sqrt(n), abs=1e-3)  # still measured/reported
    assert r.B == pytest.approx(2.9 - 2.6 * 0.1, rel=0.02)
    assert r.constants_used.get("gamma") == 0.0


def test_orthogonal_members_gamma1_override_scales_sqrt_n():
    """Legacy γ=1 stays available per-SAE: G = 1/√N ⇒ B = B_dir·√N."""
    n = 4
    ms = members(*[(i, 0.5, 0.1) for i in range(n)])
    dec = orthogonal_decoder(8)
    c = dict(DEFAULT_CONSTANTS, gamma=1.0)
    r = compute_allocation(ms, decoder=dec, constants=c)
    b_dir = 2.9 - 2.6 * 0.1
    expected = min(b_dir * math.sqrt(n), sum([2.9 - 2.6 * 0.1] * n))
    assert r.B == pytest.approx(expected, rel=0.02)


def test_single_member_reduces_to_solo_law():
    """N=1 with known frequency: B == b(f) exactly (G=1 for a unit column)."""
    ms = members((3, 0.9, 0.3))
    dec = identical_decoder(8)
    r = compute_allocation(ms, decoder=dec)
    solo = 2.9 - 2.6 * 0.3
    assert r.B == pytest.approx(solo, abs=0.01)
    assert r.strengths[0] == pytest.approx(round(solo, 1), abs=0.06)


# ── Weights & frequency edge cases ──────────────────────────────────────────

def test_equal_sims_equal_strengths_uneven_freqs():
    ms = members((0, 1.0, 0.05), (1, 1.0, 0.45))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    assert r.weights == [0.5, 0.5]
    assert abs(r.strengths[0]) == abs(r.strengths[1])


def test_sim_proportional_weights():
    ms = members((0, 0.9, 0.2), (1, 0.45, 0.2))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    assert r.weights[0] == pytest.approx(2 * r.weights[1], rel=1e-3)


def test_missing_sim_imputes_mean():
    ms = members((0, 0.8, 0.2), (1, None, 0.2), (2, 0.4, 0.2))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    # imputed sim = mean(0.8, 0.4) = 0.6 → weights 0.8 : 0.6 : 0.4
    assert r.weights[1] == pytest.approx(0.6 / 1.8, abs=1e-3)


def test_all_sims_missing_uniform():
    ms = members((0, None, 0.2), (1, None, 0.2))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    assert r.weights == [0.5, 0.5]


def test_zero_sim_member_inactive():
    ms = members((0, 0.8, 0.2), (1, 0.0, 0.2))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    assert r.strengths[1] == 0.0
    assert "inactive_member" in r.flags


def test_all_freqs_missing_default_budget():
    ms = members((0, 0.8, None), (1, 0.8, None))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    assert r.B_dir == DEFAULT_B_DIR
    assert "default_budget" in r.flags
    assert r.f_eff is None


def test_partial_freqs_renormalized():
    ms = members((0, 1.0, 0.4), (1, 1.0, None))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    assert r.f_eff == pytest.approx(0.4, abs=1e-3)  # only the known member counts


def test_out_of_range_freq_excluded():
    ms = members((0, 1.0, 1.7), (1, 1.0, 0.3))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    assert r.f_eff == pytest.approx(0.3, abs=1e-3)


# ── Gain, signs, cancellation, caps ─────────────────────────────────────────

def test_signed_member_uses_signed_weight_in_gain():
    """A suppressive member on the same direction cancels in the resultant."""
    ms = members((0, 0.8, 0.2, 1), (1, 0.8, 0.2, -1))
    dec = identical_decoder(4)
    r = compute_allocation(ms, decoder=dec)
    # +w and −w on the same unit vector ⇒ resultant ~0 ⇒ G floored
    assert r.G <= G_FLOOR + 1e-6 or r.G == pytest.approx(0.0, abs=1e-3)
    # No cancellation warning: the negative sign is explicit user intent
    assert "cancellation" not in r.flags


def test_cancellation_flag_same_sign_opposed_decoders():
    d_model, nf = 8, 4
    dec = torch.zeros(d_model, nf)
    dec[0, 0] = 1.0
    dec[0, 1] = -1.0  # opposed to feature 0
    ms = members((0, 0.8, 0.2), (1, 0.8, 0.2))
    r = compute_allocation(ms, decoder=dec)
    assert "cancellation" in r.flags
    assert set(r.cancellation_pair) == {0, 1}


def test_orthogonal_many_stays_under_cap():
    """(γ=1 override) √N growth stays below Σ solo caps for orthogonal members."""
    n = 9
    ms = members(*[(i, 1.0, 0.0) for i in range(n)])  # b*(0)=2.9 each, B_dir=2.9
    dec = orthogonal_decoder(16)
    r = compute_allocation(ms, decoder=dec, constants=dict(DEFAULT_CONSTANTS, gamma=1.0))
    assert r.B == pytest.approx(2.9 * 3, rel=0.02)  # B_dir·√9
    assert "cap_bound" not in r.flags


def test_budget_cap_binds_under_near_cancellation():
    """Near-cancelling pair floors G → B_dir/G_FLOOR would explode; the Σb* cap
    binds instead, keeping the allocation conservative."""
    d_model, nf = 8, 4
    dec = torch.zeros(d_model, nf)
    dec[0, 0] = 1.0
    dec[0, 1] = -1.0
    ms = members((0, 0.8, 0.2), (1, 0.8, 0.2))
    r = compute_allocation(ms, decoder=dec, constants=dict(DEFAULT_CONSTANTS, gamma=1.0))
    cap = 2 * (2.9 - 2.6 * 0.2)  # Σ b*(0.2) = 4.76
    assert r.B == pytest.approx(cap, abs=1e-2)
    assert "cap_bound" in r.flags
    assert "cancellation" in r.flags
    # Default γ=0: no G explosion at all — B = B_dir, cap does not bind.
    r0 = compute_allocation(ms, decoder=dec)
    assert r0.B == pytest.approx(2.9 - 2.6 * 0.2, abs=1e-2)
    assert "cap_bound" not in r0.flags


def test_mixed_layer_refused():
    ms = [AllocationMember(0, 12, 0.5, 0.1), AllocationMember(1, 13, 0.5, 0.1)]
    with pytest.raises(ValueError, match="mixed-layer"):
        compute_allocation(ms, decoder=identical_decoder(4))


def test_index_out_of_bounds_refused():
    ms = members((99, 0.5, 0.1))
    with pytest.raises(ValueError, match="out of bounds"):
        compute_allocation(ms, decoder=identical_decoder(4))


def test_decoder_unavailable_approximate_g1():
    ms = members((0, 0.8, 0.2), (1, 0.8, 0.2))
    r = compute_allocation(ms, decoder=None)
    assert r.approximate is True
    assert r.G == 1.0
    assert "approximate" in r.flags
    assert r.B == pytest.approx(r.B_dir, abs=1e-6)  # constant-budget fallback


def test_low_cohesion_flag():
    ms = members((0, 0.8, 0.2), (1, 0.8, 0.2))
    r = compute_allocation(ms, decoder=identical_decoder(4), group_cohesion=0.3)
    assert "low_cohesion" in r.flags
    r2 = compute_allocation(ms, decoder=identical_decoder(4), group_cohesion=0.9)
    assert "low_cohesion" not in r2.flags


def test_empty_members_refused():
    with pytest.raises(ValueError, match="non-empty"):
        compute_allocation([], decoder=None)


# ── Rounding & budget conservation ──────────────────────────────────────────

def test_rounding_residual_folded():
    ms = members((0, 1.0, 0.2), (1, 1.0, 0.2), (2, 1.0, 0.2))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    assert sum(abs(s) for s in r.strengths) == pytest.approx(r.B, abs=0.06)


# ── Iteration-1 review regressions ──────────────────────────────────────────

def test_fold_never_flips_signs_n20():
    """PROVEN bug: n=20, B≈3 → old single-member fold produced −0.8 on a boost
    member and Σ|s| 53% over budget. The greedy distribution must never flip."""
    n = 20
    ms = members(*[(i, 1.0, 0.0) for i in range(n)])
    dec = identical_decoder(24)
    r = compute_allocation(ms, decoder=dec)
    assert all(s >= 0 for s in r.strengths), r.strengths
    assert sum(abs(s) for s in r.strengths) == pytest.approx(r.B, abs=0.1)


def test_fold_no_single_member_budget_dump():
    """n=20 with tiny per-member shares must not dump the budget on one member."""
    n = 20
    ms = members(*[(i, 1.0, 0.48) for i in range(n)])  # B_dir≈1.65
    dec = identical_decoder(24)
    r = compute_allocation(ms, decoder=dec)
    assert max(abs(s) for s in r.strengths) <= 0.3  # ~B/n + a grain


def test_cancellation_checked_on_positive_subset_with_suppressor_present():
    """A deliberate suppressor must not disable the cancellation check for the
    positive members (old code skipped the check when ANY sign was −1)."""
    d_model, nf = 8, 4
    dec = torch.zeros(d_model, nf)
    dec[0, 0] = 1.0
    dec[0, 1] = -1.0   # positive members 0,1 oppose each other
    dec[1, 2] = 1.0    # suppressor on an orthogonal direction
    ms = members((0, 0.8, 0.2), (1, 0.8, 0.2), (2, 0.8, 0.2, -1))
    r = compute_allocation(ms, decoder=dec)
    assert "cancellation" in r.flags
    assert set(r.cancellation_pair) == {0, 1}


def test_zero_and_missing_sims_impute_true_mean():
    """Present zeros count toward the imputation mean (FTDD: mean of PRESENT sims)."""
    ms = members((0, 0.0, 0.2), (1, 0.0, 0.2), (2, None, 0.2))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    # mean(0,0)=0 → imputed 0 → all zero → uniform weights, NOT all-budget-to-unknown
    assert r.weights == [pytest.approx(1 / 3, abs=1e-3)] * 3
    assert "uniform_weights" in r.flags


def test_n1_unknown_frequency_matches_solo_default():
    """MCP/UI parity: a single feature with unknown frequency gets the solo 10."""
    ms = members((3, 0.9, None))
    r = compute_allocation(ms, decoder=identical_decoder(8))
    assert r.B == 10.0
    assert r.strengths == [10.0]
    assert "solo" in r.flags


def test_f_eff_falls_back_to_unweighted_mean_when_known_f_has_zero_weight():
    ms = members((0, 0.0, 0.3), (1, 0.8, None))
    r = compute_allocation(ms, decoder=identical_decoder(4))
    # Known f lives only on the zero-weight member → unweighted mean fallback
    assert r.f_eff == pytest.approx(0.3, abs=1e-3)
    assert "default_budget" not in r.flags


def test_nonunit_decoder_flagged_and_raw_norms_used():
    """G must be computed on RAW columns (hook parity): a 2x-norm column doubles G."""
    d_model, nf = 8, 4
    dec = torch.zeros(d_model, nf)
    dec[0, 0] = 2.0  # non-unit
    ms = members((0, 0.9, 0.2))
    r = compute_allocation(ms, decoder=dec, constants=dict(DEFAULT_CONSTANTS, gamma=1.0))
    # raw single column of norm 2 → G=2 → (γ=1) B = b(0.2)/2
    assert "nonunit_decoder" in r.flags
    assert r.G == pytest.approx(2.0, abs=1e-3)
    assert r.B == pytest.approx((2.9 - 2.6 * 0.2) / 2.0, abs=0.01)
    # Default γ=0: raw-norm G still measured + flagged, but B stays B_dir.
    r0 = compute_allocation(ms, decoder=dec)
    assert r0.G == pytest.approx(2.0, abs=1e-3)
    assert "nonunit_decoder" in r0.flags
    assert r0.B == pytest.approx(2.9 - 2.6 * 0.2, abs=0.01)


# ── Rebalance reference implementation ──────────────────────────────────────

def test_rebalance_preserves_budget():
    strengths, weights, signs = [0.8, 0.8, 0.8], [1 / 3] * 3, [1, 1, 1]
    pinned = [True, False, False]
    out, flags = rebalance([1.2, 0.8, 0.8], weights, signs, pinned, total_budget=2.4)
    assert out[0] == 1.2  # pin untouched
    assert sum(abs(s) for s in out) == pytest.approx(2.4, abs=0.1)
    assert flags == []


def test_rebalance_over_budget_zeroes_unpinned():
    out, flags = rebalance([2.0, 1.0, 1.0], [1 / 3] * 3, [1, 1, 1],
                           [True, True, False], total_budget=2.4)
    assert out[2] == 0.0
    assert "over_budget" in flags
    assert out[0] == 2.0 and out[1] == 1.0  # pins never rescaled


def test_rebalance_all_pinned_noop():
    out, flags = rebalance([1.0, 1.0], [0.5, 0.5], [1, 1], [True, True], total_budget=1.0)
    assert out == [1.0, 1.0]


# ── Constants resolution ─────────────────────────────────────────────────────

def test_resolve_constants_layering():
    cfg = '{"default": {"a": 3.5}, "per_sae": {"sae_x": {"b": 1.0}}}'
    c = resolve_constants(cfg, "sae_x")
    assert c["a"] == 3.5 and c["b"] == 1.0 and c["M"] == 3.0
    c2 = resolve_constants(cfg, "other")
    assert c2["b"] == 2.6
    c3 = resolve_constants("not json", "sae_x")
    assert c3 == DEFAULT_CONSTANTS
