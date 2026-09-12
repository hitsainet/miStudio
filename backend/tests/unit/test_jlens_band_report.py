"""
Band-report metrics, boundary derivation, and the Phase-0 gate.

Two product rules are asserted structurally here rather than trusted:

  * bands are EARNED — no default exists anywhere, so a model whose profile
    cannot support a split gets no boundaries and the product draws nothing;
  * a NO-GO is a RESULT — it constructs, records and blocks, because a gate
    that can only say yes is not a gate.

The null controls get their own tests because each is the difference between a
finding and an arithmetic artifact: adjacent positions share context, so raw
autocorrelation is high for a model with no cross-position structure at all;
and any k directions explain some variance, so raw FVE says nothing until the
random-direction control says what k random ones explain.

MUTATION CONTROLS (each must turn this file red):
  * give derive_boundaries a default when the profile is flat -> "no bands" fails
  * return the raw autocorrelation as the finding                -> "null" fails
  * return raw FVE instead of the excess                         -> "excess" fails
  * default control_seed                                         -> "seed required" fails
  * make decide_gate accept an empty rationale                   -> "rationale" fails
  * treat GO_AT_LARGER_SCALE as non-blocking                     -> "blocking" fails
  * reference agreement in a gate condition                      -> "BR-004" fails
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

from src.ml.jlens_metrics import (
    LayerProfile,
    cross_layer_cka,
    derive_boundaries,
    effective_dimensionality,
    excess_fve,
    excess_kurtosis,
    fraction_variance_explained,
    linear_cka,
    occupancy,
    shuffled_null_autocorrelation,
    top1_autocorrelation,
)
from src.services.jlens_band_report import (
    BandReport,
    GateDecision,
    build_band_report,
    decide_gate,
)


# ------------------------------------------------------------------ metrics


def test_excess_kurtosis_is_zero_for_a_normal_sample():
    torch.manual_seed(0)
    assert excess_kurtosis(torch.randn(20000)) == pytest.approx(0.0, abs=0.15)


def test_excess_kurtosis_is_positive_for_a_sharpened_distribution():
    torch.manual_seed(1)
    heavy = torch.randn(20000) * torch.where(torch.rand(20000) < 0.05, 6.0, 0.3)
    assert excess_kurtosis(heavy) > 3.0


def test_effective_dimensionality_counts_directions_not_rank():
    """Rank reports full for almost any real matrix — the same answer for every
    model. The participation ratio distinguishes them."""
    concentrated = torch.zeros(10, 10)
    concentrated[0, 0] = 10.0
    concentrated += torch.eye(10) * 1e-3
    assert effective_dimensionality(concentrated) < 2.0

    spread = torch.eye(10)
    assert effective_dimensionality(spread) == pytest.approx(10.0, abs=0.01)


def test_linear_cka_is_one_for_a_rotation_and_lower_for_unrelated():
    torch.manual_seed(2)
    x = torch.randn(64, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-6)
    assert linear_cka(x, torch.randn(64, 8)) < 0.5


def test_cross_layer_cka_is_symmetric_with_a_unit_diagonal():
    torch.manual_seed(3)
    reps = {0: torch.randn(32, 6), 1: torch.randn(32, 6)}
    grid = cross_layer_cka(reps)
    assert grid[0][0] == pytest.approx(1.0, abs=1e-6)
    assert grid[0][1] == pytest.approx(grid[1][0], abs=1e-9)


def test_occupancy_is_bounded_by_the_budget():
    assert occupancy([4, 4, 4], k=4) == pytest.approx(1.0)
    assert occupancy([2, 2], k=4) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        occupancy([1], k=0)


# ------------------------------------------------------------- null controls


def test_autocorrelation_null_is_what_makes_the_raw_figure_meaningful():
    """A repeated token gives a high RAW autocorrelation and the null matches it.

    Without the comparison this reads as cross-position structure when it is
    the arithmetic consequence of one token repeating.
    """
    repeated = [7] * 20
    raw = top1_autocorrelation(repeated)
    null = shuffled_null_autocorrelation(repeated, seed=11)

    assert raw == pytest.approx(1.0)
    assert null == pytest.approx(1.0)

    profile = LayerProfile(layer=0, autocorrelation=raw, autocorrelation_null=null)
    assert profile.excess_autocorrelation == pytest.approx(0.0, abs=1e-9)


def test_real_position_structure_survives_the_null():
    structured = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    raw = top1_autocorrelation(structured)
    null = shuffled_null_autocorrelation(structured, seed=12)
    assert raw - null > 0.2


def test_excess_fve_is_the_finding_not_the_raw_figure():
    """k random directions explain a surprising amount of variance.

    A report claiming "these directions explain X% of the variance" says
    nothing until it says what X random ones explain.
    """
    torch.manual_seed(4)
    d_model, k = 32, 8
    activations = torch.randn(256, d_model)

    random_dirs = torch.randn(k, d_model)
    raw = fraction_variance_explained(activations, random_dirs)
    assert raw > 0.15, "control directions explain nothing; fixture is degenerate"

    excess = excess_fve(activations, random_dirs, control_seed=5)
    assert abs(excess) < 0.1, "random directions should show ~no excess"


def test_excess_fve_is_positive_for_genuinely_aligned_directions():
    torch.manual_seed(6)
    d_model, k = 32, 4
    basis = torch.linalg.qr(torch.randn(d_model, k)).Q.T
    coeffs = torch.randn(256, k)
    activations = coeffs @ basis + torch.randn(256, d_model) * 0.05
    assert excess_fve(activations, basis, control_seed=7) > 0.5


def test_control_seed_is_required_not_defaulted():
    """The excess is the finding, so an unreconstructible control invalidates it."""
    params = inspect.signature(excess_fve).parameters
    assert params["control_seed"].default is inspect.Parameter.empty

    params = inspect.signature(build_band_report).parameters
    assert params["control_seed"].default is inspect.Parameter.empty


# --------------------------------------------------------------- boundaries


def flat_profiles(n=8):
    return [LayerProfile(layer=i, kurtosis=1.0) for i in range(n)]


def structured_profiles():
    # Rises through the middle, peaks late — the shape a band split describes.
    values = [0.1, 0.1, 0.2, 1.5, 2.0, 2.4, 3.0, 6.0, 2.0, 1.0]
    return [LayerProfile(layer=i, kurtosis=v) for i, v in enumerate(values)]


def test_a_flat_profile_yields_no_bands():
    """The honest answer, and what keeps unearned shading off the screen."""
    assert derive_boundaries(flat_profiles()) is None


def test_too_few_layers_yields_no_bands():
    assert derive_boundaries([LayerProfile(layer=i, kurtosis=float(i)) for i in range(3)]) is None


def test_a_profile_with_no_kurtosis_yields_no_bands():
    assert derive_boundaries([LayerProfile(layer=i) for i in range(8)]) is None


def test_boundaries_come_from_this_models_own_profile():
    bounds = derive_boundaries(structured_profiles())
    assert bounds is not None
    # Median of the profile is 1.75; L4 (2.0) is the first layer above it, and
    # L7 (6.0) is the peak. Computed from the fixture rather than asserted from
    # intuition — the first draft of this test guessed L3 and was wrong.
    assert bounds["workspace_start"] == 4
    assert bounds["motor_start"] == 7
    # The published figures, on a model with 10 layers, would be nonsense.
    assert bounds["workspace_start"] != 40 and bounds["motor_start"] != 90


def test_boundaries_track_the_profile_rather_than_a_constant():
    """Shift the profile and the boundaries must move with it.

    A single-shape test passes against a hardcoded pair — the "fixtures agree
    by construction" trap that let two mutations survive in the viewer.
    """
    shifted = [
        LayerProfile(layer=i, kurtosis=v)
        for i, v in enumerate([0.1, 0.1, 0.1, 0.1, 0.1, 1.5, 2.0, 3.0, 8.0, 1.0])
    ]
    a = derive_boundaries(structured_profiles())
    b = derive_boundaries(shifted)
    assert a != b
    assert b["workspace_start"] == 5 and b["motor_start"] == 8


def test_report_without_boundaries_says_bands_are_not_shown():
    report = build_band_report("m1", flat_profiles(), control_seed=3)
    assert report.has_bands is False
    assert report.boundaries is None
    assert "do not transfer" in report.derivation


def test_report_with_boundaries_states_how_they_were_derived():
    report = build_band_report("m1", structured_profiles(), control_seed=3)
    assert report.has_bands is True
    assert "this model's own" in report.derivation


#: The published boundaries. Measured on one specific model; BR-002 says they
#: must be impossible to port here BY CONSTRUCTION.
_PUBLISHED_BOUNDARIES = (38, 40, 90, 92)


def _jlens_modules():
    """Every jlens module, DISCOVERED — not a hand-list.

    MIS-E2E-090. This guard used to name two modules, `jlens_metrics` and
    `jlens_band_report`. Mutation M13 put the literals in
    `jlens_band_service.py`, a sibling it did not scan, and the suite stayed
    green. A guard whose scope is narrower than its claim is the shape this
    audit found three times (BR-002's two modules, `EXPECTED_CALLS`' 16 of 116,
    `REQUIRED_TABLES`' 17 of 36).
    """
    import importlib
    import pkgutil

    import src.ml as ml_pkg
    import src.services as svc_pkg

    found = []
    for pkg in (ml_pkg, svc_pkg):
        for info in pkgutil.iter_modules(pkg.__path__):
            if "jlens" not in info.name:
                continue
            found.append(importlib.import_module(f"{pkg.__name__}.{info.name}"))
    assert len(found) >= 6, (
        f"only {len(found)} jlens modules discovered — the scan is broken, and "
        f"a broken scan agrees with everything"
    )
    return found


def _embedded_numbers(module):
    """Numeric and string constants reachable in the COMPILED module.

    Compiling rather than reading the AST is what closes the second half of
    MIS-E2E-090. Mutation M12 wrote the same constants as `4 * 10` and
    `int("90")`: no `ast.Constant` node holds 40 or 90, so an AST scan for
    literals saw nothing. CPython folds `4 * 10` to `40` at compile time, and
    `int("90")` leaves `"90"` in `co_consts`, so both are visible here.
    """
    code = compile(inspect.getsource(module), module.__name__, "exec")
    seen = []

    def walk(c):
        for const in c.co_consts:
            if isinstance(const, (int, float)) and not isinstance(const, bool):
                seen.append(const)
            elif isinstance(const, str):
                try:
                    seen.append(int(const))
                except ValueError:
                    pass
            elif hasattr(const, "co_consts"):
                walk(const)

    walk(code)
    return seen


def test_no_band_constant_exists_in_any_jlens_module():
    """BR-002: porting must be impossible BY CONSTRUCTION, not by policy."""
    for module in _jlens_modules():
        for value in _embedded_numbers(module):
            assert value not in _PUBLISHED_BOUNDARIES, (
                f"{module.__name__} embeds {value}, one of the published band "
                f"boundaries. Those were measured on a different model; a "
                f"reader who sees them here will believe they transfer."
            )


def test_the_scan_sees_a_computed_constant():
    """Negative control in-test: an AST-literal scan would miss these."""
    import types

    probe = types.ModuleType("probe")
    probe_src = "def f():\n    return 4 * 10, int('90')\n"
    probe.__dict__["__source__"] = probe_src

    code = compile(probe_src, "probe", "exec")
    found = []

    def walk(c):
        for const in c.co_consts:
            if isinstance(const, (int, float)) and not isinstance(const, bool):
                found.append(const)
            elif isinstance(const, str):
                try:
                    found.append(int(const))
                except ValueError:
                    pass
            elif hasattr(const, "co_consts"):
                walk(const)

    walk(code)
    assert 40 in found, "the folder no longer catches `4 * 10`"
    assert 90 in found, "the folder no longer catches `int('90')`"


# --------------------------------------------------------------------- gate


def report_for_gate():
    return build_band_report("m1", structured_profiles(), control_seed=3)


def test_go_when_the_claim_set_replicates():
    record = decide_gate("m1", report_for_gate(), True, False, "all four claims replicate")
    assert record.decision is GateDecision.GO
    assert record.is_blocking() is False


def test_no_go_is_a_first_class_recordable_outcome():
    """A gate that can only say yes is not a gate."""
    record = decide_gate(
        "m1", report_for_gate(), False, False, "selectivity did not replicate"
    )
    assert record.decision is GateDecision.NO_GO
    assert record.is_blocking() is True
    assert record.rationale
    assert record.band_report.model_id == "m1"


def test_go_at_larger_scale_still_blocks_at_this_scale():
    """The distinction that makes it worth having rather than a softened GO."""
    record = decide_gate(
        "m1", report_for_gate(), False, True, "band structure present, capacity limits absent"
    )
    assert record.decision is GateDecision.GO_AT_LARGER_SCALE
    assert record.is_blocking() is True


def test_a_decision_without_a_rationale_is_refused():
    with pytest.raises(ValueError, match="rationale"):
        decide_gate("m1", report_for_gate(), True, False, "   ")


def test_a_no_go_model_can_still_have_no_bands():
    """The two are independent: bands describe geometry, the gate describes claims."""
    record = decide_gate(
        "m1", build_band_report("m1", flat_profiles(), control_seed=3), False, False, "flat"
    )
    assert record.decision is GateDecision.NO_GO
    assert record.band_report.has_bands is False


def test_gate_never_scores_next_token_agreement():
    """BR-004, enforced rather than documented.

    The J-lens is deliberately worse than the logit lens on agreement through
    most of the network, so a gate rewarding it fails good models and passes
    bad ones. Agreement is DESCRIBED in the per-layer profile and appears in no
    condition here.
    """
    from src.services import jlens_band_report

    tree = ast.parse(inspect.getsource(jlens_band_report))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            label = node.id if isinstance(node, ast.Name) else node.attr
            assert "agreement" not in label.lower(), (
                f"{label!r} in the gate's executable path — agreement may be "
                "described but never scored (BR-004)"
            )

    # ...and it IS available to describe, on the profile.
    assert "next_token_agreement" in LayerProfile.__dataclass_fields__
