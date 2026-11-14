"""MIS-E2E-088 — the measurement BR-002 depends on.

BR-002 is this product's load-bearing honesty rule: bands render **only** from a
measured band report, never from a constant, so that no borrowed boundary is
presented as measured. These three defects attack the measurement that rule
defends — a report can satisfy BR-002 perfectly and be wrong about its content.

  1. **A rank-deficient basis inflated FVE.** `torch.linalg.qr(...).Q` returns as
     many orthonormal columns as the input has, PADDING with arbitrary
     directions when the input is rank-deficient. Those padded directions
     explain variance the real ones do not. Reproduced: four DUPLICATE
     directions reported FVE 0.378 against a true 0.083 — 4.5x.

  2. **`excess_fve` had no production caller,** so the random-direction control
     `control_seed` exists to make reproducible never ran — while the schema
     documented it as part of the report. Both inputs were already being
     collected in the profile builder; the call was simply absent.

  3. **`derive_boundaries` documented "first sustained rise" and implemented
     first-crossing.** One noisy layer could set `workspace_start`, including at
     layer 0, which yields an EMPTY sensory band.
"""

import inspect

import pytest
import torch

from src.ml.jlens_metrics import (
    LayerProfile,
    derive_boundaries,
    excess_fve,
    fraction_variance_explained,
)


# ── (1) rank, not column count ─────────────────────────────────────────────

def test_duplicate_directions_do_not_inflate_fve():
    """The reproduction from the finding, as a test."""
    torch.manual_seed(0)
    acts = torch.randn(200, 64)
    one = torch.randn(1, 64)
    duplicated = one.repeat(4, 1)  # 4 columns, rank 1

    assert fraction_variance_explained(acts, duplicated) == pytest.approx(
        fraction_variance_explained(acts, one), abs=1e-9
    ), "four copies of one direction must explain what one direction explains"


def test_independent_directions_still_explain_more():
    """Negative control: the fix must not simply suppress FVE.

    A version that returned the rank-1 answer for everything would pass the
    test above and destroy the metric.
    """
    torch.manual_seed(0)
    acts = torch.randn(200, 64)
    one = torch.randn(1, 64)
    four = torch.randn(4, 64)

    assert fraction_variance_explained(acts, four) > fraction_variance_explained(
        acts, one
    )


def test_degenerate_directions_explain_nothing():
    torch.manual_seed(0)
    acts = torch.randn(50, 16)
    assert fraction_variance_explained(acts, torch.zeros(3, 16)) == 0.0


def test_a_full_rank_basis_is_unchanged_by_the_fix():
    """Whatever the previous behaviour was on well-conditioned input, it was
    right; this pins that the SVD path did not perturb it."""
    torch.manual_seed(1)
    acts = torch.randn(100, 8)
    dirs = torch.eye(8)
    assert fraction_variance_explained(acts, dirs) == pytest.approx(1.0, abs=1e-9)


def test_excess_fve_subtracts_a_reproducible_control():
    torch.manual_seed(2)
    acts = torch.randn(150, 32)
    dirs = torch.randn(4, 32)
    a = excess_fve(acts, dirs, control_seed=7)
    b = excess_fve(acts, dirs, control_seed=7)
    assert a == b, "the control must be reproducible from the seed"
    assert excess_fve(acts, dirs, control_seed=8) != a


# ── (2) the control actually runs ──────────────────────────────────────────

def test_the_band_profile_builder_calls_excess_fve():
    """A metric with no caller is a control that never runs."""
    from src.services import jlens_band_service as svc

    src = inspect.getsource(svc)
    assert "excess_fve(" in src, (
        "excess_fve has no production caller — control_seed makes reproducible "
        "a control that never executes"
    )
    assert "control_seed=control_seed" in src, (
        "the seed must reach the control, or it is not reconstructible"
    )


def test_the_raw_fve_is_not_published_on_its_own():
    """Its own docstring is the argument: the raw figure means nothing.

    Any k directions explain some variance and k random ones explain a
    surprising amount, so publishing the raw number invites exactly the reading
    the excess exists to prevent.
    """
    assert not hasattr(LayerProfile, "fve")
    assert "excess_fve" in LayerProfile.__dataclass_fields__


# ── (3) sustained means sustained ──────────────────────────────────────────

def _profiles(kurtoses):
    return [LayerProfile(layer=i, kurtosis=k) for i, k in enumerate(kurtoses)]


def test_a_single_noisy_layer_does_not_set_the_workspace_start():
    """The empty-sensory-band case: one spike at layer 0."""
    # Layer 0 spikes above the median, then falls back; the real sustained rise
    # is later, and the peak sits inside it.
    ks = [4.0, 0.1, 0.1, 0.1, 1.0, 5.0, 6.0, 9.0, 2.0]
    result = derive_boundaries(_profiles(ks))
    assert result is not None
    assert result["workspace_start"] != 0, (
        "a lone spike at layer 0 set workspace_start there, leaving an empty "
        "sensory band"
    )
    assert result["workspace_start"] == 5


def test_a_genuine_sustained_rise_is_found():
    """Negative control: the fix must not reject real rises."""
    ks = [0.1, 0.1, 0.1, 2.0, 3.0, 4.0, 9.0, 1.0]
    result = derive_boundaries(_profiles(ks))
    assert result is not None
    assert result["workspace_start"] == 3
    assert result["motor_start"] == 6


def test_all_isolated_spikes_yields_no_boundaries():
    """BR-002's answer to "cannot tell" is None, never a guess."""
    ks = [0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0]
    assert derive_boundaries(_profiles(ks)) is None


def test_too_few_layers_still_returns_none():
    assert derive_boundaries(_profiles([1.0, 2.0, 3.0])) is None


def test_no_band_constant_appears_anywhere_in_the_derivation():
    """BR-002, by construction. The published ~L38–92 figures were measured on
    one model, and porting them is what this function exists to prevent."""
    # Strip the docstring first. It names the published ~L38-92 figures in
    # order to say they must never be used — a bare substring scan cannot tell
    # the prohibition from the violation. (Third time this trap has appeared in
    # this remediation; the pattern is now: parse, or exclude the prose.)
    src = inspect.getsource(derive_boundaries)
    doc = inspect.getdoc(derive_boundaries) or ""
    for line in doc.splitlines():
        src = src.replace(line, "")

    for forbidden in ("38", "92", "40", "90"):
        assert forbidden not in src, (
            f"a band constant {forbidden!r} appeared in the CODE of "
            f"derive_boundaries — BR-002 forbids it by construction"
        )


def test_a_lone_late_spike_does_not_set_the_motor_start():
    """The same rule on the other boundary.

    `peak_index` was a raw argmax over every layer, so an isolated late spike
    could set `motor_start` exactly as an isolated early one could set
    `workspace_start`. Found by writing the fixture for the workspace test —
    the finding named only one of the two.
    """
    #                 sustained rise 3..5, then one isolated spike at 8
    ks = [0.1, 0.1, 0.1, 3.0, 4.0, 6.0, 0.1, 0.1, 99.0]
    result = derive_boundaries(_profiles(ks))
    assert result is not None
    assert result["motor_start"] == 5, (
        f"an isolated spike at layer 8 set motor_start to "
        f"{result['motor_start']}"
    )
