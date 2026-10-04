"""
Persisting a band report and a gate decision beside the artifact.

The property under test throughout is that ABSENCE STAYS ABSENT. A model with
no band report must not acquire boundaries by passing through serialisation, a
metric that could not be computed must not become 0.0, and a NO_GO must read
back exactly as it was written.

MUTATION CONTROLS (each must turn this file red):
  * default `boundaries` to anything when absent  -> "no bands survives" fails
  * coerce a null metric to 0.0 on save           -> "absent stays absent" fails
  * store the gate decision as a bare bool        -> "NO_GO round-trips" fails
  * return {} instead of None for a missing file  -> "missing is None" fails
  * swallow a corrupt file as an empty report     -> "corrupt is None" fails
"""

from __future__ import annotations

import json
import torch
from pathlib import Path

import pytest

from src.ml.jlens_metrics import LayerProfile
from src.services.jlens_band_report import (
    GateDecision,
    build_band_report,
    decide_gate,
)
from src.services.jlens_band_service import (
    load_band_report,
    load_gate,
    save_band_report,
    save_gate,
)


def structured():
    values = [0.1, 0.1, 0.2, 1.5, 2.0, 2.4, 3.0, 6.0, 2.0, 1.0]
    return [LayerProfile(layer=i, kurtosis=v) for i, v in enumerate(values)]


def flat():
    return [LayerProfile(layer=i, kurtosis=1.0) for i in range(8)]


# ── band report ────────────────────────────────────────────────────────────


def test_a_report_with_boundaries_round_trips(tmp_path: Path):
    report = build_band_report("m1", structured(), control_seed=7)
    save_band_report(tmp_path, report)

    loaded = load_band_report(tmp_path)
    assert loaded is not None
    assert loaded["boundaries"] == report.boundaries
    assert loaded["control_seed"] == 7
    assert len(loaded["profiles"]) == len(report.profiles)


def test_no_bands_survives_serialisation_as_null(tmp_path: Path):
    """The whole point of BR-002: absence must not become a default.

    A report that acquires boundaries by passing through JSON would draw
    authoritative-looking shading for a model whose profile never supported a
    split.
    """
    report = build_band_report("m1", flat(), control_seed=3)
    assert report.boundaries is None

    save_band_report(tmp_path, report)
    loaded = load_band_report(tmp_path)

    assert loaded["boundaries"] is None
    assert "do not transfer" in loaded["derivation"]

    raw = json.loads((tmp_path / "band-report.json").read_text())
    assert raw["boundaries"] is None


def test_an_uncomputable_metric_stays_absent_rather_than_zero(tmp_path: Path):
    """0.0 is a measurement; null is "we could not measure".

    A consumer averages the first and has to decide about the second — the same
    rule the per-layer applicability follows.
    """
    profiles = [LayerProfile(layer=0, kurtosis=1.0)]  # autocorrelation absent
    save_band_report(tmp_path, build_band_report("m1", profiles, control_seed=1))

    raw = json.loads((tmp_path / "band-report.json").read_text())
    entry = raw["profiles"][0]
    assert entry["autocorrelation"] is None
    assert entry["autocorrelation_null"] is None
    assert entry["excess_autocorrelation"] is None
    assert entry["kurtosis"] == 1.0


def test_excess_autocorrelation_is_serialised_as_the_finding(tmp_path: Path):
    """The raw figure is not the finding; the excess over the null is."""
    profiles = [
        LayerProfile(layer=0, kurtosis=1.0, autocorrelation=0.9, autocorrelation_null=0.85)
    ]
    save_band_report(tmp_path, build_band_report("m1", profiles, control_seed=1))

    entry = json.loads((tmp_path / "band-report.json").read_text())["profiles"][0]
    assert entry["excess_autocorrelation"] == pytest.approx(0.05)


def test_a_missing_report_is_None_not_an_empty_report(tmp_path: Path):
    """None renders as "no bands"; {} would render as a report with no data."""
    assert load_band_report(tmp_path) is None


def test_a_corrupt_report_is_None_rather_than_a_partial_one(tmp_path: Path):
    (tmp_path / "band-report.json").write_text("{not json")
    assert load_band_report(tmp_path) is None


# ── gate ───────────────────────────────────────────────────────────────────


def test_no_go_round_trips_exactly(tmp_path: Path):
    """A gate whose negative outcome cannot be persisted is not a gate."""
    record = decide_gate(
        "m1",
        build_band_report("m1", flat(), control_seed=3),
        claim_set_replicated=False,
        larger_scale_indicated=False,
        rationale="selectivity did not replicate",
    )
    save_gate(tmp_path, record)

    loaded = load_gate(tmp_path)
    assert loaded["decision"] == GateDecision.NO_GO.value
    assert loaded["blocking"] is True
    assert loaded["rationale"] == "selectivity did not replicate"
    assert loaded["has_bands"] is False


def test_go_at_larger_scale_round_trips_as_its_own_value(tmp_path: Path):
    """Not a softened GO: it still blocks at THIS scale."""
    record = decide_gate(
        "m1",
        build_band_report("m1", structured(), control_seed=3),
        claim_set_replicated=False,
        larger_scale_indicated=True,
        rationale="band structure present, capacity limits absent",
    )
    save_gate(tmp_path, record)

    loaded = load_gate(tmp_path)
    assert loaded["decision"] == GateDecision.GO_AT_LARGER_SCALE.value
    assert loaded["blocking"] is True


def test_go_round_trips_as_non_blocking(tmp_path: Path):
    record = decide_gate(
        "m1",
        build_band_report("m1", structured(), control_seed=3),
        claim_set_replicated=True,
        larger_scale_indicated=False,
        rationale="all four claims replicate",
    )
    save_gate(tmp_path, record)
    assert load_gate(tmp_path)["blocking"] is False


def test_a_missing_gate_is_None_and_that_is_not_a_GO(tmp_path: Path):
    """No decision recorded must never be read as permission to proceed."""
    assert load_gate(tmp_path) is None


def test_an_unrecognised_decision_raises_rather_than_rendering(tmp_path: Path):
    """A plausible-looking string in the UI is worse than an error here."""
    (tmp_path / "gate.json").write_text(json.dumps({"decision": "probably_fine"}))
    with pytest.raises(ValueError):
        load_gate(tmp_path)


# ── metrics measure the right OBJECT (review round 2) ──────────────────────


class _Svc:
    """Minimal readout-service stand-in exposing the two private hooks used."""

    def __init__(self, n_positions=4, d_model=6, n_vocab=40, n_layers=3):
        self.d_model = d_model
        self.W_U = torch.randn(n_vocab, d_model)
        self._n_positions = n_positions
        self._n_layers = n_layers

        class Tok:
            def __call__(_s, text, return_tensors=None):
                return {"input_ids": torch.ones(1, n_positions, dtype=torch.long)}

        self.tokenizer = Tok()

    def _capture_residuals(self, input_ids, layers):
        class Cap:
            pass

        cap = Cap()
        cap.by_layer = {
            l: torch.randn(self._n_positions, self.d_model) for l in layers
        }
        return cap

    def _normalize(self, x):
        return x


def test_kurtosis_is_measured_on_the_READOUT_not_the_residual():
    """BR-002 asks for the readout distribution's kurtosis.

    The residual's kurtosis describes the activation vector's shape; the
    readout's describes how sharply the layer points at particular tokens.
    Only the second is what "the distribution sharpens where reportable
    content appears" means, and only the second is what the boundary
    derivation keys on.

    A residual-based figure is measurable, plausible and answers a different
    question — so this pins the object, using a fixture whose residuals are
    near-normal (excess kurtosis ~0) while its readout is deliberately sharp.
    """
    from src.services.jlens_band_service import compute_band_report

    svc = _Svc()
    # A near-one-hot unembedding makes the readout distribution extremely
    # peaked while the residuals stay normal.
    svc.W_U = torch.zeros(40, 6)
    svc.W_U[0] = torch.ones(6) * 50.0

    report = compute_band_report(
        svc, ["abcd"], layers=[0, 1, 2], control_seed=5, model_id="m"
    )
    for p in report.profiles:
        assert p.kurtosis is not None
        # A normal sample sits near 0; this readout is far from it.
        assert p.kurtosis > 3.0, (
            "kurtosis looks like it was measured on the residuals, which are "
            "normal here, rather than on the peaked readout distribution"
        )


def test_effective_dimensionality_is_ABSENT_for_the_logit_lens():
    """The identity's effective dimensionality is d_model — a constant.

    Reporting it would put a number in the profile that varies with nothing and
    reads as a measurement.
    """
    from src.services.jlens_band_service import compute_band_report

    report = compute_band_report(
        _Svc(), ["abcd"], layers=[0, 1], control_seed=5, model_id="m"
    )
    assert all(p.effective_dimensionality is None for p in report.profiles)


def test_effective_dimensionality_comes_from_the_LENS_DICTIONARY():
    """With a Jacobian supplied, the figure describes J — not the residuals."""
    from src.services.jlens_band_service import compute_band_report

    # Layer 0's J is rank-deficient by construction; layer 1's is full rank.
    concentrated = torch.zeros(6, 6)
    concentrated[0, 0] = 1.0
    jacobians = {0: concentrated, 1: torch.eye(6)}

    report = compute_band_report(
        _Svc(), ["abcd"], layers=[0, 1], control_seed=5, model_id="m",
        jacobians=jacobians,
    )
    by_layer = {p.layer: p.effective_dimensionality for p in report.profiles}
    assert by_layer[0] == pytest.approx(1.0, abs=0.01)
    assert by_layer[1] == pytest.approx(6.0, abs=0.01)


def test_the_band_report_bounds_its_own_cost_as_a_product():
    """Prompts, positions and layers are each modest; their product is not."""
    from src.services.jlens_band_service import (
        MAX_BAND_REPORT_CELLS,
        compute_band_report,
    )

    svc = _Svc(n_positions=4096)
    with pytest.raises(ValueError, match="product of prompts"):
        compute_band_report(
            svc,
            ["x"] * 200,
            layers=list(range(26)),
            control_seed=1,
            model_id="m",
        )
    assert MAX_BAND_REPORT_CELLS > 0
