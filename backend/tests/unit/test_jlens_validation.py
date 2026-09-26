"""
Artifact validation (BR-030).

Every check here exists because the downstream consumer FAILS SILENTLY: a bad
artifact serves an empty readout rather than raising. So each class is asserted
to fail against its own violation — a suite that only ever sees good artifacts
proves that the happy path runs, which is the one thing never in doubt.

MUTATION CONTROLS (each must turn this file red):
  * make ValidationReport.passed ignore missing classes  -> "fail closed" fails
  * accept more than one lens file in a directory        -> "naming" fails
  * hardcode the envelope ceiling                        -> "envelope scales" fails
  * treat an unreachable consumer as PASS                -> "not run is not pass" fails
  * treat an empty served readout as PASS                -> "round trip" fails
  * drop the anchors from LENS_FILENAME                  -> "anchored" fails
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from src.services.jlens_validation import (
    CheckClass,
    CheckResult,
    CheckStatus,
    ValidationReport,
    check_cross_implementation,
    check_envelope,
    check_naming,
    check_round_trip,
    check_semantic,
    check_structural,
)


def passing(check: CheckClass) -> CheckResult:
    return CheckResult(check, CheckStatus.PASS, "ok")


# ---------------------------------------------------------------- fail closed


def test_report_fails_closed_when_a_class_never_ran():
    """"We did not check" and "we checked and it was fine" must differ.

    The consumer's failure mode is silence; a verdict that cannot tell an unrun
    check from a passing one reproduces exactly that failure one level up.
    """
    partial = ValidationReport([passing(c) for c in CheckClass if c is not CheckClass.ROUND_TRIP])
    assert partial.passed is False
    assert partial.missing == [CheckClass.ROUND_TRIP]
    assert "round_trip=not_run" in partial.summary()

    complete = ValidationReport([passing(c) for c in CheckClass])
    assert complete.passed is True


def test_not_run_is_not_a_pass():
    report = ValidationReport(
        [passing(c) for c in CheckClass if c is not CheckClass.CROSS_IMPLEMENTATION]
        + [CheckResult(CheckClass.CROSS_IMPLEMENTATION, CheckStatus.NOT_RUN, "unreachable")]
    )
    assert report.passed is False


# ------------------------------------------------------------------- naming


def test_naming_accepts_exactly_one_conformant_file(tmp_path: Path):
    (tmp_path / "lfm2-5-1-2b-instruct_jacobian_lens.pt").write_bytes(b"x")
    assert check_naming(tmp_path).passed


def test_naming_rejects_two_lens_files(tmp_path: Path):
    """The consumer picks among several without saying which."""
    (tmp_path / "a_jacobian_lens.pt").write_bytes(b"x")
    (tmp_path / "b_jacobian_lens.pt").write_bytes(b"x")
    result = check_naming(tmp_path)
    assert not result.passed
    assert "more than one" in result.detail


def test_naming_rejects_a_stray_non_conformant_pt(tmp_path: Path):
    (tmp_path / "a_jacobian_lens.pt").write_bytes(b"x")
    (tmp_path / "checkpoint.pt").write_bytes(b"x")
    assert not check_naming(tmp_path).passed


def test_lens_filename_pattern_is_anchored(tmp_path: Path):
    """An unanchored pattern accepts a backup file and serves it."""
    (tmp_path / "a_jacobian_lens.pt.bak").write_bytes(b"x")
    (tmp_path / "prefix_a_jacobian_lens.pt.old").write_bytes(b"x")
    result = check_naming(tmp_path)
    assert not result.passed, "a .bak was accepted as the lens"


# ----------------------------------------------------------------- structural


def test_structural_accepts_a_well_formed_artifact():
    payload = {i: torch.zeros(8, 8) for i in range(3)}
    assert check_structural(payload, d_model=8, expected_layers=range(3)).passed


def test_structural_rejects_a_non_square_matrix():
    payload = {0: torch.zeros(8, 4)}
    result = check_structural(payload, d_model=8, expected_layers=[0])
    assert not result.passed and "square" in result.detail


def test_structural_rejects_the_wrong_side():
    """Wrong-side matrices load fine and read out the wrong thing."""
    payload = {0: torch.zeros(4, 4)}
    result = check_structural(payload, d_model=8, expected_layers=[0])
    assert not result.passed and "d_model is 8" in result.detail


def test_structural_rejects_a_missing_layer():
    payload = {0: torch.zeros(8, 8), 2: torch.zeros(8, 8)}
    result = check_structural(payload, d_model=8, expected_layers=[0, 1, 2])
    assert not result.passed and "[1]" in result.detail


def test_structural_accepts_string_layer_keys():
    payload = {"0": torch.zeros(8, 8), "1": torch.zeros(8, 8)}
    assert check_structural(payload, d_model=8, expected_layers=[0, 1]).passed


def test_structural_rejects_uncoercible_layer_keys():
    result = check_structural({"final": torch.zeros(8, 8)}, 8, [0])
    assert not result.passed and "coercible" in result.detail


# ------------------------------------------------------------------- envelope


def test_envelope_accepts_the_required_size():
    d, n, v = 2048, 16, 65536
    required = d * d * 2 * n
    assert check_envelope(required, d, n, v).passed


def test_envelope_rejects_a_materialised_dictionary():
    """The defect BR-006 exists to prevent, named in the failure detail."""
    d, n, v = 2048, 16, 65536
    materialised = v * d * 2 * n
    result = check_envelope(materialised, d, n, v)
    assert not result.passed
    assert "MATERIALISED" in result.detail


def test_envelope_allows_container_overhead_without_widening_the_multiplier():
    """A file is bigger than the tensors it holds.

    The allowance is absolute so it stays constant as the model grows; a wider
    MULTIPLIER would scale with the model and open a gap a partial
    materialisation could hide in.
    """
    from src.services.jlens_validation import (
        CONTAINER_OVERHEAD_BYTES,
        container_allowance,
    )

    # Toy scale: the flat allowance alone would exceed this model's ENTIRE
    # materialised dictionary, blinding the check where the numbers are
    # smallest. The half-required cap is what stops that.
    d, n, v = 8, 3, 512
    required = d * d * 2 * n
    assert check_envelope(required + 2048, d, n, v).passed
    assert not check_envelope(v * d * 2 * n, d, n, v).passed

    # Real scale: the cap never binds and the allowance is noise.
    big_required = 2048 * 2048 * 2 * 16
    assert container_allowance(big_required) == CONTAINER_OVERHEAD_BYTES


def test_envelope_rejects_a_truncated_artifact():
    """Truncation loads fine and reads out nothing useful."""
    d, n, v = 2048, 16, 65536
    result = check_envelope(1024, d, n, v)
    assert not result.passed and "truncated" in result.detail


def test_envelope_ceiling_scales_with_the_model_not_a_constant():
    """A bound tuned on one model must not silently govern another.

    Two models with the SAME d_model and layer count but different vocabularies:
    the required size is identical, the materialised size is not, and a
    hardcoded ceiling cannot be right for both. The assertion is on the derived
    evidence, so hardcoding the ceiling breaks it.
    """
    d, n = 2048, 16
    small = check_envelope(d * d * 2 * n, d, n, n_vocab=65_536)
    large = check_envelope(d * d * 2 * n, d, n, n_vocab=256_000)

    assert small.passed and large.passed
    assert small.evidence["required_bytes"] == large.evidence["required_bytes"]
    assert large.evidence["materialised_bytes"] > small.evidence["materialised_bytes"]
    assert small.evidence["ratio"] == pytest.approx(32.0, abs=1.0)
    assert large.evidence["ratio"] == pytest.approx(125.0, abs=5.0)


def test_envelope_catches_a_materialisation_that_a_65k_bound_would_miss():
    """A 256k-vocab model materialised at 65k scale still busts its own ceiling."""
    d, n = 2048, 16
    size_at_65k_scale = 65_536 * d * 2 * n
    assert not check_envelope(size_at_65k_scale, d, n, n_vocab=256_000).passed


# ------------------------------------------------------------------- semantic


def test_semantic_recovers_an_unspoken_intermediate():
    result = check_semantic(
        lambda p, ls, k: {l: ["spider", "web", "legs"] for l in ls},
        prompt="the animal that spins webs has this many legs",
        layers=8,
        expected_intermediate="spider",
    )
    assert result.passed


def test_semantic_rejects_a_fixture_whose_answer_is_in_the_prompt():
    """A token already in the prompt is recoverable by a broken lens."""
    result = check_semantic(
        lambda p, ls, k: {l: ["spider"] for l in ls},
        prompt="a spider spins webs",
        layers=8,
        expected_intermediate="spider",
    )
    assert not result.passed and "appears in the" in result.detail


def test_semantic_fails_when_the_intermediate_is_absent():
    result = check_semantic(
        lambda p, ls, k: {l: ["the", "a", "of"] for l in ls},
        prompt="the animal that spins webs",
        layers=8,
        expected_intermediate="spider",
    )
    assert not result.passed


def test_semantic_reports_rather_than_swallows_a_raising_readout():
    def boom(*_):
        raise RuntimeError("cuda oom")

    result = check_semantic(boom, "the animal that spins webs", 8, "spider")
    assert not result.passed and "cuda oom" in result.detail


def test_semantic_scans_the_stack_rather_than_asserting_one_layer():
    """The claim is that the intermediate surfaces, not WHERE it surfaces.

    Pinning one layer discarded a converged LFM2 artifact whose mid-stack
    readout was the correct concept field. Which depth carries a bridge entity
    is a property of the model, and BR-002 forbids assuming one.

    MUTATION CONTROL: narrow `scan` in `check_semantic` back to `scan[:1]` and
    this fails — the token lives at layer 11, nowhere near the first layer.
    """

    def readout(_prompt, layers, _k):
        return {
            l: (["spider", "web"] if l == 11 else ["the", "a", "of"]) for l in layers
        }

    result = check_semantic(
        readout,
        prompt="the animal that spins webs",
        layers=[0, 4, 8, 11, 14],
        expected_intermediate="spider",
    )
    assert result.passed
    assert result.evidence["layer"] == 11
    assert result.evidence["rank"] == 0


def test_semantic_reports_where_and_at_what_rank_it_surfaced():
    """A pass that does not say where it passed cannot be judged by a human."""

    def readout(prompt, layers, _k):
        return {
            l: (["a", "b", "spider"] if ("webs" in prompt and l == 7) else ["x"])
            for l in layers
        }

    result = check_semantic(
        readout,
        prompt="the animal that spins webs",
        layers=[3, 7],
        expected_intermediate="spider",
        control_prompt="the recipe calls for two cups of",
    )
    assert result.passed
    assert result.evidence["layer"] == 7 and result.evidence["rank"] == 2
    assert "layer 7" in result.detail and "rank 2" in result.detail


def test_semantic_fails_when_the_control_prompt_recovers_it_too():
    """A lens that answers 'spider' to everything has demonstrated nothing.

    The scan is weaker than a single layer, so it needs a false-positive
    control; without this, loosening the check to a scan would be a pure
    weakening of the one class that catches a structurally perfect artifact
    with no content in it.

    MUTATION CONTROL: drop the `if ctrl_layer is not None` branch and this
    passes — the scan happily reports the hit on the real prompt.
    """
    result = check_semantic(
        lambda _p, ls, _k: {l: ["spider", "web"] for l in ls},  # spider regardless of prompt
        prompt="the animal that spins webs",
        layers=[0, 4, 8],
        expected_intermediate="spider",
        control_prompt="the recipe calls for two cups of",
    )
    assert not result.passed
    assert "control" in result.detail
    assert result.evidence["control_layer"] == 0


def test_semantic_says_so_when_no_control_was_supplied():
    """An unmeasured control must not read as a control that passed."""
    result = check_semantic(
        lambda _p, ls, _k: {l: ["spider"] for l in ls},
        prompt="the animal that spins webs",
        layers=[0],
        expected_intermediate="spider",
    )
    assert result.passed
    assert "not measured" in result.detail
    assert result.evidence["control"] == "not supplied"


def test_semantic_fails_closed_on_an_empty_layer_set():
    """No layers to scan is not a pass."""
    result = check_semantic(
        lambda _p, ls, _k: {l: ["spider"] for l in ls},
        prompt="the animal that spins webs",
        layers=[],
        expected_intermediate="spider",
    )
    assert not result.passed


# --------------------------------------------------------- cross-implementation


def test_cross_implementation_passes_on_agreement():
    assert check_cross_implementation(["a", "b", "c"], ["a", "b", "c"]).passed


def test_cross_implementation_fails_on_divergence():
    result = check_cross_implementation(["a", "b"], ["a", "z"])
    assert not result.passed and "differs" in result.detail


def test_unreachable_consumer_is_not_run_not_pass():
    """Treating an unreachable consumer as agreement makes the check silent."""
    result = check_cross_implementation(["a"], None)
    assert result.status is CheckStatus.NOT_RUN
    assert not result.passed


# ----------------------------------------------------------------- round trip


def test_round_trip_passes_on_real_content():
    assert check_round_trip([" Paris", " France"]).passed


def test_round_trip_fails_on_nothing_served():
    assert not check_round_trip(None).passed


def test_round_trip_fails_on_an_empty_readout():
    """An empty readout is exactly what an unmounted artifact produces."""
    result = check_round_trip(["", "  "])
    assert not result.passed and "empty" in result.detail


# ---------------------------------------------------------------------- BR-004


def test_no_check_scores_next_token_agreement():
    """BR-004 is a product rule, so it gets a test rather than a comment.

    The J-lens is DELIBERATELY worse than the logit lens on next-token
    agreement through most of the network. A validation check that rewarded it
    would reject good artifacts and accept bad ones, and nothing about the
    resulting artifact would look wrong.
    """
    from src.services import jlens_validation

    source = inspect.getsource(jlens_validation)
    tree = ast.parse(source)

    banned = ("agreement", "next_token_match", "kl_to_output")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            label = node.id if isinstance(node, ast.Name) else node.attr
            assert not any(b in label.lower() for b in banned), (
                f"{label!r} in the executable path: next-token agreement must "
                "never be a validation criterion (BR-004)"
            )


def test_structural_rejects_a_NON_FINITE_artifact():
    """An overflowed fp16 cast is the worst kind of bad artifact.

    Found on the first real fit: GPT-2's accumulated Jacobians exceed fp16's
    65504 ceiling, so 0.3% of entries saturated to inf. That artifact
    deserialises cleanly, is exactly the right shape and exactly the right
    size, passes NAMING and ENVELOPE, and every readout through it is garbage.
    """
    payload = {0: torch.full((8, 8), float("inf"), dtype=torch.float16)}
    result = check_structural(payload, d_model=8, expected_layers=[0])
    assert not result.passed
    assert "non-finite" in result.detail


def test_structural_rejects_a_PARTIALLY_saturated_artifact():
    """0.3% of entries is what a real overflow looked like — not all of them."""
    matrix = torch.zeros(8, 8, dtype=torch.float16)
    matrix[0, 0] = float("inf")
    result = check_structural({0: matrix}, d_model=8, expected_layers=[0])
    assert not result.passed


def test_the_scan_costs_ONE_readout_per_prompt_not_one_per_layer():
    """A readout is a forward pass; the scan must not multiply them.

    `stream` captures residuals once and reads every requested layer off them.
    Scanning layer-by-layer turned a 25-layer artifact's check into 25 forward
    passes, doubled by the control — a change that is invisible in the result
    and only shows up as validation taking minutes.

    Asserts the CALL COUNT, not merely that a call happened: "was called"
    passes just as happily against the per-layer version.

    MUTATION CONTROL: put the `readout(...)` back inside the `for layer in
    scan` loop and this fails.
    """
    calls = []

    def readout(prompt, layers, k):
        calls.append((prompt, tuple(layers)))
        return {l: ["the", "a"] for l in layers}

    scan = list(range(25))
    check_semantic(
        readout,
        prompt="the animal that spins webs",
        layers=scan,
        expected_intermediate="spider",
        control_prompt="the interest rate was raised to",
    )

    # One for the real prompt. The control never runs: the token was not found,
    # so there is nothing to control FOR.
    assert len(calls) == 1, f"{len(calls)} readouts for a 25-layer scan: {calls}"
    assert calls[0][1] == tuple(scan), "the scan did not ask for every layer at once"


def test_a_hit_scans_the_control_in_ONE_further_readout():
    """Two passes total when the token is found — one real, one control."""
    calls = []

    def readout(prompt, layers, k):
        calls.append(prompt)
        found = "webs" in prompt
        return {l: (["spider"] if found else ["x"]) for l in layers}

    result = check_semantic(
        readout,
        prompt="the animal that spins webs",
        layers=list(range(25)),
        expected_intermediate="spider",
        control_prompt="the interest rate was raised to",
    )
    assert result.passed
    assert len(calls) == 2, f"expected real+control, got {len(calls)}"
