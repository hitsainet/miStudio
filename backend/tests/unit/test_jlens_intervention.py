"""
Intervention primitives (BR-016..018).

The load-bearing requirement: A RUN WITHOUT ITS CONTROL IS INVALID. Not
"unreviewed", not "preliminary" — invalid, because every interpretation of an
intervention rests on the comparison with a size-matched random direction.

The enforcement is structural rather than a validation step, and the first test
below is the reason: under time pressure the control is what gets skipped, and a
`validate()` is what gets bypassed with a flag. An object that cannot be built
without its control cannot be reported without it.

MUTATION CONTROLS (each must turn this file red):
  * give `control_outcome` a default              -> "cannot construct" fails
  * drop the seed from ControlSpec                -> "reconstructible" fails
  * allow k <= 0                                  -> "size-matched" fails
  * include clean-pass coords in dynamic top-k    -> "ordinary behaviour" fails
  * fix the swap layer count                      -> "scale-aware" fails
  * assume a unit direction in projective ablation -> "unnormalised" fails
"""

from __future__ import annotations

import inspect

import pytest
import torch

from src.services.jlens_intervention import (
    ClampSpec,
    ControlSpec,
    InterventionResult,
    Primitive,
    apply_additive,
    apply_projective_ablation,
    build_control,
    coordinate_swap,
    default_swap_layers,
    dynamic_topk_ablation,
)


# ── the control is not optional (BR-018) ───────────────────────────────────


def test_a_result_CANNOT_BE_CONSTRUCTED_without_its_control():
    """Structural, not a validation step.

    A `validate()` is the thing that gets bypassed with a flag when someone is
    in a hurry. An unconstructable object is not.
    """
    with pytest.raises(TypeError):
        InterventionResult(  # type: ignore[call-arg]
            primitive=Primitive.ADDITIVE,
            parameters={},
            intervened_outcome=0.9,
            layers=[3],
            positions=[0],
        )


def test_control_outcome_has_no_default():
    """Asserted on the signature, so adding a default is caught immediately."""
    params = inspect.signature(InterventionResult).parameters
    for required in ("control", "control_outcome"):
        assert params[required].default is inspect.Parameter.empty, (
            f"{required} gained a default — a run could then be reported "
            "without its control (BR-018)"
        )


def test_the_finding_is_the_EXCESS_over_the_control():
    """The raw intervened outcome is not a finding.

    Moving the output says nothing until you know what moving a random
    direction of the same size does.
    """
    result = InterventionResult(
        primitive=Primitive.ADDITIVE,
        parameters={"strength": 1.0},
        control=ControlSpec(k=4, seed=7),
        intervened_outcome=0.90,
        control_outcome=0.62,
        layers=[3],
        positions=[0],
    )
    assert result.excess_over_control == pytest.approx(0.28)


def test_a_control_must_be_size_matched():
    """'A random direction' is not a control; k of them is."""
    with pytest.raises(ValueError, match="size-matched"):
        ControlSpec(k=0, seed=1)


def test_a_control_is_reconstructible_from_its_seed():
    """A figure whose control nobody else can rebuild cannot be checked."""
    a = build_control(k=4, seed=11, d_model=8)
    b = build_control(k=4, seed=11, d_model=8)
    c = build_control(k=4, seed=12, d_model=8)

    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)
    assert a.shape == (4, 8)
    # Unit-norm, so "size-matched" is about direction count and not magnitude.
    assert torch.allclose(a.norm(dim=-1), torch.ones(4), atol=1e-5)


# ── clamping (BR-016) ──────────────────────────────────────────────────────


def test_clamping_is_keyed_by_POSITION_AND_LAYER():
    """Holding a coordinate at some positions leaks the effect through the rest.

    The result is then not about the quantity it names, which is worse than no
    mediation analysis at all.
    """
    clamp = ClampSpec({(0, 3): [1, 2], (1, 3): [1, 2]})
    assert clamp.held_at(0, 3) == [1, 2]
    assert clamp.held_at(1, 3) == [1, 2]
    # A position that was never clamped reports nothing held — not the others'.
    assert clamp.held_at(2, 3) == ()
    assert clamp.held_at(0, 4) == ()


def test_an_empty_clamp_is_distinguishable_from_no_clamp():
    assert ClampSpec().is_empty is True
    assert ClampSpec({(0, 1): [0]}).is_empty is False


# ── primitives (BR-017) ────────────────────────────────────────────────────


def test_additive_steers_along_the_direction():
    act = torch.zeros(4)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(apply_additive(act, direction, 2.0), direction * 2.0)


def test_projective_ablation_removes_the_component_along_the_direction():
    act = torch.tensor([3.0, 4.0, 0.0])
    direction = torch.tensor([1.0, 0.0, 0.0])
    out = apply_projective_ablation(act, direction)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(4.0)


def test_projective_ablation_normalises_an_UNNORMALISED_direction():
    """An unnormalised direction silently scales the ablation by its magnitude.

    That looks like a stronger effect rather than a bug — the activation is
    over-removed and the run reports it as the intervention working.
    """
    act = torch.tensor([3.0, 4.0, 0.0])
    unit = apply_projective_ablation(act, torch.tensor([1.0, 0.0, 0.0]))
    scaled = apply_projective_ablation(act, torch.tensor([9.0, 0.0, 0.0]))
    assert torch.allclose(unit, scaled), "the ablation scaled with the direction's magnitude"


def test_projective_ablation_of_a_zero_direction_is_a_no_op():
    act = torch.tensor([1.0, 2.0])
    assert torch.allclose(apply_projective_ablation(act, torch.zeros(2)), act)


def test_dynamic_topk_EXCLUDES_clean_pass_coordinates():
    """Otherwise it ablates ordinary behaviour and calls the result an effect.

    Coordinate 0 is the largest, and it was already top-k in the clean pass, so
    it belongs to what the model normally does — ablating it measures the
    technique, not the intervention.
    """
    coords = torch.tensor([5.0, 4.0, 3.0, 1.0])
    out = dynamic_topk_ablation(coords, k=2, clean_pass_topk={0})

    assert out[0] == pytest.approx(5.0), "a clean-pass coordinate was ablated"
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(1.0)


def test_dynamic_topk_with_no_budget_is_a_no_op():
    coords = torch.tensor([5.0, 4.0])
    assert torch.allclose(dynamic_topk_ablation(coords, k=0, clean_pass_topk=set()), coords)


def test_coordinate_swap_replaces_the_target_with_the_source():
    coords = torch.tensor([1.0, 2.0, 3.0])
    out = coordinate_swap(coords, source=0, target=2)
    assert out[2] == pytest.approx(1.0)
    assert out[0] == pytest.approx(1.0), "the source must not be disturbed"


# ── scale-aware swap default (BR-017 v0.2) ─────────────────────────────────


def test_the_swap_layer_default_SCALES_with_the_model():
    """A constant tuned on a large model oversteers a small one.

    That is the amendment's entire reason for existing, so a single-size
    assertion would miss it.
    """
    small = default_swap_layers(16)
    large = default_swap_layers(64)
    assert small < large, "the swap default does not vary with model size"
    assert small == 4 and large == 16


def test_the_swap_default_is_at_least_one_layer():
    """A model too small to quarter still gets a usable default, not zero."""
    assert default_swap_layers(2) == 1
    assert default_swap_layers(1) == 1


def test_a_nonsensical_layer_count_is_refused():
    with pytest.raises(ValueError):
        default_swap_layers(0)


# ── the primitive is recorded (BR-017) ─────────────────────────────────────


def test_every_result_records_its_primitive_and_parameters():
    """A run whose primitive is unrecorded cannot be compared to another."""
    result = InterventionResult(
        primitive=Primitive.COORDINATE_SWAP,
        parameters={"source": 3, "target": 9, "layers": 4},
        control=ControlSpec(k=4, seed=1),
        intervened_outcome=0.5,
        control_outcome=0.4,
        layers=[1, 2, 3, 4],
        positions=[7],
    )
    assert result.primitive is Primitive.COORDINATE_SWAP
    assert result.parameters["source"] == 3


def test_all_four_primitives_exist():
    assert {p.value for p in Primitive} == {
        "additive",
        "projective_ablation",
        "dynamic_topk_ablation",
        "coordinate_swap",
    }


# ---------------------------------------------------------------------------
# A SWAP MUST SWAP.
#
# The perturbing hook had two branches — projective ablation, and
# everything-else-is-additive — so a request for `coordinate_swap` ran an
# ADDITIVE steer and the result was then labelled `coordinate_swap` in its
# `steering_recipe`. That recipe is written into `interventions.json`, which is
# built to travel with the lens, so the mislabelling would have become false
# provenance in whatever consumed it. One run in the session that found this was
# already mislabelled.
#
# MUTATION CONTROLS:
#   * return `activation + a` (an additive push)  -> "EXCHANGES" fails
#   * drop the near-parallel refusal              -> "same direction" fails
#   * skip the normalisation                      -> "unnormalised" fails
# ---------------------------------------------------------------------------


class TestCoordinateSwapExchanges:
    def _orthogonal(self):
        a = torch.zeros(8)
        a[0] = 1.0
        b = torch.zeros(8)
        b[1] = 1.0
        return a, b

    def test_it_EXCHANGES_the_two_components(self):
        """The defining property, and the one additive does not have.

        MUTATION CONTROL: return `activation + strength * a` and this fails.
        """
        from src.services.jlens_intervention import apply_coordinate_swap

        a, b = self._orthogonal()
        h = torch.zeros(8)
        h[0] = 3.0   # component along a
        h[1] = -7.0  # component along b
        h[2] = 1.5   # untouched dimension

        out = apply_coordinate_swap(h, a, b)

        assert float(out @ a) == pytest.approx(-7.0), "a did not receive b's value"
        assert float(out @ b) == pytest.approx(3.0), "b did not receive a's value"
        assert float(out[2]) == pytest.approx(1.5), (
            "a coordinate outside the swapped pair was modified"
        )

    def test_it_is_NOT_an_additive_push(self):
        """Distinguishes the two primitives by their effect, not their name."""
        from src.services.jlens_intervention import (
            apply_additive,
            apply_coordinate_swap,
        )

        a, b = self._orthogonal()
        h = torch.zeros(8)
        h[0], h[1] = 3.0, -7.0

        swapped = apply_coordinate_swap(h, a, b)
        pushed = apply_additive(h, a, 1.0)
        assert not torch.allclose(swapped, pushed), (
            "the swap produced the same activation as an additive steer — which "
            "is exactly what the unimplemented version did"
        )

    def test_swapping_EQUAL_components_is_a_no_op(self):
        """Nothing to exchange when the two already agree."""
        from src.services.jlens_intervention import apply_coordinate_swap

        a, b = self._orthogonal()
        h = torch.zeros(8)
        h[0] = h[1] = 4.0
        assert torch.allclose(apply_coordinate_swap(h, a, b), h)

    def test_two_directions_that_are_the_SAME_direction_are_refused(self):
        """A swap that cannot move anything would report a null.

        Reported as a null, it invites the reading that the concept does not
        steer — a conclusion about the model drawn from a degenerate request.

        MUTATION CONTROL: drop the cosine check and this fails.
        """
        from src.services.jlens_intervention import (
            DirectionsTooSimilar,
            apply_coordinate_swap,
        )

        a = torch.randn(8)
        nearly = a * 1.0001
        with pytest.raises(DirectionsTooSimilar, match="cosine"):
            apply_coordinate_swap(torch.randn(8), a, nearly)

    def test_an_UNNORMALISED_direction_does_not_scale_the_swap(self):
        """Otherwise a long direction silently amplifies the exchange.

        Same reasoning as `apply_projective_ablation`, which normalises inside
        for this reason.

        MUTATION CONTROL: use the raw directions and this fails.
        """
        from src.services.jlens_intervention import apply_coordinate_swap

        a, b = self._orthogonal()
        h = torch.zeros(8)
        h[0], h[1] = 3.0, -7.0

        tight = apply_coordinate_swap(h, a, b)
        loose = apply_coordinate_swap(h, a * 50.0, b * 0.02)
        assert torch.allclose(tight, loose, atol=1e-5), (
            "the direction magnitudes changed the exchange"
        )

    def test_a_zero_direction_is_refused(self):
        from src.services.jlens_intervention import apply_coordinate_swap

        a, _ = self._orthogonal()
        with pytest.raises(ValueError, match="zero direction"):
            apply_coordinate_swap(torch.randn(8), a, torch.zeros(8))
