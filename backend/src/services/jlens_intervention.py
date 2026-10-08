"""
Intervention primitives over lens directions (BR-016, BR-017, BR-018).

THE ONLY PLACE IN THE J-SPACE ARC THAT PRODUCES A CAUSAL CLAIM. Everything else
observes; this intervenes, and that is why the control requirement below is
absolute rather than advisory.

A RUN WITHOUT ITS CONTROL IS INVALID (BR-018), and that is enforced by making
the result IMPOSSIBLE TO CONSTRUCT without one. `InterventionResult` takes its
control positionally: no default, no Optional, no `validate()` to remember.

The reason is that the failure mode here is social rather than technical. Under
time pressure the control is the step that gets skipped, and a validating check
is the step that gets bypassed with a flag. An object that cannot be built
without its control cannot be reported without it either.

Every interpretation of an intervention rests on the comparison: moving the
output tells you nothing until you know what moving a RANDOM direction of the
same size does. Making the control optional makes the finding optional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

logger = logging.getLogger(__name__)


class Primitive(str, Enum):
    """The four primitives (BR-017). Recorded with every result.

    A run whose primitive is unrecorded cannot be reproduced or compared to
    another run — the parameters are as much a part of the finding as the
    numbers are.
    """

    ADDITIVE = "additive"
    PROJECTIVE_ABLATION = "projective_ablation"
    DYNAMIC_TOPK_ABLATION = "dynamic_topk_ablation"
    COORDINATE_SWAP = "coordinate_swap"


@dataclass(frozen=True)
class ControlSpec:
    """A size-matched random-direction control.

    "A random direction" is not a control; "k random directions from seed s" is.
    Both fields are required: without `k` the control is not size-matched, and
    without `seed` nobody else can reconstruct it.
    """

    k: int
    seed: int
    construction: str = "gaussian_unit_norm"

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("a control must be size-matched to the intervention (k > 0)")


@dataclass(frozen=True)
class ClampSpec:
    """Lens coordinates held at their clean-pass values (BR-016).

    Keyed by (position, layer) DELIBERATELY. Clamping a coordinate at some
    positions and not others produces a mediation result that is not about the
    thing it names — the effect leaks through the unclamped positions and is
    reported as the clamped quantity.
    """

    coordinates: Dict[Tuple[int, int], Sequence[int]] = field(default_factory=dict)

    def held_at(self, position: int, layer: int) -> Sequence[int]:
        return self.coordinates.get((position, layer), ())

    @property
    def is_empty(self) -> bool:
        return not self.coordinates


@dataclass
class InterventionResult:
    """An intervention and the control it is meaningless without.

    `control_outcome` is POSITIONAL and has no default. That is the enforcement
    of BR-018: a caller who has not run the control cannot construct a result to
    report. See the module docstring for why this is structural rather than a
    validation step.
    """

    primitive: Primitive
    parameters: Dict[str, object]
    control: ControlSpec
    intervened_outcome: float
    control_outcome: float
    layers: List[int]
    positions: List[int]
    clamp: Optional[ClampSpec] = None

    @property
    def excess_over_control(self) -> float:
        """THE finding. The raw intervened outcome is not one.

        An intervention that moves the output says nothing until compared with
        what a random direction of the same size does.
        """
        return self.intervened_outcome - self.control_outcome


def apply_additive(
    activation: torch.Tensor, direction: torch.Tensor, strength: float
) -> torch.Tensor:
    """Steer along a named direction."""
    return activation + strength * direction


def apply_projective_ablation(
    activation: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Remove the activation's component along `direction`.

    Normalised inside rather than assuming a unit vector: an unnormalised
    direction silently scales the ablation by its own magnitude, which looks
    like a stronger effect rather than a bug.
    """
    norm = torch.linalg.norm(direction)
    if norm == 0:
        return activation
    unit = direction / norm
    return activation - (activation @ unit) * unit


class DirectionsTooSimilar(ValueError):
    """Raised rather than "swapping" two directions that are the same direction."""


#: Above this cosine the two directions are the same direction for practical
#: purposes, and exchanging their coordinates is a no-op dressed as an
#: intervention. Refused rather than run: a swap that cannot move anything would
#: report a null and invite the reading that the concept does not steer.
MAX_SWAP_COSINE = 0.99


def apply_coordinate_swap(
    activation: torch.Tensor, direction_a: torch.Tensor, direction_b: torch.Tensor
) -> torch.Tensor:
    """EXCHANGE the activation's components along two directions.

    This is the primitive acting on the RESIDUAL STREAM, which is where an
    intervention has to happen — `coordinate_swap` below acts on a vector of
    lens coordinates and is a different, unwired thing.

    WHY THIS EXISTS. The perturbing hook had two branches, projective ablation
    and everything-else-is-additive, so a request for `coordinate_swap` ran an
    ADDITIVE steer and the result was then labelled `coordinate_swap` in its
    `steering_recipe`. Since the recipe is written into `interventions.json`
    and that file is built to travel with the lens, the mislabelling would have
    become false provenance in whatever consumed it.

    The exchange, with `c_x` the component along unit direction `x`:

        h' = h + (c_b - c_a) * a_hat + (c_a - c_b) * b_hat

    NON-ORTHOGONAL DIRECTIONS LEAVE CROSS-TALK, and that is stated rather than
    hidden: the new component along `a_hat` is `c_b + (c_a - c_b)(a_hat . b_hat)`,
    so the exchange is exact only when the two are orthogonal and approximate in
    proportion to their overlap. Two unembedding rows for unrelated tokens are
    close to orthogonal in a high-dimensional space, which is the case this is
    used in; near-parallel directions are refused outright because there is
    nothing to exchange.
    """
    norm_a = torch.linalg.norm(direction_a)
    norm_b = torch.linalg.norm(direction_b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("a zero direction has no coordinate to swap")

    unit_a = direction_a / norm_a
    unit_b = direction_b / norm_b

    cosine = float(torch.abs(unit_a @ unit_b))
    if cosine > MAX_SWAP_COSINE:
        raise DirectionsTooSimilar(
            f"the two directions have cosine {cosine:.4f}; exchanging their "
            "coordinates would move almost nothing. Pick two tokens that "
            "differ."
        )

    c_a = activation @ unit_a
    c_b = activation @ unit_b
    return activation + (c_b - c_a) * unit_a + (c_a - c_b) * unit_b


def dynamic_topk_ablation(
    coordinates: torch.Tensor, k: int, clean_pass_topk: Set[int]
) -> torch.Tensor:
    """Ablate the top-k J-space coordinates, EXCLUDING the clean pass's.

    The exclusion is the whole point. Without it this ablates the model's
    ordinary behaviour at that position and reports the consequence as an
    intervention effect — a finding about the technique rather than the model.
    """
    if k <= 0:
        return coordinates
    out = coordinates.clone()
    ranked = torch.argsort(coordinates.abs(), descending=True).tolist()
    ablated = 0
    for idx in ranked:
        if ablated >= k:
            break
        if idx in clean_pass_topk:
            continue
        out[idx] = 0.0
        ablated += 1
    return out


def coordinate_swap(
    coordinates: torch.Tensor, source: int, target: int
) -> torch.Tensor:
    """Replace one coordinate's value with another's."""
    out = coordinates.clone()
    out[target] = coordinates[source]
    return out


def default_swap_layers(n_layers: int) -> int:
    """How many layers a coordinate swap touches by default.

    DERIVED, never a constant (BR-017 v0.2). Swaps oversteer at small scale, so
    a default tuned on a large model applies far too much of the stack to a
    small one — which is the amendment's whole reason for existing. A quarter of
    the stack, floored at one.
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    return max(1, n_layers // 4)


def build_control(k: int, seed: int, d_model: int) -> torch.Tensor:
    """`k` size-matched random directions, reconstructible from `seed`."""
    spec = ControlSpec(k=k, seed=seed)
    generator = torch.Generator().manual_seed(spec.seed)
    directions = torch.randn(spec.k, d_model, generator=generator)
    return directions / directions.norm(dim=-1, keepdim=True)
