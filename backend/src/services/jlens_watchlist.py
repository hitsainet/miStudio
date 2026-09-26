"""
Watchlists and the cost envelope (BR-025, BR-026, BR-028).

THIS INCREMENT EMITS ONLY. Runtime evaluation is miLLM's plane; miStudio
authors, validates, versions and exports the artifact.

A WATCHLIST IS A DETECTOR DEFINITION, NOT A LIST OF WORDS. Three things travel
together or none of them mean anything:

  * the DIRECTIONS, which are artifact-specific — lens coordinates only mean
    something relative to the lens they were computed in;
  * the THRESHOLDS;
  * the SCORING DEFINITION, because a threshold applied to a differently
    computed score is a different detector at the far end, and the consumer has
    no way to notice.

THE EVALUATION-AWARENESS SCORE IS A DIFFERENCE (BR-026). The subtraction is not
a refinement of the measurement, it IS the measurement: common tokens carry high
log-probability everywhere, so the raw mean makes every prompt look
evaluation-aware — a detector that fires constantly and is therefore never
consulted. The subtraction lives inside the function rather than in the caller,
because a caller who must remember to subtract will eventually not.
"""

from __future__ import annotations

import math

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchedConcept:
    """One concept and the threshold at which it counts as detected."""

    token: str
    threshold: float


@dataclass
class Watchlist:
    """A portable detector definition (BR-025).

    `scoring_definition` and `artifact_ref` are REQUIRED. A watchlist missing
    either is refused at construction rather than exported and discovered
    later: the consumer cannot tell a mis-scored detector from a working one.
    """

    name: str
    concepts: List[WatchedConcept]
    scoring_definition: str
    artifact_ref: str
    control_set: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.scoring_definition.strip():
            raise ValueError(
                "a watchlist requires its scoring definition. A threshold "
                "applied to a differently computed score is a different "
                "detector, and the consumer has no way to notice."
            )
        if not self.artifact_ref.strip():
            raise ValueError(
                "a watchlist requires the artifact it was built against. Lens "
                "coordinates are artifact-specific and mean nothing elsewhere."
            )
        if not self.concepts:
            raise ValueError("a watchlist with no concepts detects nothing")


def evaluation_awareness_score(
    eval_token_logprobs: Sequence[float],
    control_token_logprobs: Sequence[float],
) -> float:
    """The source paper's evaluation-awareness score, per layer.

    THE SUBTRACTION IS THE MEASUREMENT. Reporting the raw mean over the
    evaluation tokens makes every prompt look evaluation-aware, because common
    tokens have high log-probability everywhere. The control set is unrelated
    concrete nouns, and the difference is what carries signal.

    Kept inside this function rather than left to the caller for the obvious
    reason: a caller who must remember to subtract will eventually not, and the
    resulting detector fires constantly while looking like it works.
    """
    if not eval_token_logprobs:
        raise ValueError("no evaluation-awareness tokens were scored")
    if not control_token_logprobs:
        raise ValueError(
            "no control tokens were scored. Without the control this is a raw "
            "mean, which is high for common tokens in any prompt."
        )
    eval_mean = sum(eval_token_logprobs) / len(eval_token_logprobs)
    control_mean = sum(control_token_logprobs) / len(control_token_logprobs)
    return eval_mean - control_mean


def score_across_layers(
    per_layer: Sequence[tuple],
) -> float:
    """Average the per-layer difference across layers.

    Averaging the DIFFERENCES, not differencing the averages — they coincide
    only when every layer contributes equally, which is not guaranteed when a
    layer is skipped.
    """
    if not per_layer:
        raise ValueError("no layers were scored")
    scores = [evaluation_awareness_score(e, c) for e, c in per_layer]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------- envelope


class OperationClass(str, Enum):
    """Every J-space operation that costs something (BR-028)."""

    ARTIFACT_CONSTRUCTION = "artifact_construction"
    READOUT = "readout"
    DECOMPOSITION = "decomposition"
    ANNOTATION_SWEEP = "annotation_sweep"
    INTERVENTION_RUN = "intervention_run"
    TEMPLATE_LENS_BUILD = "template_lens_build"


@dataclass(frozen=True)
class CostEstimate:
    """An ORDER-OF-MAGNITUDE estimate, and labelled as one.

    False precision invites planning against a number nobody measured. These
    are derived from the operation's own arithmetic, not from a benchmark, and
    the field name says so.
    """

    operation: OperationClass
    order_of_magnitude_seconds: float
    order_of_magnitude_peak_bytes: int
    basis: str

    @property
    def is_estimate(self) -> bool:
        return True


#: fp32 copies of the d_model x d_model Jacobian held PER CAPTURED LAYER: the
#: running mean, the two split-half accumulators and the working buffers the
#: update needs. Seven reproduces the 0.41 GB per layer measured on the
#: gemma-4-12B fit (7 x 3840^2 x 4 = 413 MB), which is how it was chosen.
_ACCUMULATOR_COPIES_PER_LAYER = 7

#: Seconds per (batched-backward iteration x captured layer), calibrated on the
#: ONLY end-to-end fit anyone has measured (gemma-4-12B at Q8, d_model 3840, four
#: layers, 600 prompts, 2026-09-05): 41 s/prompt at 120 iterations per prompt and
#: four layers gives 41 / (120 x 4). The estimate is therefore EXACT at that point
#: by construction and an extrapolation everywhere else, with the exponents taken
#: from the algorithm rather than guessed.
_SECONDS_PER_ITERATION_PER_LAYER = 41.0 / (120 * 4)

#: The reference point, kept next to the constant it calibrates so a future
#: measurement can be checked against it instead of replacing it silently.
FIT_REFERENCE = {
    "model": "gemma-4-12b-it @ Q8",
    "d_model": 3840,
    "n_layers": 4,
    "n_prompts": 600,
    "measured_seconds_per_prompt": 41.0,
    "measured_peak_bytes": int(16.6 * 1024 ** 3),
    "measured_weights_bytes": int(12.79 * 1024 ** 3),
    "recorded": "0xcc/tasks/DEBT_TASKS|JLens_Fit_On_24GB_2026-09-05.md",
}


def _fit_estimate(
    *,
    d_model: int,
    n_layers: int,
    n_positions: int,
    n_prompts: int,
    weights_bytes: int,
) -> CostEstimate:
    """Cost of fitting a J-lens, built on the FITTER'S OWN chunking.

    The previous model was ~14x low on the one run anyone has measured — it
    predicted 24 minutes for a fit that took nearly seven hours — and it modelled
    an algorithm the fitter does not use. Two structural facts it missed:

    * **The backward is CAPPED, not proportional.** `narrowed_chunk` shrinks the
      chunk so `chunk x seq_len x d_model x n_layers x 4` stays under
      `MAX_BACKWARD_BYTES`, so peak memory does NOT grow with layer count — only
      the per-layer accumulators do. Peak was modelled as `d_model^2 x 2 x
      n_layers`, which is 118 MB for the reference fit against 16.6 GB measured,
      because the resident weights were not in the model at all.
    * **Time grows with layer count** either way: when the cap binds, more layers
      mean a narrower chunk and proportionally more `autograd.grad` iterations;
      when it does not, each iteration backpropagates through more captured
      layers. Both are linear, so the estimate multiplies by `n_layers` once.

    This calls `narrowed_chunk` rather than restating it, so the estimate cannot
    drift from the fitter's chunking.
    """
    from ..ml.jlens_fitter import DEFAULT_CHUNK, MAX_BACKWARD_BYTES, narrowed_chunk

    chunk = narrowed_chunk(DEFAULT_CHUNK, max(1, n_positions), d_model, n_layers)
    iterations_per_prompt = math.ceil(d_model / max(1, chunk))
    seconds = (
        n_prompts * iterations_per_prompt * n_layers * _SECONDS_PER_ITERATION_PER_LAYER
    )

    accumulators = _ACCUMULATOR_COPIES_PER_LAYER * d_model * d_model * 4 * n_layers
    peak = int(max(0, weights_bytes) + MAX_BACKWARD_BYTES + accumulators)

    basis = (
        f"{n_prompts} prompts x {iterations_per_prompt} batched-backward iterations "
        f"(d_model/chunk, chunk={chunk} from the fitter's own narrowed_chunk) x "
        f"{n_layers} layers, at {_SECONDS_PER_ITERATION_PER_LAYER:.4f}s each, "
        f"calibrated on the measured gemma-4-12B fit. Peak = resident weights "
        f"({weights_bytes / 1024 ** 3:.2f} GiB as given) + the {MAX_BACKWARD_BYTES / 1024 ** 3:.0f} GiB "
        f"backward cap + {_ACCUMULATOR_COPIES_PER_LAYER} fp32 d_model^2 accumulators "
        f"per layer. Pass weights_bytes: omitted, the peak EXCLUDES the model, "
        f"which dominates it."
    )
    return CostEstimate(
        OperationClass.ARTIFACT_CONSTRUCTION,
        order_of_magnitude_seconds=max(60.0, seconds),
        order_of_magnitude_peak_bytes=peak,
        basis=basis,
    )


def estimate_cost(
    operation: OperationClass,
    *,
    d_model: int,
    n_layers: int,
    n_positions: int = 1,
    n_prompts: int = 1,
    n_features: int = 1,
    weights_bytes: int = 0,
) -> CostEstimate:
    """Estimate before committing (BR-028).

    An UNKNOWN operation class RAISES rather than returning a cheap default. A
    defaulted-cheap estimate is worse than none at all: it invites exactly the
    run it should have warned about, and an agent has no way to tell the
    difference between "cheap" and "unmeasured".
    """
    dtype_bytes = 4

    if operation is OperationClass.READOUT:
        cells = n_positions * n_layers
        return CostEstimate(
            operation,
            order_of_magnitude_seconds=max(1.0, cells / 200.0),
            order_of_magnitude_peak_bytes=n_positions * d_model * dtype_bytes,
            basis="positions x layers matvecs, plus the resident model",
        )

    if operation is OperationClass.ARTIFACT_CONSTRUCTION:
        return _fit_estimate(
            d_model=d_model,
            n_layers=n_layers,
            n_positions=n_positions,
            n_prompts=n_prompts,
            weights_bytes=weights_bytes,
        )

    if operation is OperationClass.ANNOTATION_SWEEP:
        return CostEstimate(
            operation,
            order_of_magnitude_seconds=max(10.0, n_features * n_layers / 500.0),
            order_of_magnitude_peak_bytes=d_model * dtype_bytes,
            basis="one projection per feature per layer",
        )

    if operation is OperationClass.INTERVENTION_RUN:
        # Doubled: every run executes against its control (BR-018).
        return CostEstimate(
            operation,
            order_of_magnitude_seconds=max(2.0, n_positions * n_layers / 100.0) * 2,
            order_of_magnitude_peak_bytes=n_positions * d_model * dtype_bytes * 2,
            basis="paired clean/intervened passes, doubled for the mandatory control",
        )

    if operation is OperationClass.DECOMPOSITION:
        return CostEstimate(
            operation,
            order_of_magnitude_seconds=max(5.0, n_positions * n_layers / 50.0),
            order_of_magnitude_peak_bytes=d_model * dtype_bytes,
            basis="sparse pursuit per (position, layer)",
        )

    if operation is OperationClass.TEMPLATE_LENS_BUILD:
        return CostEstimate(
            operation,
            order_of_magnitude_seconds=max(60.0, n_prompts * n_features / 10.0),
            order_of_magnitude_peak_bytes=d_model * d_model * dtype_bytes,
            basis="contexts per phrase, plus a covariance inverse",
        )

    raise ValueError(
        f"no cost estimator for {operation!r}. Refusing to default: a "
        "cheap-looking estimate invites exactly the run it should warn about."
    )
