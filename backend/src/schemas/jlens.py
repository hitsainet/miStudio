"""
Pydantic schemas for J-space lens readouts.

WIRE FORMAT IS NOT OURS TO DESIGN (BR-029, PADR IDL-45). These shapes mirror
Neuronpedia's lens stream exactly, so a miStudio stream and a Neuronpedia stream
are interchangeable at the client and the readout panel is driven by either with
no adaptation layer. Adopting the upstream shape also removes a whole class of
contract invention and satisfies part of the projection obligation structurally.

    meta  = { model, types, layers_by_type, top_n, prompt_len }
    token = { position, token, id, is_generated, results: slice[] }
    slice = { type, top_tokens[layer][k], top_probs[layer][k] }

Two details are load-bearing and are enforced here rather than by convention:

  * `top_tokens` entries are DECODED STRINGS, not token ids. Emitting ids
    type-checks fine against a looser schema and renders as unreadable cells in
    a client expecting text.
  * `layers_by_type` is per lens type and drives the client's layer axis. The
    reference panel hardcodes 21 layers at 0,5,...,100; the reference model has
    16. Nothing may assume a layer count or spacing.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The lens types the stream can carry. DIFF is a client-side rendering mode over
# two slices, never a transported type — the server does not emit it.
LensType = Literal["JACOBIAN_LENS", "LOGIT_LENS"]

# Readout mode (BR-008). These answer different questions and are not
# interchangeable; see LayerApplicability and DecompositionResult.
ReadoutMode = Literal["full_ranked", "probe", "decomposition"]


class LayerApplicability(BaseModel):
    """Which computations are meaningful at one layer.

    Layer kind is PER-LAYER state, not a model property. The reference model
    interleaves 10 convolutional layers with 6 attention layers, so
    "freeze Q/K" is undefined on 10 of 16 and attention-broadcast metrics are
    computable on 6.

    Inapplicable is expressed as None (absent), never False/0. A False would be
    averaged by downstream consumers and would silently understate; an absent
    value forces the consumer to decide what to do about it (BR-032).
    """

    model_config = ConfigDict(from_attributes=True)

    layer: int = Field(..., ge=0, description="Layer index within the model")
    has_attention: bool = Field(
        ..., description="Whether this layer performs attention at all"
    )
    frozen_qk_applicable: Optional[bool] = Field(
        None,
        description=(
            "Whether the frozen-Q/K recipe variant is meaningful here. None "
            "means INAPPLICABLE (e.g. a convolutional layer), which is not the "
            "same as False"
        ),
    )
    broadcast_metrics_applicable: Optional[bool] = Field(
        None,
        description=(
            "Whether attention-broadcast metrics can be computed here. None "
            "means inapplicable, not zero"
        ),
    )


class LensTypeSlice(BaseModel):
    """One lens type's readout for one token position, across layers.

    `top_tokens[layer_idx][k]` and `top_probs[layer_idx][k]` are parallel: the
    outer index is position in `meta.layers_by_type[type]`, NOT the model's
    absolute layer number.
    """

    model_config = ConfigDict(from_attributes=True)

    type: LensType
    top_tokens: List[List[str]] = Field(
        ...,
        description="Decoded token strings, [layer][k]. NOT token ids.",
    )
    top_probs: List[List[float]] = Field(
        ..., description="Probabilities parallel to top_tokens, [layer][k]"
    )

    @field_validator("top_tokens", mode="before")
    @classmethod
    def _tokens_must_be_strings(cls, v: Any) -> Any:
        """Reject token ids with an explanation, before pydantic's type error.

        mode="before" is deliberate. Pydantic already refuses int -> str, so an
        "after" validator here could NEVER FIRE — dead code that reads as
        protection. Running before type coercion means this fires first and
        says why the value is wrong, instead of "Input should be a valid
        string" against a bare integer.
        """
        if not isinstance(v, list):
            return v
        for layer_idx, row in enumerate(v):
            if not isinstance(row, list):
                continue
            for k, tok in enumerate(row):
                if not isinstance(tok, str):
                    raise ValueError(
                        f"top_tokens[{layer_idx}][{k}] is {type(tok).__name__} "
                        f"({tok!r}), expected a DECODED STRING. Emitting token "
                        "ids here type-checks against a looser schema and "
                        "renders as unreadable cells in the client."
                    )
        return v

    @model_validator(mode="after")
    def _shapes_must_agree(self) -> "LensTypeSlice":
        if len(self.top_tokens) != len(self.top_probs):
            raise ValueError(
                f"top_tokens has {len(self.top_tokens)} layers but top_probs has "
                f"{len(self.top_probs)}"
            )
        for i, (toks, probs) in enumerate(zip(self.top_tokens, self.top_probs)):
            if len(toks) != len(probs):
                raise ValueError(
                    f"layer index {i}: {len(toks)} tokens vs {len(probs)} probs"
                )
        return self


class LensMetaMessage(BaseModel):
    """Opening message of a readout stream."""

    model_config = ConfigDict(from_attributes=True)

    kind: Literal["meta"] = "meta"
    model: str = Field(..., description="Model identity the readout was taken from")
    types: List[LensType] = Field(..., min_length=1)
    layers_by_type: Dict[str, List[int]] = Field(
        ...,
        description=(
            "Absolute layer indices carried per lens type. Drives the client's "
            "layer axis; never assume a count or spacing"
        ),
    )
    top_n: int = Field(..., gt=0)
    prompt_len: int = Field(..., ge=0)

    # Per-layer applicability travels with the stream so a consumer never has to
    # infer homogeneity from the layer count (BR-032).
    layer_applicability: Optional[List[LayerApplicability]] = None

    @model_validator(mode="after")
    def _types_have_layers(self) -> "LensMetaMessage":
        missing = [t for t in self.types if t not in self.layers_by_type]
        if missing:
            raise ValueError(f"layers_by_type missing entries for {missing}")
        return self


class LensTokenMessage(BaseModel):
    """One token position, carrying one slice per lens type."""

    model_config = ConfigDict(from_attributes=True)

    kind: Literal["token"] = "token"
    position: int = Field(..., ge=0)
    token: str
    id: int
    is_generated: bool = False
    results: List[LensTypeSlice] = Field(..., min_length=1)


class LensDoneMessage(BaseModel):
    kind: Literal["done"] = "done"


class LensErrorMessage(BaseModel):
    """Terminal error. `reason` is required — a bare error message is unusable."""

    kind: Literal["error"] = "error"
    reason: str = Field(..., min_length=1)


class ReadoutRequest(BaseModel):
    """Request for a position x layer readout."""

    model_config = ConfigDict(from_attributes=True)

    model_id: str
    # Bounded deliberately. Readout cost is O(positions x layers x top_n) and
    # every position holds a d_model residual, so an unbounded prompt is a
    # memory-and-compute amplification from a single request.
    prompt: str = Field(..., min_length=1, max_length=8000)
    types: List[LensType] = Field(
        default_factory=lambda: ["LOGIT_LENS"], min_length=1, max_length=2
    )
    layers: Optional[List[int]] = Field(
        None,
        max_length=512,
        description="Absolute layer indices; None means every layer of the model",
    )
    top_n: int = Field(8, gt=0, le=100)

    # Identity of the J-lens artifact to use when types include JACOBIAN_LENS.
    # Absent is valid and means logit-only, which needs no artifact (BR-005).
    artifact_id: Optional[str] = None

    @model_validator(mode="after")
    def _jacobian_needs_an_artifact(self) -> "ReadoutRequest":
        if "JACOBIAN_LENS" in self.types and not self.artifact_id:
            raise ValueError(
                "JACOBIAN_LENS requested without artifact_id. The logit lens "
                "needs no artifact; the Jacobian lens does, and silently "
                "serving logit data under a Jacobian label is prohibited "
                "(BR-019 rung discipline)."
            )
        return self


class DecompositionProvenance(BaseModel):
    """Mandatory provenance for a sparse decomposition (BR-009).

    The decomposition is non-unique by construction: reproducibility is a
    provenance property, not a mathematical one. A figure without
    `control_seed` is INVALID rather than merely undocumented, because
    occupancy and excess-FVE are defined against a random-direction control.
    """

    model_config = ConfigDict(from_attributes=True)

    k: int = Field(..., gt=0, description="Sparsity level")
    solver: str = Field(..., min_length=1)
    solver_params: Dict[str, Any] = Field(default_factory=dict)
    iterations: int = Field(..., ge=0)
    convergence_criterion: str = Field(..., min_length=1)
    control_seed: int = Field(
        ...,
        description=(
            "Seed for the size-matched random-direction control. REQUIRED — "
            "occupancy and excess-FVE are defined as an excess over this "
            "control, so a figure without it cannot be reproduced or believed"
        ),
    )
    control_construction: str = Field(..., min_length=1)


class DecompositionResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    layer: int = Field(..., ge=0)
    position: int = Field(..., ge=0)
    active_tokens: List[str]
    coefficients: List[float]
    residual_norm: float
    provenance: DecompositionProvenance

    @model_validator(mode="after")
    def _active_set_is_parallel(self) -> "DecompositionResult":
        if len(self.active_tokens) != len(self.coefficients):
            raise ValueError(
                f"{len(self.active_tokens)} active tokens vs "
                f"{len(self.coefficients)} coefficients"
            )
        if len(self.active_tokens) > self.provenance.k:
            raise ValueError(
                f"active set of {len(self.active_tokens)} exceeds sparsity "
                f"level k={self.provenance.k}"
            )
        return self


class ProbeRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: str
    prompt: str = Field(..., min_length=1, max_length=8000)
    tokens: List[str] = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Named directions to score",
    )
    layers: Optional[List[int]] = Field(None, max_length=512)
    artifact_id: Optional[str] = None


class ProbeScore(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    layer: int
    position: int
    token: str
    score: float


class JLensArtifactRecipe(BaseModel):
    """Construction recipe, sufficient to rebuild the artifact (BR-007)."""

    model_config = ConfigDict(from_attributes=True)

    # DEFAULTS MATCH THE PAPER, and the fitter now implements it. An earlier
    # revision "fixed" these the wrong way round: the fitter took one source
    # position on a length-1 sub-network, so the schema was edited to say
    # `self_only_isolated` and `final` to match the CODE. That made schema and
    # implementation agree while both disagreed with the source. Alignment is
    # against the paper; the code moved to meet it.
    target_layer: Literal["final", "penultimate"] = "penultimate"
    attention_gradients: Literal["full", "frozen_qk"] = "full"
    target_position_scope: Literal[
        "self_only", "self_only_isolated", "future_only", "all_subsequent"
    ] = "all_subsequent"
    aggregation: Literal["mean", "median"] = "mean"
    corpus: str
    n_prompts: int = Field(..., ge=100, description="Floor of 100 (Appendix A.2)")
    seq_len: int = Field(..., gt=0)
    convergence_delta: Optional[float] = None
    dtype: Literal["fp16"] = "fp16"
    library_versions: Dict[str, str] = Field(default_factory=dict)

    # Recorded per layer, because a recipe choice can be inapplicable to a layer
    # rather than merely unset (BR-032). An artifact must not be described as
    # "frozen-Q/K" wholesale when the treatment reached only a subset.
    per_layer_applicability: List[LayerApplicability] = Field(default_factory=list)


class JLensArtifact(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    artifact_id: str
    model_id: str
    d_model: int = Field(..., gt=0)
    n_layers: int = Field(..., gt=0)
    n_vocab: int = Field(..., gt=0)
    layers: List[int]
    size_bytes: int = Field(..., gt=0)
    recipe: JLensArtifactRecipe

    def expected_size_bytes(self, dtype_bytes: int = 2) -> int:
        """Envelope bound derived from THIS model's dimensions.

        Never a constant: the required-vs-materialised ratio scales with
        vocabulary, so a bound hardcoded for one model passes on another while
        missing a real materialisation. Reference model ~32x, a 256k-vocab
        model ~111x (PADR IDL-42).
        """
        return self.d_model * self.d_model * dtype_bytes * self.n_layers

    def materialized_size_bytes(self, dtype_bytes: int = 2) -> int:
        """What the PROHIBITED materialised dictionary would cost."""
        return self.n_vocab * self.d_model * dtype_bytes * self.n_layers
