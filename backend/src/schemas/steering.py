"""
Pydantic schemas for Steering API endpoints.

These schemas define the structure for request/response validation
and serialization for feature steering comparison operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal, Union

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Feature Selection Schemas
# ============================================================================

class SelectedFeature(BaseModel):
    """Schema for a feature selected for steering."""

    instance_id: Optional[str] = Field(
        None,
        description="Unique identifier for this selection instance (allows duplicates of same feature)"
    )
    comparison_id: Optional[str] = Field(
        None,
        description="Links to the comparison job this instance was used in"
    )
    feature_idx: int = Field(..., ge=0, description="Feature index in the SAE")
    layer: int = Field(..., ge=0, description="Target layer for steering (L0, L1, etc.)")
    # Feature 015: per-feature SAE — a multi-layer circuit steers each feature
    # through the SAE trained on ITS layer. Omitted ⇒ the request-level sae_id
    # (back-compatible: single-layer flows send no per-feature sae_id).
    sae_id: Optional[str] = Field(
        None, description="SAE for THIS feature (015); defaults to the request sae_id")
    strength: float = Field(
        ...,
        ge=-300.0,
        le=300.0,
        description=(
            "Raw steering coefficient (Neuronpedia-compatible). "
            "Values like 0.07 for subtle effects, 80 for strong effects, 200+ for extreme. "
            "Negative values suppress the feature."
        )
    )
    additional_strengths: Optional[List[float]] = Field(
        default=None,
        description="Up to 3 additional strengths to test simultaneously"
    )
    label: Optional[str] = Field(None, description="Feature label for display")
    # Feature 011: the UI now supports up to 20 features, each drawn from a
    # 20-name palette (the original 4 first, for continuity). Colors are purely
    # cosmetic; uniqueness is not required. Kept as a Literal so the accepted
    # set stays documented and in lock-step with the frontend FeatureColor union.
    color: Literal[
        "teal", "blue", "purple", "amber", "rose", "cyan", "lime", "orange",
        "fuchsia", "sky", "emerald", "violet", "pink", "indigo", "yellow",
        "red", "green", "sapphire", "magenta", "gold",
    ] = Field(
        "teal",
        description="Color for UI display (cosmetic; one of the 20-name palette)"
    )

    @field_validator("strength")
    @classmethod
    def validate_strength(cls, v: float) -> float:
        """Validate steering strength - now Neuronpedia-compatible raw coefficients."""
        # Just validate range - warnings are handled in UI
        return v

    @field_validator("additional_strengths")
    @classmethod
    def validate_additional_strengths(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate additional strengths - max 3, each in valid range."""
        if v is None:
            return v
        if len(v) > 3:
            raise ValueError("Maximum 3 additional strengths allowed")
        for strength in v:
            if strength < -300.0 or strength > 300.0:
                raise ValueError("Additional strength must be between -300 and 300")
        return v


# ============================================================================
# Generation Config Schemas
# ============================================================================

class GenerationParams(BaseModel):
    """Schema for text generation parameters."""

    max_new_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int = Field(50, ge=0, le=500, description="Top-k sampling (0 to disable)")
    num_samples: int = Field(1, ge=1, le=10, description="Number of samples per configuration")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class AdvancedGenerationParams(BaseModel):
    """Schema for advanced generation parameters."""

    repetition_penalty: float = Field(1.15, ge=0.5, le=2.0, description="Repetition penalty (1.0=none, 1.1-1.2=mild, 1.3+=strong)")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    do_sample: bool = Field(True, description="Whether to use sampling (vs greedy)")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")


# ============================================================================
# Steering Request Schemas
# ============================================================================

class SteeringComparisonRequest(BaseModel):
    """Schema for generating a steering comparison."""

    # SAE identification
    sae_id: str = Field(..., description="SAE ID to use for steering")

    # Model identification (optional - uses SAE's linked model by default)
    model_id: Optional[str] = Field(None, description="Model ID (defaults to SAE's linked model)")

    # Prompt
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt for generation")

    # Selected features for steering (up to 20 — Feature 011)
    selected_features: List[SelectedFeature] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of features to steer with (1-20)"
    )

    # Generation parameters
    generation_params: GenerationParams = Field(
        default_factory=GenerationParams,
        description="Generation parameters"
    )
    advanced_params: AdvancedGenerationParams = Field(
        default_factory=AdvancedGenerationParams,
        description="Advanced generation parameters"
    )

    # Options
    include_unsteered: bool = Field(True, description="Include unsteered baseline output")
    compute_metrics: bool = Field(True, description="Compute evaluation metrics")

    # NOTE (Feature 011): the former unique-color validator was removed. With up
    # to 20 features and a cosmetic color palette, per-feature color uniqueness
    # can no longer hold and is not required by the compare pipeline.


class SteeringStrengthSweepRequest(BaseModel):
    """Schema for a steering strength sweep (testing multiple strengths)."""

    sae_id: str = Field(..., description="SAE ID to use for steering")
    model_id: Optional[str] = Field(None, description="Model ID")
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt")

    # Single feature to sweep
    feature_idx: int = Field(..., ge=0, description="Feature index to sweep")
    layer: int = Field(..., ge=0, description="Target layer")

    # Strength sweep range
    strength_values: List[float] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="List of strength values to test (e.g., [0, 25, 50, 100, 200])"
    )

    # Generation parameters
    generation_params: GenerationParams = Field(
        default_factory=GenerationParams,
        description="Generation parameters"
    )


# ============================================================================
# Steering Result Schemas
# ============================================================================

class GenerationMetrics(BaseModel):
    """Schema for generation quality metrics."""

    perplexity: Optional[float] = Field(None, description="Perplexity score (lower = more coherent)")
    # MIS-E2E-063: `None` means NOT MEASURED, and is the correct value when the
    # embedding model is unavailable. It must never be substituted with a
    # placeholder — every coherence score this product displayed for its whole
    # life was the constant 0.5, read by users as a measurement.
    coherence: Optional[float] = Field(
        None, description="Coherence score (0-1); null = not measured"
    )
    behavioral_score: Optional[float] = Field(
        None, description="Behavioral score (0-1); null = not measured"
    )
    token_count: int = Field(..., description="Number of tokens generated")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")


class SteeredOutput(BaseModel):
    """Schema for a single steered generation output."""

    text: str = Field(..., description="Generated text")
    feature_config: SelectedFeature = Field(..., description="Feature configuration used")
    metrics: Optional[GenerationMetrics] = Field(None, description="Generation metrics")


class MultiStrengthResult(BaseModel):
    """Schema for a single strength result in multi-strength mode."""

    strength: float = Field(..., description="Steering strength used")
    text: str = Field(..., description="Generated text")
    metrics: Optional[GenerationMetrics] = Field(None, description="Generation metrics")


class SteeredOutputMulti(BaseModel):
    """Schema for multi-strength steered generation output."""

    feature_config: SelectedFeature = Field(..., description="Feature configuration used")
    primary_result: MultiStrengthResult = Field(..., description="Result at primary strength")
    additional_results: List[MultiStrengthResult] = Field(
        default_factory=list,
        description="Results at additional strength values"
    )


class UnsteeredOutput(BaseModel):
    """Schema for unsteered baseline output."""

    text: str = Field(..., description="Generated text")
    metrics: Optional[GenerationMetrics] = Field(None, description="Generation metrics")


class SteeringComparisonResponse(BaseModel):
    """Schema for steering comparison response."""

    # Identification
    comparison_id: str = Field(..., description="Unique comparison identifier")
    sae_id: str = Field(..., description="SAE ID used")
    model_id: str = Field(..., description="Model ID used")

    # Input
    prompt: str = Field(..., description="Input prompt")

    # Results
    unsteered: Optional[UnsteeredOutput] = Field(None, description="Unsteered baseline output")
    steered: List[SteeredOutput] = Field(..., description="Steered outputs for each feature")
    steered_multi: Optional[List[SteeredOutputMulti]] = Field(
        None,
        description="Multi-strength steered outputs (when additional_strengths provided)"
    )

    # Summary metrics
    metrics_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of metrics across all outputs")

    # Timing
    total_time_ms: int = Field(..., description="Total generation time in milliseconds")
    created_at: datetime = Field(..., description="Comparison creation timestamp")


class StrengthSweepResult(BaseModel):
    """Schema for a single strength sweep result."""

    strength: float = Field(..., description="Steering strength used")
    text: str = Field(..., description="Generated text")
    metrics: Optional[GenerationMetrics] = Field(None, description="Generation metrics")


class StrengthSweepResponse(BaseModel):
    """Schema for strength sweep response."""

    sweep_id: str = Field(..., description="Unique sweep identifier")
    sae_id: str = Field(..., description="SAE ID used")
    model_id: str = Field(..., description="Model ID used")
    prompt: str = Field(..., description="Input prompt")
    feature_idx: int = Field(..., description="Feature index swept")
    layer: int = Field(..., description="Target layer")

    # Results
    unsteered: UnsteeredOutput = Field(..., description="Unsteered baseline")
    results: List[StrengthSweepResult] = Field(..., description="Results for each strength value")

    # Timing
    total_time_ms: int = Field(..., description="Total generation time")
    created_at: datetime = Field(..., description="Sweep creation timestamp")


# ============================================================================
# Experiment Save/Load Schemas
# ============================================================================

class SteeringExperimentSaveRequest(BaseModel):
    """Schema for saving a steering experiment."""

    name: str = Field(..., min_length=1, max_length=255, description="Experiment name")
    description: Optional[str] = Field(None, max_length=2000, description="Experiment description")
    comparison_id: str = Field(..., description="Comparison ID to save")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    # Include the full result since comparisons are ephemeral (stored in Redis with TTL)
    result: Optional[Dict[str, Any]] = Field(None, description="Full SteeringComparisonResponse to save")


class SteeringExperimentResponse(BaseModel):
    """Schema for a saved steering experiment."""

    id: str = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    sae_id: str = Field(..., description="SAE ID used")
    model_id: str = Field(..., description="Model ID used")
    prompt: str = Field(..., description="Input prompt")
    selected_features: List[SelectedFeature] = Field(..., description="Features used")
    generation_params: GenerationParams = Field(..., description="Generation parameters")
    results: SteeringComparisonResponse = Field(..., description="Comparison results")
    tags: List[str] = Field(default_factory=list, description="Tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = {
        "from_attributes": True,
    }


class SteeringExperimentListResponse(BaseModel):
    """Schema for paginated list of steering experiments."""

    data: List[SteeringExperimentResponse] = Field(..., description="List of experiments")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


# ============================================================================
# Real-time Progress Schemas (for WebSocket)
# ============================================================================

class SteeringProgressUpdate(BaseModel):
    """Schema for real-time steering progress updates."""

    comparison_id: str = Field(..., description="Comparison ID")
    status: str = Field(..., description="Current status (loading, generating, computing_metrics, complete)")
    current_config: Optional[str] = Field(None, description="Current configuration being generated")
    progress: float = Field(..., description="Overall progress (0-100)")
    message: Optional[str] = Field(None, description="Status message")


# ============================================================================
# Feature Activation Analysis Schemas
# ============================================================================

class FeatureActivationAnalysis(BaseModel):
    """Schema for analyzing which features activated during generation."""

    feature_idx: int = Field(..., description="Feature index")
    activation_count: int = Field(..., description="Number of tokens where feature activated")
    mean_activation: float = Field(..., description="Mean activation value")
    max_activation: float = Field(..., description="Maximum activation value")
    activated_tokens: List[str] = Field(default_factory=list, description="Tokens where feature activated")


class SteeringEffectAnalysis(BaseModel):
    """Schema for analyzing the effect of steering on activations."""

    target_feature_idx: int = Field(..., description="Target feature index")
    target_feature_activation_change: float = Field(..., description="Change in target feature activation")
    side_effects: List[FeatureActivationAnalysis] = Field(
        default_factory=list,
        description="Top unintended feature activation changes"
    )


# ============================================================================
# Async Task Schemas (for Celery-based steering)
# ============================================================================

class SteeringTaskResponse(BaseModel):
    """
    Schema for async steering task submission response.

    Returned when a steering task is submitted to the Celery queue.
    The client should then subscribe to WebSocket channel steering/{task_id}
    for progress updates, or poll the /steering/result/{task_id} endpoint.
    """

    task_id: str = Field(..., description="Celery task ID for tracking")
    task_type: Literal["compare", "sweep", "combined"] = Field(..., description="Type of steering task")
    status: str = Field("pending", description="Initial task status (pending)")
    websocket_channel: str = Field(..., description="WebSocket channel to subscribe for progress updates")
    message: str = Field(..., description="Human-readable status message")
    submitted_at: datetime = Field(..., description="Task submission timestamp")


class SteeringTaskStatus(BaseModel):
    """
    Schema for steering task status.

    Used when polling for task status or when included in result response.
    """

    task_id: str = Field(..., description="Celery task ID")
    status: Literal["pending", "started", "progress", "success", "failure", "revoked"] = Field(
        ..., description="Current task status"
    )
    percent: int = Field(0, ge=-1, le=100, description="Progress percentage (0-100, -1 for error)")
    message: str = Field("", description="Human-readable status message")
    started_at: Optional[datetime] = Field(None, description="When task started processing")
    completed_at: Optional[datetime] = Field(None, description="When task completed")
    error: Optional[str] = Field(None, description="Error message if failed")


class SteeringResultResponse(BaseModel):
    """
    Schema for async steering task result response.

    Contains the task status and, if completed, the actual result.
    """

    task_id: str = Field(..., description="Celery task ID")
    status: SteeringTaskStatus = Field(..., description="Current task status")
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="Task result (SteeringComparisonResponse or StrengthSweepResponse as dict)"
    )


class SteeringCancelResponse(BaseModel):
    """Schema for steering task cancellation response."""

    task_id: str = Field(..., description="Task ID that was cancelled")
    status: str = Field(..., description="Result of cancellation (cancelled, not_found, already_complete)")
    message: str = Field(..., description="Human-readable message")


# ============================================================================
# Combined Multi-Feature Steering Schemas
# ============================================================================


class CombinedSteeringRequest(BaseModel):
    """
    Schema for combined multi-feature steering request.

    Applies ALL selected features simultaneously in a single generation pass,
    rather than generating separate outputs for each feature.

    Use cases:
    - Test synergistic effects (e.g., "formal" + "positive" = professional tone)
    - Create complex behavioral changes with multiple influences
    - Explore feature interactions and emergent behaviors
    """

    # SAE identification
    sae_id: str = Field(..., description="SAE ID to use for steering")

    # Model identification (optional - uses SAE's linked model by default)
    model_id: Optional[str] = Field(None, description="Model ID (defaults to SAE's linked model)")

    # Prompt
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt for generation")

    # Selected features for combined steering (all applied simultaneously; up to 20 — Feature 011)
    selected_features: List[SelectedFeature] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of features to apply together (1-20)"
    )

    # Generation parameters
    generation_params: GenerationParams = Field(
        default_factory=GenerationParams,
        description="Generation parameters"
    )
    advanced_params: AdvancedGenerationParams = Field(
        default_factory=AdvancedGenerationParams,
        description="Advanced generation parameters"
    )

    # Options
    include_baseline: bool = Field(True, description="Include unsteered baseline output for comparison")
    compute_metrics: bool = Field(True, description="Compute evaluation metrics")


class CombinedFeatureApplied(BaseModel):
    """Schema for a feature that was applied in combined mode."""

    feature_idx: int = Field(..., description="Feature index")
    layer: int = Field(..., description="Target layer")
    # Feature 015: which SAE actually steered this feature (source of truth =
    # the config used at hook time, not the request) — lets the applied summary
    # group by layer and verify each member steered through its own SAE.
    sae_id: Optional[str] = Field(None, description="SAE that steered this feature (015)")
    strength: float = Field(..., description="Steering strength applied")
    label: Optional[str] = Field(None, description="Feature label")
    color: str = Field("teal", description="Color for UI display")


class CombinedSteeringResponse(BaseModel):
    """
    Schema for combined multi-feature steering response.

    Contains a single combined output where all features were applied together,
    optionally with a baseline for comparison.
    """

    # Identification
    combined_id: str = Field(..., description="Unique combined steering identifier")
    sae_id: str = Field(..., description="SAE ID used")
    model_id: str = Field(..., description="Model ID used")

    # Input
    prompt: str = Field(..., description="Input prompt")

    # Combined output
    combined_output: str = Field(..., description="Generated text with all features applied together")

    # Features that were applied
    features_applied: List[CombinedFeatureApplied] = Field(
        ...,
        description="List of features that were applied together"
    )

    # Optional baseline for comparison
    baseline_output: Optional[str] = Field(None, description="Unsteered baseline output (if requested)")

    # Metrics
    combined_metrics: Optional[GenerationMetrics] = Field(None, description="Metrics for combined output")
    baseline_metrics: Optional[GenerationMetrics] = Field(None, description="Metrics for baseline output")

    # Summary of feature contributions
    total_steering_strength: float = Field(
        ...,
        description="Sum of absolute strength values (indicates total intervention intensity)"
    )

    # Timing
    total_time_ms: int = Field(..., description="Total generation time in milliseconds")
    created_at: datetime = Field(..., description="Response creation timestamp")

# ============================================================================
# Cluster Strength Allocation Schemas (Feature 013, IDL-29)
# ============================================================================

class ClusterAllocationMember(BaseModel):
    """A cluster member submitted for strength allocation."""

    feature_idx: int = Field(..., ge=0)
    layer: int = Field(..., ge=0)
    similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Context similarity to the cluster")
    activation_frequency: Optional[float] = Field(None, description="Fraction of tokens where the feature fires")
    sign: Literal[1, -1] = Field(1, description="+1 boost, -1 suppress")
    # Feature 015: the SAE trained on THIS member's layer. Omitted ⇒ the
    # request-level sae_id (single-layer clusters send no per-member sae_id, so
    # the 013 request shape is unchanged). Required (per member) when a cluster
    # spans multiple layers — each layer partition allocates against its own SAE.
    sae_id: Optional[str] = Field(
        None, description="SAE for THIS member's layer (015); defaults to the request sae_id")


class ClusterAllocationRequest(BaseModel):
    """Request a principled starting allocation for steering a cluster."""

    sae_id: str = Field(..., description="SAE whose decoder defines the injected directions")
    members: List[ClusterAllocationMember] = Field(..., min_length=1, max_length=20)
    group_cohesion: Optional[float] = Field(None, ge=0.0, le=1.0, description="Source cluster cohesion for the gate")
    # Feature 015: when set, hazard detection promotes any stored circuit edge at
    # rung ≥2 to the PRIMARY (quantified) evidence source for the same steered
    # pair; absent ⇒ hazards fall back to the labeled weight-prior heuristic.
    circuit_id: Optional[str] = Field(
        None, description="Circuit whose validated edges quantify cross-layer hazards (015)")


class ClusterAllocationResponse(BaseModel):
    """Computed allocation: budget, gain, per-member strengths, flags."""

    B: float = Field(..., description="Total strength budget Σ|strength_i|")
    B_dir: float = Field(..., description="Direction budget from the solo law at f_eff")
    G: float = Field(..., description="Resultant-norm gain ‖Σ σᵢwᵢdᵢ‖ (1.0 when approximate)")
    f_eff: Optional[float] = Field(None, description="Similarity-weighted mean member frequency")
    weights: List[float]
    strengths: List[float] = Field(..., description="Per-member signed strengths (0.1 grain)")
    flags: List[str] = Field(default_factory=list, description="cancellation | low_cohesion | default_budget | cap_bound | approximate | uniform_weights | inactive_member")
    cancellation_pair: Optional[List[int]] = Field(None, description="Worst-opposed feature_idx pair when cancellation flagged")
    constants_used: Dict[str, float]
    formula_id: str
    approximate: bool = False


# ============================================================================
# Multi-Layer Cluster Allocation Schemas (Feature 015, IDL-29 reused per layer)
# ============================================================================

class PerLayerAllocation(BaseModel):
    """One layer's 013 allocation inside a multi-layer response.

    Byte-for-byte the same fields the single-layer response carries (so a client
    that already understands 013 reads each layer entry unchanged) plus the
    `sae_id` that steered this layer's directions.
    """

    sae_id: Optional[str] = Field(None, description="SAE that defined this layer's directions (015)")
    B: float
    B_dir: float
    G: float
    f_eff: Optional[float] = None
    weights: List[float]
    strengths: List[float]
    flags: List[str] = Field(default_factory=list)
    cancellation_pair: Optional[List[int]] = None
    constants_used: Dict[str, float]
    approximate: bool = False


class HazardModel(BaseModel):
    """A cross-layer steering hazard (Feature 015, BR-024).

    Mirrors ``steering_hazards.Hazard.to_dict()`` — evidence carries its own
    ladder label (``validated:ES=…`` vs ``heuristic:weight_prior=…``); the copy
    never claims causality for a heuristic pair (IDL-35).
    """

    type: Literal["compounding", "cancellation"]
    up: Dict[str, int] = Field(..., description="{layer, feature_idx}")
    down: Dict[str, int] = Field(..., description="{layer, feature_idx}")
    evidence: str
    rung: int = Field(0, description="edge rung (0 for a pure heuristic pair)")
    quantified_effect: Optional[float] = Field(None, description="measured ES for validated edges")
    inherited_from_cluster_edge: bool = Field(
        False,
        description=(
            "the effect size was measured on a CLUSTER-level edge and inherited "
            "by this feature pair, not measured on this pair. A supernode's "
            "activation is the max over its members (Appendix A.4), so the "
            "number belongs to the cluster pair. To get a measured per-pair "
            "figure, run the A.4 refinement restricted to the two memberships."
        ),
    )


class MultiLayerAllocationResponse(BaseModel):
    """Per-layer allocation for a cluster that spans multiple layers (015).

    Only emitted when >1 distinct layer is present; a single-layer cluster
    returns the 013 ``ClusterAllocationResponse`` shape byte-identically. Each
    partition runs the UNCHANGED IDL-29 pipeline against its own layer's SAE
    decoder — the formula is not forked.
    """

    formula_id: str = Field(..., description="freq-budget/sim-alloc/per-layer@1")
    layers: Dict[str, PerLayerAllocation] = Field(
        ..., description="layer (as string key) -> that layer's allocation")
    hazards: List[HazardModel] = Field(
        default_factory=list, description="cross-layer compounding/cancellation warnings")
    unchecked_edges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "circuit edges that could NOT be checked for hazards, each with a "
            "reason — a cluster-level edge whose profile is missing or empty. "
            "An empty `hazards` list means 'none found' only when this is also "
            "empty; otherwise it means 'none found among the edges we could "
            "read'."
        ),
    )
    strengths: List[float] = Field(
        ..., description="per-member strengths flattened in request-member order (client convenience)")


# The public allocation response is a UNION with the single-layer 013 shape
# FIRST, so existing clients deserialize a single-layer response identically to
# before Feature 015 (the multi-layer form is only chosen when layers differ).
AllocationResponseUnion = Union[ClusterAllocationResponse, MultiLayerAllocationResponse]
