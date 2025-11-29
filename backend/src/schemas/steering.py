"""
Pydantic schemas for Steering API endpoints.

These schemas define the structure for request/response validation
and serialization for feature steering comparison operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Feature Selection Schemas
# ============================================================================

class SelectedFeature(BaseModel):
    """Schema for a feature selected for steering."""

    feature_idx: int = Field(..., ge=0, description="Feature index in the SAE")
    layer: int = Field(..., ge=0, description="Target layer for steering (L0, L1, etc.)")
    strength: float = Field(
        ...,
        ge=-200.0,
        le=200.0,
        description=(
            "Raw steering coefficient (Neuronpedia-compatible). "
            "Values like 0.07 for subtle effects, 80 for strong effects. "
            "Negative values suppress the feature."
        )
    )
    label: Optional[str] = Field(None, description="Feature label for display")
    color: Literal["teal", "blue", "purple", "amber"] = Field(
        "teal",
        description="Color for UI display (teal, blue, purple, amber)"
    )

    @field_validator("strength")
    @classmethod
    def validate_strength(cls, v: float) -> float:
        """Validate steering strength - now Neuronpedia-compatible raw coefficients."""
        # Just validate range - warnings are handled in UI
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

    # Selected features for steering (max 4)
    selected_features: List[SelectedFeature] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="List of features to steer with (1-4)"
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

    @field_validator("selected_features")
    @classmethod
    def validate_selected_features(cls, v: List[SelectedFeature]) -> List[SelectedFeature]:
        """Validate selected features have unique colors."""
        colors = [f.color for f in v]
        if len(colors) != len(set(colors)):
            raise ValueError("Each selected feature must have a unique color")
        return v


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
    coherence: Optional[float] = Field(None, description="Coherence score (0-1)")
    behavioral_score: Optional[float] = Field(None, description="Behavioral score (0-1)")
    token_count: int = Field(..., description="Number of tokens generated")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")


class SteeredOutput(BaseModel):
    """Schema for a single steered generation output."""

    text: str = Field(..., description="Generated text")
    feature_config: SelectedFeature = Field(..., description="Feature configuration used")
    metrics: Optional[GenerationMetrics] = Field(None, description="Generation metrics")


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
