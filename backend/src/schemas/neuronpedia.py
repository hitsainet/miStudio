"""
Pydantic schemas for Neuronpedia Export API endpoints.

These schemas define the structure for request/response validation
and serialization for Neuronpedia export operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal, Union

from pydantic import BaseModel, Field


# ============================================================================
# Export Configuration Schemas
# ============================================================================

class NeuronpediaExportConfigRequest(BaseModel):
    """Request schema for export configuration."""

    # Feature selection
    feature_selection: Literal["all", "extracted", "custom"] = Field(
        default="all",
        description=(
            "Feature selection mode: "
            "'all' = all features in SAE, "
            "'extracted' = only features with extracted activations, "
            "'custom' = specific feature indices"
        )
    )
    feature_indices: Optional[List[int]] = Field(
        default=None,
        description="Specific feature indices to export (required if feature_selection='custom')"
    )

    # Dashboard data options
    include_logit_lens: bool = Field(
        default=True,
        description="Include logit lens data (top promoted/suppressed tokens)"
    )
    include_histograms: bool = Field(
        default=True,
        description="Include activation histograms"
    )
    include_top_tokens: bool = Field(
        default=True,
        description="Include aggregated top-activating tokens"
    )

    # Dashboard data parameters
    logit_lens_k: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of top positive/negative tokens for logit lens"
    )
    histogram_bins: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of histogram bins"
    )
    top_tokens_k: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of top tokens to aggregate"
    )

    # SAE format options
    include_saelens_format: bool = Field(
        default=True,
        description="Include SAELens-compatible format (cfg.json + weights)"
    )

    # Metadata options
    include_explanations: bool = Field(
        default=True,
        description="Include feature explanations/labels"
    )


class NeuronpediaExportRequest(BaseModel):
    """Request schema for starting a Neuronpedia export job."""

    sae_id: str = Field(..., description="ID of the SAE to export")
    config: NeuronpediaExportConfigRequest = Field(
        default_factory=NeuronpediaExportConfigRequest,
        description="Export configuration options"
    )


# ============================================================================
# Export Job Status Schemas
# ============================================================================

class NeuronpediaExportJobStatus(BaseModel):
    """Response schema for export job status."""

    id: str = Field(..., description="Export job ID")
    sae_id: str = Field(..., description="ID of the SAE being exported")
    status: Literal["pending", "computing", "packaging", "completed", "failed", "cancelled"] = Field(
        ...,
        description="Current job status"
    )
    progress: float = Field(..., ge=0, le=100, description="Progress percentage (0-100)")
    current_stage: Optional[str] = Field(None, description="Human-readable current stage")
    feature_count: Optional[int] = Field(None, description="Number of features being exported")

    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")

    # Results (only when completed)
    output_path: Optional[str] = Field(None, description="Path to output archive (when completed)")
    file_size_bytes: Optional[int] = Field(None, description="Archive size in bytes (when completed)")
    download_url: Optional[str] = Field(None, description="Download URL for archive (when completed)")

    # Error info (only when failed)
    error_message: Optional[str] = Field(None, description="Error message (when failed)")

    class Config:
        from_attributes = True


class NeuronpediaExportJobResponse(BaseModel):
    """Response schema for export job creation."""

    job_id: str = Field(..., description="Created export job ID")
    status: str = Field(..., description="Initial job status")
    message: str = Field(..., description="Status message")


class NeuronpediaExportJobListResponse(BaseModel):
    """Response schema for listing export jobs."""

    jobs: List[NeuronpediaExportJobStatus] = Field(..., description="List of export jobs")
    total: int = Field(..., description="Total number of jobs")


# ============================================================================
# Dashboard Data Schemas
# ============================================================================

class LogitLensToken(BaseModel):
    """Schema for a token in logit lens data."""

    token: str = Field(..., description="Token string")
    token_id: Optional[int] = Field(None, description="Token ID in vocabulary")
    logit: float = Field(..., description="Logit value")


class LogitLensDataResponse(BaseModel):
    """Response schema for logit lens data."""

    feature_index: int = Field(..., description="Feature index")
    top_positive: List[LogitLensToken] = Field(..., description="Top tokens with positive logits")
    top_negative: List[LogitLensToken] = Field(..., description="Top tokens with negative logits")


class HistogramDataResponse(BaseModel):
    """Response schema for histogram data."""

    feature_index: int = Field(..., description="Feature index")
    bin_edges: List[float] = Field(..., description="Histogram bin edges")
    counts: List[int] = Field(..., description="Histogram bin counts")
    total_count: int = Field(..., description="Total number of activations")
    nonzero_count: int = Field(..., description="Number of non-zero activations")
    mean: float = Field(..., description="Mean of non-zero activations")
    std: float = Field(..., description="Standard deviation of non-zero activations")
    max_value: float = Field(..., description="Maximum activation value")
    log_scale: bool = Field(..., description="Whether bins are log-scaled")


class TokenAggregationItem(BaseModel):
    """Schema for a token in top tokens aggregation."""

    token: str = Field(..., description="Token string")
    token_id: Optional[int] = Field(None, description="Token ID")
    total_activation: float = Field(..., description="Sum of activations for this token")
    count: int = Field(..., description="Number of times token activated the feature")
    mean_activation: float = Field(..., description="Average activation")
    max_activation: float = Field(..., description="Maximum activation")


class TokenAggregationResponse(BaseModel):
    """Response schema for token aggregation data."""

    feature_index: int = Field(..., description="Feature index")
    top_tokens: List[TokenAggregationItem] = Field(..., description="Top activating tokens")
    total_examples: int = Field(..., description="Number of examples processed")


class FeatureDashboardDataResponse(BaseModel):
    """Response schema for complete feature dashboard data."""

    feature_id: str = Field(..., description="Feature ID")
    feature_index: int = Field(..., description="Feature index")
    logit_lens: Optional[LogitLensDataResponse] = Field(None, description="Logit lens data")
    histogram: Optional[HistogramDataResponse] = Field(None, description="Histogram data")
    top_tokens: Optional[TokenAggregationResponse] = Field(None, description="Top tokens data")
    computed_at: Optional[datetime] = Field(None, description="When data was computed")


# ============================================================================
# Compute Request Schemas
# ============================================================================

class ComputeDashboardDataRequest(BaseModel):
    """Request schema for computing dashboard data on-demand."""

    sae_id: str = Field(..., description="SAE ID")
    feature_indices: Optional[List[int]] = Field(
        None,
        description="Feature indices to compute (None = all features)"
    )
    include_logit_lens: bool = Field(True, description="Compute logit lens data")
    include_histograms: bool = Field(True, description="Compute histograms")
    include_top_tokens: bool = Field(True, description="Compute top tokens")
    force_recompute: bool = Field(
        False,
        description="Force recompute even if cached data exists"
    )


class ComputeDashboardDataResponse(BaseModel):
    """Response schema for dashboard data computation."""

    job_id: Optional[str] = Field(None, description="Background job ID if async")
    features_computed: int = Field(..., description="Number of features computed")
    status: str = Field(..., description="Computation status")
    message: str = Field(..., description="Status message")
