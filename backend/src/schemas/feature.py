"""
Pydantic schemas for feature discovery and analysis.

This module defines request and response schemas for feature search,
filtering, and detailed feature information.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class FeatureSearchRequest(BaseModel):
    """
    Request schema for feature search and filtering.

    Validates search parameters including full-text search, sorting, and pagination.
    """
    model_config = ConfigDict(from_attributes=True)

    search: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Full-text search query on feature name and description"
    )
    sort_by: Literal["activation_freq", "interpretability", "feature_id"] = Field(
        default="activation_freq",
        description="Sort field: activation frequency, interpretability score, or feature ID"
    )
    sort_order: Literal["asc", "desc"] = Field(
        default="desc",
        description="Sort order: ascending or descending"
    )
    is_favorite: Optional[bool] = Field(
        default=None,
        description="Filter by favorite status (None = all, True = favorites only, False = non-favorites)"
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of results to return (1-500)"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip for pagination"
    )

    @field_validator("search")
    @classmethod
    def sanitize_search_query(cls, v: Optional[str]) -> Optional[str]:
        """
        Sanitize search query to prevent SQL injection.

        Removes potentially dangerous characters while preserving
        useful search functionality.
        """
        if v is None:
            return None

        # Strip leading/trailing whitespace
        v = v.strip()

        # Return None if empty after stripping
        if not v:
            return None

        # Remove SQL injection patterns (queries will use parameterized ts_query)
        # This is a defense-in-depth measure; parameterized queries are the primary defense
        dangerous_chars = ["'", '"', ";", "--", "/*", "*/", "\\"]
        for char in dangerous_chars:
            v = v.replace(char, "")

        return v if v else None


class FeatureActivationExample(BaseModel):
    """
    Single max-activating example for a feature.

    Supports both legacy (simple token list) and enhanced (context window) formats.
    Enhanced format includes prefix/prime/suffix breakdown with positions.
    """
    model_config = ConfigDict(from_attributes=True)

    # Legacy format (always present for backward compatibility)
    tokens: List[str]
    activations: List[float]
    max_activation: float
    sample_index: int

    # Enhanced context window format (optional, present in new extractions)
    prefix_tokens: Optional[List[str]] = None
    prime_token: Optional[str] = None
    suffix_tokens: Optional[List[str]] = None
    prime_activation_index: Optional[int] = None
    token_positions: Optional[List[int]] = None


class FeatureResponse(BaseModel):
    """
    Response schema for a single feature.

    Contains all feature metadata, statistics, and user annotations.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    training_id: str
    extraction_job_id: str
    neuron_index: int
    category: Optional[str] = None
    name: str
    description: Optional[str] = None
    label_source: str
    activation_frequency: float
    interpretability_score: float
    max_activation: float
    mean_activation: Optional[float] = None
    is_favorite: bool
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    # Optional: include one example for preview in list view
    example_context: Optional[FeatureActivationExample] = None


class FeatureStatistics(BaseModel):
    """Aggregate statistics for a collection of features."""
    model_config = ConfigDict(from_attributes=True)

    total_features: int
    interpretable_percentage: float
    avg_activation_frequency: float


class FeatureListResponse(BaseModel):
    """
    Response schema for paginated feature list.

    Contains features, pagination metadata, and aggregate statistics.
    """
    model_config = ConfigDict(from_attributes=True)

    features: List[FeatureResponse]
    total: int
    limit: int
    offset: int
    statistics: FeatureStatistics


class FeatureDetailResponse(BaseModel):
    """
    Response schema for detailed feature information.

    Extends FeatureResponse with computed statistics and active sample count.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    training_id: str
    extraction_job_id: str
    neuron_index: int
    category: Optional[str] = None
    name: str
    description: Optional[str] = None
    label_source: str
    activation_frequency: float
    interpretability_score: float
    max_activation: float
    mean_activation: Optional[float] = None
    is_favorite: bool
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    # Computed fields
    active_samples: int = Field(
        description="Number of samples where feature activates (computed from activation_frequency)"
    )


class FeatureUpdateRequest(BaseModel):
    """Request schema for updating feature metadata."""
    model_config = ConfigDict(from_attributes=True)

    name: Optional[str] = Field(default=None, max_length=500)
    description: Optional[str] = Field(default=None)
    notes: Optional[str] = Field(default=None)


class LogitLensResponse(BaseModel):
    """Response schema for logit lens analysis."""
    model_config = ConfigDict(from_attributes=True)

    top_tokens: List[str]
    probabilities: List[float]
    interpretation: str
    computed_at: datetime


class CorrelatedFeature(BaseModel):
    """Single correlated feature."""
    model_config = ConfigDict(from_attributes=True)

    feature_id: str
    feature_name: str
    correlation: float


class CorrelationsResponse(BaseModel):
    """Response schema for feature correlations analysis."""
    model_config = ConfigDict(from_attributes=True)

    correlated_features: List[CorrelatedFeature]
    computed_at: datetime


class AblationResponse(BaseModel):
    """Response schema for ablation analysis."""
    model_config = ConfigDict(from_attributes=True)

    perplexity_delta: float
    impact_score: float
    baseline_perplexity: float
    ablated_perplexity: float
    computed_at: datetime
