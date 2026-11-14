"""
Pydantic schemas for cross-feature grouping (Feature 010).
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class GroupingParams(BaseModel):
    """Parameters for a grouping precompute run."""

    context_window: int = Field(default=5, ge=1, le=25, description="Context tokens kept on each side of the prime token")
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Cosine threshold for context subgrouping")
    top_examples: int = Field(default=10, ge=1, le=100, description="Top activating examples read per feature")
    min_group_size: int = Field(default=2, ge=1, le=100, description="Minimum members for a group to be materialized")


class ComputeGroupsRequest(BaseModel):
    params: GroupingParams = Field(default_factory=GroupingParams)
    force: bool = False


class ComputeGroupsResponse(BaseModel):
    task_id: Optional[str] = None
    run_id: str
    status: str
    message: str


class GroupingStatusResponse(BaseModel):
    status: Literal["none", "pending", "computing", "completed", "failed"]
    run_id: Optional[str] = None
    progress: Optional[float] = None
    params: Optional[dict[str, Any]] = None
    feature_count: Optional[int] = None
    group_count: Optional[int] = None
    error_message: Optional[str] = None
    computed_at: Optional[datetime] = None


class FeatureGroupSummary(BaseModel):
    group_id: str
    normalized_token: str
    display_token: str
    member_count: int
    cohesion: float
    sample_labels: list[str] = Field(default_factory=list, description="Up to 3 member labels for scanning")


class FeatureGroupListResponse(BaseModel):
    groups: list[FeatureGroupSummary]
    total: int
    limit: int
    offset: int
    index_status: str


class FeatureGroupMemberOut(BaseModel):
    """Group member with label fields joined live from features."""

    feature_id: str
    neuron_index: int
    name: str
    category: Optional[str] = None
    label_source: Optional[str] = None
    star_color: Optional[str] = None
    is_favorite: bool = False
    max_activation: Optional[float] = None
    activation_frequency: Optional[float] = None
    similarity: float
    context_snippet: Optional[str] = None


class FeatureGroupDetailResponse(BaseModel):
    group_id: str
    normalized_token: str
    display_token: str
    cohesion: float
    member_count: int
    members: list[FeatureGroupMemberOut]


class ByTokenFeatureOut(BaseModel):
    feature_id: str
    neuron_index: int
    name: str
    category: Optional[str] = None
    raw_token: str
    normalized_token: str
    token_rank: int
    weight: float


class ByTokenResponse(BaseModel):
    features: list[ByTokenFeatureOut]
    total: int
    limit: int
    offset: int


class RelatedFeatureOut(BaseModel):
    feature_id: str
    neuron_index: Optional[int] = None
    name: Optional[str] = None
    category: Optional[str] = None
    score: float
    link_types: list[Literal["shared_token", "context", "correlation"]]


class RelatedFeaturesResponse(BaseModel):
    seed_feature_id: str
    related: list[RelatedFeatureOut]
