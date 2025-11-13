"""
Pydantic schemas for feature labeling.

This module defines request and response schemas for semantic labeling
of features extracted from SAE models.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, field_validator


class LabelingConfigRequest(BaseModel):
    """
    Request schema for feature labeling configuration.

    Validates labeling parameters before starting a feature labeling job.
    """
    model_config = ConfigDict(from_attributes=True)

    extraction_job_id: str = Field(
        description="ID of the extraction job whose features should be labeled"
    )

    labeling_method: str = Field(
        description="Feature labeling method: 'openai' (OpenAI API, fast), 'local' (local LLM, slower), 'manual' (user-provided labels)"
    )

    # OpenAI configuration
    openai_model: Optional[str] = Field(
        default="gpt-4o-mini",
        description="OpenAI model for labeling when labeling_method='openai'. Options: 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for labeling when labeling_method='openai'. If not provided, uses environment variable OPENAI_API_KEY"
    )

    # OpenAI-compatible configuration
    openai_compatible_endpoint: Optional[str] = Field(
        default=None,
        description="OpenAI-compatible endpoint URL for labeling when labeling_method='openai_compatible'. Must include /v1 suffix. Example: 'http://ollama.mcslab.io/v1'"
    )
    openai_compatible_model: Optional[str] = Field(
        default=None,
        description="Model name for OpenAI-compatible endpoint when labeling_method='openai_compatible'. Example: 'llama3.2'"
    )

    # Local model configuration
    local_model: Optional[str] = Field(
        default="meta-llama/Llama-3.2-1B",
        description="Local model to use for labeling when labeling_method='local'. Example: 'meta-llama/Llama-3.2-1B', 'microsoft/Phi-3-mini-4k-instruct'"
    )

    # Optional resource configuration
    batch_size: Optional[int] = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of features to label in parallel (1-100)"
    )

    @field_validator('labeling_method')
    @classmethod
    def validate_labeling_method(cls, v: str) -> str:
        """Validate labeling method is supported."""
        valid_methods = ['openai', 'openai_compatible', 'local', 'manual']
        if v not in valid_methods:
            raise ValueError(f"labeling_method must be one of {valid_methods}")
        return v

    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate OpenAI API key is provided when using OpenAI method."""
        # This validation will be enhanced in the service layer to check env var
        return v


class LabelingStatusResponse(BaseModel):
    """
    Response schema for labeling job status.

    Returns current status and progress of a feature labeling job.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    extraction_job_id: str
    labeling_method: str
    openai_model: Optional[str] = None
    local_model: Optional[str] = None
    status: str
    progress: float
    features_labeled: int
    total_features: Optional[int] = None
    error_message: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class LabelingListResponse(BaseModel):
    """
    Response schema for list of labeling jobs.

    Returns paginated list of labeling jobs with metadata.
    """
    model_config = ConfigDict(from_attributes=True)

    data: List[LabelingStatusResponse]
    meta: Dict[str, Any] = Field(
        description="Metadata including total count, limit, offset"
    )


class LabelingCreateResponse(BaseModel):
    """
    Response schema for labeling job creation.

    Returns the ID of the newly created labeling job.
    """
    model_config = ConfigDict(from_attributes=True)

    labeling_job_id: str
    celery_task_id: str
    status: str
    message: str


class RelabelFeaturesRequest(BaseModel):
    """
    Request schema for re-labeling features.

    Allows re-labeling of features from an existing extraction with new configuration.
    """
    model_config = ConfigDict(from_attributes=True)

    labeling_method: str = Field(
        description="Feature labeling method: 'openai' (OpenAI API), 'local' (local LLM)"
    )
    openai_model: Optional[str] = Field(
        default="gpt-4o-mini",
        description="OpenAI model for labeling"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key. If not provided, uses environment variable"
    )
    local_model: Optional[str] = Field(
        default="meta-llama/Llama-3.2-1B",
        description="Local model to use for labeling"
    )
    batch_size: Optional[int] = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of features to label in parallel (1-100)"
    )

    @field_validator('labeling_method')
    @classmethod
    def validate_labeling_method(cls, v: str) -> str:
        """Validate labeling method is supported."""
        valid_methods = ['openai', 'local']
        if v not in valid_methods:
            raise ValueError(f"labeling_method must be one of {valid_methods}")
        return v
