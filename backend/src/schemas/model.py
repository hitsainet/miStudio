"""
Pydantic schemas for Model API endpoints.

These schemas define the structure for request/response validation
and serialization for all model-related API operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, field_serializer

from ..models.model import ModelStatus


class ModelBase(BaseModel):
    """Base schema with common model fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Model name")
    architecture: str = Field(..., min_length=1, max_length=100, description="Model architecture")


class ModelCreate(ModelBase):
    """Schema for creating a new model."""

    repo_id: Optional[str] = Field(None, max_length=255, description="HuggingFace repository ID")
    quantization: str = Field("FP32", max_length=20, description="Quantization type")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ModelUpdate(BaseModel):
    """Schema for updating an existing model."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[str] = Field(None, pattern="^(downloading|loading|ready|error)$")
    progress: Optional[float] = Field(None, ge=0, le=100)
    error_message: Optional[str] = None
    file_path: Optional[str] = Field(None, max_length=512)
    params_count: Optional[int] = Field(None, ge=0)
    memory_req_bytes: Optional[int] = Field(None, ge=0)
    num_layers: Optional[int] = Field(None, ge=0)
    hidden_dim: Optional[int] = Field(None, ge=0)
    num_heads: Optional[int] = Field(None, ge=0)
    metadata: Optional[Dict[str, Any]] = None


class ModelResponse(ModelBase):
    """Schema for model response."""

    id: UUID = Field(..., description="Unique model identifier")
    repo_id: Optional[str] = Field(None, description="HuggingFace repository ID")
    quantization: str = Field(..., description="Quantization type")
    status: str = Field(..., description="Current processing status")
    progress: Optional[float] = Field(None, description="Download/loading progress (0-100)")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    file_path: Optional[str] = Field(None, description="Path to model files")
    params_count: Optional[int] = Field(None, description="Number of parameters")
    memory_req_bytes: Optional[int] = Field(None, description="Memory requirement in bytes")
    num_layers: Optional[int] = Field(None, description="Number of transformer layers")
    hidden_dim: Optional[int] = Field(None, description="Hidden dimension size")
    num_heads: Optional[int] = Field(None, description="Number of attention heads")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
        alias="extra_metadata"
    )
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")

    @field_serializer('status')
    def serialize_status(self, status: ModelStatus | str, _info) -> str:
        """Serialize status enum to uppercase name for consistency."""
        if isinstance(status, ModelStatus):
            return status.name
        return str(status).upper() if status else status

    @field_serializer('metadata')
    def serialize_metadata(self, metadata: Any, _info) -> Dict[str, Any]:
        """Serialize metadata, handling SQLAlchemy model attribute mapping."""
        if metadata is None or isinstance(metadata, dict):
            return metadata or {}
        return {}

    model_config = {
        "from_attributes": True,
        "populate_by_name": True,
    }


class ModelListResponse(BaseModel):
    """Schema for paginated list of models."""

    data: list[ModelResponse] = Field(..., description="List of models")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


class ModelDownloadRequest(BaseModel):
    """Schema for HuggingFace model download request."""

    repo_id: str = Field(..., min_length=1, description="HuggingFace repository ID (e.g., 'TinyLlama/TinyLlama-1.1B')")
    quantization: str = Field("FP16", description="Quantization format (FP32, FP16, Q8, Q4, Q2)")
    access_token: Optional[str] = Field(None, description="HuggingFace access token for gated models")

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        """Validate HuggingFace repository ID format."""
        if "/" not in v:
            raise ValueError("repo_id must be in format 'username/model-name'")
        return v

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str) -> str:
        """Validate quantization format."""
        valid_formats = ["FP32", "FP16", "Q8", "Q4", "Q2"]
        if v not in valid_formats:
            raise ValueError(f"quantization must be one of {valid_formats}")
        return v
