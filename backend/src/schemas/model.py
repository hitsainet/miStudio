"""
Pydantic schemas for Model API endpoints.

These schemas define the structure for request/response validation
and serialization for all model-related API operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator

from ..models.model import ModelStatus, QuantizationFormat


class ModelBase(BaseModel):
    """Base schema with common model fields."""

    name: str = Field(..., min_length=1, max_length=500, description="Model name")


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

    id: str = Field(..., description="Unique model identifier (format: m_{uuid})")
    repo_id: Optional[str] = Field(None, description="HuggingFace repository ID")
    architecture: str = Field(..., description="Model architecture")
    params_count: int = Field(..., description="Number of parameters")
    quantization: QuantizationFormat = Field(..., description="Quantization format")
    status: ModelStatus = Field(..., description="Current processing status")
    progress: Optional[float] = Field(None, description="Download/loading progress (0-100)")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    file_path: Optional[str] = Field(None, description="Path to raw model files")
    quantized_path: Optional[str] = Field(None, description="Path to quantized model files")
    architecture_config: Optional[Dict[str, Any]] = Field(None, description="Architecture configuration")
    memory_required_bytes: Optional[int] = Field(None, description="Estimated memory requirement")
    disk_size_bytes: Optional[int] = Field(None, description="Disk size in bytes")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")

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
    quantization: QuantizationFormat = Field(QuantizationFormat.FP16, description="Quantization format")
    access_token: Optional[str] = Field(None, description="HuggingFace access token for gated models")

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        """Validate HuggingFace repository ID format."""
        if "/" not in v:
            raise ValueError("repo_id must be in format 'username/model-name'")
        return v
