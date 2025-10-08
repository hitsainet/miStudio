"""
Pydantic schemas for Dataset API endpoints.

These schemas define the structure for request/response validation
and serialization for all dataset-related API operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, field_serializer

from ..models.dataset import DatasetStatus


class DatasetBase(BaseModel):
    """Base schema with common dataset fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Dataset name")
    source: str = Field(..., min_length=1, max_length=50, description="Source type: HuggingFace, Local, or Custom")


class DatasetCreate(DatasetBase):
    """Schema for creating a new dataset."""

    hf_repo_id: Optional[str] = Field(None, max_length=255, description="HuggingFace repository ID")
    raw_path: Optional[str] = Field(None, max_length=512, description="Path to raw dataset files")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DatasetUpdate(BaseModel):
    """Schema for updating an existing dataset."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[str] = Field(None, pattern="^(downloading|ingesting|ready|error)$")
    progress: Optional[float] = Field(None, ge=0, le=100)
    error_message: Optional[str] = None
    raw_path: Optional[str] = Field(None, max_length=512)
    tokenized_path: Optional[str] = Field(None, max_length=512)
    num_samples: Optional[int] = Field(None, ge=0)
    num_tokens: Optional[int] = Field(None, ge=0)
    avg_seq_length: Optional[float] = Field(None, ge=0)
    vocab_size: Optional[int] = Field(None, ge=0)
    size_bytes: Optional[int] = Field(None, ge=0)
    metadata: Optional[Dict[str, Any]] = None


class DatasetResponse(DatasetBase):
    """Schema for dataset response."""

    id: UUID = Field(..., description="Unique dataset identifier")
    hf_repo_id: Optional[str] = Field(None, description="HuggingFace repository ID")
    status: str = Field(..., description="Current processing status")
    progress: Optional[float] = Field(None, description="Download/processing progress (0-100)")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    raw_path: Optional[str] = Field(None, description="Path to raw dataset files")
    tokenized_path: Optional[str] = Field(None, description="Path to tokenized dataset")
    num_samples: Optional[int] = Field(None, description="Total number of samples")
    num_tokens: Optional[int] = Field(None, description="Total number of tokens")
    avg_seq_length: Optional[float] = Field(None, description="Average sequence length in tokens")
    vocab_size: Optional[int] = Field(None, description="Vocabulary size")
    size_bytes: Optional[int] = Field(None, description="Total size in bytes")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
        alias="extra_metadata"  # Map to SQLAlchemy model's extra_metadata attribute
    )
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")

    @field_serializer('status')
    def serialize_status(self, status: DatasetStatus | str, _info) -> str:
        """Serialize status enum to lowercase value for frontend compatibility."""
        if isinstance(status, DatasetStatus):
            return status.value  # Returns "downloading" for frontend enum
        return str(status).lower() if status else status

    @field_serializer('metadata')
    def serialize_metadata(self, metadata: Any, _info) -> Dict[str, Any]:
        """Serialize metadata, handling SQLAlchemy model attribute mapping."""
        if metadata is None or isinstance(metadata, dict):
            return metadata or {}
        # If it's a SQLAlchemy MetaData object or something unexpected, return empty dict
        return {}

    model_config = {
        "from_attributes": True,  # Enable ORM mode for SQLAlchemy models
        "populate_by_name": True,  # Allow populating by field name or alias
    }


class DatasetListResponse(BaseModel):
    """Schema for paginated list of datasets."""

    data: list[DatasetResponse] = Field(..., description="List of datasets")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


class DatasetDownloadRequest(BaseModel):
    """Schema for HuggingFace dataset download request."""

    repo_id: str = Field(..., min_length=1, description="HuggingFace repository ID (e.g., 'roneneldan/TinyStories')")
    access_token: Optional[str] = Field(None, description="HuggingFace access token for gated datasets")
    split: Optional[str] = Field(None, description="Dataset split to download (e.g., 'train', 'validation', 'test')")

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        """Validate HuggingFace repository ID format."""
        if "/" not in v:
            raise ValueError("repo_id must be in format 'username/dataset-name'")
        return v
