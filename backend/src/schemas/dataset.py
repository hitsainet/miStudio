"""
Pydantic schemas for Dataset API endpoints.

These schemas define the structure for request/response validation
and serialization for all dataset-related API operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, Union, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, field_serializer, ValidationError

from ..models.dataset import DatasetStatus
from .metadata import DatasetMetadata


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
    status: Optional[str] = Field(None, pattern="^(downloading|ingesting|processing|ready|error)$")
    progress: Optional[float] = Field(None, ge=0, le=100)
    error_message: Optional[str] = None
    raw_path: Optional[str] = Field(None, max_length=512)
    tokenized_path: Optional[str] = Field(None, max_length=512)
    num_samples: Optional[int] = Field(None, ge=0)
    num_tokens: Optional[int] = Field(None, ge=0)
    avg_seq_length: Optional[float] = Field(None, ge=0)
    vocab_size: Optional[int] = Field(None, ge=0)
    size_bytes: Optional[int] = Field(None, ge=0)
    metadata: Optional[Union[DatasetMetadata, Dict[str, Any]]] = Field(
        None,
        description="Dataset metadata (validated structure or raw dict for backwards compatibility)"
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, v):
        """
        Validate metadata structure if provided.

        Attempts to validate against DatasetMetadata schema for structured validation.
        Falls back to raw dict for backwards compatibility with existing data.
        """
        if v is None:
            return None

        # If already a DatasetMetadata instance, return as-is
        if isinstance(v, DatasetMetadata):
            return v

        # If it's a dict, check if it looks like structured metadata
        if isinstance(v, dict):
            # Check if dict has any of the expected metadata top-level keys
            expected_keys = {"schema", "tokenization", "download"}
            has_expected_structure = any(key in v for key in expected_keys)

            if has_expected_structure:
                # Try to validate structured metadata
                try:
                    validated = DatasetMetadata(**v)
                    return validated.model_dump(by_alias=True)  # Return as dict with aliases
                except ValidationError as e:
                    # For backwards compatibility, allow raw dicts
                    # Log validation errors for debugging but don't fail
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Metadata validation failed, storing as raw dict: {e.error_count()} errors. "
                        f"First error: {e.errors()[0] if e.errors() else 'unknown'}"
                    )
                    return v
            else:
                # Unstructured metadata (no expected keys) - keep as raw dict
                return v

        return v


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
        validation_alias="extra_metadata"  # Read from extra_metadata attribute
    )
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")

    @field_serializer('status')
    def serialize_status(self, status: DatasetStatus | str, _info) -> str:
        """Serialize status enum to lowercase value for frontend compatibility."""
        if isinstance(status, DatasetStatus):
            return status.value  # Returns "downloading" for frontend enum
        return str(status).lower() if status else status

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
    config: Optional[str] = Field(None, description="Dataset configuration name (e.g., 'en', 'zh') for datasets with multiple configs")

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        """Validate HuggingFace repository ID format."""
        if "/" not in v:
            raise ValueError("repo_id must be in format 'username/dataset-name'")
        return v


class DatasetTokenizeRequest(BaseModel):
    """Schema for dataset tokenization request."""

    tokenizer_name: str = Field(..., min_length=1, description="HuggingFace tokenizer name (e.g., 'gpt2', 'bert-base-uncased')")
    max_length: int = Field(512, ge=1, le=8192, description="Maximum sequence length in tokens")
    stride: int = Field(0, ge=0, description="Sliding window stride for long sequences (0 = no overlap)")
    padding: Literal["max_length", "longest", "do_not_pad"] = Field(
        "max_length",
        description="Padding strategy: 'max_length' pads to max_length, 'longest' pads to longest in batch, 'do_not_pad' disables padding"
    )
    truncation: Literal["longest_first", "only_first", "only_second", "do_not_truncate"] = Field(
        "longest_first",
        description="Truncation strategy: 'longest_first' truncates longest sequence first, 'only_first' truncates only first sequence, 'only_second' truncates only second sequence, 'do_not_truncate' disables truncation"
    )
    add_special_tokens: bool = Field(
        True,
        description="Add special tokens (BOS, EOS, PAD, etc.) - Recommended for most models"
    )
    return_attention_mask: bool = Field(
        True,
        description="Return attention mask - Set to False to save memory if model doesn't use attention masks"
    )
    enable_cleaning: bool = Field(
        True,
        description="Enable text cleaning (removes HTML tags, control characters, excessive punctuation, normalizes Unicode) - Recommended for better feature quality"
    )

    @field_validator("stride")
    @classmethod
    def validate_stride(cls, v: int, info) -> int:
        """Validate that stride is less than or equal to max_length."""
        max_length = info.data.get("max_length", 512)
        if v > max_length:
            raise ValueError(f"stride ({v}) must be less than or equal to max_length ({max_length})")
        return v


class TokenizePreviewRequest(BaseModel):
    """Schema for tokenization preview request."""

    tokenizer_name: str = Field(..., min_length=1, description="HuggingFace tokenizer name")
    text: str = Field(..., min_length=1, max_length=1000, description="Text to tokenize (max 1000 chars)")
    max_length: int = Field(512, ge=1, le=8192, description="Maximum sequence length")
    padding: Literal["max_length", "longest", "do_not_pad"] = Field("max_length", description="Padding strategy")
    truncation: Literal["longest_first", "only_first", "only_second", "do_not_truncate"] = Field("longest_first", description="Truncation strategy")
    add_special_tokens: bool = Field(True, description="Add special tokens (BOS, EOS, etc.)")
    return_attention_mask: bool = Field(True, description="Return attention mask")


class TokenInfo(BaseModel):
    """Information about a single token."""

    id: int = Field(..., description="Token ID")
    text: str = Field(..., description="Token text")
    type: Literal["special", "regular"] = Field(..., description="Token type")
    position: int = Field(..., ge=0, description="Position in sequence")


class TokenizePreviewResponse(BaseModel):
    """Schema for tokenization preview response."""

    tokens: list[TokenInfo] = Field(..., description="List of tokens with metadata")
    attention_mask: Optional[list[int]] = Field(None, description="Attention mask (if requested)")
    token_count: int = Field(..., ge=0, description="Total number of tokens")
    sequence_length: int = Field(..., ge=0, description="Length of tokenized sequence")
    special_token_count: int = Field(..., ge=0, description="Number of special tokens")
