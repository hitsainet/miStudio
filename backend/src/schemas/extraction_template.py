"""
Pydantic schemas for ExtractionTemplate API endpoints.

These schemas define the structure for request/response validation
and serialization for all extraction template-related API operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ExtractionTemplateBase(BaseModel):
    """Base schema with common extraction template fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Template name")
    description: Optional[str] = Field(None, description="Template description")


class ExtractionTemplateCreate(ExtractionTemplateBase):
    """Schema for creating a new extraction template."""

    layer_indices: List[int] = Field(..., min_length=1, description="List of layer indices to extract from (e.g., [0, 5, 11])")
    hook_types: List[str] = Field(..., min_length=1, description="List of hook types (residual, mlp, attention)")
    max_samples: int = Field(..., ge=1, le=100000, description="Maximum number of samples to process")
    batch_size: int = Field(8, ge=1, le=256, description="Batch size for processing")
    top_k_examples: int = Field(10, ge=1, le=100, description="Number of top activating examples to save")
    is_favorite: bool = Field(False, description="Whether this template is marked as favorite")
    extra_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("hook_types")
    @classmethod
    def validate_hook_types(cls, v: List[str]) -> List[str]:
        """Validate hook types."""
        valid_types = {"residual", "mlp", "attention"}
        for hook_type in v:
            if hook_type not in valid_types:
                raise ValueError(f"Invalid hook type: {hook_type}. Must be one of: {valid_types}")
        return v

    @field_validator("layer_indices")
    @classmethod
    def validate_layer_indices(cls, v: List[int]) -> List[int]:
        """Validate layer indices are non-negative."""
        for idx in v:
            if idx < 0:
                raise ValueError(f"Layer indices must be non-negative, got: {idx}")
        return v


class ExtractionTemplateUpdate(BaseModel):
    """Schema for updating an existing extraction template."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    layer_indices: Optional[List[int]] = Field(None, min_length=1, description="List of layer indices")
    hook_types: Optional[List[str]] = Field(None, min_length=1, description="List of hook types")
    max_samples: Optional[int] = Field(None, ge=1, le=100000, description="Maximum samples")
    batch_size: Optional[int] = Field(None, ge=1, le=256, description="Batch size")
    top_k_examples: Optional[int] = Field(None, ge=1, le=100, description="Top K examples")
    is_favorite: Optional[bool] = Field(None, description="Favorite status")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("hook_types")
    @classmethod
    def validate_hook_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate hook types."""
        if v is None:
            return v
        valid_types = {"residual", "mlp", "attention"}
        for hook_type in v:
            if hook_type not in valid_types:
                raise ValueError(f"Invalid hook type: {hook_type}. Must be one of: {valid_types}")
        return v

    @field_validator("layer_indices")
    @classmethod
    def validate_layer_indices(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """Validate layer indices are non-negative."""
        if v is None:
            return v
        for idx in v:
            if idx < 0:
                raise ValueError(f"Layer indices must be non-negative, got: {idx}")
        return v


class ExtractionTemplateResponse(ExtractionTemplateBase):
    """Schema for extraction template response."""

    id: UUID = Field(..., description="Unique template identifier (UUID)")
    layer_indices: List[int] = Field(..., description="List of layer indices to extract from")
    hook_types: List[str] = Field(..., description="List of hook types (residual, mlp, attention)")
    max_samples: int = Field(..., description="Maximum number of samples to process")
    batch_size: int = Field(..., description="Batch size for processing")
    top_k_examples: int = Field(..., description="Number of top activating examples to save")
    is_favorite: bool = Field(..., description="Whether this template is marked as favorite")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")

    model_config = {
        "from_attributes": True,
    }


class ExtractionTemplateListResponse(BaseModel):
    """Schema for paginated list of extraction templates."""

    data: list[ExtractionTemplateResponse] = Field(..., description="List of extraction templates")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


class ExtractionTemplateExport(BaseModel):
    """Schema for exporting extraction templates to JSON."""

    version: str = Field("1.0", description="Export format version")
    templates: List[ExtractionTemplateResponse] = Field(..., description="List of templates to export")
    exported_at: datetime = Field(..., description="Export timestamp")


class ExtractionTemplateImport(BaseModel):
    """Schema for importing extraction templates from JSON."""

    version: str = Field(..., description="Import format version")
    templates: List[ExtractionTemplateCreate] = Field(..., min_length=1, description="List of templates to import")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate import version compatibility."""
        if v not in ["1.0"]:
            raise ValueError(f"Unsupported import version: {v}. Supported versions: 1.0")
        return v
