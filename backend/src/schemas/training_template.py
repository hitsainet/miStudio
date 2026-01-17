"""
Pydantic schemas for TrainingTemplate API endpoints.

These schemas define the structure for request/response validation
and serialization for all training template-related API operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from .training import TrainingHyperparameters, SAEArchitectureType


class TrainingTemplateBase(BaseModel):
    """Base schema with common training template fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Template name")
    description: Optional[str] = Field(None, description="Template description")


class TrainingTemplateCreate(TrainingTemplateBase):
    """Schema for creating a new training template."""

    model_id: Optional[str] = Field(None, description="Optional reference to specific model")
    dataset_ids: List[str] = Field(default_factory=list, description="Dataset IDs for training (supports multiple)")
    encoder_type: SAEArchitectureType = Field(..., description="SAE architecture type (standard/skip/transcoder)")
    hyperparameters: TrainingHyperparameters = Field(..., description="Complete training hyperparameters")
    is_favorite: bool = Field(False, description="Whether this template is marked as favorite")
    extra_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate model ID format."""
        if v is not None and not v.startswith("m_"):
            raise ValueError("model_id must start with 'm_'")
        return v

    @field_validator("encoder_type")
    @classmethod
    def validate_encoder_type(cls, v: SAEArchitectureType) -> SAEArchitectureType:
        """Ensure encoder_type matches architecture_type in hyperparameters."""
        # This will be cross-validated in the API endpoint
        return v


class TrainingTemplateUpdate(BaseModel):
    """Schema for updating an existing training template."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    model_id: Optional[str] = Field(None, description="Optional reference to specific model")
    dataset_ids: Optional[List[str]] = Field(None, description="Dataset IDs for training (supports multiple)")
    encoder_type: Optional[SAEArchitectureType] = Field(None, description="SAE architecture type")
    hyperparameters: Optional[TrainingHyperparameters] = Field(None, description="Training hyperparameters")
    is_favorite: Optional[bool] = Field(None, description="Favorite status")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate model ID format."""
        if v is not None and not v.startswith("m_"):
            raise ValueError("model_id must start with 'm_'")
        return v


class TrainingTemplateResponse(TrainingTemplateBase):
    """Schema for training template response."""

    id: UUID = Field(..., description="Unique template identifier (UUID)")
    model_id: Optional[str] = Field(None, description="Optional reference to specific model")
    dataset_ids: List[str] = Field(default_factory=list, description="Dataset IDs for training")
    dataset_id: Optional[str] = Field(None, description="Primary dataset ID (backward compat)")
    encoder_type: str = Field(..., description="SAE architecture type (standard/skip/transcoder)")
    hyperparameters: Dict[str, Any] = Field(..., description="Complete training hyperparameters")
    is_favorite: bool = Field(..., description="Whether this template is marked as favorite")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")

    model_config = {
        "from_attributes": True,
    }


class TrainingTemplateListResponse(BaseModel):
    """Schema for paginated list of training templates."""

    data: list[TrainingTemplateResponse] = Field(..., description="List of training templates")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


class TrainingTemplateExport(BaseModel):
    """Schema for exporting training templates to JSON."""

    version: str = Field("1.0", description="Export format version")
    templates: List[TrainingTemplateResponse] = Field(..., description="List of templates to export")
    exported_at: datetime = Field(..., description="Export timestamp")


class TrainingTemplateImport(BaseModel):
    """Schema for importing training templates from JSON."""

    version: str = Field(..., description="Import format version")
    templates: List[TrainingTemplateCreate] = Field(..., min_length=1, description="List of templates to import")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate import version compatibility."""
        if v not in ["1.0"]:
            raise ValueError(f"Unsupported import version: {v}. Supported versions: 1.0")
        return v
