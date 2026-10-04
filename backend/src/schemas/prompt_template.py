"""
Pydantic schemas for PromptTemplate API endpoints.

These schemas define the structure for request/response validation
and serialization for all prompt template-related API operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class PromptTemplateBase(BaseModel):
    """Base schema with common prompt template fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Template name")
    description: Optional[str] = Field(None, description="Template description")


class PromptTemplateCreate(PromptTemplateBase):
    """Schema for creating a new prompt template."""

    prompts: List[str] = Field(
        ...,
        min_length=1,
        description="Array of prompt strings"
    )
    is_favorite: bool = Field(False, description="Whether this template is marked as favorite")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for organization")

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, v: List[str]) -> List[str]:
        """Validate prompts - filter out empty strings and require at least one non-empty prompt."""
        # Filter out empty/whitespace-only prompts
        valid_prompts = [p.strip() for p in v if p.strip()]
        if not valid_prompts:
            raise ValueError("At least one non-empty prompt is required")
        return valid_prompts

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> List[str]:
        """Validate and clean tags."""
        if v is None:
            return []
        # Filter out empty tags and strip whitespace
        return [tag.strip() for tag in v if tag.strip()]


class PromptTemplateUpdate(BaseModel):
    """Schema for updating an existing prompt template."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    prompts: Optional[List[str]] = Field(None, min_length=1, description="Array of prompt strings")
    is_favorite: Optional[bool] = Field(None, description="Favorite status")
    tags: Optional[List[str]] = Field(None, description="Tags for organization")

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate prompts if provided."""
        if v is None:
            return None
        valid_prompts = [p.strip() for p in v if p.strip()]
        if not valid_prompts:
            raise ValueError("At least one non-empty prompt is required")
        return valid_prompts

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and clean tags if provided."""
        if v is None:
            return None
        return [tag.strip() for tag in v if tag.strip()]


class PromptTemplateResponse(PromptTemplateBase):
    """Schema for prompt template response."""

    id: UUID = Field(..., description="Unique template identifier (UUID)")
    prompts: List[str] = Field(..., description="Array of prompt strings")
    is_favorite: bool = Field(..., description="Whether this template is marked as favorite")
    tags: List[str] = Field(default_factory=list, description="Tags for organization")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")

    model_config = {
        "from_attributes": True,
    }


class PromptTemplateListResponse(BaseModel):
    """Schema for paginated list of prompt templates."""

    data: list[PromptTemplateResponse] = Field(..., description="List of prompt templates")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


class PromptTemplateExport(BaseModel):
    """Schema for exporting prompt templates to JSON."""

    version: str = Field("1.0", description="Export format version")
    templates: List[PromptTemplateResponse] = Field(..., description="List of templates to export")
    exported_at: datetime = Field(..., description="Export timestamp")


class PromptTemplateImport(BaseModel):
    """Schema for importing prompt templates from JSON."""

    version: str = Field(..., description="Import format version")
    templates: List[PromptTemplateCreate] = Field(..., min_length=1, description="List of templates to import")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate import version compatibility."""
        if v not in ["1.0"]:
            raise ValueError(f"Unsupported import version: {v}. Supported versions: 1.0")
        return v
