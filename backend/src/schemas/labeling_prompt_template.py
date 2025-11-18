"""
Pydantic schemas for labeling prompt templates.

This module defines request and response schemas for creating, updating,
and managing customizable prompt templates for semantic feature labeling.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class LabelingPromptTemplateCreate(BaseModel):
    """
    Request schema for creating a labeling prompt template.

    Validates all required fields and ensures proper parameter ranges
    before creating a new template in the database.
    """
    model_config = ConfigDict(from_attributes=True)

    name: str = Field(
        description="Human-readable name for the template",
        min_length=1,
        max_length=255
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed description of what this template is for and when to use it"
    )

    # Prompt content
    system_message: str = Field(
        description="System message that sets the context and role for the LLM",
        min_length=1
    )
    user_prompt_template: str = Field(
        description="User prompt template with {tokens_table} placeholder for token data",
        min_length=1
    )

    # API parameters
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0). Lower = more deterministic, higher = more creative"
    )
    max_tokens: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="Maximum tokens in response (10-1000)"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (0.0-1.0). Controls diversity of token selection"
    )

    # Metadata
    is_default: bool = Field(
        default=False,
        description="Whether this template should be the default selection"
    )

    @field_validator('user_prompt_template')
    @classmethod
    def validate_tokens_placeholder(cls, v: str) -> str:
        """Validate that user prompt template contains the required {tokens_table} placeholder."""
        if '{tokens_table}' not in v:
            raise ValueError(
                "user_prompt_template must contain the {tokens_table} placeholder "
                "for dynamic token data insertion"
            )
        return v


class LabelingPromptTemplateUpdate(BaseModel):
    """
    Request schema for updating a labeling prompt template.

    All fields are optional to allow partial updates. Validates parameter
    ranges when provided.
    """
    model_config = ConfigDict(from_attributes=True)

    name: Optional[str] = Field(
        default=None,
        description="Updated name for the template",
        min_length=1,
        max_length=255
    )
    description: Optional[str] = Field(
        default=None,
        description="Updated description"
    )

    # Prompt content
    system_message: Optional[str] = Field(
        default=None,
        description="Updated system message",
        min_length=1
    )
    user_prompt_template: Optional[str] = Field(
        default=None,
        description="Updated user prompt template",
        min_length=1
    )

    # API parameters
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Updated temperature (0.0-2.0)"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=10,
        le=1000,
        description="Updated max tokens (10-1000)"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Updated top_p (0.0-1.0)"
    )

    # Metadata
    is_default: Optional[bool] = Field(
        default=None,
        description="Whether this template should be the default"
    )

    @field_validator('user_prompt_template')
    @classmethod
    def validate_tokens_placeholder(cls, v: Optional[str]) -> Optional[str]:
        """Validate that user prompt template contains the required {tokens_table} placeholder."""
        if v is not None and '{tokens_table}' not in v:
            raise ValueError(
                "user_prompt_template must contain the {tokens_table} placeholder "
                "for dynamic token data insertion"
            )
        return v


class LabelingPromptTemplateResponse(BaseModel):
    """
    Response schema for a single labeling prompt template.

    Returns complete template information including all fields
    and metadata.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str] = None

    # Prompt content
    system_message: str
    user_prompt_template: str

    # API parameters
    temperature: float
    max_tokens: int
    top_p: float

    # Metadata
    is_default: bool
    is_system: bool
    created_by: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: datetime

    # Usage statistics (from related labeling jobs)
    usage_count: Optional[int] = Field(
        default=None,
        description="Number of labeling jobs using this template"
    )


class LabelingPromptTemplateListResponse(BaseModel):
    """
    Response schema for a paginated list of labeling prompt templates.

    Returns templates with pagination metadata for efficient browsing.
    """
    model_config = ConfigDict(from_attributes=True)

    data: List[LabelingPromptTemplateResponse]
    meta: Dict[str, Any] = Field(
        description="Metadata including total count, limit, offset, page info"
    )


class LabelingPromptTemplateDeleteResponse(BaseModel):
    """
    Response schema for template deletion.

    Returns confirmation of deletion or error if template is in use.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    message: str
    success: bool


class LabelingPromptTemplateSetDefaultResponse(BaseModel):
    """
    Response schema for setting a template as default.

    Returns confirmation that the template is now the default selection.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    message: str
    success: bool


class LabelingPromptTemplateExportItem(BaseModel):
    """
    Schema for a single template in export format.

    Excludes system-specific fields like id, timestamps, created_by.
    """
    model_config = ConfigDict(from_attributes=True)

    name: str
    description: Optional[str] = None
    system_message: str
    user_prompt_template: str
    temperature: float
    max_tokens: int
    top_p: float
    is_default: bool


class LabelingPromptTemplateExport(BaseModel):
    """
    Response schema for exporting labeling prompt templates.

    Returns a list of templates in a portable format suitable for sharing
    and importing into other instances.
    """
    model_config = ConfigDict(from_attributes=True)

    version: str = Field(
        default="1.0",
        description="Export format version"
    )
    exported_at: datetime = Field(
        description="Timestamp when the export was created"
    )
    templates: List[LabelingPromptTemplateExportItem] = Field(
        description="List of templates in portable format"
    )


class LabelingPromptTemplateImport(BaseModel):
    """
    Request schema for importing labeling prompt templates.

    Accepts export data and optional overwrite flag.
    """
    model_config = ConfigDict(from_attributes=True)

    version: str = Field(
        description="Export format version (must match current version)"
    )
    exported_at: datetime = Field(
        description="Original export timestamp"
    )
    templates: List[LabelingPromptTemplateExportItem] = Field(
        description="List of templates to import"
    )
    overwrite_duplicates: bool = Field(
        default=False,
        description="Whether to overwrite templates with the same name"
    )


class LabelingPromptTemplateImportResult(BaseModel):
    """
    Response schema for import operation.

    Returns statistics about the import operation.
    """
    model_config = ConfigDict(from_attributes=True)

    success: bool
    message: str
    imported_count: int = Field(
        description="Number of templates successfully imported"
    )
    skipped_count: int = Field(
        description="Number of templates skipped (duplicates)"
    )
    overwritten_count: int = Field(
        description="Number of templates overwritten"
    )
    failed_count: int = Field(
        description="Number of templates that failed to import"
    )
    details: List[str] = Field(
        default_factory=list,
        description="Detailed messages about each template"
    )
