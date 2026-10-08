"""
Pydantic schemas for AppSetting API endpoints.

Defines request/response validation for application settings CRUD operations.
Sensitive values are masked in responses and encrypted at rest.
"""

from datetime import datetime
from typing import Optional, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# Valid setting categories
SettingCategory = Literal["endpoints", "api_keys", "labeling", "display", "general", "system"]


class AppSettingUpsert(BaseModel):
    """Schema for creating or updating a setting (PUT upsert semantics)."""

    key: str = Field(..., min_length=1, max_length=255, description="Unique setting key")
    value: str = Field(..., min_length=1, description="Setting value (will be encrypted if sensitive)")
    is_sensitive: bool = Field(False, description="Whether the value should be encrypted at rest")
    category: SettingCategory = Field("general", description="Setting category for grouping")


class AppSettingResponse(BaseModel):
    """Schema for setting response. Sensitive values are masked."""

    id: UUID
    key: str
    value: str = Field(..., description="Plaintext value, or masked string if sensitive")
    is_sensitive: bool
    category: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PinStatusResponse(BaseModel):
    """Whether a settings PIN is configured and whether the bypass flag is active."""

    configured: bool
    bypass_active: bool


class PinVerifyRequest(BaseModel):
    pin: str = Field(..., min_length=1, max_length=128)


class PinVerifyResponse(BaseModel):
    valid: bool


class PinSetRequest(BaseModel):
    pin: str = Field(..., min_length=4, max_length=128, description="New PIN (min 4 chars)")
    current_pin: str | None = Field(
        None, description="Current PIN — required when changing an existing PIN (waived if bypass active)"
    )


class AppSettingBulkUpsert(BaseModel):
    """Schema for upserting multiple settings at once."""

    settings: list[AppSettingUpsert] = Field(..., min_length=1, description="List of settings to upsert")


class AppSettingBulkResponse(BaseModel):
    """Response for bulk operations."""

    data: list[AppSettingResponse]
    created: int = Field(0, description="Number of new settings created")
    updated: int = Field(0, description="Number of existing settings updated")
