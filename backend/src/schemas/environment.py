"""Request and response models for the runtime environment settings routes (Feature 036).

The response shape is miForge's ``GET /settings`` plus ``label``, ``description``, ``effect``,
``env_value``, ``updated_at``/``updated_by`` and ``last_change`` (ENH-003 §3.1; D13: documented
here, no published JSON schema).
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

REASON_MAX = 500


class EnvironmentWrite(BaseModel):
    """``PUT /environment/{key}``. ``value`` may be JSON ``null`` (clears an optional URL)."""

    value: Any = Field(..., description="The new value, as JSON")
    reason: Optional[str] = Field(None, max_length=REASON_MAX, description="Why (kept in the history)")


class LastChange(BaseModel):
    actor: str
    action: Literal["set", "reset"]
    at: datetime


class EnvironmentRow(BaseModel):
    label: str
    #: The environment variable that sets it (what an operator edits in the manifest).
    env_name: str
    description: str
    group: Literal["runtime", "restart", "environment"]
    #: The operator-facing category (`config.ENVIRONMENT_CATEGORIES` id); the page groups by it.
    category: str
    value: Any = None
    env_value: Any = None
    source: Literal["env", "override", "external"]
    runtime_writable: bool
    effect: Optional[str] = None
    processes: list[str] = Field(default_factory=list)
    restart: Optional[str] = None
    restart_warning: Optional[str] = None
    secret: bool = False
    is_set: Optional[bool] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    reason: Optional[str] = None
    last_change: Optional[LastChange] = None


class EnvironmentCategory(BaseModel):
    id: str
    label: str


class EnvironmentListing(BaseModel):
    settings: dict[str, EnvironmentRow]
    #: Every category, in the order the page shows them.
    categories: list[EnvironmentCategory]
    count: int
    runtime_keys: list[str]
    overrides: dict[str, Any]


class EnvironmentWriteResult(BaseModel):
    key: str
    value: Any = None
    previous: Any = None
    source: Literal["env", "override"]


class EnvironmentChange(BaseModel):
    id: int
    key: str
    action: Literal["set", "reset"]
    previous_value: Any = None
    previous_source: str
    new_value: Any = None
    new_source: str
    actor: str
    reason: Optional[str] = None
    changed_at: datetime


class EnvironmentHistory(BaseModel):
    changes: list[EnvironmentChange]
    count: int
