"""Shapes of the old-name evidence (042 FR-042.9; FTDD §5.1, §5.2; BR-023; CF-4, CF-24).

`LegacyNamesBlock` is BR-023's block EXACTLY: four keys; `in_use` entries carry exactly `kind, old, new,
last_seen`; pairs exactly `old, new`. `extra="forbid"` everywhere, so a stray field (the withdrawn
per-entry `process`) cannot reach an answer. Serialise with `model_dump(mode="json")` — never
`exclude_none`: `legacy_names_meta.error` is ALWAYS present (`null` when clean; S001 reads a missing key
as an unmeasured day).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

NAME_PATTERN = r"^[A-Za-z0-9_.:=-]{1,255}$"
REPORTER_PATTERN = r"^[a-z]+(:[A-Za-z0-9@._-]{1,56})?$"
MAX_ENTRIES = 64

Kind = Literal["env", "host", "category"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LegacyUse(_Strict):
    kind: Kind
    old: str = Field(pattern=NAME_PATTERN)
    new: str = Field(pattern=NAME_PATTERN)
    last_seen: datetime


class LegacyPair(_Strict):
    old: str = Field(pattern=NAME_PATTERN)
    new: str = Field(pattern=NAME_PATTERN)


class LegacyNamesBlock(_Strict):
    in_use: list[LegacyUse]
    also_set: list[LegacyPair]
    conflicts: list[LegacyPair]
    retired_but_set: list[LegacyPair]


class LegacyNamesWatching(_Strict):
    env: int
    host: int
    category: int


class LegacyNamesMeta(_Strict):
    observing_since: Optional[datetime]
    in_use_window_hours: int
    watching: LegacyNamesWatching
    unobserved_hosts: list[str]
    reporters: dict[str, datetime]
    error: Optional[str] = Field(...)  # required: always serialised, `null` when clean


class LegacyNamesReport(_Strict):
    """`POST /api/v1/system/legacy-names` body: names and times only, at most 64 of each."""

    reporter: str = Field(pattern=REPORTER_PATTERN, max_length=64)
    uses: list[LegacyUse] = Field(default_factory=list, max_length=MAX_ENTRIES)
    also_set: list[LegacyPair] = Field(default_factory=list, max_length=MAX_ENTRIES)
    conflicts: list[LegacyPair] = Field(default_factory=list, max_length=MAX_ENTRIES)
