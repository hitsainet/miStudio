"""Request bodies of the blend routes (Feature 041 S2; FTDD §5.3)."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

KEY = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$"


class BlendViews(BaseModel):
    """Named views bypass resolution (FR-30)."""

    model_config = ConfigDict(extra="forbid")

    blend_fit: str
    tests: List[str] = Field(min_length=1)
    calibration: str


class BlendStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    behaviour_key: str = Field(pattern=KEY)
    model: Optional[str] = None
    dataset_version: Optional[str] = Field(default=None, max_length=255)
    member_ids: Optional[List[str]] = Field(default=None, min_length=1)
    views: Optional[BlendViews] = None
    candidates: Optional[List[str]] = Field(default=None, min_length=1)
    folds: Optional[int] = Field(default=None, ge=2, le=100)
    group_column: Optional[str] = Field(default=None, max_length=255)
    target_fpr: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    calibration_sample_rows: Optional[int] = Field(default=None, ge=100, le=200_000)


class SuggestedBlendMark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    report_id: str
    candidate: str
    reason: Optional[str] = Field(default=None, max_length=2000)


class PruneRows(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Optional[str] = Field(default=None, max_length=2000)


# ── S3: family scoring, export and publish ───────────────────────────────────────────────────


class ScoreInput(BaseModel):
    """One input: exactly one of `text` or `messages` (the single score's rule)."""

    model_config = ConfigDict(extra="forbid")

    text: Optional[str] = Field(default=None, max_length=16384)
    messages: Optional[List[Dict[str, str]]] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _exactly_one(self) -> "ScoreInput":
        if (self.text is None) == (self.messages is None):
            raise ValueError("give exactly one of text or messages")
        return self


class ScoreBatch(BaseModel):
    """`POST /probes/{id}/score-batch` (TI14): the cap is the runtime setting, enforced HERE (T19)."""

    model_config = ConfigDict(extra="forbid")

    inputs: List[ScoreInput] = Field(min_length=1)


class FamilyScoreBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    member_ids: Optional[List[str]] = Field(default=None, min_length=1)
    inputs: List[ScoreInput] = Field(min_length=1)
    blend_report_id: Optional[str] = None
    candidates: Optional[List[str]] = Field(default=None, min_length=1)
    include_traces: bool = False


class FamilyPublish(BaseModel):
    """⚠ THE TOKEN IS A REQUEST FIELD AND IS NEVER PERSISTED (as the single publish)."""

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    member_ids: Optional[List[str]] = Field(default=None, min_length=1)
    repo_id: str = Field(min_length=1, max_length=200)
    private: bool = True
    token: Optional[str] = None
