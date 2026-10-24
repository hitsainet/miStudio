"""Spot-check request bodies and the published disagreement row (Feature 038 FR-1, FR-11, FR-11b; D57).

`DisagreementRow` is `mistudio.spot-check-disagreement/v1`: ONE pydantic model that both serialises every
line of `GET /spot-checks/{id}/disagreements.jsonl` and generates `docs/schemas/spot-check-disagreement-v1.json`,
so the file and the export cannot disagree. miDataworks vendors that file byte-identical (D57).
Additive only within v1. No aliases (memory: pydantic-alias-renames-on-serialisation).
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SPOT_CHECK_DISAGREEMENT_FORMAT = "mistudio.spot-check-disagreement/v1"

#: The probe windows miLLM reports (miLLM `probe_scope.WINDOWS`).
Window = Literal["all", "prompt", "response", "last_user"]
#: What the judge was shown relative to the probe's window (D29, D51; 038 TI7).
JudgeSaw = Literal["same_window", "user_and_reply", "newest_turn"]
Stratum = Literal["fired", "near_threshold", "not_fired"]

MAX_SIZE = 500
DEFAULT_SIZE = 100


class SpotCheckStrata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fired: int = Field(ge=0, le=MAX_SIZE)
    near_threshold: int = Field(ge=0, le=MAX_SIZE)
    not_fired: int = Field(ge=0, le=MAX_SIZE)


class SpotCheckCreate(BaseModel):
    """`POST /probe-monitors/spot-checks`. Defaults: 100 turns (D31), the last 7 days, near band 0.1 (D54)."""

    model_config = ConfigDict(extra="forbid")

    behaviour: str = Field(min_length=2, max_length=64)
    prompt_id: str = Field(min_length=1, max_length=36)
    endpoint: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=255)
    #: Total turns; split evenly over the three strata unless `strata` is given. Over 500 is refused
    #: by the service with a sentence (422), not by the framework.
    size: int = Field(default=DEFAULT_SIZE, ge=1)
    strata: Optional[SpotCheckStrata] = None
    window: Window = "all"
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    near_band: float = Field(default=0.1, gt=0.0, le=1.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)
    parse_failure_limit: Optional[float] = Field(default=None, gt=0.0, lt=1.0)


class SpotCheckPin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pinned: bool


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class DisagreementRow(BaseModel):
    """`mistudio.spot-check-disagreement/v1` — one turn on which a probe and the judge disagreed.

    `messages` is the excerpt the judge was shown, in role/content chat form, so miDataworks reads it as a
    chat upload. Every other field is provenance: where the turn was (miLLM session and request), what the
    probe said (score, threshold, fired) and what the judge said, under which library prompt (key and
    content hash). `source` is always `millm-session`: miDataworks turns it into its session-data caveat.

    ⚠ The judge is a second opinion, not ground truth: a disagreement says one of the two is wrong, not which.
    """

    model_config = ConfigDict(extra="forbid")

    format: Literal["mistudio.spot-check-disagreement/v1"] = SPOT_CHECK_DISAGREEMENT_FORMAT
    messages: List[ChatMessage] = Field(min_length=1)
    behaviour: str
    spot_check_id: str
    session_id: str
    request_id: str
    turn_index: Optional[int] = None
    event_id: Optional[int] = None
    stratum: Stratum
    probe_id: str
    millm_probe_id: Optional[str] = None
    probe_score: Optional[float] = None
    threshold: Optional[float] = None
    probe_fired: bool
    judge_positive: bool
    judge_value: Optional[str] = None
    judge_score: Optional[float] = None
    prompt_key: str
    prompt_hash: str
    window: Window
    judge_saw: JudgeSaw
    source: Literal["millm-session"] = "millm-session"
