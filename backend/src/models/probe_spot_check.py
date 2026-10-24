"""Session spot-checks (Feature 038; ENH-005 D28–D35, D39, D51–D54; PADR IDL-59).

A spot-check asks miLLM for a seeded, stratified sample of RECORDED session turns where a behaviour's
primary probe ran, sends each turn's text — the window the probe scored — to a library judge prompt, and
reports where probe and judge disagree. One `ProbeSpotCheck` row per check; one `ProbeSpotCheckItem` per
(sampled turn, compared probe).

⚠ THE ITEMS HOLD USER TEXT (D35, D39). `judged_text` is the excerpt actually sent to the judge and
`judge_reply` the judge's answer. They live ONLY in `probe_spot_check_items`, cascade with the run, are
never logged, never put in Celery arguments or results and never sent on a socket, and they are nulled
seven days after the check (`excerpts_expire_at`) unless the run is pinned. Counts, scores and ids stay.
"""

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

from ..core.clock import utc_now
from ..core.database import Base


def _psc_id() -> str:
    return f"psc_{uuid.uuid4().hex[:12]}"


class ProbeSpotCheck(Base):
    __tablename__ = "probe_spot_checks"

    id = Column(String(36), primary_key=True, default=_psc_id)
    #: A spot-check is ABOUT a catalogued behaviour (FR-7): refused unless the prompt judges it.
    behaviour = Column(
        String(64), ForeignKey("probe_behaviours.key", ondelete="RESTRICT"), nullable=False
    )
    behaviour_version = Column(Integer, nullable=False)
    prompt_id = Column(
        String(36), ForeignKey("probe_judge_prompts.id", ondelete="RESTRICT"), nullable=False
    )
    #: Copied from the library row at start (`name/vN`) and its content hash; re-derived before
    #: judging, as Part A's judge runs are (`_recorded_prompt`).
    prompt_key = Column(String(80), nullable=False)
    prompt_hash = Column(String(64), nullable=False)
    endpoint = Column(String(500), nullable=False)
    model = Column(String(255), nullable=False)
    #: {seed, strata: {fired, near_threshold, not_fired}, since, until, window, near_band,
    #:  primary: {mistudio_probe_id, millm_probe_id}, compared: [{mistudio_probe_id, millm_probe_id}]}
    sample_spec = Column(JSONB, nullable=False, default=dict)
    #: What miLLM returned: the turn refs and, per stratum, requested / returned / short.
    sample_returned = Column(JSONB, nullable=True)
    status = Column(String(16), nullable=False, default="pending")
    progress = Column(Float, nullable=True)
    celery_task_id = Column(String(155), nullable=True)
    error_message = Column(Text, nullable=True)
    #: True when the run stopped (cancelled or failed) after judging some turns; the items kept are
    #: reported as a partial run, labelled so.
    partial = Column(Boolean, nullable=False, default=False)
    parse_failures = Column(Integer, nullable=False, default=0)
    parse_failure_limit = Column(Float, nullable=False, default=0.05)
    #: The report (`probe_spot_check.report`): per compared probe, overall and per stratum.
    metrics = Column(JSONB, nullable=True)
    #: D39: a pinned run keeps its excerpts past the seven days.
    pinned = Column(Boolean, nullable=False, default=False)
    excerpts_expire_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String(100), nullable=False, default="operator")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_probe_spot_checks_behaviour_status", "behaviour", "status"),
        Index("ix_probe_spot_checks_excerpts_expire_at", "excerpts_expire_at"),
    )


class ProbeSpotCheckItem(Base):
    __tablename__ = "probe_spot_check_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    spot_check_id = Column(
        String(36), ForeignKey("probe_spot_checks.id", ondelete="CASCADE"), nullable=False
    )
    session_id = Column(String(128), nullable=False)
    request_id = Column(String(128), nullable=False)
    turn_index = Column(Integer, nullable=True)
    event_id = Column(Integer, nullable=True)
    stratum = Column(String(16), nullable=False)
    #: The miStudio probe and the miLLM probe that served it.
    probe_id = Column(String(36), nullable=False)
    millm_probe_id = Column(String(64), nullable=True)
    window = Column(String(16), nullable=False)
    #: same_window | user_and_reply | newest_turn (D51, FR-4).
    judge_saw = Column(String(16), nullable=False)
    probe_score = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    threshold_revision = Column(Integer, nullable=True)
    probe_fired = Column(Boolean, nullable=True)
    #: D52: a sampled turn whose transcript was not recorded is counted, never judged.
    not_recorded = Column(Boolean, nullable=False, default=False)
    #: ⚠ USER TEXT (D35/D39): the excerpt sent, as `[{role, content}]`; nulled on expiry.
    judged_text = Column(JSONB, nullable=True)
    judge_reply = Column(Text, nullable=True)
    judge_value = Column(String(64), nullable=True)
    judge_score = Column(Float, nullable=True)
    judge_positive = Column(Boolean, nullable=True)
    parse_failed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        Index("ix_probe_spot_check_items_check_probe", "spot_check_id", "probe_id"),
        Index(
            "ix_probe_spot_check_items_disagreements",
            "spot_check_id",
            postgresql_where=text(
                "parse_failed = false AND not_recorded = false "
                "AND probe_fired IS DISTINCT FROM judge_positive"
            ),
        ),
    )
