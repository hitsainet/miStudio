"""Blend reports and the suggested blend (Feature 041 S2; FTDD §4.2, T6 – T10; migration `b7e41c2d9a05`).

* `ProbeBlendReport` (`pbr_…`) — one blend evaluation: its request, the inputs fixed at start
  (`resolved`), its status, and where its canonical `report.json` and per-row `.npy` files live. The
  AUTHORITY is the file; the row indexes it.
* `ProbeBlendReportMember` — which probes a report cites (for staleness and the UI), the ONLY relation a
  report has.
* `ProbeSuggestedBlend` — the current advice per (behaviour, model): a record sliced out of a report.
* `ProbeSuggestedBlendHistory` (`psb_…`) — every marking. APPEND-ONLY, enforced twice: ORM listeners and a
  database trigger (installed by the migration and, for `create_all`, by `after_create`).

⚠ NO FOREIGN KEY TO PROBES, RUNS, VIEWS OR BEHAVIOURS (T6). Deleting a run (which cascades its probes) or
purging a view — often a privacy action — must neither delete evidence nor be blocked by it; staleness is
computed on read instead (`services/probe_blend/service.stale_reasons`).
"""
import uuid

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

from ..core.clock import utc_now
from ..core.database import Base

REPORT_STATUSES = ("queued", "running", "done", "failed", "cancelled")
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})


def _pbr_id() -> str:
    return f"pbr_{uuid.uuid4().hex[:12]}"


def _psb_id() -> str:
    return f"psb_{uuid.uuid4().hex[:12]}"


class ProbeBlendReport(Base):
    __tablename__ = "probe_blend_reports"

    id = Column(String(36), primary_key=True, default=_pbr_id)
    behaviour_key = Column(String(64), nullable=False)
    behaviour_version = Column(Integer, nullable=False)
    model_id = Column(String(255), nullable=False)
    dataset_version = Column(String(255), nullable=True)
    status = Column(String(16), nullable=False, default="queued")
    failure_code = Column(String(48), nullable=True)
    failure_message = Column(Text, nullable=True)
    #: The caller's request as sent (`rerun_blend_report` repeats it).
    request = Column(JSONB, nullable=False, default=dict)
    #: Every input fixed at start; the task reads only this.
    resolved = Column(JSONB, nullable=False, default=dict)
    report_path = Column(String(1000), nullable=True)
    report_sha256 = Column(String(64), nullable=True)
    report_bytes = Column(Integer, nullable=True)
    rows_dir = Column(String(1000), nullable=True)
    rows_pruned_at = Column(DateTime(timezone=True), nullable=True)
    rows_pruned_by = Column(String(100), nullable=True)
    rerun_of = Column(String(36), nullable=True)
    rerun_by = Column(String(36), nullable=True)
    cancel_requested_at = Column(DateTime(timezone=True), nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    celery_task_id = Column(String(155), nullable=True)
    engine_version = Column(String(16), nullable=False, default="1")
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed', 'cancelled')", name="ck_probe_blend_reports_status"
        ),
        Index("ix_probe_blend_reports_list", behaviour_key, model_id, created_at.desc(), id.desc()),
        Index("ix_probe_blend_reports_active", "status", postgresql_where=text("status IN ('queued', 'running')")),
    )


class ProbeBlendReportMember(Base):
    __tablename__ = "probe_blend_report_members"

    report_id = Column(String(36), ForeignKey("probe_blend_reports.id", ondelete="CASCADE"), primary_key=True)
    #: No FK: a deleted probe must not delete the evidence that cited it.
    probe_id = Column(String(36), primary_key=True)

    __table_args__ = (Index("ix_probe_blend_report_members_probe_id", "probe_id"),)


class ProbeSuggestedBlend(Base):
    __tablename__ = "probe_suggested_blends"

    behaviour_key = Column(String(64), primary_key=True)
    model_id = Column(String(255), primary_key=True)
    #: The canonical record bytes (`mistudio.suggested-blend/1`), sliced from the report.
    record = Column(LargeBinary, nullable=False)
    record_sha256 = Column(String(64), nullable=False)
    report_id = Column(String(36), nullable=False)
    candidate = Column(String(64), nullable=False)
    history_id = Column(String(36), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class ProbeSuggestedBlendHistory(Base):
    """Every marking of a suggested blend. APPEND-ONLY. Who marked it, when and why live HERE — never in
    the record bytes, which must be identical across two equal evaluations (FR-15, PT-4)."""

    __tablename__ = "probe_suggested_blend_history"

    id = Column(String(36), primary_key=True, default=_psb_id)
    behaviour_key = Column(String(64), nullable=False)
    model_id = Column(String(255), nullable=False)
    report_id = Column(String(36), nullable=False)
    candidate = Column(String(64), nullable=False)
    record = Column(LargeBinary, nullable=False)
    record_sha256 = Column(String(64), nullable=False)
    actor = Column(String(100), nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        Index("ix_probe_suggested_blend_history_key_model", behaviour_key, model_id, created_at.desc()),
    )


class SuggestedBlendHistoryIsAppendOnly(RuntimeError):
    pass


@event.listens_for(ProbeSuggestedBlendHistory, "before_update")
def _refuse_update(mapper, connection, target):  # noqa: ARG001
    raise SuggestedBlendHistoryIsAppendOnly("probe_suggested_blend_history is append-only")


@event.listens_for(ProbeSuggestedBlendHistory, "before_delete")
def _refuse_delete(mapper, connection, target):  # noqa: ARG001
    raise SuggestedBlendHistoryIsAppendOnly("probe_suggested_blend_history is append-only")


HISTORY_FUNCTION = "refuse_suggested_blend_history_change"
HISTORY_TRIGGER = "trg_probe_suggested_blend_history_append_only"
HISTORY_DDL = (
    f"""
CREATE OR REPLACE FUNCTION {HISTORY_FUNCTION}() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'probe_suggested_blend_history is append-only (% refused)', TG_OP;
END;
$$ LANGUAGE plpgsql
""",
    f"""
CREATE TRIGGER {HISTORY_TRIGGER}
BEFORE UPDATE OR DELETE ON probe_suggested_blend_history
FOR EACH ROW EXECUTE FUNCTION {HISTORY_FUNCTION}()
""",
)


def _install(target, connection, **kw):  # noqa: ARG001
    if connection.dialect.name != "postgresql":
        return
    for statement in HISTORY_DDL:
        connection.execute(text(statement))


event.listen(ProbeSuggestedBlendHistory.__table__, "after_create", _install)
