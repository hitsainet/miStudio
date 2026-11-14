"""Stored evidence of the behaviour-detector feature (Feature 040 S2; FTDD §4.4, §4.7; PADR IDL-58 pattern).

* `DetectorSuccessReport` — one immutable success report per identity (040-T17, T29). No foreign key to
  any input (040-T12): a report cites probes, evaluations and views by id, and a purge of an input is
  never blocked by a report that cited it; `inputs_removed` is computed on read.
* `ProbeMonitorDatasetColumnChange` — the history of a view's declared columns when the view names no
  catalogued behaviour (FR-14; a catalogued one writes `probe_behaviour_history`). No foreign key, so a
  purge is not blocked and the history survives the view.

⚠ APPEND-ONLY, ENFORCED TWICE: an ORM `before_update`/`before_delete` refusal, and a trigger installed by
revision ``effadddf2061`` and, for `create_all`, by the ``after_create`` hooks below.
"""
import uuid

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB

from ..core.clock import utc_now
from ..core.database import Base


def _dsr_id() -> str:
    return f"dsr_{uuid.uuid4().hex[:12]}"


class DetectorSuccessReport(Base):
    __tablename__ = "detector_success_reports"

    id = Column(String(36), primary_key=True, default=_dsr_id)
    #: SHA-256 over the JCS bytes of every input that can move a figure (040-T17). UNIQUE: the same
    #: inputs return the same stored report (040-T29).
    identity_sha256 = Column(String(64), nullable=False, unique=True)
    behaviour_key = Column(String(64), nullable=False)
    behaviour_version = Column(Integer, nullable=False)
    model_id = Column(String(255), nullable=False)
    dataset_identity = Column(JSONB, nullable=False)
    report_algorithm = Column(Integer, nullable=False)
    inputs = Column(JSONB, nullable=False)
    #: The canonical JSON bytes (UTF-8 text), byte-identical on every read.
    body = Column(Text, nullable=False)
    body_sha256 = Column(String(64), nullable=False)
    overall_verdict = Column(String(16), nullable=False)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        Index("ix_detector_success_reports_key_created", "behaviour_key", "created_at"),
    )


class ProbeMonitorDatasetColumnChange(Base):
    __tablename__ = "probe_monitor_dataset_column_changes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    view_id = Column(String(36), nullable=False)
    field = Column(String(32), nullable=False)
    old_value = Column(JSONB, nullable=True)
    new_value = Column(JSONB, nullable=True)
    actor = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        Index("ix_probe_monitor_dataset_column_changes_view_created", "view_id", "created_at"),
    )


class DetectorEvidenceIsAppendOnly(RuntimeError):
    """Raised when code tries to change or remove a stored report or column-history row."""


for _model in (DetectorSuccessReport, ProbeMonitorDatasetColumnChange):
    @event.listens_for(_model, "before_update")
    def _refuse_update(mapper, connection, target):  # noqa: ARG001
        raise DetectorEvidenceIsAppendOnly(f"{target.__tablename__} is append-only")

    @event.listens_for(_model, "before_delete")
    def _refuse_delete(mapper, connection, target):  # noqa: ARG001
        raise DetectorEvidenceIsAppendOnly(f"{target.__tablename__} is append-only")


def append_only_ddl(table: str):
    function = f"{table}_append_only"
    return (
        f"""
CREATE OR REPLACE FUNCTION {function}() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '{table} is append-only (% refused)', TG_OP;
END;
$$ LANGUAGE plpgsql
""",
        f"""
CREATE TRIGGER trg_{table}_append_only
BEFORE UPDATE OR DELETE ON {table}
FOR EACH ROW EXECUTE FUNCTION {function}()
""",
    )


REPORTS_DDL = append_only_ddl("detector_success_reports")
COLUMN_CHANGES_DDL = append_only_ddl("probe_monitor_dataset_column_changes")


def _installer(ddl):
    def _install(target, connection, **kw):  # noqa: ARG001
        if connection.dialect.name != "postgresql":
            return
        for statement in ddl:
            connection.execute(text(statement))

    return _install


event.listen(DetectorSuccessReport.__table__, "after_create", _installer(REPORTS_DDL))
event.listen(ProbeMonitorDatasetColumnChange.__table__, "after_create", _installer(COLUMN_CHANGES_DDL))
