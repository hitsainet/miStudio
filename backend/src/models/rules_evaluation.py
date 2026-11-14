"""Rules ∪ probe evaluations and the rule-verdict cache (Feature 040 S3; FR-33 – FR-38; FTDD §4.7; FTID §4.1).

* `ProbeRulesEvaluation` (`pre_…`) — one request to judge a probe and a behaviour's rule set together on
  one or more views. Identified by `identity_sha256` over every input that can move a figure (040-T17); a
  PARTIAL unique index over the rows still `pending`, `running` or `completed` makes a duplicate request
  return the existing row (040-T29), while a failed or cancelled row never blocks a rerun.
* `RuleVerdictCache` — the index of cached per-view rule verdicts, keyed by (view, rule-set digest, engine
  digest, reading digest) (040-T13 plus the reading — scope and column — the verdicts were computed over).
  The FILES under ``<data_dir>/rule_verdicts/<view_id>/<set>__<engine>__<reading>/`` are the
  authority; the row says they were published. No foreign key: the cache is recomputable, and a purge of
  the view deletes its directory.

⚠ A COMPLETED BODY NEVER CHANGES (FR-38). Status columns stay writable while a row runs; once `body` is set,
an update that changes it is refused twice: an ORM `before_update` listener and a database trigger
(installed by revision ``5ed6d79015d2`` and, for `create_all`, by the ``after_create`` hook below). An
earlier result goes `stale` on read when an input moved; it is never recomputed in place.

⚠ NO ROW TEXT. A body holds counts, rates, digests, ids and positions — never a row's text or a span of it.
"""
import uuid

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Index, Integer, String, Text, event, inspect, text
from sqlalchemy.dialects.postgresql import JSONB

from ..core.clock import utc_now
from ..core.database import Base

STATUSES = ("pending", "running", "completed", "failed", "cancelled")
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
LIVE_STATUSES = ("pending", "running", "completed")


def _pre_id() -> str:
    return f"pre_{uuid.uuid4().hex[:12]}"


class ProbeRulesEvaluation(Base):
    __tablename__ = "probe_rules_evaluations"

    id = Column(String(36), primary_key=True, default=_pre_id)
    identity_sha256 = Column(String(64), nullable=False)
    probe_id = Column(String(36), nullable=False)
    view_ids = Column(JSONB, nullable=False)
    behaviour_key = Column(String(64), nullable=False)
    behaviour_version = Column(Integer, nullable=False)
    #: None when the version names no rule set (`no_rule_set`: union equals probe by construction).
    rule_set_digest = Column(String(64), nullable=True)
    engine_digest = Column(String(64), nullable=False)
    calibration = Column(Boolean, nullable=False, default=True)
    status = Column(String(16), nullable=False, default="pending")
    progress = Column(JSONB, nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    cancel_requested_at = Column(DateTime(timezone=True), nullable=True)
    task_queue_id = Column(String(36), nullable=True)
    error = Column(JSONB, nullable=True)
    #: The canonical JSON bytes (UTF-8 text) of the result, written ONCE.
    body = Column(Text, nullable=True)
    body_sha256 = Column(String(64), nullable=True)
    #: The inputs the identity was computed over (ids, digests, settings) — what staleness compares.
    inputs = Column(JSONB, nullable=False)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
                        name="ck_probe_rules_evaluations_status"),
        Index("uq_probe_rules_evaluations_identity_live", "identity_sha256", unique=True,
              postgresql_where=text("status IN ('pending', 'running', 'completed')")),
        Index("ix_probe_rules_evaluations_key_created", "behaviour_key", "created_at"),
        Index("ix_probe_rules_evaluations_status_heartbeat", "status", "heartbeat_at"),
    )


class RuleVerdictCache(Base):
    __tablename__ = "rule_verdict_cache"

    view_id = Column(String(36), primary_key=True)
    rule_set_digest = Column(String(64), primary_key=True)
    engine_digest = Column(String(64), primary_key=True)
    #: SHA-256 of WHAT the rules read on this view — the scope and the column (a planner addition to
    #: FTDD §4.7's key: two probes of one view scored over different scopes read different text, and a
    #: key without this would hand one the other's verdicts).
    reading_digest = Column(String(64), primary_key=True)
    n_rows = Column(Integer, nullable=False)
    bytes = Column(Integer, nullable=False)
    path = Column(String(1000), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    last_used_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    #: The rules evaluation reading this entry right now (pruning never removes it), or None.
    in_use_by = Column(String(36), nullable=True)

    __table_args__ = (Index("ix_rule_verdict_cache_last_used", "last_used_at"),)


class CompletedResultIsImmutable(RuntimeError):
    """Raised when code tries to change a completed rules evaluation's body."""


@event.listens_for(ProbeRulesEvaluation, "before_update")
def _refuse_body_change(mapper, connection, target):  # noqa: ARG001
    history = inspect(target).attrs.body.history
    if history.deleted and history.deleted[0] is not None and history.added != history.deleted:
        raise CompletedResultIsImmutable(f"rules evaluation {target.id}: a completed body is never rewritten")


#: ⚠ FROZEN COPY in revision `5ed6d79015d2` (pinned equal by `test_rules_evaluation_migration.py`).
BODY_IMMUTABLE_DDL = (
    """
CREATE OR REPLACE FUNCTION probe_rules_evaluations_body_immutable() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.body IS NOT NULL THEN
            RAISE EXCEPTION 'probe_rules_evaluations: a completed result is never deleted';
        END IF;
        RETURN OLD;
    END IF;
    IF OLD.body IS NOT NULL AND NEW.body IS DISTINCT FROM OLD.body THEN
        RAISE EXCEPTION 'probe_rules_evaluations: a completed body is never rewritten';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
""",
    """
CREATE TRIGGER trg_probe_rules_evaluations_body_immutable
BEFORE UPDATE OR DELETE ON probe_rules_evaluations
FOR EACH ROW EXECUTE FUNCTION probe_rules_evaluations_body_immutable()
""",
)


@event.listens_for(ProbeRulesEvaluation.__table__, "after_create")
def _install_trigger(target, connection, **kw):  # noqa: ARG001
    if connection.dialect.name != "postgresql":
        return
    for statement in BODY_IMMUTABLE_DDL:
        connection.execute(text(statement))
