"""Operator code-rule documents and their per-hash approvals (Feature 040 S4; FTDD Appendix C.5, C.6; FR-31).

* `code_rules` — content-addressed ``midataworks.runner/v1`` CodeRuleDocuments: the key is ``code_sha256`` = the
  SHA-256 of the contract's canonical JSON (keys by code point; `rule_runner.contract.canonical_document_bytes`),
  and ``canonical`` holds EXACTLY the hashed bytes, so what the runner runs is what the operator approved.
* `code_rule_approvals` — the durable per-hash list. Current state = the latest row per hash (`approve` /
  `revoke`), read FRESH at every use. ``origin`` is CHECKed equal to ``'operator'``: an agent can file a request
  (`agent_approval_requests`), never write a decision here — the database refuses ``origin='agent'``.

⚠ BOTH ARE APPEND-ONLY, ENFORCED TWICE (the `rule_sets` pattern): the ORM refuses an update or delete, and a trigger
installed by revision ``c0de5a11e004`` (and by the ``after_create`` hooks below for `create_all`) refuses raw SQL.
"""
from sqlalchemy import (
    BigInteger, CheckConstraint, Column, DateTime, ForeignKey, Index, LargeBinary, String, Text, event, text,
)
from sqlalchemy.dialects.postgresql import JSONB

from ..core.clock import utc_now
from ..core.database import Base

SOURCES = ("referenced", "inline")
ACTIONS = ("approve", "revoke")


class CodeRule(Base):
    __tablename__ = "code_rules"

    code_sha256 = Column(String(64), primary_key=True)
    #: Exactly the hashed bytes (canonical JSON, UTF-8).
    canonical = Column(LargeBinary, nullable=False)
    runner_contract = Column(String(64), nullable=False)
    #: The document's declared ``functions`` (``[{name, output}]``).
    functions = Column(JSONB, nullable=False)
    source = Column(String(16), nullable=False)
    #: ``{name, version}`` at miDataworks for a referenced document; NULL inline.
    source_ref = Column(JSONB, nullable=True)
    fetched_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        CheckConstraint("code_sha256 ~ '^[0-9a-f]{64}$'", name="ck_code_rules_sha_hex"),
        CheckConstraint("source IN ('referenced', 'inline')", name="ck_code_rules_source"),
    )


class CodeRuleApproval(Base):
    __tablename__ = "code_rule_approvals"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    code_sha256 = Column(String(64), ForeignKey("code_rules.code_sha256", ondelete="RESTRICT"), nullable=False)
    action = Column(String(16), nullable=False)
    reason = Column(Text, nullable=False)
    changed_by = Column(String(100), nullable=False)
    origin = Column(String(16), nullable=False)
    approval_request_id = Column(String(36), ForeignKey("agent_approval_requests.id", ondelete="RESTRICT"),
                                 nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        CheckConstraint("action IN ('approve', 'revoke')", name="ck_code_rule_approvals_action"),
        CheckConstraint("origin = 'operator'", name="ck_code_rule_approvals_origin_operator"),
        CheckConstraint("length(btrim(reason)) > 0", name="ck_code_rule_approvals_reason"),
        CheckConstraint("length(btrim(changed_by)) > 0", name="ck_code_rule_approvals_changed_by"),
        Index("ix_code_rule_approvals_sha_id", "code_sha256", "id"),
    )


class CodeRuleIsAppendOnly(RuntimeError):
    """Raised when code tries to change or remove a stored code rule or an approval decision."""


for _model in (CodeRule, CodeRuleApproval):
    @event.listens_for(_model, "before_update")
    def _refuse_update(mapper, connection, target):  # noqa: ARG001
        raise CodeRuleIsAppendOnly(f"{target.__tablename__} is append-only")

    @event.listens_for(_model, "before_delete")
    def _refuse_delete(mapper, connection, target):  # noqa: ARG001
        raise CodeRuleIsAppendOnly(f"{target.__tablename__} is append-only")


#: ⚠ The migration carries a FROZEN copy (`test_code_rules_migration.py` pins them equal).
APPEND_ONLY_DDL = (
    """
CREATE OR REPLACE FUNCTION code_rules_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '% is append-only (% refused)', TG_TABLE_NAME, TG_OP;
END;
$$ LANGUAGE plpgsql
""",
    """
CREATE TRIGGER trg_code_rules_append_only
BEFORE UPDATE OR DELETE ON code_rules
FOR EACH ROW EXECUTE FUNCTION code_rules_append_only()
""",
    """
CREATE TRIGGER trg_code_rule_approvals_append_only
BEFORE UPDATE OR DELETE ON code_rule_approvals
FOR EACH ROW EXECUTE FUNCTION code_rules_append_only()
""",
)


def _install_function(target, connection, **kw):  # noqa: ARG001
    if connection.dialect.name == "postgresql":
        connection.execute(text(APPEND_ONLY_DDL[0]))
        connection.execute(text(APPEND_ONLY_DDL[1]))


def _install_approvals(target, connection, **kw):  # noqa: ARG001
    if connection.dialect.name == "postgresql":
        connection.execute(text(APPEND_ONLY_DDL[0]))
        connection.execute(text(APPEND_ONLY_DDL[2]))


event.listen(CodeRule.__table__, "after_create", _install_function)
event.listen(CodeRuleApproval.__table__, "after_create", _install_approvals)
