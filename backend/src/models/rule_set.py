"""Rule sets a behaviour version names (Feature 040 FR-1, FR-45; FTDD §4.3; PADR IDL-58 pattern).

A rule set is DATA: a `midataworks.rule-set/v1` document, stored exactly once per identity. The key is
the producer's identity — the lower-case hex SHA-256 of the RFC 8785 (JCS) bytes of the document as
received (miDataworks 016, spec D38-4) — so an inline set and the same set referenced from miDataworks
are one row, and a version can never be re-pointed at changed rules under the same digest.

⚠ APPEND-ONLY, ENFORCED TWICE (the IDL-57/IDL-58 pattern of `models/probe_behaviour.py`): the ORM refuses
an update or delete, and a trigger installed by revision ``cf0910e18fd8`` and, for `create_all`, by the
``after_create`` hook below refuses raw SQL too. A version's `rule_set_digest` is a foreign key with
``ON DELETE RESTRICT`` as a second line.

Until the rule engine lands (040 S3) a stored set is reported ``unvalidated: engine_not_available``: it
is kept, never evaluated.
"""
from sqlalchemy import CheckConstraint, Column, DateTime, LargeBinary, String, event, text
from sqlalchemy.dialects.postgresql import JSONB

from ..core.clock import utc_now
from ..core.database import Base

#: `source` values: fetched once from miDataworks at write time, or sent in the request body.
SOURCES = ("referenced", "inline")
#: The one identity scheme (cross-document review 2026-10-23; spec D38-4).
DIGEST_SCHEME = "midataworks_rule_set_identity"


class RuleSet(Base):
    __tablename__ = "rule_sets"

    digest = Column(String(64), primary_key=True)
    #: The document's RFC 8785 canonical bytes — what the digest is the SHA-256 of.
    bytes = Column(LargeBinary, nullable=False)
    #: The document's own `format` (``midataworks.rule-set/v1``).
    schema_version = Column(String(64), nullable=False)
    #: The set's own required `behaviour_key`; equal to the version's key (refused otherwise, 040-P1).
    behaviour_key = Column(String(64), nullable=False)
    source = Column(String(16), nullable=False)
    #: ``{name, version, digest}`` for a referenced set; NULL inline.
    source_ref = Column(JSONB, nullable=True)
    digest_scheme = Column(String(32), nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        CheckConstraint("digest ~ '^[0-9a-f]{64}$'", name="ck_rule_sets_digest_hex"),
        CheckConstraint("source IN ('referenced', 'inline')", name="ck_rule_sets_source"),
    )


class RuleSetIsAppendOnly(RuntimeError):
    """Raised when code tries to change or remove a stored rule set."""


@event.listens_for(RuleSet, "before_update")
def _refuse_update(mapper, connection, target):  # noqa: ARG001
    raise RuleSetIsAppendOnly("rule_sets is append-only: a changed set is a new digest")


@event.listens_for(RuleSet, "before_delete")
def _refuse_delete(mapper, connection, target):  # noqa: ARG001
    raise RuleSetIsAppendOnly("rule_sets is append-only")


RULE_SETS_FUNCTION = "rule_sets_append_only"
RULE_SETS_TRIGGER = "trg_rule_sets_append_only"
RULE_SETS_DDL = (
    f"""
CREATE OR REPLACE FUNCTION {RULE_SETS_FUNCTION}() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'rule_sets is append-only (% refused): a changed set is a new digest', TG_OP;
END;
$$ LANGUAGE plpgsql
""",
    f"""
CREATE TRIGGER {RULE_SETS_TRIGGER}
BEFORE UPDATE OR DELETE ON rule_sets
FOR EACH ROW EXECUTE FUNCTION {RULE_SETS_FUNCTION}()
""",
)


def _install(target, connection, **kw):  # noqa: ARG001
    if connection.dialect.name != "postgresql":
        return
    for statement in RULE_SETS_DDL:
        connection.execute(text(statement))


event.listen(RuleSet.__table__, "after_create", _install)
