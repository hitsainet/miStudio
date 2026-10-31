"""The behaviour catalogue (Feature 037; ENH-005 D20–D25, D48–D50; PADR IDL-58).

A **behaviour** is what a probe detects and what a judge prompt judges, named by a slug
(``<behaviour-key>``). Part A
(the judge-prompt library, IDL-57) already writes that key, as a plain slug, on judge prompts, probe runs
and probes. This module adds the RECORD of each key — what it means, which probe is primary, which data
it was trained on — WITHOUT touching those columns:

⚠ ADOPTION IS A JOIN, NEVER A WRITE (D25). A prompt, run, probe or view belongs to a behaviour exactly
when its ``behaviour`` column equals ``probe_behaviours.key``. There is no foreign key from Part A's
columns to this table, and nothing here updates them: a key without a catalogue row stays valid and is
shown as "uncatalogued", and creating the row later adopts every record already using the key. A
foreign key would have forced every key to be catalogued first, and a locked judge prompt's key cannot
be changed at all (Part A's trigger).

⚠ THE MEANING IS VERSIONED AND THE HISTORY IS APPEND-ONLY, ENFORCED TWICE (D22). A meaning edit is a
new `ProbeBehaviourVersion`; an exported probe carries the version it was exported with, so changing
what a key means after the fact must not rewrite what an old export claimed. The service refuses an
edit of an existing version (409) and a trigger refuses raw SQL too — installed by migration
``e7c1a4b9d2f6`` and, through the ``after_create`` hooks below, by ``create_all`` (the unit schema).

Status (draft / validated / monitoring / unknown) is COMPUTED on read (D49) and deliberately not stored:
a stored copy of a fact derived from other rows could only drift from them. Only "archived" is set by
hand, as ``archived_at``.
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

from ..core.clock import utc_now
from ..core.database import Base

#: The slug rule as a Postgres regex. ⚠ A COPY of `judge_prompt_forms.SLUG_PATTERN` (a model module
#: may not import a service); `test_behaviour_catalogue.py::test_slug_rule_is_the_librarys` pins the
#: two equal, together with the contract's and the migration's copies.
SLUG_REGEX = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$"

#: A probe's place in its behaviour (FR-6). A probe with no row is a candidate (TI3).
PROBE_ROLES = ("primary", "candidate", "retired")
#: What an explicitly linked view is used for (FR-7) — the view's own roles.
VIEW_USES = ("train", "eval", "calibration", "blend_fit")


class ProbeBehaviour(Base):
    """The head row of one catalogued behaviour, keyed by Part A's slug."""

    __tablename__ = "probe_behaviours"

    key = Column(String(64), primary_key=True)
    #: The version the definition text is at now (`ProbeBehaviourVersion.version`).
    current_version = Column(Integer, nullable=False, default=1)
    #: The only hand-set lifecycle state (D49). Archiving keeps every key on every record.
    archived_at = Column(DateTime(timezone=True), nullable=True)
    archived_by = Column(String(100), nullable=True)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    __table_args__ = (
        CheckConstraint(f"key ~ '{SLUG_REGEX}'", name="ck_probe_behaviours_key_slug"),
    )


class ProbeBehaviourVersion(Base):
    """One version of a behaviour's definition text. APPEND-ONLY (D22)."""

    __tablename__ = "probe_behaviour_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(
        String(64), ForeignKey("probe_behaviours.key", ondelete="CASCADE"), nullable=False
    )
    version = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    positive_meaning = Column(Text, nullable=False)
    negative_meaning = Column(Text, nullable=True)
    #: The operating point the operator wants (D21): the false-positive rate a probe is calibrated to.
    target_fpr = Column(Float, nullable=True)
    # ── Feature 040 FR-1: the detector data the definition carries (revision cf0910e18fd8) ──────
    # ⚠ NULL IS "NOT DECLARED", `[]` IS "DECLARED EMPTY". No default and no backfill: a version
    # written before 040 reads with all six absent, never as invalid. Shapes: `schemas/behaviour.py`
    # (SubQuestion, SuccessCriterion); checked by `services/behaviour_detector_definition.validate`.
    sub_questions = Column(JSONB, nullable=True)
    success_criteria = Column(JSONB, nullable=True)
    held_out_columns = Column(JSONB, nullable=True)
    models = Column(JSONB, nullable=True)
    #: Overrides the deployment's calibration floor (runtime key `calibration_floor_negatives`, FR-19).
    calibration_floor = Column(Integer, nullable=True)
    #: The version's rule set: the producer's identity (RFC 8785 JCS SHA-256, spec D38-4) of a row in
    #: the append-only `rule_sets`. RESTRICT: a set a version names is never deleted.
    rule_set_digest = Column(
        String(64), ForeignKey("rule_sets.digest", ondelete="RESTRICT"), nullable=True
    )
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        UniqueConstraint("key", "version", name="uq_probe_behaviour_versions_key_version"),
        CheckConstraint("version >= 1", name="ck_probe_behaviour_versions_version_positive"),
        CheckConstraint(
            "target_fpr IS NULL OR (target_fpr > 0 AND target_fpr < 1)",
            name="ck_probe_behaviour_versions_target_fpr",
        ),
        CheckConstraint(
            "calibration_floor IS NULL OR calibration_floor >= 1",
            name="ck_probe_behaviour_versions_calibration_floor",
        ),
    )


class ProbeBehaviourProbeRole(Base):
    """A probe's role in its behaviour. One row per probe (D24: a probe has at most one behaviour)."""

    __tablename__ = "probe_behaviour_probe_roles"

    probe_id = Column(
        String(36), ForeignKey("probe_monitors.id", ondelete="CASCADE"), primary_key=True
    )
    key = Column(
        String(64), ForeignKey("probe_behaviours.key", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(16), nullable=False, default="candidate")
    updated_by = Column(String(100), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    __table_args__ = (
        CheckConstraint(
            "role IN ('primary', 'candidate', 'retired')", name="ck_probe_behaviour_probe_roles_role"
        ),
        # ⚠ AT MOST ONE PRIMARY PER BEHAVIOUR, AT THE DATABASE (FR-6). The service refuses first with
        # a message naming the current primary; this index is what holds under a race or raw SQL.
        Index(
            "uq_probe_behaviour_one_primary",
            "key",
            unique=True,
            postgresql_where=text("role = 'primary'"),
        ),
        Index("ix_probe_behaviour_probe_roles_key", "key"),
    )


class ProbeBehaviourViewLink(Base):
    """An explicit link from a behaviour to a probe dataset view (FR-7)."""

    __tablename__ = "probe_behaviour_view_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(
        String(64), ForeignKey("probe_behaviours.key", ondelete="CASCADE"), nullable=False
    )
    dataset_id = Column(
        String(36), ForeignKey("probe_monitor_datasets.id", ondelete="CASCADE"), nullable=False
    )
    use = Column(String(16), nullable=False)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        UniqueConstraint("key", "dataset_id", name="uq_probe_behaviour_view_links_key_dataset"),
        CheckConstraint(
            # Feature 040 S2 (FR-13): `blend_fit` — data a blend (041) is fitted on. Widened by
            # revision effadddf2061, whose downgrade refuses while a `blend_fit` link exists.
            "use IN ('train', 'eval', 'calibration', 'blend_fit')", name="ck_probe_behaviour_view_links_use"
        ),
    )


class ProbeBehaviourHistory(Base):
    """Every change to a behaviour's links, roles and lifecycle. APPEND-ONLY (D22, FR-8).

    ``key`` is NOT a foreign key on purpose: the history of a deleted behaviour stays readable.
    """

    __tablename__ = "probe_behaviour_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(64), nullable=False)
    #: behaviour | version | probe | view | dataset
    subject_kind = Column(String(16), nullable=False)
    subject_id = Column(String(64), nullable=True)
    #: created | version | role | behaviour | link | unlink | confirmed | archived | unarchived | deleted
    field = Column(String(32), nullable=False)
    old_value = Column(JSONB, nullable=True)
    new_value = Column(JSONB, nullable=True)
    #: "operator", or "agent:<name>" when the MCP server asked (`X-MiStudio-Agent`).
    actor = Column(String(100), nullable=False)
    approval_id = Column(String(64), nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (Index("ix_probe_behaviour_history_key_created", "key", "created_at"),)


class CatalogueRowIsAppendOnly(RuntimeError):
    """Raised when code tries to change or remove a behaviour version or a history row."""


@event.listens_for(ProbeBehaviourVersion, "before_update")
def _refuse_version_update(mapper, connection, target):  # noqa: ARG001
    raise CatalogueRowIsAppendOnly(
        "probe_behaviour_versions is append-only: make a new version instead of changing one"
    )


@event.listens_for(ProbeBehaviourHistory, "before_update")
def _refuse_history_update(mapper, connection, target):  # noqa: ARG001
    raise CatalogueRowIsAppendOnly("probe_behaviour_history is append-only")


@event.listens_for(ProbeBehaviourHistory, "before_delete")
def _refuse_history_delete(mapper, connection, target):  # noqa: ARG001
    raise CatalogueRowIsAppendOnly("probe_behaviour_history is append-only")


VERSIONS_FUNCTION = "probe_behaviour_versions_append_only"
VERSIONS_TRIGGER = "trg_probe_behaviour_versions_append_only"
#: ⚠ DELETE IS REFUSED ONLY WHILE THE BEHAVIOUR EXISTS (TI9). Deleting an UNUSED behaviour (FR-5)
#: cascades to its versions; by then the parent row is gone, so the cascade is allowed. Every other
#: DELETE, and every UPDATE, is refused.
VERSIONS_DDL = (
    f"""
CREATE OR REPLACE FUNCTION {VERSIONS_FUNCTION}() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF EXISTS (SELECT 1 FROM probe_behaviours WHERE key = OLD.key) THEN
            RAISE EXCEPTION 'probe_behaviour_versions is append-only: version % of % cannot be deleted', OLD.version, OLD.key;
        END IF;
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'probe_behaviour_versions is append-only: version % of % cannot be changed; make a new version', OLD.version, OLD.key;
END;
$$ LANGUAGE plpgsql
""",
    f"""
CREATE TRIGGER {VERSIONS_TRIGGER}
BEFORE UPDATE OR DELETE ON probe_behaviour_versions
FOR EACH ROW EXECUTE FUNCTION {VERSIONS_FUNCTION}()
""",
)

HISTORY_FUNCTION = "probe_behaviour_history_append_only"
HISTORY_TRIGGER = "trg_probe_behaviour_history_append_only"
HISTORY_DDL = (
    f"""
CREATE OR REPLACE FUNCTION {HISTORY_FUNCTION}() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'probe_behaviour_history is append-only (% refused)', TG_OP;
END;
$$ LANGUAGE plpgsql
""",
    f"""
CREATE TRIGGER {HISTORY_TRIGGER}
BEFORE UPDATE OR DELETE ON probe_behaviour_history
FOR EACH ROW EXECUTE FUNCTION {HISTORY_FUNCTION}()
""",
)


def _installer(ddl):
    def _install(target, connection, **kw):  # noqa: ARG001
        if connection.dialect.name != "postgresql":
            return
        for statement in ddl:
            connection.execute(text(statement))

    return _install


event.listen(ProbeBehaviourVersion.__table__, "after_create", _installer(VERSIONS_DDL))
event.listen(ProbeBehaviourHistory.__table__, "after_create", _installer(HISTORY_DDL))
