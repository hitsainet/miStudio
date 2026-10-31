"""The judge-prompt library (operator decisions D1/D2, 2026-10-21; PADR IDL-57).

One row is one VERSION of one judge prompt: ``name`` + ``version`` is its identity and
``prompt_key()`` (``name/vN``) is what every judge run records. An edit to a prompt that has been
used is a new version, never a change to the old one.

⚠ A USED VERSION IS IMMUTABLE, AND THAT IS ENFORCED TWICE. The first judge run that names a
version sets ``locked_at``. From then on every column except the display ones in
`MUTABLE_AFTER_LOCK` is frozen, and the row cannot be deleted:

* the service refuses (409, "make a new version"), and
* a trigger refuses at the database (`IMMUTABLE_DDL`), so raw SQL cannot do it either. It is
  installed by the migration and, through the ``after_create`` hook below, by ``create_all`` —
  the unit suite's schema — so both schemas carry it.

WHY, when PADR IDL-48 pinned measuring prompts as code constants: the judge is the instrument that
rung 3 is measured with, and an editable instrument silently invalidates every prior reading. The
library keeps that guarantee per version — a locked version's ``content_hash`` is fixed, every run
stores the prompt's id, key and hash as columns, and the worker re-derives the hash before it
judges a row (`probe_monitor_judge.execute_judge_run`). What the library adds is that a NEW
instrument can be made without a code change, for a new behaviour.
"""

import uuid

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB

from ..core.clock import utc_now
from ..core.database import Base


def _jp_id() -> str:
    return f"jp_{uuid.uuid4().hex[:12]}"


class ProbeJudgePrompt(Base):
    """One immutable-once-used version of a judge prompt."""

    __tablename__ = "probe_judge_prompts"

    id = Column(String(36), primary_key=True, default=_jp_id)
    #: Lower-case words joined by hyphens (`judge_prompt_forms.SLUG_PATTERN`).
    name = Column(String(64), nullable=False)
    version = Column(Integer, nullable=False)
    #: What a person reads in a list (the behaviour in words). Editable after use.
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    #: The behaviour key this prompt judges (a slug, `<behaviour-key>`). Rung 3 counts a judge run only when
    #: this equals the probe's own key (`ProbeMonitor.behaviour`).
    behaviour = Column(String(64), nullable=False)
    #: The behaviour in words ("Personal data present"). Display only, so editable after use.
    behaviour_label = Column(String(255), nullable=True)

    system_prompt = Column(Text, nullable=False)
    #: The judge's user message, with placeholders from `TEMPLATE_PLACEHOLDERS`.
    user_template = Column(Text, nullable=False)
    #: The answer form, normalised (every default written out): rating, binary or labels.
    answer_form = Column(JSONB, nullable=False)
    #: How much of a row the judge sees; longer rows are cut from the FRONT.
    max_interaction_chars = Column(Integer, nullable=False)
    #: `judge_prompt_forms.content_hash` of the hashed columns.
    content_hash = Column(String(64), nullable=False)

    #: Where it came from: "seed", "operator", "agent:<name>", "midataworks", "import".
    source = Column(String(100), nullable=False)
    #: Provenance detail: the rubric's id/ref/hash for a miDataworks import, the file for an import.
    source_ref = Column(JSONB, nullable=True)
    #: The version or prompt this one was made from (new version or clone).
    parent_id = Column(
        String(36), ForeignKey("probe_judge_prompts.id", ondelete="SET NULL"), nullable=True
    )
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)
    #: Set by the first judge run that names this version. Never cleared.
    locked_at = Column(DateTime(timezone=True), nullable=True)
    #: Hidden from the run form's picker and refused for new runs; every existing run keeps working.
    archived_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_probe_judge_prompts_name_version"),
        CheckConstraint("version >= 1", name="ck_probe_judge_prompts_version_positive"),
        CheckConstraint(
            "max_interaction_chars >= 200 AND max_interaction_chars <= 100000",
            name="ck_probe_judge_prompts_max_chars",
        ),
        Index("ix_probe_judge_prompts_behaviour", "behaviour"),
    )

    def __repr__(self) -> str:
        return f"<ProbeJudgePrompt({self.name}/v{self.version}, {self.id})>"


#: The ONLY columns that may change once a version is locked: what a person reads, and the
#: archive flag. Everything else — including provenance and the behaviour key — is frozen.
MUTABLE_AFTER_LOCK = ("title", "description", "behaviour_label", "archived_at", "updated_at")

#: Derived from the table, so a column added later is FROZEN by default rather than editable by
#: omission (the `labeling_fingerprint` lesson: a hand-kept list drifts toward reporting success).
FROZEN_AFTER_LOCK = tuple(
    column.name
    for column in ProbeJudgePrompt.__table__.columns
    if column.name not in MUTABLE_AFTER_LOCK
)

#: The columns the content hash covers (`judge_prompt_forms.content_hash` arguments).
HASHED_COLUMNS = (
    "name",
    "version",
    "system_prompt",
    "user_template",
    "answer_form",
    "max_interaction_chars",
)

IMMUTABLE_FUNCTION = "probe_judge_prompts_immutable"
IMMUTABLE_TRIGGER = "trg_probe_judge_prompts_immutable"


def immutable_ddl(frozen=FROZEN_AFTER_LOCK) -> tuple:
    """The trigger that refuses changing or deleting a locked version, from the column list."""
    changed = "\n        OR ".join(f"NEW.{name} IS DISTINCT FROM OLD.{name}" for name in frozen)
    function = f"""
CREATE OR REPLACE FUNCTION {IMMUTABLE_FUNCTION}() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.locked_at IS NOT NULL THEN
            RAISE EXCEPTION 'judge prompt %/v% has been used by a judge run and cannot be deleted', OLD.name, OLD.version;
        END IF;
        RETURN OLD;
    END IF;
    IF OLD.locked_at IS NOT NULL AND (
        {changed}
    ) THEN
        RAISE EXCEPTION 'judge prompt %/v% has been used by a judge run and cannot be changed; make a new version', OLD.name, OLD.version;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
"""
    trigger = f"""
CREATE TRIGGER {IMMUTABLE_TRIGGER}
BEFORE UPDATE OR DELETE ON probe_judge_prompts
FOR EACH ROW EXECUTE FUNCTION {IMMUTABLE_FUNCTION}()
"""
    return function, trigger


IMMUTABLE_DDL = immutable_ddl()


def _install_immutable_trigger(target, connection, **kw):  # noqa: ARG001
    if connection.dialect.name != "postgresql":
        return
    from sqlalchemy import text

    for statement in IMMUTABLE_DDL:
        connection.execute(text(statement))


event.listen(ProbeJudgePrompt.__table__, "after_create", _install_immutable_trigger)
