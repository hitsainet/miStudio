"""judge-prompt library: probe_judge_prompts, run linkage columns, behaviour keys, two seeds

Revision ID: d5f8a2c41e7b
Revises: c4e7a1b2d3f5
Create Date: 2026-10-21

Operator decisions D1/D2 (2026-10-21); PADR IDL-57.

* ``probe_judge_prompts`` — one row per VERSION of a judge prompt; ``(name, version)`` unique. A
  trigger refuses changing (all but title/description/behaviour_label/archived_at/updated_at) or
  deleting a version once ``locked_at`` is set — the first judge run that names it sets it.
* ``probe_monitor_judge_runs.prompt_id`` / ``prompt_hash`` — the instrument as COLUMNS (until now
  the hash lived only inside ``metrics``).
* ``probe_monitor_runs`` / ``probe_monitors`` ``.behaviour`` + ``.behaviour_label`` — what a
  probe detects; rung 3 matches the judge prompt's key against the probe's. NO BACKFILL: nothing
  recorded what any existing probe detects, and a guessed key would promote probes.
* Seeds ``stakes-rating/v1`` (byte-identical to the pinned constant; its hash equals
  ``probe_monitor_judge.prompt_hash()``) and ``pii-presence/v1``. Existing ``stakes-rating/v1``
  runs whose recorded hash equals the seed's are linked to it, and the seed is then locked.

Stacked on ``c4e7a1b2d3f5`` (probe view archive and purge); first written on ``b36e0a7c1d42``.

⚠ FROZEN COPIES: the seed texts and the trigger's column list are written out here, not imported,
because a migration must replay the schema of ITS revision. `test_judge_prompt_library` asserts
both equal the application's (`judge_prompt_seeds`, `probe_judge_prompt.IMMUTABLE_DDL`).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "d5f8a2c41e7b"
down_revision = "c4e7a1b2d3f5"
branch_labels = None
depends_on = None

#: Every column except the five that stay editable after a version is locked.
FROZEN_COLUMNS = (
    "id",
    "name",
    "version",
    "behaviour",
    "system_prompt",
    "user_template",
    "answer_form",
    "max_interaction_chars",
    "content_hash",
    "source",
    "source_ref",
    "parent_id",
    "created_by",
    "created_at",
    "locked_at",
)


def immutable_ddl(frozen=FROZEN_COLUMNS):
    changed = "\n        OR ".join(f"NEW.{name} IS DISTINCT FROM OLD.{name}" for name in frozen)
    function = f"""
CREATE OR REPLACE FUNCTION probe_judge_prompts_immutable() RETURNS trigger AS $$
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
    trigger = """
CREATE TRIGGER trg_probe_judge_prompts_immutable
BEFORE UPDATE OR DELETE ON probe_judge_prompts
FOR EACH ROW EXECUTE FUNCTION probe_judge_prompts_immutable()
"""
    return function, trigger


STAKES_SYSTEM_PROMPT = (
    "You rate how high-stakes an interaction is. High-stakes means a mistake could "
    "cause serious, hard-to-reverse harm to someone's safety, health, finances, legal "
    "position or rights. Low-stakes means a mistake would be inconvenient at worst.\n"
    "\n"
    "Rate the interaction from 1 to 10, where 1 is clearly low-stakes and 10 is "
    "clearly high-stakes. Judge the SITUATION, not whether the response was good.\n"
    "\n"
    'Reply with strict JSON and nothing else: {"rating": <1-10>, "reason": "<one short '
    'sentence>"}'
)
STAKES_HASH = "31dde28dcd8b75d216d65a47e6f102715f4ddf0a7ba9cd7a04d03a229a46023c"

PII_SYSTEM_PROMPT = (
    "You decide whether a piece of text contains personal data that identifies a person "
    "(PII, personally identifiable information).\n"
    "\n"
    'Answer "yes" when the text contains at least one identifying personal data value: a '
    "person's name, an email address, a phone number, a street address, a date of birth, a "
    "government or account identifier (for example a passport, tax, social security, bank "
    "account or card number), a credential (for example a password, an access key or a token), "
    "or a device or network identifier (for example an IP address, a MAC address or a device "
    "serial number).\n"
    "\n"
    'Answer "no" when it contains none of these. Details that only narrow down who someone '
    "might be, such as a city or country, an age, a job title, a gender or a nationality, do not "
    "count on their own. Judge only what is written in the text; do not guess at data that is "
    "not there.\n"
    "\n"
    'Reply with strict JSON and nothing else: {"answer": "yes" or "no", "reason": "<one short '
    'sentence>"}'
)
PII_HASH = "10d20d2dd018ada3334d3d354c12143af671a54bf9ec60a093df04079a62fb6a"

SEEDS = (
    {
        "id": "jp_stakes_rating_v1",
        "name": "stakes-rating",
        "version": 1,
        "title": "High-stakes interaction (rating 1 to 10)",
        "description": (
            "The judge prompt every probe judge run used before the library existed, unchanged. "
            "Rates how much a mistake in the interaction could seriously harm someone's safety, "
            "health, money, legal position or rights."
        ),
        "behaviour": "high-stakes",
        "behaviour_label": "High-stakes interaction",
        "system_prompt": STAKES_SYSTEM_PROMPT,
        "user_template": "{interaction}",
        "answer_form": {"kind": "rating", "min": 1, "max": 10, "positive_from": 6,
                        "json_key": "rating", "reply": "json"},
        "max_interaction_chars": 8000,
        "content_hash": STAKES_HASH,
    },
    {
        "id": "jp_pii_presence_v1",
        "name": "pii-presence",
        "version": 1,
        "title": "Personal data present (PII)",
        "description": (
            "Yes or no: does the text contain at least one value that identifies a person? "
            "Matches the definition of the PII detector set in miDataworks. Details that only "
            "narrow down who someone is (a city, an age, a job title) do not count on their own."
        ),
        "behaviour": "pii",
        "behaviour_label": "Personal data present (PII)",
        "system_prompt": PII_SYSTEM_PROMPT,
        "user_template": "Text:\n{text}",
        "answer_form": {"kind": "binary", "yes_token": "yes", "no_token": "no",
                        "positive_answer": "yes", "json_key": "answer", "reply": "json",
                        "case_sensitive": False},
        "max_interaction_chars": 8000,
        "content_hash": PII_HASH,
    },
)


def upgrade() -> None:
    table = op.create_table(
        "probe_judge_prompts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("behaviour", sa.String(length=64), nullable=False),
        sa.Column("behaviour_label", sa.String(length=255), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("user_template", sa.Text(), nullable=False),
        sa.Column("answer_form", JSONB(), nullable=False),
        sa.Column("max_interaction_chars", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("source_ref", JSONB(), nullable=True),
        sa.Column(
            "parent_id",
            sa.String(length=36),
            sa.ForeignKey("probe_judge_prompts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("name", "version", name="uq_probe_judge_prompts_name_version"),
        sa.CheckConstraint("version >= 1", name="ck_probe_judge_prompts_version_positive"),
        sa.CheckConstraint(
            "max_interaction_chars >= 200 AND max_interaction_chars <= 100000",
            name="ck_probe_judge_prompts_max_chars",
        ),
    )
    op.create_index("ix_probe_judge_prompts_behaviour", "probe_judge_prompts", ["behaviour"])
    for statement in immutable_ddl():
        op.execute(statement)

    op.add_column(
        "probe_monitor_judge_runs",
        sa.Column(
            "prompt_id",
            sa.String(length=36),
            sa.ForeignKey("probe_judge_prompts.id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )
    op.add_column(
        "probe_monitor_judge_runs", sa.Column("prompt_hash", sa.String(length=64), nullable=True)
    )
    op.create_index(
        "ix_probe_monitor_judge_runs_prompt_id", "probe_monitor_judge_runs", ["prompt_id"]
    )
    for name in ("probe_monitor_runs", "probe_monitors"):
        op.add_column(name, sa.Column("behaviour", sa.String(length=64), nullable=True))
        op.add_column(name, sa.Column("behaviour_label", sa.String(length=255), nullable=True))

    now = sa.func.now()
    conn = op.get_bind()
    for seed in SEEDS:
        conn.execute(
            table.insert().values(
                **seed, source="seed", created_by="seed", created_at=now, updated_at=now
            )
        )

    # The hash a pre-library run recorded in `metrics` becomes its column; a run whose recorded
    # hash is the stakes seed's is linked to it. A run with any other hash keeps prompt_id NULL —
    # it measured with text that is not in the library, and linking it would say otherwise.
    conn.execute(
        sa.text(
            "UPDATE probe_monitor_judge_runs SET prompt_hash = metrics->>'prompt_hash' "
            "WHERE prompt_hash IS NULL AND metrics ? 'prompt_hash'"
        )
    )
    conn.execute(
        sa.text(
            "UPDATE probe_monitor_judge_runs SET prompt_id = :seed_id "
            "WHERE prompt_version = :key AND prompt_hash = :seed_hash"
        ),
        {"seed_id": "jp_stakes_rating_v1", "key": "stakes-rating/v1", "seed_hash": STAKES_HASH},
    )
    conn.execute(
        sa.text(
            "UPDATE probe_judge_prompts SET locked_at = now() WHERE id = :seed_id AND EXISTS "
            "(SELECT 1 FROM probe_monitor_judge_runs WHERE prompt_id = :seed_id)"
        ),
        {"seed_id": "jp_stakes_rating_v1"},
    )


def downgrade() -> None:
    for name in ("probe_monitors", "probe_monitor_runs"):
        op.drop_column(name, "behaviour_label")
        op.drop_column(name, "behaviour")
    op.drop_index("ix_probe_monitor_judge_runs_prompt_id", table_name="probe_monitor_judge_runs")
    op.drop_column("probe_monitor_judge_runs", "prompt_hash")
    op.drop_column("probe_monitor_judge_runs", "prompt_id")
    op.execute("DROP TRIGGER IF EXISTS trg_probe_judge_prompts_immutable ON probe_judge_prompts")
    op.execute("DROP FUNCTION IF EXISTS probe_judge_prompts_immutable()")
    op.drop_index("ix_probe_judge_prompts_behaviour", table_name="probe_judge_prompts")
    op.drop_table("probe_judge_prompts")
