"""behaviour detector fields: six nullable columns on probe_behaviour_versions; rule_sets

Revision ID: cf0910e18fd8
Revises: 8a9569207fed
Create Date: 2026-10-23

Feature 040 slice S1 (FR-1, FR-45; FTDD §4.2, §4.3; FTID §4.1). Six nullable columns on the
append-only `probe_behaviour_versions` — `sub_questions`, `success_criteria`, `held_out_columns`,
`models` (JSONB), `calibration_floor` (CHECK >= 1) and `rule_set_digest` (FK to `rule_sets.digest`,
ON DELETE RESTRICT) — and the new append-only table `rule_sets` with its trigger.

NO DEFAULT AND NO BACKFILL: NULL is "not declared", so every version written before 040 reads with all
six absent, never as invalid. Adding nullable columns without a default rewrites no row. The versions
table's own append-only trigger is untouched (the columns are added by DDL, which no row trigger sees).

`down_revision` was re-read on 2026-10-23 by walking every revision's `down_revision`: the single head is
039's `8a9569207fed` (FTASKS 0.2).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "cf0910e18fd8"
down_revision = "8a9569207fed"
branch_labels = None
depends_on = None

#: ⚠ A FROZEN COPY of `models.rule_set.RULE_SETS_DDL` (a migration must not import the app, which can
#: change after it); `test_behaviour_detector_migration.py` pins the two equal.
RULE_SETS_DDL = (
    """
CREATE OR REPLACE FUNCTION rule_sets_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'rule_sets is append-only (% refused): a changed set is a new digest', TG_OP;
END;
$$ LANGUAGE plpgsql
""",
    """
CREATE TRIGGER trg_rule_sets_append_only
BEFORE UPDATE OR DELETE ON rule_sets
FOR EACH ROW EXECUTE FUNCTION rule_sets_append_only()
""",
)

VERSION_COLUMNS = (
    "sub_questions", "success_criteria", "held_out_columns", "models", "calibration_floor",
    "rule_set_digest",
)


def upgrade() -> None:
    op.create_table(
        "rule_sets",
        sa.Column("digest", sa.String(length=64), primary_key=True),
        sa.Column("bytes", sa.LargeBinary(), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("behaviour_key", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("source_ref", JSONB(), nullable=True),
        sa.Column("digest_scheme", sa.String(length=32), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("digest ~ '^[0-9a-f]{64}$'", name="ck_rule_sets_digest_hex"),
        sa.CheckConstraint("source IN ('referenced', 'inline')", name="ck_rule_sets_source"),
    )
    for statement in RULE_SETS_DDL:
        op.execute(statement)

    op.add_column("probe_behaviour_versions", sa.Column("sub_questions", JSONB(), nullable=True))
    op.add_column("probe_behaviour_versions", sa.Column("success_criteria", JSONB(), nullable=True))
    op.add_column("probe_behaviour_versions", sa.Column("held_out_columns", JSONB(), nullable=True))
    op.add_column("probe_behaviour_versions", sa.Column("models", JSONB(), nullable=True))
    op.add_column("probe_behaviour_versions", sa.Column("calibration_floor", sa.Integer(), nullable=True))
    op.add_column(
        "probe_behaviour_versions", sa.Column("rule_set_digest", sa.String(length=64), nullable=True)
    )
    op.create_check_constraint(
        "ck_probe_behaviour_versions_calibration_floor",
        "probe_behaviour_versions",
        "calibration_floor IS NULL OR calibration_floor >= 1",
    )
    op.create_foreign_key(
        "probe_behaviour_versions_rule_set_digest_fkey",
        "probe_behaviour_versions", "rule_sets",
        ["rule_set_digest"], ["digest"], ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "probe_behaviour_versions_rule_set_digest_fkey", "probe_behaviour_versions", type_="foreignkey"
    )
    op.drop_constraint(
        "ck_probe_behaviour_versions_calibration_floor", "probe_behaviour_versions", type_="check"
    )
    for column in reversed(VERSION_COLUMNS):
        op.drop_column("probe_behaviour_versions", column)
    op.execute("DROP TRIGGER IF EXISTS trg_rule_sets_append_only ON rule_sets")
    op.execute("DROP FUNCTION IF EXISTS rule_sets_append_only()")
    op.drop_table("rule_sets")
