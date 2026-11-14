"""behaviour catalogue: five tables, three view columns, two indexes on Part A's keys

Revision ID: e7c1a4b9d2f6
Revises: e6a1c9d2b4f8
Create Date: 2026-10-21

Feature 037 (ENH-005 D20–D25, D48; PADR IDL-58). Stacked on the judge-prompt library (Part A) and the live-progress revision e6a1c9d2b4f8.

* ``probe_behaviours`` (head row, keyed by Part A's slug), ``probe_behaviour_versions`` (append-only
  definition text), ``probe_behaviour_probe_roles`` (one primary per behaviour, partial unique index),
  ``probe_behaviour_view_links``, ``probe_behaviour_history`` (append-only, no FK so it survives a delete).
* ``probe_monitor_datasets.behaviour`` / ``positive_meaning`` / ``positive_labels_confirmed`` (FR-15, D48).
* Indexes on ``probe_monitor_runs.behaviour`` and ``probe_monitors.behaviour`` — Part A added both columns
  unindexed. INDEXES ONLY: ⚠ this revision never alters or rewrites Part A's three ``behaviour`` columns
  (D25, adoption is a join); `test_behaviour_migration.py` asserts that over this file's ``op.*`` calls.

⚠ FROZEN COPIES: the slug regex and both trigger bodies are written out here, not imported, because a
migration must replay the schema of ITS revision. `test_behaviour_migration.py` pins each equal to the
model module's constant.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "e7c1a4b9d2f6"
down_revision = "e6a1c9d2b4f8"
branch_labels = None
depends_on = None

SLUG_REGEX = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$"

VERSIONS_DDL = (
    """
CREATE OR REPLACE FUNCTION probe_behaviour_versions_append_only() RETURNS trigger AS $$
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
    """
CREATE TRIGGER trg_probe_behaviour_versions_append_only
BEFORE UPDATE OR DELETE ON probe_behaviour_versions
FOR EACH ROW EXECUTE FUNCTION probe_behaviour_versions_append_only()
""",
)

HISTORY_DDL = (
    """
CREATE OR REPLACE FUNCTION probe_behaviour_history_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'probe_behaviour_history is append-only (% refused)', TG_OP;
END;
$$ LANGUAGE plpgsql
""",
    """
CREATE TRIGGER trg_probe_behaviour_history_append_only
BEFORE UPDATE OR DELETE ON probe_behaviour_history
FOR EACH ROW EXECUTE FUNCTION probe_behaviour_history_append_only()
""",
)


def upgrade() -> None:
    op.create_table(
        "probe_behaviours",
        sa.Column("key", sa.String(length=64), primary_key=True),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_by", sa.String(length=100), nullable=True),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(f"key ~ '{SLUG_REGEX}'", name="ck_probe_behaviours_key_slug"),
    )
    op.create_table(
        "probe_behaviour_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "key",
            sa.String(length=64),
            sa.ForeignKey("probe_behaviours.key", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("positive_meaning", sa.Text(), nullable=False),
        sa.Column("negative_meaning", sa.Text(), nullable=True),
        sa.Column("target_fpr", sa.Float(), nullable=True),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("key", "version", name="uq_probe_behaviour_versions_key_version"),
        sa.CheckConstraint("version >= 1", name="ck_probe_behaviour_versions_version_positive"),
        sa.CheckConstraint(
            "target_fpr IS NULL OR (target_fpr > 0 AND target_fpr < 1)",
            name="ck_probe_behaviour_versions_target_fpr",
        ),
    )
    for statement in VERSIONS_DDL:
        op.execute(statement)

    op.create_table(
        "probe_behaviour_probe_roles",
        sa.Column(
            "probe_id",
            sa.String(length=36),
            sa.ForeignKey("probe_monitors.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "key",
            sa.String(length=64),
            sa.ForeignKey("probe_behaviours.key", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("updated_by", sa.String(length=100), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "role IN ('primary', 'candidate', 'retired')", name="ck_probe_behaviour_probe_roles_role"
        ),
    )
    op.create_index(
        "uq_probe_behaviour_one_primary",
        "probe_behaviour_probe_roles",
        ["key"],
        unique=True,
        postgresql_where=sa.text("role = 'primary'"),
    )
    op.create_index("ix_probe_behaviour_probe_roles_key", "probe_behaviour_probe_roles", ["key"])

    op.create_table(
        "probe_behaviour_view_links",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "key",
            sa.String(length=64),
            sa.ForeignKey("probe_behaviours.key", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "dataset_id",
            sa.String(length=36),
            sa.ForeignKey("probe_monitor_datasets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("use", sa.String(length=16), nullable=False),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("key", "dataset_id", name="uq_probe_behaviour_view_links_key_dataset"),
        sa.CheckConstraint(
            "use IN ('train', 'eval', 'calibration')", name="ck_probe_behaviour_view_links_use"
        ),
    )

    op.create_table(
        "probe_behaviour_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("subject_kind", sa.String(length=16), nullable=False),
        sa.Column("subject_id", sa.String(length=64), nullable=True),
        sa.Column("field", sa.String(length=32), nullable=False),
        sa.Column("old_value", JSONB(), nullable=True),
        sa.Column("new_value", JSONB(), nullable=True),
        sa.Column("actor", sa.String(length=100), nullable=False),
        sa.Column("approval_id", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_probe_behaviour_history_key_created", "probe_behaviour_history", ["key", "created_at"]
    )
    for statement in HISTORY_DDL:
        op.execute(statement)

    op.add_column(
        "probe_monitor_datasets", sa.Column("behaviour", sa.String(length=64), nullable=True)
    )
    op.add_column("probe_monitor_datasets", sa.Column("positive_meaning", sa.Text(), nullable=True))
    op.add_column(
        "probe_monitor_datasets", sa.Column("positive_labels_confirmed", JSONB(), nullable=True)
    )
    op.create_index(
        "ix_probe_monitor_datasets_behaviour", "probe_monitor_datasets", ["behaviour"]
    )
    # Indexes on Part A's columns — the columns themselves are not touched.
    op.create_index("ix_probe_monitor_runs_behaviour", "probe_monitor_runs", ["behaviour"])
    op.create_index("ix_probe_monitors_behaviour", "probe_monitors", ["behaviour"])


def downgrade() -> None:
    op.drop_index("ix_probe_monitors_behaviour", table_name="probe_monitors")
    op.drop_index("ix_probe_monitor_runs_behaviour", table_name="probe_monitor_runs")
    op.drop_index("ix_probe_monitor_datasets_behaviour", table_name="probe_monitor_datasets")
    op.drop_column("probe_monitor_datasets", "positive_labels_confirmed")
    op.drop_column("probe_monitor_datasets", "positive_meaning")
    op.drop_column("probe_monitor_datasets", "behaviour")
    op.drop_index("ix_probe_behaviour_history_key_created", table_name="probe_behaviour_history")
    op.drop_table("probe_behaviour_history")
    op.execute("DROP FUNCTION IF EXISTS probe_behaviour_history_append_only()")
    op.drop_table("probe_behaviour_view_links")
    op.drop_index("ix_probe_behaviour_probe_roles_key", table_name="probe_behaviour_probe_roles")
    op.drop_index("uq_probe_behaviour_one_primary", table_name="probe_behaviour_probe_roles")
    op.drop_table("probe_behaviour_probe_roles")
    op.drop_table("probe_behaviour_versions")
    op.execute("DROP FUNCTION IF EXISTS probe_behaviour_versions_append_only()")
    op.drop_table("probe_behaviours")
