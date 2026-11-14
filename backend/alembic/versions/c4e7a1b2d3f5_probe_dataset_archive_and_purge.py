"""probe dataset views: archive columns and the append-only purge tombstone (2026-10-21)

Revision ID: c4e7a1b2d3f5
Revises: b36e0a7c1d42
Create Date: 2026-10-21

Operator decision 2026-10-21 ("Keep the four 20.4 views, but also do the audit plus the
archive/delete controls"), following the pattern miForge and miDataworks shipped 2026-10-20:

* ``probe_monitor_datasets.archived_at / archived_by / archive_reason`` — archive is visibility
  only and reversible; all three NULL means "not archived".
* ``probe_monitor_dataset_purges`` — one tombstone per permanent delete of an ARCHIVED view: who,
  why, and the view's definition as it was. Append-only: a trigger refuses UPDATE and DELETE.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "c4e7a1b2d3f5"
down_revision = "b36e0a7c1d42"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "probe_monitor_datasets",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "probe_monitor_datasets", sa.Column("archived_by", sa.String(length=100), nullable=True)
    )
    op.add_column("probe_monitor_datasets", sa.Column("archive_reason", sa.Text(), nullable=True))

    op.create_table(
        "probe_monitor_dataset_purges",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("view_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("purged_by", sa.String(length=100), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("rows_removed", JSONB, nullable=False),
        sa.Column("removed", JSONB, nullable=False),
        sa.Column("kept", JSONB, nullable=False),
        sa.Column("purged_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_probe_monitor_dataset_purges_view_id", "probe_monitor_dataset_purges", ["view_id"]
    )
    op.create_index(
        "ix_probe_monitor_dataset_purges_purged_at", "probe_monitor_dataset_purges", ["purged_at"]
    )
    op.execute(
        """
CREATE OR REPLACE FUNCTION probe_monitor_dataset_purges_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'probe_monitor_dataset_purges is append-only (% refused)', TG_OP;
END;
$$ LANGUAGE plpgsql
"""
    )
    op.execute(
        """
CREATE TRIGGER trg_probe_monitor_dataset_purges_append_only
BEFORE UPDATE OR DELETE ON probe_monitor_dataset_purges
FOR EACH ROW EXECUTE FUNCTION probe_monitor_dataset_purges_append_only()
"""
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_probe_monitor_dataset_purges_append_only "
        "ON probe_monitor_dataset_purges"
    )
    op.execute("DROP FUNCTION IF EXISTS probe_monitor_dataset_purges_append_only()")
    op.drop_index(
        "ix_probe_monitor_dataset_purges_purged_at", table_name="probe_monitor_dataset_purges"
    )
    op.drop_index(
        "ix_probe_monitor_dataset_purges_view_id", table_name="probe_monitor_dataset_purges"
    )
    op.drop_table("probe_monitor_dataset_purges")
    op.drop_column("probe_monitor_datasets", "archive_reason")
    op.drop_column("probe_monitor_datasets", "archived_by")
    op.drop_column("probe_monitor_datasets", "archived_at")
