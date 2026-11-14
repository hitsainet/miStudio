"""detector reports: view columns and snapshots, sub_question/behaviour_version, calibration positions, reports

Revision ID: effadddf2061
Revises: cf0910e18fd8
Create Date: 2026-10-23

Feature 040 slice S2 (FR-4, FR-12 – FR-14, FR-19, FR-20, FR-25; FTDD §4.4 – §4.7; FTID §4.1).

* `probe_monitor_datasets`: `group_columns` (JSONB), `sub_question` (String 64), `behaviour_version`
  (Integer), `column_snapshots` (JSONB);
* `probe_monitor_runs` and `probe_monitors`: `sub_question`, `behaviour_version` (copied from the training
  view at submit and onto each probe);
* `probe_monitors`: `calibration_positions_path` (String 1000), `calibration_n_negatives` (Integer);
* new append-only tables `probe_monitor_dataset_column_changes` and `detector_success_reports` (no
  foreign key to any input, 040-T12), each with its trigger;
* `ck_probe_behaviour_view_links_use` widened to admit `blend_fit`.

NO DEFAULT AND NO BACKFILL: NULL is "not declared / not recorded" (a probe calibrated before S2 has no
recorded positions, which is never read as "aligned").

⚠ THE DOWNGRADE REFUSES WHILE A `blend_fit` LINK EXISTS (TI-21): restoring the three-value CHECK would
otherwise fail mid-migration; it raises naming the count instead.

`down_revision` re-read on 2026-10-23 by walking every revision's `down_revision`: S1's `cf0910e18fd8`.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "effadddf2061"
down_revision = "cf0910e18fd8"
branch_labels = None
depends_on = None


def _append_only(table: str):
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


#: ⚠ FROZEN COPIES of `models.detector_report.REPORTS_DDL` / `COLUMN_CHANGES_DDL`; pinned equal by
#: `test_detector_reports_migration.py`.
REPORTS_DDL = _append_only("detector_success_reports")
COLUMN_CHANGES_DDL = _append_only("probe_monitor_dataset_column_changes")


def upgrade() -> None:
    op.add_column("probe_monitor_datasets", sa.Column("group_columns", JSONB(), nullable=True))
    op.add_column("probe_monitor_datasets", sa.Column("sub_question", sa.String(length=64), nullable=True))
    op.add_column("probe_monitor_datasets", sa.Column("behaviour_version", sa.Integer(), nullable=True))
    op.add_column("probe_monitor_datasets", sa.Column("column_snapshots", JSONB(), nullable=True))
    for table in ("probe_monitor_runs", "probe_monitors"):
        op.add_column(table, sa.Column("sub_question", sa.String(length=64), nullable=True))
        op.add_column(table, sa.Column("behaviour_version", sa.Integer(), nullable=True))
    op.add_column("probe_monitors", sa.Column("calibration_positions_path", sa.String(length=1000), nullable=True))
    op.add_column("probe_monitors", sa.Column("calibration_n_negatives", sa.Integer(), nullable=True))

    op.create_table(
        "probe_monitor_dataset_column_changes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("view_id", sa.String(length=36), nullable=False),
        sa.Column("field", sa.String(length=32), nullable=False),
        sa.Column("old_value", JSONB(), nullable=True),
        sa.Column("new_value", JSONB(), nullable=True),
        sa.Column("actor", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_probe_monitor_dataset_column_changes_view_created",
                    "probe_monitor_dataset_column_changes", ["view_id", "created_at"])
    for statement in COLUMN_CHANGES_DDL:
        op.execute(statement)

    op.create_table(
        "detector_success_reports",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("identity_sha256", sa.String(length=64), nullable=False),
        sa.Column("behaviour_key", sa.String(length=64), nullable=False),
        sa.Column("behaviour_version", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("dataset_identity", JSONB(), nullable=False),
        sa.Column("report_algorithm", sa.Integer(), nullable=False),
        sa.Column("inputs", JSONB(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("body_sha256", sa.String(length=64), nullable=False),
        sa.Column("overall_verdict", sa.String(length=16), nullable=False),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("identity_sha256", name="detector_success_reports_identity_sha256_key"),
    )
    op.create_index("ix_detector_success_reports_key_created", "detector_success_reports",
                    ["behaviour_key", "created_at"])
    for statement in REPORTS_DDL:
        op.execute(statement)

    op.drop_constraint("ck_probe_behaviour_view_links_use", "probe_behaviour_view_links", type_="check")
    op.create_check_constraint(
        "ck_probe_behaviour_view_links_use", "probe_behaviour_view_links",
        "use IN ('train', 'eval', 'calibration', 'blend_fit')",
    )


def downgrade() -> None:
    bind = op.get_bind()
    count = bind.execute(sa.text("SELECT count(*) FROM probe_behaviour_view_links WHERE use = 'blend_fit'")).scalar()
    if count:
        raise RuntimeError(
            f"{count} behaviour view link(s) use 'blend_fit', which the previous schema cannot hold; unlink "
            f"them before downgrading below effadddf2061"
        )
    op.drop_constraint("ck_probe_behaviour_view_links_use", "probe_behaviour_view_links", type_="check")
    op.create_check_constraint(
        "ck_probe_behaviour_view_links_use", "probe_behaviour_view_links",
        "use IN ('train', 'eval', 'calibration')",
    )
    op.execute("DROP TRIGGER IF EXISTS trg_detector_success_reports_append_only ON detector_success_reports")
    op.execute("DROP FUNCTION IF EXISTS detector_success_reports_append_only()")
    op.drop_index("ix_detector_success_reports_key_created", table_name="detector_success_reports")
    op.drop_table("detector_success_reports")
    op.execute("DROP TRIGGER IF EXISTS trg_probe_monitor_dataset_column_changes_append_only "
               "ON probe_monitor_dataset_column_changes")
    op.execute("DROP FUNCTION IF EXISTS probe_monitor_dataset_column_changes_append_only()")
    op.drop_index("ix_probe_monitor_dataset_column_changes_view_created",
                  table_name="probe_monitor_dataset_column_changes")
    op.drop_table("probe_monitor_dataset_column_changes")
    op.drop_column("probe_monitors", "calibration_n_negatives")
    op.drop_column("probe_monitors", "calibration_positions_path")
    for table in ("probe_monitors", "probe_monitor_runs"):
        op.drop_column(table, "behaviour_version")
        op.drop_column(table, "sub_question")
    for column in ("column_snapshots", "behaviour_version", "sub_question", "group_columns"):
        op.drop_column("probe_monitor_datasets", column)
