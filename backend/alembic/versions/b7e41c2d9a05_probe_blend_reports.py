"""probe blend reports and the suggested blend (Feature 041 S2)

Revision ID: b7e41c2d9a05
Revises: effadddf2061
Create Date: 2026-10-24

Feature 041 slice S2 (FR-16, FR-19, FR-21; FTDD §4.2, T6 – T10; FTID §4):

* `probe_blend_reports` — CHECK on status; list index (behaviour, model, newest first, stable tie);
  partial index on the active statuses (admission count and janitor);
* `probe_blend_report_members` — `report_id` FK ON DELETE CASCADE; `probe_id` NO FK (T6);
* `probe_suggested_blends` — current advice, PK (behaviour_key, model_id);
* `probe_suggested_blend_history` — APPEND-ONLY, refused by a trigger on UPDATE and DELETE.

No FK to probes, runs, views or behaviours: a run delete or a view purge neither cascades into nor is
blocked by a report. The downgrade drops only these four tables and the trigger function.

`down_revision` read on 2026-10-24 by walking every revision's `down_revision` on this branch: one head,
040 S2's `effadddf2061`.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "b7e41c2d9a05"
down_revision = "effadddf2061"
branch_labels = None
depends_on = None

#: ⚠ A FROZEN COPY of `models.probe_blend.HISTORY_DDL`; pinned equal by `test_suggested_blend.py`.
HISTORY_FUNCTION = "refuse_suggested_blend_history_change"
HISTORY_DDL = (
    f"""
CREATE OR REPLACE FUNCTION {HISTORY_FUNCTION}() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'probe_suggested_blend_history is append-only (% refused)', TG_OP;
END;
$$ LANGUAGE plpgsql
""",
    """
CREATE TRIGGER trg_probe_suggested_blend_history_append_only
BEFORE UPDATE OR DELETE ON probe_suggested_blend_history
FOR EACH ROW EXECUTE FUNCTION refuse_suggested_blend_history_change()
""",
)


def upgrade() -> None:
    op.create_table(
        "probe_blend_reports",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("behaviour_key", sa.String(length=64), nullable=False),
        sa.Column("behaviour_version", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("dataset_version", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("failure_code", sa.String(length=48), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("request", JSONB(), nullable=False),
        sa.Column("resolved", JSONB(), nullable=False),
        sa.Column("report_path", sa.String(length=1000), nullable=True),
        sa.Column("report_sha256", sa.String(length=64), nullable=True),
        sa.Column("report_bytes", sa.Integer(), nullable=True),
        sa.Column("rows_dir", sa.String(length=1000), nullable=True),
        sa.Column("rows_pruned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rows_pruned_by", sa.String(length=100), nullable=True),
        sa.Column("rerun_of", sa.String(length=36), nullable=True),
        sa.Column("rerun_by", sa.String(length=36), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("celery_task_id", sa.String(length=155), nullable=True),
        sa.Column("engine_version", sa.String(length=16), nullable=False),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('queued', 'running', 'done', 'failed', 'cancelled')",
                           name="ck_probe_blend_reports_status"),
    )
    op.create_index("ix_probe_blend_reports_list", "probe_blend_reports",
                    ["behaviour_key", "model_id", sa.text("created_at DESC"), sa.text("id DESC")])
    op.create_index("ix_probe_blend_reports_active", "probe_blend_reports", ["status"],
                    postgresql_where=sa.text("status IN ('queued', 'running')"))
    op.create_table(
        "probe_blend_report_members",
        sa.Column("report_id", sa.String(length=36),
                  sa.ForeignKey("probe_blend_reports.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("probe_id", sa.String(length=36), primary_key=True),
    )
    op.create_index("ix_probe_blend_report_members_probe_id", "probe_blend_report_members", ["probe_id"])
    op.create_table(
        "probe_suggested_blends",
        sa.Column("behaviour_key", sa.String(length=64), primary_key=True),
        sa.Column("model_id", sa.String(length=255), primary_key=True),
        sa.Column("record", sa.LargeBinary(), nullable=False),
        sa.Column("record_sha256", sa.String(length=64), nullable=False),
        sa.Column("report_id", sa.String(length=36), nullable=False),
        sa.Column("candidate", sa.String(length=64), nullable=False),
        sa.Column("history_id", sa.String(length=36), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "probe_suggested_blend_history",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("behaviour_key", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("report_id", sa.String(length=36), nullable=False),
        sa.Column("candidate", sa.String(length=64), nullable=False),
        sa.Column("record", sa.LargeBinary(), nullable=False),
        sa.Column("record_sha256", sa.String(length=64), nullable=False),
        sa.Column("actor", sa.String(length=100), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_probe_suggested_blend_history_key_model", "probe_suggested_blend_history",
                    ["behaviour_key", "model_id", sa.text("created_at DESC")])
    for statement in HISTORY_DDL:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_probe_suggested_blend_history_append_only ON probe_suggested_blend_history")
    op.drop_table("probe_suggested_blend_history")
    op.execute(f"DROP FUNCTION IF EXISTS {HISTORY_FUNCTION}()")
    op.drop_table("probe_suggested_blends")
    op.drop_table("probe_blend_report_members")
    op.drop_table("probe_blend_reports")
