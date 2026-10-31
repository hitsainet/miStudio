"""probe spot-checks: probe_spot_checks and probe_spot_check_items

Revision ID: f3a9c2e5b8d1
Revises: e7c1a4b9d2f6
Create Date: 2026-10-21

Feature 038 (ENH-005 D28–D35, D39; PADR IDL-59). Two new tables; nothing else changes. The items hold
judged user text, which cascades with the run and expires after seven days unless the run is pinned.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "f3a9c2e5b8d1"
down_revision = "e7c1a4b9d2f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "probe_spot_checks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("behaviour", sa.String(length=64),
                  sa.ForeignKey("probe_behaviours.key", ondelete="RESTRICT"), nullable=False),
        sa.Column("behaviour_version", sa.Integer(), nullable=False),
        sa.Column("prompt_id", sa.String(length=36),
                  sa.ForeignKey("probe_judge_prompts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("prompt_key", sa.String(length=80), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("endpoint", sa.String(length=500), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("sample_spec", JSONB(), nullable=False),
        sa.Column("sample_returned", JSONB(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("celery_task_id", sa.String(length=155), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("partial", sa.Boolean(), nullable=False),
        sa.Column("parse_failures", sa.Integer(), nullable=False),
        sa.Column("parse_failure_limit", sa.Float(), nullable=False),
        sa.Column("metrics", JSONB(), nullable=True),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("excerpts_expire_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_probe_spot_checks_behaviour_status", "probe_spot_checks", ["behaviour", "status"])
    op.create_index("ix_probe_spot_checks_excerpts_expire_at", "probe_spot_checks", ["excerpts_expire_at"])
    op.create_table(
        "probe_spot_check_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("spot_check_id", sa.String(length=36),
                  sa.ForeignKey("probe_spot_checks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=True),
        sa.Column("event_id", sa.Integer(), nullable=True),
        sa.Column("stratum", sa.String(length=16), nullable=False),
        sa.Column("probe_id", sa.String(length=36), nullable=False),
        sa.Column("millm_probe_id", sa.String(length=64), nullable=True),
        sa.Column("window", sa.String(length=16), nullable=False),
        sa.Column("judge_saw", sa.String(length=16), nullable=False),
        sa.Column("probe_score", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("threshold_revision", sa.Integer(), nullable=True),
        sa.Column("probe_fired", sa.Boolean(), nullable=True),
        sa.Column("not_recorded", sa.Boolean(), nullable=False),
        sa.Column("judged_text", JSONB(), nullable=True),
        sa.Column("judge_reply", sa.Text(), nullable=True),
        sa.Column("judge_value", sa.String(length=64), nullable=True),
        sa.Column("judge_score", sa.Float(), nullable=True),
        sa.Column("judge_positive", sa.Boolean(), nullable=True),
        sa.Column("parse_failed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_probe_spot_check_items_check_probe", "probe_spot_check_items",
                    ["spot_check_id", "probe_id"])
    op.create_index(
        "ix_probe_spot_check_items_disagreements", "probe_spot_check_items", ["spot_check_id"],
        postgresql_where=sa.text(
            "parse_failed = false AND not_recorded = false "
            "AND probe_fired IS DISTINCT FROM judge_positive"
        ),
    )


def downgrade() -> None:
    op.drop_index("ix_probe_spot_check_items_disagreements", table_name="probe_spot_check_items")
    op.drop_index("ix_probe_spot_check_items_check_probe", table_name="probe_spot_check_items")
    op.drop_table("probe_spot_check_items")
    op.drop_index("ix_probe_spot_checks_excerpts_expire_at", table_name="probe_spot_checks")
    op.drop_index("ix_probe_spot_checks_behaviour_status", table_name="probe_spot_checks")
    op.drop_table("probe_spot_checks")
