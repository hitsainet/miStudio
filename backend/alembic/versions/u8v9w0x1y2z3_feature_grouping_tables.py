"""Feature 010: feature grouping tables + agent approval requests

Creates the cross-feature grouping data layer:
- feature_grouping_runs   (precompute run lifecycle)
- feature_token_index     (token→feature inverted index)
- feature_groups          (context-similarity subgroups per shared token)
- feature_group_members   (group membership)
- agent_approval_requests (MCP steering operator-approval mode)

Revision ID: u8v9w0x1y2z3
Revises: f2b3c4d5e6f7
Create Date: 2026-07-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "u8v9w0x1y2z3"
down_revision = "f2b3c4d5e6f7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feature_grouping_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("extraction_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("params", JSONB, nullable=False),
        sa.Column("params_hash", sa.String(64), nullable=False),
        sa.Column("feature_count", sa.Integer, nullable=True),
        sa.Column("group_count", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_feature_grouping_runs_extraction_id", "feature_grouping_runs", ["extraction_id"])

    op.create_table(
        "feature_token_index",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("feature_grouping_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("extraction_id", sa.String(255), nullable=False),
        sa.Column(
            "feature_id",
            sa.String(255),
            sa.ForeignKey("features.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("neuron_index", sa.Integer, nullable=False),
        sa.Column("raw_token", sa.Text, nullable=False),
        sa.Column("normalized_token", sa.Text, nullable=False),
        sa.Column("token_rank", sa.Integer, nullable=False),
        sa.Column("weight", sa.Float, nullable=False),
        sa.Column("context_tokens", JSONB, nullable=True),
    )
    op.create_index("ix_feature_token_index_run_id", "feature_token_index", ["run_id"])
    op.create_index("ix_feature_token_index_feature_id", "feature_token_index", ["feature_id"])
    op.create_index("ix_fti_ext_token", "feature_token_index", ["extraction_id", "normalized_token"])

    op.create_table(
        "feature_groups",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("feature_grouping_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("extraction_id", sa.String(255), nullable=False),
        sa.Column("normalized_token", sa.Text, nullable=False),
        sa.Column("display_token", sa.Text, nullable=False),
        sa.Column("member_count", sa.Integer, nullable=False),
        sa.Column("cohesion", sa.Float, nullable=False),
    )
    op.create_index("ix_feature_groups_run_id", "feature_groups", ["run_id"])
    op.create_index("ix_feature_groups_extraction_id", "feature_groups", ["extraction_id"])
    op.create_index("ix_fg_ext_token", "feature_groups", ["extraction_id", "normalized_token"])

    op.create_table(
        "feature_group_members",
        sa.Column(
            "group_id",
            sa.String(36),
            sa.ForeignKey("feature_groups.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "feature_id",
            sa.String(255),
            sa.ForeignKey("features.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("similarity", sa.Float, nullable=False),
        sa.Column("context_snippet", sa.Text, nullable=True),
    )

    op.create_table(
        "agent_approval_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tool_name", sa.String(50), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("status", sa.String(10), nullable=False, server_default="pending"),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("steering_task_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_agent_approval_requests_status", "agent_approval_requests", ["status"])


def downgrade() -> None:
    op.drop_table("agent_approval_requests")
    op.drop_table("feature_group_members")
    op.drop_table("feature_groups")
    op.drop_table("feature_token_index")
    op.drop_table("feature_grouping_runs")
