"""add enhanced_labeling_jobs table and enhanced_llm label source

Revision ID: r5s6t7u8v9w0
Revises: q4r5s6t7u8v9
Create Date: 2026-04-19 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "r5s6t7u8v9w0"
down_revision = "q4r5s6t7u8v9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Extend label_source_enum with enhanced_llm value
    op.execute("ALTER TYPE label_source_enum ADD VALUE IF NOT EXISTS 'enhanced_llm'")

    # 2. Create enhanced_labeling_jobs table
    op.create_table(
        "enhanced_labeling_jobs",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "feature_id",
            sa.String(255),
            sa.ForeignKey("features.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("status", sa.String(50), nullable=False, server_default="queued"),
        sa.Column("phase", sa.String(50), nullable=True),
        sa.Column("examples_total", sa.Integer, nullable=False, server_default="20"),
        sa.Column("examples_completed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("workers", sa.Integer, nullable=False, server_default="8"),
        sa.Column("endpoint", sa.String(500), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("pass1_summaries", postgresql.JSONB, nullable=True),
        sa.Column("raw_synthesis", sa.Text, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_elj_feature_id", "enhanced_labeling_jobs", ["feature_id"])
    op.create_index("idx_elj_status", "enhanced_labeling_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("idx_elj_status", table_name="enhanced_labeling_jobs")
    op.drop_index("idx_elj_feature_id", table_name="enhanced_labeling_jobs")
    op.drop_table("enhanced_labeling_jobs")
    # Note: PostgreSQL does not support removing enum values; enhanced_llm remains in the type
