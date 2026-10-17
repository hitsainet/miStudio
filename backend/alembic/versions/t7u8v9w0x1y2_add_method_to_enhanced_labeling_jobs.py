"""add method to enhanced_labeling_jobs

Revision ID: t7u8v9w0x1y2
Revises: s6t7u8v9w0x1
Create Date: 2026-04-24 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "t7u8v9w0x1y2"
down_revision = "s6t7u8v9w0x1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "enhanced_labeling_jobs",
        sa.Column(
            "method",
            sa.String(50),
            nullable=False,
            server_default="openai_compatible",
        ),
    )


def downgrade() -> None:
    op.drop_column("enhanced_labeling_jobs", "method")
