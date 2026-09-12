"""add api_timeout column to labeling_jobs

Revision ID: q4r5s6t7u8v9
Revises: p3q4r5s6t7u8
Create Date: 2026-04-09 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "q4r5s6t7u8v9"
down_revision = "p3q4r5s6t7u8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "ALTER TABLE labeling_jobs ADD COLUMN IF NOT EXISTS api_timeout FLOAT NOT NULL DEFAULT 120.0"
        )
    )


def downgrade() -> None:
    op.drop_column("labeling_jobs", "api_timeout")
