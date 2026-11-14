"""add max_tokens column to labeling_jobs (missing migration)

Revision ID: p3q4r5s6t7u8
Revises: o2p3q4r5s6t7
Create Date: 2026-03-29 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "p3q4r5s6t7u8"
down_revision = "o2p3q4r5s6t7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "ALTER TABLE labeling_jobs ADD COLUMN IF NOT EXISTS max_tokens INTEGER NOT NULL DEFAULT 300"
        )
    )


def downgrade() -> None:
    op.drop_column("labeling_jobs", "max_tokens")
