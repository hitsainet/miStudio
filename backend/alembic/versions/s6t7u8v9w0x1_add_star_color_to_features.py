"""add star_color to features

Revision ID: s6t7u8v9w0x1
Revises: r5s6t7u8v9w0
Create Date: 2026-04-22 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "s6t7u8v9w0x1"
down_revision = "r5s6t7u8v9w0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "features",
        sa.Column("star_color", sa.String(20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("features", "star_color")
