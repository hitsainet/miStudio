"""Add extraction_ids JSONB column to trainings

Revision ID: l8m9n0o1p2q3
Revises: k7l8m9n0o1p2
Create Date: 2026-02-15

Supports multi-extraction training: one extraction per dataset.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision = 'l8m9n0o1p2q3'
down_revision = 'k7l8m9n0o1p2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('trainings', sa.Column('extraction_ids', JSONB, nullable=True, server_default='[]'))


def downgrade() -> None:
    op.drop_column('trainings', 'extraction_ids')
