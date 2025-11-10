"""add extraction filter fields

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2025-11-10 00:00:01.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add extraction filter fields to extractions table
    op.add_column('extractions', sa.Column('extraction_filter_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('extractions', sa.Column('extraction_filter_mode', sa.String(20), nullable=False, server_default='standard'))


def downgrade() -> None:
    op.drop_column('extractions', 'extraction_filter_mode')
    op.drop_column('extractions', 'extraction_filter_enabled')
