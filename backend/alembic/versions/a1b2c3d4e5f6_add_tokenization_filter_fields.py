"""add tokenization filter fields

Revision ID: a1b2c3d4e5f6
Revises: 7282abcac53a
Create Date: 2025-11-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '7282abcac53a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add tokenization filter fields to datasets table
    op.add_column('datasets', sa.Column('tokenization_filter_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('datasets', sa.Column('tokenization_filter_mode', sa.String(20), nullable=False, server_default='conservative'))
    op.add_column('datasets', sa.Column('tokenization_junk_ratio_threshold', sa.Float(), nullable=False, server_default='0.7'))


def downgrade() -> None:
    op.drop_column('datasets', 'tokenization_junk_ratio_threshold')
    op.drop_column('datasets', 'tokenization_filter_mode')
    op.drop_column('datasets', 'tokenization_filter_enabled')
