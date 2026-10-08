"""add token filter customization fields to dataset_tokenizations

Revision ID: 578abb790b30
Revises: b2c3d4e5f6g7
Create Date: 2025-11-11 14:52:27.529845

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '578abb790b30'
down_revision: Union[str, None] = 'b2c3d4e5f6g7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add token filtering configuration columns to dataset_tokenizations table
    op.add_column('dataset_tokenizations', sa.Column('remove_all_punctuation', sa.Boolean(), server_default='false', nullable=False, comment='If true, removes ALL punctuation characters from tokens'))
    op.add_column('dataset_tokenizations', sa.Column('custom_filter_chars', sa.String(length=255), nullable=True, comment="Custom characters to filter (e.g., '~@#$%')"))


def downgrade() -> None:
    # Remove token filtering configuration columns from dataset_tokenizations table
    op.drop_column('dataset_tokenizations', 'custom_filter_chars')
    op.drop_column('dataset_tokenizations', 'remove_all_punctuation')
