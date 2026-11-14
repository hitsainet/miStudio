"""add_token_filter_columns_to_labeling_jobs

Revision ID: c351738faecf
Revises: e2a5af6a2b11
Create Date: 2025-11-17 10:22:26.304013

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c351738faecf'
down_revision: Union[str, None] = 'e2a5af6a2b11'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add token filtering configuration columns to labeling_jobs table
    op.add_column('labeling_jobs', sa.Column('filter_special', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('labeling_jobs', sa.Column('filter_single_char', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('labeling_jobs', sa.Column('filter_punctuation', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('labeling_jobs', sa.Column('filter_numbers', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('labeling_jobs', sa.Column('filter_fragments', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('labeling_jobs', sa.Column('filter_stop_words', sa.Boolean(), nullable=False, server_default='false'))


def downgrade() -> None:
    # Remove token filtering configuration columns from labeling_jobs table
    op.drop_column('labeling_jobs', 'filter_stop_words')
    op.drop_column('labeling_jobs', 'filter_fragments')
    op.drop_column('labeling_jobs', 'filter_numbers')
    op.drop_column('labeling_jobs', 'filter_punctuation')
    op.drop_column('labeling_jobs', 'filter_single_char')
    op.drop_column('labeling_jobs', 'filter_special')
