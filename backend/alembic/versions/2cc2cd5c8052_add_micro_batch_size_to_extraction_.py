"""add_micro_batch_size_to_extraction_tables

Revision ID: 2cc2cd5c8052
Revises: 8131e563f5fe
Create Date: 2025-11-01 12:29:44.350810

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2cc2cd5c8052'
down_revision: Union[str, None] = '8131e563f5fe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add micro_batch_size column to activation_extractions table
    op.add_column('activation_extractions', sa.Column('micro_batch_size', sa.Integer(), nullable=True))

    # Add micro_batch_size column to extraction_templates table
    op.add_column('extraction_templates', sa.Column('micro_batch_size', sa.Integer(), nullable=True))


def downgrade() -> None:
    # Remove micro_batch_size column from extraction_templates table
    op.drop_column('extraction_templates', 'micro_batch_size')

    # Remove micro_batch_size column from activation_extractions table
    op.drop_column('activation_extractions', 'micro_batch_size')
