"""add_layer_index_to_extraction_jobs

Revision ID: 56ee0619fe2e
Revises: d4e5f6a7b8c9
Create Date: 2026-01-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '56ee0619fe2e'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add layer_index column to extraction_jobs table.

    This column stores the layer index (e.g., 10, 17, 24) for multi-layer
    trainings where the user needs to select which layer's SAE to use
    for feature extraction.
    """
    op.add_column(
        'extraction_jobs',
        sa.Column('layer_index', sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    """
    Remove layer_index column from extraction_jobs table.
    """
    op.drop_column('extraction_jobs', 'layer_index')
