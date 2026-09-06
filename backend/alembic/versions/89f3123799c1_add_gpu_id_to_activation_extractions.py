"""add gpu_id to activation_extractions

Revision ID: 89f3123799c1
Revises: 7d3539aaf247
Create Date: 2025-12-21 02:31:10.604384

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '89f3123799c1'
down_revision: Union[str, None] = '7d3539aaf247'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add gpu_id column to activation_extractions table."""
    op.add_column(
        'activation_extractions',
        sa.Column('gpu_id', sa.Integer(), nullable=False, server_default='0')
    )


def downgrade() -> None:
    """Remove gpu_id column from activation_extractions table."""
    op.drop_column('activation_extractions', 'gpu_id')
