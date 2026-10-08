"""add_layer_idx_to_training_metrics

Revision ID: 2d4f3b18cc10
Revises: 09d85441a622
Create Date: 2025-10-21 23:13:05.899341

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2d4f3b18cc10'
down_revision: Union[str, None] = '09d85441a622'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add layer_idx column to training_metrics table
    # NULL values represent aggregated metrics across all layers
    op.add_column(
        'training_metrics',
        sa.Column('layer_idx', sa.Integer, nullable=True, comment='Layer index (NULL for aggregated metrics)')
    )

    # Add index for efficient querying by layer
    op.create_index(
        'idx_training_metrics_layer_idx',
        'training_metrics',
        ['training_id', 'layer_idx', 'step']
    )


def downgrade() -> None:
    # Remove index and column
    op.drop_index('idx_training_metrics_layer_idx', table_name='training_metrics')
    op.drop_column('training_metrics', 'layer_idx')
