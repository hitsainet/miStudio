"""add_fvu_column_to_training_metrics

Revision ID: 92780ad64734
Revises: 9b02b81e6908
Create Date: 2025-11-26 21:35:26.855981

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '92780ad64734'
down_revision: Union[str, None] = '9b02b81e6908'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add FVU (Fraction of Variance Unexplained) column to training_metrics
    # FVU = var(residuals) / var(original) is a key metric for SAE evaluation
    op.add_column(
        'training_metrics',
        sa.Column('fvu', sa.Float(), nullable=True,
                  comment='Fraction of Variance Unexplained (var_residuals / var_original)')
    )


def downgrade() -> None:
    op.drop_column('training_metrics', 'fvu')
