"""add_fvu_to_training_metrics

Revision ID: da9fc8175694
Revises: 4a1844011c28
Create Date: 2026-01-14

Add 'fvu' (Fraction of Variance Unexplained) column to training_metrics table.
FVU = var(residuals) / var(original) - a key metric for SAE reconstruction quality.
Lower FVU indicates better reconstruction.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'da9fc8175694'
down_revision: Union[str, None] = '4a1844011c28'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add fvu column to training_metrics table.

    FVU (Fraction of Variance Unexplained) measures reconstruction quality:
    - FVU = var(x - x_reconstructed) / var(x)
    - FVU = 0 means perfect reconstruction
    - FVU = 1 means reconstruction is no better than mean
    - FVU < 1 indicates the SAE captures some variance
    """
    op.add_column(
        'training_metrics',
        sa.Column('fvu', sa.Float(), nullable=True)
    )


def downgrade() -> None:
    """Remove fvu column from training_metrics table."""
    op.drop_column('training_metrics', 'fvu')
