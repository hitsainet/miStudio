"""add max_length column to dataset_tokenizations

Revision ID: 9c0d1e2f3a4b
Revises: 8b9c0d1e2f3a
Create Date: 2026-01-01 18:30:00.000000

This migration:
1. Adds max_length column to dataset_tokenizations table
2. Creates unique constraint (dataset_id, model_id, max_length)

Note: No existing unique constraint to drop - only had indexes before.
The new constraint allows same dataset+model with different max_lengths.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9c0d1e2f3a4b'
down_revision: Union[str, None] = '8b9c0d1e2f3a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add max_length column with default value of 512
    op.add_column(
        'dataset_tokenizations',
        sa.Column(
            'max_length',
            sa.Integer(),
            nullable=False,
            server_default='512',
            comment='Maximum sequence length used for tokenization'
        )
    )

    # Create unique constraint including max_length
    # This enables multiple tokenizations per dataset+model with different max_lengths
    op.create_unique_constraint(
        'uq_dataset_model_maxlen_tokenization',
        'dataset_tokenizations',
        ['dataset_id', 'model_id', 'max_length']
    )


def downgrade() -> None:
    # Drop the unique constraint
    op.drop_constraint(
        'uq_dataset_model_maxlen_tokenization',
        'dataset_tokenizations',
        type_='unique'
    )

    # Remove max_length column
    op.drop_column('dataset_tokenizations', 'max_length')
