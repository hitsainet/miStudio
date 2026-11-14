"""allow_multiple_extractions_per_training

Revision ID: 126732716f92
Revises: 075931c81831
Create Date: 2025-11-19 10:35:31.881812

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '126732716f92'
down_revision: Union[str, None] = '075931c81831'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Update the unique constraint on features table to allow multiple extractions per training.
    Changes from (training_id, neuron_index) to (training_id, extraction_job_id, neuron_index).
    """
    # Drop the old constraint
    op.drop_constraint('uq_features_training_neuron', 'features', type_='unique')

    # Create the new constraint
    op.create_unique_constraint(
        'uq_features_extraction_neuron',
        'features',
        ['training_id', 'extraction_job_id', 'neuron_index']
    )


def downgrade() -> None:
    """
    Revert back to single extraction per training constraint.
    """
    # Drop the new constraint
    op.drop_constraint('uq_features_extraction_neuron', 'features', type_='unique')

    # Recreate the old constraint
    op.create_unique_constraint(
        'uq_features_training_neuron',
        'features',
        ['training_id', 'neuron_index']
    )
