"""remove_unique_constraint_dataset_model_tokenization

Revision ID: b83ef173c7a4
Revises: 578abb790b30
Create Date: 2025-11-12 00:27:18.745518

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b83ef173c7a4'
down_revision: Union[str, None] = '578abb790b30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Remove the UNIQUE constraint that prevents multiple tokenizations
    # for the same dataset+model combination with different settings
    op.drop_constraint('uq_dataset_model_tokenization', 'dataset_tokenizations', type_='unique')


def downgrade() -> None:
    # Re-add the UNIQUE constraint if rolling back
    # Note: This will fail if there are multiple tokenizations for the same dataset+model
    op.create_unique_constraint('uq_dataset_model_tokenization', 'dataset_tokenizations', ['dataset_id', 'model_id'])
