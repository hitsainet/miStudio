"""remove_tokenization_fields_from_datasets

Revision ID: 7282abcac53a
Revises: 2e1feb9cc451
Create Date: 2025-11-08 19:28:11.589849

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7282abcac53a'
down_revision: Union[str, None] = '2e1feb9cc451'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Remove tokenization-related columns from datasets table.

    These fields have been moved to the dataset_tokenizations table to support
    multiple tokenizations per dataset (one per model).
    """
    # Drop tokenization-related columns
    op.drop_column('datasets', 'tokenized_path')
    op.drop_column('datasets', 'vocab_size')
    op.drop_column('datasets', 'num_tokens')
    op.drop_column('datasets', 'avg_seq_length')


def downgrade() -> None:
    """
    Restore tokenization-related columns to datasets table.

    Note: This will not restore the actual data - that is preserved in
    the dataset_tokenizations table.
    """
    # Re-add tokenization-related columns
    op.add_column('datasets', sa.Column('tokenized_path', sa.String(512), nullable=True))
    op.add_column('datasets', sa.Column('vocab_size', sa.Integer(), nullable=True))
    op.add_column('datasets', sa.Column('num_tokens', sa.BigInteger(), nullable=True))
    op.add_column('datasets', sa.Column('avg_seq_length', sa.Float(), nullable=True))
