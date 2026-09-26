"""add_default_partition_to_feature_activations

Revision ID: 1b09fdca19e8
Revises: 76918d8aa763
Create Date: 2025-10-25 13:37:33.503984

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1b09fdca19e8'
down_revision: Union[str, None] = '76918d8aa763'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create default partition for feature_activations to handle feature IDs
    # that don't match the expected numeric range (e.g., feat_train_56_00000)
    op.execute("""
        CREATE TABLE IF NOT EXISTS feature_activations_default
        PARTITION OF feature_activations DEFAULT;
    """)


def downgrade() -> None:
    # Drop default partition
    op.execute("DROP TABLE IF EXISTS feature_activations_default;")
