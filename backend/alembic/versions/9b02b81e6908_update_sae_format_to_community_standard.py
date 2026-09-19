"""update_sae_format_to_community_standard

Revision ID: 9b02b81e6908
Revises: a1c2e3f4g5h6
Create Date: 2025-11-26 11:55:39.968889

This migration:
1. Updates the default value of external_saes.format from 'saelens' to 'community_standard'
2. Migrates existing records with format='saelens' to 'community_standard'
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9b02b81e6908'
down_revision: Union[str, None] = 'a1c2e3f4g5h6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update existing records from 'saelens' to 'community_standard'
    op.execute(
        "UPDATE external_saes SET format = 'community_standard' WHERE format = 'saelens'"
    )

    # Update the default value for the format column
    op.alter_column(
        'external_saes',
        'format',
        server_default='community_standard'
    )


def downgrade() -> None:
    # Revert existing records from 'community_standard' back to 'saelens'
    op.execute(
        "UPDATE external_saes SET format = 'saelens' WHERE format = 'community_standard'"
    )

    # Revert the default value for the format column
    op.alter_column(
        'external_saes',
        'format',
        server_default='saelens'
    )
