"""add_category_column_to_features

Revision ID: de2e3ad17dc1
Revises: 6819dd3caeb3
Create Date: 2025-11-08 17:28:26.277334

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'de2e3ad17dc1'
down_revision: Union[str, None] = '6819dd3caeb3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add category column to features table for dual-label system."""
    op.add_column(
        'features',
        sa.Column('category', sa.String(length=255), nullable=True)
    )


def downgrade() -> None:
    """Remove category column from features table."""
    op.drop_column('features', 'category')
