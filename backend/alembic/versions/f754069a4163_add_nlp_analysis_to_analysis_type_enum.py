"""add_nlp_analysis_to_analysis_type_enum

Revision ID: f754069a4163
Revises: b3c4d5e6f7g8
Create Date: 2025-12-08 02:06:27.460062

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f754069a4163'
down_revision: Union[str, None] = 'b3c4d5e6f7g8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add 'nlp_analysis' value to analysis_type_enum."""
    # PostgreSQL requires ALTER TYPE to add new enum values
    # This must be done outside a transaction block in some cases,
    # but Alembic handles this correctly with op.execute
    op.execute("ALTER TYPE analysis_type_enum ADD VALUE IF NOT EXISTS 'nlp_analysis'")


def downgrade() -> None:
    """
    Note: PostgreSQL does not support removing enum values directly.
    To fully downgrade, you would need to:
    1. Create a new enum without 'nlp_analysis'
    2. Update the column to use the new enum
    3. Drop the old enum

    For safety, we just leave the enum value in place.
    """
    pass
