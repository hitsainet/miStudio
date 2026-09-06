"""add missing label_source_enum values

Revision ID: 7a8b9c0d1e2f
Revises: 89f3123799c1
Create Date: 2025-12-21 18:45:00.000000

NOTE: This migration exists for databases created before this fix.
The original migration (76918d8aa763) has been updated to include all enum values,
so fresh installs will have the correct enum from the start.
This migration uses IF NOT EXISTS and is idempotent.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '7a8b9c0d1e2f'
down_revision: Union[str, None] = '89f3123799c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add missing enum values to label_source_enum: llm, local_llm, openai."""
    # PostgreSQL requires ALTER TYPE to add enum values
    # Each value must be added separately
    op.execute("ALTER TYPE label_source_enum ADD VALUE IF NOT EXISTS 'llm'")
    op.execute("ALTER TYPE label_source_enum ADD VALUE IF NOT EXISTS 'local_llm'")
    op.execute("ALTER TYPE label_source_enum ADD VALUE IF NOT EXISTS 'openai'")


def downgrade() -> None:
    """
    Note: PostgreSQL does not support removing enum values directly.
    To downgrade, you would need to:
    1. Create a new enum type without the removed values
    2. Update the column to use the new type
    3. Drop the old type

    For now, we'll leave the enum values in place as they don't cause harm.
    """
    pass
