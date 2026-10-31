"""add_extraction_unique_constraint

Revision ID: 8131e563f5fe
Revises: 1b09fdca19e8
Create Date: 2025-10-28 01:42:59.663464

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8131e563f5fe'
down_revision: Union[str, None] = '1b09fdca19e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add partial unique index: only one QUEUED or EXTRACTING job per training
    # This prevents duplicate active extractions for the same training
    op.create_index(
        'idx_extraction_active_training_unique',
        'extraction_jobs',
        ['training_id'],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'extracting')")
    )


def downgrade() -> None:
    op.drop_index('idx_extraction_active_training_unique', table_name='extraction_jobs')
