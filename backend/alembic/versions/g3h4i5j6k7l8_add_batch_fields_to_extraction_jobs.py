"""Add batch fields to extraction_jobs table.

Adds fields to support batch extraction operations where multiple
SAEs can be extracted in a single batch.

Revision ID: g3h4i5j6k7l8
Revises: f2g3h4i5j6k7
Create Date: 2026-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'g3h4i5j6k7l8'
down_revision: Union[str, None] = 'f2g3h4i5j6k7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    conn = op.get_bind()
    result = conn.execute(
        sa.text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = :table_name AND column_name = :column_name
            )
        """),
        {"table_name": table_name, "column_name": column_name}
    )
    return result.scalar()


def upgrade() -> None:
    """Add batch fields to extraction_jobs table."""

    # Add batch_id column - groups related extraction jobs
    if not column_exists('extraction_jobs', 'batch_id'):
        op.add_column(
            'extraction_jobs',
            sa.Column('batch_id', sa.String(255), nullable=True)
        )
        # Create index for batch_id lookups
        op.create_index(
            'ix_extraction_jobs_batch_id',
            'extraction_jobs',
            ['batch_id']
        )

    # Add batch_position column - position within batch (1-indexed)
    if not column_exists('extraction_jobs', 'batch_position'):
        op.add_column(
            'extraction_jobs',
            sa.Column('batch_position', sa.Integer, nullable=True)
        )

    # Add batch_total column - total jobs in batch
    if not column_exists('extraction_jobs', 'batch_total'):
        op.add_column(
            'extraction_jobs',
            sa.Column('batch_total', sa.Integer, nullable=True)
        )


def downgrade() -> None:
    """Remove batch fields from extraction_jobs table."""

    # Drop index first
    try:
        op.drop_index('ix_extraction_jobs_batch_id', table_name='extraction_jobs')
    except Exception:
        pass  # Index may not exist

    # Drop columns
    if column_exists('extraction_jobs', 'batch_total'):
        op.drop_column('extraction_jobs', 'batch_total')

    if column_exists('extraction_jobs', 'batch_position'):
        op.drop_column('extraction_jobs', 'batch_position')

    if column_exists('extraction_jobs', 'batch_id'):
        op.drop_column('extraction_jobs', 'batch_id')
