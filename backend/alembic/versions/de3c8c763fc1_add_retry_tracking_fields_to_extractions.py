"""add_retry_tracking_fields_to_extractions

Revision ID: de3c8c763fc1
Revises: 456bdad91d81
Create Date: 2025-10-15 04:28:31.809745

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'de3c8c763fc1'
down_revision: Union[str, None] = '456bdad91d81'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add retry tracking fields to activation_extractions table."""
    # Add retry_count column (default 0 for existing records)
    op.add_column(
        'activation_extractions',
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0')
    )

    # Add original_extraction_id column (nullable for original extractions)
    op.add_column(
        'activation_extractions',
        sa.Column('original_extraction_id', sa.String(length=255), nullable=True)
    )

    # Add retry_reason column (nullable)
    op.add_column(
        'activation_extractions',
        sa.Column('retry_reason', sa.Text(), nullable=True)
    )

    # Add auto_retried flag (default False for existing records)
    op.add_column(
        'activation_extractions',
        sa.Column('auto_retried', sa.Boolean(), nullable=False, server_default='false')
    )

    # Add error_type column for error classification (nullable)
    op.add_column(
        'activation_extractions',
        sa.Column('error_type', sa.String(length=50), nullable=True)
    )

    # Create index on original_extraction_id for faster retry lookups
    op.create_index(
        'ix_activation_extractions_original_extraction_id',
        'activation_extractions',
        ['original_extraction_id']
    )


def downgrade() -> None:
    """Remove retry tracking fields from activation_extractions table."""
    # Drop index
    op.drop_index('ix_activation_extractions_original_extraction_id', table_name='activation_extractions')

    # Drop columns
    op.drop_column('activation_extractions', 'error_type')
    op.drop_column('activation_extractions', 'auto_retried')
    op.drop_column('activation_extractions', 'retry_reason')
    op.drop_column('activation_extractions', 'original_extraction_id')
    op.drop_column('activation_extractions', 'retry_count')
