"""add indexes to activation_extractions table

Revision ID: 456bdad91d81
Revises: bf3c8f1dc38c
Create Date: 2025-10-14 11:05:57.411657

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '456bdad91d81'
down_revision: Union[str, None] = 'bf3c8f1dc38c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add indexes for common query patterns

    # Index on model_id (foreign key - used in most queries)
    op.create_index(
        'idx_activation_extractions_model_id',
        'activation_extractions',
        ['model_id']
    )

    # Composite index on (model_id, status) for active extraction queries
    # Queries like: WHERE model_id = ? AND status IN ('QUEUED', 'LOADING', 'EXTRACTING', 'SAVING')
    op.create_index(
        'idx_activation_extractions_model_status',
        'activation_extractions',
        ['model_id', 'status']
    )

    # Index on created_at for sorting (list queries)
    op.create_index(
        'idx_activation_extractions_created_at',
        'activation_extractions',
        ['created_at']
    )

    # Index on status for global status queries (e.g., count active extractions)
    op.create_index(
        'idx_activation_extractions_status',
        'activation_extractions',
        ['status']
    )


def downgrade() -> None:
    # Drop indexes in reverse order
    op.drop_index('idx_activation_extractions_status', table_name='activation_extractions')
    op.drop_index('idx_activation_extractions_created_at', table_name='activation_extractions')
    op.drop_index('idx_activation_extractions_model_status', table_name='activation_extractions')
    op.drop_index('idx_activation_extractions_model_id', table_name='activation_extractions')
