"""Add hook_type column to external_saes table.

Adds hook_type column to track which hook point the SAE was trained on
(e.g., hook_resid_pre, hook_resid_post, hook_mlp_out).
This is needed for multi-SAE import where a training produces SAEs
for multiple layers AND multiple hook types.

Revision ID: h4i5j6k7l8m9
Revises: g3h4i5j6k7l8
Create Date: 2026-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'h4i5j6k7l8m9'
down_revision: Union[str, None] = 'g3h4i5j6k7l8'
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


def index_exists(index_name: str) -> bool:
    """Check if an index exists."""
    conn = op.get_bind()
    result = conn.execute(
        sa.text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE indexname = :index_name
            )
        """),
        {"index_name": index_name}
    )
    return result.scalar()


def upgrade() -> None:
    """Add hook_type column to external_saes table."""

    # Add hook_type column - stores the hook point (e.g., hook_resid_pre)
    if not column_exists('external_saes', 'hook_type'):
        op.add_column(
            'external_saes',
            sa.Column('hook_type', sa.String(100), nullable=True)
        )

    # Create composite index for layer + hook_type queries
    if not index_exists('ix_external_saes_layer_hook_type'):
        op.create_index(
            'ix_external_saes_layer_hook_type',
            'external_saes',
            ['layer', 'hook_type']
        )

    # Create index for training_id lookups (useful for listing SAEs from training)
    if not index_exists('ix_external_saes_training_id'):
        op.create_index(
            'ix_external_saes_training_id',
            'external_saes',
            ['training_id']
        )


def downgrade() -> None:
    """Remove hook_type column from external_saes table."""

    # Drop indexes first
    if index_exists('ix_external_saes_training_id'):
        op.drop_index('ix_external_saes_training_id', table_name='external_saes')

    if index_exists('ix_external_saes_layer_hook_type'):
        op.drop_index('ix_external_saes_layer_hook_type', table_name='external_saes')

    # Drop column
    if column_exists('external_saes', 'hook_type'):
        op.drop_column('external_saes', 'hook_type')
