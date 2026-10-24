"""Make training_id nullable in extraction_jobs and features tables.

Extraction jobs and features can now be created for SAEs that don't have an
associated training (e.g., externally imported SAEs). This requires training_id
to be nullable in both tables.

Revision ID: j6k7l8m9n0o1
Revises: i5j6k7l8m9n0
Create Date: 2026-01-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'j6k7l8m9n0o1'
down_revision: Union[str, None] = 'i5j6k7l8m9n0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def constraint_exists(constraint_name: str, table_name: str) -> bool:
    """Check if a constraint exists on a table."""
    conn = op.get_bind()
    result = conn.execute(
        sa.text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = :constraint_name
                AND table_name = :table_name
            )
        """),
        {"constraint_name": constraint_name, "table_name": table_name}
    )
    return result.scalar()


def column_is_nullable(table_name: str, column_name: str) -> bool:
    """Check if a column is nullable."""
    conn = op.get_bind()
    result = conn.execute(
        sa.text("""
            SELECT is_nullable FROM information_schema.columns
            WHERE table_name = :table_name AND column_name = :column_name
        """),
        {"table_name": table_name, "column_name": column_name}
    )
    row = result.fetchone()
    return row[0] == 'YES' if row else False


def upgrade() -> None:
    """Make training_id nullable and drop foreign key constraints."""

    # =========================================================================
    # extraction_jobs table
    # =========================================================================

    # Drop foreign key constraint if it exists
    if constraint_exists('extraction_jobs_training_id_fkey', 'extraction_jobs'):
        op.drop_constraint(
            'extraction_jobs_training_id_fkey',
            'extraction_jobs',
            type_='foreignkey'
        )

    # Make training_id nullable
    if not column_is_nullable('extraction_jobs', 'training_id'):
        op.alter_column(
            'extraction_jobs',
            'training_id',
            existing_type=sa.String(255),
            nullable=True
        )

    # =========================================================================
    # features table
    # =========================================================================

    # Drop foreign key constraint if it exists
    if constraint_exists('features_training_id_fkey', 'features'):
        op.drop_constraint(
            'features_training_id_fkey',
            'features',
            type_='foreignkey'
        )

    # Make training_id nullable
    if not column_is_nullable('features', 'training_id'):
        op.alter_column(
            'features',
            'training_id',
            existing_type=sa.String(255),
            nullable=True
        )


def downgrade() -> None:
    """Restore NOT NULL constraints and foreign keys."""

    # =========================================================================
    # extraction_jobs table
    # =========================================================================

    op.alter_column(
        'extraction_jobs',
        'training_id',
        existing_type=sa.String(255),
        nullable=False
    )

    op.create_foreign_key(
        'extraction_jobs_training_id_fkey',
        'extraction_jobs',
        'trainings',
        ['training_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # =========================================================================
    # features table
    # =========================================================================

    op.alter_column(
        'features',
        'training_id',
        existing_type=sa.String(255),
        nullable=False
    )

    op.create_foreign_key(
        'features_training_id_fkey',
        'features',
        'trainings',
        ['training_id'],
        ['id'],
        ondelete='CASCADE'
    )
