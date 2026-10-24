"""add_dataset_ids_array_to_trainings

Revision ID: f2g3h4i5j6k7
Revises: e1f2g3h4i5j6
Create Date: 2025-01-17 10:00:00

Adds dataset_ids JSONB array column to trainings table to support
training on multiple datasets simultaneously.

This migration is IDEMPOTENT:
- Checks if column exists before adding
- Migrates existing dataset_id values to dataset_ids array
- Keeps dataset_id column for backward compatibility

Works across all deployment modes:
- Native development
- Docker Compose containers
- Kubernetes pods
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'f2g3h4i5j6k7'
down_revision: Union[str, None] = 'e1f2g3h4i5j6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    conn = op.get_bind()
    result = conn.execute(sa.text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = :table_name AND column_name = :column_name
    """), {"table_name": table_name, "column_name": column_name})
    return result.fetchone() is not None


def upgrade() -> None:
    conn = op.get_bind()

    # Step 1: Add dataset_ids column if it doesn't exist
    if not column_exists('trainings', 'dataset_ids'):
        print("Adding dataset_ids column to trainings table...")
        op.add_column(
            'trainings',
            sa.Column('dataset_ids', JSONB, nullable=True)
        )

        # Step 2: Migrate existing dataset_id values to dataset_ids array
        print("Migrating existing dataset_id values to dataset_ids array...")
        conn.execute(sa.text("""
            UPDATE trainings
            SET dataset_ids = jsonb_build_array(dataset_id)
            WHERE dataset_id IS NOT NULL AND dataset_id != ''
        """))

        # Set empty array for any remaining nulls
        conn.execute(sa.text("""
            UPDATE trainings
            SET dataset_ids = '[]'::jsonb
            WHERE dataset_ids IS NULL
        """))

        # Step 3: Make column NOT NULL with default
        print("Setting dataset_ids column constraints...")
        op.alter_column(
            'trainings',
            'dataset_ids',
            nullable=False,
            server_default=sa.text("'[]'::jsonb")
        )

        print("Migration complete: dataset_ids column added and populated")
    else:
        print("dataset_ids column already exists, skipping")

    # Also update training_templates table if it exists
    if column_exists('training_templates', 'dataset_id') and not column_exists('training_templates', 'dataset_ids'):
        print("Adding dataset_ids column to training_templates table...")
        op.add_column(
            'training_templates',
            sa.Column('dataset_ids', JSONB, nullable=True)
        )

        # Migrate existing values
        conn.execute(sa.text("""
            UPDATE training_templates
            SET dataset_ids = CASE
                WHEN dataset_id IS NOT NULL AND dataset_id != '' THEN jsonb_build_array(dataset_id)
                ELSE '[]'::jsonb
            END
        """))

        # Make NOT NULL with default
        op.alter_column(
            'training_templates',
            'dataset_ids',
            nullable=False,
            server_default=sa.text("'[]'::jsonb")
        )

        print("Migration complete: training_templates.dataset_ids column added")


def downgrade() -> None:
    # Downgrade: Remove dataset_ids columns (data loss warning)
    print("WARNING: Downgrading will remove dataset_ids columns. Multi-dataset info will be lost.")

    if column_exists('training_templates', 'dataset_ids'):
        op.drop_column('training_templates', 'dataset_ids')

    if column_exists('trainings', 'dataset_ids'):
        op.drop_column('trainings', 'dataset_ids')
