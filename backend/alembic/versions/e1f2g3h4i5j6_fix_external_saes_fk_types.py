"""fix_external_saes_fk_types

Revision ID: e1f2g3h4i5j6
Revises: da9fc8175694
Create Date: 2025-01-16 10:00:00

This migration fixes type mismatches for external_saes.training_id and model_id
columns. These columns should be VARCHAR(255) to match the referenced primary
keys in trainings and models tables.

This issue occurred when the K8s database was created manually with INTEGER
types before migrations existed. The original migration (a1c2e3f4g5h6) correctly
defines these as VARCHAR(255), but didn't run on K8s.

This migration is IDEMPOTENT:
- On databases with correct types (VARCHAR): No-op, columns already correct
- On databases with INTEGER types: Converts to VARCHAR(255) safely

For INTEGER -> VARCHAR conversion:
1. Drops foreign key constraint
2. Alters column type with USING clause to cast values
3. Recreates foreign key constraint
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = 'e1f2g3h4i5j6'
down_revision: Union[str, None] = 'da9fc8175694'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def get_column_type(table_name: str, column_name: str) -> str:
    """Get the PostgreSQL type name for a column."""
    conn = op.get_bind()
    result = conn.execute(sa.text("""
        SELECT data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = :table_name AND column_name = :column_name
    """), {"table_name": table_name, "column_name": column_name})
    row = result.fetchone()
    if row:
        data_type = row[0]
        if data_type == 'character varying':
            return f"varchar({row[1]})"
        return data_type
    return "unknown"


def has_foreign_key(table_name: str, constraint_name: str) -> bool:
    """Check if a foreign key constraint exists."""
    conn = op.get_bind()
    result = conn.execute(sa.text("""
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = :table_name
        AND constraint_name = :constraint_name
        AND constraint_type = 'FOREIGN KEY'
    """), {"table_name": table_name, "constraint_name": constraint_name})
    return result.fetchone() is not None


def upgrade() -> None:
    conn = op.get_bind()

    # Check and fix training_id column
    training_id_type = get_column_type('external_saes', 'training_id')
    print(f"external_saes.training_id current type: {training_id_type}")

    if training_id_type == 'integer':
        print("Converting training_id from INTEGER to VARCHAR(255)...")

        # Drop FK constraint if exists
        if has_foreign_key('external_saes', 'fk_external_saes_training_id'):
            op.drop_constraint('fk_external_saes_training_id', 'external_saes', type_='foreignkey')

        # Convert column type (USING casts existing integer values to text)
        conn.execute(sa.text("""
            ALTER TABLE external_saes
            ALTER COLUMN training_id TYPE VARCHAR(255)
            USING training_id::VARCHAR(255)
        """))

        # Recreate FK constraint
        op.create_foreign_key(
            'fk_external_saes_training_id',
            'external_saes',
            'trainings',
            ['training_id'],
            ['id'],
            ondelete='SET NULL'
        )
        print("training_id converted successfully")
    else:
        print(f"training_id already correct type ({training_id_type}), skipping")

    # Check and fix model_id column
    model_id_type = get_column_type('external_saes', 'model_id')
    print(f"external_saes.model_id current type: {model_id_type}")

    if model_id_type == 'integer':
        print("Converting model_id from INTEGER to VARCHAR(255)...")

        # Drop FK constraint if exists
        if has_foreign_key('external_saes', 'fk_external_saes_model_id'):
            op.drop_constraint('fk_external_saes_model_id', 'external_saes', type_='foreignkey')

        # Convert column type
        conn.execute(sa.text("""
            ALTER TABLE external_saes
            ALTER COLUMN model_id TYPE VARCHAR(255)
            USING model_id::VARCHAR(255)
        """))

        # Recreate FK constraint
        op.create_foreign_key(
            'fk_external_saes_model_id',
            'external_saes',
            'models',
            ['model_id'],
            ['id'],
            ondelete='SET NULL'
        )
        print("model_id converted successfully")
    else:
        print(f"model_id already correct type ({model_id_type}), skipping")


def downgrade() -> None:
    # Downgrade is intentionally a no-op
    # We don't want to convert back to INTEGER as that would break FK relationships
    # and potentially lose data (VARCHAR IDs won't fit in INTEGER)
    print("Downgrade is a no-op - VARCHAR(255) is the correct type for these columns")
    pass
