"""add_external_sae_id_to_extraction_and_features

Revision ID: 79dd2387ebeb
Revises: a04bee8d5640
Create Date: 2025-11-28 15:23:18.486198

This migration adds support for feature extraction from external SAEs
(downloaded from HuggingFace) in addition to miStudio-trained SAEs.

Changes:
- Add external_sae_id FK to extraction_jobs table
- Add external_sae_id FK to features table
- Make training_id nullable in both tables
- Add CHECK constraint ensuring exactly one source is set
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '79dd2387ebeb'
down_revision: Union[str, None] = 'a04bee8d5640'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === EXTRACTION_JOBS TABLE ===

    # Add external_sae_id column
    op.add_column(
        'extraction_jobs',
        sa.Column('external_sae_id', sa.String(255), nullable=True)
    )

    # Add foreign key constraint
    op.create_foreign_key(
        'fk_extraction_jobs_external_sae_id',
        'extraction_jobs',
        'external_saes',
        ['external_sae_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # Create index for external_sae_id
    op.create_index(
        'ix_extraction_jobs_external_sae_id',
        'extraction_jobs',
        ['external_sae_id']
    )

    # Make training_id nullable
    op.alter_column(
        'extraction_jobs',
        'training_id',
        existing_type=sa.String(255),
        nullable=True
    )

    # Add CHECK constraint: exactly one of training_id or external_sae_id must be set
    op.execute("""
        ALTER TABLE extraction_jobs
        ADD CONSTRAINT check_extraction_single_source
        CHECK (
            (training_id IS NOT NULL AND external_sae_id IS NULL) OR
            (training_id IS NULL AND external_sae_id IS NOT NULL)
        )
    """)

    # === FEATURES TABLE ===

    # Add external_sae_id column
    op.add_column(
        'features',
        sa.Column('external_sae_id', sa.String(255), nullable=True)
    )

    # Add foreign key constraint
    op.create_foreign_key(
        'fk_features_external_sae_id',
        'features',
        'external_saes',
        ['external_sae_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # Create index for external_sae_id
    op.create_index(
        'ix_features_external_sae_id',
        'features',
        ['external_sae_id']
    )

    # Make training_id nullable
    op.alter_column(
        'features',
        'training_id',
        existing_type=sa.String(255),
        nullable=True
    )

    # Add CHECK constraint: exactly one of training_id or external_sae_id must be set
    op.execute("""
        ALTER TABLE features
        ADD CONSTRAINT check_feature_single_source
        CHECK (
            (training_id IS NOT NULL AND external_sae_id IS NULL) OR
            (training_id IS NULL AND external_sae_id IS NOT NULL)
        )
    """)


def downgrade() -> None:
    # === FEATURES TABLE ===

    # Remove CHECK constraint
    op.execute("ALTER TABLE features DROP CONSTRAINT IF EXISTS check_feature_single_source")

    # Make training_id NOT NULL again (may fail if null values exist)
    op.alter_column(
        'features',
        'training_id',
        existing_type=sa.String(255),
        nullable=False
    )

    # Drop index
    op.drop_index('ix_features_external_sae_id', table_name='features')

    # Drop foreign key
    op.drop_constraint('fk_features_external_sae_id', 'features', type_='foreignkey')

    # Drop column
    op.drop_column('features', 'external_sae_id')

    # === EXTRACTION_JOBS TABLE ===

    # Remove CHECK constraint
    op.execute("ALTER TABLE extraction_jobs DROP CONSTRAINT IF EXISTS check_extraction_single_source")

    # Make training_id NOT NULL again (may fail if null values exist)
    op.alter_column(
        'extraction_jobs',
        'training_id',
        existing_type=sa.String(255),
        nullable=False
    )

    # Drop index
    op.drop_index('ix_extraction_jobs_external_sae_id', table_name='extraction_jobs')

    # Drop foreign key
    op.drop_constraint('fk_extraction_jobs_external_sae_id', 'extraction_jobs', type_='foreignkey')

    # Drop column
    op.drop_column('extraction_jobs', 'external_sae_id')
