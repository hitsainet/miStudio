"""create_external_saes_table

Revision ID: a1c2e3f4g5h6
Revises: 678f7e8bbeb6
Create Date: 2025-11-26 10:00:00

Creates the external_saes table for managing SAEs downloaded from HuggingFace
or converted from local training. Supports the SAEs tab functionality for
downloading, uploading, and managing SAEs for steering.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1c2e3f4g5h6'
down_revision: Union[str, None] = '678f7e8bbeb6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create external_saes table
    op.create_table(
        'external_saes',
        # Primary key
        sa.Column('id', sa.String(length=255), nullable=False),

        # Display info
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),

        # Source type: huggingface, local, trained
        sa.Column('source', sa.String(length=50), nullable=False),

        # Status: pending, downloading, converting, ready, error, deleted
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),

        # HuggingFace source fields
        sa.Column('hf_repo_id', sa.String(length=500), nullable=True),
        sa.Column('hf_filepath', sa.String(length=1000), nullable=True),
        sa.Column('hf_revision', sa.String(length=255), nullable=True),

        # Trained source reference
        sa.Column('training_id', sa.String(length=255), nullable=True),

        # Model compatibility
        sa.Column('model_name', sa.String(length=255), nullable=True),
        sa.Column('model_id', sa.String(length=255), nullable=True),

        # SAE architecture info
        sa.Column('layer', sa.Integer(), nullable=True),
        sa.Column('n_features', sa.Integer(), nullable=True),
        sa.Column('d_model', sa.Integer(), nullable=True),
        sa.Column('architecture', sa.String(length=100), nullable=True),  # standard, gated, etc.

        # Format info
        sa.Column('format', sa.String(length=50), nullable=False, server_default='saelens'),

        # Local storage
        sa.Column('local_path', sa.String(length=1000), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),

        # Download/upload progress
        sa.Column('progress', sa.Float(), nullable=False, server_default='0.0'),

        # Error handling
        sa.Column('error_message', sa.Text(), nullable=True),

        # Flexible metadata (activation stats, neuronpedia links, etc.)
        # Named sae_metadata to avoid SQLAlchemy reserved name conflict
        sa.Column('sae_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('downloaded_at', sa.DateTime(timezone=True), nullable=True),

        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for common queries
    op.create_index('idx_external_saes_status', 'external_saes', ['status'])
    op.create_index('idx_external_saes_source', 'external_saes', ['source'])
    op.create_index('idx_external_saes_model_name', 'external_saes', ['model_name'])
    op.create_index('idx_external_saes_hf_repo_id', 'external_saes', ['hf_repo_id'])
    op.create_index('idx_external_saes_training_id', 'external_saes', ['training_id'])
    op.create_index('idx_external_saes_created_at', 'external_saes', ['created_at'])

    # Add foreign key constraints
    # FK to trainings table (for SAEs exported from training)
    op.create_foreign_key(
        'fk_external_saes_training_id',
        'external_saes',
        'trainings',
        ['training_id'],
        ['id'],
        ondelete='SET NULL'
    )

    # FK to models table (for model compatibility tracking)
    op.create_foreign_key(
        'fk_external_saes_model_id',
        'external_saes',
        'models',
        ['model_id'],
        ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    # Drop foreign keys
    op.drop_constraint('fk_external_saes_model_id', 'external_saes', type_='foreignkey')
    op.drop_constraint('fk_external_saes_training_id', 'external_saes', type_='foreignkey')

    # Drop indexes
    op.drop_index('idx_external_saes_created_at', table_name='external_saes')
    op.drop_index('idx_external_saes_training_id', table_name='external_saes')
    op.drop_index('idx_external_saes_hf_repo_id', table_name='external_saes')
    op.drop_index('idx_external_saes_model_name', table_name='external_saes')
    op.drop_index('idx_external_saes_source', table_name='external_saes')
    op.drop_index('idx_external_saes_status', table_name='external_saes')

    # Drop table
    op.drop_table('external_saes')
