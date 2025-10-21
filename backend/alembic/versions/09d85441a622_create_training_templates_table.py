"""create_training_templates_table

Revision ID: 09d85441a622
Revises: 456bdad91d81
Create Date: 2025-10-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '09d85441a622'
down_revision: Union[str, None] = '5523f486e7f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create training_templates table
    op.create_table(
        'training_templates',
        sa.Column('id', sa.UUID(), nullable=False, comment='Unique template identifier'),
        sa.Column('name', sa.String(length=255), nullable=False, comment='Template name'),
        sa.Column('description', sa.Text(), nullable=True, comment='Template description'),

        # Optional model and dataset references (templates can be generic or specific)
        sa.Column('model_id', sa.String(255), sa.ForeignKey('models.id', ondelete='SET NULL'), nullable=True, comment='Optional reference to specific model'),
        sa.Column('dataset_id', sa.String(255), nullable=True, comment='Optional reference to specific dataset (no FK due to type mismatch)'),

        # SAE Architecture type
        sa.Column('encoder_type', sa.String(length=20), nullable=False, comment='SAE architecture type (standard/skip/transcoder)'),

        # Training hyperparameters (stored as JSONB for flexibility)
        sa.Column('hyperparameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment='Complete training hyperparameters'),
        # Contains: hidden_dim, latent_dim, l1_alpha, learning_rate, batch_size,
        #           num_steps, warmup_steps, save_every, log_every, etc.

        # Template metadata
        sa.Column('is_favorite', sa.Boolean(), server_default=sa.text('false'), nullable=False, comment='Whether this template is marked as favorite'),
        sa.Column('extra_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='Additional metadata'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Record creation timestamp'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Record last update timestamp'),

        sa.PrimaryKeyConstraint('id'),

        # CHECK constraint to ensure encoder_type is valid
        sa.CheckConstraint(
            "encoder_type IN ('standard', 'skip', 'transcoder')",
            name='ck_training_templates_encoder_type'
        )
    )

    # Create indexes for efficient queries
    op.create_index('idx_training_templates_name', 'training_templates', ['name'])
    op.create_index('idx_training_templates_is_favorite', 'training_templates', ['is_favorite'])
    op.create_index('idx_training_templates_created_at', 'training_templates', ['created_at'])
    op.create_index('idx_training_templates_encoder_type', 'training_templates', ['encoder_type'])
    op.create_index('idx_training_templates_model_id', 'training_templates', ['model_id'])

    # GIN index for JSONB extra_metadata queries
    op.create_index(
        'idx_training_templates_extra_metadata_gin',
        'training_templates',
        ['extra_metadata'],
        postgresql_using='gin'
    )

    # GIN index for JSONB hyperparameters queries
    op.create_index(
        'idx_training_templates_hyperparameters_gin',
        'training_templates',
        ['hyperparameters'],
        postgresql_using='gin'
    )

    # Create trigger to auto-update updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_training_templates_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER trigger_update_training_templates_updated_at
        BEFORE UPDATE ON training_templates
        FOR EACH ROW
        EXECUTE FUNCTION update_training_templates_updated_at();
    """)


def downgrade() -> None:
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS trigger_update_training_templates_updated_at ON training_templates;")
    op.execute("DROP FUNCTION IF EXISTS update_training_templates_updated_at();")

    # Drop indexes
    op.drop_index('idx_training_templates_hyperparameters_gin', table_name='training_templates', postgresql_using='gin')
    op.drop_index('idx_training_templates_extra_metadata_gin', table_name='training_templates', postgresql_using='gin')
    op.drop_index('idx_training_templates_model_id', table_name='training_templates')
    op.drop_index('idx_training_templates_encoder_type', table_name='training_templates')
    op.drop_index('idx_training_templates_created_at', table_name='training_templates')
    op.drop_index('idx_training_templates_is_favorite', table_name='training_templates')
    op.drop_index('idx_training_templates_name', table_name='training_templates')

    # Drop table
    op.drop_table('training_templates')
