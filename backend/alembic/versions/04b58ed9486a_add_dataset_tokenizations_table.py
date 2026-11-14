"""add_dataset_tokenizations_table

Revision ID: 04b58ed9486a
Revises: de2e3ad17dc1
Create Date: 2025-11-08 18:53:54.249828

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '04b58ed9486a'
down_revision: Union[str, None] = 'de2e3ad17dc1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create dataset_tokenizations table
    # Note: sa.Enum will automatically create the enum type if it doesn't exist
    op.create_table(
        'dataset_tokenizations',
        sa.Column('id', sa.String(length=255), nullable=False, comment='Unique tokenization identifier (format: tok_{dataset_id}_{model_id})'),
        sa.Column('dataset_id', sa.UUID(), nullable=False, comment='Parent dataset ID'),
        sa.Column('model_id', sa.String(length=255), nullable=False, comment='Model whose tokenizer was used'),
        sa.Column('tokenized_path', sa.String(length=512), nullable=True, comment='Path to tokenized dataset (Arrow format)'),
        sa.Column('tokenizer_repo_id', sa.String(length=255), nullable=False, comment='HuggingFace tokenizer repository ID'),
        sa.Column('vocab_size', sa.Integer(), nullable=True, comment='Vocabulary size for this tokenization'),
        sa.Column('num_tokens', sa.BigInteger(), nullable=True, comment='Total number of tokens in tokenized dataset'),
        sa.Column('avg_seq_length', sa.Float(), nullable=True, comment='Average sequence length in tokens'),
        sa.Column('status', sa.Enum('QUEUED', 'PROCESSING', 'READY', 'ERROR', name='tokenization_status_enum', create_type=True), nullable=False, server_default='QUEUED', comment='Current tokenization status'),
        sa.Column('progress', sa.Float(), nullable=True, server_default='0.0', comment='Tokenization progress (0-100)'),
        sa.Column('error_message', sa.Text(), nullable=True, comment='Error message if status is ERROR'),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True, comment='Celery task ID for async tokenization'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Record creation timestamp'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Record last update timestamp'),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True, comment='Timestamp when tokenization completed'),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['model_id'], ['models.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('dataset_id', 'model_id', name='uq_dataset_model_tokenization')
    )

    # Create indexes
    op.create_index('idx_tokenizations_created_at', 'dataset_tokenizations', ['created_at'], unique=False)
    op.create_index('idx_tokenizations_dataset_id', 'dataset_tokenizations', ['dataset_id'], unique=False)
    op.create_index('idx_tokenizations_model_id', 'dataset_tokenizations', ['model_id'], unique=False)
    op.create_index('idx_tokenizations_status', 'dataset_tokenizations', ['status'], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_tokenizations_status', table_name='dataset_tokenizations')
    op.drop_index('idx_tokenizations_model_id', table_name='dataset_tokenizations')
    op.drop_index('idx_tokenizations_dataset_id', table_name='dataset_tokenizations')
    op.drop_index('idx_tokenizations_created_at', table_name='dataset_tokenizations')

    # Drop table (this will also drop the enum type if no other tables use it)
    op.drop_table('dataset_tokenizations')

    # Manually drop enum type to ensure cleanup
    op.execute("DROP TYPE IF EXISTS tokenization_status_enum")
