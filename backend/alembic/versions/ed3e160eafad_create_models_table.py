"""create models table

Revision ID: ed3e160eafad
Revises: 118f85d483dd
Create Date: 2025-10-08 06:05:44.672752

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'ed3e160eafad'
down_revision: Union[str, None] = '118f85d483dd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create models table
    op.create_table('models',
        sa.Column('id', sa.UUID(), nullable=False, comment='Unique model identifier'),
        sa.Column('name', sa.String(length=255), nullable=False, comment='Model name'),
        sa.Column('repo_id', sa.String(length=255), nullable=True, comment='HuggingFace repository ID'),
        sa.Column('architecture', sa.String(length=100), nullable=False, comment='Model architecture (e.g., GPT-2, LLaMA)'),
        sa.Column('params_count', sa.BigInteger(), nullable=True, comment='Number of parameters'),
        sa.Column('quantization', sa.String(length=20), nullable=False, comment='Quantization type (FP32, FP16, INT8, INT4)'),
        sa.Column('memory_req_bytes', sa.BigInteger(), nullable=True, comment='Estimated memory requirement in bytes'),
        sa.Column('status', sa.Enum('downloading', 'loading', 'ready', 'error', name='model_status_enum'), nullable=False, comment='Current processing status'),
        sa.Column('progress', sa.Float(), nullable=True, comment='Download/loading progress (0-100)'),
        sa.Column('error_message', sa.Text(), nullable=True, comment='Error message if status is ERROR'),
        sa.Column('file_path', sa.String(length=512), nullable=True, comment='Path to model files'),
        sa.Column('num_layers', sa.Integer(), nullable=True, comment='Number of layers'),
        sa.Column('hidden_dim', sa.Integer(), nullable=True, comment='Hidden dimension size'),
        sa.Column('num_heads', sa.Integer(), nullable=True, comment='Number of attention heads'),
        sa.Column('extra_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment='Additional metadata'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Record creation timestamp'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Record last update timestamp'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('idx_models_status', 'models', ['status'], unique=False)
    op.create_index('idx_models_architecture', 'models', ['architecture'], unique=False)
    op.create_index('idx_models_created_at', 'models', ['created_at'], unique=False)
    # GIN index for JSONB extra_metadata queries
    op.create_index('idx_models_metadata_gin', 'models', ['extra_metadata'], unique=False, postgresql_using='gin')


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_models_metadata_gin', table_name='models', postgresql_using='gin')
    op.drop_index('idx_models_created_at', table_name='models')
    op.drop_index('idx_models_architecture', table_name='models')
    op.drop_index('idx_models_status', table_name='models')

    # Drop table
    op.drop_table('models')

    # Drop enum type
    op.execute('DROP TYPE model_status_enum')
