"""create_neuronpedia_export_tables

Revision ID: b3c4d5e6f7g8
Revises: 7aa4e03d0d11
Create Date: 2025-12-04 12:00:00

Creates tables for Neuronpedia export functionality:
- neuronpedia_export_jobs: Tracks export job status and progress
- feature_dashboard_data: Stores computed dashboard data (logit lens, histograms, top tokens)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b3c4d5e6f7g8'
down_revision: Union[str, None] = '7aa4e03d0d11'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create export_status enum type
    op.execute("""
        CREATE TYPE export_status AS ENUM (
            'pending', 'computing', 'packaging', 'completed', 'failed', 'cancelled'
        )
    """)

    # Create neuronpedia_export_jobs table
    op.create_table(
        'neuronpedia_export_jobs',
        # Primary key - UUID
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),

        # SAE reference - can be external_sae or training
        sa.Column('sae_id', sa.String(length=255), nullable=False),
        sa.Column('source_type', sa.String(length=50), nullable=False),  # 'external_sae' or 'training'

        # Job configuration stored as JSONB
        # Contains: format, feature_selection, include_logit_lens, include_histograms, etc.
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),

        # Status tracking
        sa.Column('status', postgresql.ENUM('pending', 'computing', 'packaging', 'completed', 'failed', 'cancelled', name='export_status', create_type=False), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('current_stage', sa.String(length=100), nullable=True),  # Human-readable stage description

        # Results
        sa.Column('output_path', sa.Text(), nullable=True),  # Path to generated ZIP archive
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('feature_count', sa.Integer(), nullable=True),

        # Timing
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),

        # Error handling
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),

        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for common queries
    op.create_index('idx_export_jobs_sae_id', 'neuronpedia_export_jobs', ['sae_id'])
    op.create_index('idx_export_jobs_status', 'neuronpedia_export_jobs', ['status'])
    op.create_index('idx_export_jobs_created_at', 'neuronpedia_export_jobs', ['created_at'])

    # Create feature_dashboard_data table
    op.create_table(
        'feature_dashboard_data',
        # Primary key - auto-incrementing BIGINT
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),

        # Feature reference - links to features table
        sa.Column('feature_id', sa.String(length=255), sa.ForeignKey('features.id', ondelete='CASCADE'), nullable=False),

        # Logit lens data - top tokens influenced by this feature
        # Format: {"top_positive": [{"token": "...", "logit": 1.23}, ...], "top_negative": [...]}
        sa.Column('logit_lens_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),

        # Histogram data - activation distribution
        # Format: {"bin_edges": [...], "counts": [...], "total_count": N, "nonzero_count": N}
        sa.Column('histogram_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),

        # Top tokens aggregated across all examples
        # Format: [{"token": "...", "total_activation": 12.3, "count": 5, "mean_activation": 2.46, "max_activation": 4.5}, ...]
        sa.Column('top_tokens', postgresql.JSONB(astext_type=sa.Text()), nullable=True),

        # Computation metadata
        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('computation_version', sa.String(length=50), nullable=True),  # Version of computation algorithm

        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('feature_id', name='uq_feature_dashboard_feature_id')
    )

    # Create index for feature lookup
    op.create_index('idx_feature_dashboard_feature_id', 'feature_dashboard_data', ['feature_id'])


def downgrade() -> None:
    # Drop feature_dashboard_data
    op.drop_index('idx_feature_dashboard_feature_id', table_name='feature_dashboard_data')
    op.drop_table('feature_dashboard_data')

    # Drop neuronpedia_export_jobs
    op.drop_index('idx_export_jobs_created_at', table_name='neuronpedia_export_jobs')
    op.drop_index('idx_export_jobs_status', table_name='neuronpedia_export_jobs')
    op.drop_index('idx_export_jobs_sae_id', table_name='neuronpedia_export_jobs')
    op.drop_table('neuronpedia_export_jobs')

    # Drop enum type
    op.execute("DROP TYPE export_status")
