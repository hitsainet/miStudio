"""add_nlp_analysis_persistence_columns

Revision ID: 7d3539aaf247
Revises: f754069a4163
Create Date: 2025-12-14 08:00:00.000000

Add columns for NLP analysis persistence:
- features.nlp_analysis (JSONB) - stores computed NLP analysis results
- features.nlp_processed_at (TIMESTAMP) - when NLP was last processed
- extraction_jobs.nlp_status (VARCHAR) - NLP processing status
- extraction_jobs.nlp_progress (FLOAT) - NLP processing progress 0.0-1.0
- extraction_jobs.nlp_processed_count (INTEGER) - features with NLP completed
- extraction_jobs.nlp_error_message (TEXT) - error details if failed
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '7d3539aaf247'
down_revision: Union[str, None] = 'f754069a4163'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add NLP analysis persistence columns to features and extraction_jobs tables."""

    # Add NLP analysis columns to features table
    op.add_column(
        'features',
        sa.Column('nlp_analysis', postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    )
    op.add_column(
        'features',
        sa.Column('nlp_processed_at', sa.DateTime(timezone=True), nullable=True)
    )

    # Add NLP status columns to extraction_jobs table
    op.add_column(
        'extraction_jobs',
        sa.Column('nlp_status', sa.String(50), nullable=True)
    )
    op.add_column(
        'extraction_jobs',
        sa.Column('nlp_progress', sa.Float(), nullable=True, server_default='0.0')
    )
    op.add_column(
        'extraction_jobs',
        sa.Column('nlp_processed_count', sa.Integer(), nullable=True, server_default='0')
    )
    op.add_column(
        'extraction_jobs',
        sa.Column('nlp_error_message', sa.Text(), nullable=True)
    )

    # Create index for efficient queries on NLP status
    op.create_index(
        'idx_extraction_jobs_nlp_status',
        'extraction_jobs',
        ['nlp_status'],
        unique=False
    )

    # Create index for features with NLP analysis (for efficient lookups)
    op.create_index(
        'idx_features_nlp_processed',
        'features',
        ['nlp_processed_at'],
        unique=False,
        postgresql_where=sa.text('nlp_processed_at IS NOT NULL')
    )


def downgrade() -> None:
    """Remove NLP analysis persistence columns."""

    # Drop indexes
    op.drop_index('idx_features_nlp_processed', table_name='features')
    op.drop_index('idx_extraction_jobs_nlp_status', table_name='extraction_jobs')

    # Remove columns from extraction_jobs
    op.drop_column('extraction_jobs', 'nlp_error_message')
    op.drop_column('extraction_jobs', 'nlp_processed_count')
    op.drop_column('extraction_jobs', 'nlp_progress')
    op.drop_column('extraction_jobs', 'nlp_status')

    # Remove columns from features
    op.drop_column('features', 'nlp_processed_at')
    op.drop_column('features', 'nlp_analysis')
