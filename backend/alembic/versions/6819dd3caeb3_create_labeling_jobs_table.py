"""create_labeling_jobs_table

Revision ID: 6819dd3caeb3
Revises: 2cc2cd5c8052
Create Date: 2025-11-08 01:08:27.438941

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6819dd3caeb3'
down_revision: Union[str, None] = '2cc2cd5c8052'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create labeling_jobs table for independent semantic labeling of features."""
    # Create labeling_jobs table
    op.create_table(
        'labeling_jobs',
        # Primary key
        sa.Column('id', sa.String(length=255), nullable=False),

        # Foreign key to extraction_jobs
        sa.Column('extraction_job_id', sa.String(length=255), nullable=False),

        # Status and progress
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('progress', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('features_labeled', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_features', sa.Integer(), nullable=True),

        # Configuration
        sa.Column('labeling_method', sa.String(length=50), nullable=False),
        sa.Column('openai_model', sa.String(length=100), nullable=True),
        sa.Column('openai_api_key', sa.String(length=500), nullable=True),
        sa.Column('local_model', sa.String(length=100), nullable=True),

        # Metadata
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('statistics', sa.JSON(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),

        # Primary key constraint
        sa.PrimaryKeyConstraint('id'),

        # Foreign key constraint
        sa.ForeignKeyConstraint(['extraction_job_id'], ['extraction_jobs.id'], ondelete='CASCADE')
    )

    # Create indexes for efficient querying
    op.create_index(
        'idx_labeling_jobs_extraction_job_id',
        'labeling_jobs',
        ['extraction_job_id']
    )

    op.create_index(
        'idx_labeling_jobs_status',
        'labeling_jobs',
        ['status']
    )

    op.create_index(
        'idx_labeling_jobs_created_at',
        'labeling_jobs',
        ['created_at']
    )

    # Add labeling_job_id column to features table for tracking which labeling job created each label
    op.add_column(
        'features',
        sa.Column('labeling_job_id', sa.String(length=255), nullable=True)
    )

    # Add labeled_at timestamp to features table
    op.add_column(
        'features',
        sa.Column('labeled_at', sa.DateTime(timezone=True), nullable=True)
    )

    # Create index on labeling_job_id for faster lookups
    op.create_index(
        'idx_features_labeling_job_id',
        'features',
        ['labeling_job_id']
    )


def downgrade() -> None:
    """Remove labeling_jobs table and related columns."""
    # Drop indexes on features table
    op.drop_index('idx_features_labeling_job_id', table_name='features')

    # Drop columns from features table
    op.drop_column('features', 'labeled_at')
    op.drop_column('features', 'labeling_job_id')

    # Drop indexes on labeling_jobs table
    op.drop_index('idx_labeling_jobs_created_at', table_name='labeling_jobs')
    op.drop_index('idx_labeling_jobs_status', table_name='labeling_jobs')
    op.drop_index('idx_labeling_jobs_extraction_job_id', table_name='labeling_jobs')

    # Drop labeling_jobs table
    op.drop_table('labeling_jobs')
