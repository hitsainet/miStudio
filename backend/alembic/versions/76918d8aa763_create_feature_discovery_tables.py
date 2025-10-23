"""create_feature_discovery_tables

Revision ID: 76918d8aa763
Revises: 4665467878d9
Create Date: 2025-10-23 06:20:29.430917

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '76918d8aa763'
down_revision: Union[str, None] = '4665467878d9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum for extraction status
    op.execute("CREATE TYPE extraction_status_enum AS ENUM ('queued', 'extracting', 'completed', 'failed', 'cancelled')")

    # Create enum for label source
    op.execute("CREATE TYPE label_source_enum AS ENUM ('auto', 'user')")

    # Create enum for analysis type
    op.execute("CREATE TYPE analysis_type_enum AS ENUM ('logit_lens', 'correlations', 'ablation')")

    # ========================================
    # Table 1: extraction_jobs
    # ========================================
    op.create_table(
        'extraction_jobs',
        sa.Column('id', sa.String(255), primary_key=True, comment='Extraction job ID (format: ext_{training_id}_{timestamp})'),
        sa.Column('training_id', sa.String(255), sa.ForeignKey('trainings.id', ondelete='CASCADE'), nullable=False, comment='Reference to completed training'),
        sa.Column('celery_task_id', sa.String(255), nullable=True, comment='Celery task ID for tracking'),

        # Extraction configuration
        sa.Column('config', postgresql.JSONB, nullable=False, comment='Extraction config: {evaluation_samples, top_k_examples}'),

        # Processing status
        sa.Column('status', postgresql.ENUM('queued', 'extracting', 'completed', 'failed', 'cancelled', name='extraction_status_enum', create_type=False), nullable=False, server_default='queued', comment='Extraction status'),
        sa.Column('progress', sa.Float, nullable=True, server_default='0.0', comment='Extraction progress (0-100)'),
        sa.Column('features_extracted', sa.Integer, nullable=True, server_default='0', comment='Number of features extracted so far'),
        sa.Column('total_features', sa.Integer, nullable=True, comment='Total number of features to extract'),
        sa.Column('error_message', sa.Text, nullable=True, comment='Error message if status is failed'),

        # Output statistics
        sa.Column('statistics', postgresql.JSONB, nullable=True, comment='Extraction statistics: {total_features, avg_interpretability, avg_activation_freq, interpretable_count}'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )

    # Indexes for extraction_jobs
    op.create_index('idx_extraction_jobs_training_id', 'extraction_jobs', ['training_id'])
    op.create_index('idx_extraction_jobs_status', 'extraction_jobs', ['status'])
    op.create_index('idx_extraction_jobs_created_at', 'extraction_jobs', ['created_at'])

    # Unique constraint: only one active extraction per training (using partial unique index)
    op.execute("""
        CREATE UNIQUE INDEX uq_extraction_jobs_active_training
        ON extraction_jobs (training_id)
        WHERE status IN ('queued', 'extracting')
    """)

    # ========================================
    # Table 2: features
    # ========================================
    op.create_table(
        'features',
        sa.Column('id', sa.String(255), primary_key=True, comment='Feature ID (format: feat_{training_id}_{neuron_index})'),
        sa.Column('training_id', sa.String(255), sa.ForeignKey('trainings.id', ondelete='CASCADE'), nullable=False, comment='Reference to training'),
        sa.Column('extraction_job_id', sa.String(255), sa.ForeignKey('extraction_jobs.id', ondelete='CASCADE'), nullable=False, comment='Reference to extraction job'),
        sa.Column('neuron_index', sa.Integer, nullable=False, comment='SAE neuron index (0 to d_sae-1)'),

        # Feature identification
        sa.Column('name', sa.String(500), nullable=False, comment='Feature name/label (auto-generated or user-edited)'),
        sa.Column('description', sa.Text, nullable=True, comment='Feature description'),
        sa.Column('label_source', postgresql.ENUM('auto', 'user', name='label_source_enum', create_type=False), nullable=False, server_default='auto', comment='Label source: auto (heuristic) or user (edited)'),

        # Feature statistics
        sa.Column('activation_frequency', sa.Float, nullable=False, comment='Fraction of samples where feature activates (>0.01)'),
        sa.Column('interpretability_score', sa.Float, nullable=False, comment='Interpretability score (0.0-1.0) based on consistency and sparsity'),
        sa.Column('max_activation', sa.Float, nullable=False, comment='Maximum activation value across all samples'),
        sa.Column('mean_activation', sa.Float, nullable=True, comment='Mean activation value when active'),

        # User metadata
        sa.Column('is_favorite', sa.Boolean, nullable=False, server_default='false', comment='User favorited flag'),
        sa.Column('notes', sa.Text, nullable=True, comment='User notes about feature'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Indexes for features
    op.create_index('idx_features_training_id', 'features', ['training_id'])
    op.create_index('idx_features_extraction_job_id', 'features', ['extraction_job_id'])
    op.create_index('idx_features_activation_freq', 'features', ['activation_frequency'])
    op.create_index('idx_features_interpretability', 'features', ['interpretability_score'])
    op.create_index('idx_features_favorite', 'features', ['is_favorite'])
    op.create_index('idx_features_neuron_index', 'features', ['neuron_index'])

    # GIN index for full-text search on name and description
    op.execute("""
        CREATE INDEX idx_features_fulltext_search
        ON features
        USING GIN (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, '')))
    """)

    # Unique constraint: one feature per (training_id, neuron_index)
    op.create_unique_constraint(
        'uq_features_training_neuron',
        'features',
        ['training_id', 'neuron_index']
    )

    # ========================================
    # Table 3: feature_activations (partitioned)
    # ========================================
    # Create parent table with partitioning by feature_id range
    op.execute("""
        CREATE TABLE feature_activations (
            id BIGSERIAL,
            feature_id VARCHAR(255) NOT NULL,
            sample_index INTEGER NOT NULL,
            max_activation FLOAT NOT NULL,
            tokens JSONB NOT NULL,
            activations JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            PRIMARY KEY (id, feature_id)
        ) PARTITION BY RANGE (feature_id);
    """)

    # Add foreign key to features table
    op.execute("""
        ALTER TABLE feature_activations
        ADD CONSTRAINT fk_feature_activations_feature_id
        FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE;
    """)

    # Create partitions for feature_id ranges (1000 features per partition)
    # This allows efficient querying since queries typically filter by feature_id
    # We'll create 16 partitions initially (supports up to 16,000 features)
    for i in range(16):
        start_range = f"feat_{i * 1000:05d}"
        end_range = f"feat_{(i + 1) * 1000:05d}"
        partition_name = f"feature_activations_p{i:02d}"

        op.execute(f"""
            CREATE TABLE {partition_name}
            PARTITION OF feature_activations
            FOR VALUES FROM ('{start_range}') TO ('{end_range}');
        """)

        # Create indexes on each partition
        op.execute(f"""
            CREATE INDEX idx_{partition_name}_feature_id
            ON {partition_name} (feature_id);
        """)

        op.execute(f"""
            CREATE INDEX idx_{partition_name}_max_activation
            ON {partition_name} (max_activation DESC);
        """)

    # ========================================
    # Table 4: feature_analysis_cache
    # ========================================
    op.create_table(
        'feature_analysis_cache',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('feature_id', sa.String(255), sa.ForeignKey('features.id', ondelete='CASCADE'), nullable=False, comment='Reference to feature'),
        sa.Column('analysis_type', postgresql.ENUM('logit_lens', 'correlations', 'ablation', name='analysis_type_enum', create_type=False), nullable=False, comment='Type of analysis'),

        # Analysis results
        sa.Column('result', postgresql.JSONB, nullable=False, comment='Analysis result (structure depends on analysis_type)'),

        # Cache metadata
        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False, comment='Cache expiration (7 days from computed_at)'),
    )

    # Indexes for feature_analysis_cache
    op.create_index('idx_feature_analysis_cache_feature_id', 'feature_analysis_cache', ['feature_id'])
    op.create_index('idx_feature_analysis_cache_analysis_type', 'feature_analysis_cache', ['analysis_type'])
    op.create_index('idx_feature_analysis_cache_expires_at', 'feature_analysis_cache', ['expires_at'])

    # Unique constraint: one cached result per (feature_id, analysis_type)
    op.create_unique_constraint(
        'uq_feature_analysis_cache_feature_type',
        'feature_analysis_cache',
        ['feature_id', 'analysis_type']
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('feature_analysis_cache')

    # Drop feature_activations partitions
    for i in range(16):
        partition_name = f"feature_activations_p{i:02d}"
        op.execute(f"DROP TABLE IF EXISTS {partition_name};")

    # Drop parent table
    op.execute("DROP TABLE IF EXISTS feature_activations;")

    op.drop_table('features')

    # Drop partial unique index before dropping table
    op.execute("DROP INDEX IF EXISTS uq_extraction_jobs_active_training;")
    op.drop_table('extraction_jobs')

    # Drop enums
    op.execute("DROP TYPE IF EXISTS analysis_type_enum;")
    op.execute("DROP TYPE IF EXISTS label_source_enum;")
    op.execute("DROP TYPE IF EXISTS extraction_status_enum;")
