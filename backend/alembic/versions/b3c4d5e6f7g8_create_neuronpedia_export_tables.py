"""create_neuronpedia_export_tables

Revision ID: b3c4d5e6f7g8
Revises: 7aa4e03d0d11
Create Date: 2025-12-04 12:00:00

Creates tables for Neuronpedia export functionality:
- neuronpedia_export_jobs: Tracks export job status and progress
- feature_dashboard_data: Stores computed dashboard data (logit lens, histograms, top tokens)

NOTE: This migration is idempotent - it uses IF NOT EXISTS to safely re-run
if tables were created manually or partially applied.
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
    # Create export_status enum type (idempotent)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'export_status') THEN
                CREATE TYPE export_status AS ENUM (
                    'pending', 'computing', 'packaging', 'completed', 'failed', 'cancelled'
                );
            END IF;
        END
        $$;
    """)

    # Create neuronpedia_export_jobs table (idempotent)
    op.execute("""
        CREATE TABLE IF NOT EXISTS neuronpedia_export_jobs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            sae_id VARCHAR(255) NOT NULL,
            source_type VARCHAR(50) NOT NULL,
            config JSONB NOT NULL DEFAULT '{}',
            status export_status NOT NULL DEFAULT 'pending',
            progress FLOAT NOT NULL DEFAULT 0.0,
            current_stage VARCHAR(100),
            output_path TEXT,
            file_size_bytes BIGINT,
            feature_count INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            error_message TEXT,
            error_details JSONB
        )
    """)

    # Create indexes for common queries (idempotent)
    op.execute("CREATE INDEX IF NOT EXISTS idx_export_jobs_sae_id ON neuronpedia_export_jobs(sae_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_export_jobs_status ON neuronpedia_export_jobs(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_export_jobs_created_at ON neuronpedia_export_jobs(created_at)")

    # Create feature_dashboard_data table (idempotent)
    op.execute("""
        CREATE TABLE IF NOT EXISTS feature_dashboard_data (
            id BIGSERIAL PRIMARY KEY,
            feature_id VARCHAR(255) NOT NULL REFERENCES features(id) ON DELETE CASCADE,
            logit_lens_data JSONB,
            histogram_data JSONB,
            top_tokens JSONB,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            computation_version VARCHAR(50),
            CONSTRAINT uq_feature_dashboard_feature_id UNIQUE (feature_id)
        )
    """)

    # Create index for feature lookup (idempotent)
    op.execute("CREATE INDEX IF NOT EXISTS idx_feature_dashboard_feature_id ON feature_dashboard_data(feature_id)")


def downgrade() -> None:
    # Drop feature_dashboard_data
    op.execute("DROP INDEX IF EXISTS idx_feature_dashboard_feature_id")
    op.execute("DROP TABLE IF EXISTS feature_dashboard_data")

    # Drop neuronpedia_export_jobs
    op.execute("DROP INDEX IF EXISTS idx_export_jobs_created_at")
    op.execute("DROP INDEX IF EXISTS idx_export_jobs_status")
    op.execute("DROP INDEX IF EXISTS idx_export_jobs_sae_id")
    op.execute("DROP TABLE IF EXISTS neuronpedia_export_jobs")

    # Drop enum type
    op.execute("DROP TYPE IF EXISTS export_status")
