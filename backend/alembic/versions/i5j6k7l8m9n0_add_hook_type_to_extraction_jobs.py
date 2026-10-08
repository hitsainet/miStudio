"""Add hook_type column to extraction_jobs table

Revision ID: i5j6k7l8m9n0
Revises: h4i5j6k7l8m9
Create Date: 2025-01-18

This migration adds a hook_type column to extraction_jobs to support
multi-hook SAE trainings. When a training job trains SAEs on multiple
hook types (e.g., residual and mlp), the user needs to select which
hook's SAE to extract features from.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'i5j6k7l8m9n0'
down_revision = 'h4i5j6k7l8m9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add hook_type column to extraction_jobs table
    # Nullable because existing extractions and single-hook trainings won't have this set
    op.add_column(
        'extraction_jobs',
        sa.Column('hook_type', sa.String(50), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('extraction_jobs', 'hook_type')
