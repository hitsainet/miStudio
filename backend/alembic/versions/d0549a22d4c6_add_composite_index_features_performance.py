"""add_composite_index_features_performance

Revision ID: d0549a22d4c6
Revises: l8m9n0o1p2q3
Create Date: 2026-02-26 03:18:45.968317

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd0549a22d4c6'
down_revision: Union[str, None] = 'l8m9n0o1p2q3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Composite index for the most common query pattern:
    # SELECT * FROM features WHERE extraction_job_id = ? ORDER BY activation_frequency DESC
    # This replaces two separate index scans with one efficient index scan.
    op.create_index(
        'idx_features_extjob_actfreq',
        'features',
        ['extraction_job_id', sa.text('activation_frequency DESC')],
        if_not_exists=True
    )


def downgrade() -> None:
    op.drop_index('idx_features_extjob_actfreq', table_name='features')
