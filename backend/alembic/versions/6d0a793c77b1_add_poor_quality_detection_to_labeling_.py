"""add_poor_quality_detection_to_labeling_jobs

Revision ID: 6d0a793c77b1
Revises: 9af302845be9
Create Date: 2025-11-23 14:25:35.277527

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6d0a793c77b1'
down_revision: Union[str, None] = '9af302845be9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add poor quality detection columns to labeling_jobs table
    # save_poor_quality_labels: Enable saving debug files for poor quality labels (default: false)
    # poor_quality_sample_rate: Sample rate for saving poor quality labels, 0.0-1.0 (default: 1.0 = save all)
    op.add_column(
        'labeling_jobs',
        sa.Column('save_poor_quality_labels', sa.Boolean(), nullable=False, server_default='false')
    )
    op.add_column(
        'labeling_jobs',
        sa.Column('poor_quality_sample_rate', sa.Float(), nullable=False, server_default='1.0')
    )


def downgrade() -> None:
    # Remove poor quality detection columns from labeling_jobs table
    op.drop_column('labeling_jobs', 'poor_quality_sample_rate')
    op.drop_column('labeling_jobs', 'save_poor_quality_labels')
