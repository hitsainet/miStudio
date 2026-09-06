"""add save_requests_sample_rate to labeling_jobs

Revision ID: 92326e3cf5a7
Revises: 6d0a793c77b1
Create Date: 2025-11-23 21:42:19.913375

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '92326e3cf5a7'
down_revision: Union[str, None] = '6d0a793c77b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add save_requests_sample_rate column to labeling_jobs table
    op.add_column(
        'labeling_jobs',
        sa.Column('save_requests_sample_rate', sa.Float(), nullable=False, server_default='1.0')
    )


def downgrade() -> None:
    # Remove save_requests_sample_rate column from labeling_jobs table
    op.drop_column('labeling_jobs', 'save_requests_sample_rate')
