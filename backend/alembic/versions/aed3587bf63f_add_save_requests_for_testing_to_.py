"""add_save_requests_for_testing_to_labeling_jobs

Revision ID: aed3587bf63f
Revises: c351738faecf
Create Date: 2025-11-17 11:06:57.297586

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aed3587bf63f'
down_revision: Union[str, None] = 'c351738faecf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add save_requests_for_testing column to labeling_jobs table
    op.add_column('labeling_jobs', sa.Column('save_requests_for_testing', sa.Boolean(), nullable=False, server_default='false'))


def downgrade() -> None:
    # Remove save_requests_for_testing column from labeling_jobs table
    op.drop_column('labeling_jobs', 'save_requests_for_testing')
