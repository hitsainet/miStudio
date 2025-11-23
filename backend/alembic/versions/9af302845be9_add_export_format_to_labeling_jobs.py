"""add_export_format_to_labeling_jobs

Revision ID: 9af302845be9
Revises: cf2dc8a57416
Create Date: 2025-11-23 11:39:05.567161

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9af302845be9'
down_revision: Union[str, None] = 'cf2dc8a57416'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add export_format column to labeling_jobs table
    # Options: 'postman', 'curl', 'both'
    # Default: 'both' for backward compatibility
    op.add_column(
        'labeling_jobs',
        sa.Column('export_format', sa.String(20), nullable=False, server_default='both')
    )


def downgrade() -> None:
    # Remove export_format column from labeling_jobs table
    op.drop_column('labeling_jobs', 'export_format')
