"""add extraction filter fields

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2025-11-10 00:00:01.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add extraction filter fields to extraction_jobs table (if it exists)
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if 'extraction_jobs' in inspector.get_table_names():
        op.add_column('extraction_jobs', sa.Column('extraction_filter_enabled', sa.Boolean(), nullable=False, server_default='false'))
        op.add_column('extraction_jobs', sa.Column('extraction_filter_mode', sa.String(20), nullable=False, server_default='standard'))


def downgrade() -> None:
    # Drop columns if table exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if 'extraction_jobs' in inspector.get_table_names():
        op.drop_column('extraction_jobs', 'extraction_filter_mode')
        op.drop_column('extraction_jobs', 'extraction_filter_enabled')
