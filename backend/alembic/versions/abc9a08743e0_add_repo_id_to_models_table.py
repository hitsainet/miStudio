"""add repo_id to models table

Revision ID: abc9a08743e0
Revises: c8c7653233ee
Create Date: 2025-10-12 01:31:35.294372

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'abc9a08743e0'
down_revision: Union[str, None] = 'c8c7653233ee'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add repo_id column to models table
    op.add_column('models', sa.Column('repo_id', sa.String(length=500), nullable=True))


def downgrade() -> None:
    # Remove repo_id column from models table
    op.drop_column('models', 'repo_id')
