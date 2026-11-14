"""add_openai_compatible_fields

Revision ID: 7d94bbc742c5
Revises: b83ef173c7a4
Create Date: 2025-11-12 09:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7d94bbc742c5'
down_revision: Union[str, None] = 'b83ef173c7a4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns for OpenAI-compatible endpoint support
    op.add_column('labeling_jobs', sa.Column('openai_compatible_endpoint', sa.String(length=500), nullable=True))
    op.add_column('labeling_jobs', sa.Column('openai_compatible_model', sa.String(length=100), nullable=True))


def downgrade() -> None:
    # Remove the columns if rolling back
    op.drop_column('labeling_jobs', 'openai_compatible_model')
    op.drop_column('labeling_jobs', 'openai_compatible_endpoint')
