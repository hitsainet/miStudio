"""Add circuits.model_hf_id — cross-instance-stable model provenance (R3-B2).

model_id is instance-local; hf_id is what a foreign import carries and what
015's model-mismatch hazard check needs. Downgrade drops the column (loses
imported hf provenance).

Revision ID: b2c3d4e5f6a7
Revises: 3e9c439d9085
Create Date: 2026-07-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = '3e9c439d9085'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('circuits', sa.Column('model_hf_id', sa.String(length=500), nullable=True))


def downgrade() -> None:
    op.drop_column('circuits', 'model_hf_id')
