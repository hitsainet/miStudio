"""Add circuits.version — optimistic-concurrency for edge writes (017 Task 3.0).

017's validation writer and a user editing a circuit in the panel would
otherwise silently clobber each other. `version` increments on every
CircuitService.update(); a stale expected_version 409s. Backfill existing
rows to 1. Downgrade drops the column.

Revision ID: 9c8683b365f6
Revises: e5f6a7b8c9da
Create Date: 2026-07-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '9c8683b365f6'
down_revision: Union[str, None] = 'e5f6a7b8c9da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('circuits', sa.Column(
        'version', sa.Integer(), nullable=False, server_default='1'))


def downgrade() -> None:
    op.drop_column('circuits', 'version')
