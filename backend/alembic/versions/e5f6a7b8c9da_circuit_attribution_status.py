"""Separate attribution lifecycle on circuit_discovery_runs (016 R1 QA-P2).

A failed/cancelled attribution pass must not make the completed DISCOVERY
present as failed. Adds attribution_status/progress/error. Downgrade drops them.

Revision ID: e5f6a7b8c9da
Revises: c3d4e5f6a7b8
Create Date: 2026-07-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'e5f6a7b8c9da'
down_revision: Union[str, None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE circuit_discovery_runs ADD COLUMN IF NOT EXISTS "
               "attribution_status VARCHAR(16)")
    op.execute("ALTER TABLE circuit_discovery_runs ADD COLUMN IF NOT EXISTS "
               "attribution_progress DOUBLE PRECISION")
    op.execute("ALTER TABLE circuit_discovery_runs ADD COLUMN IF NOT EXISTS "
               "attribution_error TEXT")
    op.execute("ALTER TABLE circuit_discovery_runs ADD COLUMN IF NOT EXISTS "
               "attribution_task_id VARCHAR(155)")


def downgrade() -> None:
    # IF EXISTS: tolerant of a DB where this revision was applied at an
    # earlier column set (dev drift while the migration was being authored).
    op.execute("ALTER TABLE circuit_discovery_runs "
               "DROP COLUMN IF EXISTS attribution_task_id")
    op.execute("ALTER TABLE circuit_discovery_runs "
               "DROP COLUMN IF EXISTS attribution_error")
    op.execute("ALTER TABLE circuit_discovery_runs "
               "DROP COLUMN IF EXISTS attribution_progress")
    op.execute("ALTER TABLE circuit_discovery_runs "
               "DROP COLUMN IF EXISTS attribution_status")
