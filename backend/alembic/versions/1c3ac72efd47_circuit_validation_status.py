"""Separate validation lifecycle on circuit_discovery_runs (Feature 017).

A failed validation pass must not make the completed discovery present as
failed (same pattern as attribution). Idempotent add. Downgrade drops.

Revision ID: 1c3ac72efd47
Revises: 85f73dbda900
Create Date: 2026-07-20
"""
from typing import Sequence, Union

from alembic import op

revision: str = '1c3ac72efd47'
down_revision: Union[str, None] = '85f73dbda900'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE circuit_discovery_runs ADD COLUMN IF NOT EXISTS "
               "validation_status VARCHAR(16)")
    op.execute("ALTER TABLE circuit_discovery_runs ADD COLUMN IF NOT EXISTS "
               "validation_progress DOUBLE PRECISION")
    op.execute("ALTER TABLE circuit_discovery_runs ADD COLUMN IF NOT EXISTS "
               "validation_error TEXT")
    op.execute("ALTER TABLE circuit_discovery_runs ADD COLUMN IF NOT EXISTS "
               "validation_task_id VARCHAR(155)")


def downgrade() -> None:
    for col in ("validation_task_id", "validation_error",
                "validation_progress", "validation_status"):
        op.execute(f"ALTER TABLE circuit_discovery_runs DROP COLUMN IF EXISTS {col}")
