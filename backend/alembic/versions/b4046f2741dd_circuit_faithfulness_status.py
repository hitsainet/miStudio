"""Faithfulness lifecycle on circuits (Feature 017 R2 B-5).

Faithfulness runs on a circuit, not a discovery run — its in-flight marker
lives here so the single-GPU guard sees it, cleanup reclaims a stuck one, and
two runs can't race. Idempotent add. Downgrade drops.

Revision ID: b4046f2741dd
Revises: 1c3ac72efd47
Create Date: 2026-07-20
"""
from typing import Sequence, Union

from alembic import op

revision: str = 'b4046f2741dd'
down_revision: Union[str, None] = '1c3ac72efd47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE circuits ADD COLUMN IF NOT EXISTS "
               "faithfulness_status VARCHAR(16)")
    op.execute("ALTER TABLE circuits ADD COLUMN IF NOT EXISTS "
               "faithfulness_task_id VARCHAR(155)")


def downgrade() -> None:
    op.execute("ALTER TABLE circuits DROP COLUMN IF EXISTS faithfulness_task_id")
    op.execute("ALTER TABLE circuits DROP COLUMN IF EXISTS faithfulness_status")
