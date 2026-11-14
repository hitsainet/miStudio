"""Circuit strength calibration columns (Feature 20 / IDL-37).

Calibration finds the usable steering band (onset..cliff) and ships it in the
circuit contract. `calibration` is the CircuitCalibration JSONB snapshot; the
status/task_id pair mirrors faithfulness because calibration also runs on the
circuit and holds the GPU (single-GPU guard + cleanup + no-race).

Idempotent add. Downgrade drops. Additive + nullable — existing rows are
unaffected and existing exported documents (calibration absent) stay valid.

Revision id chosen distinctively ("ca11b" ≈ "calib") to avoid the placeholder-
hex collision class: `a1b2c3d4e5f6` was already taken by an unrelated migration,
which produced an alembic revision cycle.

Revision ID: ca11b7a7e020
Revises: f9a1c2d3e4b5
Create Date: 2026-07-21
"""
from typing import Sequence, Union

from alembic import op

revision: str = "ca11b7a7e020"
down_revision: Union[str, None] = "f9a1c2d3e4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE circuits ADD COLUMN IF NOT EXISTS calibration JSONB")
    op.execute("ALTER TABLE circuits ADD COLUMN IF NOT EXISTS "
               "calibration_status VARCHAR(16)")
    op.execute("ALTER TABLE circuits ADD COLUMN IF NOT EXISTS "
               "calibration_task_id VARCHAR(155)")


def downgrade() -> None:
    op.execute("ALTER TABLE circuits DROP COLUMN IF EXISTS calibration_task_id")
    op.execute("ALTER TABLE circuits DROP COLUMN IF EXISTS calibration_status")
    op.execute("ALTER TABLE circuits DROP COLUMN IF EXISTS calibration")
