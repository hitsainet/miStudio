"""probe runs and judge runs: progress_detail (live rows, set, rate, ETA)

Revision ID: e6a1c9d2b4f8
Revises: d5f8a2c41e7b
Create Date: 2026-10-21

Operator, 2026-10-21: "The probe run does not show progress." A judge run wrote progress once per
set and a training run's percentage said nothing about what it counted. ``progress_detail`` holds
the live reading (``services/live_progress.py``): for a judge run, rows judged / planned, the
current set, rate, ETA and parse failures; for a training run, the stage, its units, the item
inside it, rate and ETA.

Nullable JSONB, no default and no backfill: NULL means "has not reported", which is the truth for
every existing row. A new column rather than a key in ``metrics`` (the judge's RESULT, read by the
rung) or ``environment`` (the run's reproducibility record, rewritten at every stage boundary).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "e6a1c9d2b4f8"
down_revision = "d5f8a2c41e7b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("probe_monitor_runs", sa.Column("progress_detail", JSONB(), nullable=True))
    op.add_column("probe_monitor_judge_runs", sa.Column("progress_detail", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("probe_monitor_judge_runs", "progress_detail")
    op.drop_column("probe_monitor_runs", "progress_detail")
