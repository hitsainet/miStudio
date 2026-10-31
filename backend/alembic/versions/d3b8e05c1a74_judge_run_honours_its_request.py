"""judge runs record the caps they were asked for

Revision ID: d3b8e05c1a74
Revises: c9f2a71e4b08
Create Date: 2026-09-26

⚠ FOUND BY STAGE 2 ACCEPTANCE, NOT BY READING. `JudgeRunCreate` has declared
`max_rows_per_set` (ge=20, le=50000) and `parse_failure_limit` since 032 shipped. Both were
validated and then dropped: no column held them, the endpoint did not pass them, and
`execute_judge_run` read every row of every set and used `judge_rows`' own default limit.

Measured on `pmj_da0220514811`, submitted with `max_rows_per_set: 200` over five sets:
**5,288 chat completions in 75 minutes** against an expected 1,000 — it was judging all 5,737
rows. 5.7x the work, 77 minutes instead of about 13, on a served 7B model sharing the node's
only GPU.

NULLABLE, not backfilled with the schema defaults. A historical row genuinely does not know
what it was asked for, and writing 2000/0.05 into it would assert that it honoured a cap it
never saw — the `chat_format NOT NULL DEFAULT 'auto'` mistake, which made every historical row
claim it had used the tokenizer's real template. A NULL here reads as "not recorded", which is
the truth.
"""
from alembic import op
import sqlalchemy as sa

revision = "d3b8e05c1a74"
down_revision = "c9f2a71e4b08"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "probe_monitor_judge_runs",
        sa.Column("max_rows_per_set", sa.Integer(), nullable=True),
    )
    op.add_column(
        "probe_monitor_judge_runs",
        sa.Column("parse_failure_limit", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("probe_monitor_judge_runs", "parse_failure_limit")
    op.drop_column("probe_monitor_judge_runs", "max_rows_per_set")
