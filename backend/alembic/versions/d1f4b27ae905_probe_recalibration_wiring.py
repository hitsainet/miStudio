"""probe_monitors: what a threshold needs to move without a GPU

Revision ID: d1f4b27ae905
Revises: c4e7a1b93d52
Create Date: 2026-10-02

Two columns, both nullable, both with no server default and no backfill.

`calibration_lengths_path` — the per-row scored-token counts the negatives came from.
`calibration_scores_path` has been kept since September so the operating point could be
re-derived "without a GPU — the quantile of an array already computed". For the GLOBAL
threshold that was true. For `length_bands` it was not: `length_band_decisions` refuses on a
length/score mismatch and its boundaries are quantiles of the lengths actually observed, so
without them a per-length re-cut still cost a full GPU pass to recover 8 KB that had been in
memory beside the scores the whole time.

`calibration_history` — every operating point the probe has served, append-only.

⚠ NO DEFAULT, AND THAT IS THE POINT. NULL means "this probe predates recalibration" /
"its bar is still the one its run cut" — both true of every existing row, and both different
from `[]`, which would mean "re-cut, and the history came back empty".

This estate has shipped the other choice and paid for it: `chat_format` was added
`NOT NULL DEFAULT 'auto'`, making every historical row claim it had used the tokenizer's real
chat template — a column added to stop a silent misattribution introducing one. A default here
would make every probe ever trained claim a calibration provenance nobody recorded.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "d1f4b27ae905"
down_revision = "c4e7a1b93d52"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "probe_monitors",
        sa.Column("calibration_lengths_path", sa.String(length=1000), nullable=True),
    )
    op.add_column(
        "probe_monitors",
        sa.Column(
            "calibration_history", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_column("probe_monitors", "calibration_history")
    op.drop_column("probe_monitors", "calibration_lengths_path")
