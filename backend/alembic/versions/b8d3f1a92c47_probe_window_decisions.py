"""A threshold per contract window on probe_monitors.

A probe scores the tokens in a window and takes the mean, and its threshold is the
`(1 - target_fpr)` quantile of negatives aggregated under ONE window. miLLM can now read the same
weights over `all`, `prompt` and `response` separately — because the prompt says something about
the user and the response about the model — and over a different window that single number no
longer names the same false-positive rate. It is a quantile of a different distribution.

Until a per-window threshold exists, those verdicts fire against the whole-request bar and are
reported `provisional`. This column is what retires that flag for the windows it covers.

⚠ NULLABLE, AND NULL IS MEANINGFUL. "This probe predates per-window calibration" must stay
distinguishable from "its windows were calibrated and came back empty": only the first should make
a consumer fall back to the single `threshold`. So no server default and no backfill — there is no
honest value to write for rows whose negatives were never scored under these windows, and
inventing one is how this estate once made every historical row claim a `chat_format` it had never
used.

Revision ID: b8d3f1a92c47
Revises: f7a2c9d41b60
Create Date: 2026-09-30
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "b8d3f1a92c47"
down_revision = "f7a2c9d41b60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "probe_monitors",
        sa.Column("window_decisions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("probe_monitors", "window_decisions")
