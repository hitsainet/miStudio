"""record the render form each probe was trained under

Revision ID: a3c9e1f05b72
Revises: e7a2c5d913f4
Create Date: 2026-10-08

One column, nullable, no server default and no backfill: `probe_monitors.render_form` (JSONB),
e.g. `{"generation_prompt": true, "add_special_tokens": false}`.

Operator decision, 2026-10-08: probe training, calibration, evaluation and test vectors render a
conversation the way miLLM SERVES it — with the model's generation prompt after a user-ended
conversation (`probe_monitor_render.SERVED_RENDER_FORM`). Before that, miStudio rendered without
it, so every probe was calibrated on a form it never sees live.

⚠ NO DEFAULT, AND NO BACKFILL. NULL means NOT RECORDED — and for every pre-existing row that means
"rendered WITHOUT the generation prompt", which the code proves. Writing a value into those rows
would turn an inference into a recorded fact (the `chat_format NOT NULL DEFAULT 'auto'` mistake),
and writing the served form would be false. `probe_monitor_run.probe_render` reads NULL as the old
form and refuses work that would mix it with the served one.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a3c9e1f05b72"
down_revision = "e7a2c5d913f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "probe_monitors",
        sa.Column("render_form", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("probe_monitors", "render_form")
