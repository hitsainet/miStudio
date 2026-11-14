"""approval forecast: what the asking tool forecast for a request, shown to the operator

Revision ID: e012a15f7c01
Revises: e011a7c3d9f2
Create Date: 2026-11-09

ENH-006 G15. One nullable JSONB column on `agent_approval_requests`, no default, no backfill:

* `forecast` — filed beside the payload by the tool that asks (a probe run's time forecast: its one-line
  summary and the evaluation rows it assumes) and shown on the approval card. Never part of what is approved
  or consumed. NULL on every row before this revision.

`ADD COLUMN … NULL` with no default rewrites nothing; the migration runner's lock timeout covers a blocked
ALTER (`src/db/migration_safety.py`).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "e012a15f7c01"
down_revision = "e011a7c3d9f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_approval_requests",
        sa.Column("forecast", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_approval_requests", "forecast")
