"""view evidence spans: the span column a probe dataset view declares, and its profile

Revision ID: 0b576a6b9bda
Revises: e012a15f7c01
Create Date: 2026-11-09

ENH-006 G39 (operator, 2026-11-09). Two nullable columns on ``probe_monitor_datasets``; every existing
view reads NULL, which means "no span column declared", and nothing is recommended for it. Additive,
so the revision is safe before any deploy window.

* ``span_column``   — the split column holding each row's evidence offsets (``[[start, end], ...]``).
* ``evidence_spans`` — its profile, read once at registration (numbers only, never text).

Downgrade drops both.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0b576a6b9bda"
down_revision = "e012a15f7c01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("probe_monitor_datasets", sa.Column("span_column", sa.String(255), nullable=True))
    op.add_column(
        "probe_monitor_datasets",
        sa.Column("evidence_spans", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("probe_monitor_datasets", "evidence_spans")
    op.drop_column("probe_monitor_datasets", "span_column")
