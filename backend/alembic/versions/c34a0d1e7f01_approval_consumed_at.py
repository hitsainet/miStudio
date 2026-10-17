"""agent approvals can be consumed once (Feature 034, FR-20)

Revision ID: c34a0d1e7f01
Revises: e2e024a11e5b
Create Date: 2026-10-16

One nullable column, no default, no backfill: `agent_approval_requests.consumed_at`.

`millm_load_model` through MCP now waits for operator approval. The backend cannot execute the
load itself (it has no miLLM URL — T-98), so approval only marks the row `approved` and the MCP
tool CONSUMES it on re-call: one conditional `UPDATE … WHERE status = 'approved'`, which is the
only once-guarantee. `consumed_at` records when. The new status value `consumed` fits the existing
`String(10)` column, so it needs no migration.

NULL is truthful for every existing row: they are steering approvals, which are executed by the
backend on approve and never consumed.
"""

from alembic import op
import sqlalchemy as sa

revision = "c34a0d1e7f01"
down_revision = "e2e024a11e5b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_approval_requests",
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_approval_requests", "consumed_at")
