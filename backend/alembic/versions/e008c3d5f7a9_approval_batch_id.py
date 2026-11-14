"""approval batch id: requests approved together by one "Approve all" call

Revision ID: e008c3d5f7a9
Revises: e008b7c9d1e3
Create Date: 2026-10-29

ENH-006 D81. One nullable column on `agent_approval_requests`, no backfill, and its index:

* `batch_id` — the id shared by every request one `POST /mcp/approvals/approve-batch` approved; NULL for a request
  decided on its own (every row before this revision).

`ADD COLUMN … NULL` with no default rewrites nothing. The index is built on a table of at most a few thousand rows;
the migration runner's lock timeout covers a blocked ALTER (`src/db/migration_safety.py`).
"""

from alembic import op
import sqlalchemy as sa

revision = "e008c3d5f7a9"
down_revision = "e008b7c9d1e3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_approval_requests", sa.Column("batch_id", sa.String(36), nullable=True))
    op.create_index(op.f("ix_agent_approval_requests_batch_id"), "agent_approval_requests", ["batch_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_approval_requests_batch_id"), table_name="agent_approval_requests")
    op.drop_column("agent_approval_requests", "batch_id")
