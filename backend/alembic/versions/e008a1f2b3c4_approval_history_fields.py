"""approval history fields: who asked and who decided an agent approval request

Revision ID: e008a1f2b3c4
Revises: c0de5a11e004
Create Date: 2026-10-28

ENH-008 (§3.2 step 7). Two nullable columns on `agent_approval_requests`, no backfill:

* `requested_by` — the `X-MiStudio-Agent` value the request was filed with ("mcp"), or "operator" for one filed
  from the UI; NULL on rows from before this revision (the Approvals page shows "asked by an agent").
* `decided_by` — "operator" on approve or deny, "system" when a pending request expired.

`ADD COLUMN … NULL` with no default takes ACCESS EXCLUSIVE only for the catalogue update (no rewrite); the
migration runner's lock timeout covers a blocked ALTER (`src/db/migration_safety.py`).
"""

from alembic import op
import sqlalchemy as sa

revision = "e008a1f2b3c4"
down_revision = "c0de5a11e004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_approval_requests", sa.Column("requested_by", sa.String(64), nullable=True))
    op.add_column("agent_approval_requests", sa.Column("decided_by", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_approval_requests", "decided_by")
    op.drop_column("agent_approval_requests", "requested_by")
