"""Feature 010: add 'mcp_agent' to label_source_enum

Kept as its own revision: ALTER TYPE ... ADD VALUE cannot run in the same
transaction that later uses the value.

Revision ID: v9w0x1y2z3a4
Revises: u8v9w0x1y2z3
Create Date: 2026-07-12
"""

from alembic import op

revision = "v9w0x1y2z3a4"
down_revision = "u8v9w0x1y2z3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE label_source_enum ADD VALUE IF NOT EXISTS 'mcp_agent'")


def downgrade() -> None:
    # Postgres cannot remove enum values; leaving the value in place is harmless.
    pass
