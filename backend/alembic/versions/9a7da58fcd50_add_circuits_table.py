"""add circuits table

Revision ID: 9a7da58fcd50
Revises: w0x1y2z3a4b5
Create Date: 2026-07-19 20:26:12.513964

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '9a7da58fcd50'
down_revision: Union[str, None] = 'w0x1y2z3a4b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "circuits",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("narrative", sa.Text(), nullable=True),
        sa.Column("granularity", sa.String(16), nullable=False, server_default="feature"),
        sa.Column("saes", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("members", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("edges", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("budget", postgresql.JSONB(), nullable=True),
        sa.Column("faithfulness", postgresql.JSONB(), nullable=True),
        sa.Column("rung", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("promoted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("discovery_run_id", sa.String(36), nullable=True),
        sa.Column("model_id", sa.String(255), nullable=True),
        sa.Column("schema_version", sa.String(8), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_circuits_promoted_rung", "circuits", ["promoted", "rung"])


def downgrade() -> None:
    op.drop_index("idx_circuits_promoted_rung", table_name="circuits")
    op.drop_table("circuits")
