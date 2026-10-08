"""add app_settings table

Create the app_settings table for persistent application configuration.
Supports encrypted storage for sensitive values (API keys, tokens).

Revision ID: f3a7b1c2d4e5
Revises: 9e045d0a94ef
Create Date: 2026-03-08 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "f3a7b1c2d4e5"
down_revision = "9e045d0a94ef"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("key", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("is_sensitive", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("category", sa.String(50), nullable=False, server_default="general", index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
