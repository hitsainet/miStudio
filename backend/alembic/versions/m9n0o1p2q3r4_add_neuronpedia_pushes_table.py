"""Add neuronpedia_pushes table for tracking local push jobs

Revision ID: m9n0o1p2q3r4
Revises: l8m9n0o1p2q3
Create Date: 2026-03-28

Tracks Neuronpedia local push jobs so they appear in Active Operations monitor.
"""

from alembic import op
import sqlalchemy as sa

revision = "m9n0o1p2q3r4"
down_revision = "l8m9n0o1p2q3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "neuronpedia_pushes",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("sae_id", sa.String(255), nullable=False, index=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float, nullable=False, server_default="0"),
        sa.Column("features_pushed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_features", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("neuronpedia_pushes")
