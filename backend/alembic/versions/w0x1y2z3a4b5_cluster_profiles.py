"""cluster_profiles table (Feature 014, IDL-30)

Durable, user-authored cluster captures (name + narrative + tuned member
strengths + budget snapshot), decoupled from the recomputable grouping tables.
FK to external_saes is RESTRICT: deleting an SAE with profiles must be an
explicit, surfaced act (service returns a structured 409).

Revision ID: w0x1y2z3a4b5
Revises: v9w0x1y2z3a4
Create Date: 2026-07-16
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "w0x1y2z3a4b5"
down_revision = "v9w0x1y2z3a4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cluster_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "sae_id",
            sa.String(255),
            sa.ForeignKey("external_saes.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("model_id", sa.String(255), nullable=True),
        sa.Column("extraction_id", sa.String(255), nullable=True),
        sa.Column("source_group_id", sa.String(36), nullable=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("narrative", sa.Text(), nullable=True),
        sa.Column("display_token", sa.String(255), nullable=True),
        sa.Column("members", JSONB(), nullable=False),
        sa.Column("budget", JSONB(), nullable=True),
        sa.Column("schema_version", sa.String(8), nullable=False, server_default="1"),
        sa.Column("imported_from", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_cluster_profiles_sae_id", "cluster_profiles", ["sae_id"])
    op.create_index("ix_clp_sae_name", "cluster_profiles", ["sae_id", "name"])


def downgrade() -> None:
    op.drop_index("ix_clp_sae_name", table_name="cluster_profiles")
    op.drop_index("ix_cluster_profiles_sae_id", table_name="cluster_profiles")
    op.drop_table("cluster_profiles")
