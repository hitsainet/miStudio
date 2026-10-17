"""persist J-space feature annotations (024 tasks 4.3/4.4)

Revision ID: e2e024a11e5b
Revises: a3c9e1f05b72
Create Date: 2026-10-16

One new table, `jlens_feature_annotations`: a feature's lens readout per SURFACE (decoder
or encoder, BR-015) and lens type, with the disagreement score against the label it was
compared with. The disagreement queue (BR-013) is a query over it. Additive: no existing
table changes.

⚠ PARALLEL BRANCH NOTE (2026-10-16): written on `e2e/outstanding` while Feature 034 may add
its own revision on `e2e/034` from the same parent. If both land, re-parent one onto the
other at integration so `alembic heads` stays single.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "e2e024a11e5b"
down_revision = "a3c9e1f05b72"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jlens_feature_annotations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("sae_id", sa.String(length=255), nullable=False),
        sa.Column("feature_index", sa.Integer(), nullable=False),
        sa.Column("surface", sa.String(length=16), nullable=False),
        sa.Column("lens_type", sa.String(length=32), nullable=False),
        sa.Column("layer", sa.Integer(), nullable=False),
        sa.Column("lens_kurtosis", sa.Float(), nullable=True),
        sa.Column("workspace_class", sa.String(length=16), nullable=False),
        sa.Column("top_tokens", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("compared_label", sa.String(length=500), nullable=True),
        sa.Column("disagreement_score", sa.Float(), nullable=True),
        sa.Column("has_disagreement", sa.Boolean(), nullable=False),
        sa.Column("sweep_task_id", sa.String(length=155), nullable=True),
        sa.Column("evidence_rung", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "sae_id", "feature_index", "surface", "lens_type",
            name="uq_jlens_feature_annotation",
        ),
    )
    op.create_index(
        "ix_jlens_feature_annotations_sae_id", "jlens_feature_annotations", ["sae_id"]
    )
    op.create_index(
        "ix_jlens_feature_annotations_disagreement_score",
        "jlens_feature_annotations",
        ["disagreement_score"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_jlens_feature_annotations_disagreement_score",
        table_name="jlens_feature_annotations",
    )
    op.drop_index("ix_jlens_feature_annotations_sae_id", table_name="jlens_feature_annotations")
    op.drop_table("jlens_feature_annotations")
