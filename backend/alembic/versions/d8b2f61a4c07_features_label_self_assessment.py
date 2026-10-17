"""features.label_fit_count / label_confidence — the judge's own self-assessment

Revision ID: d8b2f61a4c07
Revises: c3f1a9d27b64
Create Date: 2026-09-22

WHY. Labeling templates ask the judge for `fit_count` (how many of the examples
shown its hypothesis explains, "N/D") and `confidence`. The parser carried both
through — `_enforce_refusal` even uses `fit_count` to turn a poorly-supported
label into a refusal — and then the write path DROPPED them. So the only signal
in the pipeline for "the judge was unsure" existed for one function call and
was gone: a confident wrong label and a hedged right one were indistinguishable
on every row, and no labeling run could be validated against its own
self-assessment. The refusal function's docstring claimed the original verdict
was "preserved in fit_count/confidence"; nothing preserved it.

NULLABLE, NO DEFAULT, NO server_default, NO BACKFILL — the provenance convention
(b9d4e7a2c815). NULL means "not recorded": true of every row labelled before
this column existed, and of any row labelled by a template that never asks.

CATALOGUE-ONLY: nullable columns with no default do not rewrite the table.
"""
from alembic import op
import sqlalchemy as sa

revision = "d8b2f61a4c07"
down_revision = "c3f1a9d27b64"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("features", sa.Column("label_fit_count", sa.String(length=32), nullable=True))
    op.add_column("features", sa.Column("label_confidence", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column("features", "label_confidence")
    op.drop_column("features", "label_fit_count")
