"""record the precision every model ran at, and each checkpoint's own

Revision ID: e7a2c5d913f4
Revises: d1f4b27ae905
Create Date: 2026-10-03

Three columns, all nullable, all with no server default and no backfill.

`activation_extractions.model_dtype` and `trainings.model_dtype` — the precision the model ran at
when these activations were read (`ml/native_dtype.py`). `models.native_dtype` — what the
checkpoint's own config.json records, read from the file at download.

Before 2026-10-03 every miStudio load path cast 16-bit rows to float16 while the checkpoints are
bfloat16 and miLLM serves bfloat16, and nothing recorded which, so a probe failing parity on
import was the first sign. Recording it is half the fix.

⚠ NO DEFAULT, AND NO BACKFILL OF `model_dtype`. NULL means NOT RECORDED. Every pre-existing extraction and
training did in fact run float16 — the code proves it — and the UI may SAY so, labelled as
inferred (`services/artifact_dtype.py`). Writing "float16" into these rows would turn an
inference into a recorded fact, which is the `chat_format NOT NULL DEFAULT 'auto'` mistake this
estate has already paid for once.

`models.native_dtype` IS backfilled — from each checkpoint's config.json, a fact read from a file,
not an inference (`src/db/native_dtype_backfill.py`, which cannot fail the migration).
"""

from alembic import op
import sqlalchemy as sa

revision = "e7a2c5d913f4"
down_revision = "d1f4b27ae905"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("activation_extractions", sa.Column("model_dtype", sa.String(length=16), nullable=True))
    op.add_column("trainings", sa.Column("model_dtype", sa.String(length=16), nullable=True))
    op.add_column("models", sa.Column("native_dtype", sa.String(length=16), nullable=True))
    # A fact, read from each checkpoint's own config.json; see src/db/native_dtype_backfill.py.
    from src.db.native_dtype_backfill import backfill_native_dtype

    backfill_native_dtype(op.get_bind())


def downgrade() -> None:
    op.drop_column("models", "native_dtype")
    op.drop_column("trainings", "model_dtype")
    op.drop_column("activation_extractions", "model_dtype")
