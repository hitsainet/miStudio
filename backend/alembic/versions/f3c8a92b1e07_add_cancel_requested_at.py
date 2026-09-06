"""cancel_requested_at on the three native-enum lifecycles

Revision ID: f3c8a92b1e07
Revises: e4a1c7b2f5d9
Create Date: 2026-09-05

WHY A TIMESTAMP AND NOT A NEW ENUM MEMBER.

`datasets.status`, `models.status` and `dataset_tokenizations.status` are NATIVE
POSTGRES ENUMS, and none of them has a CANCELLED member. Adding one needs
`ALTER TYPE ... ADD VALUE`, which is non-transactional — it cannot be rolled
back inside a migration, and until recently could not run inside one at all.
The other twelve lifecycles already spell "cancelled" in their status column and
are untouched by this.

A nullable TIMESTAMPTZ is one additive column, and it is the better model
anyway: it separates "the operator asked" from "the job stopped". That
conflation is exactly what produced today's `status = ERROR` +
"Cancelled by user" — a deliberate stop recorded as a crash, indistinguishable
afterwards from a real failure.

The timestamp also answers a question the status never could: HOW LONG the job
kept running after the request. That is the number that tells an operator
whether cooperative cancellation is working.
"""
import sqlalchemy as sa
from alembic import op

revision = "f3c8a92b1e07"
down_revision = "e4a1c7b2f5d9"
branch_labels = None
depends_on = None

_TABLES = ("datasets", "models", "dataset_tokenizations")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(
            table,
            sa.Column(
                "cancel_requested_at",
                sa.DateTime(timezone=True),
                nullable=True,
                comment=(
                    "When the operator asked this job to stop. The job polls "
                    "this and stops at its next checkpoint; status records "
                    "where it actually ended up."
                ),
            ),
        )


def downgrade() -> None:
    for table in _TABLES:
        op.drop_column(table, "cancel_requested_at")
