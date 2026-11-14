"""add dismissed_operations

Lets a failed operation be cleared from the Monitor without deleting the job
record it came from.

WHY A SEPARATE TABLE. The Monitor's Failed Operations list federates rows from
trainings, extraction_jobs, labeling_jobs and neuronpedia_pushes. Those rows
are read-only there (can_retry=False) and the UI told the user to "manage in
its panel" — but for neuronpedia_pushes no such control exists anywhere: the
only DELETE in the Neuronpedia API targets neuronpedia_exports, a different
table. Four failures from 2026-03-28 were therefore unclearable, reported
2026-07-26.

The alternatives were worse. Deleting the source row destroys the failure
record and its error message. Adding a `dismissed_at` column to each federated
table means one migration per table now and another for every table federated
later. One keyed marker table covers all current and future sources uniformly
and is trivially reversible — undismissing is a DELETE.

Revision ID: c7e2a4f18b93
Revises: b4d19f0c73ae
Create Date: 2026-07-26

"""
from alembic import op
import sqlalchemy as sa


revision = "c7e2a4f18b93"
down_revision = "b4d19f0c73ae"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dismissed_operations",
        sa.Column(
            "source_type",
            sa.String(50),
            primary_key=True,
            comment="Federated source: training | extraction | labeling | neuronpedia_push",
        ),
        sa.Column(
            "source_id",
            sa.String(255),
            primary_key=True,
            comment="Primary key of the row in the source table",
        ),
        sa.Column(
            "dismissed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    # The failed-operations list looks these up on every poll.
    op.create_index(
        "ix_dismissed_operations_source_type",
        "dismissed_operations",
        ["source_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_dismissed_operations_source_type", table_name="dismissed_operations")
    op.drop_table("dismissed_operations")
