"""Restore the three foreign keys the ORM declares and the database lacks

Revision ID: d7f3a91c2e08
Revises: c7e2a4f18b93
Create Date: 2026-08-23

MIS-E2E-033. Three foreign keys are declared on the ORM models and absent from
the database:

    features.training_id        -> trainings.id      ON DELETE CASCADE
    features.labeling_job_id    -> labeling_jobs.id  ON DELETE SET NULL
    extraction_jobs.training_id -> trainings.id      ON DELETE CASCADE

The first two were dropped by ``j6k7l8m9n0o1`` and never re-created. The third
(``features.labeling_job_id``) was never created at all — ``6819dd3caeb3`` added
the column as a plain indexed VARCHAR. ``ondelete=`` in SQLAlchemy is DDL-only,
so with no constraint present it does nothing: deleting a labeling job leaves
dangling ids, in Postgres and in Python alike.

Why this matters beyond integrity: the unit suite builds its schema with
``Base.metadata.create_all()``, which reads the ORM. So the tests have run
against a schema WITH these constraints while production ran WITHOUT them —
the divergence was structurally unobservable. A guard test added alongside this
migration (``test_orm_matches_migrated_schema.py``) now fails when the two
diverge again.

ORPHANS: a dangling value cannot satisfy a new constraint, and production data
could not be inspected from the audit environment. Rather than risk a failed
deploy or delete rows, this NULLs any orphaned reference first and reports the
count. NULLing is the conservative choice — the referenced parent is already
gone, so the value is meaningless, and deleting the child (what CASCADE would
have done at the time) would destroy data that has survived without it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d7f3a91c2e08"
down_revision: Union[str, None] = "c7e2a4f18b93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (constraint, child table, child column, parent table, ondelete)
_FKS = [
    ("fk_features_training_id", "features", "training_id", "trainings", "CASCADE"),
    ("fk_features_labeling_job_id", "features", "labeling_job_id", "labeling_jobs", "SET NULL"),
    ("fk_extraction_jobs_training_id", "extraction_jobs", "training_id", "trainings", "CASCADE"),
]


def _existing_constraints(bind, table: str) -> set:
    return {
        fk["name"]
        for fk in sa.inspect(bind).get_foreign_keys(table)
        if fk.get("name")
    }


def upgrade() -> None:
    bind = op.get_bind()

    for name, child, col, parent, ondelete in _FKS:
        if name in _existing_constraints(bind, child):
            print(f"  {name}: already present, skipping")
            continue

        orphans = bind.execute(
            sa.text(
                f"UPDATE {child} SET {col} = NULL "
                f"WHERE {col} IS NOT NULL "
                f"  AND NOT EXISTS (SELECT 1 FROM {parent} p WHERE p.id = {child}.{col})"
            )
        ).rowcount
        if orphans:
            print(
                f"  {name}: NULLed {orphans} orphaned {child}.{col} value(s) — "
                f"the referenced {parent} row no longer exists"
            )

        op.create_foreign_key(
            name, child, parent, [col], ["id"], ondelete=ondelete
        )
        print(f"  {name}: created ({col} -> {parent}.id ON DELETE {ondelete})")


def downgrade() -> None:
    bind = op.get_bind()
    for name, child, _col, _parent, _ondelete in reversed(_FKS):
        if name in _existing_constraints(bind, child):
            op.drop_constraint(name, child, type_="foreignkey")
