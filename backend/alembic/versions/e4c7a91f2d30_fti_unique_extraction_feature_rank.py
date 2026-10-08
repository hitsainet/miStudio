"""feature_token_index: one row per (extraction, feature, token_rank)

Revision ID: e4c7a91f2d30
Revises: d8b2f61a4c07
Create Date: 2026-09-25

WHY. `feature_token_index` is the inverted index the hard-negative donor query
reads: it selects donors `WHERE normalized_token = <target's> AND token_rank = 1`.
Nothing stopped two rows existing for one feature's rank-1 token, and a grouping
run that fails partway and is re-run is exactly how that happens. A duplicated
row is then drawn twice, so one donor's passages carry double weight in every
detection score — silently, with no error and no visible symptom.

MEASURED BEFORE ADDING (2026-09-25, production): 117,528 rows, 117,528 distinct
`(extraction_id, feature_id, token_rank)` keys. **No duplicates exist**, so this
adds the constraint without a dedupe step. If a future database does hold
duplicates this migration will fail loudly on that database, which is the correct
outcome: it means rows were double-counted and the operator must decide which to
keep, rather than have a migration choose silently.

Not a unique INDEX on purpose — a named constraint states the intent in the
catalogue, and the ORM declares the same name so the schema-parity guard
compares like with like.
"""
from alembic import op

revision = "e4c7a91f2d30"
down_revision = "d8b2f61a4c07"
branch_labels = None
depends_on = None

CONSTRAINT = "uq_fti_extraction_feature_rank"


def upgrade() -> None:
    op.create_unique_constraint(
        CONSTRAINT,
        "feature_token_index",
        ["extraction_id", "feature_id", "token_rank"],
    )


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT, "feature_token_index", type_="unique")
