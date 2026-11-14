"""Which stored examples the judge is shown, and how their strength is stated.

Every retrieval in this codebase is `ORDER BY max_activation DESC LIMIT
max_examples` (labeling_service._retrieve_top_examples_batch_sync,
feature_service.get_feature_examples, nlp_analysis_tasks). So a feature's label
is an inference from the extreme upper tail of its activation distribution --
the top 10 of a stored top-100, or of a stored top-20 on the eb48 extraction.

`example_sampling='stratified'` spreads the same `max_examples` budget across
the stored range instead of taking its head. `activation_display` is separate
and deliberately so: widening the evidence and restating its strength are two
variables, and the 2026-09-01 triage experiment moved two at once (SAE
expansion AND context width) with the result that nothing was attributable
until one was held fixed. The display knob can therefore be run alone as an
inert control before the sampling knob is trusted.

ADDITIVE ONLY. Both columns are NOT NULL with a server default matching
today's behaviour, so PostgreSQL adds them as catalogue-only changes and every
existing template keeps behaving exactly as it did. Nothing here reads or
rewrites a row; there is no backfill to separate out.

⚠ THIS CHANGES EVERY TEMPLATE'S FINGERPRINT, AND THAT IS CORRECT.

`labeling_fingerprint.prompt_fingerprint_fields` hashes the whole non-identity
column set derived from `__table__.columns`, so a new key changes the hash even
at its default value. Consequence: the coverage endpoint will report every
existing verdict as STALE the moment this lands.

That is cosmetic and safe. Staleness is an opt-in query -- routine resume
selects on `label_status IN ('pending','failed')` -- so nothing is re-labelled
automatically. Do NOT "fix" it by adding these columns to IDENTITY_FIELDS: that
denylist is for fields which cannot change a single byte the judge is sent, and
these two change which examples it sees.

Revision ID: eb82dd9fd5a4
Revises: e617abdd7897
Create Date: 2026-09-10
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "eb82dd9fd5a4"
down_revision = "e617abdd7897"
branch_labels = None
depends_on = None


_COLUMNS = (
    ("example_sampling", "VARCHAR(24) NOT NULL DEFAULT 'top_k'"),
    ("activation_display", "VARCHAR(24) NOT NULL DEFAULT 'absolute'"),
)


def upgrade() -> None:
    for column, ddl in _COLUMNS:
        op.execute(
            "ALTER TABLE labeling_prompt_templates "
            f"ADD COLUMN IF NOT EXISTS {column} {ddl}"
        )


def downgrade() -> None:
    for column, _ddl in reversed(_COLUMNS):
        op.execute(
            f"ALTER TABLE labeling_prompt_templates DROP COLUMN IF EXISTS {column}"
        )
