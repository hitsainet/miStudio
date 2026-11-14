"""The activation frequency, counted before the junk-token filter.

`features.activation_frequency` is derived from the examples extraction EMITS,
and emission happens after `is_junk_token` has discarded any (sample, feature)
pair whose peak landed on a filtered token
(extraction_vectorized.py:331-341, then extraction_service.py:1675-1676). So the
column has always meant "fraction of samples where this feature's peak was on a
NON-JUNK token", not "fraction of samples where the feature fired".

Five of six filters are on by default, and `filter_fragments` matches a
~500-entry blocklist containing ordinary words (`land`, `field`, `like`, `some`,
`one`). Measured on feature eb48_00046: the blocklist deletes documents whose
peak landed on a CONVENTIONAL affix while letting unconventional splits through,
which is how that feature came to be labelled "technical identifiers" when what
it actually detects is a tokenizer seam.

TWO COLUMNS, TWO MEANINGS, BOTH STATED. The existing column is NOT corrected in
place. It feeds the dead-neuron gate and the frontend's steering auto-baseline
`clamp(2.9 - 2.6*freq, 1, 3)`, whose constants were FITTED against measured
values of that very column over the range 0.037-0.484. Redefining it would shift
every auto-baseline strength in the estate with nothing reporting the change,
and the four pinned calibration points in steeringStrength.test.ts would then be
anchored to frequencies the pipeline no longer produces.

NULL on every existing row, and that is the honest encoding: those extractions
did not measure this. No backfill revision follows, because there is nothing to
derive it from — the discarded pairs were never recorded.

ADDITIVE ONLY. Both columns are nullable with no default, so PostgreSQL adds
them as catalogue-only changes with no table rewrite.

Revision ID: 9e44fb6b6ba9
Revises: eb82dd9fd5a4
Create Date: 2026-09-10
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "9e44fb6b6ba9"
down_revision = "eb82dd9fd5a4"
branch_labels = None
depends_on = None


_COLUMNS = (
    ("activation_frequency_true", "DOUBLE PRECISION"),
    ("activation_count_true", "BIGINT"),
)
_INDEX = "idx_features_activation_frequency_true"
#: `features` is ~100 MB across the estate, well inside what builds in a
#: startup probe. The limit is stated rather than assumed so the day it grows,
#: the build steps aside instead of taking the API down.
_INLINE_BUILD_MAX_BYTES = 2_000_000_000


def upgrade() -> None:
    for column, ddl in _COLUMNS:
        op.execute(
            f"ALTER TABLE features ADD COLUMN IF NOT EXISTS {column} {ddl}"
        )

    # THE SAME HELPER ITS SIBLING USES, not a second hand-rolled copy.
    #
    # An earlier version of this revision got CONCURRENTLY but none of the
    # invalid-index handling written one file over — which is precisely the
    # "fixed one representative, never generalized" pattern this repo has a
    # memory for. `features` is ~356k rows so the window is narrow, but a pod
    # eviction mid-build is enough to leave `indisvalid = false`, after which
    # `IF NOT EXISTS` skips it forever and Alembic stamps success over a dead
    # index.
    #
    # The comparison between the two frequency columns is the whole point of
    # having both, and it should not be a sequential scan.
    from src.db.index_build import ensure_index_concurrently

    ensure_index_concurrently(
        op,
        index=_INDEX,
        table="features",
        columns="(activation_frequency_true)",
        revision=revision,
        inline_build_max_bytes=_INLINE_BUILD_MAX_BYTES,
    )


def downgrade() -> None:
    from src.db.index_build import drop_index_concurrently

    drop_index_concurrently(op, index=_INDEX)
    for column, _ddl in reversed(_COLUMNS):
        op.execute(f"ALTER TABLE features DROP COLUMN IF EXISTS {column}")
