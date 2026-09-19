"""Index feature_activations by (feature_id, max_activation, id).

THE TABLE HAS 35.6M ROWS IN ONE PARTITION AND ONE INDEX ON THE WRONG COLUMNS.

`feature_activations` is RANGE-partitioned by `feature_id`, and the partitions
were created for keys `feat_00000`..`feat_16000` (76918d8aa763). Real ids are
`feat_sae_...` and `feat_train_...`, which sort ABOVE every one of those bounds
under text ordering, so all 16 range partitions are empty and every row lands in
`feature_activations_default` (added later by 1b09fdca19e8) — which never
received the two per-partition indexes the range partitions got.

Measured on production 2026-09-10:

    feature_activations_default   69 GB   1 index   (the PK)
    feature_activations_p00..p15  32 kB   3 indexes each

The one index is the composite primary key `(id, feature_id)`. A btree cannot
serve a lookup on its SECOND column, so `WHERE feature_id = ANY(...)` degrades
to a full scan of a 35.6M-row index (`EXPLAIN` cost 1,404,642 for ~100 rows).

Sized honestly: batching amortises this for bulk labeling (~7 ms/feature against
~5 s of judging). INTERACTIVE reads pay full price — the feature detail modal,
`get_feature_examples`, the grouping service.

`id` is the third column because the windows order by
`feature_id, max_activation DESC, fa.id ASC`; without it the planner adds an
Incremental Sort. (`WEAKEST_EXAMPLES_SQL` orders `max_activation ASC` and cannot
use this index's order either way — a backward scan yields `id DESC`. It is the
one consumer this index does not order for.)

⚠ THIS MIGRATION WILL NOT BUILD A LARGE INDEX. THAT IS THE POINT.

`docker-entrypoint.sh` runs `alembic upgrade heads` BEFORE uvicorn, and
`k8s/base/backend.yaml` gives the startupProbe 30 x 10s = 300 seconds. A
`CREATE INDEX CONCURRENTLY` on 69 GB does two heap passes plus a sort at the
stock `maintenance_work_mem` of 64 MB, AND waits out every transaction that can
see the table — including the three Celery containers in the same pod, which a
backend restart does not stop.

An earlier version of this migration built it inline and raised if the result
was invalid. Both halves were wrong:
  * killed mid-build, the process dies before the raise can run, leaving
    `indisvalid = false`;
  * the next start drops the dead index and rebuilds FROM ZERO, is killed
    again, and loops forever — with `strategy: Recreate` the old pod is already
    gone, so the API is 503 throughout.
It came up on 2026-09-10 only because every labeling job had been cancelled
minutes earlier, so nothing held a conflicting transaction. That is luck, not a
deployment strategy.

A PERFORMANCE INDEX MUST NEVER BE ABLE TO TAKE DOWN THE API. On a large table
this migration reports what to run and stamps; correctness does not depend on
the index existing, only speed does. Build it deliberately, out of band, with
the workers quiesced:

    CREATE INDEX CONCURRENTLY idx_feature_activations_default_feature_max
      ON feature_activations_default (feature_id, max_activation DESC, id);

Revision ID: e617abdd7897
Revises: e5b91d4a2c73
Create Date: 2026-09-10
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "e617abdd7897"
down_revision = "e5b91d4a2c73"
branch_labels = None
depends_on = None


#: The partition that actually holds the data. Named directly rather than
#: created on the parent: a parent-level CREATE INDEX cascades to all 17
#: children, and CONCURRENTLY is unsupported on a partitioned parent.
_INDEX = "idx_feature_activations_default_feature_max"
_TABLE = "feature_activations_default"
_COLUMNS = "(feature_id, max_activation DESC, id)"

#: Above this on-disk size the build is left to an operator. 2 GB builds
#: comfortably inside a 300 s probe on this hardware; 69 GB does not.
_INLINE_BUILD_MAX_BYTES = 2_000_000_000


def upgrade() -> None:
    from src.db.index_build import ensure_index_concurrently

    ensure_index_concurrently(
        op,
        index=_INDEX,
        table=_TABLE,
        columns=_COLUMNS,
        revision=revision,
        inline_build_max_bytes=_INLINE_BUILD_MAX_BYTES,
    )


def downgrade() -> None:
    from src.db.index_build import drop_index_concurrently

    drop_index_concurrently(op, index=_INDEX)
