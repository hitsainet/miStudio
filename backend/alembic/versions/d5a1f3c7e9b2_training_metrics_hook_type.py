"""training_metrics: the metric row carries its hook type, and the unique key includes it

Revision ID: d5a1f3c7e9b2
Revises: c4e8b2d6f1a9
Create Date: 2026-09-15

SAE training remediation, review round 1, finding A5 (HIGH).

THE DEFECT. A training trains one SAE per (layer, hook type) and logs a metric row
per SAE at every log step. Those rows carried only ``layer_idx``, and the table was
unique on (training_id, step, layer_idx). Two hook types on one layer wrote the same
key, so a training over ``hook_types=["residual", "mlp"]`` failed with IntegrityError
at its first log step. The training form offers several hook types, so users could
reach it. The held-out row of the second hook hit the same key inside a broad
``except`` and was silently lost.

THE COLUMN. ``hook_type`` (VARCHAR(50), nullable): the SAE's hook type on every
per-SAE row, in-sample and held-out. NULL on aggregated rows, which describe every
SAE at once, and on every row written before this revision. There is no backfill,
because nothing recorded which hook a historical row described. A single-hook run's
rows could be inferred from its hyperparameters, but a guessed value is the wrong
trade in a column whose purpose is to stop rows being misattributed.

THE KEY, AND HOW NULLs BEHAVE. The unique constraint
``uq_training_metrics_tid_step_layer`` (training_id, step, layer_idx) is replaced by
the unique EXPRESSION index

    uq_training_metrics_tid_step_layer_hook
        ON training_metrics (training_id, step, layer_idx, COALESCE(hook_type, ''))

What each kind of row is guaranteed, before and after:

* Per-SAE rows (layer_idx >= 0) and held-out rows (layer_idx = -1 - layer). Before:
  unique per (training, step, layer), so a second hook on the layer collided. After:
  unique per (training, step, layer, hook). The held-out encoding stays unambiguous
  per hook, because the hook is on the row beside it.
* Historical rows (hook_type NULL, layer_idx not NULL). COALESCE maps NULL to '', so
  they keep EXACTLY the old guarantee: unique per (training, step, layer).
* Aggregated rows (layer_idx NULL). Before: NOT unique at all. PostgreSQL treats NULLs
  as distinct in a unique constraint, so any number of aggregated rows could exist at
  one step; the resume's metric discard exists partly because of that. After: the
  same. layer_idx is deliberately NOT coalesced, so this revision changes no guarantee
  for aggregated rows. Making them unique would first need the historical duplicates
  that a pre-discard resume wrote to be deleted. That is a separate decision, recorded
  as debt in the A5 record.

WHY COALESCE AND NOT ``NULLS NOT DISTINCT``. ``UNIQUE NULLS NOT DISTINCT`` needs
PostgreSQL 15+. CI and k8s run 15, but ``docker-compose.yml``,
``docker-compose.hub.yml`` and the local development database run 14, where it is a
syntax error. Even on 15 it would do the wrong thing here: applied to the whole key,
it would make aggregated rows unique too, which is a change of guarantee (above).

WHY COALESCE AND NOT A PLAIN 4-COLUMN UNIQUE CONSTRAINT. With NULLs distinct, a plain
(training_id, step, layer_idx, hook_type) key accepts any number of per-SAE rows whose
hook_type is NULL. A call site that forgot to pass the hook would then write rows the
database accepts while nothing can tell them apart, and the failure would go quiet.
Under COALESCE the same omission on a multi-hook run writes two rows of
(layer, '') and fails loudly, as the old key did.

WHY NOT TWO PARTIAL INDEXES (hook NULL / hook NOT NULL). The guarantees are the same,
but there are two objects to keep in step, and a NULL-hook row could coexist with a
'residual' row for the same layer. One expression index states the rule in one place.

DOWNGRADE. Restores the 3-column unique constraint and drops the column. Rows that a
multi-hook run writes after this revision violate the old key, and the old schema
cannot represent them. So the downgrade first keeps ONE row per
(training_id, step, layer_idx): the 'residual' row when there is one (the default hook
and the only one that can be spliced), otherwise the lowest id. It leaves aggregated
rows (NULL layer) alone, since the old key never constrained them, and logs the number
of rows removed as a WARNING on the ``alembic`` logger, which the Alembic CLI prints (a
server-side RAISE NOTICE was tried first: neither the CLI nor psycopg2 surfaced it).
Metric rows are a display-only time series; the SAEs, checkpoints and exports of those
runs are unaffected.
"""
import logging

from alembic import op
import sqlalchemy as sa

revision = "d5a1f3c7e9b2"
down_revision = "c4e8b2d6f1a9"
branch_labels = None
depends_on = None

OLD_KEY = "uq_training_metrics_tid_step_layer"
NEW_KEY = "uq_training_metrics_tid_step_layer_hook"
HOOK_TYPE_COMMENT = (
    "The SAE's hook type (residual, mlp, attention) on per-SAE and held-out rows. "
    "NULL on aggregated rows and on rows written before 2026-09-15."
)

#: One row survives per (training_id, step, layer_idx): the 'residual' row if there
#: is one, else the lowest id. A row goes when a better row exists in its group.
#:
#: IS NOT DISTINCT FROM, never =: hook_type = 'residual' is NULL on a historical row,
#: and NULL compared with anything is NULL, so a historical NULL-hook row and a later
#: 'mlp' row at the same layer would BOTH survive and the old constraint could not be
#: re-created. (Found tracing this SQL by hand; pinned by
#: test_training_metrics_hook_key.py, which mixes the two.)
DOWNGRADE_DEDUP = """
DELETE FROM training_metrics tm
WHERE tm.layer_idx IS NOT NULL
  AND EXISTS (
    SELECT 1 FROM training_metrics keep
    WHERE keep.training_id = tm.training_id
      AND keep.step = tm.step
      AND keep.layer_idx = tm.layer_idx
      AND keep.id <> tm.id
      AND (
            (    (keep.hook_type IS NOT DISTINCT FROM 'residual')
             AND NOT (tm.hook_type IS NOT DISTINCT FROM 'residual'))
         OR (    (keep.hook_type IS NOT DISTINCT FROM 'residual')
               = (tm.hook_type IS NOT DISTINCT FROM 'residual')
             AND keep.id < tm.id)
          )
  )
"""

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    op.add_column(
        "training_metrics",
        sa.Column("hook_type", sa.String(length=50), nullable=True, comment=HOOK_TYPE_COMMENT),
    )
    op.drop_constraint(OLD_KEY, "training_metrics", type_="unique")
    op.create_index(
        NEW_KEY,
        "training_metrics",
        ["training_id", "step", "layer_idx", sa.text("COALESCE(hook_type, '')")],
        unique=True,
    )


def downgrade() -> None:
    removed = op.get_bind().execute(sa.text(DOWNGRADE_DEDUP)).rowcount
    logger.warning(
        "d5a1f3c7e9b2 downgrade: removed %s per-hook metric rows the old key cannot hold", removed
    )
    op.drop_index(NEW_KEY, table_name="training_metrics")
    op.create_unique_constraint(OLD_KEY, "training_metrics", ["training_id", "step", "layer_idx"])
    op.drop_column("training_metrics", "hook_type")
