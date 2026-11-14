"""dataworks provenance routes: pinned dataset revision, per-evaluation array paths, run filter index (Feature 035)

Revision ID: a35d0e1f2c3b
Revises: c34b5e2a9d02
Create Date: 2026-10-18

ONE revision for the whole feature (035 FTDD §4.4):

* ``datasets.revision`` VARCHAR(40) NULL, checked to be a 40-character lowercase hex commit or NULL.
  NO default and NO backfill: every existing row was downloaded without a revision, so NULL — read
  "revision not recorded" — is the only truthful value (FR-5, D1). Writing "main" or a guessed head
  here would turn a missing fact into a false one.
* ``probe_monitor_evaluations.positions_path``, ``n_scored_path``, ``labels_path`` VARCHAR(1000) NULL.
  NULL means the evaluation predates 035; the scores route refuses it with ``positions_not_recorded``
  rather than replaying positions (D15, D29). ``labels_path`` is its own column by the operator's
  2026-10-18 amendment of D44 (the planner default derived it from ``scores_path``).
* ``ix_probe_monitor_runs_train_dataset_id`` for ``GET /probe-monitors/runs?train_dataset_id=`` (D46).
"""

from alembic import op
import sqlalchemy as sa

revision = "a35d0e1f2c3b"
down_revision = "c34b5e2a9d02"
branch_labels = None
depends_on = None

_EVALUATION_PATHS = ("positions_path", "n_scored_path", "labels_path")


def upgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column(
            "revision",
            sa.String(length=40),
            nullable=True,
            comment="Hugging Face commit the files were downloaded at; NULL = not recorded",
        ),
    )
    op.create_check_constraint(
        "ck_datasets_revision_is_commit",
        "datasets",
        "revision IS NULL OR revision ~ '^[0-9a-f]{40}$'",
    )
    for name in _EVALUATION_PATHS:
        op.add_column(
            "probe_monitor_evaluations", sa.Column(name, sa.String(length=1000), nullable=True)
        )
    op.create_index(
        "ix_probe_monitor_runs_train_dataset_id", "probe_monitor_runs", ["train_dataset_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_probe_monitor_runs_train_dataset_id", table_name="probe_monitor_runs")
    for name in reversed(_EVALUATION_PATHS):
        op.drop_column("probe_monitor_evaluations", name)
    op.drop_constraint("ck_datasets_revision_is_commit", "datasets", type_="check")
    op.drop_column("datasets", "revision")
