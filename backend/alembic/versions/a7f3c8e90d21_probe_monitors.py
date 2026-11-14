"""probe monitors: five tables for probe datasets, runs, probes, evaluations and judge runs

Revision ID: a7f3c8e90d21
Revises: e4c7a91f2d30
Create Date: 2026-09-25

WHY (PADR IDL-51). Feature 032 trains a linear probe over one decoder layer's
`resid_post` activations and evaluates it on data it was not fitted on. Five
tables, and each exists for a reason a JSONB blob on an existing table would not
serve:

`probe_monitor_datasets` is a VIEW, not a copy — the columns, label mapping and
filter that turn a downloaded dataset into labelled examples, plus the counts that
mapping produced. Several views over one download is the normal case (one repo's
`training` config trains while its six `*_balanced` configs evaluate), and the
counts are stored because `excluded` and `unparseable` are precisely the numbers a
refusal has to cite.

`probe_monitor_runs` carries the full layer × pool selection grid, not only the
winning layer. A sweep that records only its argmax cannot be audited for a
near-tie, and a near-tie is exactly when the chosen layer is arbitrary.

`probe_monitors` is one row per (layer, rule, variant) because that is the unit
033 exports. `threshold` is an operating point, not a quality number, so
`realised_fpr` and `threshold_source` sit beside it: a threshold from validation
negatives and one from a held-out calibration corpus are different claims.

`probe_monitor_evaluations` exists so that A REFUSAL IS A ROW. Below 20 per class
the row is written with its reason and both counts. A missing row reads as "not run
yet" and a 0.5 reads as "measured, and chance" — neither is what happened, and this
estate has already paid for a metric that reported a comfortable 0.5 instead of
refusing.

`probe_monitor_judge_runs` records the LLM monitor as the BASELINE a probe is
compared against (FR-13 rung 3 is "judge-compared", which claims the two were
measured on the same data and nothing about the judge being right).
`parse_failures` is a column rather than a log line because a judge whose output
stops parsing yields a silently smaller sample.

STATUS COLUMNS ARE STRINGS, NOT NATIVE PG ENUMS, following `circuit_capture_runs`.
A native enum needs a migration to gain a value, and this repo has already paid for
three tables (`datasets`, `models`, `dataset_tokenizations`) whose enums lacked
CANCELLED when cooperative cancellation arrived — each needed its own column to
work around the enum it could not extend.

ON DELETE, chosen per edge rather than uniformly:
  · run → probes, probe → evaluations: CASCADE. A probe without its run is
    unreproducible and an evaluation without its probe is unreadable.
  · probe → `sae_id`: SET NULL. Deleting an SAE must not delete the record that a
    probe was trained over it; the probe becomes unbuildable and that is a state to
    REPORT (033 refuses it), not to erase.
  · run → `calibration_dataset_id`: SET NULL, same reasoning.

The downgrade drops all five in reverse dependency order, so it is a real
downgrade and not a stub.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a7f3c8e90d21"
down_revision = "e4c7a91f2d30"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "probe_monitor_datasets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        # UUID, not String(36): `datasets.id` is a real PG uuid column, unlike every
        # other id on this estate. A varchar FK to it is refused by Postgres with
        # DatatypeMismatchError — caught here by the upgrade/downgrade round trip.
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("config", sa.String(length=255), nullable=True),
        sa.Column("split", sa.String(length=255), nullable=True),
        sa.Column("input_column", sa.String(length=255), nullable=False),
        sa.Column("label_column", sa.String(length=255), nullable=False),
        sa.Column("label_mapping", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("keyword_filter", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("pair_column", sa.String(length=255), nullable=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("distribution", sa.String(length=24), nullable=True),
        sa.Column("counts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_probe_monitor_datasets_dataset_id", "probe_monitor_datasets", ["dataset_id"]
    )
    op.create_index("ix_probe_monitor_datasets_role", "probe_monitor_datasets", ["role"])

    op.create_table(
        "probe_monitor_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),   # models.id is varchar(255)
        sa.Column("train_dataset_id", sa.String(length=36), nullable=False),
        sa.Column("eval_dataset_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("calibration_dataset_id", sa.String(length=36), nullable=True),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("celery_task_id", sa.String(length=155), nullable=True),
        sa.Column("gpu_request", sa.String(length=64), nullable=True),
        sa.Column("gpu_uuid", sa.String(length=64), nullable=True),
        sa.Column("gpu_uuids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("artifact_dir", sa.String(length=1000), nullable=True),
        sa.Column("environment", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("layer_selection", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["train_dataset_id"], ["probe_monitor_datasets.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["calibration_dataset_id"], ["probe_monitor_datasets.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_probe_monitor_runs_model_status", "probe_monitor_runs", ["model_id", "status"]
    )

    op.create_table(
        "probe_monitors",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("layer", sa.Integer(), nullable=False),
        sa.Column("rule", sa.String(length=32), nullable=False),
        sa.Column("rule_params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("variant", sa.String(length=8), nullable=False),
        sa.Column("sae_id", sa.String(length=255), nullable=True),      # external_saes.id is varchar(255)
        sa.Column("sae_feature_indices", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("val_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("selected", sa.Boolean(), nullable=False),
        sa.Column("weights_path", sa.String(length=1000), nullable=True),
        sa.Column("norm_path", sa.String(length=1000), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("target_fpr", sa.Float(), nullable=True),
        sa.Column("realised_fpr", sa.Float(), nullable=True),
        sa.Column("threshold_source", sa.String(length=32), nullable=True),
        sa.Column("calibration_dataset_id", sa.String(length=36), nullable=True),
        sa.Column("streamable", sa.Boolean(), nullable=False),
        sa.Column("rung", sa.Integer(), nullable=False),
        sa.Column("rung_reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["probe_monitor_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["sae_id"], ["external_saes.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["calibration_dataset_id"], ["probe_monitor_datasets.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_probe_monitors_run_id", "probe_monitors", ["run_id"])
    op.create_index("ix_probe_monitors_run_selected", "probe_monitors", ["run_id", "selected"])

    op.create_table(
        "probe_monitor_evaluations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("probe_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),  # probe_monitor_datasets.id
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("n_positive", sa.Integer(), nullable=True),
        sa.Column("n_negative", sa.Integer(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("scores_path", sa.String(length=1000), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["probe_id"], ["probe_monitors.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["probe_monitor_datasets.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_probe_monitor_evaluations_probe_id", "probe_monitor_evaluations", ["probe_id"]
    )
    op.create_index(
        "ix_probe_monitor_evaluations_dataset_id", "probe_monitor_evaluations", ["dataset_id"]
    )

    op.create_table(
        "probe_monitor_judge_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("endpoint", sa.String(length=500), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("dataset_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("probe_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("celery_task_id", sa.String(length=155), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("parse_failures", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["probe_id"], ["probe_monitors.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_probe_monitor_judge_runs_probe_id", "probe_monitor_judge_runs", ["probe_id"]
    )


def downgrade() -> None:
    # Reverse dependency order: judge runs and evaluations point at probes,
    # probes at runs, runs at datasets.
    op.drop_index("ix_probe_monitor_judge_runs_probe_id", table_name="probe_monitor_judge_runs")
    op.drop_table("probe_monitor_judge_runs")

    op.drop_index(
        "ix_probe_monitor_evaluations_dataset_id", table_name="probe_monitor_evaluations"
    )
    op.drop_index("ix_probe_monitor_evaluations_probe_id", table_name="probe_monitor_evaluations")
    op.drop_table("probe_monitor_evaluations")

    op.drop_index("ix_probe_monitors_run_selected", table_name="probe_monitors")
    op.drop_index("ix_probe_monitors_run_id", table_name="probe_monitors")
    op.drop_table("probe_monitors")

    op.drop_index("ix_probe_monitor_runs_model_status", table_name="probe_monitor_runs")
    op.drop_table("probe_monitor_runs")

    op.drop_index("ix_probe_monitor_datasets_role", table_name="probe_monitor_datasets")
    op.drop_index("ix_probe_monitor_datasets_dataset_id", table_name="probe_monitor_datasets")
    op.drop_table("probe_monitor_datasets")
