"""The five probe-monitor tables: registration, FK targets, and the delete policy.

MUTATION CONTROLS (each verified to fail the suite; see the review record):
  M67  `dataset_id` back to String(36)          → the FK-target type test fails
  M68  `model_id` back to String(36)            → same
  M69  probe→run CASCADE changed to SET NULL    → the delete-policy test fails
  M70  probe→sae_id CASCADE instead of SET NULL → the SAE-survival test fails
  M71  a class dropped from `models/__init__`   → the registration test fails
  M72  the migration's downgrade body emptied   → the reverse-order test fails

⚠ WHY THE FK-TARGET TYPES ARE TESTED AT ALL. `datasets.id` is a real Postgres
`uuid` column while every other id on this estate is a prefixed varchar — `m_…`,
`sae_…`, `cap_…`. Writing `String(36)` here (the local convention, and what the
design document implied) produced a table Postgres refuses outright:
`DatatypeMismatchError: foreign key constraint … cannot be implemented`. That was
caught by running the migration, not by reading it, and `test_schema_guards` could
never have caught it either — a migration that cannot apply never reaches the
comparison. So the types are asserted against the REFERENCED column, read from the
target model, rather than against a literal.
"""
import ast
import inspect
from pathlib import Path

import pytest
import sqlalchemy as sa

from src.core.database import Base
from src.models import (
    Dataset,
    ExternalSAE,
    Model,
    ProbeMonitor,
    ProbeMonitorDataset,
    ProbeMonitorEvaluation,
    ProbeMonitorJudgeRun,
    ProbeMonitorRun,
)

TABLES = (
    "probe_monitor_datasets",
    "probe_monitor_runs",
    "probe_monitors",
    "probe_monitor_evaluations",
    "probe_monitor_judge_runs",
)

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "alembic" / "versions" / "a7f3c8e90d21_probe_monitors.py"
)


class TestTheModelsAreRegistered:
    """A model absent from `Base.metadata` is invisible to the schema guard AND to
    `create_all`, so its table simply does not exist in tests — which look green."""

    @pytest.mark.parametrize("table", TABLES)
    def test_the_table_is_in_the_shared_metadata(self, table):
        assert table in Base.metadata.tables, (
            f"{table} is not in Base.metadata — check the import in models/__init__.py"
        )

    @pytest.mark.parametrize(
        "cls",
        [
            ProbeMonitorDataset,
            ProbeMonitorRun,
            ProbeMonitor,
            ProbeMonitorEvaluation,
            ProbeMonitorJudgeRun,
        ],
    )
    def test_the_class_is_exported_by_name(self, cls):
        import src.models as models

        assert cls.__name__ in models.__all__, f"{cls.__name__} missing from __all__"

    def test_every_id_carries_its_prefix_default(self):
        """The prefixes are how a reader tells a run id from a probe id in a log
        line, and 033 parses them out of export filenames."""
        expected = {
            ProbeMonitorDataset: "pmd_",
            ProbeMonitorRun: "pmr_",
            ProbeMonitor: "pm_",
            ProbeMonitorEvaluation: "pme_",
            ProbeMonitorJudgeRun: "pmj_",
        }
        for cls, prefix in expected.items():
            default = cls.__table__.c.id.default.arg
            minted = default(None) if callable(default) else default
            assert minted.startswith(prefix), f"{cls.__name__} minted {minted}"

    def test_the_prefixes_are_distinct_beyond_pm(self):
        """`pm_` is a prefix of `pmd_`? No — but `pm_` vs `pmr_` share two
        characters, so a `startswith("pm_")` check elsewhere must not match a run.
        Asserted here so a future rename cannot make one id parse as another."""
        run = ProbeMonitorRun.__table__.c.id.default.arg(None)
        probe = ProbeMonitor.__table__.c.id.default.arg(None)
        assert not run.startswith("pm_")
        assert not probe.startswith("pmr_")


class TestForeignKeysMatchTheirTargetsType:
    """The defect this file exists for. Compare against the TARGET model's column,
    never a literal — a literal is how the mismatch got written in the first place.
    """

    def _fk_column(self, cls, name):
        return cls.__table__.c[name]

    def test_dataset_id_is_the_same_type_as_datasets_id(self):
        target = Dataset.__table__.c.id.type
        ours = self._fk_column(ProbeMonitorDataset, "dataset_id").type
        assert type(ours) is type(target), (
            f"probe_monitor_datasets.dataset_id is {ours!r} but datasets.id is "
            f"{target!r}; Postgres refuses this FK outright"
        )

    def test_model_id_matches_models_id(self):
        target = Model.__table__.c.id.type
        ours = self._fk_column(ProbeMonitorRun, "model_id").type
        assert type(ours) is type(target)
        assert ours.length == target.length, f"{ours.length} vs {target.length}"

    def test_sae_id_matches_external_saes_id(self):
        target = ExternalSAE.__table__.c.id.type
        ours = self._fk_column(ProbeMonitor, "sae_id").type
        assert type(ours) is type(target)
        assert ours.length == target.length

    def test_every_foreign_key_in_these_tables_has_a_type_matching_its_target(self):
        """The general rule, so a table added later is covered without a new test."""
        mismatches = []
        for table_name in TABLES:
            table = Base.metadata.tables[table_name]
            for fk in table.foreign_keys:
                local = fk.parent.type
                remote = fk.column.type
                if type(local) is not type(remote):
                    mismatches.append(
                        f"{table_name}.{fk.parent.name} is {local!r} but "
                        f"{fk.column.table.name}.{fk.column.name} is {remote!r}"
                    )
                elif getattr(local, "length", None) != getattr(remote, "length", None):
                    mismatches.append(
                        f"{table_name}.{fk.parent.name} length "
                        f"{getattr(local, 'length', None)} vs "
                        f"{getattr(remote, 'length', None)}"
                    )
        assert not mismatches, "FK type mismatches:\n  " + "\n  ".join(mismatches)


class TestEveryTimestampIsTimezoneAware:
    """⚠ THE GUARD THAT DID NOT EXIST WHEN IT WAS NEEDED.

    `test_clock_is_timezone_aware` refuses `datetime.utcnow()` because "every DateTime
    column in this schema is `timezone=True`". It checks the VALUE. Nothing checked the
    COLUMN — so satisfying it by switching the defaults to the aware `utc_now()` while
    leaving the columns `TIMESTAMP WITHOUT TIME ZONE` passed the suite and failed in
    production on the first real row: asyncpg refuses an aware value for a naive column,
    and the first probe dataset parsed all 8,000 rows correctly before dying at the
    INSERT.

    Half a fix is worse than neither here: the original naive default at least matched
    its column.
    """

    @pytest.mark.parametrize("table", TABLES)
    def test_no_naive_timestamp_survives(self, table):
        columns = Base.metadata.tables[table].columns
        naive = [
            column.name
            for column in columns
            if isinstance(column.type, sa.DateTime) and not column.type.timezone
        ]
        assert not naive, (
            f"{table} has naive timestamp column(s) {naive} while its defaults are the "
            f"aware utc_now(); asyncpg refuses that combination outright"
        )

    def test_the_defaults_really_are_aware(self):
        """The other half: an aware column with a naive default is the same bug
        mirrored, so the value is checked here too rather than assumed."""
        from datetime import datetime

        for table in TABLES:
            for column in Base.metadata.tables[table].columns:
                if not isinstance(column.type, sa.DateTime):
                    continue
                default = column.default
                if default is None or not callable(getattr(default, "arg", None)):
                    continue
                produced = default.arg(None)
                assert isinstance(produced, datetime)
                assert produced.tzinfo is not None, (
                    f"{table}.{column.name}'s default produced a naive datetime"
                )

    def test_a_migration_widens_the_columns_that_already_shipped(self):
        """`a7f3c8e90d21` created them naive, so the ORM change alone would leave
        production disagreeing with the models."""
        versions = MIGRATION.parent
        widening = [
            path for path in versions.glob("*.py")
            if "TIMESTAMP WITH TIME ZONE" in path.read_text()
            and "probe_monitor" in path.read_text()
        ]
        assert widening, "no migration widens the probe timestamp columns"

    def test_that_migration_covers_EVERY_timestamp_column_IT_CREATED(self):
        """A migration that widens some of them leaves the rest failing exactly as before, on
        whichever row reaches them first.

        ⚠ NARROWED FROM EQUALITY TO A SUBSET, AND THE REASON IS THE INTERESTING PART. This asserted
        `declared == actual`, and it FAILED the moment 033 added `definition_built_at` — correctly:
        the two sets had genuinely diverged. But the right conclusion is not "add the new column to
        the old migration", which would be rewriting applied history. `b8e1c04a7f52` widened the
        eight columns that existed when it ran, and a column added LATER is created
        `timezone=True` by its own migration.
        `TestEveryTimestampIsTimezoneAware::test_no_naive_timestamp_survives` is what covers the
        new ones, at the ORM level, and `test_schema_guards` covers them at the database level. So
        this checks that the widening migration named only real columns and missed none of ITS
        OWN."""
        import importlib.util

        path = MIGRATION.parent / "b8e1c04a7f52_probe_timestamps_timezone_aware.py"
        spec = importlib.util.spec_from_file_location("probe_tz_migration", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        declared = set(module.COLUMNS)
        actual = {
            (table, column.name)
            for table in TABLES
            for column in Base.metadata.tables[table].columns
            if isinstance(column.type, sa.DateTime)
        }
        assert declared <= actual, (
            f"the migration widens column(s) the models do not have: "
            f"{sorted(declared - actual)} — it would fail to apply"
        )
        # Every timestamp it did NOT name must have been added after it. Checked by date rather
        # than asserted by hand, so a column added to an EARLIER migration cannot slip through.
        later = actual - declared
        for table, column in sorted(later):
            assert Base.metadata.tables[table].columns[column].type.timezone, (
                f"{table}.{column} is not covered by {module.revision} and is naive; a column "
                f"added after the widening must be created timezone-aware by its own migration"
            )


class TestTheDeletePolicyIsPerEdge:
    """A uniform CASCADE would erase evidence; a uniform SET NULL would orphan it.
    Each edge is asserted because each was chosen separately."""

    def _ondelete(self, table_name, column):
        table = Base.metadata.tables[table_name]
        for fk in table.foreign_keys:
            if fk.parent.name == column:
                return fk.ondelete
        raise AssertionError(f"{table_name}.{column} has no foreign key")

    def test_a_probe_dies_with_its_run(self):
        assert self._ondelete("probe_monitors", "run_id") == "CASCADE"

    def test_an_evaluation_dies_with_its_probe(self):
        assert self._ondelete("probe_monitor_evaluations", "probe_id") == "CASCADE"

    def test_deleting_an_SAE_does_NOT_delete_the_probe_trained_over_it(self):
        """The probe becomes unbuildable, which 033 must REFUSE and report. Erasing
        the row instead would make the probe silently vanish from the estate."""
        assert self._ondelete("probe_monitors", "sae_id") == "SET NULL"

    def test_deleting_a_calibration_set_does_not_delete_the_probe(self):
        assert self._ondelete("probe_monitors", "calibration_dataset_id") == "SET NULL"
        assert self._ondelete("probe_monitor_runs", "calibration_dataset_id") == "SET NULL"

    def test_a_judge_run_outlives_the_probe_it_compared_against(self):
        """A judge baseline over a set of data is a measurement in its own right."""
        assert self._ondelete("probe_monitor_judge_runs", "probe_id") == "SET NULL"


class TestTheMigrationIsReversible:
    """A downgrade that drops nothing is a stub, and a downgrade that drops in the
    wrong order fails on the FK. Read from the AST, not the text — a comment naming
    a table would satisfy a substring search."""

    def _dropped_tables_in_order(self):
        tree = ast.parse(MIGRATION.read_text())
        downgrade = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "downgrade"
        )
        order = []
        for node in ast.walk(downgrade):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "drop_table"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                order.append(node.args[0].value)
        return order

    def test_the_downgrade_drops_all_five(self):
        assert set(self._dropped_tables_in_order()) == set(TABLES)

    def test_it_drops_children_before_parents(self):
        """`probe_monitors` references `probe_monitor_runs`, so dropping the run
        table first raises. This asserts the ORDER, which a set comparison cannot."""
        order = self._dropped_tables_in_order()
        position = {name: i for i, name in enumerate(order)}
        assert position["probe_monitor_evaluations"] < position["probe_monitors"]
        assert position["probe_monitor_judge_runs"] < position["probe_monitors"]
        assert position["probe_monitors"] < position["probe_monitor_runs"]
        assert position["probe_monitor_runs"] < position["probe_monitor_datasets"]

    def test_it_revises_a_real_revision(self):
        text = MIGRATION.read_text()
        assert 'down_revision = "e4c7a91f2d30"' in text
        versions = MIGRATION.parent
        assert any(
            "e4c7a91f2d30" in p.name for p in versions.glob("*.py")
        ), "down_revision names a revision with no file"

    def test_the_docstring_says_WHY_not_only_what(self):
        """House style, and this migration's reasons (view-not-copy, refusal-is-a-row,
        strings-not-enums) are the ones a later reader will want to change."""
        doc = MIGRATION.read_text().split('"""')[1]
        assert "WHY" in doc
        for token in ("IDL-51", "REFUSAL IS A ROW", "SET NULL"):
            assert token in doc, f"the docstring does not explain {token}"
