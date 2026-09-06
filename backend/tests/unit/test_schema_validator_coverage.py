"""The startup schema validator must cover the schema it claims to.

WHY THIS EXISTS

`REQUIRED_TABLES` was a hand-maintained literal of 17 tables against 36 mapped
ones. Every table added in the last year was unchecked, so the validator logged
"Schema validation passed" on a database missing `circuits`,
`validation_manifests`, `cluster_profiles`, `steering_record_runs`,
`agent_approval_requests` and `dismissed_operations` (MIS-E2E-032).

It also had **no tests at all** — mutation M2 removed `"models"` from the dict
and 155 tests stayed green (MIS-E2E-051). And PADR IDL-16 claims the validator
"compares live DB schema against SQLAlchemy model metadata", which it did not
(MIS-E2E-157).

The list is now derived from `Base.metadata`. These tests pin that it stays
derived — a future refactor back to a literal fails here.

NEGATIVE CONTROL: replace `_required_tables()` with a literal subset and
`test_every_mapped_table_is_checked` must fail. Verified 2026-08-23.
"""

import pytest

from src.core.database import Base
from src.db.schema_validator import _required_tables, validate_schema
from src.models import *  # noqa: F401,F403 — registers every table


class TestTheValidatorCoversTheWholeSchema:
    def test_every_mapped_table_is_checked(self):
        """Derived, not hand-listed. This is the whole point."""
        mapped = set(Base.metadata.tables)
        checked = set(_required_tables())
        missing = mapped - checked
        assert not missing, (
            f"{len(missing)} mapped tables are not checked by the startup "
            f"validator: {sorted(missing)}. The list must be derived from "
            f"Base.metadata, never hand-maintained."
        )

    def test_it_covers_the_tables_that_were_missing(self):
        """The specific tables MIS-E2E-032 found unchecked.

        Named explicitly so a regression is legible in the failure output
        rather than showing up as an off-by-N in the count above.
        """
        checked = _required_tables()
        for table in (
            "circuits",
            "validation_manifests",
            "cluster_profiles",
            "steering_record_runs",
            "agent_approval_requests",
            "dismissed_operations",
        ):
            assert table in checked, f"{table} is not covered by the validator"

    def test_it_checks_a_meaningful_column_set(self):
        """A table entry with no columns would pass vacuously."""
        for table, cols in _required_tables().items():
            assert cols, f"{table} has an empty required-column list"

    def test_the_required_columns_are_actually_non_nullable(self):
        """The bar is NOT NULL: a missing nullable column is survivable at
        runtime, a missing NOT NULL column is not."""
        required = _required_tables()
        for name, table in Base.metadata.tables.items():
            non_nullable = {c.name for c in table.columns if not c.nullable}
            if not non_nullable:
                continue  # falls back to the primary key; covered above
            assert set(required[name]) == non_nullable, (
                f"{name}: required columns drifted from the non-nullable set"
            )


class TestTheValidatorActuallyDetectsAMissingTable:
    async def test_a_missing_table_is_reported(self, async_session):
        """The behaviour the validator exists for, exercised end to end.

        M2 removed a table from the list and nothing failed, because nothing
        ever called `validate_schema`. This calls it.
        """
        is_valid, missing_tables, _missing_cols = await validate_schema(
            async_session,
            required_tables={"a_table_that_does_not_exist": ["id"]},
            raise_on_error=False,
        )
        assert is_valid is False
        assert "a_table_that_does_not_exist" in missing_tables

    async def test_a_present_table_passes(self, async_session):
        """The negative half — otherwise the test above passes on a validator
        that reports everything as missing."""
        is_valid, missing_tables, missing_cols = await validate_schema(
            async_session,
            required_tables={"features": ["id"]},
            raise_on_error=False,
        )
        assert is_valid is True, f"missing={missing_tables} cols={missing_cols}"
