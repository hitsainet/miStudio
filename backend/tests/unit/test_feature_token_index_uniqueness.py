"""OSD-11 — one feature_token_index row per (extraction, feature, token_rank).

The hard-negative donor query selects donors `WHERE normalized_token = <target's>
AND token_rank = 1`. Nothing stopped two rows existing for one feature's rank-1
token, and a grouping run that fails partway and is re-run is exactly how that
happens. The duplicate is then drawn twice and one donor's passages carry double
weight in every detection score — silently.

Measured before adding the constraint (production, 2026-09-25): 117,528 rows and
117,528 distinct keys, so no dedupe was needed.
"""
import os

import psycopg2
import pytest

from src.models.feature_grouping import FeatureTokenIndex

CONSTRAINT = "uq_fti_extraction_feature_rank"
EXPECTED_COLUMNS = ["extraction_id", "feature_id", "token_rank"]


class TestTheOrmDeclaresIt:

    def test_the_constraint_is_on_the_table(self):
        found = {
            constraint.name: [column.name for column in constraint.columns]
            for constraint in FeatureTokenIndex.__table__.constraints
            if constraint.name == CONSTRAINT
        }
        assert CONSTRAINT in found, (
            f"{CONSTRAINT} is not declared on feature_token_index; a re-run "
            f"grouping job can duplicate a donor and double-weight its passages"
        )
        assert sorted(found[CONSTRAINT]) == sorted(EXPECTED_COLUMNS)


class TestTheMigratedDatabaseHasIt:
    """The ORM declaring it proves nothing about the database the app talks to.

    This reads the MIGRATED schema — the same database the parity guards compare
    against — so it fails if the model was changed without a migration.
    """

    @pytest.fixture
    def migrated_dsn(self):
        dsn = os.getenv("SCHEMA_CHECK_DATABASE_URL")
        if not dsn:
            pytest.skip("SCHEMA_CHECK_DATABASE_URL is not set")
        scheme, separator, rest = dsn.partition("://")
        return f"{scheme.split('+', 1)[0]}{separator}{rest}"

    def test_the_constraint_exists_in_the_migrated_schema(self, migrated_dsn):
        connection = psycopg2.connect(migrated_dsn)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT c.conname,
                           array_agg(a.attname ORDER BY a.attname) AS cols
                    FROM pg_constraint c
                    JOIN pg_class t ON t.oid = c.conrelid
                    JOIN unnest(c.conkey) AS k(attnum) ON TRUE
                    JOIN pg_attribute a
                      ON a.attrelid = t.oid AND a.attnum = k.attnum
                    WHERE t.relname = 'feature_token_index'
                      AND c.contype = 'u'
                    GROUP BY c.conname
                    """
                )
                rows = dict(cursor.fetchall())
        finally:
            connection.close()

        assert CONSTRAINT in rows, (
            "the migrated database has no unique constraint on "
            f"feature_token_index; found {sorted(rows)}. The ORM declaring one is "
            "not the same as the database enforcing it"
        )
        assert rows[CONSTRAINT] == sorted(EXPECTED_COLUMNS)
