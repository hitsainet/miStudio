"""Revision b7e41c2d9a05 (Feature 041 S2, FTASKS 5.2): the migrated schema carries exactly the ORM's columns.

`test_orm_matches_migrated_schema` audits constraints of three older tables only, so a column the ORM declares
but the migration forgot was invisible to the suite (control 5.2 survived: `rows_dir` dropped from the
migration, suite green). This reflects `SCHEMA_CHECK_DATABASE_URL` (migrated to head) and compares the four
041 tables column for column, nullability included, plus the append-only trigger on the history table.
"""
from __future__ import annotations

import pytest
import sqlalchemy as sa

from src.models import probe_blend as models
from tests.support.test_databases import migrated_url

TABLES = (models.ProbeBlendReport, models.ProbeBlendReportMember, models.ProbeSuggestedBlend,
          models.ProbeSuggestedBlendHistory)


@pytest.fixture(scope="module")
def inspector():
    engine = sa.create_engine(migrated_url().replace("+asyncpg", "").replace("+psycopg2", ""))
    try:
        yield sa.inspect(engine), engine
    finally:
        engine.dispose()


@pytest.mark.parametrize("model", TABLES, ids=lambda m: m.__tablename__)
def test_columns_match_the_orm(inspector, model):
    found, _engine = inspector
    table = model.__table__
    migrated = {c["name"]: c["nullable"] for c in found.get_columns(table.name)}
    declared = {c.name: c.nullable for c in table.columns}
    assert migrated == declared


def test_history_trigger_installed(inspector):
    _found, engine = inspector
    with engine.connect() as conn:
        names = conn.execute(sa.text("SELECT tgname FROM pg_trigger WHERE tgrelid = "
                                     "'probe_suggested_blend_history'::regclass AND NOT tgisinternal")).scalars().all()
    assert models.HISTORY_TRIGGER in names
