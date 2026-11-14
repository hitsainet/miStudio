"""The remediation's three columns match their migration in type, nullability and comment (review R1-C).

`test_orm_matches_migrated_schema.py` is the general drift guard, and it did not
notice the ORM declaring `trainings.current_fvu_centred` an INTEGER over a DOUBLE
PRECISION column (mutation C35 survived it). An integer column would round every
FVU to 0 or 1 on write — a silent loss of the one number the column exists for —
so these three are checked here directly against the MIGRATED database.

NEGATIVE CONTROL (applied alone, this file run, restored, sha256 verified):
  C35 `current_fvu_centred = Column(Integer, ...)` in models/training.py -> RED here.
"""

import os

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from src.models.training import Training
from src.models.training_metric import TrainingMetric

COLUMNS = [
    (TrainingMetric, "fvu_centred", sa.Float),
    (Training, "current_fvu_centred", sa.Float),
    (Training, "evaluation", postgresql.JSONB),
]


@pytest.fixture(scope="module")
def inspector():
    url = os.environ.get("SCHEMA_CHECK_DATABASE_URL") or os.environ.get("DATABASE_URL_SYNC")
    if not url:
        pytest.skip("no migrated database configured")
    engine = sa.create_engine(url)
    try:
        yield sa.inspect(engine)
    finally:
        engine.dispose()


@pytest.mark.parametrize("model, name, kind", COLUMNS, ids=[c[1] for c in COLUMNS])
def test_the_orm_column_is_the_migrated_column(inspector, model, name, kind):
    table = model.__table__.name
    reflected = {c["name"]: c for c in inspector.get_columns(table)}
    assert name in reflected, f"{table}.{name} is not in the migrated database"
    column = model.__table__.c[name]

    assert isinstance(column.type, kind), f"ORM {table}.{name} is {column.type!r}"
    assert isinstance(reflected[name]["type"], kind), f"migrated {table}.{name} is {reflected[name]['type']!r}"
    assert column.nullable is True and reflected[name]["nullable"] is True
    assert column.comment and reflected[name]["comment"] == column.comment
