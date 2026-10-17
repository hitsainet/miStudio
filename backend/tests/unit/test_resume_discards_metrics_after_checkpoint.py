"""A resume re-runs every step after its checkpoint, so the metrics logged past it go first.

Review round 1 (R1-A), 2026-09-15. `training_metrics` is unique on
(training_id, step, layer_idx). A pause lands wherever the operator presses it —
with the defaults (log_interval 100, checkpoint_interval 1000) typically several
logged steps after the newest checkpoint — and the resumed run starts at
checkpoint + 1 and logs those steps again. The per-layer insert then raised
IntegrityError, which the task records as a FAILED training: full resume failed on
the ordinary pause. The resume-equivalence tests could not see it, because their
fake session appended anything and their pause fell on the step right after the
checkpoint, where nothing had been logged yet.

These tests run against REAL Postgres (the table's own DDL, less its foreign key),
in a database named after this run's DATABASE_URL_SYNC so concurrent suites keep
apart. They fail, never skip, when Postgres is unreachable.

MUTATION CONTROLS (R1-A; the target applied alone, this file and the
resume-equivalence file run, bytes restored, sha256 verified):
  C1 the call in the resume branch removed     -> the equivalence pause-at-13 test (IntegrityError)
  C2 `step > int(step)` -> `>=`                 -> test_discard_removes_..., both equivalence configs,
                                                   the pause-at-13 test
"""

import contextlib
import os
import re

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateIndex, CreateTable

from src.models.training_metric import TrainingMetric
from src.workers import base_task, training_tasks


def _database():
    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio"
    server, name = url.rsplit("/", 1)
    return server, f"{name}_resume_metrics"


@pytest.fixture
def factory(monkeypatch):
    server, name = _database()
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()

    engine = create_engine(server + "/" + name)
    ddl = str(CreateTable(TrainingMetric.__table__).compile(engine))
    ddl = re.sub(r",\s*FOREIGN KEY\s*\([^)]*\)\s*REFERENCES\s+\w+\s*\([^)]*\)(\s+ON DELETE \w+)?", "", ddl)
    # The table these tests rely on: the mapped columns and the unique key. The key is
    # an expression INDEX since review R1-A A5 (it includes COALESCE(hook_type, '')), so
    # CreateTable no longer carries it: the table's indexes are created beside it.
    indexes = {index.name: index for index in TrainingMetric.__table__.indexes}
    assert "FOREIGN KEY" not in ddl and "uq_training_metrics_tid_step_layer_hook" in indexes, (ddl, sorted(indexes))
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS training_metrics"))
        conn.execute(text(ddl))
        for index in indexes.values():
            conn.execute(CreateIndex(index))

    make = sessionmaker(bind=engine, expire_on_commit=False)

    @contextlib.contextmanager
    def session():
        db = make()
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    monkeypatch.setattr(base_task, "get_sync_db", session)
    yield session
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS training_metrics"))
    engine.dispose()


def _log(training_id, step, layer_idx, loss=1.0):
    training_tasks.train_sae_task.log_metric(
        training_id=training_id, step=step, loss=loss, layer_idx=layer_idx,
    )


def _rows(factory, training_id):
    with factory() as db:
        return sorted(
            (m.step, -10**9 if m.layer_idx is None else m.layer_idx)
            for m in db.query(TrainingMetric).filter(TrainingMetric.training_id == training_id).all()
        )


def test_the_table_refuses_a_step_logged_twice(factory):
    """The failure mode, through the task's own write path."""
    _log("train_a", 11, 0)
    with pytest.raises(IntegrityError, match="uq_training_metrics_tid_step_layer"):
        _log("train_a", 11, 0)


def test_discard_removes_this_trainings_rows_after_the_step_and_nothing_else(factory):
    for step in (9, 10, 11, 12):
        for layer in (None, 0, -1):  # aggregated, per-layer, held-out
            _log("train_a", step, layer)
    for step in (11, 12):
        for layer in (None, 0, -1):
            _log("train_b", step, layer)

    with factory() as db:
        removed = training_tasks.discard_metrics_after_step(db, "train_a", 10)

    assert removed == 6
    # The checkpoint's own step stays: it ran before the checkpoint was written.
    assert _rows(factory, "train_a") == sorted((s, l) for s in (9, 10) for l in (-10**9, 0, -1))
    assert len(_rows(factory, "train_b")) == 6
    # And the step the resumed run logs next is accepted.
    _log("train_a", 11, 0)
