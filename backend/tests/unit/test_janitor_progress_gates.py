"""Every janitor's progress gate must be WIRED, not merely present.

The gates were added so a slow-but-advancing job is not reaped. They are also
the easiest thing in this change to ship dead: `conftest` stubs Redis away for
unit tests, so `progress_stalled_seconds` returns None everywhere and the gate
never fires. A suite can be fully green over six gates that do nothing.

So each test here patches the gate IN THE JANITOR'S OWN MODULE and asserts the
row is spared. Removing the gate from that janitor turns its test red — which is
the only thing that proves the capability exists for a caller.

Origin: on 2026-08-28 the third member of a three-SAE batch was failed at 186
minutes as a "crashed worker" while its batch was working normally. Every
janitor shared the same defect — a quiet row read as a dead job.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

NOW = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(hours=6)


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows

    def first(self):
        return None

    def count(self):
        return 0


class _DB:
    def __init__(self, rows):
        self.rows = rows
        self.commits = 0

    def query(self, *a, **k):
        return _Query(self.rows)

    def commit(self):
        self.commits += 1


class _Ctx:
    def __init__(self, db):
        self._db = db

    def __enter__(self):
        return self._db

    def __exit__(self, *a):
        return False


def drive(mod, task, row, *, stalled):
    """Run one janitor with its progress gate pinned, everything else inert."""
    with patch.object(task, "get_db", lambda: _Ctx(_DB([row]))), \
         patch.object(mod, "progress_stalled_seconds", lambda *a, **k: stalled), \
         patch("src.workers.task_heartbeat.task_looks_alive", lambda *a, **k: False):
        for emitter in (
            "emit_extraction_job_progress",
            "emit_training_progress",
            "emit_extraction_failed",
            "emit_tokenization_status",
            "emit_enhanced_labeling_progress",
        ):
            if hasattr(mod, emitter):
                setattr(mod, emitter, lambda **k: True)
        return task.run()


def terminal(row):
    """Did this janitor condemn the row? Works across the differing enums."""
    status = str(getattr(row, "status", "") or "")
    nlp = str(getattr(row, "nlp_status", "") or "")
    return "fail" in status.lower() or "error" in status.lower() or "fail" in nlp.lower()


# (module path, task attribute, row factory) — one per gated janitor.
def _training():
    return SimpleNamespace(
        id="train_1", status="running", celery_task_id="t1", current_step=10300,
        progress=20.6, updated_at=OLD, created_at=OLD, error_message=None,
        completed_at=None,
    )


def _activation():
    return SimpleNamespace(
        id="ext_1", status="EXTRACTING", celery_task_id="t1", samples_processed=4336,
        progress=44.7, updated_at=OLD, created_at=OLD, error_message=None,
        error_type=None, completed_at=None, model_id="m1", extraction_id="ext_1",
    )


def _tokenization():
    return SimpleNamespace(
        id="tok_1", status="PROCESSING", celery_task_id="t1", progress=80.0,
        updated_at=OLD, created_at=OLD, error_message=None, completed_at=None,
        dataset_id="ds1",
    )


def _labeling():
    return SimpleNamespace(
        id="lab_1", status="processing", phase="pass1", celery_task_id="t1",
        examples_completed=12, examples_total=40, updated_at=OLD, created_at=OLD,
        error_message=None, completed_at=None, feature_id="f1",
    )


CASES = [
    ("src.workers.cleanup_stuck_trainings", "cleanup_stuck_trainings_task", _training),
    ("src.workers.cleanup_stuck_activations", "cleanup_stuck_activations_task", _activation),
    ("src.workers.cleanup_stuck_tokenizations", "cleanup_stuck_tokenizations_task", _tokenization),
    ("src.workers.cleanup_stuck_enhanced_labeling", "cleanup_stuck_enhanced_labeling_task", _labeling),
]


@pytest.mark.parametrize("module_path,task_name,make_row", CASES)
class TestTheGateIsWired:
    def _load(self, module_path, task_name):
        import importlib

        mod = importlib.import_module(module_path)
        return mod, getattr(mod, task_name)

    def test_a_row_that_advanced_recently_is_spared(
        self, module_path, task_name, make_row
    ):
        """Six hours stale by the clock, but the counter moved a minute ago."""
        mod, task = self._load(module_path, task_name)
        row = make_row()

        drive(mod, task, row, stalled=60.0)

        assert not terminal(row), (
            f"{module_path} reaped a job whose progress counter had just "
            "advanced — its gate is not wired"
        )

    def test_a_row_whose_counter_is_frozen_is_still_reclaimed(
        self, module_path, task_name, make_row
    ):
        """The gate must not become a blanket amnesty."""
        mod, task = self._load(module_path, task_name)
        row = make_row()

        drive(mod, task, row, stalled=99999.0)

        assert terminal(row), (
            f"{module_path} spared a job that has not advanced in 27 hours"
        )


class TestTheCircuitGateIsWired:
    """Circuit runs decide through `_is_abandoned`, shared by every lifecycle."""

    def test_a_run_that_advanced_recently_is_not_abandoned(self):
        from src.workers import cleanup_stuck_circuit_runs as mod

        run = SimpleNamespace(
            id="cap_1", status="running", celery_task_id="t1", progress=45.6,
            updated_at=OLD, created_at=OLD,
        )
        with patch.object(mod, "progress_stalled_seconds", lambda *a, **k: 60.0):
            assert mod._is_abandoned(run, "running") is False

    def test_a_frozen_run_is_still_abandoned(self):
        from src.workers import cleanup_stuck_circuit_runs as mod

        run = SimpleNamespace(
            id="cap_1", status="running", celery_task_id=None, progress=45.6,
            updated_at=OLD, created_at=OLD,
        )
        with patch.object(mod, "progress_stalled_seconds", lambda *a, **k: 99999.0):
            assert mod._is_abandoned(run, "running") is True

    def test_a_lifecycle_with_no_counter_falls_through_unchanged(self):
        """Steering record runs have no progress column. The gate must be a
        no-op there, not a crash and not an amnesty."""
        from src.workers import cleanup_stuck_circuit_runs as mod

        run = SimpleNamespace(id="rec_1", status="running", celery_task_id=None,
                              updated_at=OLD, created_at=OLD)
        assert mod._is_abandoned(run, "running") is True
