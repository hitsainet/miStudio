"""
The tokenization janitor: reachability, and what it does to a stranded row.

WHY THIS EXISTS. Tokenization was the one long-running status with no janitor.
Six others have one. A tokenization whose worker died therefore held PROCESSING
forever -- a frozen progress bar, no error, no retry, and the parent dataset
stuck PROCESSING alongside it. Observed live on 2026-08-25: a 446,762-sample job
reached 100%, lost its worker, and sat at "Processing 80.0%" with nothing in the
system that would ever change it.

A janitor is the archetype of a capability that can be fully implemented and
never fire. Nothing goes red when it doesn't run; it just stays dark, which is
the failure it exists to prevent. So reachability is asserted against the live
registry after autodiscovery, in a fresh interpreter, per the repo rule.

MUTATION CONTROLS (each must go red):
  * drop the module from autodiscover_tasks   -> registration test fails
  * delete the beat entry                     -> schedule test fails
  * delete the short-name route               -> routing test fails
  * skip the dataset release                  -> dataset test fails
  * drop the task_looks_alive guard           -> live-task test fails
"""

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.core.celery_app import celery_app
from src.models.dataset import DatasetStatus
from src.models.dataset_tokenization import TokenizationStatus

TASK_NAME = "cleanup_stuck_tokenizations"
BEAT_ENTRY = "cleanup-stuck-tokenizations"


def _queue_name(route: dict) -> str:
    q = route.get("queue")
    return getattr(q, "name", str(q))


class TestTheJanitorIsReachable:
    def test_autodiscovery_registers_it_in_a_fresh_interpreter(self):
        """
        The strong form. A subprocess that imports ONLY the celery app proves
        the task is reachable through autodiscovery -- not because some other
        test in this session happened to import the module.
        """
        code = (
            "from src.core.celery_app import celery_app;"
            "celery_app.loader.import_default_modules();"
            f"print({TASK_NAME!r} in celery_app.tasks)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
        )
        assert out.stdout.strip().endswith("True"), (
            f"{TASK_NAME} is not reachable through autodiscovery; beat would "
            f"fire a name no worker can execute.\nstderr: {out.stderr[-600:]}"
        )

    def test_it_is_scheduled(self):
        entry = celery_app.conf.beat_schedule.get(BEAT_ENTRY)
        assert entry is not None, (
            "the tokenization janitor has no beat entry, so nothing reclaims a "
            "row whose worker died and the dataset stays busy indefinitely"
        )
        assert entry["task"] == TASK_NAME
        assert entry["schedule"] > 0

    def test_it_routes_off_the_gpu_queue_without_beat_options(self):
        """task_routes globs match the TASK NAME; a short name needs an exact
        entry, or a direct .delay() lands CPU work on the GPU worker."""
        assert _queue_name(celery_app.amqp.router.route({}, TASK_NAME)) == (
            "low_priority"
        )


def _tok(**kw):
    base = dict(
        id="tok_abc_m_1_512",
        dataset_id="ds-1",
        status=TokenizationStatus.PROCESSING,
        celery_task_id=None,
        error_message=None,
        completed_at=None,
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=180),
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Query:
    """Minimal stand-in: .all() yields the stuck rows, .first() the dataset,
    .count() the sibling count."""

    def __init__(self, db, model):
        self._db, self._model = db, model

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._db.stuck

    def first(self):
        return self._db.dataset

    def count(self):
        return self._db.sibling_count


class _DB:
    def __init__(self, stuck, dataset, sibling_count=0):
        self.stuck, self.dataset, self.sibling_count = stuck, dataset, sibling_count
        self.commits = 0

    def query(self, model):
        return _Query(self, model)

    def commit(self):
        self.commits += 1


class _Ctx:
    def __init__(self, db):
        self._db = db

    def __enter__(self):
        return self._db

    def __exit__(self, *a):
        return False


def _run(db):
    """bind=True means `self` is the task instance, so patching get_db on the
    task object is what injects the fake session."""
    from src.workers import cleanup_stuck_tokenizations as mod

    with patch.object(
        mod.cleanup_stuck_tokenizations_task, "get_db", lambda: _Ctx(db)
    ), patch.object(mod, "emit_tokenization_status", lambda **kw: True):
        return mod.cleanup_stuck_tokenizations_task.run()


class TestItReclaimsAStrandedRow:
    def test_a_dead_tokenization_becomes_an_error(self):
        tok = _tok()
        db = _DB([tok], SimpleNamespace(id="ds-1", status=DatasetStatus.PROCESSING))

        assert _run(db) == {"cleaned": 1}
        assert tok.status == TokenizationStatus.ERROR
        assert tok.completed_at is not None
        assert "Re-run" in (tok.error_message or "")

    def test_it_releases_the_parent_dataset(self):
        """The row alone is not enough -- the card badge reads the dataset."""
        tok = _tok()
        dataset = SimpleNamespace(id="ds-1", status=DatasetStatus.PROCESSING)
        db = _DB([tok], dataset)

        _run(db)
        assert dataset.status == DatasetStatus.READY, (
            "the dataset stayed PROCESSING, so the card still shows a busy "
            "dataset with no job behind it"
        )

    def test_it_leaves_the_dataset_alone_while_a_sibling_is_live(self):
        tok = _tok()
        dataset = SimpleNamespace(id="ds-1", status=DatasetStatus.PROCESSING)
        db = _DB([tok], dataset, sibling_count=1)

        _run(db)
        assert dataset.status == DatasetStatus.PROCESSING, (
            "released a dataset that still has a running tokenization"
        )

    def test_it_skips_a_row_whose_task_is_still_alive(self):
        """A slow job is not a dead one."""
        tok = _tok(celery_task_id="live-task")
        db = _DB([tok], SimpleNamespace(id="ds-1", status=DatasetStatus.PROCESSING))

        with patch("src.workers.task_heartbeat.task_looks_alive", return_value=True):
            assert _run(db) == {"cleaned": 0}
        assert tok.status == TokenizationStatus.PROCESSING
