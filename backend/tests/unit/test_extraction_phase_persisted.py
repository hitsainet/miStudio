"""An activation extraction's step survives a reload, the merge heartbeats, and
the janitor judges a started job by its own clock.

Reported by the user, 2026-09-13: "I just saw a more detailed status that said it
was in the statistics phase. But when I left the page and came back, it was
'Extracting 20000/20000' again." And earlier: "I wish it would change from
'extracting' to 'merging' then to 'statistics'".

Three defects, each pinned below.

1. THE STEP LIVED ONLY IN WEBSOCKET MESSAGES. After the GPU pass reaches N/N the
   job merges its per-batch shards, then computes statistics, while the row's
   status stays EXTRACTING and samples_processed stays N/N. Nothing stored the
   step, so a reload fell back to "Extracting N/N". The merge did not even have
   a WebSocket message. `activation_extractions.phase` now records extracting,
   merging or statistics, through `record_progress`, so a terminal row is never
   moved back.

2. THE MERGE WROTE NOTHING TO THE ROW. Statistics had a 60 s heartbeat; the merge
   (disk-bound, long on a 10,000 x 2,048-token extraction) had none, so the row
   went stale for its whole duration.

3. THE JANITOR COULD NOT REAP A DEAD EXTRACTION. It passed
   ``started=str(status).lower() in ("extracting", "running", "processing")``.
   The column loads an ``ExtractionStatus`` member and ``str()`` of that is
   ``"ExtractionStatus.EXTRACTING"``, so ``started`` was always False,
   ``task_looks_alive`` never read the row's age, Celery's PENDING for a dead
   worker read as alive, and nothing was ever reaped. The existing janitor
   fixture (`test_janitor_progress_gates._activation`) used the plain string
   "EXTRACTING", for which the expression happens to work: a fixture agreeing
   with the defect by construction.

MUTATION CONTROLS, 2026-09-13. Each was applied to the committed tree (the
target text required to occur exactly once, so each one provably landed), this
file was run, and the original bytes were restored and hash-checked; `git status`
was clean afterwards. Every one turned this file red:

  M1  heartbeat writes only updated_at, no phase (the pre-change statistics
      heartbeat)                                  -> 5 failed (first-call phase
      x2, three phases in order, throttle, real merge loop)
  M1b merge heartbeat writes phase "extracting"   -> 4 failed
  M2  merge loop drops the per-copy callback      -> 2 failed (real merge loop
      write count 4 -> 1, announce-each-layer calls)
  M2b task builds the merge heartbeat but does not pass
      merge_progress_callback                     -> 1 failed (task wiring)
  M3  janitor `started` reverted to the old expression
                                                  -> 5 failed (dead job reaped
      x3 phases, loading and saving)
  M4  "phase" removed from the active-extraction response
                                                  -> 2 failed
  M6  phase write bypasses record_progress (direct unguarded setattr+commit)
                                                  -> 9 failed (terminal row
      refused x6, helper reports refusal, plus the two throttle tests, whose
      record_progress patch the bypass no longer goes through)
  M7  mark_completed leaves the phase             -> 1 failed
  M8  phase write drops updated_at (the previous round's statistics-heartbeat
      fix)                                        -> 3 failed
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio
import numpy as np
import pytest

import _cancel_ast as A
from src.models.activation_extraction import ExtractionStatus
from src.services.activation_service import ActivationService
from src.services.extraction_db_service import ExtractionDatabaseService
from src.workers import model_tasks as MT

LONG_AGO = datetime.now(timezone.utc) - timedelta(hours=2)


# ── fakes ──────────────────────────────────────────────────────────────────

class _Session:
    """Just enough of a SQLAlchemy session for record_progress, the DB service
    and the janitor's query. Filters are ignored: there is one row."""

    def __init__(self, row):
        self.row = row
        self.commits = 0

    def query(self, *_):
        return self

    def filter(self, *a, **k):
        return self

    def filter_by(self, **k):
        return self

    def populate_existing(self):
        return self

    def first(self):
        return self.row

    def all(self):
        return [self.row]

    def commit(self):
        self.commits += 1

    def refresh(self, _):
        pass


@contextmanager
def _ctx(session):
    yield session


def _row(status=ExtractionStatus.EXTRACTING, phase=None, updated_at=None):
    return SimpleNamespace(
        id="ext_1", model_id="m_1", status=status, phase=phase, progress=90.0,
        samples_processed=20000, updated_at=updated_at or LONG_AGO,
        celery_task_id="celery-1", error_message=None, error_type=None,
        completed_at=None, statistics={}, saved_files=[],
    )


def _never_cancelled(polls=None):
    def raise_if_cancelled(detail=""):
        if polls is not None:
            polls.append(detail)

    return SimpleNamespace(raise_if_cancelled=raise_if_cancelled)


@pytest.fixture
def row_db(monkeypatch):
    """Route `record_progress`'s own session (db=None) to a one-row fake."""

    def install(row):
        session = _Session(row)
        monkeypatch.setattr("src.core.database.get_sync_db", lambda: _ctx(session))
        return session

    return install


@pytest.fixture
def emitted(monkeypatch):
    sent = []
    monkeypatch.setattr(MT, "emit_extraction_progress", lambda **k: sent.append(k))
    return sent


HEARTBEATS = {
    "merging": MT.build_merge_heartbeat,
    "statistics": MT.build_statistics_heartbeat,
}


# ── 1. each transition writes the row ──────────────────────────────────────

class TestEachTransitionWritesThePhase:
    def test_the_gpu_pass_records_extracting(self, emitted):
        row = _row(status=ExtractionStatus.LOADING, phase=None)
        session = _Session(row)
        task = SimpleNamespace(get_db=lambda: _ctx(session))

        callback = MT.build_extraction_progress_callback(task, "m_1", "ext_1", _never_cancelled())
        callback(10, 20000)

        assert row.status == ExtractionStatus.EXTRACTING
        assert row.phase == "extracting"
        assert emitted[-1]["phase"] == "extracting"

    @pytest.mark.parametrize("phase", ["merging", "statistics"])
    def test_a_post_gpu_phase_is_written_on_its_first_call(self, row_db, emitted, phase):
        """The first call is the transition, so it is written at once, not a
        minute later, and it moves the clock the janitor reads."""
        row = _row(phase="extracting")
        session = row_db(row)

        heartbeat = HEARTBEATS[phase]("m_1", "ext_1", _never_cancelled(), clock=lambda: 0.0)
        heartbeat(0, 3, 0, 40)

        assert row.phase == phase
        assert row.updated_at > LONG_AGO
        assert row.status == ExtractionStatus.EXTRACTING, "status must stay EXTRACTING"
        assert session.commits == 1
        assert [m["phase"] for m in emitted] == [phase]

    def test_the_three_phases_follow_each_other_on_one_row(self, row_db, emitted):
        row = _row(status=ExtractionStatus.LOADING, phase=None)
        row_db(row)
        seen = []

        task = SimpleNamespace(get_db=lambda: _ctx(_Session(row)))
        MT.build_extraction_progress_callback(task, "m_1", "ext_1", _never_cancelled())(20000, 20000)
        seen.append(row.phase)
        MT.build_merge_heartbeat("m_1", "ext_1", _never_cancelled(), clock=lambda: 0.0)(0, 2, 0, 5)
        seen.append(row.phase)
        MT.build_statistics_heartbeat("m_1", "ext_1", _never_cancelled(), clock=lambda: 0.0)(0, 2, 0, 1)
        seen.append(row.phase)

        assert seen == ["extracting", "merging", "statistics"]

    def test_completion_clears_the_phase(self):
        row = _row(phase="statistics")
        ExtractionDatabaseService.mark_completed(
            db=_Session(row), extraction_id="ext_1", statistics={}, saved_files=["a.npy"]
        )
        assert row.status == ExtractionStatus.COMPLETED
        assert row.phase is None, "a completed row must not claim a step is still running"

    def test_failure_keeps_the_phase_it_stopped_in(self):
        row = _row(phase="merging")
        ExtractionDatabaseService.mark_failed(db=_Session(row), extraction_id="ext_1", error_message="disk full")
        assert row.status == ExtractionStatus.FAILED
        assert row.phase == "merging"

    def test_an_unknown_phase_is_refused(self):
        with pytest.raises(ValueError):
            MT.write_extraction_phase("ext_1", "saving")
        with pytest.raises(ValueError):
            ExtractionDatabaseService.update_progress(
                db=_Session(_row()), extraction_id="ext_1", progress=50.0,
                status=ExtractionStatus.EXTRACTING, samples_processed=1, phase="bogus",
            )


class TestATerminalRowIsNeverMovedBack:
    """A heartbeat in flight when the operator presses Stop must not revive the row.

    The cancel checker is stubbed to NOT raise, which is exactly the race: the
    endpoint has written CANCELLED and the task has not polled yet.
    """

    @pytest.mark.parametrize("phase", ["merging", "statistics"])
    @pytest.mark.parametrize(
        "status", [ExtractionStatus.CANCELLED, ExtractionStatus.FAILED, ExtractionStatus.COMPLETED]
    )
    def test_a_phase_write_onto_a_terminal_row_is_refused(self, row_db, emitted, status, phase):
        row = _row(status=status, phase="extracting")
        session = row_db(row)

        HEARTBEATS[phase]("m_1", "ext_1", _never_cancelled(), clock=lambda: 0.0)(0, 3, 0, 10)

        assert row.status == status
        assert row.phase == "extracting"
        assert row.updated_at == LONG_AGO
        assert session.commits == 0

    def test_the_write_helper_reports_the_refusal(self, row_db):
        row_db(_row(status=ExtractionStatus.CANCELLED, phase="statistics"))
        assert MT.write_extraction_phase("ext_1", "merging") is False


# ── 2. the merge heartbeats on time ────────────────────────────────────────

def _capture_writes(monkeypatch):
    writes = []
    monkeypatch.setattr(
        MT, "record_progress", lambda kind, target, **kw: writes.append((kind, target, kw)) or True
    )
    return writes


def _shards(tmp_path, n_layers=2, n_batches=5, rows=2):
    """Real per-batch .npy files, as the GPU pass leaves them."""
    rng = np.random.default_rng(0)
    accumulated, expected = {}, {}
    for layer in range(n_layers):
        name = f"layer_{layer}_residual"
        arrays = [rng.normal(size=(rows, 3, 4)).astype(np.float16) for _ in range(n_batches)]
        paths = []
        for b, array in enumerate(arrays):
            path = tmp_path / f"tmp_{name}_batch{b}.npy"
            np.save(path, array)
            paths.append(str(path))
        accumulated[name] = paths
        expected[name] = np.concatenate(arrays, axis=0)
    return accumulated, expected


class TestTheMergeHeartbeatsOnTime:
    def test_writes_are_throttled_on_time(self, monkeypatch, emitted):
        writes = _capture_writes(monkeypatch)
        polls = []
        clock = iter([0.0, 10.0, 59.0, 61.0, 90.0, 125.0]).__next__
        heartbeat = MT.build_merge_heartbeat("m_1", "ext_1", _never_cancelled(polls), clock=clock, interval_s=60)

        for i in range(6):
            heartbeat(0, 3, i, 100)

        assert len(writes) == 3, "expected writes at t=0, t=61 and t=125 only"
        for kind, target, fields in writes:
            assert (kind, target) == ("activation_extraction", "ext_1")
            assert fields["phase"] == "merging"
            assert "updated_at" in fields, "the write must move the janitor's clock"
        assert len(polls) == 6, "the throttle must not delay a Stop"
        assert len(emitted) == 3

    def test_a_cancel_stops_before_anything_is_written(self, monkeypatch, emitted):
        writes = _capture_writes(monkeypatch)

        class Cancelled(BaseException):
            pass

        def raise_if_cancelled(detail=""):
            raise Cancelled(detail)

        heartbeat = MT.build_merge_heartbeat(
            "m_1", "ext_1", SimpleNamespace(raise_if_cancelled=raise_if_cancelled), clock=lambda: 0.0
        )
        with pytest.raises(Cancelled, match="stopped merging batch files at layer 2 of 3"):
            heartbeat(1, 3, 0, 10)
        assert writes == []

    def test_the_real_merge_loop_heartbeats_on_time(self, tmp_path, monkeypatch, emitted):
        """Drives the service's own merge over real shard files with a fake clock.

        2 layers x 5 batches = 12 callbacks (one before each layer, one after each
        copy), 20 s apart: t = 0, 20, ..., 220. At a 60 s interval the row is
        written at t = 0, 60, 120 and 180, which is 4 writes.
        """
        writes = _capture_writes(monkeypatch)
        polls = []
        ticks = iter([20.0 * k for k in range(12)])
        heartbeat = MT.build_merge_heartbeat(
            "m_1", "ext_1", _never_cancelled(polls), clock=lambda: next(ticks), interval_s=60
        )
        accumulated, expected = _shards(tmp_path)

        svc = ActivationService.__new__(ActivationService)
        merged = svc._merge_batch_files(accumulated, tmp_path, on_merge_progress=heartbeat)

        assert len(polls) == 12
        assert len(writes) == 4
        assert all(fields["phase"] == "merging" for _, _, fields in writes)
        for name, array in expected.items():
            np.testing.assert_array_equal(np.asarray(merged[name]), array)
        assert not any((tmp_path / f"tmp_{name}_batch0.npy").exists() for name in expected), (
            "shards must be deleted as they are merged"
        )

    def test_the_merge_announces_each_layer_before_copying(self, tmp_path):
        calls = []
        accumulated, _ = _shards(tmp_path, n_layers=2, n_batches=3)

        svc = ActivationService.__new__(ActivationService)
        svc._merge_batch_files(accumulated, tmp_path, on_merge_progress=lambda *a: calls.append(a))

        assert calls == [
            (0, 2, 0, 3), (0, 2, 1, 3), (0, 2, 2, 3), (0, 2, 3, 3),
            (1, 2, 0, 3), (1, 2, 1, 3), (1, 2, 2, 3), (1, 2, 3, 3),
        ]

    def test_the_heartbeat_interval_is_inside_the_janitors_window(self):
        from src.workers.task_heartbeat import STALE_AFTER_SECONDS

        assert MT.MERGE_HEARTBEAT_SECONDS < STALE_AFTER_SECONDS
        assert MT.STATISTICS_HEARTBEAT_SECONDS < STALE_AFTER_SECONDS


class TestTheMergeIsWiredToTheLiveJob:
    """The callback is built in the task and reaches the merge loop, link by link."""

    def test_the_task_builds_the_merge_heartbeat_and_passes_it(self):
        task = MT.extract_activations
        assert A.calls_named(task, "build_merge_heartbeat"), "the task never builds the merge heartbeat"
        assert A.passes_real_value(task, "extract_activations", "merge_progress_callback"), (
            "the merge heartbeat is built and never handed to the extraction"
        )

    def test_the_service_hands_it_to_the_extraction_pass(self):
        calls = A.calls_named(ActivationService.extract_activations, "_run_extraction")
        passed = [A.keyword_of(c, "on_merge_progress") for c in calls]
        assert any(getattr(v, "id", None) == "merge_progress_callback" for v in passed)

    def test_the_extraction_pass_merges_with_it(self):
        calls = A.calls_named(ActivationService._run_extraction, "_merge_batch_files")
        passed = [A.keyword_of(c, "on_merge_progress") for c in calls]
        assert any(getattr(v, "id", None) == "on_merge_progress" for v in passed), (
            "_run_extraction no longer merges through _merge_batch_files with the callback"
        )


# ── 3. the responses carry the phase ───────────────────────────────────────

def _endpoints(monkeypatch, phase):
    from src.api.v1.endpoints import models as endpoint
    from src.services import activation_service

    extraction = SimpleNamespace(
        id="ext_1", model_id="m_1", dataset_id="ds_1", celery_task_id="celery-1",
        status=SimpleNamespace(value="extracting"), phase=phase, progress=90.0,
        samples_processed=20000, max_samples=20000, layer_indices=[11], hook_types=["residual"],
        batch_size=16, gpu_request="auto", gpu_uuid=None, created_at=LONG_AGO, updated_at=LONG_AGO,
        completed_at=None, error_message=None, statistics={}, saved_files=[],
    )

    class _NoTrainings:
        def query(self, *columns):
            return self

        def filter(self, *conditions):
            return self

        def all(self):
            return []

    monkeypatch.setattr(endpoint.ModelService, "get_model", AsyncMock(return_value=SimpleNamespace(id="m_1", name="LFM2.5-1.2B-Instruct")))
    monkeypatch.setattr(endpoint, "get_sync_db", lambda: _ctx(_NoTrainings()))
    monkeypatch.setattr(
        endpoint.ExtractionDatabaseService, "list_extractions_for_model",
        staticmethod(lambda db, model_id, limit=50: [extraction]),
    )
    monkeypatch.setattr(
        endpoint.ExtractionDatabaseService, "get_active_extraction_for_model",
        staticmethod(lambda db, model_id: extraction),
    )
    monkeypatch.setattr(activation_service.ActivationService, "__init__", lambda self: None)
    monkeypatch.setattr(activation_service.ActivationService, "list_extractions", lambda self: [])
    return endpoint


class TestTheResponsesCarryThePhase:
    @pytest.mark.parametrize("phase", ["merging", "statistics"])
    def test_the_active_extraction_reports_its_phase(self, monkeypatch, phase):
        endpoint = _endpoints(monkeypatch, phase)
        active = asyncio.run(endpoint.get_active_extraction("m_1", db=None))
        assert active["data"]["status"] == "extracting"
        assert active["data"]["phase"] == phase

    def test_the_list_reports_each_extractions_phase(self, monkeypatch):
        endpoint = _endpoints(monkeypatch, "merging")
        listed = asyncio.run(endpoint.list_model_extractions("m_1", db=None))
        rows = [e for e in listed["extractions"] if e["extraction_id"] == "ext_1"]
        assert len(rows) == 1
        assert rows[0]["phase"] == "merging"


# ── 4. the janitor ─────────────────────────────────────────────────────────

def _sweep(monkeypatch, row, *, celery_state="PENDING"):
    """One real janitor sweep over one row. Liveness is decided by the real
    `task_looks_alive`; only Celery's answer is faked."""
    from src.core.celery_app import celery_app
    from src.workers import cleanup_stuck_activations as J

    session = _Session(row)
    failed = []
    monkeypatch.setattr(J.cleanup_stuck_activations_task, "get_db", lambda: _ctx(session))
    monkeypatch.setattr(J, "emit_extraction_failed", lambda **k: failed.append(k))
    monkeypatch.setattr(
        celery_app, "AsyncResult", lambda task_id, app=None: SimpleNamespace(state=celery_state, info=None)
    )
    result = J.cleanup_stuck_activations_task.run()
    return result, failed, session


class TestTheJanitorJudgesAStartedJobByItsClock:
    def test_the_old_expression_was_false_for_every_status(self):
        from src.workers.cleanup_stuck_activations import extraction_has_started

        def old(status):
            return str(status).lower() in ("extracting", "running", "processing")

        assert [s for s in ExtractionStatus if old(s)] == [], (
            "str() of an ExtractionStatus member is 'ExtractionStatus.X', never 'x'"
        )
        assert {s.value for s in ExtractionStatus if extraction_has_started(s)} == {
            "loading", "extracting", "saving",
        }

    @pytest.mark.parametrize("phase", ["extracting", "merging", "statistics"])
    def test_a_dead_job_with_a_stale_heartbeat_is_reaped(self, monkeypatch, phase):
        """Two hours without a heartbeat and Celery says PENDING: the worker is gone.

        Before the fix `started` was False, so the row's age was never
        consulted, PENDING read as alive, and this row was kept forever.
        """
        row = _row(status=ExtractionStatus.EXTRACTING, phase=phase, updated_at=LONG_AGO)

        result, failed, _ = _sweep(monkeypatch, row)

        assert result == {"cleaned": 1}
        assert row.status == ExtractionStatus.FAILED
        assert row.error_type == "TIMEOUT"
        assert row.phase == phase, "the reaped row keeps the step it died in"
        assert len(failed) == 1

    @pytest.mark.parametrize("status", [ExtractionStatus.LOADING, ExtractionStatus.SAVING])
    def test_every_started_status_is_judged_by_its_clock(self, monkeypatch, status):
        row = _row(status=status, updated_at=LONG_AGO)
        result, _, _ = _sweep(monkeypatch, row)
        assert result == {"cleaned": 1}
        assert row.status == ExtractionStatus.FAILED

    @pytest.mark.parametrize("phase", ["merging", "statistics"])
    def test_a_live_job_with_a_recent_heartbeat_is_kept(self, monkeypatch, phase):
        """The heartbeat wrote 30 s ago. Celery still says PENDING (these tasks
        never report state), and that must not condemn it."""
        row = _row(phase=phase, updated_at=datetime.now(timezone.utc) - timedelta(seconds=30))

        result, failed, session = _sweep(monkeypatch, row)

        assert result == {"cleaned": 0}
        assert row.status == ExtractionStatus.EXTRACTING
        assert failed == []
        assert session.commits == 0

    def test_a_queued_row_waiting_behind_a_gpu_job_is_kept(self, monkeypatch):
        """A queued row's age is time spent waiting, not evidence of a dead task."""
        row = _row(status=ExtractionStatus.QUEUED, updated_at=LONG_AGO)
        result, _, _ = _sweep(monkeypatch, row)
        assert result == {"cleaned": 0}
        assert row.status == ExtractionStatus.QUEUED
