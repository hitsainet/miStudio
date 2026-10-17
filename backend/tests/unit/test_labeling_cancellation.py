"""Cancelling a labeling job must actually STOP it.

WHY THIS EXISTS
---------------
`cancel_labeling_job` set status=CANCELLED and called
`celery_app.control.revoke(terminate=True)` — and the job kept running for
hours. Observed in production: a job cancelled at 11:45:58 was still calling the
LLM at 11:48:25, ~150 features later, and only died when the pod was restarted.

The reason is the worker's pool. It runs `--pool=solo -c 1`:
  * a task executes in the worker's MAIN process, so there is no child for
    Celery to signal — `terminate=True` cannot stop a running task
  * while the task runs, the main process never services control messages, so
    the revoke is not even read (`celery inspect ping` times out)
  * with `-c 1` that one task blocks EVERY queue, so nothing else can start

So revoke can never be the mechanism here. The loop has to notice by itself.

MUTATION CONTROLS:
  * remove `self._raise_if_cancelled(...)` from a batch loop -> the loop-stops
    test fails
  * make the checker swallow the CANCELLED status -> same
  * re-raise instead of returning in the task's cancel branch -> the
    clean-cancel test fails
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.models.labeling_job import LabelingStatus
from src.services.labeling_service import LabelingService


class _Query:
    """A query that models SQLAlchemy's IDENTITY MAP (MIS-E2E-057).

    The previous fake returned a fresh row on every `first()`, so it could not
    exhibit the behaviour under test: the real session is configured
    `expire_on_commit=False`, and a plain re-query hands back the object already
    loaded rather than fresh database state. A cancel written by the API on
    another connection was therefore invisible to the running job, which then
    wrote COMPLETED over it.

    This fake now behaves the same way — `first()` returns the CACHED row unless
    `populate_existing()` was called — so the test can fail when the fix is
    removed. That it could not before is why the defect shipped.
    """

    def __init__(self, session, model):
        self._session = session
        self._model = model
        self._populate = False

    def populate_existing(self):
        self._populate = True
        return self

    def filter(self, *a):
        return self

    def first(self):
        if self._populate:
            # Refresh from "the database", i.e. whatever another connection
            # has since written.
            self._session.cached = self._session.db_row
            self._session.refreshes += 1
        return self._session.cached


class _Session:
    def __init__(self, row):
        # `cached` is what the identity map holds; `db_row` is the real state.
        # They start identical and diverge when something else writes.
        self.cached = row
        self.db_row = row
        self.queries = 0
        self.refreshes = 0

    # Kept for tests that read or SET the loaded row. A setter writes through
    # to both, because a test assigning `session.row = None` means "the row is
    # gone", not "the cache is stale".
    @property
    def row(self):
        return self.cached

    @row.setter
    def row(self, value):
        self.cached = value
        self.db_row = value

    def query(self, model):
        self.queries += 1
        return _Query(self, model)


def _service(status):
    row = SimpleNamespace(id="job_1", status=status)
    svc = LabelingService.__new__(LabelingService)   # no __init__ side effects
    svc.db = _Session(row)
    return svc, row


class TestCooperativeCancellation:
    def test_raises_when_the_job_has_been_cancelled(self):
        svc, _ = _service(LabelingStatus.CANCELLED.value)
        with pytest.raises(LabelingService._LabelingCancelled):
            svc._raise_if_cancelled("job_1")

    @pytest.mark.parametrize(
        "status", [LabelingStatus.LABELING.value, LabelingStatus.QUEUED.value]
    )
    def test_does_not_raise_while_the_job_is_live(self, status):
        svc, _ = _service(status)
        svc._raise_if_cancelled("job_1")  # must not raise

    def test_a_deleted_row_stops_the_job(self):
        """INVERTED 2026-09-05. This asserted that a vanished row must NOT
        raise, on the reasoning that it would be a spurious cancellation.

        It was pinning the defect. `delete_labeling_job` REVOKES (inert on a
        solo pool) and then DELETES THE ROW — deletion is how that path stops a
        job. Returning quietly meant the loop labelled every remaining feature
        against a row that no longer existed, writing results nobody could read
        and holding the single worker for the duration.

        The "spurious cancellation" it guarded against cannot happen here: the
        endpoint commits the row before `.delay()` (`endpoints/labeling.py`),
        so by the time a worker polls, a missing row means deleted, not
        not-yet-created. The `labeling` scope carries `missing_row="cancelled"`
        for exactly this, and reports the reason as "deleted" rather than
        flattening it into "cancelled".
        """
        svc, _ = _service(LabelingStatus.LABELING.value)
        svc.db.row = None
        with pytest.raises(LabelingService._LabelingCancelled) as exc:
            svc._raise_if_cancelled("job_1")
        assert exc.value.reason == "deleted"

    def test_a_db_error_never_breaks_the_job(self):
        """The check is a safety net, not a new failure mode."""
        svc, _ = _service(LabelingStatus.LABELING.value)

        def boom(_model):
            raise RuntimeError("connection reset")

        svc.db.query = boom
        svc._raise_if_cancelled("job_1")  # swallowed

    def test_a_batch_loop_stops_once_the_job_is_cancelled(self):
        """THE regression: the loop must abandon its remaining batches.

        Simulates the real shape — a long batch loop that checks each iteration.
        Before the fix the loop ran to completion regardless of status.
        """
        svc, row = _service(LabelingStatus.LABELING.value)
        processed = []

        with pytest.raises(LabelingService._LabelingCancelled):
            for batch_start in range(0, 1000, 1):
                svc._raise_if_cancelled("job_1")
                processed.append(batch_start)
                if batch_start == 4:            # user cancels mid-run
                    row.status = LabelingStatus.CANCELLED.value

        assert processed == [0, 1, 2, 3, 4], (
            "the loop continued past the cancellation instead of stopping"
        )


class TestTaskTreatsCancelAsCleanStop:
    def test_the_task_catches_cancellation_and_does_not_re_raise(self):
        """A user-initiated stop must not be recorded as a failed run.

        Parsed with AST rather than string matching: a substring check happily
        matched `return`/`raise` belonging to the NEXT except-block, so the test
        passed even when the cancel branch re-raised.
        """
        import ast
        import inspect
        import textwrap

        from src.workers import labeling_tasks

        tree = ast.parse(textwrap.dedent(inspect.getsource(labeling_tasks.label_features_task)))

        cancel_handlers = [
            h for h in ast.walk(tree)
            if isinstance(h, ast.ExceptHandler)
            and h.type is not None
            and "_LabelingCancelled" in ast.dump(h.type)
        ]
        assert cancel_handlers, (
            "the task does not distinguish a cancellation from a failure"
        )

        for handler in cancel_handlers:
            raises = [n for n in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
                      if isinstance(n, ast.Raise)]
            returns = [n for n in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
                       if isinstance(n, ast.Return)]
            assert not raises, (
                "the cancel branch re-raises — a user-initiated stop would be "
                "recorded as a failed run"
            )
            assert returns, "the cancel branch must return so the worker is freed"


class TestLoopsAreInstrumented:
    def test_every_labeling_batch_loop_checks_for_cancellation(self):
        """All three loops must check — one unguarded loop still hangs the worker."""
        import inspect

        src = inspect.getsource(LabelingService)
        loops = src.count("for batch_start in range(0, total_features, LABEL_BATCH_SIZE):")
        checks = src.count("self._raise_if_cancelled(labeling_job_id)")
        assert loops > 0, "loop pattern changed; update this test"
        assert checks >= loops, (
            f"{loops} labeling batch loop(s) but only {checks} cancellation check(s)"
        )


class TestTheFixtureCanActuallyExhibitTheDefect:
    """MIS-E2E-057's second half: the fixture, not just the code.

    The finding's sharpest point was not the missing `populate_existing()` — it
    was that the test's fake session had no identity map, so it could not have
    caught the bug in either direction. These tests hold the FAKE to the
    behaviour it is standing in for.
    """

    def test_a_stale_identity_map_hides_an_external_write(self):
        """Without `populate_existing`, a cancel written elsewhere is invisible.

        This is the production symptom, reproduced against the fake.
        """
        svc, row = _service(LabelingStatus.LABELING.value)
        # Another connection cancels the job.
        svc.db.db_row = SimpleNamespace(
            id="job_1", status=LabelingStatus.CANCELLED.value
        )

        # A plain query still sees the cached row...
        assert svc.db.query(None).filter().first().status == (
            LabelingStatus.LABELING.value
        )
        # ...and only a refreshing one sees the truth.
        assert svc.db.query(None).populate_existing().filter().first().status == (
            LabelingStatus.CANCELLED.value
        )

    def test_the_cancel_check_observes_an_external_write(self):
        """End to end: the loop must notice a cancel it did not write."""
        svc, _ = _service(LabelingStatus.LABELING.value)
        svc.db.db_row = SimpleNamespace(
            id="job_1", status=LabelingStatus.CANCELLED.value
        )

        with pytest.raises(LabelingService._LabelingCancelled):
            svc._raise_if_cancelled("job_1")

        assert svc.db.refreshes == 1, (
            "the check did not refresh, so it read the identity map and could "
            "never have seen the cancel"
        )


class TestACancelledJobIsNeverRedelivered:
    """A CANCELLED row must refuse a requeued message instead of restarting.

    WHY THIS EXISTS
    ---------------
    Two defects compounded into a loop that ran for 22 hours against an
    operator's explicit stop, and neither was visible from the other's side.

    `cancel_labeling_job` issued `revoke(terminate=True, signal='SIGTERM')` on
    top of the plain `revoke()` that `request_cancel` already sends. SIGTERM to
    a `--pool=solo` worker reaches the MAIN process, because solo has no child
    to signal, so the worker died mid-task. `task_acks_late=True` +
    `task_reject_on_worker_lost=True` then requeued the still-unacked message.

    And the task's pre-flight guard branched on COMPLETED and on LABELING and
    nothing else, so a CANCELLED row fell through to the work. The replacement
    worker restarted the job from feature 0.

    Measured on 2026-09-09/10: the 53k-feature L46 job was cancelled at 02:22,
    resurrected by the 10:57 deploy (the same requeue, a different trigger), and
    was still calling the judge 13 hours later — having re-labelled 7,324
    features it already held verdicts for. Cancelling it again restarted the
    worker seven seconds later, which is how the second half was found.

    Both halves are pinned here, because either alone leaves the loop closed:
    without the guard any worker restart resurrects the job, and without
    dropping the terminate the cancel supplies the restart itself.

    MUTATION CONTROLS (each verified to fail this class):
      * delete the CANCELLED branch from `label_features_task` -> the
        redelivery test fails
      * restore `terminate=True, signal='SIGTERM'` in `cancel_labeling_job` ->
        the revoke test fails
    """

    @staticmethod
    def _run_task(monkeypatch, row):
        """Invoke the REAL task body against a row in a given state.

        Deliberately not an AST scrape. A source guard proves a branch is
        written, not that it is reached, and this repo has shipped that mistake
        more than once. `label_features_for_extraction` is the work; if it is
        called, the job restarted.
        """
        from contextlib import contextmanager

        from src.services import labeling_service as labeling_service_module
        from src.workers import labeling_tasks

        session = _Session(row)

        @contextmanager
        def _fake_db():
            yield session

        did_work = []

        def _explode(self, job_id):
            did_work.append(job_id)
            return {"features_labeled": 0}

        monkeypatch.setattr(
            labeling_service_module.LabelingService,
            "label_features_for_extraction",
            _explode,
        )
        monkeypatch.setattr(
            labeling_service_module.LabelingService, "__init__",
            lambda self, db: setattr(self, "db", db),
        )

        from src.workers.base_task import DatabaseTask
        monkeypatch.setattr(DatabaseTask, "get_db", lambda self: _fake_db())

        result = labeling_tasks.label_features_task.run("job_1")
        return result, did_work

    def test_a_redelivered_cancelled_job_does_not_restart(self, monkeypatch):
        row = SimpleNamespace(
            id="job_1",
            status=LabelingStatus.CANCELLED.value,
            updated_at=None,
            completed_at=None,
            statistics={"features_labeled": 7366},
        )

        result, did_work = self._run_task(monkeypatch, row)

        assert did_work == [], (
            "a cancelled job was restarted by a redelivered message — the "
            "operator's stop is undone by any worker restart"
        )
        assert result == {"features_labeled": 7366}
        assert row.status == LabelingStatus.CANCELLED.value, (
            "the task moved a cancelled row"
        )

    def test_a_live_job_still_runs(self, monkeypatch):
        """The negative control: the guard must not swallow real work.

        A guard that refuses everything passes the test above and ships a
        labeling feature that never labels.
        """
        row = SimpleNamespace(
            id="job_1",
            status=LabelingStatus.QUEUED.value,
            updated_at=None,
            completed_at=None,
            statistics=None,
        )

        _, did_work = self._run_task(monkeypatch, row)

        assert did_work == ["job_1"], "the guard blocked a job that should run"

    @pytest.mark.asyncio
    async def test_cancel_never_sends_a_terminating_signal(self, monkeypatch):
        """SIGTERM to a solo pool kills the worker and requeues the message.

        `request_cancel` issues the one revoke that is defensible — for a task
        that has not started yet — and passes no `terminate`. Anything here that
        adds one turns a stop into a restart.
        """
        from src.core import celery_app as celery_app_module

        calls = []

        class _Control:
            def revoke(self, task_id, **kwargs):
                calls.append((task_id, kwargs))

        monkeypatch.setattr(
            celery_app_module.celery_app, "control", _Control(), raising=False
        )

        row = SimpleNamespace(
            id="job_1",
            status=LabelingStatus.LABELING.value,
            celery_task_id="task-abc",
            labeling_method="local",
            completed_at=None,
            error_message=None,
        )

        svc = LabelingService.__new__(LabelingService)

        class _AsyncDb:
            async def execute(self, *a, **k):
                return SimpleNamespace(scalar_one_or_none=lambda: row)

            async def refresh(self, *a, **k):
                return None

        svc.db = _AsyncDb()

        recorded = {}

        def _fake_request_cancel(kind, target_id, **kwargs):
            recorded["kind"] = kind
            recorded["celery_task_id"] = kwargs.get("celery_task_id")
            row.status = LabelingStatus.CANCELLED.value
            return SimpleNamespace(ok=True)

        from src.core import cancellation as cancellation_module
        monkeypatch.setattr(
            cancellation_module, "request_cancel", _fake_request_cancel
        )

        await svc.cancel_labeling_job("job_1")

        assert recorded["kind"] == "labeling", (
            "cancel no longer goes through the registry, so the word it writes "
            "is no longer the word the checker reads"
        )
        assert recorded["celery_task_id"] == "task-abc", (
            "the task id must still reach the central revoke, which is what "
            "stops a job that has not started"
        )

        terminating = [
            (tid, kw) for tid, kw in calls
            if kw.get("terminate") or kw.get("signal")
        ]
        assert not terminating, (
            f"cancel sent a terminating revoke {terminating} — on --pool=solo "
            "that kills the worker itself, and acks_late then requeues the job"
        )

    @pytest.mark.asyncio
    async def test_delete_never_sends_a_terminating_signal_either(
        self, monkeypatch
    ):
        """The same defect lived twice in this file, 100 lines apart.

        `delete_labeling_job` carried its own copy of
        `revoke(terminate=True, signal='SIGTERM')`. Fixing only the
        representative I happened to find is this repo's recorded
        anti-pattern, and it matters MORE here: delete then removes the row,
        which is the only channel a running task has (the `labeling` scope
        carries `missing_row="cancelled"` for exactly that). Killing the pool
        on the way out replaces an orderly stop with a dead worker.
        """
        from src.core import celery_app as celery_app_module

        calls = []

        class _Control:
            def revoke(self, task_id, **kwargs):
                calls.append((task_id, kwargs))

        monkeypatch.setattr(
            celery_app_module.celery_app, "control", _Control(), raising=False
        )

        row = SimpleNamespace(
            id="job_1",
            status=LabelingStatus.LABELING.value,
            celery_task_id="task-abc",
        )

        svc = LabelingService.__new__(LabelingService)

        class _AsyncDb:
            async def execute(self, *a, **k):
                return SimpleNamespace(scalar_one_or_none=lambda: row)

            async def delete(self, *a, **k):
                return None

            async def commit(self, *a, **k):
                return None

        svc.db = _AsyncDb()

        await svc.delete_labeling_job("job_1")

        assert calls, (
            "delete no longer revokes at all — a queued task that has not "
            "started would still run against a deleted row"
        )
        terminating = [
            (tid, kw) for tid, kw in calls
            if kw.get("terminate") or kw.get("signal")
        ]
        assert not terminating, (
            f"delete sent a terminating revoke {terminating} — on --pool=solo "
            "that kills the worker itself"
        )

    def test_no_terminating_revoke_survives_anywhere_in_the_labeling_service(
        self,
    ):
        """A ratchet over the whole module, so a third copy cannot appear.

        AST, not a regex: `terminate=True` split across lines is what the
        original looked like, and a substring search for `revoke(terminate`
        would have matched neither copy. This is a BACKSTOP for instances the
        two behavioural tests above do not reach — it is not the protection,
        because a source guard proves what is written, not what runs.
        """
        import ast
        import inspect

        from src.services import labeling_service as module

        tree = ast.parse(inspect.getsource(module))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "revoke"):
                continue
            bad = [
                kw.arg for kw in node.keywords
                if kw.arg in ("terminate", "signal")
            ]
            if bad:
                offenders.append((node.lineno, bad))

        assert not offenders, (
            f"terminating revoke(s) at {offenders} in labeling_service.py — "
            "solo has no pool child, so the signal lands on the worker itself"
        )
