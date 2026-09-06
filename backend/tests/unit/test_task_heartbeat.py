"""
A task whose worker died must stop claiming to be in progress.

OBSERVED, NOT IMAGINED. A band report was killed when a deploy rolled the
backend pod. Celery keeps whatever the task last reported and nothing writes a
terminal state when a worker vanishes, so `/models/tasks/{id}` returned
PROGRESS for forty minutes afterwards — and three separate status checks
reported the job as healthy.

MUTATION CONTROLS (each must turn this file red):
  * beat() omits the timestamp                  -> "stamps" fails
  * looks_orphaned ignores the threshold        -> "only when stale" fails
  * a task with NO heartbeat reads as orphaned  -> "never beat" fails
  * a terminal state is called orphaned         -> "terminal" fails
"""

import time

import pytest

from src.workers.task_heartbeat import (
    STALE_AFTER_SECONDS,
    beat,
    looks_orphaned,
    seconds_since_beat,
)


class TestTheBeat:
    def test_it_stamps_the_clock_alongside_the_caller_s_meta(self):
        meta = beat({"stage": "profiling", "prompt": 3})
        assert meta["stage"] == "profiling"
        assert meta["prompt"] == 3
        assert isinstance(meta["heartbeat"], float)
        assert abs(meta["heartbeat"] - time.time()) < 5

    def test_it_works_with_no_caller_meta_at_all(self):
        assert "heartbeat" in beat()


class TestStaleness:
    def test_a_fresh_beat_is_not_orphaned(self):
        assert looks_orphaned("PROGRESS", beat({"stage": "x"})) is False

    def test_a_task_that_stopped_reporting_IS_orphaned(self):
        old = {"stage": "profiling", "heartbeat": time.time() - STALE_AFTER_SECONDS - 60}
        assert looks_orphaned("PROGRESS", old) is True

    def test_it_is_orphaned_only_once_PAST_the_threshold(self):
        """A slow task must not be declared dead.

        A false 'dead' is worse than a slow truth: it sends someone to re-run a
        job that was going to finish.
        """
        just_inside = {"heartbeat": time.time() - (STALE_AFTER_SECONDS - 30)}
        assert looks_orphaned("PROGRESS", just_inside) is False

    def test_a_task_that_NEVER_beat_is_not_called_orphaned(self):
        """No heartbeat is not the same as a stale one.

        Short tasks never beat at all, and tasks predating this have none.
        Reporting those as dead would condemn most of the queue.
        """
        assert looks_orphaned("PROGRESS", {"stage": "loading_model"}) is False
        assert looks_orphaned("PROGRESS", None) is False
        assert seconds_since_beat({"stage": "x"}) is None

    @pytest.mark.parametrize("state", ["SUCCESS", "FAILURE", "PENDING", "RETRY"])
    def test_a_terminal_or_unstarted_state_is_never_orphaned(self, state):
        """A report finished last week is not orphaned, it is done."""
        ancient = {"heartbeat": time.time() - 86400}
        assert looks_orphaned(state, ancient) is False


class TestTheStatusEndpointActuallyRuns:
    """The endpoint that READS the heartbeat had no test, and I broke it.

    `from ...workers.task_heartbeat import ...` resolves to `src.api.workers`,
    one package short of `src.workers`. Because the import sits INSIDE the
    handler it does not fail at module import, so nothing noticed: the whole
    backend suite passed, CI went green, the image shipped, and every call to
    /models/tasks/{id} returned 500 — which is how a band report that had
    finished successfully read as unreachable.

    MUTATION CONTROLS:
      * restore the ...workers depth  -> "resolves its imports" fails
      * drop the orphan branch        -> "reports orphaned" fails
    """

    def _status(self, state, info):
        import asyncio
        from unittest.mock import MagicMock, patch

        from src.api.v1.endpoints.models import get_task_status

        result = MagicMock()
        result.state = state
        result.info = info
        result.ready.return_value = state in ("SUCCESS", "FAILURE")
        result.successful.return_value = state == "SUCCESS"
        result.failed.return_value = state == "FAILURE"
        result.result = {}

        with patch("celery.result.AsyncResult", return_value=result):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(get_task_status("t-1"))
            finally:
                loop.close()

    def test_it_resolves_its_imports_and_returns(self):
        """Exercises the handler BODY. A module-import check would not have
        caught this — the bad import is inside the function."""
        body = self._status("PROGRESS", beat({"stage": "profiling"}))
        assert body["task_id"] == "t-1"
        assert body["state"] == "PROGRESS"
        assert body["seconds_since_heartbeat"] is not None

    def test_it_reports_orphaned_when_the_beat_goes_stale(self):
        stale = {"stage": "profiling", "heartbeat": time.time() - STALE_AFTER_SECONDS - 60}
        body = self._status("PROGRESS", stale)
        assert body["state"] == "ORPHANED"
        assert body["ready"] is True
        assert "stopped reporting" in body["error"]

    def test_a_healthy_task_is_untouched(self):
        body = self._status("SUCCESS", {"heartbeat": time.time() - 99999})
        assert body["state"] == "SUCCESS"
        assert "error" not in body


class TestActiveOperationsAgreesWithTheHeartbeat:
    """The two surfaces must not disagree about what is running.

    OBSERVED: a fit killed by a pod roll showed "running 21.5%" in Active
    Operations while /models/tasks/{id} correctly reported ORPHANED. The row is
    written when the task is queued and moved by the task itself — so a worker
    that dies writes nothing and the row sits at its last progress forever.

    MUTATION CONTROLS:
      * drop the reconciliation        -> "reports orphaned" fails
      * reconcile QUEUED rows too      -> "queued is not orphaned" fails
    """

    def _row(self, status, state, info):
        from unittest.mock import MagicMock, patch

        from src.api.v1.endpoints.task_queue import _celery_view

        task = MagicMock()
        task.status = status
        task.task_id = "t-1"

        result = MagicMock()
        result.state = state
        result.info = info
        with patch("celery.result.AsyncResult", return_value=result):
            orphaned, _live = _celery_view(task)
            return orphaned

    def test_it_reports_a_running_row_whose_task_stopped_beating(self):
        stale = {"stage": "fitting", "heartbeat": time.time() - STALE_AFTER_SECONDS - 60}
        assert self._row("running", "PROGRESS", stale) is True

    def test_a_live_row_is_left_alone(self):
        assert self._row("running", "PROGRESS", beat({"stage": "fitting"})) is False

    def test_a_QUEUED_row_is_never_orphaned(self):
        """A task waiting behind a long job has not started and cannot beat.

        Condemning it would mark everything in the queue as dead.
        """
        stale = {"heartbeat": time.time() - STALE_AFTER_SECONDS - 60}
        assert self._row("queued", "PENDING", stale) is False
        assert self._row("queued", "PROGRESS", stale) is False

    def test_a_row_with_no_task_id_is_never_orphaned(self):
        """Federated rows from other job tables carry no Celery id."""
        from unittest.mock import MagicMock

        from src.api.v1.endpoints.task_queue import _celery_view

        task = MagicMock()
        task.status = "running"
        task.task_id = None
        assert _celery_view(task) == (False, {})


class TestTheJanitorClosesGhostRows:
    """Read-time reconciliation makes the LISTING honest; this makes it STOP.

    A GET must not have side effects, so the write belongs in a janitor. Without
    one the row stays "running" in the database forever and every consumer that
    reads it directly — not just the reconciled listing — keeps believing it.

    MUTATION CONTROLS:
      * janitor also closes QUEUED rows      -> "only running" fails
      * janitor RETRIES instead of failing   -> "does not resubmit" fails
      * task not registered / wrong queue    -> "reachable" fails
    """

    def test_the_janitor_is_registered_and_routed_to_a_real_queue(self):
        from src.core.celery_app import celery_app

        assert "cleanup_orphaned_tasks" in celery_app.tasks, (
            "the janitor is not in the live registry, so beat would schedule a "
            "task no worker can execute"
        )
        # Short name + explicit queue: a module-path glob never matches these,
        # which this file's own celery_app comment records as a shipped defect.
        assert celery_app.conf.task_routes["cleanup_orphaned_tasks"]["queue"] == (
            "low_priority"
        )
        entry = celery_app.conf.beat_schedule["cleanup-orphaned-tasks"]
        assert entry["task"] == "cleanup_orphaned_tasks"
        assert entry["schedule"] <= STALE_AFTER_SECONDS, (
            "the sweep must run at least as often as the staleness threshold, "
            "or a ghost row stays visible for the sum of both"
        )

    def test_it_closes_only_rows_that_claim_to_be_RUNNING(self):
        """A queued task has not started and cannot beat.

        On a single-GPU queue everything waits behind a long job, so condemning
        queued rows would fail the entire backlog.

        THE FILTER IS INSPECTED, not assumed. An earlier version of this test
        used a fake that returned the same row whatever it was asked, so
        deleting the status filter from the janitor changed nothing observable
        and the mutation survived.
        """
        import src.workers.cleanup_orphaned_tasks as janitor
        from contextlib import contextmanager
        from unittest.mock import MagicMock, patch

        running = MagicMock(status="running", task_id="t-dead", error_message=None)
        criteria = []

        class _Q:
            def query(self, _m):
                return self

            def filter(self, *args, **_kw):
                criteria.extend(str(a) for a in args)
                return self

            def all(self):
                return [running]

            def commit(self):
                pass

        @contextmanager
        def fake_db():
            yield _Q()

        dead = MagicMock()
        dead.state = "PROGRESS"
        dead.info = {"heartbeat": time.time() - STALE_AFTER_SECONDS - 60}

        with patch("src.core.database.get_sync_db", fake_db), patch(
            "celery.result.AsyncResult", return_value=dead
        ):
            out = janitor.cleanup_orphaned_tasks_task.run()

        assert out["closed"] == 1
        assert running.status == "failed"
        assert "presumed gone" in running.error_message

        joined = " ".join(criteria)
        assert "status" in joined, (
            f"the janitor did not filter on status; criteria were {criteria}. "
            "Without it every QUEUED row waiting behind a long job would be "
            "marked failed."
        )
        assert "task_id" in joined, (
            "the janitor did not filter out rows with no Celery id — federated "
            "rows from other job tables have none and cannot be checked"
        )

    def test_it_does_not_resubmit_the_work(self):
        """A dead fit lost every prompt it processed; the accumulator was in
        worker memory. Re-running automatically would take the GPU for another
        hour without being asked."""
        import inspect

        import src.workers.cleanup_orphaned_tasks as janitor

        source = inspect.getsource(janitor)
        for resubmit in (".delay(", ".apply_async(", ".retry("):
            assert resubmit not in source, (
                f"the janitor calls {resubmit} — it must mark the row and stop, "
                "not silently reclaim the GPU"
            )


# ---------------------------------------------------------------------------
# The half of the problem the heartbeat rule cannot see.
#
# A janitor written specifically to clear a dead row could not clear it: the
# worker died AND its result-backend entry expired, so Celery answered PENDING
# with no info. `looks_orphaned` only ever inspects PROGRESS/STARTED, so it
# returned False and the row sat at "running 21.5%" for hours with the GPU idle
# at 0%. Observed on hardware, through a janitor, twice reported by the user.
# ---------------------------------------------------------------------------


def test_a_pending_task_with_an_old_row_is_abandoned():
    """The exact shape that survived the janitor.

    MUTATION CONTROL: delete the `state == "PENDING"` branch of
    `looks_abandoned` and this fails — which is the state the code was in when
    the ghost row was reported.
    """
    from src.workers.task_heartbeat import STALE_AFTER_SECONDS, looks_abandoned

    assert looks_abandoned("PENDING", None, STALE_AFTER_SECONDS + 60)


def test_a_pending_task_whose_row_is_fresh_is_left_alone():
    """A task that has just been dispatched has not failed."""
    from src.workers.task_heartbeat import looks_abandoned

    assert not looks_abandoned("PENDING", None, 5)


def test_a_pending_task_with_no_row_age_is_not_condemned():
    """Absent evidence is not evidence of death — fail safe, not closed."""
    from src.workers.task_heartbeat import looks_abandoned

    assert not looks_abandoned("PENDING", None, None)


def test_a_terminal_task_is_never_abandoned_however_old_its_row():
    """A row left open behind a SUCCESS is stale bookkeeping, not a dead job.

    Marking it failed would tell the user their completed fit died.

    MUTATION CONTROL: drop the TERMINAL_STATES guard and this fails.
    """
    from src.workers.task_heartbeat import looks_abandoned

    for state in ("SUCCESS", "FAILURE", "REVOKED"):
        assert not looks_abandoned(state, None, 999_999), state


def test_a_stale_heartbeat_is_still_caught():
    """The original rule must keep working — this is a widening, not a swap."""
    from src.workers.task_heartbeat import STALE_AFTER_SECONDS, looks_abandoned

    stale = {"heartbeat": 1_000.0}
    now = 1_000.0 + STALE_AFTER_SECONDS + 1
    assert looks_abandoned("PROGRESS", stale, None, now=now)


def test_a_beating_task_is_never_abandoned():
    """A working job must not be condemned by the widened rule."""
    from src.workers.task_heartbeat import looks_abandoned

    fresh = {"heartbeat": 1_000.0}
    assert not looks_abandoned("PROGRESS", fresh, 999_999, now=1_010.0)


def test_row_age_reads_naive_and_aware_timestamps_alike():
    """The column has carried both; comparing across them raises.

    A janitor that raises on one row stops sweeping every row after it.
    """
    from datetime import datetime, timedelta, timezone

    from src.workers.task_heartbeat import seconds_since_row_update

    class _Row:
        def __init__(self, stamp):
            self.updated_at = stamp
            self.created_at = None

    naive = _Row(datetime.utcnow() - timedelta(seconds=900))
    aware = _Row(datetime.now(timezone.utc) - timedelta(seconds=900))

    for row in (naive, aware):
        age = seconds_since_row_update(row)
        assert age is not None and 800 < age < 1000, age


def test_the_janitor_uses_the_widened_rule_not_the_heartbeat_one():
    """Reachability: the fix must be the rule the sweep actually calls.

    The previous janitor imported `looks_orphaned` and was therefore blind to
    the case it existed to clear. Asserting the module imports the widened rule
    is not enough — this drives the sweep and asserts the row is closed.

    MUTATION CONTROL: point the janitor back at `looks_orphaned` and this fails.
    """
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock, patch

    import src.workers.cleanup_orphaned_tasks as janitor

    class _Row:
        status = "running"
        task_id = "dead-task"
        error_message = None
        updated_at = datetime.now(timezone.utc) - timedelta(hours=3)
        created_at = updated_at

    row = _Row()
    db = MagicMock()
    db.query.return_value.filter.return_value.filter.return_value.all.return_value = [row]

    from contextlib import contextmanager

    @contextmanager
    def fake_db():
        yield db

    # The shape that beat the old janitor: no record left in the backend.
    dead = MagicMock(state="PENDING", info=None)

    with patch("src.core.database.get_sync_db", fake_db), patch(
        "celery.result.AsyncResult", return_value=dead
    ):
        out = janitor.cleanup_orphaned_tasks_task.run()

    assert out["closed"] == 1, (
        "the sweep left a PENDING row with a three-hour-old timestamp open — "
        "this is the ghost job the janitor was written to clear"
    )
    assert row.status == "failed"
    assert "presumed gone" in (row.error_message or "")


class TestTheListingCarriesLiveProgress:
    """`/active` fetched the worker's meta and threw it away.

    `_looks_orphaned` built an AsyncResult per running row, asked it one
    question, and discarded `result.info` — the only place `prompts_seen`,
    `total_prompts` and `last_delta` exist. So the listing had a percentage and
    nothing else, and a J-lens fit rendered as `jlens_fit 24.4%` against a raw
    model id. The read was already paid for; only the answer was discarded.

    MUTATION CONTROLS:
      * return `(orphaned, {})` from `_celery_view`  -> "carries the counts" fails
      * keep `heartbeat` in the merged meta          -> "no raw timestamp" fails
      * render absent counts as 0 in the subtitle    -> "absent, never zero" fails
    """

    def _view(self, state, info, status="running"):
        from unittest.mock import MagicMock, patch

        from src.api.v1.endpoints.task_queue import _celery_view

        task = MagicMock()
        task.status = status
        task.task_id = "t-1"
        result = MagicMock()
        result.state = state
        result.info = info
        with patch("celery.result.AsyncResult", return_value=result):
            return _celery_view(task)

    def test_it_carries_the_counts_the_tile_needs(self):
        _orphaned, live = self._view(
            "PROGRESS",
            beat({"stage": "fitting", "prompts_seen": 634, "total_prompts": 1200}),
        )
        assert live["prompts_seen"] == 634
        assert live["total_prompts"] == 1200, (
            "without the denominator a reader can show a percentage but not "
            "'634 / 1200' except by reconstructing it from a rounded number"
        )

    def test_the_raw_heartbeat_timestamp_does_not_leak_into_the_listing(self):
        """An epoch float is not a thing to render; its AGE is."""
        _orphaned, live = self._view("PROGRESS", beat({"prompts_seen": 1}))
        assert "heartbeat" not in live
        assert isinstance(live["seconds_since_heartbeat"], float)

    def test_a_queued_row_reports_nothing_live(self):
        """A task that has not started has no progress to describe."""
        assert self._view("PENDING", None, status="queued") == (False, {})


class TestTheProgressSubtitle:
    def test_absent_counts_render_as_ABSENT_never_zero(self):
        """"0 / 1200" claims the fit has done nothing. It has not been asked.

        MUTATION CONTROL: default the counts to 0 and this fails.
        """
        from src.api.v1.endpoints.task_queue import _progress_details

        assert _progress_details({}) is None
        assert "0" not in (_progress_details({"stage": "fitting"}) or "")

    def test_it_names_the_counts_and_the_threshold_the_delta_is_racing(self):
        from src.api.v1.endpoints.task_queue import _progress_details

        text = _progress_details(
            {
                "prompts_seen": 634,
                "total_prompts": 1200,
                "last_delta": 0.00103,
                "convergence_delta": 1e-3,
            }
        )
        assert "634 / 1,200 prompts" in text
        assert "1.03e-03" in text and "target 1e-03" in text, (
            f"a delta with no target cannot be judged by a reader: {text}"
        )

    def test_a_stage_alone_is_still_worth_saying(self):
        """"validating" explains a bar that has stopped moving at 53%."""
        from src.api.v1.endpoints.task_queue import _progress_details

        assert _progress_details({"stage": "validating"}) == "validating"


class TestTheRowCarriesItsOwnClock:
    """`started_at` and `completed_at` were never written for J-space work.

    Verified on hardware: every J-lens row had both as None. Elapsed time had to
    be derived from `created_at`, which is QUEUE time — an LFM2 fit that waited
    three hours behind gemma would have reported a four-hour fit after one hour
    of work.

    MUTATION CONTROLS:
      * stop stamping started_at   -> "a running row carries a start" fails
      * stamp it on every update   -> "the start is not moved" fails
      * stop stamping completed_at -> "a finished row carries an end" fails
    """

    def _row_after(self, transitions):
        """Drive `update_row` through `transitions` and return (row, snapshots).

        `snapshots` is `started_at` AFTER each transition, because the "is not
        moved" property is about successive writes and cannot be observed from
        the final state alone.
        """
        from contextlib import contextmanager
        from unittest.mock import MagicMock, patch

        from src.workers import jlens_progress

        row = MagicMock()
        row.started_at = None
        row.completed_at = None
        row.status = "queued"

        db = MagicMock()
        # `update_row` is a shim over `record_progress`, which re-reads with
        # populate_existing() to defeat the identity map. The fake must model
        # that chain or the writer operates on a MagicMock, not on `row`.
        db.query.return_value.filter.return_value.populate_existing.return_value.first.return_value = row
        db.query.return_value.filter.return_value.first.return_value = row

        @contextmanager
        def fake_db():
            yield db

        snapshots = []
        with patch("src.core.database.get_sync_db", fake_db):
            for status in transitions:
                jlens_progress.update_row("t-1", status=status)
                snapshots.append(row.started_at)
        return row, snapshots

    def test_a_running_row_carries_a_start_time(self):
        row, _ = self._row_after(["running"])
        assert row.started_at is not None

    def test_the_start_time_is_not_moved_by_later_progress(self):
        """Otherwise elapsed resets to zero on every progress report.

        A fit reports every few seconds for hours; re-stamping on each would
        make the tile read "Elapsed 3s" forever.

        MUTATION CONTROL: drop the `row.started_at is None` condition and this
        fails. The first version of this test never called `update_row` again
        and asserted a value equalled itself, which no mutation could break.
        """
        _row, snapshots = self._row_after(["running", "running", "running"])
        assert len(snapshots) == 3
        assert snapshots[0] is not None
        assert snapshots[1] is snapshots[0] and snapshots[2] is snapshots[0], (
            f"the start time moved across progress reports: {snapshots}"
        )

    def test_a_finished_row_carries_an_end_time(self):
        row, _ = self._row_after(["running", "completed"])
        assert row.completed_at is not None

    def test_a_failed_row_also_carries_an_end_time(self):
        """A job that died has a duration too, and a row left open reads as
        still running."""
        row, _ = self._row_after(["running", "failed"])
        assert row.completed_at is not None

    def test_a_queued_row_has_neither(self):
        row, _ = self._row_after([])
        assert row.started_at is None and row.completed_at is None


class TestAQueuedRowIsAlsoSwept:
    """A row that never reached RUNNING was invisible to the sweep.

    `update_row(status="running")` fires on a task's FIRST progress report, so a
    task rejected before it reports one leaves its row at "queued" — and no
    J-space task writes a failed status at all.

    OBSERVED IN THE PRODUCT: five interventions refused at validation showed as
    "5 queued, 0%" indefinitely on an idle GPU, while every Celery task behind
    them had reported FAILURE. The janitor written to clear ghost rows filtered
    on `status == "running"` and could not see one of them.

    MUTATION CONTROLS:
      * filter on "running" only        -> "a queued row is swept" fails
      * sweep queued rows by AGE        -> "waiting work is not condemned" fails
    """

    def _sweep(self, rows_and_states):
        from contextlib import contextmanager
        from datetime import datetime, timedelta, timezone
        from unittest.mock import MagicMock, patch

        import src.workers.cleanup_orphaned_tasks as janitor

        rows = []
        results = {}
        for i, (status, state) in enumerate(rows_and_states):
            row = MagicMock()
            row.status = status
            row.task_id = f"t-{i}"
            row.error_message = None
            row.updated_at = datetime.now(timezone.utc) - timedelta(hours=3)
            row.created_at = row.updated_at
            rows.append(row)
            results[f"t-{i}"] = MagicMock(state=state, info=None)

        db = MagicMock()
        db.query.return_value.filter.return_value.filter.return_value.all.return_value = rows

        @contextmanager
        def fake_db():
            yield db

        with patch("src.core.database.get_sync_db", fake_db), patch(
            "celery.result.AsyncResult", side_effect=lambda tid, app=None: results[tid]
        ):
            out = janitor.cleanup_orphaned_tasks_task.run()

        # Kept so a test can inspect WHAT WAS ASKED FOR: a MagicMock returns its
        # rows whatever the filter said, so the outcome alone cannot show that
        # the query changed.
        self._first_filter_arg = db.query.return_value.filter.call_args[0][0]
        return out, rows

    def test_the_sweep_ASKS_FOR_queued_rows_as_well_as_running(self):
        """Asserted on the QUERY, because the outcome cannot see the filter.

        The first version of this checked only that a queued row came back
        swept — but the mock returns whatever rows it is given regardless of
        what was filtered, so narrowing the query back to `status == "running"`
        left it green. The filter is the thing under test, so the filter is what
        has to be inspected.

        MUTATION CONTROL: filter on "running" only and this fails.
        """
        from src.models.task_queue import TaskQueue  # noqa: F401

        _out, _rows = self._sweep([("queued", "FAILURE")])
        # The criterion handed to the FIRST .filter() call, rendered with its
        # literal values so the statuses are visible.
        import src.workers.cleanup_orphaned_tasks as janitor  # noqa: F401

        criterion = self._first_filter_arg
        rendered = str(
            criterion.compile(compile_kwargs={"literal_binds": True})
        ).lower()
        assert "queued" in rendered, f"the sweep never asks for queued rows: {rendered}"
        assert "running" in rendered, f"the sweep stopped asking for running rows: {rendered}"

    def test_a_QUEUED_row_whose_task_FAILED_is_swept(self):
        """The exact shape that sat at "0% queued" on an idle GPU."""
        out, rows = self._sweep([("queued", "FAILURE")])
        assert out["closed"] == 1
        assert rows[0].status == "failed"
        assert "rejected before it started" in (rows[0].error_message or "")

    def test_work_WAITING_on_the_queue_is_never_condemned(self):
        """A job behind a long fit is PENDING and legitimately hours old.

        Judging a queued row by AGE would fail the entire queue on a
        single-GPU box, which is the normal state of the world here.

        MUTATION CONTROL: sweep queued rows on row age and this fails.
        """
        out, rows = self._sweep([("queued", "PENDING")])
        assert out["closed"] == 0
        assert rows[0].status == "queued"

    def test_a_revoked_queued_row_is_swept_too(self):
        out, rows = self._sweep([("queued", "REVOKED")])
        assert out["closed"] == 1

    def test_a_running_row_still_uses_the_HEARTBEAT_rule(self):
        """Widening the filter must not change how running rows are judged."""
        out, rows = self._sweep([("running", "PENDING")])
        assert out["closed"] == 1
        assert "stopped reporting" in (rows[0].error_message or "")


class TestATaskOwnsItsOwnFailure:
    """The janitor could only ever close half of these.

    A task that fails BEFORE its first progress report leaves a `queued` row —
    now swept. A task that fails AFTER one leaves a `running` row, and the sweep
    deliberately never closes those: `looks_abandoned` returns False for a
    TERMINAL Celery state, because a finished task is not an orphan. So a fit
    that OOMs at prompt 800 sat at "running 42%" on an idle GPU permanently, and
    the fix for the easier half read as complete.

    Routing failures through the janitor also discards the REASON. The user gets
    prose about the bookkeeping defect instead of "swap partner is 2 tokens".

    MUTATION CONTROLS:
      * remove @owns_its_failure from a task -> "records its own failure" fails
      * swallow the exception in the wrapper -> "still raises" fails
      * write a generic reason               -> "carries the real reason" fails
    """

    def _run_failing(self):
        from contextlib import contextmanager
        from unittest.mock import MagicMock, patch

        from src.workers import jlens_progress

        recorded = {}

        def spy(task_id, status=None, progress=None, error_message=None):
            recorded["status"] = status
            recorded["error"] = error_message

        import src.workers.jlens_intervention_tasks as task_mod

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None

        @contextmanager
        def fake_db():
            yield db

        # A REQUEST ID, because the wrapper needs one. Outside a worker
        # `self.request.id` is None, so without this the decorator correctly
        # declines to write a row and the test measures the harness.
        task_mod.run_intervention_task.push_request(id="t-fail")
        with patch.object(jlens_progress, "update_row", spy), patch(
            "src.core.database.get_sync_db", fake_db
        ), patch.object(task_mod.run_intervention_task, "update_state", MagicMock()):
            try:
                task_mod.run_intervention_task.run(
                    model_id="m_missing",
                    prompt="x",
                    primitive="additive",
                    layers=[0],
                    direction_token="Paris",
                )
                raised = None
            except Exception as exc:  # noqa: BLE001
                raised = exc
            finally:
                task_mod.run_intervention_task.pop_request()
        return recorded, raised

    def test_a_failing_task_RECORDS_ITS_OWN_FAILURE(self):
        recorded, _ = self._run_failing()
        assert recorded.get("status") == "failed", (
            "the task left its row for the janitor, which cannot close a row "
            "whose Celery state is terminal"
        )

    def test_it_carries_THE_REAL_REASON_not_the_bookkeeping_one(self):
        """"No model with id 'm_missing'" is actionable. "nothing moves a row on
        failure" is not."""
        recorded, _ = self._run_failing()
        assert "m_missing" in (recorded.get("error") or ""), recorded.get("error")

    def test_the_exception_STILL_RAISES(self):
        """Celery must see the FAILURE. Swallowing it to record a row would
        trade one silent state for another."""
        _recorded, raised = self._run_failing()
        assert raised is not None

    def test_every_jspace_task_is_decorated(self):
        """A task added later inherits this by construction, not by memory.

        FROM THE LIVE REGISTRY, NOT FROM SOURCE. The first version of this
        scraped the files with a regex that allowed at most one decorator line
        between `@celery_app.task(...)` and `def`. A task carrying any second
        decorator matched NOTHING, so it never entered the list this assertion
        checks — the scan failed OPEN and the undecorated task shipped green.
        That is the failure mode this whole class exists to prevent, reproduced
        in its own guard.

        `celery_app.tasks` is what the worker will actually dispatch, and
        `__wrapped__` is what `functools.wraps` leaves behind, so neither can be
        fooled by how the decorators are laid out.

        MUTATION CONTROL: drop @owns_its_failure from any J-Space task and this
        fails, naming it.
        """
        import importlib
        import pathlib as _pathlib

        from src.core.celery_app import celery_app
        from src.workers.jlens_progress import OWNERSHIP_MARKER

        root = _pathlib.Path(__file__).resolve().parents[2] / "src" / "workers"
        modules = sorted(p.stem for p in root.glob("jlens_*_tasks.py"))
        assert modules, "no J-Space task modules found; the glob is wrong"
        for name in modules:
            importlib.import_module(f"src.workers.{name}")

        registered = {
            name: task
            for name, task in celery_app.tasks.items()
            if any(name.startswith(f"{m.replace('_tasks', '')}") for m in modules)
            or "jlens" in name
        }
        assert registered, "no J-Space tasks are registered with Celery at all"

        missing = [
            name
            for name, task in sorted(registered.items())
            if not getattr(task.run, OWNERSHIP_MARKER, False)
        ]
        assert not missing, (
            f"these registered J-Space tasks do not record their own failure: "
            f"{missing}. A task that dies after its first progress report "
            f"leaves a 'running' row the janitor will never close."
        )

    def test_the_guard_can_SEE_a_task_that_carries_other_decorators(self):
        """The regex it replaced could not, and said everything was fine.

        MUTATION CONTROL: revert the check to a source scrape and this fails.
        """
        import functools

        from src.workers.jlens_progress import OWNERSHIP_MARKER, owns_its_failure

        def extra(fn):
            @functools.wraps(fn)
            def w(*a, **k):
                return fn(*a, **k)

            return w

        # DECORATED, then wrapped by something else — the layout the scrape
        # missed. `functools.wraps` copies __dict__, so the marker survives.
        decorated = extra(owns_its_failure(lambda self: None))
        assert getattr(decorated, OWNERSHIP_MARKER, False) is True

        # And an undecorated one is still seen as undecorated.
        assert getattr(extra(lambda self: None), OWNERSHIP_MARKER, False) is False
