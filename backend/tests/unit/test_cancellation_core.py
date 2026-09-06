"""The shared cooperative-cancellation mechanism.

`revoke(terminate=True)` cannot stop a running task here — every worker is
`--pool=solo -c 1`, terminate signals a pool child, and solo has none. So the
row is the channel and the task polls it. These tests pin the three properties
that make that work, each of which was learned from a real failure:

  * `OperatorCancelled` is a BaseException, so the three existing
    `except Exception` handlers on these paths cannot swallow it.
  * `record_progress` refuses to move a terminal row, so a straggling progress
    report cannot overwrite a cancellation. Without this the checker is theatre.
  * the poll defeats the SQLAlchemy identity map, so a long-lived task session
    can observe a write made by the API process (MIS-E2E-057).
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.core import cancellation as C


class _Row:
    def __init__(self, status="running", **kw):
        self.status = status
        self.nlp_status = status
        self.progress = 0.0
        self.error_message = None
        self.started_at = None
        self.completed_at = None
        for k, v in kw.items():
            setattr(self, k, v)


class _Query:
    """Records whether populate_existing() was used — the identity-map defeat."""

    def __init__(self, row, log):
        self._row = row
        self._log = log

    def filter(self, *a, **k):
        return self

    def populate_existing(self):
        self._log.append("populate_existing")
        return self

    def first(self):
        return self._row


def _session(row, log=None):
    log = log if log is not None else []
    db = MagicMock()
    db.query.side_effect = lambda model: _Query(row, log)
    db._log = log
    return db


def _ctx(db):
    c = MagicMock()
    c.__enter__.return_value = db
    c.__exit__.return_value = False
    return c


class TestTheExceptionCannotBeSwallowed:
    def test_operator_cancelled_is_not_an_Exception(self):
        """THE decision. `services/activation_service.py` wraps the progress
        callback in `except Exception` — the natural checkpoint of the
        worst-broken task. An Exception-derived cancel dies there silently."""
        assert issubclass(C.OperatorCancelled, BaseException)
        assert not issubclass(C.OperatorCancelled, Exception)

    def test_an_except_Exception_block_does_not_catch_it(self):
        """Written as the real handler shape rather than an issubclass check,
        because that is the code it has to survive."""
        caught = None
        with pytest.raises(C.OperatorCancelled):
            try:
                raise C.OperatorCancelled("labeling", "job1", detail="at 4 of 1000")
            except Exception as exc:          # noqa: BLE001 - the shape under test
                caught = exc
        assert caught is None, "an except Exception handler swallowed the cancellation"

    def test_it_carries_what_the_task_boundary_needs(self):
        exc = C.OperatorCancelled("training", "t1", reason="deleted", detail="step 40")
        assert (exc.scope, exc.target_id, exc.reason, exc.detail) == (
            "training", "t1", "deleted", "step 40")


class TestRegistry:
    def test_every_scope_resolves_to_a_real_model_and_columns(self):
        for kind, scope in C.SCOPES.items():
            model = scope.model()
            assert hasattr(model, scope.id_field), f"{kind}: no {scope.id_field}"
            assert hasattr(model, scope.status_field), f"{kind}: no {scope.status_field}"

    def test_an_unknown_scope_fails_loudly(self):
        """Silence here would mean a task that cannot be cancelled and says so
        to nobody."""
        with pytest.raises(KeyError, match="no cancel scope registered"):
            C.get_scope("not_a_thing")

    def test_training_treats_paused_as_terminal(self):
        """A straggling progress write must not un-pause a job."""
        assert "paused" in C.SCOPES["training"].terminal_values


class TestChecker:
    def test_the_first_call_always_polls(self):
        """Throttling from zero would skip the opening checks, so a job
        cancelled before it started runs to its Nth checkpoint first."""
        db = _session(_Row("cancelled"))
        check = C.cancel_checker("labeling", "j1", db=db, min_interval_s=999)
        assert check() is True

    def test_it_defeats_the_identity_map(self):
        log = []
        db = _session(_Row("running"), log)
        C.cancel_checker("labeling", "j1", db=db)()
        assert "populate_existing" in log, (
            "without populate_existing a task session never sees the API's write"
        )

    def test_the_throttle_suppresses_a_second_poll_inside_the_window(self):
        log = []
        db = _session(_Row("running"), log)
        check = C.cancel_checker("labeling", "j1", db=db, min_interval_s=999)
        for _ in range(50):
            check()
        assert log.count("populate_existing") == 1, "the throttle did not hold"

    def test_the_throttle_releases_once_the_window_passes(self):
        log = []
        db = _session(_Row("running"), log)
        check = C.cancel_checker("labeling", "j1", db=db, min_interval_s=0.01)
        check()
        time.sleep(0.02)
        check()
        assert log.count("populate_existing") == 2

    def test_poll_now_ignores_the_throttle(self):
        log = []
        db = _session(_Row("running"), log)
        check = C.cancel_checker("labeling", "j1", db=db, min_interval_s=999)
        check()
        check.poll_now()
        assert log.count("populate_existing") == 2

    def test_once_cancelled_it_stays_cancelled_without_polling_again(self):
        log = []
        db = _session(_Row("cancelled"), log)
        check = C.cancel_checker("labeling", "j1", db=db)
        assert check() is True
        assert check() is True
        assert log.count("populate_existing") == 1

    def test_a_failed_poll_does_not_kill_the_work(self):
        db = MagicMock()
        db.query.side_effect = RuntimeError("transaction aborted")
        assert C.cancel_checker("labeling", "j1", db=db)() is False

    def test_raise_if_cancelled_raises_with_the_detail(self):
        db = _session(_Row("cancelled"))
        check = C.cancel_checker("labeling", "j1", db=db)
        with pytest.raises(C.OperatorCancelled) as e:
            check.raise_if_cancelled("stopped at 4 of 1000")
        assert e.value.detail == "stopped at 4 of 1000"
        assert e.value.reason == "cancelled"


class TestMissingRowPolicy:
    def test_a_deleted_row_stops_the_job_where_deletion_is_the_signal(self):
        """`delete_labeling_job` deletes the row; today the check finds nothing
        and returns, so the job runs on against a deleted row."""
        check = C.cancel_checker("labeling", "gone", db=_session(None))
        assert check() is True
        assert check.reason == "deleted"

    def test_a_deleted_row_is_ignored_where_deletion_is_unrelated(self):
        check = C.cancel_checker("activation_extraction", "gone", db=_session(None))
        assert check() is False


class TestTerminalGuard:
    @pytest.mark.parametrize("terminal", ["cancelled", "completed", "failed"])
    def test_a_terminal_row_is_never_moved_back(self, terminal):
        row = _Row(terminal)
        assert C.record_progress(
            "labeling", "j1", status="labeling", progress=42.0, db=_session(row)
        ) is False
        assert row.status == terminal
        assert row.progress == 0.0, "a refused update must write nothing at all"

    def test_a_live_row_updates_normally(self):
        row = _Row("labeling")
        assert C.record_progress(
            "labeling", "j1", status="labeling", progress=12.0, db=_session(row)
        ) is True
        assert row.progress == 12.0

    def test_terminal_to_terminal_is_allowed_so_janitors_still_work(self):
        row = _Row("cancelled")
        assert C.record_progress("labeling", "j1", status="failed", db=_session(row)) is True
        assert row.status == "failed"

    def test_an_error_message_only_write_survives_on_a_terminal_row(self):
        """A cancelled task records WHERE it stopped."""
        row = _Row("cancelled")
        assert C.record_progress(
            "labeling", "j1", error_message="stopped at 4", db=_session(row)
        ) is True
        assert row.error_message == "stopped at 4"
        assert row.status == "cancelled"

    def test_paused_training_is_not_resurrected_by_a_progress_report(self):
        row = _Row("paused")
        assert C.record_progress(
            "training", "t1", status="running", progress=50.0, db=_session(row)
        ) is False
        assert row.status == "paused"

    def test_progress_is_clamped(self):
        row = _Row("labeling")
        C.record_progress("labeling", "j1", progress=250.0, db=_session(row))
        assert row.progress == 100.0

    def test_completed_at_is_stamped_on_a_terminal_transition(self):
        row = _Row("labeling")
        C.record_progress("labeling", "j1", status="completed", db=_session(row))
        assert row.completed_at is not None


class TestRequestCancel:
    def test_it_writes_the_flag_and_reports_it_was_running(self):
        row = _Row("labeling")
        out = C.request_cancel("labeling", "j1", reason="operator", db=_session(row))
        assert out.requested is True
        assert out.was_running is True
        assert row.status == "cancelled"
        assert row.error_message == "operator"

    def test_a_queued_job_reports_it_never_started(self):
        row = _Row("queued")
        out = C.request_cancel("labeling", "j1", db=_session(row))
        assert out.requested is True
        assert out.was_running is False
        assert "will not run" in out.detail

    def test_a_finished_job_is_refused_rather_than_pretended_at(self):
        row = _Row("completed")
        out = C.request_cancel("labeling", "j1", db=_session(row))
        assert out.requested is False
        assert "already completed" in out.detail
        assert row.status == "completed"

    def test_a_missing_row_is_reported_not_raised(self):
        out = C.request_cancel("labeling", "nope", db=_session(None))
        assert out.requested is False

    def test_revoke_is_issued_without_terminate(self):
        """revoke only helps for a task that never started. Passing terminate=
        would invite the reader to believe it stops a running one."""
        row = _Row("labeling")
        fake = MagicMock()
        with patch.dict("sys.modules"), patch("src.core.celery_app.celery_app", fake):
            C.request_cancel("labeling", "j1", celery_task_id="abc", db=_session(row))
        fake.control.revoke.assert_called_once_with("abc")


class TestCooperativeCancelDecorator:
    def test_it_converts_the_exception_into_the_canonical_result(self):
        @C.cooperative_cancel("labeling")
        def task():
            raise C.OperatorCancelled("labeling", "j1", detail="at 4 of 1000")

        with patch.object(C, "record_progress", return_value=True) as rec:
            out = task()
        assert out["status"] == "cancelled"
        assert out["scope"] == "labeling"
        assert out["target_id"] == "j1"
        assert out["detail"] == "at 4 of 1000"
        rec.assert_called_once()

    def test_it_does_not_swallow_a_real_error(self):
        @C.cooperative_cancel("labeling")
        def task():
            raise ValueError("a genuine bug")

        with pytest.raises(ValueError):
            task()

    def test_a_normal_return_is_untouched(self):
        @C.cooperative_cancel("labeling")
        def task():
            return {"status": "completed"}

        assert task() == {"status": "completed"}

    def test_the_scope_is_discoverable_for_the_registry_test(self):
        """Shape D must find tasks via the imported object, never a source
        regex — a source-scraping guard fails open, twice-observed here."""
        @C.cooperative_cancel("labeling")
        def task():
            return None

        assert task.__cooperative_cancel_scope__ == "labeling"


class TestThePermanentAliases:
    """Phase 5 kept two names alive as ALIASES, not subclasses.

    Both are caught by name in worker code and asserted by name in existing
    behavioural tests, so the names had to survive the shim. Pointing them at
    `OperatorCancelled` also upgrades both from `Exception` to `BaseException`
    — which is the MIS-E2E-058 fix generalised to the two remaining paths whose
    outer `except Exception` could still turn an operator's stop into a crash.

    Mutation controls P5-C4 and P5-C5 restore them as plain Exceptions.
    """

    def test_jlens_TaskCancelled_is_the_core_exception(self):
        from src.workers.jlens_progress import TaskCancelled

        assert TaskCancelled is C.OperatorCancelled
        assert not issubclass(TaskCancelled, Exception), (
            "a J-space cancel raised inside a task whose handler catches "
            "Exception would be recorded as a crash"
        )

    def test_labeling_cancelled_is_the_core_exception(self):
        from src.services.labeling_service import LabelingService

        assert LabelingService._LabelingCancelled is C.OperatorCancelled
        assert not issubclass(LabelingService._LabelingCancelled, Exception), (
            "MIS-E2E-058: labeling's outer except Exception caught its own "
            "cancellation and wrote FAILED"
        )


class TestTheCircuitShim:
    """`circuit_capture_tasks._cancel_checker` keeps its signature and four
    call sites, but the vocabulary now lives in the registry."""

    def _row(self, **kw):
        row = _Row(kw.pop("status", "running"))
        for k, v in kw.items():
            setattr(row, k, v)
        return row

    def test_each_model_and_column_pair_resolves_to_its_own_scope(self):
        from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
        from src.workers.circuit_capture_tasks import _SCOPE_FOR

        assert _SCOPE_FOR[(CircuitCaptureRun.__name__, "status")] == "circuit_capture"
        assert _SCOPE_FOR[(CircuitDiscoveryRun.__name__, "status")] == "circuit_discovery"
        assert _SCOPE_FOR[
            (CircuitDiscoveryRun.__name__, "attribution_status")
        ] == "circuit_attribution"
        assert _SCOPE_FOR[
            (CircuitDiscoveryRun.__name__, "validation_status")
        ] == "circuit_validation"

    def test_the_attribution_checker_reads_attribution_status(self):
        """A checker pointed at the run's main `status` would satisfy a
        "returns a callable" test and never fire for an attribution cancel."""
        from src.models.circuit_runs import CircuitDiscoveryRun
        from src.workers.circuit_capture_tasks import _cancel_checker

        row = self._row(status="running", attribution_status="cancelled")
        check = _cancel_checker(
            _session(row), CircuitDiscoveryRun, "run_1",
            status_field="attribution_status",
        )
        assert check() is True

    def test_it_does_not_fire_on_an_unrelated_column(self):
        from src.models.circuit_runs import CircuitDiscoveryRun
        from src.workers.circuit_capture_tasks import _cancel_checker

        row = self._row(status="cancelled", attribution_status="running")
        check = _cancel_checker(
            _session(row), CircuitDiscoveryRun, "run_1",
            status_field="attribution_status",
        )
        assert check() is False, (
            "the attribution checker fired on the discovery run's own status, "
            "so cancelling a discovery would abort an unrelated attribution"
        )

    def test_it_really_delegates(self):
        """A stub returning False would pass every test above that asserts
        False. This one requires a real poll."""
        from src.models.circuit_runs import CircuitCaptureRun
        from src.workers.circuit_capture_tasks import _cancel_checker

        row = self._row(status="cancelled")
        check = _cancel_checker(_session(row), CircuitCaptureRun, "run_1")
        assert check() is True

    def test_an_unregistered_pair_fails_loudly(self):
        """Silently returning a never-firing checker is how a new lifecycle
        ships with cancellation that does nothing."""
        from src.models.circuit_runs import CircuitCaptureRun
        from src.workers.circuit_capture_tasks import _cancel_checker

        with pytest.raises(KeyError, match="no cancel scope"):
            _cancel_checker(
                _session(self._row()), CircuitCaptureRun, "run_1",
                status_field="not_a_column",
            )
