"""Phase 2 — `extract_activations` actually stops.

THE ROW CHANGING IS NOT THE TEST. The endpoint writes CANCELLED; asserting the
row says "cancelled" only proves the endpoint ran, which it always did. It
returned 200 and emitted a "cancelled" WebSocket event while the GPU kept
running for hours. What has to be proven is that the WORK STOPS.

  * Shape A — STOPS.        Drive the real callback N times, flip the flag at 4,
                            assert five units executed and no more.
  * Shape B — NOT OVERWRITTEN. Covered for this writer in test_progress_guard.
  * Shape C — THE ENDPOINT REACHES THE CHANNEL. What the endpoint writes is what
                            the checker reads. Cannot pass if the two disagree
                            about the vocabulary.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.core import cancellation as C


class _Row:
    def __init__(self, status="extracting"):
        from src.models.activation_extraction import ExtractionStatus
        self.status = ExtractionStatus(status)
        self.progress = 0.0
        self.samples_processed = 0
        self.error_message = None
        self.completed_at = None
        self.celery_task_id = "celery-abc"


class _Query:
    def __init__(self, row):
        self._row = row

    def filter(self, *a, **k):
        return self

    filter_by = filter

    def populate_existing(self):
        return self

    def first(self):
        return self._row


def _session(row):
    db = MagicMock()
    db.query.side_effect = lambda model: _Query(row)
    return db


class TestShapeAItStops:
    """THE PRODUCTION CALLBACK, not a reconstruction of it.

    This drove a hand-written copy of the callback until 2026-09-05. Deleting
    the `raise_if_cancelled` from the real one then turned exactly one test red
    — a source scrape — so by the reachability rule the capability was not
    shipped. `build_extraction_progress_callback` exists at module level so this
    can call the thing production calls.
    """

    def _drive(self, row, cancel_at, total=1000):
        """Run the real callback `total` times, flipping the flag at `cancel_at`.

        Wraps each call in `except Exception`, which is not decoration: that is
        the exact shape `activation_service.py` wraps the callback in.
        """
        from src.core.cancellation import cancel_checker
        from src.models.activation_extraction import ExtractionStatus
        from src.workers.model_tasks import build_extraction_progress_callback

        task = MagicMock()

        @contextmanager
        def _db():
            yield _session(row)

        task.get_db = _db

        checker = cancel_checker(
            "activation_extraction", "ext_1", db=_session(row), min_interval_s=0.0
        )
        callback = build_extraction_progress_callback(task, "m_1", "ext_1", checker)

        executed = []
        raised = None
        with patch("src.workers.model_tasks.emit_extraction_progress"):
            for unit in range(total):
                if unit == cancel_at:
                    row.status = ExtractionStatus.CANCELLED
                try:
                    callback(unit, total)
                except Exception:  # noqa: BLE001 - the production swallow site
                    pass
                except C.OperatorCancelled as exc:
                    raised = exc
                    break
                executed.append(unit)
        return executed, raised

    def test_exactly_the_units_before_the_flag_ran(self):
        """Not "it stopped eventually" — four units, then nothing."""
        executed, raised = self._drive(_Row(), cancel_at=4)
        assert executed == [0, 1, 2, 3], (
            f"expected the loop to abandon at the first checkpoint after the "
            f"flag; {len(executed)} units executed"
        )
        assert raised is not None, "the cancellation never propagated"

    def test_an_except_Exception_around_the_callback_cannot_swallow_it(self):
        """THE EMPIRICAL PROOF OF THE BaseException DECISION.

        `activation_service.py` has wrapped this callback in
        `except Exception: logger.warning("Progress callback failed")` since long
        before cancellation existed. An `Exception`-derived cancel raised at the
        one checkpoint this task has would be logged at WARNING and the
        extraction would run to completion — the operator told it stopped, the
        GPU busy for hours. `_drive` catches Exception on every unit and the
        cancellation still escapes.
        """
        assert issubclass(C.OperatorCancelled, BaseException)
        assert not issubclass(C.OperatorCancelled, Exception)
        _executed, raised = self._drive(_Row(), cancel_at=4)
        assert raised is not None, (
            "an except Exception around the callback swallowed the cancellation"
        )

    def test_the_detail_says_where_it_stopped(self):
        _executed, raised = self._drive(_Row(), cancel_at=4)
        assert "4 of 1000" in raised.detail

    def test_an_uncancelled_run_reports_every_unit(self):
        """The negative side: the checkpoint must not stop healthy work."""
        executed, raised = self._drive(_Row(), cancel_at=None, total=50)
        assert raised is None
        assert len(executed) == 50


class TestShapeCTheEndpointReachesTheChannel:
    """What the endpoint writes must be what the worker's checker reads.

    This is the test that catches a vocabulary split — `saes.py` writes FAILED
    on cancel while `models.py` writes CANCELLED for the sibling lifecycle. A
    checker looking for one cannot see the other.
    """

    def test_the_endpoint_write_is_visible_to_the_workers_checker(self):
        row = _Row()
        db = _session(row)

        outcome = C.request_cancel(
            "activation_extraction", "ext_1",
            reason="Extraction cancelled by user", db=db,
        )
        assert outcome.requested is True

        # A DIFFERENT checker, as the worker would construct it.
        checker = C.cancel_checker("activation_extraction", "ext_1", db=db)
        assert checker() is True, (
            "the worker's checker cannot see what the endpoint wrote — the two "
            "sides disagree about what 'cancelled' is"
        )

    def test_the_endpoint_writes_the_columns_own_enum_member(self):
        from src.models.activation_extraction import ExtractionStatus

        row = _Row()
        C.request_cancel("activation_extraction", "ext_1", db=_session(row))
        assert row.status is ExtractionStatus.CANCELLED

    def test_a_queued_job_is_reported_as_never_going_to_run(self):
        row = _Row("queued")
        out = C.request_cancel("activation_extraction", "ext_1", db=_session(row))
        assert out.was_running is False
        assert "will not run" in out.detail

    def test_a_running_job_is_reported_as_stopping_at_a_checkpoint(self):
        """The endpoint must not claim the GPU has already stopped."""
        row = _Row("extracting")
        out = C.request_cancel("activation_extraction", "ext_1", db=_session(row))
        assert out.was_running is True
        assert "next checkpoint" in out.detail
        assert "cancelled successfully" not in out.detail


class TestTheTaskIsWiredForCancellation:
    """Reachability: the capability must not be removable without a red."""

    def test_the_task_carries_the_cooperative_cancel_decorator(self):
        import inspect

        from src.workers.model_tasks import extract_activations

        found = None
        fn = extract_activations
        for _ in range(6):
            found = getattr(fn, "__cooperative_cancel_scope__", None)
            if found:
                break
            nxt = getattr(fn, "__wrapped__", None)
            if nxt is None or nxt is fn:
                break
            fn = nxt
        assert found == "activation_extraction", (
            "extract_activations is not decorated; an OperatorCancelled would "
            "escape as a task FAILURE and the acks_late message would be "
            "redelivered instead of acked"
        )

    def test_the_task_uses_the_real_callback_factory(self):
        """The factory is only worth anything if the task actually calls it.

        Shape A proves the callback the factory returns stops the work. This
        proves the task hands THAT callback to the extraction, rather than
        building its own copy without the checkpoint — the "declaring is not
        wiring" failure this repo has shipped three times.
        """
        import inspect

        from src.workers.model_tasks import extract_activations

        src = inspect.getsource(inspect.unwrap(extract_activations))
        assert 'cancel_checker("activation_extraction"' in src, (
            "the task never constructs a checker"
        )
        assert "build_extraction_progress_callback(" in src, (
            "the task builds its own callback instead of the one with the "
            "cancellation checkpoint in it"
        )
        assert "progress_callback=on_extraction_progress" in src, (
            "the built callback is never passed to the extraction"
        )

    def test_it_polls_before_the_model_load(self):
        """The callback is not reached until the model is on the GPU.

        Everything from the checker's construction to the first callback is one
        indivisible `extract_activations` call, and loading a large model into
        it takes minutes. Without a poll before that, cancelling a job that has
        only just started still costs a full load.
        """
        import inspect

        from src.workers.model_tasks import extract_activations

        src = inspect.getsource(inspect.unwrap(extract_activations))
        poll = src.find("cancelled.poll_now()")
        run = src.find("activation_service.extract_activations(")
        assert poll != -1, "nothing polls before the extraction pass begins"
        assert run != -1, "shape changed — re-read the task"
        assert poll < run, (
            "the pre-extraction poll happens after the extraction call, which "
            "is no earlier than the callback it was meant to precede"
        )

    def test_the_endpoint_no_longer_pretends_terminate_works(self):
        import inspect

        from src.api.v1.endpoints import models as models_endpoint

        src = inspect.getsource(models_endpoint.cancel_extraction)
        # The endpoint dispatches through `run_in_threadpool` now, so
        # `"request_cancel(" in src` is false while the behaviour is unchanged —
        # the moved-not-deleted trap again. Assert the CALL.
        import _cancel_ast as A

        assert "activation_extraction" in A.scopes_passed_to(
            models_endpoint.cancel_extraction, "run_in_threadpool"
        ), "the endpoint no longer asks the registry to cancel anything"

        # COMMENTS STRIPPED FIRST. The first version of this assertion matched
        # the comment that EXPLAINS why terminate is wrong, so it failed against
        # correct code — and would equally have passed against a re-added call
        # sitting under a comment that did not mention it.
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert "terminate=True" not in code, (
            "terminate signals a POOL CHILD and this worker is --pool=solo; "
            "it returns cleanly and does nothing"
        )


class TestAFinishedRunDoesNotOverwriteTheCancellation:
    def test_mark_completed_leaves_a_cancelled_row_cancelled(self):
        """Cancelling during SAVING leaves no checkpoint, so the task runs to
        the end and reaches mark_completed. terminal -> terminal is allowed by
        the guard (janitors need it), so this has to be refused here."""
        from src.models.activation_extraction import ExtractionStatus
        from src.services.extraction_db_service import ExtractionDatabaseService

        row = _Row("cancelled")
        ExtractionDatabaseService.mark_completed(
            db=_session(row),
            extraction_id="ext_1",
            statistics={"n": 1},
            saved_files=["layer_0.npy", "layer_1.npy"],
        )
        assert row.status is ExtractionStatus.CANCELLED
        assert row.saved_files == ["layer_0.npy", "layer_1.npy"], (
            "the artifact is real and must not be orphaned"
        )
        assert "already finished" in row.error_message

    def test_a_live_run_still_completes_normally(self):
        from src.models.activation_extraction import ExtractionStatus
        from src.services.extraction_db_service import ExtractionDatabaseService

        row = _Row("saving")
        ExtractionDatabaseService.mark_completed(
            db=_session(row), extraction_id="ext_1",
            statistics={"n": 1}, saved_files=["layer_0.npy"],
        )
        assert row.status is ExtractionStatus.COMPLETED
        assert row.progress == 100.0


class TestTheReturnedIdIsTheRowsId:
    """Found on hardware, 2026-09-05. The endpoint GUESSED the id.

    It generated `ext_{model}_{now}` and did NOT pass it to the task, which
    generated its own from a second `datetime.now()`. They agree only when both
    land inside the same second — observed straddling one: the endpoint
    returned `..._185140` and the row was created as `..._185141`.

    CANCEL KEYS ON THIS ID. So an operator cancelling with the id they were
    handed got `404 Extraction not found` and the extraction ran on — the exact
    class of silent failure this whole arc exists to remove, sitting in the
    endpoint that Phase 2 rewired. Four static review rounds did not find it;
    the first real GPU run did, which is the repo's own recorded pattern.
    """

    def test_the_endpoint_hands_its_id_to_the_task(self):
        import _cancel_ast as A

        from src.api.v1.endpoints import models as models_endpoint

        calls = A.calls_named(models_endpoint.extract_model_activations, "delay")
        assert calls, "the endpoint no longer dispatches the task"
        passed = [A.keyword_of(c, "extraction_id") for c in calls]
        assert any(v is not None for v in passed), (
            "the endpoint returns an extraction_id it never passes to the "
            "task, so the task invents a different one and the id the caller "
            "holds — and cancels with — does not exist"
        )

    def test_the_task_only_invents_an_id_when_it_is_given_none(self):
        """The other half: if the task ignored the argument the fix is void."""
        import inspect

        from src.workers.model_tasks import extract_activations

        src = inspect.getsource(inspect.unwrap(extract_activations))
        assert "if extraction_id is None:" in src, (
            "the task no longer honours a caller-supplied id"
        )

    def test_two_dispatches_in_one_second_would_still_collide(self):
        """HONEST LIMIT, recorded rather than hidden.

        Passing the id fixes the endpoint/task disagreement. It does NOT make
        the id unique: the format is second-granular, so two extractions
        started for the same model inside one second still collide. That was
        also observed on 2026-09-05 (both jobs came back as `..._184829`).
        Fixing it means changing the id format, which is a wider change than
        this arc — recorded here so the next reader knows it is known.
        """
        import inspect

        from src.api.v1.endpoints import models as models_endpoint

        src = inspect.getsource(models_endpoint.extract_model_activations)
        assert "%Y%m%d_%H%M%S" in src, (
            "the id format changed — if it now carries sub-second precision or "
            "a uuid, this collision note is obsolete and should be deleted"
        )
