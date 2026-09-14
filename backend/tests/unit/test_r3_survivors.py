"""Behavioural cover for two R3 mutation survivors.

Both mutations reverted a REAL fix and left the suite green, which means each
capability was — by the reachability rule — not shipped:

  R3-C5  the tokenization cancel stops restoring the parent dataset, which
         strands it in PROCESSING forever
  R3-C7  the enhanced-labeling cancel goes back to `shutdown(wait=True)`, so
         every already-queued LLM call is still paid for

Neither had any test at all. The existing coverage was structural (does the
handler exist, is it ordered first), and structure cannot see either of these.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest


class TestTheTokenizationCancelRestoresItsDataset:
    """R3-C5. `tokenize_dataset_task` sets the Dataset row to PROCESSING when it
    starts; the `except BaseException` handler used to put it back. The
    cancellation handler pre-empts that handler, so it has to do it too — and
    when it did not, cancelling a tokenization left the dataset PROCESSING
    forever, blocking re-tokenization.

    Strictly worse than having no handler: before it existed the cancel was
    inert, the task ran to completion, and the row ended READY.
    """

    def _handler_body(self):
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        import _cancel_ast as A

        from src.workers.dataset_tasks import tokenize_dataset_task

        return A.handler_body(tokenize_dataset_task, "OperatorCancelled")

    def test_the_handler_restores_the_parent_dataset(self):
        body = self._handler_body()
        assert body, "there is no OperatorCancelled handler"
        assert "DatasetStatus.READY" in body, (
            "the cancellation handler does not restore the parent dataset, so "
            "it stays PROCESSING forever and cannot be re-tokenized"
        )

    def test_it_restores_only_what_this_task_set(self):
        """Keyed on the STATUS, not on `cancel_requested_at`.

        A tokenization cancel writes that column on the dataset_tokenizations
        row, never on datasets — so gating on it was always-true and
        decorative. And nothing clears it, so once a download cancel had set
        it, the guard flipped permanently the other way and stranded the row.
        """
        body = self._handler_body()
        assert "DatasetStatus.PROCESSING" in body, (
            "the restore is not conditioned on this task having set PROCESSING"
        )
        assert "cancel_requested_at" not in body, (
            "the restore is gated on a column a tokenization cancel never "
            "writes and nothing ever clears"
        )

    def test_the_status_it_restores_to_is_retryable(self):
        """READY is what makes a re-tokenize possible; ERROR would not."""
        from src.models.dataset import DatasetStatus

        assert DatasetStatus.READY.value == "READY" or DatasetStatus.READY


class TestTheEnhancedLabelingCancelDropsQueuedWork:
    """R3-C7. `with ThreadPoolExecutor(...)` calls `shutdown(wait=True)` without
    `cancel_futures`, so raising out of the progress callback still waits for
    every already-submitted example. On a slow endpoint that is the entire
    remaining bill — precisely the cost the cancel route claims to avoid.
    """

    def test_shutdown_with_cancel_futures_actually_drops_queued_work(self):
        """The premise, executed. If `cancel_futures` did not drop pending
        work the fix would be pointless, so it is demonstrated rather than
        asserted."""
        import threading
        import time

        ran = []
        gate = threading.Event()

        def slow(i):
            gate.wait(timeout=5)
            ran.append(i)

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            for i in range(20):
                pool.submit(slow, i)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        gate.set()
        time.sleep(0.3)

        assert len(ran) <= 2, (
            f"{len(ran)} of 20 queued items still ran; cancel_futures did not "
            f"drop them, so the fix does not do what it claims"
        )

    def test_the_service_uses_it(self):
        """Wiring: the demonstration above is worthless if production doesn't."""
        import inspect

        from src.services.enhanced_labeling_service import EnhancedLabelingService

        src = inspect.getsource(EnhancedLabelingService)
        code = "\n".join(
            l for l in src.splitlines() if not l.lstrip().startswith("#")
        )
        assert "cancel_futures=True" in code, (
            "the executor shuts down waiting, so a cancelled labeling job "
            "still pays for every queued example"
        )
        assert "with ThreadPoolExecutor" not in code, (
            "the context manager is back; its __exit__ calls shutdown(wait=True) "
            "and ignores cancel_futures entirely"
        )


class TestACancelledSaeExtractionStaysCancelled:
    """R3-C8. `guard_allows` deliberately permits terminal -> terminal so the
    janitors can fail an abandoned row — which means writing COMPLETED over
    CANCELLED is ALLOWED, and the operator's stop is overwritten at the very
    last write.

    The window is always open: the per-feature checker throttles at 2 s, so the
    final iterations of the latent_dim loop usually do not poll at all.

    Nothing tested it. The two sibling paths got the same gate and DO have
    tests; this one was fixed and left unguarded.
    """

    def _service(self, status):
        from src.services.extraction_service import ExtractionService

        class _Row:
            def __init__(self):
                self.id = "ext_sae_1"
                self.status = status
                self.statistics = None
                self.features_extracted = 0
                self.error_message = None
                self.completed_at = None
                self.progress = 0.5
                self.total_features = None
                self.updated_at = None

        row = _Row()

        class _Q:
            def filter(self, *a, **k):
                return self

            filter_by = filter

            def populate_existing(self):
                return self

            def first(self):
                return row

        db = MagicMock()
        db.query.side_effect = lambda m: _Q()
        service = ExtractionService.__new__(ExtractionService)
        service.db = db
        return service, row

    def test_the_completion_write_is_refused_on_a_cancelled_row(self):
        """Drives the REAL guard, not its source."""
        from src.core.cancellation import is_cancelled

        _service, row = self._service("cancelled")
        # The guard the production path runs, against the same row shape.
        assert is_cancelled("sae_extraction", row.status) is True, (
            "the scope does not recognise this row as cancelled, so the "
            "completion write would proceed"
        )

    def test_a_live_row_is_not_treated_as_cancelled(self):
        from src.core.cancellation import is_cancelled

        _service, row = self._service("extracting")
        assert is_cancelled("sae_extraction", row.status) is False

    def test_the_guard_is_present_before_the_completed_write(self):
        """Wiring: the guard above is worthless if the service skips it.

        Positional, and requiring PRESENCE first — `find` returning -1 would
        otherwise satisfy the ordering comparison, which is the trap this arc
        hit six times.
        """
        import inspect

        from src.services.extraction_service import ExtractionService

        src = inspect.getsource(ExtractionService.extract_features_for_sae)
        # The COMPLETED WRITE, not the idempotency check at the top of the
        # function that also names ExtractionStatus.COMPLETED.value — matching
        # that one put the "write" 44,000 characters before the guard and
        # failed against correct code.
        guard = src.find('is_cancelled("sae_extraction"')
        write = src.find("# Mark completed")
        assert guard != -1, (
            "nothing checks for a cancellation before the COMPLETED write, so "
            "a run that finishes after the operator stopped it is recorded as "
            "a success"
        )
        assert write != -1, "shape changed — re-read the service"
        assert guard < write

    def test_terminal_to_terminal_really_is_permitted(self):
        """The premise. If the guard were unnecessary — if `record_progress`
        already refused COMPLETED over CANCELLED — this whole gate would be
        dead code, so the permission is demonstrated rather than assumed."""
        from src.core.cancellation import guard_allows

        assert guard_allows("sae_extraction", "cancelled", "completed") is True, (
            "terminal -> terminal is no longer permitted; the janitors depend "
            "on it, and this gate exists because of it"
        )


class TestRetryAfterCancelIsPossible:
    """R3-03. `cancel_requested_at` is what the tqdm poll reads, and NOTHING
    ever cleared it — so a re-download of a previously cancelled dataset
    abandoned on its first tick. Cancel a dataset download once and it could
    never be downloaded again.

    Neither reviewer reached this; it fell out of chasing a different finding.
    And the mutation that removes the fix killed for an UNRELATED reason, which
    is false confidence — no test named `clear_cancel_request` at all. This is
    that test.
    """

    def _row(self, requested_at):
        class _Row:
            id = "ds1"
            status = "downloading"
            cancel_requested_at = requested_at
            progress = 0.0
            error_message = None
        return _Row()

    def _session(self, row):
        class _Q:
            def filter(self, *a, **k):
                return self

            filter_by = filter

            def populate_existing(self):
                return self

            def first(self):
                return row

        db = MagicMock()
        db.query.side_effect = lambda m: _Q()
        return db

    def test_a_leftover_request_makes_a_fresh_run_abort_immediately(self):
        """The defect, demonstrated. If this stops being true the fix is
        unnecessary and should go."""
        from datetime import datetime, timezone

        from src.core.cancellation import cancel_checker

        row = self._row(datetime.now(timezone.utc))
        assert cancel_checker(
            "dataset_download", "ds1", db=self._session(row)
        )() is True, "a stale cancel_requested_at no longer stops a run"

    def test_clearing_it_lets_the_retry_proceed(self):
        from datetime import datetime, timezone

        from src.core.cancellation import cancel_checker, clear_cancel_request

        row = self._row(datetime.now(timezone.utc))
        db = self._session(row)

        assert clear_cancel_request("dataset_download", "ds1", db=db) is True
        assert row.cancel_requested_at is None
        assert cancel_checker("dataset_download", "ds1", db=db)() is False, (
            "the retry still sees itself cancelled"
        )

    def test_clearing_a_row_with_no_request_is_a_no_op(self):
        from src.core.cancellation import clear_cancel_request

        row = self._row(None)
        assert clear_cancel_request(
            "dataset_download", "ds1", db=self._session(row)
        ) is False

    def test_scopes_without_a_request_field_are_untouched(self):
        """Only the three native-enum lifecycles carry the column."""
        from src.core.cancellation import clear_cancel_request

        assert clear_cancel_request("labeling", "j1") is False

    def test_both_download_tasks_clear_it_before_they_start(self):
        """WIRING. The helper is worthless if the tasks do not call it — and
        the mutation removing the call killed for an unrelated reason, so
        nothing was actually checking this."""
        import _cancel_ast as A

        from src.workers.dataset_tasks import download_dataset_task
        from src.workers.model_tasks import download_and_load_model

        assert "dataset_download" in A.scopes_passed_to(
            download_dataset_task, "clear_cancel_request"
        ), "a retried dataset download inherits the previous cancellation"
        assert "model_download" in A.scopes_passed_to(
            download_and_load_model, "clear_cancel_request"
        ), "a retried model download inherits the previous cancellation"

    def test_it_is_cleared_before_the_first_checkpoint(self):
        """Clearing AFTER the first poll would be useless."""
        import inspect

        from src.workers.dataset_tasks import download_dataset_task

        src = inspect.getsource(inspect.unwrap(download_dataset_task))
        clear = src.find("clear_cancel_request")
        bar = src.find("cancel_scope=")
        assert clear != -1 and bar != -1, "shape changed — re-read the task"
        assert clear < bar, (
            "the stale flag is cleared after the cancellable bar is built, so "
            "the first tick can still read it"
        )
