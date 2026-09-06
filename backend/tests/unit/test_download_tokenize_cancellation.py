"""Phase 4 — the fork-pool lifecycles: downloads and tokenization.

THE HARD PART IS WHERE A CHECK CAN EVEN RUN. `Dataset.map(num_proc=N)` forks a
worker pool and the mapper executes in the children; a child cannot coordinate a
stop with its siblings, and one that dies becomes "One of the subprocesses has
abruptly died" — a cancellation indistinguishable from a crash. But `datasets`
funnels every batch's progress back through a manager queue and ticks the
progress bar in the PARENT. That tick is the only owner-process checkpoint that
exists, and for a HuggingFace download tqdm is the only in-process callback of
any kind.

THE STATUS COLUMN CANNOT SAY "CANCELLED". `datasets`, `models` and
`dataset_tokenizations` are native Postgres enums without the member, and
`ALTER TYPE ... ADD VALUE` is non-transactional. They carry
`cancel_requested_at` instead (migration f3c8a92b1e07), which also separates
"the operator asked" from "the job stopped" — the conflation behind today's
`status = ERROR` + "Cancelled by user".
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.core import cancellation as C
from src.core.cancellation import OperatorCancelled


class _Row:
    def __init__(self, status="processing", cancel_requested_at=None):
        self.id = "tok_1"
        self.status = status
        self.progress = 10.0
        self.error_message = None
        self.completed_at = None
        self.cancel_requested_at = cancel_requested_at
        self.celery_task_id = "celery-tok"


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


class TestTheRequestLivesInItsOwnColumn:
    @pytest.mark.parametrize(
        "kind", ["dataset_download", "model_download", "dataset_tokenization"]
    )
    def test_the_request_is_a_timestamp_not_a_status(self, kind):
        scope = C.get_scope(kind)
        assert scope.request_field == "cancel_requested_at"

    @pytest.mark.parametrize(
        "kind", ["dataset_download", "model_download", "dataset_tokenization"]
    )
    def test_ready_and_error_are_this_familys_terminal_states(self, kind):
        """Using the default {completed, failed, cancelled} here would mean the
        guard never considered any of these rows terminal, so a straggling
        progress write could revive a finished download."""
        scope = C.get_scope(kind)
        assert "ready" in scope.terminal_values
        assert "error" in scope.terminal_values

    def test_requesting_a_cancel_writes_the_timestamp_and_not_the_status(self):
        row = _Row("processing")
        out = C.request_cancel("dataset_tokenization", "tok_1", db=_session(row))
        assert out.requested is True
        assert row.cancel_requested_at is not None
        assert row.status == "processing", (
            "the native enum has no CANCELLED member; writing one would fail "
            "at flush on the cancellation path itself"
        )

    def test_the_checker_reads_the_timestamp(self):
        row = _Row("processing", cancel_requested_at=datetime.now(timezone.utc))
        assert C.cancel_checker("dataset_tokenization", "tok_1", db=_session(row))() is True

    def test_a_live_row_is_not_cancelled(self):
        assert C.cancel_checker(
            "dataset_tokenization", "tok_1", db=_session(_Row("processing"))
        )() is False

    def test_a_finished_download_is_not_cancellable(self):
        out = C.request_cancel("dataset_download", "ds_1", db=_session(_Row("ready")))
        assert out.requested is False
        assert "already ready" in out.detail


class TestTheTqdmBridgeIsTheCheckpoint:
    def _bar(self, row, total=1000):
        from src.workers.tqdm_websocket_bridge import TqdmWebSocketCallback

        with patch("src.core.database.get_sync_db") as gsd:
            gsd.return_value.__enter__.return_value = _session(row)
            bar = TqdmWebSocketCallback(
                total=total, dataset_id="ds-1", tokenization_id="tok_1",
                cancel_scope="dataset_tokenization", cancel_target="tok_1",
                disable=True, desc="test",
            )
        return bar

    def test_a_cancelled_run_raises_from_update(self):
        row = _Row("processing", cancel_requested_at=datetime.now(timezone.utc))
        bar = self._bar(row)
        with patch("src.core.database.get_sync_db") as gsd:
            gsd.return_value.__enter__.return_value = _session(row)
            with pytest.raises(OperatorCancelled):
                bar.update(1)

    def test_the_raise_survives_an_except_Exception_around_the_tick(self):
        """This module and `datasets` both wrap progress ticks in
        `except Exception`. An Exception-derived cancel would be logged as a
        dropped tick and the map would run to completion."""
        row = _Row("processing", cancel_requested_at=datetime.now(timezone.utc))
        bar = self._bar(row)
        escaped = None
        with patch("src.core.database.get_sync_db") as gsd:
            gsd.return_value.__enter__.return_value = _session(row)
            try:
                try:
                    bar.update(1)
                except Exception:  # noqa: BLE001 - the production shape
                    pass
            except OperatorCancelled as exc:
                escaped = exc
        assert escaped is not None

    def test_a_live_run_is_not_interrupted(self):
        """The negative side: the checkpoint must not stop healthy work.

        `n` is not asserted — a tqdm built with `disable=True` does not track
        it, and this bar is disabled to keep the test silent.
        """
        row = _Row("processing")
        bar = self._bar(row)
        with patch("src.core.database.get_sync_db") as gsd:
            gsd.return_value.__enter__.return_value = _session(row)
            bar.update(1)  # must not raise

    def test_a_bar_with_no_scope_never_polls(self):
        """Most tqdm instances in the process are not cancellable jobs."""
        from src.workers.tqdm_websocket_bridge import TqdmWebSocketCallback

        bar = TqdmWebSocketCallback(total=10, disable=True, desc="test")
        assert bar._cancel is None
        bar.update(1)

    def test_the_poll_precedes_the_parent_bookkeeping(self):
        import inspect

        from src.workers.tqdm_websocket_bridge import TqdmWebSocketCallback

        src = inspect.getsource(TqdmWebSocketCallback.update)
        # COMMENTS STRIPPED. This matched "raise_if_cancelled" inside the
        # comment that EXPLAINS the ordering — and that comment stays above
        # `super().update()` however far the actual call moves, so the
        # assertion held while the poll sat after the bookkeeping.
        code = "\n".join(
            l for l in src.splitlines() if not l.lstrip().startswith("#")
        )
        poll = code.index("raise_if_cancelled")
        parent = code.index("super().update(n)")
        assert poll < parent, (
            "the poll runs after the parent bookkeeping, so a cancelled run "
            "still advances the bar and emits one more progress event"
        )


class TestTheTaskDoesNotEatTheCancellation:
    def test_operator_cancelled_is_handled_before_the_baseexception_catch_all(self):
        """That handler exists for SystemExit from the signal handler and
        catches BaseException to get it — so it also catches OperatorCancelled,
        and would record a cancel as `Tokenization failed: ...`, or as a
        SUCCESS if the data happened to be saved already."""
        import inspect

        from src.workers.dataset_tasks import tokenize_dataset_task

        src = inspect.getsource(inspect.unwrap(tokenize_dataset_task))
        cancelled = src.index("except OperatorCancelled")
        catch_all = src.index("except BaseException")
        assert cancelled < catch_all, (
            "the BaseException catch-all precedes the cancellation handler, so "
            "it swallows every cancel"
        )

    def test_the_signal_handler_is_still_installed(self):
        """It is NOT dead code — it is what lets `Dataset.map`'s pool children
        die to their own default handler instead of raising SystemExit into the
        owner's inherited one."""
        import inspect

        from src.workers.dataset_tasks import tokenize_dataset_task

        src = inspect.getsource(inspect.unwrap(tokenize_dataset_task))
        assert "make_tokenization_signal_handler" in src


class TestTheTasksActuallyWireTheBar:
    """A CancellableTqdm nobody passes a scope to is an ordinary progress bar.

    Both mutations that removed the wiring left the suite green: the bridge was
    tested in isolation and nothing asserted that either task asks for it.
    """

    def _task_source(self, name):
        import inspect

        from src.workers import dataset_tasks

        return inspect.getsource(inspect.unwrap(getattr(dataset_tasks, name)))

    def test_tokenization_asks_for_a_cancellable_bar(self):
        src = self._task_source("tokenize_dataset_task")
        assert 'cancel_scope="dataset_tokenization"' in src
        assert "cancel_target=tokenization_id" in src

    def test_the_download_asks_for_a_cancellable_bar(self):
        src = self._task_source("download_dataset_task")
        assert 'cancel_scope="dataset_download"' in src
        assert "cancel_target=dataset_id" in src


class TestTheCancelledDownloadCleansUpAfterItself:
    """The other half of moving deletion out of the endpoint.

    Removing the endpoint's rmtree without this is not a fix, it is a storage
    leak: the partial directory would survive with nobody left to remove it.
    """

    def test_it_really_deletes_the_directory(self, tmp_path):
        """DRIVEN, NOT SCRAPED. A source assertion could not see the mutation
        that wrapped the deletion in `if False:` — the text was still there."""
        from src.workers.dataset_tasks import remove_partial_download

        partial = tmp_path / "raw_data"
        partial.mkdir()
        (partial / "data.arrow").write_text("half a download")

        with patch("src.workers.dataset_tasks.settings") as settings:
            settings.resolve_deletable_path.return_value = partial
            assert remove_partial_download(str(partial)) is True
        assert not partial.exists()

    def test_it_refuses_a_path_outside_the_deletable_roots(self, tmp_path):
        """`raw_path` is API-writable, so it arrives stored, not trusted
        (MIS-E2E-071). A refusal must not delete and must not raise."""
        from src.workers.dataset_tasks import remove_partial_download

        protected = tmp_path / "precious"
        protected.mkdir()

        with patch("src.workers.dataset_tasks.settings") as settings:
            settings.resolve_deletable_path.side_effect = ValueError("not deletable")
            assert remove_partial_download(str(protected)) is False
        assert protected.exists()

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        from src.workers.dataset_tasks import remove_partial_download

        gone = tmp_path / "never_written"
        with patch("src.workers.dataset_tasks.settings") as settings:
            settings.resolve_deletable_path.return_value = gone
            assert remove_partial_download(str(gone)) is False

    def test_it_targets_what_is_actually_on_disk(self):
        """R1-05(c). The handler cleaned `raw_path` — which does not exist yet.

        The tqdm checkpoint fires during `load_dataset`, and `raw_path` is only
        created by `save_to_disk` AFTER that returns. So the cleanup deleted
        nothing while HuggingFace's real cache — `downloads/` and the
        `repo___id` arrow tree — leaked with the endpoint no longer removing it
        either. A no-op cleanup is worse than the live-directory delete it
        replaced.
        """
        import _cancel_ast as A

        from src.workers.dataset_tasks import download_dataset_task

        import inspect

        body = A.handler_body(download_dataset_task, "OperatorCancelled")
        assert "_cleanup_paths" in body, (
            "the handler no longer cleans up the paths the task recorded"
        )
        # The list itself is built next to the assignments it depends on, so
        # the handler cannot reference a name that was never bound. Assert what
        # goes INTO it — the transfer cache, not just raw_path, which does not
        # exist at cancel time.
        src = inspect.getsource(inspect.unwrap(download_dataset_task))
        built = src[src.index("_cleanup_paths = ["):]
        built = built[: built.index("]")]

        # THIS JOB'S OUTPUT, not the shared cache. R1 cleaned `downloads/`
        # unconditionally — but `data_dir` is `settings.datasets_dir`, the
        # cache_dir every dataset shares, so that threw away other jobs'
        # resumable chunks and ignored the operator's auto_cleanup setting.
        # What is genuinely this job's is keyed on repo_id.
        assert "___" in built, (
            "the per-repo arrow tree is not cleaned up, so the cancelled "
            "download's real output leaks — raw_path alone does not exist at "
            "cancel time, which made the cleanup a no-op"
        )
        assert "raw_path" in built

        after = src[src.index("_cleanup_paths = ["):]
        assert "auto_cleanup_after_download" in after, (
            "the shared downloads/ cache is cleaned unconditionally again; it "
            "holds other datasets' resumable chunks"
        )
        assert '"downloads"' not in built, (
            "the shared transfer cache is in the unconditional list"
        )

    def test_the_cancellation_handler_calls_it(self):
        """Wiring: the helper is worthless if the handler does not reach it."""
        import inspect

        from src.workers.dataset_tasks import download_dataset_task

        # R1-08. `src[handler:]` is the rest of the WHOLE function, so moving
        # the cleanup out of the cancel handler into the generic one left this
        # green — and leaked the partial directory, which is the specific thing
        # this class exists to prevent.
        import _cancel_ast as A

        body = A.handler_body(download_dataset_task, "OperatorCancelled")
        assert body, "there is no OperatorCancelled handler to clean up in"
        assert "remove_partial_download" in body, (
            "the cleanup is not in the cancellation handler; a cancelled "
            "download leaves its partial output behind and the endpoint no "
            "longer deletes it either"
        )

    def test_the_handler_precedes_the_generic_one(self):
        import inspect

        from src.workers.dataset_tasks import download_dataset_task

        import _cancel_ast as A

        assert A.catches_before(
            download_dataset_task, "OperatorCancelled", "Exception"
        ), (
            "no single try lists OperatorCancelled before the generic handler, "
            "so a cancel is recorded as a download failure"
        )


class TestTheEndpointDoesNotDeleteALiveDirectory:
    def test_the_cancel_path_leaves_a_started_download_to_clean_up_after_itself(self):
        import inspect

        from src.workers.dataset_tasks import cancel_dataset_download

        # R1-09: index() found the ASSIGNMENT, trivially before the rmtree.
        import _cancel_ast as A

        guarded, total = A.guard_counts(
            cancel_dataset_download, "job_had_started", "rmtree"
        )
        assert total >= 1, "the cancel path deletes nothing at all now"
        assert guarded >= 1, (
            "no rmtree is guarded by job_had_started; the cancel path deletes "
            "the directory a live download is writing into, and "
            "revoke(terminate=) is inert on a solo pool so it IS still running"
        )

    def test_the_tokenization_endpoint_no_longer_sigkills(self):
        import inspect

        from src.api.v1.endpoints import datasets as ds

        src = inspect.getsource(ds.cancel_dataset_tokenization)
        code = "\n".join(
            l for l in src.splitlines() if not l.lstrip().startswith("#")
        )
        assert "request_cancel" in code
        assert "SIGKILL" not in code, (
            "killing a solo worker crashed the pool and stranded an acks_late "
            "message for the full 12-hour visibility timeout"
        )
