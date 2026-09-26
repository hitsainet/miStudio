"""Shape B — A CANCELLATION IS NOT OVERWRITTEN BY THE NEXT PROGRESS WRITE.

Cancellation here is cooperative: the endpoint writes a terminal status and the
task notices at its next checkpoint. In between, the task is still narrating.
Every writer it narrates through must refuse to move a terminal row, or the
request is lost and the operator is told it worked — which is worse than having
no cancel at all.

These tests drive the REAL writers, not `record_progress`. `test_cancellation_core`
already covers the rule; what is unproven without these is that each writer is
actually standing behind it. Removing a guard must turn one of these red.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core import cancellation as C


class _Query:
    """Models the IDENTITY MAP, so dropping populate_existing is detectable.

    R1-06: this returned the same row whether or not `populate_existing()` was
    called, so the guard could be reading stale state and every test still
    passed. On a real SQLAlchemy Query `populate_existing()` returns a NEW
    Query — `q.populate_existing()` with the result discarded reads the row as
    the session first loaded it, which is MIS-E2E-057 exactly.
    """

    def __init__(self, session, log):
        self._session = session
        self._log = log
        self._populate = False

    def filter(self, *a, **k):
        return self

    filter_by = filter

    def order_by(self, *a, **k):
        return self

    def populate_existing(self):
        self._log.append("populate_existing")
        fresh = _Query(self._session, self._log)
        fresh._populate = True
        return fresh

    def first(self):
        if self._populate:
            self._session.cached = self._session.db_row
        return self._session.cached


class _Sess:
    def __init__(self, row):
        self.cached = row
        self.db_row = row


def _session(row, log=None):
    log = [] if log is None else log
    state = _Sess(row)
    db = MagicMock()
    db.query.side_effect = lambda model: _Query(state, log)
    db._log = log
    db._state = state
    return db


class _ExtractionRow:
    """Shaped like ActivationExtraction, whose status is a REAL python enum."""

    def __init__(self, status):
        from src.models.activation_extraction import ExtractionStatus
        self.status = ExtractionStatus(status)
        self.progress = 12.0
        self.samples_processed = 40
        self.error_message = None
        self.completed_at = None


class TestActivationExtractionWriter:
    """`ExtractionDatabaseService.update_progress` — called every ~10 samples."""

    def _call(self, row, status_name, progress, samples):
        from src.models.activation_extraction import ExtractionStatus
        from src.services.extraction_db_service import ExtractionDatabaseService

        return ExtractionDatabaseService.update_progress(
            db=_session(row),
            extraction_id="ext_1",
            progress=progress,
            status=ExtractionStatus[status_name],
            samples_processed=samples,
        )

    @pytest.mark.parametrize("terminal", ["CANCELLED", "COMPLETED", "FAILED"])
    def test_a_terminal_extraction_is_not_dragged_back_to_extracting(self, terminal):
        row = _ExtractionRow(terminal.lower())
        assert self._call(row, "EXTRACTING", 55.0, 500) is None
        assert row.status.value == terminal.lower()
        assert row.progress == 12.0, "a refused write must move nothing"
        assert row.samples_processed == 40

    def test_a_live_extraction_still_records_progress(self):
        row = _ExtractionRow("extracting")
        self._call(row, "EXTRACTING", 55.0, 500)
        assert row.progress == 55.0
        assert row.samples_processed == 500

    def test_the_status_is_written_as_the_columns_own_enum_member(self):
        """`activation_extractions.status` is SQLEnum WITHOUT values_callable,
        so SQLAlchemy persists the member NAME. A bare "cancelled" string is not
        a key it can look up, and it would fail at flush on the cancel path."""
        from src.models.activation_extraction import ExtractionStatus

        row = _ExtractionRow("extracting")
        C.request_cancel("activation_extraction", "ext_1", db=_session(row))
        assert row.status is ExtractionStatus.CANCELLED


class _JobRow:
    def __init__(self, status):
        self.status = status
        self.progress = 0.2
        self.features_extracted = 100
        self.total_features = None
        self.statistics = None
        self.error_message = None
        self.completed_at = None
        self.updated_at = None


class TestSaeExtractionWriter:
    """`ExtractionService.update_extraction_status_sync` — the SAE lifecycle."""

    def _service(self, row):
        from src.services.extraction_service import ExtractionService

        service = ExtractionService.__new__(ExtractionService)
        service.db = _session(row)
        return service

    @pytest.mark.parametrize("terminal", ["cancelled", "completed", "failed"])
    def test_a_terminal_job_is_not_dragged_back_to_extracting(self, terminal):
        row = _JobRow(terminal)
        with patch("src.services.extraction_service.emit_progress", create=True):
            self._service(row).update_extraction_status_sync(
                extraction_id="ex1",
                status="extracting",
                progress=0.9,
                features_extracted=9000,
            )
        assert row.status == terminal
        assert row.progress == 0.2
        assert row.features_extracted == 100

    def test_a_cancelled_job_emits_no_further_websocket_progress(self):
        """The refusal must reach the UI too: a card that keeps advancing after
        the operator cancelled reads as a cancel that did not take."""
        row = _JobRow("cancelled")
        with patch("src.workers.websocket_emitter.emit_progress") as emit:
            self._service(row).update_extraction_status_sync(
                extraction_id="ex1", status="extracting", progress=0.9
            )
        emit.assert_not_called()

    def test_a_live_job_still_records_progress(self):
        row = _JobRow("extracting")
        with patch("src.workers.websocket_emitter.emit_progress"):
            self._service(row).update_extraction_status_sync(
                extraction_id="ex1", status="extracting", progress=0.9
            )
        assert row.progress == 0.9


class _ExportRow:
    def __init__(self, status):
        self.status = status
        self.progress = 10.0
        self.current_stage = "computing"
        self.error_message = None
        self.completed_at = None


class TestNeuronpediaExportWriter:
    """`NeuronpediaTask.update_export_progress` writes NO status at all, so it
    can only be guarded by the progress-move half of the rule."""

    def _task(self, row):
        from contextlib import contextmanager
        from src.workers.neuronpedia_tasks import NeuronpediaTask

        task = NeuronpediaTask.__new__(NeuronpediaTask)

        @contextmanager
        def _db():
            yield _session(row)

        task.get_db = _db
        return task

    @pytest.mark.parametrize("terminal", ["cancelled", "completed", "failed"])
    def test_a_terminal_export_stops_advancing(self, terminal):
        row = _ExportRow(terminal)
        with patch("src.workers.neuronpedia_tasks.emit_export_progress") as emit:
            self._task(row).update_export_progress("j1", 60.0, "packaging")
        assert row.progress == 10.0
        assert row.current_stage == "computing"
        emit.assert_not_called()

    def test_a_live_export_still_advances(self):
        row = _ExportRow("computing")
        with patch("src.workers.neuronpedia_tasks.emit_export_progress") as emit:
            self._task(row).update_export_progress("j1", 60.0, "packaging")
        assert row.progress == 60.0
        assert row.current_stage == "packaging"
        emit.assert_called_once()


class _TrainingRow:
    def __init__(self, status):
        self.status = status
        self.progress = 4.0
        self.current_step = 100
        self.current_loss = 1.0
        self.current_l0_sparsity = None
        self.current_dead_neurons = None
        self.current_learning_rate = None
        self.current_fvu = 0.5
        self.started_at = None
        self.completed_at = None
        self.error_message = None


class TestTrainingTracker:
    def _task(self, row):
        from contextlib import contextmanager
        from src.workers.training_tasks import TrainingTask

        task = TrainingTask.__new__(TrainingTask)

        @contextmanager
        def _db():
            yield _session(row)

        task.get_db = _db
        return task

    @pytest.mark.parametrize("terminal", ["paused", "cancelled", "completed", "failed"])
    def test_a_stopped_training_accrues_no_further_steps(self, terminal):
        row = _TrainingRow(terminal)
        self._task(row).update_training_progress(
            training_id="t1", step=500, total_steps=1000, loss=0.3
        )
        assert row.status == terminal
        assert row.current_step == 100, "a paused row must match its checkpoint"
        assert row.progress == 4.0

    def test_a_live_training_still_records_metrics(self):
        row = _TrainingRow("running")
        self._task(row).update_training_progress(
            training_id="t1", step=500, total_steps=1000, loss=0.3
        )
        assert row.current_step == 500
        assert row.progress == 50.0
        assert row.current_loss == 0.3

    def test_a_reported_fvu_of_none_does_not_erase_a_good_reading(self):
        row = _TrainingRow("running")
        self._task(row).update_training_progress(
            training_id="t1", step=500, total_steps=1000, loss=0.3, fvu=None
        )
        assert row.current_fvu == 0.5


class TestTheGuardCanActuallySeeTheCancel:
    """The guard is only as good as the read in front of it.

    Each writer runs on a Celery task session that has held the row open for
    hours. SQLAlchemy's identity map returns THAT instance, so without
    `populate_existing()` the guard compares against the status the task saw at
    startup and never observes the endpoint's write. The guard would then be
    present, readable, reviewed — and inert.
    """

    def test_the_activation_extraction_writer_rereads_the_row(self):
        from src.models.activation_extraction import ExtractionStatus
        from src.services.extraction_db_service import ExtractionDatabaseService

        db = _session(_ExtractionRow("extracting"))
        ExtractionDatabaseService.update_progress(
            db=db,
            extraction_id="ext_1",
            progress=50.0,
            status=ExtractionStatus.EXTRACTING,
            samples_processed=5,
        )
        assert "populate_existing" in db._log

    def test_the_sae_extraction_writer_rereads_the_row(self):
        from src.services.extraction_service import ExtractionService

        service = ExtractionService.__new__(ExtractionService)
        service.db = _session(_JobRow("extracting"))
        with patch("src.workers.websocket_emitter.emit_progress"):
            service.update_extraction_status_sync(
                extraction_id="ex1", status="extracting", progress=0.5
            )
        assert "populate_existing" in service.db._log

    def test_the_export_writer_rereads_the_row(self):
        from contextlib import contextmanager
        from src.workers.neuronpedia_tasks import NeuronpediaTask

        db = _session(_ExportRow("computing"))
        task = NeuronpediaTask.__new__(NeuronpediaTask)

        @contextmanager
        def _db():
            yield db

        task.get_db = _db
        with patch("src.workers.neuronpedia_tasks.emit_export_progress"):
            task.update_export_progress("j1", 20.0, "computing")
        assert "populate_existing" in db._log

    @pytest.mark.asyncio
    async def test_the_async_sae_writer_asks_for_a_fresh_read(self):
        """The async path cannot call `populate_existing()` on a Query — it sets
        the same option on the statement, so that is what is asserted."""
        from unittest.mock import AsyncMock
        from src.services.extraction_service import ExtractionService

        row = _JobRow("extracting")
        seen = {}

        async def _execute(stmt, *a, **k):
            seen["options"] = stmt.get_execution_options()
            result = MagicMock()
            result.scalar_one_or_none.return_value = row
            return result

        service = ExtractionService.__new__(ExtractionService)
        service.db = MagicMock()
        service.db.execute = _execute
        service.db.commit = AsyncMock()

        await service.update_extraction_status(
            extraction_id="ex1", status="extracting", progress=0.5
        )
        assert seen["options"].get("populate_existing") is True

    @pytest.mark.asyncio
    async def test_the_async_sae_writer_also_refuses_a_terminal_row(self):
        from unittest.mock import AsyncMock
        from src.services.extraction_service import ExtractionService

        row = _JobRow("cancelled")

        async def _execute(stmt, *a, **k):
            result = MagicMock()
            result.scalar_one_or_none.return_value = row
            return result

        service = ExtractionService.__new__(ExtractionService)
        service.db = MagicMock()
        service.db.execute = _execute
        service.db.commit = AsyncMock()

        await service.update_extraction_status(
            extraction_id="ex1", status="extracting", progress=0.9
        )
        assert row.status == "cancelled"
        assert row.progress == 0.2
        service.db.commit.assert_not_awaited()


class TestTheGuardSeesWhatAnotherConnectionWrote:
    """MIS-E2E-057, driven. The one shape that matters and was never exercised.

    R1-06: `assert "populate_existing" in db._log` is a CALL-HAPPENED check.
    On a real SQLAlchemy Query `populate_existing()` returns a NEW Query, so
    `q.populate_existing()` with the result discarded reads the row the session
    first loaded — the call is in the log and the guard is blind. These tests
    diverge the identity map from the database, which is the only way to tell
    the two apart.
    """

    def _diverged(self, loaded_status, db_status):
        """A session holding `loaded_status` while the database says otherwise."""
        row_loaded = _ExtractionRow(loaded_status)
        row_db = _ExtractionRow(db_status)
        db = _session(row_loaded)
        db._state.db_row = row_db
        return db, row_loaded, row_db

    def test_a_cancel_written_elsewhere_stops_the_progress_write(self):
        from src.models.activation_extraction import ExtractionStatus
        from src.services.extraction_db_service import ExtractionDatabaseService

        # The task loaded the row while it was EXTRACTING; the API has since
        # written CANCELLED on another connection.
        db, _loaded, row_db = self._diverged("extracting", "cancelled")

        result = ExtractionDatabaseService.update_progress(
            db=db, extraction_id="ext_1", progress=90.0,
            status=ExtractionStatus.EXTRACTING, samples_processed=900,
        )
        assert result is None, (
            "the writer accepted a progress update onto a row another "
            "connection had already cancelled — it is reading the identity map"
        )
        assert row_db.progress == 12.0, "the refused write still moved the row"

    def test_a_live_row_still_updates_when_the_database_agrees(self):
        from src.models.activation_extraction import ExtractionStatus
        from src.services.extraction_db_service import ExtractionDatabaseService

        db, _loaded, row_db = self._diverged("extracting", "extracting")
        ExtractionDatabaseService.update_progress(
            db=db, extraction_id="ext_1", progress=90.0,
            status=ExtractionStatus.EXTRACTING, samples_processed=900,
        )
        assert row_db.progress == 90.0


class TestJlensShim:
    """`jlens_progress.update_row` is now a shim; the contract must not move."""

    def test_a_cancelled_fit_row_is_not_dragged_back_to_running(self):
        from src.workers import jlens_progress

        row = _TrainingRow("cancelled")
        with patch("src.core.database.get_sync_db") as gsd:
            gsd.return_value.__enter__.return_value = _session(row)
            assert jlens_progress.update_row("celery-1", status="running", progress=50.0) is False
        assert row.status == "cancelled"
        assert row.progress == 4.0

    def test_it_still_stamps_started_at_on_the_running_transition(self):
        from src.workers import jlens_progress

        row = _TrainingRow("queued")
        with patch("src.core.database.get_sync_db") as gsd:
            gsd.return_value.__enter__.return_value = _session(row)
            assert jlens_progress.update_row("celery-1", status="running") is True
        assert row.started_at is not None
