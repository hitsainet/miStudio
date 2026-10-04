"""Phase 3 — the Neuronpedia export stops, and its cancel is not relabelled.

TWO THINGS ABOUT THIS PATH WERE NOT WHAT THE PLAN ASSUMED.

The Celery task `neuronpedia.execute_export` is DEAD CODE — nothing dispatches
it. The live path is a FastAPI `BackgroundTasks` call running `execute_export`
inside the API process. So the checkpoint belongs in the SERVICE, which both
paths share; a checkpoint added to the Celery task would fire on nothing.

And `execute_export` loads its row once with `db.get(...)`, which returns the
identity-mapped instance without emitting SQL. With `expire_on_commit=False` on
both session factories, `job.status` is frozen at whatever it was when the
export began — so a checkpoint reading it could never observe a cancellation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core import cancellation as C
from src.core.cancellation import OperatorCancelled


class _Job:
    def __init__(self, status="computing"):
        self.id = "exp-1"
        self.status = status
        self.progress = 10.0
        self.current_stage = "computing"
        self.error_message = None
        self.completed_at = None


def _service_with(row):
    """A service whose async session returns `row` from a populate_existing read."""
    from src.services.neuronpedia_export_service import NeuronpediaExportService

    service = NeuronpediaExportService.__new__(NeuronpediaExportService)
    seen = {}

    async def _execute(stmt, *a, **k):
        seen["options"] = stmt.get_execution_options()
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        return result

    db = MagicMock()
    db.execute = _execute
    db.commit = AsyncMock()
    return service, db, seen


class TestTheCheckpointCanActuallySeeACancel:
    @pytest.mark.asyncio
    async def test_it_re_reads_rather_than_trusting_the_loaded_row(self):
        service, db, seen = _service_with(_Job("computing"))
        await service._cancel_point(db, "exp-1")
        assert seen["options"].get("populate_existing") is True, (
            "the checkpoint reads the identity-mapped row, which is frozen at "
            "the status the export started with"
        )

    @pytest.mark.asyncio
    async def test_a_cancelled_row_stops_the_export(self):
        service, db, _ = _service_with(_Job("cancelled"))
        with pytest.raises(OperatorCancelled) as exc:
            await service._cancel_point(db, "exp-1")
        assert exc.value.reason == "cancelled"

    @pytest.mark.asyncio
    async def test_a_deleted_row_also_stops_the_export(self):
        """DELETE /export/{id} removes the row; there is then nothing to write
        results to and nobody waiting for them."""
        service, db, _ = _service_with(None)
        with pytest.raises(OperatorCancelled) as exc:
            await service._cancel_point(db, "exp-1")
        assert exc.value.reason == "deleted"

    @pytest.mark.asyncio
    async def test_a_live_row_does_not_stop_it(self):
        service, db, _ = _service_with(_Job("packaging"))
        await service._cancel_point(db, "exp-1")

    @pytest.mark.asyncio
    async def test_every_stage_boundary_is_a_checkpoint(self):
        """`_update_stage` is the only checkpoint `execute_export` owns — the
        per-feature loops are one level down in three services that each
        swallow with `except Exception`."""
        service, db, _ = _service_with(_Job("cancelled"))
        job = _Job("cancelled")
        with pytest.raises(OperatorCancelled):
            await service._update_stage(db, job, "packaging", 90.0)
        assert job.current_stage == "computing", "a refused stage write moved it"
        assert job.progress == 10.0

    def test_the_completion_write_is_preceded_by_a_checkpoint(self):
        import inspect

        from src.services.neuronpedia_export_service import NeuronpediaExportService

        src = inspect.getsource(NeuronpediaExportService.execute_export)
        complete = src.index("job.status = ExportStatus.COMPLETED.value")
        # PRESENCE FIRST. This used `rfind`, which returns -1 when the
        # checkpoint has been deleted — and -1 < complete is true, so the
        # assertion passed in exactly the case it existed to catch. A mutation
        # removing the checkpoint left the whole suite green.
        check = src.rfind("_cancel_point", 0, complete)
        assert check != -1, (
            "nothing checks for a cancellation before the completion write, so "
            "a finished export overwrites the operator's cancellation with "
            "COMPLETED at the very last write"
        )


class TestTheFailureWriterDoesNotEatTheCancel:
    def _task(self, row):
        from contextlib import contextmanager

        from src.workers.neuronpedia_tasks import NeuronpediaTask

        class _Q:
            def __init__(self, r):
                self._r = r

            def filter_by(self, *a, **k):
                return self

            def populate_existing(self):
                return self

            def first(self):
                return self._r

        db = MagicMock()
        db.query.side_effect = lambda m: _Q(row)
        task = NeuronpediaTask.__new__(NeuronpediaTask)

        @contextmanager
        def _db():
            yield db

        task.get_db = _db
        return task

    def test_a_cancelled_export_is_not_relabelled_failed(self):
        row = _Job("cancelled")
        with patch("src.workers.neuronpedia_tasks.emit_export_progress"):
            self._task(row).mark_export_failed("exp-1", "boom")
        assert row.status == "cancelled", (
            "the bare except Exception around the export turned the operator's "
            "cancellation into a crash report"
        )
        assert row.error_message is None

    def test_a_genuine_failure_is_still_recorded(self):
        row = _Job("computing")
        with patch("src.workers.neuronpedia_tasks.emit_export_progress"):
            self._task(row).mark_export_failed("exp-1", "boom")
        assert row.status == "failed"
        assert row.error_message == "boom"


class TestTheScope:
    def test_a_deleted_export_row_reads_as_cancelled(self):
        check = C.cancel_checker("neuronpedia_export", "gone", db=_missing_session())
        assert check() is True
        assert check.reason == "deleted"


def _missing_session():
    class _Q:
        def filter(self, *a, **k):
            return self

        filter_by = filter

        def populate_existing(self):
            return self

        def first(self):
            return None

    db = MagicMock()
    db.query.side_effect = lambda m: _Q()
    return db
