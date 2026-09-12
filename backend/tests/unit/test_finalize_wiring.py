"""Behavioural wiring tests for finalize.

These exist because assertion-of-existence tests let two real regressions
through in review: deleting the `stop_and_finalize` handler branch, and
hard-coding progress=100 in the finalize emitter, BOTH left the suite green.
A schema Literal that accepts an action proves nothing about whether anything
handles it.

MUTATION CONTROLS:
  * delete the `elif action == "stop_and_finalize"` branch -> control tests fail
  * hard-code "progress": 100.0 in the emitter payload -> progress test fails
  * drop the finalized_from_step write -> status test fails
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


class FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter_by(self, **kw):
        return FakeQuery([
            r for r in self._rows
            if all(getattr(r, k, None) == v for k, v in kw.items())
        ])

    def filter(self, *a):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.commits = 0

    def query(self, model):
        return FakeQuery(self._rows.get(model.__name__, []))

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestStopAndFinalizeIsActuallyHandled:
    """The flagship button's action must reach a handler, not just validate."""

    async def test_dispatches_finalize_with_the_right_arguments(self):
        from src.api.v1.endpoints import trainings as ep

        training = SimpleNamespace(
            id="train_x", status="cancelled", celery_task_id="celery-1"
        )

        with patch.object(ep.TrainingService, "stop_training", return_value=training), \
             patch.object(ep, "revoke_task") as revoke, \
             patch(
                 "src.workers.training_finalize_tasks."
                 "finalize_training_from_checkpoint_task.delay"
             ) as delay, \
             patch(
                 "src.services.training_finalize_service.list_checkpoint_steps",
                 return_value=[2000, 4000],
             ):
            result = await ep.control_training(
                control_request=SimpleNamespace(action="stop_and_finalize"),
                training_id="train_x",
                db=MagicMock(),
            )

        # Payload AND call count — "was called" would pass on wrong arguments.
        delay.assert_called_once_with("train_x", None)
        revoke.assert_called_once()
        assert result["success"] is True
        assert "finaliz" in result["message"].lower()

    async def test_does_not_promise_a_finalize_when_there_are_no_checkpoints(self):
        """Telling the user their SAE was saved when nothing was written is worse
        than telling them there was nothing to save."""
        from src.api.v1.endpoints import trainings as ep

        training = SimpleNamespace(id="train_y", status="cancelled", celery_task_id=None)

        with patch.object(ep.TrainingService, "stop_training", return_value=training), \
             patch(
                 "src.workers.training_finalize_tasks."
                 "finalize_training_from_checkpoint_task.delay"
             ) as delay, \
             patch(
                 "src.services.training_finalize_service.list_checkpoint_steps",
                 return_value=[],
             ):
            result = await ep.control_training(
                control_request=SimpleNamespace(action="stop_and_finalize"),
                training_id="train_y",
                db=MagicMock(),
            )

        delay.assert_not_called()
        assert "no checkpoints" in result["message"].lower()

    async def test_unknown_action_still_rejected(self):
        from fastapi import HTTPException

        from src.api.v1.endpoints import trainings as ep

        with pytest.raises(HTTPException) as exc:
            await ep.control_training(
                control_request=SimpleNamespace(action="definitely_not_an_action"),
                training_id="t",
                db=MagicMock(),
            )
        assert exc.value.status_code == 400


class TestFinalizeWorkerBehaviour:
    """The worker's DB writes and emit payload — previously zero coverage."""

    def _run(self, training, emit, return_session=False):
        from src.workers import training_finalize_tasks as tft

        session = FakeSession({"Training": [training]})
        # PromiseProxy resolves to the real task object on first attribute access.
        task = tft.finalize_training_from_checkpoint_task
        task_cls = type(task.__class__ and task._get_current_object()) \
            if hasattr(task, "_get_current_object") else type(task)

        with patch.object(task_cls, "get_db", lambda self: session, create=True), \
             patch.object(
                 tft, "finalize_from_checkpoint",
                 return_value={"checkpoint_step": 10000, "sae_count": 3,
                               "community_format_dir": "/x",
                               "training_id": training.id, "outputs": {}},
             ), \
             patch("src.workers.websocket_emitter.emit_training_progress", emit):
            # .run() invokes the task body with `self` bound to the task.
            result = task.run(training.id, None)
        return session if return_session else result

    def test_sets_completed_and_records_the_step_without_touching_progress(self):
        from src.models.training import TrainingStatus

        training = SimpleNamespace(
            id="train_z", status="cancelled", progress=20.6, current_step=10300,
            completed_at=None, finalized_from_step=None,
            error_message=None, error_traceback=None, hyperparameters={},
        )
        session = self._run(training, MagicMock(), return_session=True)

        # Without this, replacing db.commit() with `pass` leaves the suite green:
        # the assertions below read attributes off an in-memory object that
        # persist whether or not anything was ever written to the database.
        assert session.commits == 1, "the finalize was never committed"

        assert training.status == TrainingStatus.COMPLETED.value
        assert training.finalized_from_step == 10000
        # The honesty invariant: a salvaged run must not claim it went the distance.
        assert training.progress == 20.6
        assert training.current_step == 10300

    def test_emits_the_real_progress_not_a_hardcoded_100(self):
        """Round 2 mutation survivor: replacing this with 100.0 stayed green."""
        training = SimpleNamespace(
            id="train_z", status="cancelled", progress=20.6, current_step=10300,
            completed_at=None, finalized_from_step=None,
            error_message=None, error_traceback=None, hyperparameters={},
        )
        emit = MagicMock()
        self._run(training, emit)

        emit.assert_called_once()
        payload = emit.call_args.kwargs["data"]
        assert payload["progress"] == 20.6, (
            "emitting 100 repaints a full progress bar on a run stopped at 20%"
        )
        assert payload["current_step"] == 10300
        assert payload["finalized_from_step"] == 10000

    def test_does_not_promote_a_run_that_is_no_longer_terminal(self):
        """F6: the dying trainer races us; never blindly overwrite its status.

        stop_and_finalize revokes the trainer, whose except-handler writes
        FAILED. Whoever committed last used to win — so a valid export could
        land on a run showing FAILED with import locked.
        """
        from src.models.training import TrainingStatus

        training = SimpleNamespace(
            id="train_r", status=TrainingStatus.RUNNING.value, progress=40.0,
            current_step=20000, completed_at=None, finalized_from_step=None,
            error_message=None, error_traceback=None, hyperparameters={},
        )
        emit = MagicMock()
        self._run(training, emit)

        assert training.status == TrainingStatus.RUNNING.value, (
            "a non-terminal run was promoted to COMPLETED underneath a live task"
        )
        assert training.finalized_from_step is None
        emit.assert_not_called()

    def test_promotes_a_cancelled_run(self):
        from src.models.training import TrainingStatus

        training = SimpleNamespace(
            id="train_c", status=TrainingStatus.CANCELLED.value, progress=20.6,
            current_step=10300, completed_at=None, finalized_from_step=None,
            error_message=None, error_traceback=None, hyperparameters={},
        )
        self._run(training, MagicMock())
        assert training.status == TrainingStatus.COMPLETED.value

    def test_finalizing_over_a_failed_run_clears_its_crash_record(self):
        """A COMPLETED run must not carry a stack trace beside a green badge."""
        from src.models.training import TrainingStatus

        training = SimpleNamespace(
            id="train_f", status=TrainingStatus.FAILED.value, progress=6.0,
            current_step=3000, completed_at=None, finalized_from_step=None,
            error_message="CUDA out of memory", error_traceback="Traceback...",
            hyperparameters={},
        )
        self._run(training, MagicMock())

        assert training.status == TrainingStatus.COMPLETED.value
        assert training.error_message is None
        assert training.error_traceback is None
        # Preserved, not erased — the run really did crash.
        assert training.hyperparameters["finalized_over_error"] == "CUDA out of memory"


class TestFinalizeEndpointGuards:
    """The /finalize 409 guards had ZERO backend coverage.

    Deleting any guard body left the suite green — the only test touching these
    paths was a frontend test against a MOCKED store, which never exercises the
    server. Each test asserts the status code AND whether .delay fired.
    """

    async def _call(self, status, **kwargs):
        from src.api.v1.endpoints import trainings as ep

        training = SimpleNamespace(id="t1", status=status)
        with patch.object(ep.TrainingService, "get_training", return_value=training), \
             patch(
                 "src.workers.training_finalize_tasks."
                 "finalize_training_from_checkpoint_task.delay"
             ) as delay:
            try:
                result = await ep.finalize_training(
                    training_id="t1", db=MagicMock(), **kwargs
                )
                return None, delay, result
            except Exception as e:
                return e, delay, None

    @pytest.mark.parametrize("status", ["running", "initializing", "pending", "paused"])
    async def test_active_statuses_are_refused(self, status):
        """PAUSED belongs here: its worker is alive and resumable."""
        exc, delay, _ = await self._call(
            status, checkpoint_step=None, allow_failed=False, force=False
        )
        assert exc is not None and exc.status_code == 409
        delay.assert_not_called()

    async def test_completed_requires_force(self):
        """Re-finalizing would replace a finished run's FINAL weights."""
        exc, delay, _ = await self._call(
            "completed", checkpoint_step=None, allow_failed=False, force=False
        )
        assert exc is not None and exc.status_code == 409
        delay.assert_not_called()

    async def test_completed_proceeds_with_force(self):
        exc, delay, result = await self._call(
            "completed", checkpoint_step=None, allow_failed=False, force=True
        )
        assert exc is None
        delay.assert_called_once_with("t1", None)

    async def test_failed_requires_allow_failed(self):
        exc, delay, _ = await self._call(
            "failed", checkpoint_step=None, allow_failed=False, force=False
        )
        assert exc is not None and exc.status_code == 409
        delay.assert_not_called()

    async def test_failed_proceeds_with_allow_failed(self):
        exc, delay, result = await self._call(
            "failed", checkpoint_step=None, allow_failed=True, force=False
        )
        assert exc is None
        delay.assert_called_once_with("t1", None)

    async def test_cancelled_needs_no_escape_hatch(self):
        """The normal rescue path must stay frictionless."""
        exc, delay, result = await self._call(
            "cancelled", checkpoint_step=None, allow_failed=False, force=False
        )
        assert exc is None
        delay.assert_called_once_with("t1", None)

    async def test_explicit_checkpoint_step_is_passed_through(self):
        exc, delay, _ = await self._call(
            "cancelled", checkpoint_step=8000, allow_failed=False, force=False
        )
        assert exc is None
        delay.assert_called_once_with("t1", 8000)
