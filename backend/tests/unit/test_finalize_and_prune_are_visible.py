"""OSD-36c — Finalize and Prune must appear in Active Operations.

Neither created a `task_queue` row, and Active Operations is built from that table
plus the entity tables. So an operator pressed Finalize, received a 202, and
watched a page that showed nothing running: the work was happening and the only
evidence was the worker log.

THE ORDERING IS THE POINT, and it lives in one helper because this repo has
already paid for that assumption living apart from its code — a guard once
required a row "created by the API before dispatch" which nothing created, and
every UI extraction was silently refused for a week.
"""
import ast
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.v1.endpoints import trainings


class TestTheRowIsWrittenBeforeTheTaskIsSent:

    @pytest.mark.asyncio
    async def test_the_row_exists_before_dispatch(self):
        order = []

        async def fake_create(db, **kwargs):
            order.append(("row", kwargs["task_type"], kwargs["entity_id"], kwargs["task_id"]))
            return MagicMock(id="tq_1")

        def fake_dispatch(task_id):
            order.append(("dispatch", task_id))
            return MagicMock(id=task_id)

        with patch.object(trainings.TaskQueueService, "create_task_entry", fake_create):
            returned = await trainings._queue_visible(
                AsyncMock(), task_type="training_finalize",
                training_id="train_x", dispatch=fake_dispatch,
            )

        assert [step[0] for step in order] == ["row", "dispatch"], (
            "dispatching before the row exists lets a task move a row that is not there"
        )
        assert order[0][1:3] == ("training_finalize", "train_x")
        assert order[0][3] == returned == order[1][1], (
            "the row must carry the SAME celery id the task runs under, or Active "
            "Operations cannot reconcile the row against the task"
        )

    @pytest.mark.asyncio
    async def test_the_entity_is_the_training(self):
        captured = {}

        async def fake_create(db, **kwargs):
            captured.update(kwargs)
            return MagicMock(id="tq_1")

        with patch.object(trainings.TaskQueueService, "create_task_entry", fake_create):
            await trainings._queue_visible(
                AsyncMock(), task_type="checkpoint_prune",
                training_id="train_y", dispatch=lambda task_id: MagicMock(id=task_id),
            )
        assert captured["entity_type"] == "training"
        assert captured["entity_id"] == "train_y"


class TestBothRoutesUseIt:
    """Assert the CALL, by AST — a name search matches the docstrings above."""

    def _calls_in(self, fn) -> set:
        tree = ast.parse(inspect.getsource(fn))
        return {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    def test_finalize_queues_visibly(self):
        assert "_queue_visible" in self._calls_in(trainings.finalize_training), (
            "finalize dispatches without a task_queue row, so it is invisible again"
        )

    def test_prune_queues_visibly(self):
        assert "_queue_visible" in self._calls_in(trainings.prune_checkpoints_now)

    def test_neither_route_still_calls_delay_directly(self):
        """`.delay` cannot carry a chosen task id, so it cannot keep the row and
        the task in agreement."""
        for fn in (trainings.finalize_training, trainings.prune_checkpoints_now):
            source = inspect.getsource(fn)
            tree = ast.parse(source)
            delays = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "delay"
            ]
            assert not delays, f"{fn.__name__} still dispatches with .delay"


class TestTheTaskTypesAreDistinct:
    """Active Operations groups by task_type; two operations sharing one label
    cannot be told apart in the list."""

    @pytest.mark.asyncio
    async def test_finalize_and_prune_do_not_share_a_label(self):
        seen = []

        async def fake_create(db, **kwargs):
            seen.append(kwargs["task_type"])
            return MagicMock(id="tq_1")

        with patch.object(trainings.TaskQueueService, "create_task_entry", fake_create):
            for task_type in ("training_finalize", "checkpoint_prune"):
                await trainings._queue_visible(
                    AsyncMock(), task_type=task_type, training_id="t",
                    dispatch=lambda task_id: MagicMock(id=task_id),
                )
        assert len(set(seen)) == 2
