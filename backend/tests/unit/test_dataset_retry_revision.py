"""A retried dataset download keeps its commit (035 FR-7, D12).

Drives the REAL `POST /api/v1/task-queue/{id}/retry` with a failed row shaped exactly as the
download task's failure handler writes it (`test_dataset_task_revision.py` pins that shape), and
records the re-dispatch: payload and call count.

MUTATION CONTROL: C11 drop `revision=retry_params.get("revision")` from the dispatch → red.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.models.task_queue import TaskQueue

pytestmark = pytest.mark.asyncio

COMMIT = "c" * 40


async def test_the_retry_dispatches_the_failed_attempts_commit(client, async_session):
    dataset_id = str(uuid.uuid4())
    entry = TaskQueue(
        id=f"tq_{uuid.uuid4().hex[:12]}", task_id="old", task_type="download",
        entity_id=dataset_id, entity_type="dataset", status="failed", progress=0.0,
        error_message="connection reset",
        retry_params={"repo_id": "org/ds", "split": "train", "config": "c", "revision": COMMIT},
        retry_count=0,
    )
    async_session.add(entry)
    await async_session.commit()

    with patch("src.workers.dataset_tasks.download_dataset_task.delay") as delay:
        delay.return_value = MagicMock(id="new-task")
        r = await client.post(f"/api/v1/task-queue/{entry.id}/retry", json={})
    assert r.status_code == 200, r.text
    assert delay.call_count == 1
    kwargs = delay.call_args.kwargs
    assert kwargs["revision"] == COMMIT
    assert (kwargs["dataset_id"], kwargs["repo_id"], kwargs["split"], kwargs["config"]) == (
        dataset_id, "org/ds", "train", "c",
    )


async def test_a_pre_035_failure_retries_with_no_revision(client, async_session):
    """No key → None; the task then reads the row's column (TI5) rather than inventing one."""
    entry = TaskQueue(
        id=f"tq_{uuid.uuid4().hex[:12]}", task_id="old", task_type="download",
        entity_id=str(uuid.uuid4()), entity_type="dataset", status="failed", progress=0.0,
        retry_params={"repo_id": "org/ds", "split": None, "config": None}, retry_count=0,
    )
    async_session.add(entry)
    await async_session.commit()
    with patch("src.workers.dataset_tasks.download_dataset_task.delay") as delay:
        delay.return_value = MagicMock(id="new-task")
        r = await client.post(f"/api/v1/task-queue/{entry.id}/retry", json={})
    assert r.status_code == 200, r.text
    assert delay.call_args.kwargs["revision"] is None
