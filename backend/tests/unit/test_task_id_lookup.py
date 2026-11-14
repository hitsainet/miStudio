"""A task id a tool handed out resolves (MCP gap, 2026-11-10).

Found live: `start_probe_training_run` answered `{id, task_id}` (its `task_id` is the Celery id), and
`get_task_status` with that id answered `{"data": []}` — the 25-row task-queue list — because the tool had no
`task_id` argument (the name was silently dropped; the server-wide refusal now names it) and a Celery id is not a
task-queue id (`tq_…`) anyway. The run itself was readable through `get_probe_training_run` all along.

Now:
* `GET /api/v1/task-queue/lookup/{task_id}` finds a Celery id wherever a row records it: the task queue's
  `task_id` column, and every mapped table with a `celery_task_id` column (DISCOVERED from the metadata, never
  listed), answering each row's table, id and status;
* `get_task_status(task_id=…)` reads it: a task-queue row comes back as the task-queue record; a job row comes
  back as a POINTER — the table, the row id, its status and a `next` call of the tool that reads that row
  (validated against the live registry); nothing found is `status: "not_found"` with what to try, never a list;
* `get_task_status(task_queue_id=…)` with an id the queue does not know tries the same lookup before saying
  not found (agents pass the Celery id there too).

Through the real routes over the real database, and the real tool on the built server with its client on the
real app. Wiring controls (recorded in the 042 controls file): drop the route's table discovery → the job-row
case is red; drop the tool's `task_id` branch → red.
"""
from __future__ import annotations

import uuid

import httpx
import pytest

from tests.support.next_calls import assert_next_calls_are_live

pytestmark = pytest.mark.asyncio
CELERY_ID = "959734fc-1d2e-4f50-9a6b-0c1d2e3f4a5b"


async def _run_with_celery_id(session, celery_id: str = CELERY_ID) -> str:
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="llama", status=ModelStatus.READY, file_path="/x",
                  architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="ssn", source="local", status=DatasetStatus.READY, raw_path="/x",
                      extra_metadata={})
    session.add_all([model, dataset])
    await session.flush()
    view = ProbeMonitorDataset(name="train", dataset_id=dataset.id, input_column="text", label_column="label",
                               label_mapping={"1": "positive"}, role="train", counts={})
    session.add(view)
    await session.flush()
    run = ProbeMonitorRun(id=f"pmr_{uuid.uuid4().hex[:12]}", model_id=model.id, train_dataset_id=view.id,
                          eval_dataset_ids=[], config={}, status="queued", celery_task_id=celery_id)
    session.add(run)
    await session.commit()
    return run.id


def _tool(client):
    """`call(arguments)`: `get_task_status` on the built server, its backend client on the real app."""
    from src.main import app
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, studio = build_server(MCPSettings(tool_categories="read,jobs", allow_anonymous=True, _env_file=None),
                               stdio=True)
    studio._http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),  # noqa: SLF001
                                     base_url="http://testserver/api/v1")

    async def call(arguments):
        result = await mcp._tool_manager.call_tool("get_task_status", arguments, context=None)  # noqa: SLF001
        return result

    return call


async def test_the_route_finds_a_celery_id_on_a_job_row(client, async_session):
    run_id = await _run_with_celery_id(async_session)
    body = (await client.get(f"/api/v1/task-queue/lookup/{CELERY_ID}")).json()
    assert body["task_id"] == CELERY_ID
    assert {"table": "probe_monitor_runs", "row_id": run_id, "status": "queued"} in body["found"]
    assert body["searched"] and "probe_monitor_runs" in body["searched"] and "task_queue" in body["searched"]
    # Discovered, not listed: every mapped table with a celery_task_id column is searched.
    from src.core.database import Base
    import src.models  # noqa: F401

    discovered = {t.name for t in Base.metadata.sorted_tables if "celery_task_id" in t.c and "id" in t.c}
    assert len(discovered) >= 10 and discovered <= set(body["searched"])


async def test_the_route_answers_an_unknown_id_with_an_empty_found(client, async_session):
    body = (await client.get("/api/v1/task-queue/lookup/not-a-task")).json()
    assert body["found"] == [] and body["task_id"] == "not-a-task"


async def test_get_task_status_task_id_points_at_the_run_reader(client, async_session):
    run_id = await _run_with_celery_id(async_session)
    call = _tool(client)
    out = await call({"task_id": CELERY_ID})
    assert out["status"] == "queued" and out["found_in"] == "probe_monitor_runs" and out["row_id"] == run_id
    assert out["next"] == [{"tool": "get_probe_training_run", "arguments": {"run_id": run_id},
                            "why": "probe_monitor_runs row of this task: read it with get_probe_training_run"}]
    assert_next_calls_are_live(out["next"])


async def test_a_celery_id_passed_as_task_queue_id_is_resolved_too(client, async_session):
    run_id = await _run_with_celery_id(async_session)
    out = await _tool(client)({"task_queue_id": CELERY_ID})
    assert out["found_in"] == "probe_monitor_runs" and out["row_id"] == run_id


async def test_nothing_found_is_not_a_list(client, async_session):
    out = await _tool(client)({"task_id": "no-such-task"})
    assert out["status"] == "not_found" and "list_active_jobs" in out["next"]
    out = await _tool(client)({"task_queue_id": "tq_nosuch"})
    assert out["status"] == "not_found"


async def test_a_task_queue_row_comes_back_as_its_record(client, async_session):
    from src.models.task_queue import TaskQueue

    row = TaskQueue(id=f"tq_{uuid.uuid4().hex[:12]}", task_id=CELERY_ID, task_type="download",
                    entity_id="m_x", entity_type="model", status="running", progress=5.0)
    async_session.add(row)
    await async_session.commit()
    out = await _tool(client)({"task_id": CELERY_ID})
    assert out["data"]["id"] == row.id and out["data"]["status"] == "running"


def test_every_reader_named_by_the_tool_is_live():
    from src.mcp_server.tools.jobs import TASK_ROW_READERS

    assert TASK_ROW_READERS, "no reader mapped"
    assert_next_calls_are_live([{"tool": tool, "arguments": {arg: "x_1"}, "why": table}
                                for table, (tool, arg) in TASK_ROW_READERS.items()])
