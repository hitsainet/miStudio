"""Every agent-reachable cancel records who asked, not "cancelled by operator" (ENH-006 D116 G9 debt).

Each route, through the real app, with `request_cancel` recording its arguments: the reason it was given and
how many times. An operator keeps the route's existing text. A malformed header is a 400 that cancels nothing.
A row and its stored field are exercised end to end for a probe run in `test_probe_cancel_names_the_actor.py`
and for a dataset below. The six the debt named have FK-heavy rows, so their row read is stubbed: what is under
test is the line passing the reason. Controls: `0xcc/reviews/mcp_gaps_controls_2026-11-02.md`.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from mcp import Client
from sqlalchemy import select

from src.core.cancellation import CancelOutcome
from src.core.deps import get_db
from src.main import app
from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server
from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus
from src.models.task_queue import TaskQueue

ROW = SimpleNamespace(id="row-1", status="running", celery_task_id="t-1", task_id="t-1", faithfulness_status="running",
                      calibration_status="running", faithfulness_task_id="t-1", calibration_task_id="t-1")

ROUTES = [
    pytest.param("POST", "/feature-groups/runs/row-1/cancel", id="feature grouping"),
    pytest.param("POST", "/circuits/row-1/faithfulness/cancel", id="circuit faithfulness"),
    pytest.param("POST", "/circuits/row-1/calibration/cancel", id="circuit calibration"),
    pytest.param("POST", "/circuits/steering-samples/row-1/cancel", id="steering record"),
    pytest.param("POST", "/enhanced-labeling/row-1/cancel", id="enhanced labeling"),
    pytest.param("POST", "/probe-monitors/spot-checks/row-1/cancel", id="spot check"),
    pytest.param("POST", "/probe-monitors/blend-reports/row-1/cancel", id="blend"),
]


@pytest.fixture
async def stubbed(monkeypatch):
    """The real app with the row read stubbed and `request_cancel` recording."""
    calls = []

    def recording(kind, target, *, reason="(the default)", celery_task_id=None, db=None):
        calls.append((kind, target, reason))
        return CancelOutcome(True, True, "running", "ok")

    monkeypatch.setattr("src.core.cancellation.request_cancel", recording)

    result = MagicMock()
    result.scalar_one_or_none.return_value = ROW
    session = MagicMock(execute=AsyncMock(return_value=result), get=AsyncMock(return_value=ROW),
                        refresh=AsyncMock(), commit=AsyncMock())

    async def stub_db():
        yield session

    app.dependency_overrides[get_db] = stub_db
    monkeypatch.setattr("src.api.v1.endpoints.circuits._get_or_404", AsyncMock(return_value=ROW))
    monkeypatch.setattr("src.api.v1.endpoints.probe_spot_checks._get", AsyncMock(return_value=ROW))
    monkeypatch.setattr("src.services.probe_spot_check.ACTIVE", ("running",), raising=False)
    monkeypatch.setattr("src.services.probe_blend.service.get_row", AsyncMock(return_value=ROW))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, calls
    app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", ROUTES)
class TestTheRoutes:
    async def test_an_agent_is_named_once(self, stubbed, method, path):
        client, calls = stubbed
        response = await client.request(method, f"/api/v1{path}", headers={"X-MiStudio-Agent": "mcp"})
        assert response.status_code < 300, response.text
        assert [c[2] for c in calls] == ["cancelled by agent:mcp"]

    async def test_the_operator_keeps_the_text(self, stubbed, method, path):
        client, calls = stubbed
        response = await client.request(method, f"/api/v1{path}")
        assert response.status_code < 300, response.text
        assert [c[2] for c in calls] == ["cancelled by operator"]

    async def test_a_malformed_header_cancels_nothing(self, stubbed, method, path):
        client, calls = stubbed
        response = await client.request(method, f"/api/v1{path}", headers={"X-MiStudio-Agent": "NOT A NAME!"})
        assert response.status_code == 400, response.text
        assert calls == []


class TestJlensAndModelDownload:
    """Agent tools with the same defect, through real rows."""

    @pytest.fixture
    async def queued_task(self, async_session):
        task_id = str(uuid.uuid4())
        async_session.add(TaskQueue(id=f"tq_{uuid.uuid4().hex[:8]}", task_id=task_id, task_type="jlens_fit",
                                    entity_id="m", entity_type="model", status="running", retry_params={}, retry_count=0))
        await async_session.commit()
        return task_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize("headers,expected", [({"X-MiStudio-Agent": "mcp"}, "cancelled by agent:mcp"),
                                                  ({}, "cancelled by operator")])
    async def test_a_jlens_task(self, client, async_session, queued_task, monkeypatch, headers, expected):
        monkeypatch.setattr("src.core.celery_app.celery_app.control.revoke", MagicMock())
        response = await client.post(f"/api/v1/jlens/tasks/{queued_task}/cancel", headers=headers)
        assert response.status_code == 200, response.text
        async_session.expire_all()
        row = (await async_session.execute(select(TaskQueue).where(TaskQueue.task_id == queued_task))).scalar_one()
        assert row.error_message == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("headers,expected", [({"X-MiStudio-Agent": "mcp"}, "cancelled by agent:mcp"),
                                                  ({}, "Cancelled by user")])
    async def test_a_model_download(self, client, async_session, monkeypatch, headers, expected):
        model_id = f"m_{uuid.uuid4().hex[:12]}"
        async_session.add(Model(id=model_id, name="big", status=ModelStatus.DOWNLOADING, architecture="L", params_count=1))
        await async_session.commit()
        monkeypatch.setattr("src.workers.model_tasks.send_progress_update", MagicMock(), raising=False)
        response = await client.delete(f"/api/v1/models/{model_id}/cancel", headers=headers)
        assert response.status_code == 200, response.text
        async_session.expire_all()
        assert (await async_session.get(Model, model_id)).error_message == expected


class TestDatasetDownload:
    """The real route, the real worker body and a real row, from the tool on the built server."""

    @pytest.fixture
    async def downloading(self, async_session, tmp_path):
        row = Dataset(id=uuid.uuid4(), name="big", source="HuggingFace", status=DatasetStatus.DOWNLOADING,
                      extra_metadata={})
        async_session.add(row)
        await async_session.commit()
        return row

    @pytest.fixture
    def agent(self, client, monkeypatch):
        monkeypatch.setattr("src.workers.dataset_tasks.emit_dataset_progress", MagicMock())
        mcp, studio = build_server(MCPSettings(tool_categories="probe_monitors", allow_anonymous=True, _env_file=None),
                                   stdio=True)
        studio._http = httpx.AsyncClient(  # noqa: SLF001
            transport=client._transport, base_url="http://test/api/v1",  # noqa: SLF001
            headers=dict(studio._http.headers),  # noqa: SLF001 - the header the client really sends
        )

        async def call(dataset_id):
            async with Client(mcp) as session:
                return await session.call_tool("cancel_dataset_download", {"dataset_id": dataset_id})

        return call

    @pytest.mark.asyncio
    async def test_the_tool_stops_it_and_the_row_names_the_agent(self, agent, async_session, downloading):
        row_id = downloading.id
        result = await agent(str(row_id))
        assert not result.is_error, result.content
        async_session.expire_all()
        row = await async_session.get(Dataset, row_id)
        assert row.status == DatasetStatus.ERROR
        assert row.error_message == "cancelled by agent:mcp"
        assert row.cancel_requested_at is not None

    @pytest.mark.asyncio
    async def test_the_operator_keeps_the_text_the_ui_shows(self, client, async_session, downloading, monkeypatch):
        monkeypatch.setattr("src.workers.dataset_tasks.emit_dataset_progress", MagicMock())
        row_id = downloading.id
        response = await client.delete(f"/api/v1/datasets/{row_id}/cancel")
        assert response.status_code == 200, response.text
        async_session.expire_all()
        assert (await async_session.get(Dataset, row_id)).error_message == "Cancelled by user"

    @pytest.mark.asyncio
    async def test_a_dataset_not_downloading_is_refused_and_reaches_the_agent(self, agent, async_session, downloading):
        downloading.status = DatasetStatus.READY
        await async_session.commit()
        result = await agent(str(downloading.id))
        assert result.is_error and "backend 400" in result.content[0].text

    def test_the_tool_is_on_the_server_the_deployed_manifest_builds(self):
        import asyncio

        from tests.unit.test_reachability import TestTheDEPLOYMENTEnablesWhatTheCodeRegisters as T

        mcp, _ = build_server(MCPSettings(tool_categories=",".join(sorted(T()._deployed_categories())),
                                          allow_anonymous=True, _env_file=None), stdio=True)
        assert "cancel_dataset_download" in {t.name for t in asyncio.run(mcp.list_tools())}
