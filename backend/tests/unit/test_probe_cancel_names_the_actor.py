"""A cancel records who asked (ENH-006 D116 G9).

`cancel_probe_training_run` went through the route with `request_cancel`'s default reason, so an agent's
cancel was stored on the row as "cancelled by operator". The operator then reads a statement about
themselves that is false. The route now reads `X-MiStudio-Agent`, as the view archive routes do.

Through the real route and the real `request_cancel`, and once from the tool on the built server, so
the header the row depends on is the one the MCP client really sends.

Controls are recorded in `0xcc/reviews/mcp_gaps_controls_2026-11-02.md`.
"""
from __future__ import annotations

import uuid

import httpx
import pytest
from mcp import Client

from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server
from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus
from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorJudgeRun, ProbeMonitorRun


@pytest.fixture
async def run_id(async_session):
    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY, file_path="/nowhere",
                  architecture="LlamaForCausalLM", params_count=1_000)
    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="local", status=DatasetStatus.READY, raw_path="/nowhere",
                      extra_metadata={})
    async_session.add_all([model, dataset])
    await async_session.commit()
    view = ProbeMonitorDataset(name="train", dataset_id=dataset.id, input_column="t", label_column="l",
                               label_mapping={"1": "positive", "0": "negative"}, role="train", counts={})
    async_session.add(view)
    await async_session.commit()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[], config={},
                          status="running", environment={})
    async_session.add(run)
    await async_session.commit()
    return run.id


@pytest.fixture
async def judge_run_id(async_session):
    run = ProbeMonitorJudgeRun(endpoint="http://judge", model="m", prompt_version="v1", dataset_ids=[], status="running")
    async_session.add(run)
    await async_session.commit()
    return run.id


async def _recorded(async_session, model, row_id):
    async_session.expire_all()
    return (await async_session.get(model, row_id)).error_message


class TestTheRecord:
    @pytest.mark.asyncio
    async def test_an_operators_cancel_keeps_the_existing_text(self, client, async_session, run_id):
        response = await client.post(f"/api/v1/probe-monitors/runs/{run_id}/cancel")
        assert response.status_code == 202, response.text
        assert await _recorded(async_session, ProbeMonitorRun, run_id) == "cancelled by operator"

    @pytest.mark.asyncio
    async def test_an_agents_cancel_names_the_agent(self, client, async_session, run_id):
        response = await client.post(f"/api/v1/probe-monitors/runs/{run_id}/cancel",
                                     headers={"X-MiStudio-Agent": "mcp"})
        assert response.status_code == 202, response.text
        assert await _recorded(async_session, ProbeMonitorRun, run_id) == "cancelled by agent:mcp"

    @pytest.mark.asyncio
    async def test_a_malformed_header_is_refused_and_cancels_nothing(self, client, async_session, run_id):
        """Reading it as the operator is the one wrong answer (`lifecycle.actor_of`)."""
        response = await client.post(f"/api/v1/probe-monitors/runs/{run_id}/cancel",
                                     headers={"X-MiStudio-Agent": "NOT A NAME!"})
        assert response.status_code == 400
        assert await _recorded(async_session, ProbeMonitorRun, run_id) is None

    @pytest.mark.asyncio
    async def test_the_tool_through_the_real_client_is_recorded_as_the_agent(self, client, async_session, run_id):
        mcp, studio = build_server(MCPSettings(tool_categories="probe_monitors", allow_anonymous=True, _env_file=None),
                                   stdio=True)
        studio._http = httpx.AsyncClient(  # noqa: SLF001
            transport=client._transport, base_url="http://test/api/v1",  # noqa: SLF001
            headers=dict(studio._http.headers),  # noqa: SLF001 - the header the client really sends
        )
        async with Client(mcp) as agent:
            result = await agent.call_tool("cancel_probe_training_run", {"run_id": run_id})
        assert not result.is_error, result.content
        assert await _recorded(async_session, ProbeMonitorRun, run_id) == "cancelled by agent:mcp"


class TestTheJudgeRouteBesideIt:
    @pytest.mark.asyncio
    async def test_operator(self, client, async_session, judge_run_id):
        response = await client.post(f"/api/v1/probe-monitors/judge-runs/{judge_run_id}/cancel")
        assert response.status_code == 202, response.text
        assert await _recorded(async_session, ProbeMonitorJudgeRun, judge_run_id) == "cancelled by operator"

    @pytest.mark.asyncio
    async def test_agent(self, client, async_session, judge_run_id):
        response = await client.post(f"/api/v1/probe-monitors/judge-runs/{judge_run_id}/cancel",
                                     headers={"X-MiStudio-Agent": "mcp"})
        assert response.status_code == 202, response.text
        assert await _recorded(async_session, ProbeMonitorJudgeRun, judge_run_id) == "cancelled by agent:mcp"
