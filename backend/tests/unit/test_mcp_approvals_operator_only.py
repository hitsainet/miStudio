"""OSD-56 (040 Appendix C.6, 040-A5): an agent can never approve or deny an approval request.

`POST /mcp/approvals/{id}/approve` and `/deny` refuse any caller sending `X-MiStudio-Agent` (the header
every MCP request carries) with 403 `approval_operator_only`, and the row stays `pending`. Agents still
FILE requests (`POST /mcp/approvals`) and SPEND caller-executed ones (`/consume`) with the header.

One session per request (the `api` fixture of `test_mcp_approvals_consume.py`), parametrised over one
backend-executed (steering) and one caller-executed (`start_probe_training_run`; `millm_load_model` until ENH-008
moved miLLM's approvals into miLLM) tool.

Wiring controls (recorded in `0xcc/reviews/040_implementation_controls_2026-10-23.md`, S4 table, O1/O2):
delete `dependencies=[Depends(refuse_agent_decisions)]` from `approve_request`'s decorator → the approve
cases go red (200 `approved`); the same for `deny_request` → the deny cases red.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.v1.endpoints import mcp_approvals
from src.core.deps import get_db
from src.main import app
from src.mcp_server import client as mcp_client
from src.models.agent_approval import AgentApprovalRequest

LOAD = {"tool_name": "start_probe_training_run", "payload": {"model_id": 7, "train_dataset_id": "pmd_1"}}
STEER = {"tool_name": "steer_compare",
         "payload": {"sae_id": "s", "prompt": "hi", "selected_features": []}}
AGENT = {mcp_client.AGENT_HEADER: "mcp"}


@pytest.fixture
def events(monkeypatch):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_approvals, "emit_progress",
                        lambda channel, event, data: seen.append((event, data)))
    return seen


@pytest.fixture
def self_calls(monkeypatch):
    """Records any steering self-call instead of making it (a refused approve must make none)."""
    calls: list[tuple[str, dict]] = []

    class _Response:
        status_code = 200
        headers = {"content-type": "application/json"}

        @staticmethod
        def json():
            return {"task_id": "steer-task-1"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            calls.append((url, json))
            return _Response()

    monkeypatch.setattr(mcp_approvals.httpx, "AsyncClient", _Client)
    return calls


@pytest_asyncio.fixture
async def api(async_engine):
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def per_request_session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db] = per_request_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test/api/v1") as client:
        yield client, maker
    app.dependency_overrides.clear()


async def _status(maker, request_id: str) -> str:
    async with maker() as session:
        row = (await session.execute(
            select(AgentApprovalRequest).where(AgentApprovalRequest.id == request_id))).scalar_one()
        return row.status


def test_the_header_refused_is_the_one_the_mcp_client_sends():
    assert mcp_approvals.AGENT_HEADER == mcp_client.AGENT_HEADER == "X-MiStudio-Agent"


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [STEER, LOAD], ids=["backend_executed", "caller_executed"])
@pytest.mark.parametrize("decision", ["approve", "deny"])
async def test_an_agent_cannot_decide(api, events, self_calls, body, decision):
    client, maker = api
    # Agents FILE requests with the header: that must keep working.
    created = await client.post("/mcp/approvals", json=body, headers=AGENT)
    assert created.status_code == 201, created.text
    rid = created.json()["id"]
    events.clear()

    extra = {"json": {"reason": "agent says no"}} if decision == "deny" else {}
    refused = await client.post(f"/mcp/approvals/{rid}/{decision}", headers=AGENT, **extra)
    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"]["code"] == "approval_operator_only"
    assert await _status(maker, rid) == "pending"
    assert self_calls == [] and events == []

    # A malformed (empty) header is still an agent's header.
    empty = await client.post(f"/mcp/approvals/{rid}/{decision}", headers={mcp_client.AGENT_HEADER: ""}, **extra)
    assert empty.status_code == 403
    assert await _status(maker, rid) == "pending"

    # The operator (no header) decides.
    done = await client.post(f"/mcp/approvals/{rid}/{decision}", **extra)
    assert done.status_code == 200, done.text
    want = "approved" if decision == "approve" else "denied"
    assert done.json()["status"] == want
    assert await _status(maker, rid) == want
    assert [e for e, _ in events] == ["approval:resolved"]
    if decision == "approve" and body is STEER:
        assert len(self_calls) == 1  # the operator's approve submitted exactly once


@pytest.mark.asyncio
async def test_an_agent_still_consumes_a_caller_executed_approval(api, events, self_calls):
    client, maker = api
    rid = (await client.post("/mcp/approvals", json=LOAD, headers=AGENT)).json()["id"]
    assert (await client.post(f"/mcp/approvals/{rid}/approve")).status_code == 200
    spent = await client.post(f"/mcp/approvals/{rid}/consume", json=LOAD, headers=AGENT)
    assert spent.status_code == 200, spent.text
    assert spent.json()["status"] == "consumed"
