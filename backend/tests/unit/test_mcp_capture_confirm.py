"""Circuit-capture confirm acts on the ESTIMATED run (live GPU checks, 2026-10-20 22:11-22:13 ET).

Live: `start_circuit_capture(confirm=false)` made `cap_4ac3dc9e5eed` (`estimated`); the follow-up
`start_circuit_capture(confirm=true)` — what the tool text told the agent to do — made a NEW run
`cap_27876e548ac5` and left the estimate orphaned. The REST route that confirms THE SAME run
(`POST /circuit-capture/{id}/confirm`) existed and no tool called it.

Chosen fix: a `confirm_circuit_capture(run_id, gpu=None)` tool on that route, and the text of
`start_circuit_capture` says confirm=true there is a one-step NEW run; an estimate's answer names
the next call. Chosen over "match by identical arguments" because a match would silently pick one
of several identical estimates, and over making confirm=true reuse a row because that changes a
REST contract the UI already relies on.

Controls (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`, restored from saved bytes):
  C-CONFIRM  point confirm_circuit_capture at POST /circuit-capture → payload test and the
             real-route test red
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from tests.unit.test_circuits_run_on_the_chosen_gpu import (
    CARDS,
    RTX_UUID,
    TI_UUID,
    _capture,
    _sync_bridge,
)


class Recorder:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer if answer is not None else {}

    async def post(self, path, json_body=None, **params):
        self.calls.append(("POST", path, json_body))
        return self.answer


def _tools(client):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import circuits

    mcp = MCPServer("t")
    circuits.register(mcp, client, MCPSettings(auth_token="x" * 32, _env_file=None))
    return mcp._tool_manager  # noqa: SLF001


def test_confirm_posts_to_THE_RUNS_confirm_route_once():
    rec = Recorder({"id": "cap_1", "status": "queued"})
    asyncio.run(_tools(rec).call_tool("confirm_circuit_capture", {"run_id": "cap_1"}, context=None))
    assert rec.calls == [("POST", "/circuit-capture/cap_1/confirm", None)]


def test_a_card_named_at_confirm_is_sent():
    rec = Recorder()
    asyncio.run(_tools(rec).call_tool("confirm_circuit_capture",
                                      {"run_id": "cap_1", "gpu": RTX_UUID}, context=None))
    assert rec.calls == [("POST", "/circuit-capture/cap_1/confirm", {"gpu": RTX_UUID})]


def test_an_estimate_names_the_confirm_call_for_its_own_run():
    rec = Recorder({"id": "cap_est", "task_id": "t", "status": "queued", "confirmed": False})
    out = asyncio.run(_tools(rec).call_tool("start_circuit_capture", {
        "dataset_id": "d", "layers": [{"layer": 1, "sae_id": "s"}], "confirm": False}, context=None))
    assert len(rec.calls) == 1 and rec.calls[0][1] == "/circuit-capture"
    assert "confirm_circuit_capture(run_id='cap_est')" in out["next"]


def test_a_one_step_capture_gets_no_confirm_hint():
    rec = Recorder({"id": "cap_x", "status": "queued", "confirmed": True})
    out = asyncio.run(_tools(rec).call_tool("start_circuit_capture", {
        "dataset_id": "d", "layers": [], "confirm": True}, context=None))
    assert "next" not in out


def test_the_start_description_no_longer_says_re_call_with_confirm_true():
    tools = {t.name: t for t in _tools(Recorder()).list_tools()}
    text = tools["start_circuit_capture"].description
    assert "confirm_circuit_capture" in text and "creates a NEW run" in text
    assert "there is no separate confirm" not in text


async def test_confirm_moves_THE_ESTIMATED_ROW_through_the_real_route(monkeypatch):
    """Tool → real MiStudioClient → the real FastAPI route: the same row goes to `pending`
    and the full capture is dispatched for THAT id — no new row."""
    from src.core.database import get_db
    from src.main import app
    from src.mcp_server.client import MiStudioClient
    from src.services import gpu_placement

    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)
    run = _capture(gpu_request=TI_UUID)
    session = AsyncMock()

    async def _override():
        yield session

    app.dependency_overrides[get_db] = _override
    real = MiStudioClient("http://test")
    real._http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),  # noqa: SLF001
                                   base_url="http://test/api/v1")
    try:
        with patch("src.api.v1.endpoints.circuit_discovery._capture_or_404",
                   new=AsyncMock(return_value=run)), \
             patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                   new=_sync_bridge(run)), \
             patch("src.workers.circuit_capture_tasks.capture_circuit_activations.delay",
                   return_value=MagicMock(id="task_cc")) as delay:
            tools = _tools(real)
            out = await tools.call_tool("confirm_circuit_capture", {"run_id": "cap_test1"}, context=None)
    finally:
        app.dependency_overrides.pop(get_db, None)
        await real.close()
    assert out == {"id": "cap_test1", "task_id": "task_cc", "status": "queued"}
    assert run.status == "pending" and run.gpu_request == TI_UUID
    delay.assert_called_once_with("cap_test1", confirmed=True)
