"""G9/G10 MCP half (2026-10-21): `get_gpu_status`, `list_active_jobs`, `gpu_preflight` and the
unconditional `get_approval_status`, through the LIVE registry of the built server.

* Reachability: the tools are on `build_server()`'s `list_tools()` under the DEFAULT categories
  and the deployed manifest's categories — with steering approval OFF (the deployed setting),
  which is exactly where `get_approval_status` used to be missing.
* Payload + call count: each tool issues exactly the documented calls with exactly the documented
  arguments (a recording client; the preflight's sibling reads go over `httpx.MockTransport`
  through the REAL MiLLMClient and the real miForge path).
* Standalone: with no MILLM_API_URL and no MIFORGE_API_URL the preflight answers
  `configured: false` for each and still gives a verdict.

Controls (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`, restored from saved bytes):
  C-REG     drop `gpu_ops` from CATEGORY_MODULES["jobs"] → the registry tests red
  C-APPREG  drop `approvals` from CATEGORY_MODULES["read"] → test_get_approval_status_is_on_… red
  C-MILLM   make `resident_on_this_card` always False → test_a_resident_millm_model_makes_it_busy red
  C-FORGE   skip the miForge runs in `verdict` → test_miforge_runs_make_it_uncertain red
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

CARD = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
OTHER = "GPU-f47ba814-49a2-603f-3595-275284140251"


def _built(categories: str, monkeypatch, **settings) -> set[str]:
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    for var in ("MILLM_API_URL", "MIDATAWORKS_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    mcp, _ = build_server(MCPSettings(tool_categories=categories, allow_anonymous=True,
                                      **settings), stdio=True)
    return {t.name for t in asyncio.run(mcp.list_tools())}


NEW = {"get_gpu_status", "list_active_jobs", "gpu_preflight", "get_approval_status"}


def test_the_new_tools_are_on_the_DEFAULT_build(monkeypatch):
    assert NEW <= _built("read,groups,steering,labeling,experiments,profiles,circuits,jlens,"
                         "jobs,models,probe_monitors", monkeypatch)


def test_the_new_tools_are_on_the_DEPLOYED_build(monkeypatch):
    from tests.unit.test_reachability import TestTheDEPLOYMENTEnablesWhatTheCodeRegisters as T

    deployed = ",".join(sorted(T()._deployed_categories()))
    assert NEW <= _built(deployed, monkeypatch, steering_approval=False)


def test_get_approval_status_is_on_a_read_only_build_with_approval_off(monkeypatch):
    names = _built("read", monkeypatch, steering_approval=False)
    assert "get_approval_status" in names
    assert "steer_sweep" not in names


def test_get_approval_status_is_registered_ONCE_with_steering_approval_on(monkeypatch):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(tool_categories="read,steering", allow_anonymous=True,
                                      steering_approval=True), stdio=True)
    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert names.count("get_approval_status") == 1


def test_the_gpu_tools_are_in_jobs_and_absent_without_it(monkeypatch):
    assert {"get_gpu_status", "list_active_jobs", "gpu_preflight"} <= _built("jobs", monkeypatch)
    assert not ({"get_gpu_status", "gpu_preflight"} & _built("read", monkeypatch))


# ── payload + call count ────────────────────────────────────────────────────────────────────


class Recorder:
    def __init__(self, answers=None):
        self.calls: list[tuple[str, str, dict]] = []
        self.answers = answers or {}

    async def get(self, path, **params):
        self.calls.append(("GET", path, params))
        answer = self.answers.get(path, {})
        if isinstance(answer, Exception):
            raise answer
        return answer


def _tools(client, **settings):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import approvals, gpu_ops

    mcp = MCPServer("t")
    s = MCPSettings(auth_token="x" * 32, _env_file=None, **settings)
    gpu_ops.register(mcp, client, s)
    approvals.register(mcp, client, s)
    return mcp._tool_manager  # noqa: SLF001


def _call(manager, name, **args):
    return asyncio.run(manager.call_tool(name, args, context=None))


def test_get_gpu_status_payload():
    rec = Recorder()
    _call(_tools(rec), "get_gpu_status", gpu=CARD)
    assert rec.calls == [("GET", "/operations/gpu-status", {"gpu": CARD})]
    rec = Recorder()
    _call(_tools(rec), "get_gpu_status")
    assert rec.calls == [("GET", "/operations/gpu-status", {"gpu": None})]


def test_list_active_jobs_payload():
    rec = Recorder()
    _call(_tools(rec), "list_active_jobs", kind="probe_training", gpu=CARD)
    assert rec.calls == [("GET", "/operations/active-jobs", {"kind": "probe_training", "gpu": CARD})]


def _studio(card_in_use=False, jobs=()):
    card = {"uuid": CARD, "in_use": card_in_use,
            "in_use_by": ["miStudio lease held by training:t1"] if card_in_use else []}
    return {
        "/operations/gpu-status": {"cards": [card], "reason": None},
        "/operations/active-jobs": {"jobs": list(jobs), "errors": [], "read_at": "now"},
    }


def test_preflight_STANDALONE_is_clear_with_both_siblings_unconfigured(monkeypatch):
    for var in ("MILLM_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    rec = Recorder(_studio())
    out = _call(_tools(rec), "gpu_preflight", gpu=CARD)
    assert [c[:2] for c in rec.calls] == [("GET", "/operations/gpu-status"),
                                         ("GET", "/operations/active-jobs")]
    assert rec.calls[0][2] == {"gpu": CARD} and rec.calls[1][2] == {}
    assert out["verdict"] == "clear", out
    assert out["millm"]["configured"] is False and out["miforge"]["configured"] is False
    assert out["busy_reasons"] == [] and out["uncertain_reasons"] == []


def test_preflight_refuses_an_index():
    rec = Recorder(_studio())
    out = _call(_tools(rec), "gpu_preflight", gpu="0000")
    assert out["refused"] is True and rec.calls == []


def test_a_leased_card_is_busy(monkeypatch):
    for var in ("MILLM_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    out = _call(_tools(Recorder(_studio(card_in_use=True))), "gpu_preflight", gpu=CARD)
    assert out["verdict"] == "busy"
    assert "lease" in out["busy_reasons"][0]


def test_a_queued_job_with_no_card_makes_it_uncertain(monkeypatch):
    for var in ("MILLM_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    queued = {"kind": "probe_training", "id": "pmr_1", "status": "pending", "gpu_uuids": []}
    placed = {"kind": "sae_training", "id": "t2", "status": "queued", "gpu_uuids": [OTHER]}
    out = _call(_tools(Recorder(_studio(jobs=[queued, placed]))), "gpu_preflight", gpu=CARD)
    assert out["verdict"] == "uncertain"
    assert "pmr_1" in out["uncertain_reasons"][0] and "t2" not in out["uncertain_reasons"][0]


def test_a_backend_refusal_is_returned_not_raised(monkeypatch):
    from src.mcp_server.client import BackendError

    rec = Recorder({"/operations/gpu-status": BackendError(404, {"detail": "No GPU"})})
    out = _call(_tools(rec), "gpu_preflight", gpu=CARD)
    assert out["refused"] is True and out["status_code"] == 404


# ── siblings over the real clients ──────────────────────────────────────────────────────────


def _sibling_transports(monkeypatch, millm_handler=None, forge_handler=None):
    from src.mcp_server.millm_client import MiLLMClient
    from src.mcp_server.tools import gpu_ops

    seen = {"millm": [], "forge": []}

    def millm_factory(url):
        client = MiLLMClient(url)

        def handler(request):
            seen["millm"].append((request.method, request.url.path))
            return millm_handler(request)

        client._http = httpx.AsyncClient(base_url=url, transport=httpx.MockTransport(handler))  # noqa: SLF001
        return client

    def forge_factory(url):
        def handler(request):
            seen["forge"].append((request.method, request.url.path, dict(request.url.params)))
            return forge_handler(request)

        return httpx.AsyncClient(base_url=url, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(gpu_ops, "_millm_client", millm_factory)
    monkeypatch.setattr(gpu_ops, "_forge_client", forge_factory)
    return seen


def _millm(resident_on=None, model="llama"):
    def handler(request):
        if request.url.path == "/api/health/detailed":
            return httpx.Response(200, json={"model_loaded": bool(resident_on), "model_name": model,
                                             "lease": None, "model_placement": None})
        cards = [{"smi_index": 0, "uuid": CARD.lower(), "torch_measured": resident_on == CARD,
                  "millm_allocated_mb": 15000 if resident_on == CARD else None,
                  "engine_memory": None}]
        return httpx.Response(200, json={"read_at": "x", "cards": cards})
    return handler


def test_a_resident_millm_model_makes_it_busy(monkeypatch):
    monkeypatch.delenv("MIFORGE_API_URL", raising=False)
    seen = _sibling_transports(monkeypatch, millm_handler=_millm(resident_on=CARD))
    out = _call(_tools(Recorder(_studio()), millm_api_url_override="http://millm.test"),
                "gpu_preflight", gpu=CARD)
    assert seen["millm"] == [("GET", "/api/health/detailed"), ("GET", "/api/health/gpus")]
    assert out["millm"]["resident_on_this_card"] is True
    assert out["verdict"] == "busy" and "miLLM" in out["busy_reasons"][0]


def test_a_millm_model_on_ANOTHER_card_leaves_this_one_clear(monkeypatch):
    monkeypatch.delenv("MIFORGE_API_URL", raising=False)
    _sibling_transports(monkeypatch, millm_handler=_millm(resident_on=OTHER))
    out = _call(_tools(Recorder(_studio()), millm_api_url_override="http://millm.test"),
                "gpu_preflight", gpu=CARD)
    assert out["verdict"] == "clear", out


def test_a_configured_millm_that_is_down_makes_it_uncertain_not_clear(monkeypatch):
    monkeypatch.delenv("MIFORGE_API_URL", raising=False)

    def down(request):
        raise httpx.ConnectError("refused")

    _sibling_transports(monkeypatch, millm_handler=down)
    out = _call(_tools(Recorder(_studio()), millm_api_url_override="http://millm.test"),
                "gpu_preflight", gpu=CARD)
    assert out["millm"] == {**out["millm"], "configured": True, "reachable": False}
    assert out["verdict"] == "uncertain"


def _forge(running=(), pending=(), body=None):
    def handler(request):
        if body is not None:
            return httpx.Response(200, content=body)
        status = request.url.params["status"]
        rows = running if status == "running" else pending
        return httpx.Response(200, json={"runs": [{"run_id": r, "status": status} for r in rows],
                                         "total": len(rows)})
    return handler


def test_miforge_runs_make_it_uncertain(monkeypatch):
    monkeypatch.delenv("MILLM_API_URL", raising=False)
    seen = _sibling_transports(monkeypatch, forge_handler=_forge(running=["run_a"],
                                                                 pending=["run_b"]))
    out = _call(_tools(Recorder(_studio()), miforge_api_url_override="http://forge.test"),
                "gpu_preflight", gpu=CARD)
    assert [(m, p, q["status"]) for m, p, q in seen["forge"]] == [
        ("GET", "/api/v1/runs", "running"), ("GET", "/api/v1/runs", "pending")]
    assert out["miforge"]["active_run_count"] == 2
    assert out["verdict"] == "uncertain"


def test_miforge_idle_is_clear(monkeypatch):
    monkeypatch.delenv("MILLM_API_URL", raising=False)
    _sibling_transports(monkeypatch, forge_handler=_forge())
    out = _call(_tools(Recorder(_studio()), miforge_api_url_override="http://forge.test"),
                "gpu_preflight", gpu=CARD)
    assert out["verdict"] == "clear" and out["miforge"]["active_run_count"] == 0


def test_a_miforge_html_page_is_NOT_an_empty_run_list(monkeypatch):
    monkeypatch.delenv("MILLM_API_URL", raising=False)
    _sibling_transports(monkeypatch, forge_handler=_forge(body=b"<html>splash</html>"))
    out = _call(_tools(Recorder(_studio()), miforge_api_url_override="http://forge.test"),
                "gpu_preflight", gpu=CARD)
    assert out["miforge"]["reachable"] is False and out["verdict"] == "uncertain"


def test_the_preflight_reads_the_env_urls(monkeypatch):
    monkeypatch.setenv("MILLM_API_URL", "http://millm.env")
    monkeypatch.setenv("MIFORGE_API_URL", "http://forge.env/")
    from src.mcp_server.config import MCPSettings

    s = MCPSettings(auth_token="x" * 32, _env_file=None)
    assert s.millm_api_url == "http://millm.env" and s.miforge_api_url == "http://forge.env"


# ── get_approval_status ─────────────────────────────────────────────────────────────────────


ROW = {"id": "a1", "tool_name": "start_probe_training_run", "payload": {"model_id": "m"},
       "status": "approved", "reason": None, "steering_task_id": None,
       "created_at": "2026-10-20T21:47:24Z", "resolved_at": "2026-10-20T21:58:00Z",
       "consumed_at": None}


def test_get_approval_status_payload_and_shape():
    rec = Recorder({"/mcp/approvals/a1": ROW})
    out = _call(_tools(rec), "get_approval_status", approval_request_id="a1")
    assert rec.calls == [("GET", "/mcp/approvals/a1", {})]
    assert out["status"] == "approved" and out["decided"] is True
    assert out["decided_at"] == ROW["resolved_at"] and out["decided_by"].startswith("operator")
    assert "Re-call" in out["next"]


@pytest.mark.parametrize("status,reason,who", [
    ("pending", None, None),
    ("denied", "too big", "operator"),
    ("consumed", None, "operator"),
    ("expired", "expired: pending for more than 24 h", "expiry"),
    ("expired", "expired: approved but not used within 24 h", "operator"),
])
def test_who_decided_per_status(status, reason, who):
    from src.mcp_server.tools.approvals import describe

    out = describe({**ROW, "status": status, "reason": reason})
    assert out["decided"] is (status != "pending")
    if who is None:
        assert out["decided_by"] is None
    else:
        assert out["decided_by"].startswith(who)
    assert out["next"]


def test_an_unknown_approval_is_not_found_not_an_exception():
    from src.mcp_server.client import BackendError

    rec = Recorder({"/mcp/approvals/zz": BackendError(404, "nope")})
    out = _call(_tools(rec), "get_approval_status", approval_request_id="zz")
    assert out["status"] == "not_found"


async def test_get_approval_status_against_the_REAL_route_never_consumes(async_session, client):
    """Through the real approvals route: create, approve, read with the tool — and the row is
    still `approved` afterwards (reading never spends it)."""
    from mcp.server.mcpserver import MCPServer

    from src.main import app
    from src.mcp_server.client import MiStudioClient
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import approvals

    created = (await client.post("/api/v1/mcp/approvals", json={
        "tool_name": "start_probe_training_run", "payload": {"model_id": "m_x"}})).json()
    real = MiStudioClient("http://test")
    real._http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),  # noqa: SLF001
                                   base_url="http://test/api/v1")
    mcp = MCPServer("t")
    approvals.register(mcp, real, MCPSettings(auth_token="x" * 32, _env_file=None))
    tool = mcp._tool_manager._tools["get_approval_status"]  # noqa: SLF001

    pending = await tool.fn(approval_request_id=created["id"])
    assert pending["status"] == "pending" and pending["decided"] is False
    await client.post(f"/api/v1/mcp/approvals/{created['id']}/approve")
    approved = await tool.fn(approval_request_id=created["id"])
    again = await tool.fn(approval_request_id=created["id"])
    assert approved["status"] == again["status"] == "approved"
    assert approved["decided_at"] and approved["consumed_at"] is None
    row = (await client.get(f"/api/v1/mcp/approvals/{created['id']}")).json()
    assert row["status"] == "approved", "reading must never consume"
    await real.close()
