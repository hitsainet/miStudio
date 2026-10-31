"""miStudio's `millm_*` proxies RELAY miLLM's own approval answer (ENH-008 S2, S3; spec §3.2 steps 2-4).

miLLM decides its own requests (D1). The six tools that change what miLLM serves — `millm_load_model`,
`millm_unload_model`, `millm_delete_model`, `millm_request_probe_limits`, `millm_measure_probe_capacity` and
`millm_import_rule_set` for operator code — send the request to miLLM with `X-miLLM-Agent`; miLLM stores it and
answers 202 with `X-miLLM-Approval: pending`; the tool returns miLLM's body unchanged plus `"decide_in": "millm"`.

Everything here runs through the LIVE registry: `build_server` with the categories the deployed pod enables
(`k8s/base/mcp.yaml`) and a miLLM URL, the real `MiLLMClient` on an `httpx.MockTransport` standing in for miLLM, and
miStudio's own client on a recorder. So each assertion covers the registration, the client's header and the tool:

* S2: a pending answer is returned as miLLM sent it (+ `decide_in`), with exactly TWO miLLM requests — the
  fail-closed `GET /api/approvals/actions`, then the route with its exact body — and ZERO requests to miStudio's
  `/mcp/approvals`;
* S3: a miLLM without `/api/approvals/actions` (404), or one whose list lacks the action, gets the
  `millm_has_no_approval_queue` refusal and ZERO requests on the action route;
* the deprecated `approval_request_id` answers `approval_moved_to_millm` and sends nothing anywhere;
* the two new read tools issue exactly one request each, with the parameters given.

Mutation controls (0xcc/reviews/enh008_controls_2026-10-28.md): skip the queue check in one tool; make
`ApprovalQueueCheck.refusal` return None for a missing action; return the body without `relayed()`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

BACKEND = Path(__file__).resolve().parents[2]
MISTUDIO_URL = "http://mistudio.test"
MILLM_URL = "http://millm.test"
ALL_ACTIONS = ["model_load", "model_unload", "model_delete", "probe_limits_write", "capacity_measure",
               "code_rule_import"]
CODE_SHA = "c" * 64


def _manifest_categories() -> str:
    import yaml

    for doc in yaml.safe_load_all((BACKEND.parent / "k8s" / "base" / "mcp.yaml").read_text()):
        for container in (((doc or {}).get("spec") or {}).get("template") or {}).get("spec", {}).get("containers", []):
            for e in container.get("env", []):
                if e.get("name") == "MCP_TOOL_CATEGORIES":
                    return e["value"]
    raise AssertionError("no MCP_TOOL_CATEGORIES in k8s/base/mcp.yaml")


def _pending(action: str) -> dict[str, Any]:
    """miLLM's 202 body (spec §3.1, `ApiResponse.ok({...})`), unwrapped by the client."""
    return {"approval_id": f"apr_{action}", "status": "pending", "action": action,
            "summary": {"model": "m"}, "request_digest": "sha256:" + "a" * 64,
            "expires_at": "2026-10-29T12:00:00Z",
            "hint": "Waiting for the operator to approve this in miLLM's Approvals page."}


class _MiLLM:
    """miLLM with (or without) its approval queue. Records every request except the gate's health probe."""

    def __init__(self, actions: Any = "all", *, pending: bool = True):
        self.actions = ALL_ACTIONS if actions == "all" else actions
        self.pending = pending
        self.seen: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content) if request.content else None
        self.seen.append({"method": request.method, "path": path, "body": body,
                          "params": dict(request.url.params), "headers": dict(request.headers)})
        if path == "/api/approvals/actions":
            if self.actions is None:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json={"success": True, "data": {
                "actions": [{"action": a, "label": a} for a in self.actions]}, "error": None})
        if path.startswith("/api/approvals"):
            return httpx.Response(200, json={"success": True, "data": {"approvals": [], "count": 0, "pending": 0}})
        action = _ACTION_FOR.get((request.method, path))
        if action and self.pending:
            return httpx.Response(202, json={"success": True, "data": _pending(action), "error": None},
                                  headers={"X-miLLM-Approval": "pending"})
        return httpx.Response(200, json={"success": True, "data": {"id": 7, "status": "loading"}, "error": None})


_ACTION_FOR = {
    ("POST", "/api/models/7/load"): "model_load",
    ("POST", "/api/models/7/unload"): "model_unload",
    ("DELETE", "/api/models/7"): "model_delete",
    ("PUT", "/api/models/7/probe-limits"): "probe_limits_write",
    ("POST", "/api/models/7/probe-capacity/measure"): "capacity_measure",
    ("POST", "/api/rule-sets/import"): "code_rule_import",
}


def _code_rule_set() -> dict[str, Any]:
    return {"format": "midataworks.rule-set/v1", "name": "code", "behaviour_key": "<behaviour-key>",
            "signals": [{"id": "c", "kind": "code_rule", "code_sha256": CODE_SHA, "function": "f"}],
            "expressions": {"hit": {"signal": "c"}}, "outputs": [], "cases": []}


#: tool → (arguments, the ONE action request it must send: method, path, body, params).
RELAYED = {
    "millm_load_model": ({"model_id": 7}, ("POST", "/api/models/7/load", None, {})),
    "millm_unload_model": ({"model_id": 7}, ("POST", "/api/models/7/unload", None, {})),
    "millm_delete_model": ({"model_id": 7, "acknowledge_permanent": True}, ("DELETE", "/api/models/7", None, {})),
    "millm_request_probe_limits": (
        {"model_id": 7, "reason": "5% of the model's own time", "decode_budget": {"value": 5, "unit": "pct"}},
        ("PUT", "/api/models/7/probe-limits",
         {"reason": "5% of the model's own time", "set_by": "mistudio-agent", "source": "operator",
          "decode_budget": {"value": 5.0, "unit": "pct"}}, {})),
    "millm_measure_probe_capacity": ({"model_id": 7}, ("POST", "/api/models/7/probe-capacity/measure", None, {})),
    "millm_import_rule_set": ({"document": _code_rule_set(), "version": 2},
                              ("POST", "/api/rule-sets/import", _code_rule_set(), {"version": "2"})),
}


@pytest.fixture
def live(monkeypatch):
    """`build(millm)` → `(call, mistudio_seen)`: tool calls against the deployed-category server."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    def build(millm: _MiLLM):
        mistudio_seen: list[tuple[str, str]] = []

        def mistudio(request: httpx.Request) -> httpx.Response:
            mistudio_seen.append((request.method, request.url.path))
            return httpx.Response(200, json={"status": "ok"})

        real = httpx.AsyncClient

        class Routed(real):
            def __init__(self, *args, **kwargs):
                base = str(kwargs.get("base_url", ""))
                kwargs["transport"] = httpx.MockTransport(mistudio if base.startswith(MISTUDIO_URL) else millm)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", Routed)
        monkeypatch.setenv("MISTUDIO_API_URL", MISTUDIO_URL)
        monkeypatch.setenv("MILLM_API_URL", MILLM_URL)
        monkeypatch.setenv("MCP_TOOL_CATEGORIES", _manifest_categories())
        mcp, _ = build_server(MCPSettings(auth_token="x" * 32), stdio=True)

        def call(tool: str, args: dict) -> Any:
            return asyncio.run(mcp._tool_manager.call_tool(tool, args, context=None))  # noqa: SLF001

        return call, mistudio_seen

    return build


def _action_requests(millm: _MiLLM) -> list[dict]:
    return [r for r in millm.seen if not r["path"].startswith("/api/approvals")]


# ── S2: a pending answer is relayed, with two miLLM calls and none to miStudio's queue ─────────────────────────────


@pytest.mark.parametrize("tool", sorted(RELAYED))
def test_a_pending_answer_is_relayed_unchanged_with_decide_in(live, tool):
    args, (method, path, body, params) = RELAYED[tool]
    millm = _MiLLM()
    call, mistudio_seen = live(millm)

    result = call(tool, args)

    action = _ACTION_FOR[(method, path)]
    assert result == {**_pending(action), "decide_in": "millm"}
    # Exactly two miLLM requests, in this order: the fail-closed check, then the action with ITS body.
    assert [(r["method"], r["path"]) for r in millm.seen] == [("GET", "/api/approvals/actions"), (method, path)]
    sent = millm.seen[1]
    assert sent["body"] == body and sent["params"] == params
    assert all(r["headers"].get("x-millm-agent") == "agent:mistudio-mcp" for r in millm.seen)
    assert not [p for p in mistudio_seen if p[1].startswith("/api/v1/mcp/approvals")], (
        "the relay filed in miStudio's queue: miLLM decides its own requests now")


def test_the_load_sends_the_lease_ONLY_as_a_header(live):
    millm = _MiLLM()
    call, _ = live(millm)
    call("millm_load_model", {"model_id": 7, "lease_id": "L-secret"})
    load = _action_requests(millm)
    assert len(load) == 1
    assert load[0]["headers"].get("x-millm-lease") == "L-secret"
    assert load[0]["body"] is None and "L-secret" not in json.dumps(load[0]["params"])


def test_a_load_miLLM_simply_starts_reads_as_started(live):
    """No `X-miLLM-Approval: pending` (an older pending-less answer, or an operator call): the 034 shape."""
    millm = _MiLLM(pending=False)
    call, _ = live(millm)
    result = call("millm_load_model", {"model_id": 7})
    assert result["load_started"] is True
    assert "decide_in" not in result


def test_a_202_without_the_pending_header_is_NOT_a_pending_answer():
    from src.mcp_server.millm_client import MiLLMResponse
    from src.mcp_server.tools.millm_approvals import relayed

    assert relayed(MiLLMResponse(202, {"status": "loading"}, {})) is None
    assert relayed(MiLLMResponse(200, {"x": 1}, {"x-millm-approval": "pending"})) is None
    assert relayed(MiLLMResponse(202, {"approval_id": "a"}, {"X-miLLM-Approval": "Pending"})) == {
        "approval_id": "a", "decide_in": "millm"}


def test_a_declarative_import_is_ONE_request_and_needs_no_queue(live):
    doc = {"format": "midataworks.rule-set/v1", "signals": [{"id": "p", "kind": "pattern"}]}
    millm = _MiLLM(actions=None)  # even a miLLM with no queue: nothing here is gated
    call, _ = live(millm)
    call("millm_import_rule_set", {"document": doc})
    assert [(r["method"], r["path"]) for r in millm.seen] == [("POST", "/api/rule-sets/import")]


# ── S3: fail closed against a miLLM without its queue ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool", sorted(RELAYED))
def test_a_miLLM_without_the_queue_is_refused_and_sent_nothing(live, tool):
    args, (method, path, _body, _params) = RELAYED[tool]
    millm = _MiLLM(actions=None)
    call, _ = live(millm)
    result = call(tool, args)
    assert result["refused"] == "millm_has_no_approval_queue", result
    assert _action_requests(millm) == [], f"{tool} sent {method} {path} to a miLLM that cannot gate it"


@pytest.mark.parametrize("tool", sorted(RELAYED))
def test_a_queue_that_does_not_list_the_action_is_refused_too(live, tool):
    args, (method, path, _body, _params) = RELAYED[tool]
    action = _ACTION_FOR[(method, path)]
    millm = _MiLLM(actions=[a for a in ALL_ACTIONS if a != action])
    call, _ = live(millm)
    result = call(tool, args)
    assert result["refused"] == "millm_has_no_approval_queue" and result["action"] == action
    assert _action_requests(millm) == []


def test_an_unreadable_queue_is_refused_not_trusted():
    """A 500 on the actions route is not an empty list and not a yes: refuse, and do not cache it."""
    from unittest.mock import MagicMock

    from src.mcp_server.client import BackendError
    from src.mcp_server.tools.millm_approvals import ApprovalQueueCheck

    calls = []

    class Millm:
        async def get(self, path, **_):
            calls.append(path)
            raise BackendError(500, "boom")

    gate = MagicMock()
    gate.ttl_s = 60.0
    check = ApprovalQueueCheck(Millm(), gate)
    assert asyncio.run(check.refusal("model_load"))["refused"] == "millm_has_no_approval_queue"
    asyncio.run(check.refusal("model_load"))
    assert calls == ["/api/approvals/actions"] * 2


def test_the_answer_is_cached_for_the_gate_window_only():
    from unittest.mock import MagicMock

    from src.mcp_server.tools.millm_approvals import ApprovalQueueCheck

    calls = []

    class Millm:
        async def get(self, path, **_):
            calls.append(path)
            return {"actions": ["model_load"]}

    gate = MagicMock()
    gate.ttl_s = 60.0
    check = ApprovalQueueCheck(Millm(), gate)
    assert asyncio.run(check.refusal("model_load")) is None
    assert asyncio.run(check.refusal("model_load")) is None
    assert len(calls) == 1
    gate.ttl_s = 0.0
    asyncio.run(check.refusal("model_load"))
    assert len(calls) == 2


@pytest.mark.parametrize("body,expected", [
    (["model_load"], {"model_load"}),
    ([{"action": "model_load"}, {"name": "model_unload"}, {"id": "model_delete"}],
     {"model_load", "model_unload", "model_delete"}),
    ({"actions": [{"action": "model_load"}]}, {"model_load"}),
    ({"actions": {"model_load": {"label": "x"}}}, {"model_load"}),
    ({"model_load": "Load a model"}, {"model_load"}),
    (None, set()),
    ("model_load", set()),
])
def test_the_action_list_is_read_in_any_documented_shape(body, expected):
    from src.mcp_server.tools.millm_approvals import _action_names

    assert _action_names(body) == expected


# ── the deprecated parameter ──────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool", ["millm_load_model", "millm_request_probe_limits", "millm_measure_probe_capacity",
                                  "millm_import_rule_set"])
def test_the_old_approval_id_is_refused_and_names_the_new_tool(live, tool):
    args = dict(RELAYED[tool][0], approval_request_id="ap_old")
    millm = _MiLLM()
    call, mistudio_seen = live(millm)
    result = call(tool, args)
    assert result["refused"] == "approval_moved_to_millm" and result["next"] == "millm_get_approval_status"
    assert millm.seen == [] and mistudio_seen == []


# ── the two new read tools ────────────────────────────────────────────────────────────────────────────────────────


def test_list_approvals_is_one_GET_with_its_filters(live):
    millm = _MiLLM()
    call, _ = live(millm)
    call("millm_list_approvals", {"status": "pending", "limit": 200})
    assert [(r["method"], r["path"], r["params"]) for r in millm.seen] == [
        ("GET", "/api/approvals", {"status": "pending", "limit": "200"})]


def test_get_approval_status_is_one_GET_and_says_what_next(live):
    millm = _MiLLM()

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/approvals/apr_1":
            millm.seen.append({"method": request.method, "path": request.url.path})
            return httpx.Response(200, json={"success": True, "data": {"id": "apr_1", "status": "rejected",
                                                                       "reason": "no"}})
        return millm(request)

    call, _ = live(answer)
    result = call("millm_get_approval_status", {"approval_id": "apr_1"})
    assert [(r["method"], r["path"]) for r in millm.seen] == [("GET", "/api/approvals/apr_1")]
    assert result["status"] == "rejected" and "Rejected" in result["next"]


def test_an_unknown_id_reads_not_found(live):
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"success": False, "error": {"code": "approval_not_found", "message": "x"}})

    call, _ = live(answer)
    result = call("millm_get_approval_status", {"approval_id": "apr_x"})
    assert result["status"] == "not_found"


def test_both_new_tools_are_in_the_live_registry_once(live):
    call, _ = live(_MiLLM())
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32), stdio=True)
    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert names.count("millm_list_approvals") == 1 and names.count("millm_get_approval_status") == 1
    for tool in RELAYED:
        assert names.count(tool) == 1, tool
