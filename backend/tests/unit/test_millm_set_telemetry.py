"""`millm_set_telemetry` relays an agent's telemetry-settings change to miLLM's own approval queue
(miLLM MCP contract v1.23; suite decisions ENH-006 D75-D78).

miLLM gates an agent's write of a TELEMETRY_* setting as `telemetry_write`; an agent's write of any
OTHER runtime setting runs at once (ENH-003 D5). So the tool's client-side key check is what keeps
the MCP from becoming an ungated settings writer, and the fail-closed queue check is what keeps an
older miLLM (no queue, or no `telemetry_write`) from running the change with nobody approving it.

Everything runs through the LIVE registry: `build_server` with the categories the deployed pod
enables (`k8s/base/mcp.yaml`), the real `MiLLMClient` over an `httpx.MockTransport` standing in for
miLLM. Payloads and request counts are asserted, not just "was called".

Mutation controls (0xcc/reviews/millm_telemetry_proxy_controls_2026-10-29.md): delete the tool's
`@mcp.tool()` line; drop the client-side key check; drop the fail-closed queue check.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.support.sibling_repos import millm_repo

BACKEND = Path(__file__).resolve().parents[2]
MISTUDIO_URL = "http://mistudio.test"
MILLM_URL = "http://millm.test"
ALL_ACTIONS = ["model_load", "model_unload", "model_delete", "probe_limits_write", "capacity_measure",
               "code_rule_import", "telemetry_write"]


def _manifest_categories() -> str:
    import yaml

    for doc in yaml.safe_load_all((BACKEND.parent / "k8s" / "base" / "mcp.yaml").read_text()):
        for container in (((doc or {}).get("spec") or {}).get("template") or {}).get("spec", {}).get("containers", []):
            for e in container.get("env", []):
                if e.get("name") == "MCP_TOOL_CATEGORIES":
                    return e["value"]
    raise AssertionError("no MCP_TOOL_CATEGORIES in k8s/base/mcp.yaml")


PENDING = {"approval_id": "apr_tel", "status": "pending", "action": "telemetry_write",
           "summary": {"TELEMETRY_TRACES_ENABLED": {"now": False, "after": True}},
           "request_digest": "sha256:" + "a" * 64, "expires_at": "2026-10-30T12:00:00Z",
           "hint": "Waiting for the operator to approve this in miLLM's Approvals page."}


class _MiLLM:
    """miLLM with (or without) its approval queue. Records every request except the gate's probe."""

    def __init__(self, actions: Any = "all", *, put_status: int = 202, put_body: Any = None):
        self.actions = ALL_ACTIONS if actions == "all" else actions
        self.put_status = put_status
        self.put_body = put_body
        self.seen: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content) if request.content else None
        self.seen.append({"method": request.method, "path": path, "body": body,
                          "headers": dict(request.headers)})
        if path == "/api/approvals/actions":
            if self.actions is None:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json={"success": True, "data": {
                "actions": [{"action": a, "label": a} for a in self.actions]}, "error": None})
        if (request.method, path) == ("PUT", "/api/environment"):
            if self.put_status == 202:
                return httpx.Response(202, json={"success": True, "data": PENDING, "error": None},
                                      headers={"X-miLLM-Approval": "pending"})
            return httpx.Response(self.put_status, json=self.put_body)
        return httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture
def live(monkeypatch):
    """`build(millm)` → `(call, names)`: tool calls against the deployed-category server."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    def build(millm):
        real = httpx.AsyncClient

        def mistudio(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok"})

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

        def call(args: dict) -> Any:
            out = asyncio.run(mcp._tool_manager.call_tool("millm_set_telemetry", args, context=None))  # noqa: SLF001
            return out

        names = [t.name for t in asyncio.run(mcp.list_tools())]
        return call, names

    return build


# ── wiring ────────────────────────────────────────────────────────────────────────────────────────


def test_the_tool_is_in_the_live_registry_once_with_the_deployed_categories(live):
    _call, names = live(_MiLLM())
    assert names.count("millm_set_telemetry") == 1


def test_it_is_absent_without_millm(monkeypatch):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    monkeypatch.delenv("MILLM_API_URL", raising=False)
    mcp, _ = build_server(MCPSettings(tool_categories="read,millm_runtime", allow_anonymous=True),
                          stdio=True)
    assert "millm_set_telemetry" not in {t.name for t in asyncio.run(mcp.list_tools())}


# ── the relay: one check, one PUT, miLLM's pending answer plus decide_in ───────────────────────────


def test_a_pending_answer_is_relayed_with_decide_in_after_exactly_two_requests(live):
    millm = _MiLLM()
    call, _ = live(millm)
    result = call({"values": {"TELEMETRY_TRACES_ENABLED": True,
                              "TELEMETRY_ENDPOINT": "https://otel.example:4318"},
                   "reset": ["TELEMETRY_SAMPLE_RATIO"], "reason": "turn traces on"})

    assert result == {**PENDING, "decide_in": "millm"}
    assert [(r["method"], r["path"]) for r in millm.seen] == [
        ("GET", "/api/approvals/actions"), ("PUT", "/api/environment")]
    assert millm.seen[1]["body"] == {
        "values": {"TELEMETRY_TRACES_ENABLED": True, "TELEMETRY_ENDPOINT": "https://otel.example:4318"},
        "reset": ["TELEMETRY_SAMPLE_RATIO"], "reason": "turn traces on"}
    assert all(r["headers"].get("x-millm-agent") == "agent:mistudio-mcp" for r in millm.seen)


def test_reset_and_reason_default_to_an_empty_list_and_null(live):
    millm = _MiLLM()
    call, _ = live(millm)
    call({"values": {"TELEMETRY_LOGS_ENABLED": False}})
    puts = [r for r in millm.seen if r["method"] == "PUT"]
    assert len(puts) == 1
    assert puts[0]["body"] == {"values": {"TELEMETRY_LOGS_ENABLED": False}, "reset": [], "reason": None}


def test_a_reset_alone_is_sent(live):
    millm = _MiLLM()
    call, _ = live(millm)
    call({"values": {}, "reset": ["TELEMETRY_ENDPOINT"]})
    puts = [r for r in millm.seen if r["method"] == "PUT"]
    assert [p["body"] for p in puts] == [{"values": {}, "reset": ["TELEMETRY_ENDPOINT"], "reason": None}]


# ── client-side refusals: zero HTTP requests ──────────────────────────────────────────────────────


@pytest.mark.parametrize("args,key,code", [
    ({"values": {"LOG_LEVEL": "DEBUG"}}, "LOG_LEVEL", "not_a_telemetry_setting"),
    ({"values": {"TELEMETRY_TRACES_ENABLED": True, "PROBE_MAX_ARMED": 9}}, "PROBE_MAX_ARMED",
     "not_a_telemetry_setting"),
    ({"values": {}, "reset": ["MAX_BATCH_SIZE"]}, "MAX_BATCH_SIZE", "not_a_telemetry_setting"),
    ({"values": {"telemetry_traces_enabled": True}}, "telemetry_traces_enabled", "not_a_telemetry_setting"),
    ({"values": {"TELEMETRY_HEADERS": "Authorization=Bearer x"}}, "TELEMETRY_HEADERS", "setting_not_runtime"),
    ({"values": {"TELEMETRY_TLS_CLIENT_KEY_FILE": "/k"}}, "TELEMETRY_TLS_CLIENT_KEY_FILE", "setting_not_runtime"),
    ({"values": {"TELEMETRY_TLS_CA_FILE": "/ca"}}, "TELEMETRY_TLS_CA_FILE", "setting_not_runtime"),
    ({"values": {"TELEMETRY_TLS_CLIENT_CERT_FILE": "/c"}}, "TELEMETRY_TLS_CLIENT_CERT_FILE", "setting_not_runtime"),
    ({"values": {"OTEL_SDK_DISABLED": True}}, "OTEL_SDK_DISABLED", "setting_not_runtime"),
    ({"values": {}, "reset": ["TELEMETRY_HEADERS"]}, "TELEMETRY_HEADERS", "setting_not_runtime"),
])
def test_a_key_it_may_not_write_is_refused_and_nothing_is_sent(live, args, key, code):
    millm = _MiLLM()
    call, _ = live(millm)
    result = call(args)
    assert result["refused"] == code and result["key"] == key, result
    assert key in result["message"]
    assert millm.seen == [], f"{key} was refused only after {millm.seen}"


def test_naming_nothing_is_refused_and_nothing_is_sent(live):
    millm = _MiLLM()
    call, _ = live(millm)
    assert call({"values": {}})["refused"] == "nothing_to_change"
    assert millm.seen == []


# ── fail closed ───────────────────────────────────────────────────────────────────────────────────


def test_a_queue_without_telemetry_write_is_refused_with_zero_puts(live):
    millm = _MiLLM(actions=[a for a in ALL_ACTIONS if a != "telemetry_write"])
    call, _ = live(millm)
    result = call({"values": {"TELEMETRY_TRACES_ENABLED": True}})
    assert result["refused"] == "millm_has_no_approval_queue" and result["action"] == "telemetry_write"
    assert [(r["method"], r["path"]) for r in millm.seen] == [("GET", "/api/approvals/actions")]


def test_a_miLLM_with_no_queue_is_refused_with_zero_puts(live):
    millm = _MiLLM(actions=None)
    call, _ = live(millm)
    result = call({"values": {"TELEMETRY_TRACES_ENABLED": True}})
    assert result["refused"] == "millm_has_no_approval_queue"
    assert not [r for r in millm.seen if r["method"] == "PUT"]


# ── miLLM's own answers ───────────────────────────────────────────────────────────────────────────


def test_a_millm_refusal_is_passed_through_clearly(live):
    millm = _MiLLM(put_status=422, put_body={"success": False, "data": None, "error": {
        "code": "SETTING_INVALID", "message": "TELEMETRY_ENDPOINT must not carry a password",
        "details": {"key": "TELEMETRY_ENDPOINT"}}})
    call, _ = live(millm)
    result = call({"values": {"TELEMETRY_ENDPOINT": "https://u:p@otel.example"}})
    assert result == {"refused": "millm_refused", "status_code": 422, "code": "SETTING_INVALID",
                      "message": "TELEMETRY_ENDPOINT must not carry a password",
                      "details": {"key": "TELEMETRY_ENDPOINT"}}
    assert len([r for r in millm.seen if r["method"] == "PUT"]) == 1


# ── F-3 (2026-10-29): every refusal shape reaches the agent; an over-long reason never leaves ────
#
# The GPU window saw miLLM's 422 for a 600-character `reason` relayed as `code: null, message: null`:
# the tool read only the unwrapped envelope, and miLLM answers body validation in FastAPI's own
# `{"detail": [...]}`. Controls (0xcc/reviews/telemetry_error_relay_controls_2026-10-29.md): read
# only the envelope shape → the detail-list and text tests go red; drop the client-side length
# check → the over-long-reason test goes red.

#: Exactly what miLLM's management routes answer for `reason` over the limit (pydantic 2,
#: `exception_handlers.openai_validation_error_handler`, non-/v1 branch).
TOO_LONG_DETAIL = {"detail": [{
    "type": "string_too_long", "loc": ["body", "reason"],
    "msg": "String should have at most 500 characters", "input": "x" * 501,
    "ctx": {"max_length": 500}}]}


def test_a_fastapi_detail_list_relays_its_message(live):
    millm = _MiLLM(put_status=422, put_body=TOO_LONG_DETAIL)
    call, _ = live(millm)
    result = call({"values": {"TELEMETRY_TRACES_ENABLED": True}, "reason": "short"})
    assert result == {"refused": "millm_refused", "status_code": 422, "code": "VALIDATION_ERROR",
                      "message": "reason: String should have at most 500 characters",
                      "details": TOO_LONG_DETAIL["detail"]}
    assert len([r for r in millm.seen if r["method"] == "PUT"]) == 1


def test_a_fastapi_detail_string_relays_it(live):
    call, _ = live(_MiLLM(put_status=409, put_body={"detail": "Another save is in progress"}))
    result = call({"values": {"TELEMETRY_TRACES_ENABLED": True}})
    assert result["refused"] == "millm_refused" and result["status_code"] == 409
    assert result["message"] == "Another save is in progress"


class _TextMiLLM(_MiLLM):
    """A PUT answered by something that is not miLLM's JSON: a proxy's plain-text page."""

    def __init__(self, status: int, text: str):
        super().__init__()
        self.status, self.text = status, text

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if (request.method, request.url.path) == ("PUT", "/api/environment"):
            self.seen.append({"method": "PUT", "path": request.url.path})
            return httpx.Response(self.status, text=self.text,
                                  headers={"content-type": "text/plain"})
        return super().__call__(request)


def test_a_non_json_body_relays_the_status_and_its_text(live):
    millm = _TextMiLLM(413, "413 Request Entity Too Large (nginx)")
    call, _ = live(millm)
    result = call({"values": {"TELEMETRY_TRACES_ENABLED": True}})
    assert result == {"refused": "millm_refused", "status_code": 413, "code": None,
                      "message": "miLLM answered HTTP 413: 413 Request Entity Too Large (nginx)",
                      "details": None}
    assert len([r for r in millm.seen if r["method"] == "PUT"]) == 1


def test_a_long_non_json_body_is_trimmed_not_dropped(live):
    from src.mcp_server.millm_client import ERROR_TEXT_MAX_CHARS

    call, _ = live(_TextMiLLM(400, "<html>" + "y" * 5000 + "</html>"))
    message = call({"values": {"TELEMETRY_TRACES_ENABLED": True}})["message"]
    assert message.startswith("miLLM answered HTTP 400: <html>yyy")
    assert len(message) < ERROR_TEXT_MAX_CHARS + 100
    assert "5013 characters in all" in message


def test_an_over_long_reason_is_refused_with_zero_requests(live):
    from src.mcp_server.tools.millm_runtime import REASON_MAX_CHARS

    requests: list[str] = []
    millm = _MiLLM()

    def counting(request: httpx.Request) -> httpx.Response:
        requests.append(f"{request.method} {request.url.path}")
        return millm(request)

    call, _ = live(counting)
    result = call({"values": {"TELEMETRY_TRACES_ENABLED": True}, "reason": "z" * (REASON_MAX_CHARS + 1)})
    assert result["refused"] == "reason_too_long"
    assert result["length"] == REASON_MAX_CHARS + 1 and result["max"] == REASON_MAX_CHARS
    assert "Nothing was sent" in result["message"]
    assert requests == []


def test_a_reason_at_the_limit_is_sent_verbatim(live):
    from src.mcp_server.tools.millm_runtime import REASON_MAX_CHARS

    millm = _MiLLM()
    call, _ = live(millm)
    reason = "z" * REASON_MAX_CHARS
    result = call({"values": {"TELEMETRY_TRACES_ENABLED": True}, "reason": reason})
    assert result["status"] == "pending"
    puts = [r for r in millm.seen if r["method"] == "PUT"]
    assert len(puts) == 1 and puts[0]["body"]["reason"] == reason


@pytest.mark.parametrize("detail,expected", [
    ({"code": "SETTING_INVALID", "message": "bad", "details": {"k": 1}},
     {"code": "SETTING_INVALID", "message": "bad", "details": {"k": 1}}),
    ({"success": False, "data": None, "error": {"code": "C", "message": "m", "details": None}},
     {"code": "C", "message": "m", "details": None}),
    ({"error": {"message": "Invalid value for 'x'", "type": "invalid_request_error", "code": "invalid_parameter"}},
     {"code": "invalid_parameter", "message": "Invalid value for 'x'", "details": None}),
    ([{"loc": ["body", "values"], "msg": "Field required"}],
     {"code": "VALIDATION_ERROR", "message": "values: Field required",
      "details": [{"loc": ["body", "values"], "msg": "Field required"}]}),
    ("", {"code": None, "message": "miLLM answered HTTP 422 with an empty body", "details": None}),
    (None, {"code": None, "message": "miLLM answered HTTP 422 with an empty body", "details": None}),
    ({"weird": 1}, {"code": None, "message": 'miLLM answered HTTP 422 without a message: {"weird": 1}',
                    "details": None}),
])
def test_error_fields_reads_every_shape(detail, expected):
    from src.mcp_server.millm_client import error_fields

    assert error_fields(422, detail) == expected


def test_the_send_refusal_reads_a_fastapi_body_too():
    """The shared reader is wired into millm_send too: a 503 from FastAPI used to carry message ''."""
    from src.mcp_server.client import BackendError
    from src.mcp_server.millm_send import refusal

    out = refusal(BackendError(503, {"detail": "model is loading"}))
    assert out["message"] == "model is loading"
    out = refusal(BackendError(503, "Service Unavailable"))
    assert out["message"] == "miLLM answered HTTP 503: Service Unavailable"


def test_millm_down_answers_unavailable_and_sends_nothing(live):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    call, _ = live(down)
    result = call({"values": {"TELEMETRY_TRACES_ENABLED": True}})
    assert result["unavailable"] == "millm"


def test_a_closed_gate_sends_nothing():
    """The hand-rolled gate (the key check runs first, so the decorator is not used)."""
    from unittest.mock import AsyncMock, MagicMock

    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import millm_runtime
    from tests.support.external_ctx import make_ctx

    calls = []

    class Millm:
        async def get(self, path, **_):
            calls.append(("GET", path))
            return {"actions": ALL_ACTIONS}

        async def put(self, path, **_):
            calls.append(("PUT", path))
            raise AssertionError("PUT with the gate closed")

    gate = MagicMock()
    gate.check = AsyncMock(return_value=(False, "connection refused"))
    mcp = MCPServer("t")
    millm_runtime.register(mcp, Millm(), gate, make_ctx())
    result = asyncio.run(mcp._tool_manager.call_tool(  # noqa: SLF001
        "millm_set_telemetry", {"values": {"TELEMETRY_TRACES_ENABLED": True}}, context=None))
    assert result == {"unavailable": "millm", "reason": "connection refused"}
    assert calls == []


# ── cross-repo: the environment-only set is miLLM's ───────────────────────────────────────────────

_DUMP = (
    "import json\n"
    "from millm.services import runtime_settings as r\n"
    "from millm.api.routes.management.environment import EnvironmentSave as S\n"
    "limit = [m.max_length for m in S.model_fields['reason'].metadata"
    " if getattr(m, 'max_length', None) is not None]\n"
    "print('@@KEYS@@' + json.dumps({'env_only': sorted(r.TELEMETRY_ENV_ONLY),"
    " 'runtime': sorted(k for k in r.RUNTIME_KEYS if r.is_telemetry_key(k)),"
    " 'reason_max': limit}))\n"
)


def test_the_environment_only_set_matches_miLLMs():
    """CROSS-REPO. The tool's env-only list must be miLLM's `TELEMETRY_ENV_ONLY`, and every telemetry
    key miLLM makes writable must pass the tool's check. Skips loudly without the sibling; fails under
    MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1."""
    from src.mcp_server.tools.millm_runtime import (
        REASON_MAX_CHARS, TELEMETRY_ENV_ONLY, telemetry_key_refusal)

    repo = millm_repo()
    python = Path(os.environ.get("MILLM_PYTHON") or repo / "venv" / "bin" / "python")
    if not (repo / "millm" / "services" / "runtime_settings.py").is_file() or not python.is_file():
        reason = f"no miLLM checkout/interpreter at {repo} / {python} (set MILLM_REPO, MILLM_PYTHON)"
        if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(reason)
        pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + reason)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_URL", "PYTHONPATH"))}
    proc = subprocess.run([str(python), "-B", "-c", _DUMP], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=240)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@KEYS@@")]
    if proc.returncode != 0 or not lines:
        if "TELEMETRY_ENV_ONLY" in proc.stderr and "has no attribute" in proc.stderr:
            pytest.skip(f"CROSS-REPO, SKIPPED LOUDLY: the miLLM at {repo} predates contract v1.23")
        pytest.fail(f"miLLM's settings did not load (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    keys = json.loads(lines[0][len("@@KEYS@@"):])
    assert sorted(TELEMETRY_ENV_ONLY) == keys["env_only"]
    # F-3: the client-side `reason` limit is the one PUT /api/environment's body schema enforces.
    assert keys["reason_max"] == [REASON_MAX_CHARS]
    assert keys["runtime"], "miLLM lists no writable telemetry key: the probe sees nothing"
    blocked = [k for k in keys["runtime"] if telemetry_key_refusal({k: 1}, []) is not None]
    assert not blocked, f"telemetry keys miLLM lets the operator write that the tool refuses: {blocked}"
