"""`millm_load_model` waits for operator approval (Feature 034, FR-4/FR-20, SC-5).

Loading evicts the model miLLM serves. With `MCP_MILLM_LOAD_APPROVAL` on (the default, X-10) an
agent's first call files a request in miStudio's banner and loads NOTHING; the re-call with the
approval id consumes it — once, for exactly that model — and only then loads.

Order (FTID §3.3, TI4): arguments → (no id: create approval, return) → gate → consume → load. The
gate runs before the consume so a down miLLM does not cost an approval.

Controls (recorded 2026-10-16, task 2.8; each red, each restore verified by sha256 + git diff):
  * delete the `_consume(...)` call → `test_with_an_id_it_consumes_then_loads` red;
  * put `lease_id` into the approval `payload` → `test_the_lease_id_never_enters_the_approval` red;
  * flip `if approval_on and approval_request_id is None` to `if not approval_on and …` →
    `test_flag_on_without_an_id_files_ONE_request_and_loads_nothing` red;
  * flag default `True → False` (`config.py`) → `test_the_flag_defaults_ON` red.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.mcp_server.client import BackendError, MiStudioClient
from src.mcp_server.config import MCPSettings
from src.mcp_server.millm_client import MiLLMResponse
from tests.support.external_ctx import make_ctx


class _Recorder:
    """Records (client, method, path, kwargs) across BOTH clients, in order. Required arguments
    stay required, as on the real clients."""

    def __init__(self, log: list, name: str, responses: dict | None = None):
        self.log = log
        self.name = name
        self.responses = responses or {}

    def _answer(self, method: str, path: str) -> Any:
        answer = self.responses.get((method, path))
        if isinstance(answer, Exception):
            raise answer
        return answer if answer is not None else {}

    async def post(self, path, json_body=None, *, headers=None, full=False, **params):
        entry = {"json_body": json_body, **params}
        if headers is not None:
            entry["headers"] = headers
        self.log.append((self.name, "POST", path, entry))
        answer = self._answer("POST", path)
        return MiLLMResponse(200, answer, {}) if full else answer

    async def get(self, path, **params):
        self.log.append((self.name, "GET", path, params))
        return self._answer("GET", path)



def _gate(open_: bool = True):
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, "ok") if open_ else (False, "connection refused"))
    gate.invalidate = MagicMock()
    return gate


def _call(args: dict, *, flag: bool = True, gate_open: bool = True,
          mistudio_responses: dict | None = None, millm_responses: dict | None = None):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import millm_models

    log: list = []
    mistudio = _Recorder(log, "mistudio", mistudio_responses or {
        ("POST", "/mcp/approvals"): {"id": "ap_1", "status": "pending"},
    })
    millm = _Recorder(log, "millm", millm_responses or {
        ("POST", "/api/models/7/load"): {"data": {"id": 7}},
    })
    gate = _gate(gate_open)
    mcp = MCPServer("t")
    millm_models.register(mcp, millm, gate, make_ctx(mistudio, millm_load_approval=flag))
    fn = mcp._tool_manager._tools["millm_load_model"].fn  # noqa: SLF001
    return asyncio.run(fn(**args)), log, gate


# ── 2.1 the flag ─────────────────────────────────────────────────────────────────────────────


class TestTheFlag:
    def test_the_flag_defaults_ON(self, monkeypatch):
        monkeypatch.delenv("MCP_MILLM_LOAD_APPROVAL", raising=False)
        assert MCPSettings(_env_file=None).millm_load_approval is True

    def test_the_env_var_turns_it_off(self, monkeypatch):
        monkeypatch.setenv("MCP_MILLM_LOAD_APPROVAL", "false")
        assert MCPSettings(_env_file=None).millm_load_approval is False

    def test_an_unparseable_value_fails_LOUDLY(self, monkeypatch):
        """Fail closed at startup, never a silent permissive default."""
        monkeypatch.setenv("MCP_MILLM_LOAD_APPROVAL", "maybe")
        with pytest.raises(Exception):
            MCPSettings(_env_file=None)


# ── 2.8 the two shapes and the sequence ──────────────────────────────────────────────────────


class TestTheApprovalSequence:
    def test_flag_on_without_an_id_files_ONE_request_and_loads_nothing(self):
        result, log, gate = _call({"model_id": 7})
        assert log == [(
            "mistudio", "POST", "/mcp/approvals",
            {"json_body": {"tool_name": "millm_load_model", "payload": {"model_id": 7}}},
        )]
        assert result["status"] == "pending_approval"
        assert result["approval_request_id"] == "ap_1"
        assert "approval_request_id" in result["hint"]

    def test_with_an_id_it_consumes_then_loads(self):
        result, log, _ = _call({"model_id": 7, "approval_request_id": "ap_1"})
        assert log == [
            ("mistudio", "POST", "/mcp/approvals/ap_1/consume",
             {"json_body": {"tool_name": "millm_load_model", "payload": {"model_id": 7}}}),
            ("millm", "POST", "/api/models/7/load", {"json_body": None}),
        ]
        assert result["load_started"] is True

    def test_the_lease_id_travels_as_a_header_only(self):
        result, log, _ = _call(
            {"model_id": 7, "approval_request_id": "ap_1", "lease_id": "lease-secret"}
        )
        assert log[1] == ("millm", "POST", "/api/models/7/load",
                          {"json_body": None, "headers": {"X-miLLM-Lease": "lease-secret"}})
        assert result["load_started"] is True

    def test_the_lease_id_never_enters_the_approval(self):
        """miLLM FR-29.1.6: the lease id is the only proof of holding. An approval row is
        readable by anyone on the LAN."""
        _, log, _ = _call({"model_id": 7, "lease_id": "lease-secret"})
        _, log2, _ = _call({"model_id": 7, "approval_request_id": "ap_1",
                            "lease_id": "lease-secret"})
        for entry in log + [log2[0]]:
            assert "lease-secret" not in repr(entry), entry

    @pytest.mark.parametrize("status", ["pending", "denied", "expired", "consumed"])
    def test_a_refused_consume_LOADS_NOTHING_and_says_why(self, status):
        refusal = BackendError(409, {"status": status, "reason": "r"})
        result, log, _ = _call(
            {"model_id": 7, "approval_request_id": "ap_1"},
            mistudio_responses={("POST", "/mcp/approvals/ap_1/consume"): refusal},
        )
        assert [e[0] for e in log] == ["mistudio"], log
        assert result["status"] == status
        assert result["hint"]
        if status in ("denied", "expired", "consumed"):
            assert "without approval_request_id" in result["hint"]

    def test_a_mismatched_payload_loads_nothing(self):
        refusal = BackendError(422, "the payload differs")
        result, log, _ = _call(
            {"model_id": 8, "approval_request_id": "ap_1"},
            mistudio_responses={("POST", "/mcp/approvals/ap_1/consume"): refusal},
        )
        assert [e[0] for e in log] == ["mistudio"]
        assert result["status"] == "mismatch"

    def test_a_closed_gate_with_an_id_CONSUMES_NOTHING(self):
        """A down miLLM must not cost the operator's approval."""
        result, log, _ = _call({"model_id": 7, "approval_request_id": "ap_1"},
                               gate_open=False)
        assert log == []
        assert result == {"unavailable": "millm", "reason": "connection refused"}

    def test_a_double_recall_loads_once(self):
        """The second re-call hits the consumed approval: one load in total."""
        _, first, _ = _call({"model_id": 7, "approval_request_id": "ap_1"})
        refusal = BackendError(409, {"status": "consumed", "reason": None})
        second, log, _ = _call(
            {"model_id": 7, "approval_request_id": "ap_1"},
            mistudio_responses={("POST", "/mcp/approvals/ap_1/consume"): refusal},
        )
        loads = [e for e in first + log if e[0] == "millm"]
        assert len(loads) == 1 and second["status"] == "consumed"

    def test_a_refused_load_reports_the_approval_as_SPENT(self):
        locked = BackendError(409, {"code": "MODEL_LOCKED", "message": "locked for steering"})
        result, log, _ = _call(
            {"model_id": 7, "approval_request_id": "ap_1"},
            millm_responses={("POST", "/api/models/7/load"): locked},
        )
        assert result["load_started"] is False and result["approval_consumed"] is True
        assert result["error"]["code"] == "MODEL_LOCKED"

    def test_flag_off_issues_ONE_load_and_no_approval_call(self):
        result, log, _ = _call({"model_id": 7}, flag=False)
        assert log == [("millm", "POST", "/api/models/7/load", {"json_body": None})]
        assert result["load_started"] is True

    def test_flag_off_a_closed_gate_issues_nothing(self):
        result, log, _ = _call({"model_id": 7}, flag=False, gate_open=False)
        assert log == [] and result["unavailable"] == "millm"

    def test_an_unreachable_miStudio_reads_as_unavailable(self):
        down = BackendError(0, "miStudio backend unreachable: boom")
        result, log, _ = _call(
            {"model_id": 7}, mistudio_responses={("POST", "/mcp/approvals"): down},
        )
        assert result["unavailable"] == "mistudio" and not [e for e in log if e[0] == "millm"]

    def test_the_description_states_both_shapes_and_both_refusals(self):
        from mcp.server.mcpserver import MCPServer

        from src.mcp_server.tools import millm_models

        mcp = MCPServer("t")
        millm_models.register(mcp, MagicMock(), _gate(), make_ctx())
        d = mcp._tool_manager._tools["millm_load_model"].description  # noqa: SLF001
        for needle in ("pending_approval", "approval_request_id", "LOCKED", "LEASE", "EVICTS"):
            assert needle in d, needle


# ── 2.7 the miStudio client refuses a 2xx that is not JSON ───────────────────────────────────


class TestTheMiStudioClientRefusesNonJson:
    """Control (recorded 2026-10-16, task 2.7): replacing `if unparseable:` with `if False:` in
    `MiStudioClient.request` → `test_an_HTML_200_RAISES` red."""

    def _client(self, handler) -> MiStudioClient:
        client = MiStudioClient("http://backend")
        client._http = httpx.AsyncClient(  # noqa: SLF001
            base_url="http://backend/api/v1", transport=httpx.MockTransport(handler)
        )
        return client

    def test_an_HTML_200_RAISES(self):
        client = self._client(lambda r: httpx.Response(
            200, text="<html>splash</html>", headers={"content-type": "text/html"}))
        with pytest.raises(BackendError) as exc:
            asyncio.run(client.post("/mcp/approvals", json_body={}))
        assert "non-JSON" in str(exc.value) and "NOT an empty result" in str(exc.value)

    def test_an_empty_200_is_still_an_empty_dict(self):
        client = self._client(lambda r: httpx.Response(204))
        assert asyncio.run(client.get("/x")) == {}

    def test_JSON_still_parses(self):
        client = self._client(lambda r: httpx.Response(200, json={"id": "ap_1"}))
        assert asyncio.run(client.post("/mcp/approvals", json_body={})) == {"id": "ap_1"}
