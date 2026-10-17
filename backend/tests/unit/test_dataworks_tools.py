"""The `dataworks` proxy tools (Feature 034, FR-10 – FR-14, SC-3, SC-4).

* **Approvals pass through** (FR-11): a recorded `202` comes back unchanged (a `hint` is added
  only when one is missing); no tool calls miStudio's `/mcp/approvals`; no tool references an
  approve or reject path (AST); `expired` is reported verbatim.
* **Arguments before the gate** (FR-14): bad arguments with a closed gate report the arguments.
* **No credential parameter** on any `dataworks_*` or Feature 034 `millm_*` tool (FR-13).
* **Unset URL → skipped with one warning; set but down → registered, structured unavailable**
  (FR-6, FR-7).

Controls (recorded 2026-10-16; each red, each restore verified):
  * `with_poll_hint` returns `{}` for a pending body → `test_a_202_comes_back_UNCHANGED` red;
  * add `hf_token: str = ""` to `dataworks_publish_version` → `test_no_credential_parameter` red;
  * move the reserved-key check after the gate → `test_bad_options_with_a_CLOSED_gate…` red.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.support.external_ctx import make_ctx

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "tools"
PENDING = {"approval_id": "ap1", "status": "pending", "action": "hub_push",
           "request_digest": "sha256:" + "0" * 64, "expires_at": "2026-10-17T00:00:00+00:00",
           "hint": "Waiting for the operator to approve this in miDataworks."}


class _Dw:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer if answer is not None else {}

    async def _r(self, method, path, json_body=None, **params):
        self.calls.append((method, path, json_body, params))
        return self.answer

    async def get(self, path, **params):
        return await self._r("GET", path, **params)

    async def post(self, path, json_body=None, **params):
        return await self._r("POST", path, json_body, **params)

    async def delete(self, path, json_body=None, **params):
        return await self._r("DELETE", path, json_body, **params)


def _tools(dw, gate_open=True):
    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.tools import dataworks

    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, "ok") if gate_open else (False, "refused"))
    gate.invalidate = MagicMock()
    mcp = FastMCP("t")
    dataworks.register(mcp, dw, gate, make_ctx())
    return mcp._tool_manager._tools, gate  # noqa: SLF001


def _run(name, dw, gate_open=True, **args):
    tools, gate = _tools(dw, gate_open)
    return asyncio.run(tools[name].fn(**args)), gate


ACT = {
    "dataworks_publish_version": {"version_id": "v1", "build_id": "b1", "repo_id": "o/n"},
    "dataworks_send_detector_set": {"set_id": "ds1"},
    "dataworks_delete_version": {"version_id": "v1", "reason": "old"},
    "dataworks_start_label_run": {"input_version_id": "v1", "role": "classifier"},
}


class TestApprovalsPassThrough:
    @pytest.mark.parametrize("name", sorted(ACT))
    def test_a_202_comes_back_UNCHANGED(self, name):
        result, _ = _run(name, _Dw(dict(PENDING)), **ACT[name])
        assert result == PENDING

    @pytest.mark.parametrize("name", sorted(ACT))
    def test_a_202_WITHOUT_a_hint_gains_one_naming_the_poll_tool(self, name):
        bare = {k: v for k, v in PENDING.items() if k != "hint"}
        result, _ = _run(name, _Dw(bare), **ACT[name])
        assert "dataworks_get_approval_status" in result["hint"]
        assert {k: v for k, v in result.items() if k != "hint"} == bare

    def test_expired_is_reported_VERBATIM(self):
        expired = {"id": "ap1", "status": "expired", "action": "hub_push"}
        result, _ = _run("dataworks_get_approval_status", _Dw(expired), approval_id="ap1")
        assert result == expired

    def test_no_tool_calls_miStudios_approval_queue_or_an_approve_path(self):
        """AST over the module's string constants and f-string parts: no `/mcp/approvals`, no
        `/approve`, no `/reject`. The proxy can neither approve nor queue."""
        tree = ast.parse((TOOLS / "dataworks.py").read_text())
        strings = [n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        paths = [s for s in strings if s.startswith("/")]
        assert paths, "no paths found — the scan is checking nothing"
        bad = [s for s in strings if "/mcp/approvals" in s or re.search(r"/(approve|reject)\b", s)]
        assert not bad, bad

    def test_the_proxy_never_reads_ctx_mistudio(self):
        """miDataworks holds these approvals; miStudio's queue is not involved (FR-11)."""
        tree = ast.parse((TOOLS / "dataworks.py").read_text())
        uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "mistudio"]
        assert not uses


class TestArgumentsComeFirst:
    def test_bad_options_with_a_CLOSED_gate_report_the_arguments(self):
        dw = _Dw()
        result, gate = _run("dataworks_start_label_run", dw, gate_open=False,
                            input_version_id="v1", role="judge", options={"who": "operator"})
        assert result["error"] == "invalid_options" and "who" in result["message"]
        assert gate.check.await_count == 0 and not dw.calls

    def test_a_blank_reason_with_a_closed_gate_reports_the_reason(self):
        dw = _Dw()
        result, gate = _run("dataworks_delete_version", dw, gate_open=False,
                            version_id="v1", reason="   ")
        assert result["error"] == "invalid_reason" and gate.check.await_count == 0

    def test_a_closed_gate_is_dataworks_unavailable(self):
        dw = _Dw()
        result, _ = _run("dataworks_get_version", dw, gate_open=False, version_id="v1")
        assert result == {"unavailable": "dataworks", "reason": "refused"} and not dw.calls


def _feature_034_tools():
    """Every dataworks_* tool and every Feature 034 millm_* tool, from the live registry."""
    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.tools import (
        dataworks, millm_batches, millm_inference, millm_models, millm_probes,
    )

    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, "ok"))
    mcp = FastMCP("t")
    for module in (dataworks, millm_batches, millm_inference, millm_models, millm_probes):
        module.register(mcp, MagicMock(), gate, make_ctx())
    new_millm = {"millm_load_model", "millm_acquire_lease", "millm_renew_lease",
                 "millm_release_lease", "millm_lease_status", "millm_score_probes"}
    return [t for t in asyncio.run(mcp.list_tools())
            if t.name.startswith("dataworks_") or t.name in new_millm
            or t.name.startswith(("millm_score", "millm_generate", "millm_upload",
                                  "millm_submit", "millm_batch", "millm_cancel_batch"))]


#: FR-13's pattern, made precise: a parameter NAMED like a credential. The literal `token`
#: substring also matches tokenizer vocabulary (`allowed_token_ids`, `max_tokens`,
#: `add_special_tokens`), which carries no secret; `_token` / `token` as the final word does.
CREDENTIAL = re.compile(r"(?:^|_)token$|secret|password|api_?key|access_token|bearer", re.I)


def test_the_credential_pattern_sees_credentials_and_not_vocabulary():
    for name in ("hf_token", "token", "api_key", "apikey", "client_secret", "password",
                 "access_token", "bearer"):
        assert CREDENTIAL.search(name), name
    for name in ("allowed_token_ids", "max_tokens", "add_special_tokens", "return_token_ids",
                 "top_logprobs"):
        assert not CREDENTIAL.search(name), name


def test_no_credential_parameter():
    """FR-13: no tool here carries a secret through an agent transcript."""
    pattern = CREDENTIAL
    tools = _feature_034_tools()
    assert len(tools) >= 13 + 14, len(tools)
    offenders = [(t.name, p) for t in tools for p in (t.inputSchema.get("properties") or {})
                 if pattern.search(p)]
    assert not offenders, offenders


class TestRegistrationTopology:
    def _settings(self, **kw):
        from src.mcp_server.config import MCPSettings

        return MCPSettings(auth_token="secret-token", _env_file=None, **kw)

    def test_UNSET_url_skips_the_category_with_ONE_warning(self, monkeypatch, caplog):
        import logging

        from src.mcp_server.server import build_server

        monkeypatch.delenv("MIDATAWORKS_API_URL", raising=False)
        with caplog.at_level(logging.WARNING, logger="src.mcp_server.server"):
            mcp, _ = build_server(self._settings(tool_categories="read,dataworks"))
        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert not any(n.startswith("dataworks_") for n in names)
        warnings = [r.getMessage() for r in caplog.records if "MIDATAWORKS_API_URL" in r.getMessage()]
        assert len(warnings) == 1

    def test_SET_but_down_keeps_tools_and_answers_unavailable(self, monkeypatch):
        """Live SC-4's unit twin: an unreachable miDataworks → structured unavailable."""
        from src.mcp_server.server import build_server

        monkeypatch.setenv("MIDATAWORKS_API_URL", "http://127.0.0.1:1")
        mcp, _ = build_server(self._settings(tool_categories="dataworks"))
        tool = mcp._tool_manager._tools["dataworks_list_datasets"]  # noqa: SLF001
        text = str(asyncio.run(tool.run({})))
        assert "unavailable" in text and "dataworks" in text


def test_the_SERVER_INSTRUCTIONS_name_the_034_shapes():
    """FR-30. The instructions are the first thing an agent reads; they must name the dataworks
    unavailable shape beside miLLM's, the pending-approval answer and its poll tool, and the
    inference tools' `refused` shape. Review round 2026-10-16: rewording the dataworks sentence
    SURVIVED every test until this one. Negative control: the same rewording → red."""
    from src.mcp_server.server import _server_instructions

    text = _server_instructions()
    assert '{"unavailable": "dataworks", "reason": …}' in text
    assert '{"unavailable": "millm", "reason": …}' in text
    assert "dataworks_get_approval_status" in text and "never re-submit" in text
    assert '"refused": "model_not_resident"' in text and "millm_load_model" in text
