"""Operator code reaches miLLM only behind the operator — in miLLM since ENH-008 (miLLM 039 FTASKS 19.1; §4l; D1).

`millm_import_rule_set` proxies `POST /api/rule-sets/import`. A set that names OPERATOR CODE — a `code_rule` signal,
or miLLM's envelope `{rule_set, code_modules}` — used to wait in miStudio's queue (keyed by the body's content hash)
and be consumed on a re-call. ENH-008 moved the decision into miLLM: the body goes to miLLM with `X-miLLM-Agent`,
miLLM stores it (its `request_digest` covers the exact body) and answers pending, and the tool relays that answer.
The relay, the fail-closed queue check and the deprecated `approval_request_id` are asserted for this tool beside the
other five in `test_millm_approvals_relay.py`, through the live registry.

What stays here: `names_operator_code` decides which bodies need the queue (and fails closed on shapes it did not
foresee), the pod-side hash is still miStudio's rule-set identity, the pod's registry carries the tool, and a
declarative set is one request with miLLM's outages answered `unavailable`.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.millm_client import MiLLMResponse
from tests.support.external_ctx import make_ctx

BACKEND = Path(__file__).resolve().parents[2]
CODE_SHA = "c" * 64
OTHER_SHA = "e" * 64
IMPORT = "/api/rule-sets/import"


def _declarative() -> dict[str, Any]:
    return {"format": "midataworks.rule-set/v1", "name": "decl", "behaviour_key": "<behaviour-key>",
            "description": "d", "signals": [{"id": "p", "kind": "pattern", "pattern": "x", "reads": "text"}],
            "expressions": {"hit": {"signal": "p"}},
            "outputs": [{"column": "hit", "type": "bool", "value": {"fired": "hit"}}], "cases": []}


def _with_code() -> dict[str, Any]:
    doc = _declarative()
    doc["name"] = "code"
    doc["signals"].append({"id": "c", "kind": "code_rule", "code_sha256": CODE_SHA, "function": "f",
                           "runner_contract": "midataworks.runner/v1", "params": {}, "outputs": [{"name": "count"}]})
    doc["expressions"]["hit"] = {"any": [{"signal": "p"}, {"signal": "c"}]}
    return doc


def _envelope(source: str = "def f(text, params):\n    return 0\n") -> dict[str, Any]:
    return {"rule_set": _with_code(), "code_modules": {CODE_SHA: source}}


# ── the tool against recorders ────────────────────────────────────────────────────────────────────────────────────


class _Recorder:
    def __init__(self, log: list, name: str, responses: dict):
        self.log, self.name, self.responses = log, name, responses

    async def post(self, path, json_body=None, *, headers=None, full=False, **params):
        self.log.append((self.name, "POST", path, json_body, params))
        answer = self.responses.get(path, {})
        if isinstance(answer, Exception):
            raise answer
        return MiLLMResponse(200, answer, {}) if full else answer


def _call(args: dict, *, gate_open: bool = True):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import millm_behaviours

    log: list = []
    mistudio = _Recorder(log, "mistudio", {"/mcp/approvals": {"id": "ap_9"}})
    millm = _Recorder(log, "millm", {IMPORT: {"id": "rs_000000000001"}})
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, "ok") if gate_open else (False, "connection refused"))
    mcp = MCPServer("t")
    millm_behaviours.register(mcp, millm, gate, make_ctx(mistudio))
    return asyncio.run(mcp._tool_manager._tools["millm_import_rule_set"].fn(**args)), log  # noqa: SLF001


def test_a_declarative_set_is_ONE_import_request_and_no_approval():
    result, log = _call({"document": _declarative(), "version": 2, "origin": "authored"})
    assert log == [("millm", "POST", IMPORT, _declarative(),
                    {"version": 2, "digest": None, "origin": "authored"})]
    assert result == {"id": "rs_000000000001"}


def test_the_detection_fails_closed_on_a_shape_it_did_not_foresee():
    from src.mcp_server.tools.millm_behaviours import names_operator_code

    assert not names_operator_code(_declarative())
    nested = _declarative()
    nested["extra"] = [{"deep": {"kind": "code_rule"}}]
    assert names_operator_code(nested)
    assert names_operator_code({"rule_set": _declarative(), "code_modules": {}})


def test_the_pod_side_hash_is_miStudios_rule_set_identity():
    """The MCP pod hashes with `rfc8785` directly (it cannot import the backend); the digest must be the one
    `src/rule_signals/canonical.py` and the vendored identity cases give."""
    from src.mcp_server.tools.millm_behaviours import content_sha256
    from src.rule_signals.canonical import identity

    cases = json.loads((BACKEND / "src/schemas/vendored/rule-signal-cases-v1.json").read_text())["rule_set_identity"]
    checked = 0
    for case in cases:
        if isinstance(case.get("document"), dict) and case.get("sha256"):
            assert content_sha256(case["document"]) == case["sha256"] == identity(case["document"]), case["id"]
            checked += 1
    assert checked >= 3
    assert content_sha256(_envelope()) == identity(_envelope())


# ── the pod's registry ────────────────────────────────────────────────────────────────────────────────────────────

PROBE = """
import asyncio, json
from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server
mcp, _ = build_server(MCPSettings(), stdio=True)
tool = next(t for t in asyncio.run(mcp.list_tools()) if t.name == 'millm_import_rule_set')
print('PARAMS=' + json.dumps(sorted(tool.input_schema['properties'])))
"""


def _manifest_categories() -> str:
    import yaml

    for doc in yaml.safe_load_all((BACKEND.parent / "k8s" / "base" / "mcp.yaml").read_text()):
        for container in (((doc or {}).get("spec") or {}).get("template") or {}).get("spec", {}).get("containers", []):
            for e in container.get("env", []):
                if e.get("name") == "MCP_TOOL_CATEGORIES":
                    return e["value"]
    raise AssertionError("no MCP_TOOL_CATEGORIES in k8s/base/mcp.yaml")


def test_the_pods_registry_carries_the_gated_import():
    keep = {"PATH", "HOME", "LANG", "VIRTUAL_ENV", "TMPDIR"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(MCP_AUTH_TOKEN="x" * 32, PYTHONPATH=str(BACKEND), MCP_ALLOW_ANONYMOUS="true",
               MCP_TOOL_CATEGORIES=_manifest_categories(), MILLM_API_URL="http://127.0.0.1:9")
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=BACKEND, env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    params = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("PARAMS="))[len("PARAMS="):])
    assert params == ["approval_request_id", "digest", "document", "origin", "version"]


# ── Review R1 (19.1): paths the build's tests did not reach ──────────────────────────────────────────────────────


def _call_with_outage(args: dict):
    """The tool with a healthy gate whose miLLM import then fails to connect (a mid-TTL outage)."""
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.client import BackendError
    from src.mcp_server.tools import millm_behaviours

    log: list = []
    mistudio = _Recorder(log, "mistudio", {"/mcp/approvals": {"id": "ap_9"}})
    millm = _Recorder(log, "millm", {IMPORT: BackendError(0, "miLLM backend unreachable: refused")})
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, "ok"))
    mcp = MCPServer("t")
    millm_behaviours.register(mcp, millm, gate, make_ctx(mistudio))
    result = asyncio.run(mcp._tool_manager._tools["millm_import_rule_set"].fn(**args))  # noqa: SLF001
    return result, log, gate


@pytest.mark.parametrize("args", [{"document": _declarative()}], ids=["declarative"])
def test_a_mid_call_outage_answers_unavailable_and_expires_the_gate(args):
    """R1-F1: the tool dropped `@gated` for its own branch, so it re-implements the mid-TTL outage answer. Without it
    the agent gets a raised error instead of `{"unavailable": "millm"}`, and the stale gate keeps saying ok."""
    result, log, gate = _call_with_outage(args)
    assert result == {"unavailable": "millm", "reason": "backend 0: miLLM backend unreachable: refused"}
    assert [e[2] for e in log if e[0] == "millm"] == [IMPORT]
    gate.invalidate.assert_called_once_with("millm")


def test_a_declarative_set_with_miLLM_down_sends_nothing():
    result, log = _call({"document": _declarative()}, gate_open=False)
    assert log == [] and result == {"unavailable": "millm", "reason": "connection refused"}
