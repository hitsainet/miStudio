"""The judge gate fails CLOSED (042 FR-042.38 P0 half; FTID §3.7; CF-9, CF-25 m14; TPD-12, TPD-30; FTASKS 4.1).

A judge run (or a spot-check) on miServer's card queues every live chat behind thousands of generations and
may evict the served model, so the gate errs towards asking the operator. The old `judge_endpoint_is_millm`
matched only `"millm" in host` and let every host-less endpoint through (FTDD §17 C8, C14).

The resolver is stubbed at module level (`probe_monitors._resolve`) and its calls are asserted (count and
arguments); the tool cases go through the REAL tools registered on an MCPServer.

Controls K-4.5 – K-4.10.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.mcp_server.config import MCPSettings
from src.mcp_server.tools import probe_monitors as pm
from src.mcp_server.tools.probe_monitors import judge_endpoint_is_miserver

CONFIGURED = "http://gpu-box.lan:8000"


class Resolver:
    """A stand-in for `socket.getaddrinfo`: a fixed table; records every call."""

    def __init__(self, table=None, raises=None, delay=0.0):
        self.table = table or {}
        self.raises = raises
        self.delay = delay
        self.calls: list[tuple] = []

    def __call__(self, host, port):
        import time

        self.calls.append((host, port))
        if self.delay:
            time.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        return [(2, 1, 6, "", (a, 0)) for a in self.table.get(host, ["198.51.100.99"])]


@pytest.fixture
def resolver(monkeypatch):
    r = Resolver({"gpu-box.lan": ["10.0.0.5"], "10.0.0.5": ["10.0.0.5"], "10.0.0.9": ["10.0.0.9"],
                  "api.example.org": ["203.0.113.7"]})
    monkeypatch.setattr(pm, "_resolve", r)
    return r


def _gate(endpoint, configured=CONFIGURED):
    return asyncio.run(judge_endpoint_is_miserver(endpoint, SimpleNamespace(miserver_api_url=configured)))


@pytest.mark.parametrize("endpoint,configured", [
    ("http://k8s-millm.hitsai.local/v1", ""),
    ("http://k8s-miserver.hitsai.local/v1", ""),
    ("http://miserver-backend.miserver.svc:8000/v1", ""),
    ("http://gpu-box.lan:8000/v1", CONFIGURED),
    ("http://gpu-box2.lan:8000/v1", "http://gpu-box2.lan:8000"),
], ids=["k8s-millm", "k8s-miserver", "miserver-backend.miserver.svc", "configured-new", "configured-old"])
def test_stems_and_configured_hosts_are_gated(endpoint, configured, resolver):
    assert _gate(endpoint, configured) == (True, None)
    assert resolver.calls == [], "a name match needs no lookup"


def test_the_retired_old_variable_configures_nothing(monkeypatch, resolver):
    """ENH-009 P7: an address set only under the OLD name (MILLM_API_URL, retired) is not configuration any more,
    so the gate has no configured address and falls back to name stems, saying so (until P7 it gated)."""
    monkeypatch.delenv("MISERVER_API_URL", raising=False)
    monkeypatch.setenv("MILLM_API_URL", "http://gpu-box2.lan:8000")
    settings = MCPSettings(_env_file=None)
    assert asyncio.run(judge_endpoint_is_miserver("http://gpu-box2.lan:8000/v1", settings)) == \
        (False, "name_stem_only")
    assert resolver.calls == []


def test_unrelated_host_not_gated(resolver):
    assert _gate("https://api.example.org/v1") == (False, None)
    assert resolver.calls == [("api.example.org", None), ("gpu-box.lan", None)]


@pytest.mark.parametrize("endpoint", ["", "not a url", "miserver-backend:8000/v1"],
                         ids=["empty", "not-a-url", "scheme-less-miserver-backend"])
def test_hostless_endpoint_is_gated(endpoint, resolver):
    assert _gate(endpoint) == (True, None)
    assert _gate(endpoint, configured="") == (True, None)
    assert resolver.calls == []


def test_ip_literal_resolving_to_configured_is_gated(resolver):
    assert _gate("http://10.0.0.5:8000/v1") == (True, None)


def test_unrelated_ip_not_gated(resolver):
    assert _gate("http://10.0.0.9:8000/v1") == (False, None)


@pytest.mark.parametrize("failure", ["raises", "timeout"])
def test_lookup_failure_or_timeout_is_gated(failure, monkeypatch):
    if failure == "raises":
        r = Resolver(raises=OSError("Name or service not known"))
    else:
        monkeypatch.setattr(pm, "JUDGE_RESOLVE_TIMEOUT_S", 0.05)
        r = Resolver(delay=0.5)
    monkeypatch.setattr(pm, "_resolve", r)
    assert _gate("https://api.example.org/v1") == (True, None)


def test_resolve_called_with_both_hosts(resolver):
    _gate("http://10.0.0.9:8000/v1")
    assert resolver.calls == [("10.0.0.9", None), ("gpu-box.lan", None)], resolver.calls


def test_standalone_ip_not_gated_and_says_name_stem_only(resolver):
    assert _gate("http://10.0.0.5:8000/v1", configured="") == (False, "name_stem_only")
    assert resolver.calls == [], "_resolve must not be called with no address configured"


# ── through the two real tools ─────────────────────────────────────────────────────────────────────────
class _Client:
    def __init__(self):
        self.calls = []

    async def post(self, path, json_body=None, **params):
        self.calls.append(("POST", path, json_body))
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        return {"id": "job_1", "status": "queued"}


TOOL_ARGS = {
    "start_probe_judge_run": {"endpoint": "http://10.0.0.5:8000/v1", "model": "qwen", "dataset_ids": ["pmd_2"],
                              "probe_id": "pm_1", "prompt_id": "jp_1", "max_rows_per_set": 10},
    "start_probe_spot_check": {"behaviour": "pii", "prompt_id": "jp_1", "endpoint": "http://10.0.0.5:8000/v1",
                               "model": "qwen"},
}


def _tool(name, args, *, configured=CONFIGURED):
    from mcp.server.mcpserver import MCPServer

    client = _Client()
    mcp = MCPServer("t")
    pm.register(mcp, client, MCPSettings(auth_token="x" * 32, probe_gpu_approval=True,
                                         miserver_api_url_override=configured, _env_file=None))
    result = asyncio.run(mcp._tool_manager._tools[name].fn(**args))  # noqa: SLF001
    return result, client.calls


@pytest.mark.parametrize("name", sorted(TOOL_ARGS))
def test_through_both_real_tools(name, resolver, monkeypatch):
    # The configured address is the override below, never one another test left in the environment.
    monkeypatch.delenv("MISERVER_API_URL", raising=False)
    monkeypatch.delenv("MILLM_API_URL", raising=False)
    result, calls = _tool(name, TOOL_ARGS[name])
    assert result["status"] == "pending_approval", result
    assert [c[1] for c in calls] == ["/mcp/approvals"], "a gated judge submits NOTHING before approval"
    # standalone: not gated, submitted once, and the result says the gate checked names only
    result, calls = _tool(name, TOOL_ARGS[name], configured="")
    assert len(calls) == 1 and calls[0][1] != "/mcp/approvals", calls
    assert calls[0][2]["endpoint"] == "http://10.0.0.5:8000/v1", "the endpoint is forwarded unchanged"
    assert result["judge_gate"] == "name_stem_only" and "not checked" in result["judge_gate_reason"]
