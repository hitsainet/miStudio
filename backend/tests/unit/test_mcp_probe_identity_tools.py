"""Feature 039's MCP arguments travel, and the identity fields reach the agent unchanged (FTASKS 6.1, 6.2).

Through the same harness as `test_reachability.py` (every module registered on one recording client):
the digest filters of `list_probe_monitors` and `expect_decision_digest` of `recalibrate_probe_monitor`
are asserted by PAYLOAD and CALL COUNT; a preview's identity fields are returned to the agent as the
backend sent them.
"""
from __future__ import annotations

import asyncio

import pytest

from tests.unit.test_reachability import _tools_by_name

D = "d" * 64
E = "e" * 64


def _call(tool, kwargs):
    manager, client = _tools_by_name()
    result = asyncio.run(manager.call_tool(tool, kwargs, context=None))
    return client.calls, result


def test_the_digest_filters_travel_as_query_parameters():
    calls, _ = _call("list_probe_monitors", {"detector_digest": D, "decision_digest": E})
    assert calls == [("GET", "/probe-monitors/probes",
                      {"run_id": None, "selected_only": None, "detector_digest": D,
                       "decision_digest": E})]


def test_expect_decision_digest_travels_in_the_body_only_when_given():
    calls, _ = _call("recalibrate_probe_monitor",
                     {"probe_id": "pm_1", "target_fpr": 0.005, "preview": False,
                      "expect_decision_digest": E})
    assert calls == [("POST", "/probe-monitors/probes/pm_1/recalibrate", {"json_body": {
        "target_fpr": 0.005, "preview": False, "reason": "", "allow_fire_on_nothing": False,
        "expect_decision_digest": E}})]


@pytest.mark.parametrize("bad", ["E" * 64, "e" * 63, "not-a-digest"])
def test_a_malformed_digest_is_refused_before_any_call(bad):
    manager, client = _tools_by_name()
    with pytest.raises(Exception):
        asyncio.run(manager.call_tool("list_probe_monitors", {"decision_digest": bad}, context=None))
    assert client.calls == []


def test_the_preview_identity_reaches_the_agent_unchanged(monkeypatch):
    """FR-10 through MCP: the tool returns the backend's preview verbatim — the identity fields are
    not dropped or renamed on the way to the agent."""
    preview = {
        "probe_id": "pm_1",
        "current": {"threshold": 1.0, "revision": 1, "decision_digest": E, "decision_revision": 1},
        "proposed": {"threshold": 2.0, "revision": 2, "decision_digest": "f" * 64,
                     "decision_revision": 2},
        "detector_digest": D, "detector_unchanged": True, "unchanged": False,
        "committed": False,
    }
    manager, client = _tools_by_name()

    async def post(path, json_body=None, **_k):
        client.calls.append(("POST", path, {"json_body": json_body}))
        return preview

    monkeypatch.setattr(client, "post", post)
    result = asyncio.run(manager.call_tool(
        "recalibrate_probe_monitor", {"probe_id": "pm_1", "target_fpr": 0.005}, context=None))
    text = str(result)
    for value in (E, "f" * 64, D, "detector_unchanged", "decision_revision", "unchanged"):
        assert value in text, value
    assert len(client.calls) == 1
