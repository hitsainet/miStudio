"""`millm_recalibrate_probe` is a thin HTTP push, and both halves of its payload are load-bearing.

⚠ **THE FIRST VERSION OF THIS TOOL DID THE RE-CUT ITSELF, AND COULD NOT HAVE RUN.**

It imported `src.services.probe_recalibration` and opened a `SyncSessionLocal`. The MCP server is
a SEPARATE deployment with no `DATABASE_URL` — `src.core.config.Settings` fails there with six
missing fields — so the tool was registered in all four layers, payload-asserted, present in the
live registry of the deployed server, and would have raised on its first real call. Every check
passed because every check ran in a process that happened to have a database.

`test_mcp_tools_stay_on_http.py` is the guard that now makes that impossible. This file asserts
what the tool does instead: one POST, carrying a decision assembled from the proposal that
`recalibrate_probe_monitor` returns.

Why two tools rather than one convenient call: the MCP server can only reach miLLM. It is also the
right shape — a bar should not move without somebody reading what the candidate does to every
evaluation set, which is what the miStudio proposal carries.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.tools import millm_probes as probe_tools

MILLM_ID = "pr_abc123"
MISTUDIO_ID = "pm_36d1a65f7953"

#: Straight out of a `recalibrate_probe_monitor` proposal's `proposed` block.
PROPOSED = {
    "threshold": 14.5201,
    "target_fpr": 0.005,
    "realised_fpr": 0.005,
    "threshold_source": "calibration_set",
}


def _register():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test")
    client = MagicMock()
    for verb in ("get", "post", "put", "delete"):
        setattr(client, verb, AsyncMock(return_value={}))
    gate = MagicMock()
    # ⚠ `(ok, reason)`, NOT `None`. `@gated` unpacks the result, so a stand-in returning None
    # raises inside the decorator and every test fails before reaching the tool.
    gate.check = AsyncMock(return_value=(True, None))
    probe_tools.register(mcp, client, gate)
    return mcp, client


def _call(**over):
    mcp, client = _register()
    try:
        fn = asyncio.run(mcp.get_tool("millm_recalibrate_probe")).fn
    except Exception:
        fn = mcp._tool_manager._tools["millm_recalibrate_probe"].fn
    kwargs = {
        "probe_id": MILLM_ID,
        "mistudio_probe_id": MISTUDIO_ID,
        "reason": "tighter budget",
        **PROPOSED,
    }
    kwargs.update(over)
    asyncio.run(fn(**kwargs))
    return client


class TestItIsRegisteredAndStaysOnHttp:
    def test_it_is_in_the_built_server(self):
        mcp, _client = _register()
        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert "millm_recalibrate_probe" in names

    def test_it_issues_exactly_ONE_call(self):
        """The re-cut happens in miStudio, through its own tool. This is the push, and nothing
        else — a second call here would mean the tool had grown a second responsibility."""
        client = _call()
        assert client.post.await_count == 1
        assert client.get.await_count == 0

    def test_it_posts_to_the_probes_own_path(self):
        client = _call()
        assert client.post.await_args.args[0] == f"/api/probes/{MILLM_ID}/recalibrate"

    def test_it_uses_json_body_and_NOT_a_field_literally_named_params(self):
        """⚠ THE DEFECT THAT SHIPPED IN THIS CATEGORY BEFORE. `MiLLMClient.post` takes
        `json_body`; `params={...}` would send a query field named "params", and a path-only
        assertion could not tell."""
        client = _call()
        assert "json_body" in client.post.await_args.kwargs
        assert "params" not in client.post.await_args.kwargs


class TestThePayloadIsTheWholeBar:
    def test_the_decision_carries_every_field_miLLM_refuses_without(self):
        """A threshold with no budget and no named source is refused by miLLM — a number nobody
        calibrated is not a threshold. A tool that dropped either field would make every push a
        409 that looks like a server problem."""
        body = _call().post.await_args.kwargs["json_body"]
        assert body["decision"] == PROPOSED
        assert body["mistudio_probe_id"] == MISTUDIO_ID
        assert body["reason"] == "tighter budget"

    def test_the_MISTUDIO_id_travels_and_is_not_confused_with_miLLMs(self):
        """⚠ THE IDS DIFFER ON THE TWO SIDES and miLLM compares the cut against the stored
        `provenance.probe_id`. An earlier version of this tool passed miLLM's `pr_…` to the
        miStudio re-cut, which would have asked for a probe it has never heard of."""
        body = _call().post.await_args.kwargs["json_body"]
        assert body["mistudio_probe_id"].startswith("pm_")
        assert body["mistudio_probe_id"] != MILLM_ID

    def test_per_window_and_per_length_bars_travel_when_given(self):
        """⚠ BOTH TAKE PRECEDENCE OVER THE GLOBAL THRESHOLD. Pushing only the global would move
        the number miLLM stores and leave every verdict judged against the old per-window bar —
        a success report over a no-op."""
        windows = {"prompt": {"threshold": 7.0}}
        bands = [{"min_tokens": 0, "max_tokens": None, "threshold": 9.0}]
        body = _call(windows=windows, length_bands=bands).post.await_args.kwargs["json_body"]
        assert body["decision"]["windows"] == windows
        assert body["decision"]["length_bands"] == bands

    def test_absent_window_and_length_blocks_are_OMITTED_not_sent_as_null(self):
        """miLLM's parsers treat an absent block and an empty one differently. Sending `None` for
        windows a probe HAS would read as "it has none" and silently drop its per-window bars."""
        decision = _call().post.await_args.kwargs["json_body"]["decision"]
        assert "windows" not in decision
        assert "length_bands" not in decision

    def test_a_NULL_threshold_is_passed_through_as_null(self):
        """`threshold=None` means the probe ranks but does not decide. It is a real operating
        point and must not be coerced to 0.0 — the NULL-is-not-zero confusion the probe model
        warns about."""
        decision = _call(
            threshold=None, target_fpr=None, realised_fpr=None, threshold_source=None
        ).post.await_args.kwargs["json_body"]["decision"]
        assert decision["threshold"] is None

    def test_the_calibration_id_travels_when_given_and_is_null_otherwise(self):
        assert _call().post.await_args.kwargs["json_body"]["calibration_id"] is None
        body = _call(calibration_id="pmd_3222ea3bc591").post.await_args.kwargs["json_body"]
        assert body["calibration_id"] == "pmd_3222ea3bc591"
