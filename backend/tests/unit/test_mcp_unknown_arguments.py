"""Every tool on this MCP server refuses an argument name it does not declare (MCP gap, 2026-11-10).

Found live: `start_probe_training_run` called with `layers`, `rules`, `rolling_windows`, `calibration_max_rows`,
`sae_k`, `seed`, `max_length`, `target_fpr`, `top_n_layers` and `val_fraction` at the TOP level (they belong inside
`config`) returned success and queued a GPU run with DEFAULT settings: the SDK's argument model ignores names it
does not know, so the agent's settings vanished without a word. And `get_task_status(task_id=…)` answered the
25-row task list because `task_id` was dropped the same way.

SERVER-WIDE, NOT PER TOOL: `server.refuse_unknown_arguments` wraps the tool manager's `call_tool` (the one entry
every `tools/call` goes through) and, before anything runs, refuses a name that is not a property of the tool's
advertised `inputSchema`, naming the unknown names, the accepted ones, and — when an unknown name is a field of
an object parameter — where it belongs (`config.layers`).

Read through the LIVE registry (every category on, every product URL set). Wiring control: delete the
`refuse_unknown_arguments(mcp)` call in `build_server` → `test_every_registered_tool_refuses_an_unknown_name` red.
"""
from __future__ import annotations

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools.base import Tool

UNKNOWN = "definitely_not_an_argument"
TOP_LEVEL_CONFIG = ["layers", "rules", "rolling_windows", "calibration_max_rows", "sae_k", "seed", "max_length",
                    "target_fpr", "top_n_layers", "val_fraction"]


@pytest.fixture
def built(monkeypatch):
    """`(mcp, runs)`: the real build, with `Tool.run` recording instead of running (nothing may run)."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    for spec in PRODUCTS:
        monkeypatch.setenv(spec.url_env, f"http://{spec.name}.test")
    runs: list[tuple[str, dict]] = []

    async def recording_run(self, arguments, context, convert_result=False):
        runs.append((self.name, dict(arguments)))
        return {"ran": self.name}

    monkeypatch.setattr(Tool, "run", recording_run)
    settings = MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True,
                           _env_file=None)
    mcp, _ = build_server(settings, stdio=True)
    return mcp, runs


def _call(mcp, name, arguments):
    return asyncio.run(mcp._tool_manager.call_tool(name, arguments, context=None))  # noqa: SLF001


def test_every_registered_tool_refuses_an_unknown_name(built):
    mcp, runs = built
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) > 300, "the live registry is nearly empty: the check would read nothing"
    for tool in tools:
        accepted = sorted((tool.input_schema or {}).get("properties", {}))
        with pytest.raises(ToolError) as info:
            _call(mcp, tool.name, {UNKNOWN: 1})
        message = str(info.value)
        assert f"unknown argument: {UNKNOWN}" in message, (tool.name, message)
        assert "Nothing was run" in message, (tool.name, message)
        for name in accepted:
            assert f"`{name}`" in message, (tool.name, name, message)
        if not accepted:
            assert "takes no arguments" in message, (tool.name, message)
    assert runs == [], f"a tool ran with an unknown argument: {runs[:3]}"


def test_the_reported_case_is_refused_and_points_into_config(built):
    mcp, runs = built
    args = {"probe_dataset_id": "pd_1", **{k: 1 for k in TOP_LEVEL_CONFIG}}
    with pytest.raises(ToolError) as info:
        _call(mcp, "start_probe_training_run", args)
    message = str(info.value)
    for key in TOP_LEVEL_CONFIG:
        assert f"`config.{key}`" in message, (key, message)
    assert runs == []


def test_a_misspelt_name_is_refused_naming_what_the_tool_accepts(built):
    """The shape of the live `get_task_status(task_id=…)` miss: the call answered the 25-row list."""
    mcp, runs = built
    with pytest.raises(ToolError) as info:
        _call(mcp, "get_task_status", {"taskid": "959734fc-0000-4000-8000-000000000000", "bogus": 2})
    message = str(info.value)
    assert "unknown arguments: bogus, taskid" in message
    assert "`task_queue_id`" in message and "`list_active`" in message
    assert runs == []


def test_declared_arguments_still_reach_the_tool_once(built):
    mcp, runs = built
    assert _call(mcp, "get_task_status", {"list_active": True}) == {"ran": "get_task_status"}
    assert _call(mcp, "miworkbench_howto", {}) == {"ran": "miworkbench_howto"}
    assert runs == [("get_task_status", {"list_active": True}), ("miworkbench_howto", {})]


def test_the_wrapper_is_applied_once():
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server, refuse_unknown_arguments

    mcp, _ = build_server(MCPSettings(tool_categories="read", allow_anonymous=True, _env_file=None), stdio=True)
    assert getattr(mcp._tool_manager.call_tool, "refuses_unknown_arguments", False) is True  # noqa: SLF001
    assert refuse_unknown_arguments(mcp) == 0, "a second pass must not wrap the wrapper"
