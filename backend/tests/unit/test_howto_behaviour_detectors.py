"""The `behaviour_detectors` how-to topic names only live tools (Feature 040 S2 FR-42; FTASKS 12.5)."""
from __future__ import annotations

import re

from src.mcp_server.tools import howto
from tests.support.next_calls import live_input_schemas


def named_tools(text: str) -> set:
    return set(re.findall(r"`([a-z][a-z0-9_]+)`", text))


def test_the_topic_exists_and_names_only_live_tools():
    text = howto.TOPICS["behaviour_detectors"]
    tools = named_tools(text) - {"base_version", "behaviour", "group_columns", "sub_question", "next", "stale"}
    assert len(tools) >= 10, tools
    live = set(live_input_schemas())
    assert tools <= live, sorted(tools - live)


def test_a_removed_tool_still_named_is_caught():
    live = set(live_input_schemas())
    assert not (named_tools("use `get_probe_breakdownx`") <= live)
