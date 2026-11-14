"""Both ids of this app's MCP server classify as an agent (042 FR-042.28, DR-17; FTASKS 12.23, control K-12.42).

Stored rows keep `agent:mistudio-mcp` forever (DR-17); rows written since the P3 cut carry `agent:miworkbench-mcp`.
The backend's classifier (`judge_prompt_library._source_for`, which sets a judge-prompt version's `source`) must
call both "agent", and a person "operator". The route-level twin is `test_frozen_identifiers.py::test_dr17_…`.
"""
from __future__ import annotations

from src.mcp_server.tools.judge_prompts import AGENT_ACTOR
from src.services.judge_prompt_library import _source_for
from tests.support import legacy_export


def test_both_ids_classify_as_agent():
    (old,) = next(r["literals"] for r in legacy_export.load().entries() if r["id"] == "DR-17")
    assert old == "agent:mistudio-mcp"
    assert AGENT_ACTOR == "agent:miworkbench-mcp"
    for actor in (old, AGENT_ACTOR, "agent:miforge-mcp"):
        assert _source_for(actor) == "agent", actor
    for person in ("sean", "operator", "mistudio-mcp"):
        assert _source_for(person) == "operator", person
