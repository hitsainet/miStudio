"""Every MCP tool that pages with `offset` reaches its live route's paging (review R2D-2).

`list_trainings` sent `offset` to `/trainings`, which pages by `page`, and `list_experiments`
sent `offset` to `/steering/experiments`, which reads `skip`. FastAPI drops query parameters a
route does not declare, so every page an agent asked for was the first one, with no error.

The tools come from the REAL `build_server()` (review R3-D, R3D-14: the first version registered
four hand-listed modules, which proved the modules and said nothing about what the server
exposes), and the routes from the live `app.openapi()`. Calls are recorded through the server's
own client object, so the payload is what an agent's call would send.

MUTATION CONTROLS (integration, 2026-09-15; each line broken alone, this file run, bytes
restored and sha256 verified):
  MP-1 list_trainings sends offset again              -> its paging case red
  MP-2 list_experiments sends offset again            -> its paging case red
  MP-3 list_trainings keeps the rows before offset     -> the mid-page offset test red
  MP-4 the experiments module is dropped from its MCP category -> the coverage test red
"""

import asyncio
import inspect
import re

import pytest

#: Values for the required arguments of the paging tools; a new required argument fails loudly.
REQUIRED_ARGUMENTS = {"extraction_id": "extr_test", "token": "hello"}
#: The categories that expose the paging tools.
PAGING_CATEGORIES = "read,groups,experiments"


def _built_server(monkeypatch, rows=None):
    """The REAL server, with its client's reads recorded instead of sent."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    mcp, client = build_server(MCPSettings(tool_categories=PAGING_CATEGORIES, allow_anonymous=True), stdio=True)
    calls = []

    async def get(path, **params):
        # As the real client does: None values are not sent.
        calls.append(("GET", path, {k: v for k, v in params.items() if v is not None}))
        return {"data": list(rows or [])}

    async def post(path, *args, **kwargs):
        calls.append(("POST", path, kwargs))
        return {}

    monkeypatch.setattr(client, "get", get)
    monkeypatch.setattr(client, "post", post)
    return mcp, calls


def _tools(mcp):
    """Name -> the registered function, for every tool the built server lists."""
    listed = {tool.name for tool in asyncio.run(mcp.list_tools())}
    registered = mcp._tool_manager._tools
    return {name: registered[name].fn for name in listed}


def _offset_tools(monkeypatch):
    mcp, _ = _built_server(monkeypatch)
    return sorted(name for name, fn in _tools(mcp).items() if "offset" in inspect.signature(fn).parameters)


def _live_get_operation(path):
    """The live route a concrete path resolves to, preferring literal segments over templates."""
    from src.main import app

    candidates = []
    for template, operations in app.openapi()["paths"].items():
        pattern = "^" + re.sub(r"\{[^}]+\}", "[^/]+", template) + "$"
        if "get" in operations and re.match(pattern, "/api/v1" + path):
            candidates.append((template.count("{"), template, operations["get"]))
    assert candidates, f"no live GET route for /api/v1{path}"
    _, template, operation = min(candidates)
    return template, operation


OFFSET_TOOLS = [
    "find_features_by_token", "get_feature_groups", "list_experiments",
    "list_extractions", "list_trainings", "search_features",
]


def test_the_built_server_exposes_exactly_these_offset_paging_tools(monkeypatch):
    assert _offset_tools(monkeypatch) == OFFSET_TOOLS


@pytest.mark.parametrize("tool", OFFSET_TOOLS)
def test_an_offset_tool_sends_only_what_its_route_declares_and_the_page_arrives(tool, monkeypatch):
    mcp, calls = _built_server(monkeypatch)
    fn = _tools(mcp)[tool]
    arguments = {
        name: REQUIRED_ARGUMENTS[name]
        for name, parameter in inspect.signature(fn).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    arguments.update(limit=10, offset=20)
    asyncio.run(fn(**arguments))

    gets = [call for call in calls if call[0] == "GET"]
    assert len(gets) == 1, calls
    _, path, sent = gets[0]
    template, operation = _live_get_operation(path)
    declared = {p["name"] for p in operation.get("parameters", []) if p.get("in") == "query"}
    assert set(sent) <= declared, (
        f"{tool} sends {sorted(set(sent) - declared)} to GET {template}, which declares only "
        f"{sorted(declared)}: FastAPI drops the rest, so the agent's paging is ignored"
    )
    assert sent.get("limit") == 10, sent
    paging = {key: sent[key] for key in ("offset", "skip", "page") if key in sent}
    assert paging in ({"offset": 20}, {"skip": 20}, {"page": 3}), (tool, sent)


def test_a_mid_page_offset_on_trainings_returns_the_rows_from_that_offset(monkeypatch):
    page_three = [{"id": f"train_{i}"} for i in range(20, 30)]
    mcp, calls = _built_server(monkeypatch, rows=page_three)
    fn = _tools(mcp)["list_trainings"]

    result = asyncio.run(fn(limit=10, offset=25))

    assert calls == [("GET", "/trainings", {"limit": 10, "page": 3})]
    assert [row["id"] for row in result["data"]] == [f"train_{i}" for i in range(25, 30)]
