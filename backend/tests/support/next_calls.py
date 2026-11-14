"""Check a `next` list (Feature 039 D-T1, BR-030) against the LIVE MCP registry.

Every entry must name a tool the REAL built server registers (every category on, every product URL
set — `mcp_route_map.registered_tools`'s recipe), and its `arguments` must validate against that
tool's own `inputSchema` with `jsonschema`. A `needs` list (arguments only the agent can supply)
must name real properties of that schema and must NOT be filled in `arguments`.

Read from `list_tools()`, never from the source: a renamed tool or argument then fails here rather
than in an agent (memory: derive-from-the-registry-not-the-source).
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable

import jsonschema
import pytest

_SCHEMAS: Dict[str, Dict[str, Any]] = {}


def live_input_schemas() -> Dict[str, Dict[str, Any]]:
    """`{tool name: inputSchema}` from the real built server (cached per process)."""
    if _SCHEMAS:
        return _SCHEMAS
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    mp = pytest.MonkeyPatch()
    try:
        for spec in PRODUCTS:
            mp.setenv(spec.url_env, f"http://{spec.name}.test")
        settings = MCPSettings(
            tool_categories=",".join(sorted(all_categories())), allow_anonymous=True
        )
        mcp, _client = build_server(settings, stdio=True)
        # `asyncio.run` cannot nest inside an async test's running loop: list in a worker thread.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            tools = pool.submit(asyncio.run, mcp.list_tools()).result()
    finally:
        mp.undo()
    _SCHEMAS.update({tool.name: dict(getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}) for tool in tools})
    return _SCHEMAS


def assert_next_calls_are_live(next_list: Iterable[Dict[str, Any]]) -> None:
    """Every entry names a registered tool and its arguments validate against that tool."""
    schemas = live_input_schemas()
    for entry in next_list:
        assert set(entry) <= {"tool", "arguments", "why", "needs"}, entry
        tool = entry["tool"]
        assert tool in schemas, f"`next` names {tool!r}, which the live MCP registry does not have"
        assert isinstance(entry.get("why"), str) and entry["why"].strip(), entry
        schema = schemas[tool]
        # A REQUIRED argument the agent must supply (Feature 041: `register_probe_dataset` needs the
        # agent's own data and columns) is named in `needs` and validated as supplied — never filled
        # in. Every other required argument must still be present.
        needed = set(entry.get("needs") or [])
        if needed and schema.get("required"):
            schema = {**schema, "required": [r for r in schema["required"] if r not in needed]}
        try:
            jsonschema.validate(entry["arguments"], schema)
        except jsonschema.ValidationError as exc:  # pragma: no cover - the failure message
            raise AssertionError(
                f"`next` call {tool}({entry['arguments']}) does not validate against the tool's "
                f"live input schema: {exc.message}"
            ) from exc
        properties = schema.get("properties") or {}
        # A tool's schema need not forbid extra keys, so a RENAMED argument would still validate;
        # every argument must be a property the live tool declares.
        unknown = sorted(set(entry["arguments"]) - set(properties))
        assert not unknown, f"`next` passes {unknown} to {tool}, which takes no such argument"
        for name in entry.get("needs") or []:
            assert name in properties, f"`needs` names {name!r}, which {tool} does not take"
            assert name not in entry["arguments"], (
                f"{name!r} is listed as needed from the agent AND filled in — an acknowledgement "
                "nobody gave"
            )
