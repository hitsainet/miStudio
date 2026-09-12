"""Every route an MCP tool calls must exist in the served API.

MIS-E2E-114 (adjacent). The tool `get_extraction_summary` issued
`GET /extractions/{id}`. That path served only DELETE, so every agent
invocation returned 405 — and `docs/mcp-contract.md`, the published description
of the MCP surface, advertised the GET as real.

This is the reachability rule pointed the other way. The existing harness asks
"is the tool registered with the server"; this asks "does the thing the tool
calls exist". A tool can be perfectly registered, unit-tested and documented
while calling a route nobody wrote.

Read from the SERVED OpenAPI schema. `app.routes` is not usable for this: the
build wraps included routers in `_IncludedRouter` objects carrying no `.path`,
so it reports 10 entries for a 230-path API and every check passes by finding
nothing to disagree with.
"""

import ast
import re
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "tools"

#: The MCP client prefixes every call with the API root.
API_ROOT = "/api/v1"


def _served() -> set:
    from src.main import app

    spec = app.openapi()
    return {
        (method.upper(), re.sub(r"\{[^}]+\}", "{}", path.rstrip("/")))
        for path, operations in spec["paths"].items()
        for method in operations
    }


def _tool_calls() -> list:
    """Every `client.<verb>("/path")` in the tool modules, with its location."""
    calls = []
    for module in sorted(TOOLS.glob("*.py")):
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("get", "post", "put", "patch", "delete"):
                continue
            if getattr(node.func.value, "id", None) != "client":
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                path = arg.value
            elif isinstance(arg, ast.JoinedStr):
                path = "".join(
                    part.value if isinstance(part, ast.Constant) else "{}"
                    for part in arg.values
                )
            else:
                continue
            if not path.startswith("/"):
                continue
            calls.append((f"{module.name}:{node.lineno}", node.func.attr.upper(), path))
    return calls


class TestTheProbeCanSee:
    """A blind probe agrees with everything. Both halves must be non-empty."""

    def test_the_served_schema_is_populated(self):
        served = _served()
        assert len(served) > 200, (
            f"only {len(served)} (method, path) pairs — the schema read is "
            f"broken, and a broken read makes every assertion below vacuous"
        )

    def test_tool_calls_are_found(self):
        calls = _tool_calls()
        assert len(calls) > 50, (
            f"only {len(calls)} client calls found across the tool modules — "
            f"the AST scan is broken"
        )

    def test_the_scan_would_notice_a_missing_route(self):
        """Negative control baked in: a path nobody serves must not match."""
        served = _served()
        assert ("GET", f"{API_ROOT}/definitely-not-a-route") not in served


class TestEveryToolRouteExists:
    def test_no_tool_calls_a_route_that_does_not_exist(self):
        served = _served()
        broken = []
        for where, method, path in _tool_calls():
            normalised = re.sub(r"\{[^}]+\}", "{}", path.split("?")[0].rstrip("/"))
            if (method, API_ROOT + normalised) not in served:
                broken.append(f"{where}  {method} {path}")

        assert not broken, (
            "these MCP tools call routes the API does not serve, so every agent "
            "invocation fails at the transport layer:\n  " + "\n  ".join(broken)
        )

    def test_the_extraction_summary_route_specifically_exists(self):
        """The one this was written for — a GET, not just the DELETE."""
        served = _served()
        assert ("GET", f"{API_ROOT}/extractions/{{}}") in served, (
            "GET /extractions/{id} is gone again; get_extraction_summary will "
            "return 405 from the DELETE-only path"
        )
        assert ("DELETE", f"{API_ROOT}/extractions/{{}}") in served, (
            "the DELETE that was there all along has been lost"
        )
