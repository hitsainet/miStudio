"""Which HTTP route each MCP tool calls, joined to the LIVE registry (route→tool parity guards).

Membership comes from the authority: `registered_tools()` builds the REAL server with every
category enabled and reads `list_tools()` — the same call an agent makes. The AST walk here only
ENRICHES a registered tool with the routes it calls (`list_tools()` cannot report endpoints); it
never decides what is registered. A tool whose registration line is deleted therefore drops out
of the covered set and its routes go uncovered, which is the red the parity guards exist for
(memory: derive-from-the-registry-not-the-source, source-scrape-guards-fail-open).

What the walk reads, per `@mcp.tool` function:

* `client.<verb>("/path")`, `millm.<verb>(...)`, `dw.<verb>(...)` and `ctx.mistudio.<verb>(...)`
  with a literal or f-string path — `{…}` parts become `{}`;
* calls to a helper defined in the same module, followed transitively, including a helper that
  receives the path as an argument (`_submit("steer_sweep", "/steering/async/sweep", body)`).

`raw_get` is a GET. Calls whose path the walk cannot read are reported by `unreadable_calls()`,
which the guards require to be empty — an unread call is a route the guard cannot see.
"""
from __future__ import annotations

import ast
import asyncio
import re
from functools import lru_cache
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "tools"

VERBS = {"get", "post", "put", "patch", "delete", "raw_get"}

#: receiver name → the product the call goes to.
RECEIVERS = {"client": "mistudio", "millm": "millm", "dw": "dataworks"}


def normalise(path: str) -> str:
    """`/a/{x}/b?q=1` → `/a/{}/b`; a `{repo_id:path}` converter is a parameter like any other."""
    return re.sub(r"\{[^}]*\}", "{}", path.split("?")[0]).rstrip("/") or "/"


def _product(receiver: ast.AST) -> str | None:
    if isinstance(receiver, ast.Name):
        return RECEIVERS.get(receiver.id)
    if isinstance(receiver, ast.Attribute) and receiver.attr == "mistudio":
        return "mistudio"
    return None


def _path_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "{}" for part in node.values
        )
    return None


def _is_tool(fn: ast.AST) -> bool:
    for dec in getattr(fn, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "tool":
            return True
    return False


Route = tuple[str, str, str]  # (product, METHOD, path as written)


def _module_routes(path: Path) -> tuple[dict[str, set[Route]], list[str]]:  # noqa: C901
    tree = ast.parse(path.read_text())
    funcs = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    unreadable: set[str] = set()

    def direct(fn) -> tuple[set[Route], set[tuple[str, str, str]], list[tuple[str, ast.Call]]]:
        routes: set[Route] = set()
        by_param: set[tuple[str, str, str]] = set()  # (product, METHOD, parameter name)
        helpers: list[tuple[str, ast.Call]] = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in VERBS:
                product = _product(func.value)
                if product and node.args:
                    method = "GET" if func.attr == "raw_get" else func.attr.upper()
                    literal = _path_of(node.args[0])
                    if literal is not None and literal.startswith("/"):
                        routes.add((product, method, literal))
                    elif isinstance(node.args[0], ast.Name):
                        by_param.add((product, method, node.args[0].id))
                    else:
                        unreadable.add(f"{path.name}:{node.lineno}")
            if isinstance(func, ast.Name) and func.id in funcs and funcs[func.id] is not fn:
                helpers.append((func.id, node))
        return routes, by_param, helpers

    def resolve(fn, seen: frozenset[str]) -> set[Route]:
        routes, _, helpers = direct(fn)
        for name, call in helpers:
            if name in seen:
                continue
            helper = funcs[name]
            routes |= resolve(helper, seen | {name})
            _, by_param, _ = direct(helper)
            params = [a.arg for a in helper.args.args]
            for product, method, param in by_param:
                if param not in params:
                    unreadable.add(f"{path.name}:{helper.lineno} ({param})")
                    continue
                values = []
                index = params.index(param)
                if index < len(call.args):
                    values.append(call.args[index])
                values += [kw.value for kw in call.keywords if kw.arg == param]
                for value in values:
                    literal = _path_of(value)
                    if literal is not None and literal.startswith("/"):
                        routes.add((product, method, literal))
        return routes

    out = {}
    for name, fn in funcs.items():
        if isinstance(fn, ast.AsyncFunctionDef) and _is_tool(fn):
            out[name] = resolve(fn, frozenset({name}))
            # A tool's OWN call with a variable path has no call site to resolve it from.
            for _product_name, _method, param in direct(fn)[1]:
                unreadable.add(f"{path.name}:{fn.lineno} ({param})")
    return out, sorted(unreadable)


@lru_cache(maxsize=1)
def tool_routes() -> dict[str, frozenset[Route]]:
    """`{tool function name: {(product, METHOD, path)}}` for every `@mcp.tool` in tools/."""
    out: dict[str, frozenset[Route]] = {}
    for module in sorted(TOOLS.glob("*.py")):
        routes, _ = _module_routes(module)
        for name, found in routes.items():
            assert name not in out, f"tool function {name} is defined twice"
            out[name] = frozenset(found)
    return out


def unreadable_calls() -> list[str]:
    """Client calls whose path is neither a literal, an f-string nor a helper parameter."""
    found: list[str] = []
    for module in sorted(TOOLS.glob("*.py")):
        found += _module_routes(module)[1]
    return found


def registered_tools(monkeypatch) -> set[str]:
    """Every tool on the REAL built server with EVERY category enabled and every product's URL
    set — the set the agent sees, not the set the source declares."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    for spec in PRODUCTS:
        monkeypatch.setenv(spec.url_env, f"http://{spec.name}.test")
    settings = MCPSettings(
        tool_categories=",".join(sorted(all_categories())), allow_anonymous=True
    )
    mcp, _client = build_server(settings, stdio=True)
    return {t.name for t in asyncio.run(mcp.list_tools())}


def called_routes(registered: set[str], product: str, prefix: str = "") -> dict[tuple[str, str], set[str]]:
    """`{(METHOD, prefix + normalised path): {tools}}` for REGISTERED tools calling `product`."""
    routes = tool_routes()
    out: dict[tuple[str, str], set[str]] = {}
    for tool in registered:
        for prod, method, path in routes.get(tool, ()):
            if prod == product:
                out.setdefault((method, normalise(prefix + path)), set()).add(tool)
    return out
