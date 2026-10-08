"""Generate the normative MCP contract from the LIVE tool registry.

WHY GENERATED
-------------
miLLM's `docs/mcp-contract.md` is hand-maintained, and for an entire increment
it carried ✅ marks for 16 tools that were never registered with the server.
miStudio had no contract at all: 32 `millm_*` rows existed there, zero rows for
the 58 native tools.

Writing a second hand-maintained document would repeat a pattern that has now
failed four times in this codebase (the copy audit's SURFACES list, the
reachability harness's category list, the contract row scraper, and the
hand-written tool map). So this derives the document, and a test regenerates it
and diffs — a stale contract fails the build instead of quietly lying.

WHAT IT IS NOT
--------------
Not a replacement for docstrings. This is the inventory an operator or reviewer
reads; an AGENT should call `mistudio_howto` instead, which carries the
workflows and the failure modes a table cannot express.
"""

import ast
import inspect
from typing import Any

HEADER = """<!-- GENERATED FILE — DO NOT EDIT BY HAND.

Regenerate with:
    python -c "from src.mcp_server.contract import write_contract; write_contract()"

`backend/tests/unit/test_mcp_contract_generated.py` regenerates this and fails
if it differs from what is committed, so an edit here is reverted by the next
run rather than silently kept.
-->

# miStudio MCP contract

Every tool the miStudio MCP server registers, its category, and the backend
endpoint it calls. Derived from the live registry — see
`src/mcp_server/contract.py`.

**Agents: call `mistudio_howto` instead of reading this.** A table cannot carry
ordering constraints, GPU-lock contention, id namespaces, or the failure modes
that mislead. This document is the inventory; that tool is the guidance.

Categories are gated by `MCP_TOOL_CATEGORIES`. The `millm_*` categories also
require `MILLM_API_URL` and are never enabled by default.
"""


def _endpoints_for(func_node: ast.AST) -> list[str]:
    """HTTP calls a tool body issues, as `METHOD /path`."""
    found: list[str] = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func

        # Tools that route through a submit helper pass the endpoint as an
        # ARGUMENT: `_submit("steer_sweep", "/steering/async/sweep", body)`.
        # The guard caught these three under-reporting with no endpoint at all,
        # which is the failure mode that matters — a contract row that says
        # nothing about what the tool reaches.
        if isinstance(fn, ast.Name) and fn.id.startswith("_submit"):
            for a in node.args:
                if (isinstance(a, ast.Constant) and isinstance(a.value, str)
                        and a.value.startswith("/")):
                    found.append(f"POST {a.value}")
            continue

        if not isinstance(fn, ast.Attribute) or fn.attr not in (
            "get", "post", "put", "delete", "patch", "raw_get"
        ):
            continue
        method = {"raw_get": "GET"}.get(fn.attr, fn.attr.upper())
        path = None
        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                path = arg.value
            elif isinstance(arg, ast.JoinedStr):
                # f"/circuits/{id}/export" → /circuits/{…}/export
                parts = []
                for v in arg.values:
                    if isinstance(v, ast.Constant):
                        parts.append(str(v.value))
                    else:
                        parts.append("{…}")
                path = "".join(parts)
        # AN HTTP PATH STARTS WITH "/" (MIS-E2E-114).
        #
        # `fn.attr in ("get", "post", ...)` also matches a plain
        # `dict.get("kind")`, so the scraper recorded ordinary dictionary
        # access as endpoints. The committed contract carried three:
        # `GET kind`, `GET manifests`, `GET /validation-manifests` on
        # `get_steering_samples` — and `test_mcp_contract_generated.py` pinned
        # them as correct, so the contract defended whatever path was recorded
        # rather than the real one.
        if path and path.startswith("/"):
            found.append(f"{method} {path}")
    return sorted(set(found))


def collect() -> dict[str, list[dict[str, Any]]]:
    """`{category: [{name, summary, endpoints, destructive}]}` from the registry."""
    from .tools import CATEGORY_MODULES, MILLM_CATEGORY_MODULES

    # The REGISTRY is the authority on what exists; the AST pass only enriches
    # rows with endpoints, which `list_tools()` cannot report. An earlier
    # version derived membership from the AST alone and an adversarial pass
    # walked straight through it.
    from .tools.howto import _all_tools

    index = _all_tools()

    out: dict[str, list[dict[str, Any]]] = {}
    for category, modules in {**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}.items():
        served = {n: d for n, d in index.get(category, [])}
        seen_in_source: set[str] = set()
        rows: list[dict[str, Any]] = []
        for module in modules:
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    continue
                is_tool = any(
                    isinstance(d.func if isinstance(d, ast.Call) else d, ast.Attribute)
                    and (d.func if isinstance(d, ast.Call) else d).attr == "tool"
                    for d in node.decorator_list
                )
                if not is_tool:
                    continue
                doc = ast.get_docstring(node) or ""
                summary = " ".join(doc.split())
                cut = summary.find(". ")
                if 0 < cut < 150:
                    summary = summary[: cut + 1]
                if node.name not in served:
                    # Registered-by-helper tools are invisible to this AST
                    # pass; they are picked up from `served` below. A tool in
                    # the source but NOT served (conditional registration) is
                    # correctly omitted — the contract records what an agent
                    # can actually call.
                    continue
                seen_in_source.add(node.name)
                rows.append({
                    "name": node.name,
                    "summary": summary[:160],
                    "endpoints": _endpoints_for(node),
                    # An operator scanning this needs the irreversible ones to
                    # stand out; the word is the tools' own convention.
                    # Keyword set widened after the first generated pass
                    # flagged only 4 of 5: millm_delete_circuit says
                    # "permanently" and "cannot be undone" without ever using
                    # the word DESTRUCTIVE, and it is the most dangerous tool
                    # in the surface.
                    "destructive": any(
                        k in doc
                        for k in ("DESTRUCTIVE", "Irreversible", "IRREVERSIBLE",
                                  "permanently", "cannot be undone",
                                  "irreversible")
                    ),
                })
        # Anything the server serves that the AST could not see still gets a
        # row — without an endpoint, but never omitted.
        for name, summary in served.items():
            if name not in seen_in_source:
                rows.append({"name": name, "summary": summary,
                             "endpoints": [], "destructive": False})
        if rows:
            out[category] = sorted(rows, key=lambda r: r["name"])
    return out


def render() -> str:
    """The full contract document."""
    index = collect()
    total = sum(len(v) for v in index.values())
    lines = [HEADER, f"\n**{total} tools across {len(index)} categories.**\n"]
    for category in sorted(index):
        rows = index[category]
        lines.append(f"\n## `{category}` ({len(rows)} tools)\n")
        lines.append("| Tool | Endpoint | Summary |")
        lines.append("|---|---|---|")
        for r in rows:
            eps = "<br>".join(f"`{e}`" for e in r["endpoints"]) or "—"
            name = f"`{r['name']}`"
            if r["destructive"]:
                name += " ⚠️"
            summary = r["summary"].replace("|", "\\|")
            lines.append(f"| {name} | {eps} | {summary} |")
    lines.append("\n⚠️ = destructive or irreversible.\n")
    return "\n".join(lines)


def contract_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[3] / "docs" / "mcp-contract.md"


def write_contract() -> str:
    path = contract_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render()
    path.write_text(text)
    return str(path)
