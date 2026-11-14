"""Every product key in MCP tool code comes from the product row, never a literal (042 FR-042.16; FTID §3.10;
FTASKS 12.7, control K-12.33).

At the P3 cut the miServer row's name changed from `"millm"` to `"miserver"`. A tool that spelled the key as a
string would keep answering `{"unavailable": "millm"}` (or gate on a product the health gate no longer knows)
while every guard that reads the row stayed green. So the AST of every module under `src/mcp_server/` must hold
NO string constant `"millm"` or `"miserver"`:

* as an argument of `gated(...)`, or of a health gate's `check(...)` / `invalidate(...)`;
* as the value of an `"unavailable"` or `"decide_in"` key in a dict literal.

The walk must find what it guards (memory `source-scrape-guards-fail-open`): at least one `gated(...)` call, one
`check(...)`, and one `unavailable` dict whose value comes from the row.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "mcp_server"
LITERAL_KEYS = {"millm", "miserver"}
CALLS = {"gated", "check", "invalidate"}
VALUE_KEYS = {"unavailable", "decide_in"}


def _name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _scan() -> tuple[list[str], dict[str, int]]:
    findings: list[str] = []
    seen = {"gated": 0, "check": 0, "row_value": 0}
    files = sorted(SRC.rglob("*.py"))
    assert files, f"no module under {SRC}"
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(SRC.parents[1])
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func) in CALLS:
                name = _name(node.func)
                if name in seen:
                    seen[name] += 1
                for arg in [*node.args, *(k.value for k in node.keywords)]:
                    if isinstance(arg, ast.Constant) and arg.value in LITERAL_KEYS:
                        findings.append(f"{rel}:{node.lineno}: {name}({arg.value!r}) — use MISERVER.name")
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value in VALUE_KEYS:
                        if isinstance(value, ast.Constant) and value.value in LITERAL_KEYS:
                            findings.append(f"{rel}:{node.lineno}: {{{key.value!r}: {value.value!r}}} — use the row")
                        elif isinstance(value, ast.Attribute) and value.attr == "name":
                            seen["row_value"] += 1
            # `out["decide_in"] = "millm"` — the subscript form of the same value.
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and \
                    node.value.value in LITERAL_KEYS:
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and \
                            target.slice.value in VALUE_KEYS:
                        findings.append(f"{rel}:{node.lineno}: [{target.slice.value!r}] = {node.value.value!r}")
    return findings, seen


def test_no_product_key_literal_in_mcp_server():
    findings, seen = _scan()
    assert seen["gated"] >= 50, f"the walk found {seen['gated']} gated(...) calls: it is not reading the tools"
    assert seen["check"] >= 10, f"the walk found {seen['check']} check(...) calls"
    assert seen["row_value"] >= 10, f"the walk found {seen['row_value']} row-valued unavailable/decide_in dicts"
    assert findings == [], "\n".join(findings)


def test_the_row_names_miserver():
    """The value agents read is the row's: `unavailable`/`decide_in` say `miserver` after the cut."""
    from src.mcp_server.products import MISERVER, PRODUCTS

    assert MISERVER.name == "miserver" and MISERVER in PRODUCTS
    assert MISERVER.approval_owner == "product"
    assert MISERVER.identity == "agent:miworkbench-mcp"
