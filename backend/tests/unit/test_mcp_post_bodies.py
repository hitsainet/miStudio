"""Every MCP tool that POSTs must send a BODY, not a query parameter.

`MiStudioClient.post(path, json_body=None, **params)` funnels unknown keyword
arguments into the QUERY STRING. So `client.post(path, json=body)` type-checks,
runs, and silently sends no body at all; the failure surfaces only as a server
-side `422 {"loc": ["body"], "msg": "Field required"}`, far from the typo.

Found 2026-09-05: three jlens tools were written with `json=` --
`/jlens/acquire/preview`, `/jlens/publish` and `/jlens/acquire`. That is the
ENTIRE artifact-acquisition path, so no pre-fitted lens could be previewed,
acquired or published over MCP at all, while `list` and `validate` worked fine
and made the category look healthy.

Asserting "the client was called" would pass against the broken version. These
assert the BODY ARRIVED.
"""

import ast
import pathlib

import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "tools"
CLIENT = pathlib.Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "client.py"


def _body_carrying_calls(tree: ast.AST):
    """Yield (lineno, method, kwargs) for every client.post/patch/put call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr not in ("post", "patch", "put"):
            continue
        if not isinstance(fn.value, ast.Name) or "client" not in fn.value.id.lower():
            continue
        yield node.lineno, fn.attr, {k.arg for k in node.keywords if k.arg}


def test_client_post_still_swallows_unknown_kwargs_into_params():
    """The premise. If this ever stops being true, this whole file can go."""
    src = CLIENT.read_text()
    assert "async def post(self, path: str, json_body:" in src
    assert "**params" in src, (
        "post() no longer absorbs stray kwargs; the silent-failure mode these "
        "tests guard may not exist any more"
    )


@pytest.mark.parametrize("path", sorted(p for p in TOOLS.glob("*.py")), ids=lambda p: p.name)
def test_no_tool_passes_json_instead_of_json_body(path):
    tree = ast.parse(path.read_text())
    offenders = [
        (line, method) for line, method, kwargs in _body_carrying_calls(tree)
        if "json" in kwargs and "json_body" not in kwargs
    ]
    assert not offenders, (
        f"{path.name}: client.{offenders[0][1]}() called with `json=` at line "
        f"{offenders[0][0]}. The client takes `json_body=`; `json=` becomes a "
        f"query parameter and the request is sent with NO BODY. All: {offenders}"
    )


def test_the_three_acquisition_tools_send_a_body():
    """Pins the specific regression, by path, so a refactor cannot resurrect it."""
    tree = ast.parse((TOOLS / "jlens.py").read_text())
    want = {"/jlens/acquire/preview", "/jlens/publish", "/jlens/acquire"}
    seen = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "post" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in want):
            seen[node.args[0].value] = {k.arg for k in node.keywords if k.arg}
    assert want <= set(seen), f"missing acquisition POSTs: {want - set(seen)}"
    for p, kwargs in seen.items():
        assert "json_body" in kwargs, f"{p} does not send json_body (got {kwargs})"
