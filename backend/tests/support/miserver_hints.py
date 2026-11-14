"""miServer's `error.next` hints, found and checked against THIS server's live MCP registry
(042 FR-042.20, BR-W07; FTDD §6.8; FTID §3.11; TPD-11).

miServer's management errors carry `error.next`, the name of the MCP tool that resolves the refusal. The
names are tools of THIS repository's MCP server (`millm_*` today, `miserver_*` after the P3 cut), so a
rename here that miServer's hints do not follow leaves an agent told to call a tool that does not exist.

* **Authority:** `docs/schemas/next-tool-hints.json` in miServer's checkout (`{"version": 1,
  "next_tools": [sorted names]}`, 040 FR-040.17), when it exists.
* **Cross-check, so the file cannot omit a hint:** the AST of every `.py` under miServer's package (either
  name, `sibling_repos.miserver_path`): a class-level `next_tool = <expr>` and every call keyword
  `next_tool=<expr>`. `<expr>` is a string literal, `None` (no hint), or a name / attribute naming a
  module-level string constant of `<package>/core/next_tools.py`; anything else raises `DynamicHint`
  (a hint computed at run time cannot be checked, so it fails).
* **Transition:** with neither the JSON nor `next_tools.py` (040's P0 not landed), the AST set is the
  authority and `hints()` says so; `next_tools.py` WITHOUT the JSON is 040 shipped half and raises
  `HalfShipped` (a failure under the require flag).
* **Liveness:** every name must be a tool of the server `build_server()` builds with every category on.
  Between this repo's P3 cut and miServer's FR-040.35 push a `millm_<x>` passes only when `miserver_<x>`
  is live AND a product client the server built carries it in its relay set (read from the built client).
  `LIVENESS_MODE` flipped to `"exact"` in the P7 commit that removed the relay rewrite (TI-21): every hint must
  now be a live tool by its exact name.
"""
from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path

from tests.support.sibling_repos import miserver_package, miserver_path

HINTS_FILE = Path("docs") / "schemas" / "next-tool-hints.json"
LIVENESS_MODE = "exact"  # flipped from "transition" by the P7 relay-removal commit (TI-21; FTASKS 18.6)


class DynamicHint(Exception):
    def __init__(self, file: Path, line: int, why: str) -> None:
        super().__init__(f"{file}:{line}: dynamic next_tool ({why}) — a hint must be a constant this test can read")
        self.file, self.line = file, line


class HalfShipped(Exception):
    """miServer has `core/next_tools.py` but not the published `next-tool-hints.json` (040 shipped half)."""


def published_hints(repo: Path) -> set[str] | None:
    path = Path(repo) / HINTS_FILE
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or set(doc) != {"version", "next_tools"}:
        raise ValueError(f"{path}: keys must be exactly version, next_tools; got {sorted(doc) if isinstance(doc, dict) else doc!r}")
    if doc["version"] != 1:
        raise ValueError(f"{path}: unknown version {doc['version']!r}")
    names = doc["next_tools"]
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise ValueError(f"{path}: next_tools must be a list of names")
    if names != sorted(set(names)):
        raise ValueError(f"{path}: next_tools must be sorted with no duplicates")
    return set(names)


def _next_tools_module(repo: Path) -> Path:
    return miserver_path(repo, "core", "next_tools.py")


def next_tools_constants(repo: Path) -> dict[str, str] | None:
    """Module-level string constants of `<package>/core/next_tools.py`, or None when the file is absent."""
    path = _next_tools_module(repo)
    if not path.is_file():
        return None
    out: dict[str, str] = {}
    for node in ast.parse(path.read_text(encoding="utf-8"), str(path)).body:
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
            out[target.id] = value.value
    return out


def _aliases(tree: ast.AST, pkg: str) -> tuple[set[str], dict[str, str]]:
    """Names bound to the next_tools MODULE, and names bound to one of its constants."""
    modules: set[str] = set()
    constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                if a.name == "next_tools" and (mod in (f"{pkg}.core", "core") or (node.level and mod in ("core", ""))):
                    modules.add(a.asname or a.name)
                elif mod in (f"{pkg}.core.next_tools", "core.next_tools", "next_tools") or (
                        node.level and mod.endswith("next_tools")):
                    constants[a.asname or a.name] = a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == f"{pkg}.core.next_tools":
                    if a.asname:
                        modules.add(a.asname)
    return modules, constants


def _resolve(expr: ast.AST, file: Path, modules: set[str], const_names: dict[str, str],
             consts: dict[str, str]) -> str | None:
    if isinstance(expr, ast.Constant):
        if expr.value is None:
            return None
        if isinstance(expr.value, str):
            return expr.value
        raise DynamicHint(file, expr.lineno, f"a {type(expr.value).__name__} constant")
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id in modules:
        if expr.attr in consts:
            return consts[expr.attr]
        raise DynamicHint(file, expr.lineno, f"next_tools.{expr.attr} is not a string constant of next_tools.py")
    if isinstance(expr, ast.Name) and expr.id in const_names and const_names[expr.id] in consts:
        return consts[const_names[expr.id]]
    raise DynamicHint(file, getattr(expr, "lineno", 0), ast.unparse(expr))


def source_hints(repo: Path) -> set[str]:
    """Every hint miServer's package can send, by AST; raises DynamicHint on a hint it cannot read."""
    pkg = miserver_package(repo)
    if pkg is None:
        raise FileNotFoundError(f"{repo} holds neither miserver/ nor millm/")
    consts = next_tools_constants(repo) or {}
    root = miserver_path(repo)
    files = sorted(root.rglob("*.py"))
    if not files:
        raise FileNotFoundError(f"no Python file under {root}: discovery read nothing")
    found: set[str] = set()
    for file in files:
        tree = ast.parse(file.read_text(encoding="utf-8"), str(file))
        modules, const_names = _aliases(tree, pkg)
        for node in ast.walk(tree):
            exprs: list[ast.AST] = []
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "next_tool"
                                                            for t in stmt.targets):
                        exprs.append(stmt.value)
                    elif (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                          and stmt.target.id == "next_tool" and stmt.value is not None):
                        exprs.append(stmt.value)
            elif isinstance(node, ast.Call):
                exprs += [k.value for k in node.keywords if k.arg == "next_tool"]
            for expr in exprs:
                name = _resolve(expr, file, modules, const_names, consts)
                if name is not None:
                    found.add(name)
    return found


def hints(repo: Path) -> tuple[set[str], str]:
    """`(names, authority)`. The published file when it exists (and the source set must be a subset of
    it); the source set when neither file exists; `HalfShipped` when only `next_tools.py` exists."""
    published = published_hints(repo)
    source = source_hints(repo)
    if published is not None:
        missing = source - published
        if missing:
            raise AssertionError(f"miServer's source sends hints its published {HINTS_FILE} omits: {sorted(missing)}")
        return published, f"published ({HINTS_FILE})"
    if next_tools_constants(repo) is not None:
        raise HalfShipped(f"{_next_tools_module(repo)} exists but {Path(repo) / HINTS_FILE} does not")
    return source, "source (040 P0 not landed)"


def built_relay_sets() -> list[frozenset[str]]:
    """The relay set of every product client a real `build_server()` builds (every category on, every
    product URL set). Read from the BUILT clients: before the P3 cut no client has one."""
    import asyncio  # noqa: F401  (the server module needs a loop-free build only)

    import pytest

    from src.mcp_server import server as server_mod
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.tools import all_categories

    built: list[object] = []
    real = server_mod._client_for
    mp = pytest.MonkeyPatch()
    try:
        for spec in PRODUCTS:
            mp.setenv(spec.url_env, f"http://{spec.name}.test")

        def capture(spec, url):
            client = real(spec, url)
            built.append(client)
            return client

        mp.setattr(server_mod, "_client_for", capture)
        settings = MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True)
        server_mod.build_server(settings, stdio=True)
    finally:
        mp.undo()
    assert built, "build_server built no product client: the relay check would read nothing"
    return [frozenset(getattr(c, "relay_names", None) or ()) for c in built]


def liveness_problems(names: Iterable[str], live: set[str], relay_sets: list[frozenset[str]] | None = None,
                      mode: str | None = None) -> list[str]:
    mode = mode or LIVENESS_MODE
    out = []
    for name in sorted(names):
        if name in live:
            continue
        if mode == "transition" and name.startswith("millm_"):
            twin = "miserver_" + name[len("millm_"):]
            relayed = any(twin in s for s in (relay_sets or []))
            if twin in live and relayed:
                continue
            out.append(f"{name}: not live; its twin {twin} is {'live' if twin in live else 'not live'} and "
                       f"{'is' if relayed else 'is not'} in a built client's relay set")
            continue
        out.append(f"{name}: not a tool of the live registry")
    return out


def checked_hints(repo: Path) -> tuple[set[str], str]:
    """`hints(repo)` for a cross-repo test: a missing checkout or a half-shipped 040 skips loudly, and FAILS
    under the require flag (`require_cross_repo()`)."""
    import pytest

    from tests.support.cross_repo import require_cross_repo

    def absent(reason: str):
        if require_cross_repo():
            pytest.fail("cross-repo checks are required but the miServer hint check cannot run: " + reason)
        pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + reason)

    if miserver_package(repo) is None:
        absent(f"no miServer checkout at {repo} (set MISERVER_REPO)")
    try:
        return hints(repo)
    except HalfShipped as exc:
        absent(f"040 shipped half: {exc}")
        raise  # unreachable: absent() always raises
