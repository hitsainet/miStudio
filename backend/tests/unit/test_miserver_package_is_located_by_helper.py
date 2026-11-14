"""miServer's Python package is located only through `tests/support/sibling_repos.py` (042 FR-042.36,
BR-W11; FTDD §6.9; FTASKS 5.5, control K-5.14).

miServer renames its package `millm` → `miserver` at ENH-009 P8 (miServer 041). A test that hard-codes
`<repo>/millm/…`, `from millm.main import …` or `import millm` would then stop finding miServer and SKIP
(green while checking nothing) — exactly the failure the cross-repo checks exist to prevent. The helpers
`miserver_package()`, `miserver_path()`, `miserver_import()` resolve either name; this guard keeps every
other test on them.

AST over `backend/tests/` and `rule_runner/tests/` (never a text search: a comment or a message may name
the package). Outside `sibling_repos.py` it refuses:
* `"millm"` / `"miserver"` as an operand of `/`, or as an argument of `Path(…)`, `.joinpath(…)`,
  `os.path.join(…)`;
* a string constant (incl. folded concatenations and f-string constant parts) holding `from millm`,
  `import millm`, `millm/<…>.py`, `millm.main` (and the `miserver` forms);
* a real `import millm…` / `from millm… import`, and `import_module("millm…")`.
One exemption, by FILE AND LINE and count, with its reason; it must still match.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ROOTS = (REPO / "backend" / "tests", REPO / "rule_runner" / "tests")
HELPER = "backend/tests/support/sibling_repos.py"
PACKAGES = ("millm", "miserver")
TEXT = re.compile(r"\b(?:from|import)\s+(?:millm|miserver)\b|\b(?:millm|miserver)/[\w./-]*\.py\b"
                  r"|\b(?:millm|miserver)\.main\b")
PATH_CALLS = {"Path", "PurePath", "joinpath", "join"}

#: (file, the exact stripped text of the reported line) -> (count, reason). Keyed by the line's TEXT, not its
#: number: the exemption still covers exactly one line, and an edit above it (042 P0 builder b added lines to
#: that file, moving the sample from :156 to :176) no longer turns it stale. Never a whole file.
EXEMPT: dict[tuple[str, str], tuple[int, str]] = {
    ("backend/tests/unit/test_mcp_tools_stay_on_http.py", '"import midataworks\\n"'): (
        1, "a PLANTED refusal sample (FTID 3.12): that guard proves an MCP tool importing millm is refused"),
}


def _fold(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
    return None


def _is_pkg(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in PACKAGES


def scan(text: str, rel: str) -> list[tuple[str, int, str]]:
    tree = ast.parse(text)
    out: set[tuple[str, int, str]] = set()
    folded_children: set[int] = set()
    for node in ast.walk(tree):  # a bare string statement (a docstring) is prose
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            folded_children.add(id(node.value))
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.JoinedStr)) and _fold(node) is not None:
            for child in ast.walk(node):
                if child is not node:
                    folded_children.add(id(child))
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and (_is_pkg(node.left) or _is_pkg(node.right)):
            out.add((rel, node.lineno, "a package name as a path segment (`/`)"))
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in PATH_CALLS and any(_is_pkg(a) for a in node.args):
                out.add((rel, node.lineno, f"a package name as a path segment ({name}(…))"))
            if name in ("import_module", "__import__") and node.args:
                first = _fold(node.args[0])
                if first and first.split(".")[0] in PACKAGES:
                    out.add((rel, node.lineno, f"{name}({first!r})"))
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in PACKAGES:
                    out.add((rel, node.lineno, f"import {alias.name}"))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in PACKAGES and not node.level:
            out.add((rel, node.lineno, f"from {node.module} import …"))
        if id(node) in folded_children:
            continue
        folded = _fold(node) if isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr)) else None
        if folded and TEXT.search(folded):
            out.add((rel, node.lineno, f"a string naming the package: {TEXT.search(folded).group(0)!r}"))
    return sorted(out)


def _files() -> list[Path]:
    return sorted(p for root in ROOTS for p in root.rglob("*.py"))


def test_no_package_located_outside_the_helper():
    files = _files()
    assert len(files) > 500, f"only {len(files)} test files discovered"
    assert any(str(p).endswith(HELPER) for p in files), "the helper itself was not discovered"
    found, counts = [], {}
    for path in files:
        rel = str(path.relative_to(REPO))
        if rel == HELPER:
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for item in scan(text, rel):
            key = (item[0], lines[item[1] - 1].strip())
            if key in EXEMPT:
                counts[key] = counts.get(key, 0) + 1
            else:
                found.append(f"{item[0]}:{item[1]}: {item[2]}")
    assert not found, ("locate miServer's package with tests/support/sibling_repos.py "
                       "(miserver_package / miserver_path / miserver_import):\n" + "\n".join(found))
    for key, (count, reason) in EXEMPT.items():
        assert counts.get(key, 0) == count, f"stale exemption {key} ({reason}): matched {counts.get(key, 0)}"


def test_the_helper_names_both_packages():
    held = {n.value for n in ast.walk(ast.parse((REPO / HELPER).read_text(encoding="utf-8")))
            if isinstance(n, ast.Constant) and n.value in PACKAGES}
    assert held == set(PACKAGES), held


def _j(*parts: str) -> str:
    """Join at runtime, so a planted sample is never a string constant of this file."""
    return "".join(parts)


M = _j("mil", "lm")
S = _j("mise", "rver")


@pytest.mark.parametrize("source", [
    _j('p = repo / "', M, '" / "main.py"\n'),
    _j('p = "', S, '" / x\n'),
    _j('p = Path(repo, "', M, '", "services")\n'),
    _j('p = repo.joinpath("', M, '", "x.py")\n'),
    _j('p = os.path.join(repo, "', S, '")\n'),
    _j('C = "from ', M, '.main import create_app"\n'),
    _j('C = "import json; from " + "', M, '.services import x"\n'),
    _j('C = f"{root}/', M, '/main.py"\n'),
    _j('C = "python -c \'import ', M, '\'"\n'),
    _j('import ', M, '.services.probe_runtime\n'),
    _j('from ', M, '.ml.probe_head import ProbeHead\n'),
    _j('m = importlib.import_module("', M, '.api.schemas.probe")\n'),
])
def test_the_guard_finds_each_form(source):
    assert scan(source, "planted.py"), source


@pytest.mark.parametrize("source", [
    'p = miserver_path(repo, "services", "x.py")\n',
    'C = f"from {pkg}.main import create_app"\n',
    'k = "millm.steering-set/v1"\n',
    _j('"""Docstring: ', M, '/services/probe_runtime.py and from ', M, '.main import x."""\n'),
    'h = "k8s-millm.hitsai.local"\n',
    'tool = "miserver_list_probes"\n',
    'm = importlib.import_module(f"{pkg}.ml.probe_head")\n',
])
def test_other_millm_strings_pass(source):
    assert scan(source, "ok.py") == []
