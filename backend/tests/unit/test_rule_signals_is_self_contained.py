"""The rule engine imports nothing from the app (Feature 040 FR-27; FTDD §6.1; FTID §2.3; FTASKS 17.1).

`src/rule_signals/` is a pure function of (rule set, text, scope, limits). It may import the standard library,
`re2`, `jsonschema`, `rfc8785` and itself — nothing from `models`, `services`, `workers`, `db`, `core`, `api`,
`ml` or `schemas` (the vendored files are READ by path), and no sibling app.

Two readings, so neither can pass alone by accident (suite memory `source-scrape-guards-fail-open`):

1. the AST of every module (relative imports resolved against the module's real package);
2. the LIVE import: a fresh interpreter imports `src.rule_signals` and every submodule, and the modules then
   in `sys.modules` under `src.` must all be `src.rule_signals.*` (plus the `src` package itself).

Control (FTASKS 17.1): add ``from src.services import …`` to a module → both tests red.
"""
from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = BACKEND / "src" / "rule_signals"
FORBIDDEN_ROOTS = ("models", "services", "workers", "db", "core", "api", "ml", "schemas", "mcp_server", "utils")
ALLOWED_THIRD_PARTY = {"re2", "jsonschema", "rfc8785"}


def _modules():
    return sorted(PACKAGE.rglob("*.py"))


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = ".".join(path.resolve().relative_to(BACKEND).parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                yield ".".join(base + ([node.module] if node.module else []))
            else:
                yield node.module or ""


def test_the_scan_sees_every_module():
    names = {p.name for p in _modules()}
    assert {"__init__.py", "dialect.py", "compose.py", "lexical.py", "reduce.py", "cases.py", "canonical.py",
            "runner.py", "document.py"} <= names
    assert len(list((PACKAGE / "validators").glob("*.py"))) >= 4


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_module_imports_the_app(path):
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    bad = []
    for name in _imports(path):
        root = name.split(".")[0]
        if name.startswith("src."):
            if not name.startswith("src.rule_signals"):
                bad.append(name)
        elif root not in stdlib and root not in ALLOWED_THIRD_PARTY:
            bad.append(name)
    assert not bad, f"{path.name} imports {bad}; rule_signals may import only the stdlib, re2, jsonschema, rfc8785"


def test_the_live_import_loads_nothing_else_from_the_app():
    code = (
        "import json, sys, pkgutil, importlib\n"
        "import src.rule_signals as p\n"
        "for m in pkgutil.walk_packages(p.__path__, 'src.rule_signals.'): importlib.import_module(m.name)\n"
        "print(json.dumps(sorted(n for n in sys.modules if n.startswith('src.'))))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert loaded and all(n.startswith("src.rule_signals") for n in loaded), [
        n for n in loaded if not n.startswith("src.rule_signals")]
