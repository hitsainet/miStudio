"""The runner host imports nothing from the app, and the app never loads the runner's Wasmtime (Feature 040 S4;
FTDD Appendix C.4; FR-31; FTASKS 25.4).

`src/rule_runner/` ships ALONE in `hitsai/mistudio-rule-runner`, installed as the top-level package
`mistudio_rule_runner` (no `src/__init__`, which imports SQLAlchemy). So:

1. every import in it is RELATIVE (within the package), the standard library, or one of ``wasmtime``,
   ``starlette``, ``uvicorn``, ``anyio`` — never ``src.…`` (AST of every module);
2. ``contract`` is standard library ONLY: the backend imports it (hash rule, floor, reasons) without Wasmtime;
3. LIVE: a fresh interpreter importing ``src.rule_runner.contract`` loads no third-party module of the runner, and
   importing the API app (``src.main``) and the rules worker's task module loads NO ``wasmtime`` (miStudio never runs
   operator code in-process, FR-31);
4. no ``.wasm`` / ``.cwasm`` is tracked by git (the guest is fetched by digest at build time).

Controls (040 controls record, S4): add ``from src.core.config import settings`` to a runner module → red; import
``wasmtime`` at the top of ``contract.py`` → red.
"""
from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
PACKAGE = BACKEND / "src" / "rule_runner"
ALLOWED_THIRD_PARTY = {"wasmtime", "starlette", "uvicorn", "anyio"}
STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def _modules():
    return sorted(PACKAGE.glob("*.py"))


def _imports(path: pathlib.Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield 0, alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.level, node.module or ""


def test_the_scan_sees_every_module():
    names = {p.name for p in _modules()}
    assert {"__init__.py", "contract.py", "config.py", "wasi.py", "reply.py", "host.py", "lanes.py", "bundle.py",
            "documents.py", "server.py", "fetch.py"} <= names
    subpackages = [p for p in PACKAGE.glob("*/") if p.name != "__pycache__"]
    assert not subpackages, "a subpackage would need the scan extended"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_imports_the_app(path):
    bad = []
    for level, name in _imports(path):
        if level:
            continue  # relative: within mistudio_rule_runner
        root = name.split(".")[0]
        if root not in STDLIB and root not in ALLOWED_THIRD_PARTY:
            bad.append(name)
    assert not bad, f"{path.name} imports {bad}: the runner image has only the stdlib, {sorted(ALLOWED_THIRD_PARTY)}"


def test_the_contract_module_is_standard_library_only():
    bad = [name for level, name in _imports(PACKAGE / "contract.py")
           if level or name.split(".")[0] not in STDLIB]
    assert not bad, bad


def _live(code: str) -> list:
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_backend_never_loads_wasmtime():
    loaded = _live(
        "import json, sys\n"
        "import src.rule_runner.contract\n"
        "first = sorted(n for n in sys.modules if n.split('.')[0] in ('wasmtime','starlette','uvicorn','anyio'))\n"
        "import src.main, src.workers.rules_evaluation_tasks, src.services.rules_evaluation\n"
        "import src.services.code_rule_runner, src.services.code_rules\n"
        "print(json.dumps({'contract': first, 'wasmtime': sorted(n for n in sys.modules if n.startswith('wasmtime'))}))\n"
    )
    assert loaded["contract"] == [], loaded["contract"]
    assert loaded["wasmtime"] == [], "the backend loaded wasmtime: operator code must never run in-process (FR-31)"


def test_no_webassembly_binary_is_tracked():
    out = subprocess.run(["git", "ls-files", "*.wasm", "*.cwasm", "**/*.wasm", "**/*.cwasm"], cwd=REPO,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0
    assert out.stdout.strip() == "", out.stdout
