"""Read Celery's wiring from the CALL, not from the text of the module.

⚠ WHY. `celery_app.autodiscover_tasks(force=True)` imports its module list immediately
and keeps no record of it: `conf.include` and `conf.imports` are both empty here. So a
test that wants to know whether a worker will import a module has nothing to read back
at runtime, and two workarounds have been used in this suite:

  * importing the module in a fixture and asserting `name in celery_app.tasks` — which
    GUARANTEES its own result. Verified 2026-09-26: deleting
    `"src.workers.probe_monitor_tasks"` from the list left
    `test_probe_monitor_wiring.py` at 41 passed while no worker would have run a single
    probe task. The same failure as the 16 unregistered `millm_circuit_*` MCP tools.

  * `inspect.getsource(celery_app)` and a substring check (`test_jlens_reachable.py`) —
    which is satisfied by ANY occurrence of the string, including one inside
    `task_routes`, a comment, or a docstring quoting the very line it is checking for.
    "A guard satisfied by the wrong occurrence" is a named, repeated failure here.

Reading the argument list of the `autodiscover_tasks` call is immune to both: it cannot
be satisfied by prose, and it does not depend on what a shared test process happens to
have imported.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import List


def _celery_app_source() -> Path:
    """`src/core/celery_app.py`, whether pytest runs from `backend/` or the repo root."""
    for candidate in (
        Path("src/core/celery_app.py"),
        Path("backend/src/core/celery_app.py"),
        Path(__file__).resolve().parents[2] / "src" / "core" / "celery_app.py",
    ):
        if candidate.exists():
            return candidate
    raise AssertionError("celery_app.py not found from any expected root")


@lru_cache(maxsize=1)
def autodiscovered_modules() -> tuple:
    """Every string literal passed to `celery_app.autodiscover_tasks(...)`."""
    tree = ast.parse(_celery_app_source().read_text())
    found: List[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "autodiscover_tasks" or not node.args:
            continue
        listing = node.args[0]
        if not isinstance(listing, (ast.List, ast.Tuple)):
            continue
        found.extend(
            element.value
            for element in listing.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
    assert found, "no autodiscover_tasks(...) call with a literal module list was found"
    return tuple(found)


def assert_autodiscovered(module: str) -> None:
    listing = autodiscovered_modules()
    assert module in listing, (
        f"{module} is not in celery_app.autodiscover_tasks(...), so no worker will "
        f"import it and none of its tasks will exist for a real caller"
    )
