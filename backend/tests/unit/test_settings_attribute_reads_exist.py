"""Every read of the `settings` object must name a field `Settings` really has.

WHY THIS EXISTS

`_host_may_receive_stored_key` (api/v1/endpoints/labeling.py) built the set of
hosts allowed to receive the operator's stored OpenAI key from two candidates:
the database setting, and `getattr(settings, "openai_compatible_endpoint",
None)`. `Settings` has no such field, so the second candidate was always None.
It read like an environment override and never was one (found by the ENH-003
settings inventory, 2026-10-20).

`getattr(obj, name, default)` is exactly the shape that hides this: a typo or a
field that was never added returns the default instead of raising, and nothing
anywhere reports it. A plain `settings.missing` raises AttributeError, but only
on the line's first execution, which on a rarely-taken branch is production.

HOW IT WORKS

AST, not a regex (a regex over source matches nothing on an unexpected layout and
asserts nothing). Every module under `src/` that binds `settings` from
`core.config` — at module level or inside a function, under any alias — is
parsed, and each `settings.<name>`, `getattr(settings, "<name>"...)`,
`hasattr(...)` and `setattr(...)` is resolved against the LIVE class:
`Settings.model_fields` plus the attributes the class itself defines (methods,
properties, the pydantic API). A function that rebinds the name locally
(`settings = [...]`) is skipped for that scope, since there it is not the config
object.

It fails closed in three ways: a floor on how many reads it found (a scanner that
silently found nothing would otherwise pass), a check that the scan reached the
modules known to read settings, and a refusal of any `getattr` whose name is not
a string literal (it cannot be checked, so it must not be written).

NEGATIVE CONTROL (2026-10-20): restoring the old two-candidate loop in
`labeling.py` makes `test_every_settings_read_names_a_real_field` fail with
`labeling.py:<line> getattr 'openai_compatible_endpoint'`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# Reads found on 2026-10-20: 203. The floor sits well below so ordinary
# refactors do not trip it, and far above zero so a scanner that matches nothing
# cannot pass.
_MIN_READS = 120

_ACCESSORS = ("getattr", "hasattr", "setattr")


def _is_config_import(node: ast.ImportFrom) -> bool:
    module = node.module or ""
    return module == "core.config" or module.endswith(".core.config")


def _imported_names(body_nodes) -> set[str]:
    """Names bound to the config `settings` by imports directly in a scope
    (not inside nested functions or classes)."""
    names: set[str] = set()
    stack = list(body_nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.ImportFrom) and _is_config_import(node):
            for alias in node.names:
                if alias.name == "settings":
                    names.add(alias.asname or "settings")
        stack.extend(ast.iter_child_nodes(node))
    return names


def _locally_rebound(func) -> set[str]:
    """Names a function assigns or takes as parameters — local everywhere in it."""
    bound: set[str] = set()
    args = func.args
    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
        if a is not None:
            bound.add(a.arg)
    stack = list(func.body) if isinstance(func.body, list) else [func.body]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return bound


def scan_source(source: str, filename: str = "<string>") -> list[tuple[str, int, str, str | None]]:
    """Every access to the config `settings` in one module.

    Returns `(filename, lineno, kind, name)`: kind is "attr" or an accessor
    name; name is None when a getattr-style call's name is not a literal.
    """
    tree = ast.parse(source, filename=filename)
    found: list[tuple[str, int, str, str | None]] = []

    def visit(node, live: frozenset[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            body = node.body if isinstance(node.body, list) else [node.body]
            rebound = _locally_rebound(node)
            imported = _imported_names(body)
            live = frozenset((live - rebound) | imported)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in live:
            found.append((filename, node.lineno, "attr", node.attr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _ACCESSORS and node.args \
                and isinstance(node.args[0], ast.Name) and node.args[0].id in live:
            name_node = node.args[1] if len(node.args) > 1 else None
            name = (name_node.value
                    if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)
                    else None)
            found.append((filename, node.lineno, node.func.id, name))
        for child in ast.iter_child_nodes(node):
            visit(child, live)

    visit(tree, frozenset(_imported_names(tree.body)))
    return found


def _valid_names() -> set[str]:
    from src.core.config import Settings

    return set(Settings.model_fields) | set(dir(Settings))


def _scan_tree() -> list[tuple[str, int, str, str | None]]:
    reads = []
    for path in sorted(SRC.rglob("*.py")):
        reads.extend(scan_source(path.read_text(), str(path.relative_to(SRC))))
    return reads


def test_every_settings_read_names_a_real_field():
    valid = _valid_names()
    reads = _scan_tree()

    assert len(reads) >= _MIN_READS, (
        f"found only {len(reads)} reads of `settings` under src/ — the scanner "
        "has stopped seeing them, so this guard would pass vacuously"
    )

    unknown = [f"{f}:{ln} {kind} {name!r}" for f, ln, kind, name in reads
               if name is not None and name not in valid]
    assert not unknown, (
        "these read a field `Settings` does not have; getattr returns its default "
        "forever and a bare attribute raises on first use:\n  " + "\n  ".join(unknown)
    )

    dynamic = [f"{f}:{ln} {kind}" for f, ln, kind, name in reads if name is None]
    assert not dynamic, (
        "getattr/hasattr/setattr on `settings` with a non-literal name cannot be "
        "checked against the field list:\n  " + "\n  ".join(dynamic)
    )


def test_the_scan_reaches_modules_known_to_read_settings():
    """If the import detection broke, these files would drop out silently."""
    files = {f for f, *_ in _scan_tree()}
    for expected in ("api/v1/endpoints/labeling.py",
                     "api/v1/endpoints/settings.py",
                     "services/gpu_dispatch.py"):
        assert expected in files, f"{expected} reads settings but was not scanned"


class TestTheScannerItself:
    """Negative controls on synthetic modules: each case must be SEEN."""

    def test_it_flags_a_getattr_of_a_missing_field(self):
        src = (
            "from src.core.config import settings\n"
            "x = getattr(settings, 'openai_compatible_endpoint', None)\n"
        )
        reads = scan_source(src)
        assert reads == [("<string>", 2, "getattr", "openai_compatible_endpoint")]
        assert "openai_compatible_endpoint" not in _valid_names()

    def test_it_sees_a_function_local_aliased_import(self):
        src = (
            "def f():\n"
            "    from ...core.config import settings as cfg\n"
            "    return cfg.no_such_field\n"
        )
        assert scan_source(src) == [("<string>", 3, "attr", "no_such_field")]

    def test_it_reports_a_non_literal_name(self):
        src = (
            "from ..core.config import settings\n"
            "def f(k):\n"
            "    return getattr(settings, k)\n"
        )
        assert scan_source(src) == [("<string>", 3, "getattr", None)]

    def test_a_local_rebinding_is_not_the_config_object(self):
        src = (
            "from ..core.config import settings\n"
            "def f(rows):\n"
            "    settings = rows\n"
            "    return settings.append\n"
            "y = settings.data_dir\n"
        )
        assert scan_source(src) == [("<string>", 5, "attr", "data_dir")]

    def test_other_settings_objects_are_not_confused_for_it(self):
        src = "from x.y import settings\nz = settings.anything\n"
        assert scan_source(src) == []

    @pytest.mark.parametrize("name", ["database_url", "bypass_settings_pin", "model_dump"])
    def test_real_fields_and_the_model_api_are_valid(self, name):
        assert name in _valid_names()
