"""`src/core/legacy_names.py`'s hand-kept tables equal the census export (042 FR-042.8; CF-3, CF-24, CF-25 m13;
T-04; FTASKS 7.7).

The runtime module and the export are two INDEPENDENT copies of one register fact (CF-3): census is the
authority, the module is what the running app reads. This test compares them both ways, with `observed` for
hosts (CF-24), and proves the independence by AST: the module defines its tables as literals, imports only the
standard library, and reads no file (CF-25 m13) — a table built from the export would agree by construction and
prove nothing.

On the public mirror the export withholds some host records (they name `.local` hosts); those rows are skipped
BY NAME (`legacy_export.require`), never compared against a list silently missing them.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from src.core import legacy_names
from tests.support import legacy_export as le

MODULE = Path(legacy_names.__file__)
#: Calls and names that would read a file or the export at import time.
FILE_READERS = {"open", "Path", "read_text", "read_bytes", "loads", "load", "files", "import_module",
                "__import__", "exec", "eval"}
FILE_MODULES = {"json", "tomllib", "importlib", "pathlib", "io", "yaml", "csv", "configparser"}


@pytest.fixture(scope="module")
def export() -> le.Export:
    return le.load()


def test_env_rows_equal_the_export(export):
    module = {(r.old, r.new, r.status) for r in legacy_names.ENV}
    exported = {(r["old"], r["new"], r["status"]) for r in export.env()}
    assert exported, "the export lists no env record"
    assert module == exported, (f"only in legacy_names.ENV: {sorted(module - exported)}; "
                                f"only in the export: {sorted(exported - module)}")
    assert len(legacy_names.ENV) == len(module), "a duplicate ENV row"


def test_host_rows_equal_the_export_with_observed(export):
    module = {h.old: (h.new, h.status, h.observed) for h in legacy_names.HOSTS}
    assert len(module) == len(legacy_names.HOSTS), "a duplicate HOSTS row"
    exported = {r["old"]: (r["new"], r["status"], r["observed"]) for r in export.hosts()}
    assert exported, "the export lists no host record"
    assert set(exported) <= set(module), f"export hosts missing from the module: {sorted(set(exported) - set(module))}"
    for old, row in module.items():
        if old not in exported:
            # Only a host the PUBLIC variant withholds may be absent, and it is skipped by name.
            assert export.is_public and export.is_withheld("host", old), f"module-only host {old}"
            continue
        assert row == exported[old], (old, row, exported[old])
    withheld = [old for old in module if old not in exported]
    if withheld:
        le.require(export, "host", withheld[0])


def _imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add("." * node.level + (node.module or ""))
            else:
                names.add((node.module or "").split(".")[0])
    return names


def test_module_is_stdlib_only():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imports = _imports(tree)
    assert imports, "the module imports nothing: the AST walk found no import (a broken scan asserts nothing)"
    outside = sorted(n for n in imports if n.startswith(".") or n not in sys.stdlib_module_names
                     and n != "__future__")
    assert not outside, f"src/core/legacy_names.py imports outside the standard library: {outside}"


def _table_value(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id == name:
            return node.value
    raise AssertionError(f"no module-level assignment of {name}")


def test_tables_are_literals_and_no_file_is_read():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    # ENV / HOSTS: a tuple of constructor calls whose arguments are all constants.
    for name, ctor in (("ENV", "EnvName"), ("HOSTS", "HostAlias")):
        value = _table_value(tree, name)
        assert isinstance(value, ast.Tuple) and value.elts, f"{name} is not a literal tuple"
        for elt in value.elts:
            assert isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name) and elt.func.id == ctor, \
                f"{name} row is not a {ctor}(...) literal: {ast.unparse(elt)}"
            args = [*elt.args, *(k.value for k in elt.keywords)]
            assert all(isinstance(a, ast.Constant) for a in args), f"{name} row computes a value: {ast.unparse(elt)}"
    aliases = _table_value(tree, "CATEGORY_ALIASES")
    assert isinstance(aliases, ast.Dict) and all(isinstance(x, ast.Constant) for x in [*aliases.keys, *aliases.values])
    # No file read anywhere in the module, and no module that reads files imported.
    readers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            called = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if called in FILE_READERS:
                readers.append(f"line {node.lineno}: {ast.unparse(node)[:80]}")
    assert not readers, readers
    assert not (_imports(tree) & FILE_MODULES), sorted(_imports(tree) & FILE_MODULES)
