"""No code may union two category maps by hand — it reads `all_categories()` (Feature 034, FR-2).

**Why.** Before 034 the expression `{**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}` (or its
`set(A) | set(B)` cousin) was written at ten sites: the server's tool count, the contract
generator, `mistudio_howto`'s index and seven test guards. Each was a hand-maintained statement of
"which products exist". A second product — miDataworks — would have been seen by whichever sites
someone remembered and missed by the rest, and the guards built to catch a missed site were
themselves sites.

`tools.all_categories()` is the one answer, and `TestTheProductRegistry` pins that it IS the
union. This file keeps the sites on it: it walks every module under `backend/src` and
`backend/tests` by AST (not text, so a docstring that quotes the old expression — this one —
does not trip it) and refuses:

* a dict display unpacking two or more category maps: `{**A_CATEGORY_MODULES, **B}`;
* a `|` whose two sides are category maps, bare or wrapped: `A | B`, `set(A) | set(B)`;
* `.update(<category map>)` — the same union spelled imperatively.

A "category map" is any name or attribute ending in `CATEGORY_MODULES`.

Control (recorded 2026-10-16, task 1.8): reintroducing
`{**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}` in `contract.collect` →
`test_no_module_unions_category_maps_by_hand` red naming `src/mcp_server/contract.py`; restored,
`git diff --stat` clean.
"""

from __future__ import annotations

import ast
import pathlib

BACKEND = pathlib.Path(__file__).resolve().parents[2]
ROOTS = (BACKEND / "src", BACKEND / "tests")


def _is_map(node: ast.AST) -> bool:
    """A category map, bare or wrapped once in `set()`/`dict()`/`list()`/`frozenset()`."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
        "set", "dict", "list", "frozenset"
    ) and len(node.args) == 1:
        node = node.args[0]
    if isinstance(node, ast.Name):
        return node.id.endswith("CATEGORY_MODULES")
    if isinstance(node, ast.Attribute):
        return node.attr.endswith("CATEGORY_MODULES")
    return False


def _unions(source: str) -> list[int]:
    """Line numbers of every hand-written union of category maps in `source`."""
    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            unpacked = [v for k, v in zip(node.keys, node.values) if k is None]
            if len(unpacked) >= 2 and any(_is_map(v) for v in unpacked):
                found.append(node.lineno)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            if _is_map(node.left) and _is_map(node.right):
                found.append(node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "update" and any(_is_map(a) for a in node.args):
                found.append(node.lineno)
    return found


def _modules() -> list[pathlib.Path]:
    return sorted(p for root in ROOTS for p in root.rglob("*.py") if "venv" not in p.parts)


class TestTheScannerCanSeeEachForm:
    """A scan that matches nothing asserts nothing — three guards in this estate shipped one."""

    def test_it_sees_a_dict_display_union(self) -> None:
        assert _unions("x = {**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}\n") == [1]

    def test_it_sees_a_set_union(self) -> None:
        assert _unions("x = set(CATEGORY_MODULES) | set(MILLM_CATEGORY_MODULES)\n") == [1]

    def test_it_sees_a_bare_union_and_an_attribute(self) -> None:
        assert _unions("x = tools.CATEGORY_MODULES | DATAWORKS_CATEGORY_MODULES\n") == [1]

    def test_it_sees_an_update(self) -> None:
        assert _unions("d = dict(CATEGORY_MODULES)\nd.update(MILLM_CATEGORY_MODULES)\n") == [2]

    def test_it_ignores_a_docstring_and_a_single_map(self) -> None:
        src = '"""{**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}"""\nx = dict(CATEGORY_MODULES)\n'
        assert _unions(src) == []

    def test_it_walks_both_trees(self) -> None:
        names = {p.relative_to(BACKEND).as_posix() for p in _modules()}
        assert "src/mcp_server/contract.py" in names
        assert "tests/unit/test_reachability.py" in names
        assert len(names) > 500, len(names)


def test_no_module_unions_category_maps_by_hand() -> None:
    offenders = []
    for path in _modules():
        try:
            lines = _unions(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        offenders += [f"{path.relative_to(BACKEND)}:{n}" for n in lines]
    assert not offenders, (
        "hand-written unions of category maps — read `tools.all_categories()` (or "
        "`external_categories()`) instead, so a product added to the table reaches every site:\n  "
        + "\n  ".join(offenders)
    )
