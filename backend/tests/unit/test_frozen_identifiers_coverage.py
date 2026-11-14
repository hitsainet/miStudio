"""Every frozen entry is exercised, and every exercise is named by an entry (042 FR-042.7; T-05; FTASKS 6.5).

The export (`docs/legacy-identifiers.md`, census) names, per entry this repository carries, one `exercise`:
`pytest:backend/tests/unit/test_frozen_identifiers.py::<function>` or
`vitest:frontend/src/<…>.test.ts::<test name>` — or the `history_only` flag (TI-6). This file checks the
bookkeeping in both directions, resolving ids by AST (pytest) and by the TypeScript-shaped declaration of a
test (Vitest; the frontend half `legacyExport.test.ts` resolves the same ids with the TypeScript compiler):

* no record is `pending` (CF-1: the first export after handoff is red until every exercise exists);
* each exercise resolves to a test that runs: a module-level function of that name in that file (a bracketed
  parametrize id must be one of its decorator's ids), or an `it`/`test` of that exact title that is not
  skipped, not todo, not under a skipped `describe`, in a file with no `.only`;
* every test function in `test_frozen_identifiers.py` is named by some entry (TI-7), so an exercise
  cannot be deleted from the register while its test stays, nor a test added nobody owns.

Controls: K-6.13a (a record flagged pending in the export → red), K-6.17 (an exercise naming a missing test
→ red).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.support import legacy_export
from tests.support.legacy_export import REPO_ROOT

FROZEN_FILE = "backend/tests/unit/test_frozen_identifiers.py"


def _export() -> legacy_export.Export:
    return legacy_export.load()


def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _param_ids(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str] | None:
    """The ids a `@pytest.mark.parametrize(..., ids=[...])` decorator gives, or None when it gives none."""
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "parametrize":
            for kw in dec.keywords:
                if kw.arg == "ids" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    return {e.value for e in kw.value.elts if isinstance(e, ast.Constant)}
            values = dec.args[1] if len(dec.args) > 1 else None
            if isinstance(values, (ast.List, ast.Tuple)):
                return {str(e.value) for e in values.elts if isinstance(e, ast.Constant)}
    return None


_COMMENT = re.compile(r"""//[^\n]*|/\*.*?\*/|(?P<s>'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"|`(?:\\.|[^`\\])*`)""", re.DOTALL)
#: Chain members that leave a test or describe running unconditionally; any other (`skip`, `todo`, `skipIf`,
#: `runIf`, `only`, `fails`) refuses (review rev-mistudio-p0 F1: ten skip forms read green).
_NEUTRAL = {"concurrent", "sequential", "shuffle", "each", "for"}
_CONDITIONAL_SCOPE = re.compile(r"(?:\b(?:if|else|for|while|do|switch|try|catch|finally)\b[^{;]*|\bfunction\s+\w+\s*\([^)]*\)\s*)$")


def _strip_comments(source: str) -> str:
    """Comments removed, strings kept (a `//` inside a URL string is not a comment)."""
    return _COMMENT.sub(lambda m: m.group("s") or "", source)


def _open_scopes(code: str, end: int) -> list[str]:
    """The text just before each `{` still open at offset `end` (strings blanked first)."""
    blank = re.sub(r"""'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"|`(?:\\.|[^`\\])*`""", lambda m: '""', code[:end])
    stack: list[str] = []
    last = 0
    for i, ch in enumerate(blank):
        if ch == "{":
            stack.append(blank[max(last, i - 200):i])
            last = i + 1
        elif ch == "}":
            if stack:
                stack.pop()
            last = i + 1
        elif ch in ";":
            last = i + 1
    return stack


def vitest_problem(source: str, name: str) -> str | None:
    """Why a Vitest id would not run as a test of that exact name, or None. Comments are stripped first, so a
    commented-out test never counts (the TypeScript-compiler version lives in `legacyExport.ts`). Refuses any
    skip/todo/skipIf/runIf/only/fails member on the test or on any describe in the file, a skip/todo/only/fails
    options object, a runtime `skip()` call, a declaration under if/else/loop/try or in a named function."""
    code = _strip_comments(source)
    title = re.escape(name)
    quote = r"""(['"`])"""
    declared = re.compile(rf"\b(x?it|x?test)((?:\.\w+)*)\(\s*{quote}{title}\3")
    matches = list(declared.finditer(code))
    if not matches:
        return f"no it/test named {name!r}"
    for m in matches:
        parts = [p for p in m.group(2).split(".") if p]
        if m.group(1).startswith("x") or any(p not in _NEUTRAL for p in parts):
            return f"the test {name!r} is skipped or todo"
        after = code[m.end():m.end() + 400]
        if re.match(r"\s*,\s*\{[^}]*\b(?:skip|todo|only|fails)\s*:", after):
            return f"the test {name!r} is skipped or todo (an options object)"
        if any(_CONDITIONAL_SCOPE.search(s.rstrip()) for s in _open_scopes(code, m.start())):
            return f"the test {name!r} is declared conditionally or inside a named function"
    if re.search(r"\b(?:describe|suite)(?:\.\w+)*\.(?:skip|todo|skipIf|runIf|fails)\b|\bxdescribe\s*\(", code):
        return "a describe block in the file is skipped"
    if re.search(r"\b(?:describe|suite)\s*\(\s*(['\"`]).*?\1\s*,\s*\{[^}]*\b(?:skip|todo|only)\s*:", code):
        return "a describe block in the file is skipped (an options object)"
    if re.search(r"\.skip\s*\(\s*\)|(?<![.\w])skip\s*\(\s*\)", code):
        return "a test in the file calls skip() at run time"
    if re.search(r"\b(?:it|test|describe|suite)(?:\.\w+)*\.only\s*\(", code):
        return "the file uses .only, which skips every other test"
    return None


def exercise_problem(exercise: str, root: Path = REPO_ROOT) -> str | None:
    runner, rel, name = legacy_export.parse_exercise(exercise)
    path = root / rel
    if not path.is_file():
        return f"{rel} does not exist"
    if runner == "vitest":
        return vitest_problem(path.read_text(encoding="utf-8"), name)
    base, _, param = name.partition("[")
    if "::" in base:
        return f"{name}: class-scoped exercises are not used here"
    fn = _functions(path).get(base)
    if fn is None or not base.startswith("test_"):
        return f"no module-level test function {base} in {rel}"
    if param:
        ids = _param_ids(fn)
        if ids is None or param.rstrip("]") not in ids:
            return f"{base} has no parametrize id {param.rstrip(']')!r} (ids: {sorted(ids or [])})"
    return None


def _entry_id_of(test_name: str) -> str | None:
    m = re.match(r"test_dr(\d+)_", test_name)
    return f"DR-{m.group(1)}" if m else None


def test_no_pending_record():
    export = _export()
    assert export.entries(), "the export lists no entry"
    pending = export.pending_ids()
    assert not pending, f"unexercised frozen entries (flag `pending`): {pending}"


def test_every_entry_has_a_resolvable_exercise():
    export = _export()
    problems = []
    exercised = 0
    for entry in export.entries():
        if entry["exercise"] is None:
            if "history_only" not in entry["flags"]:
                problems.append(f"{entry['id']}: no exercise and not history_only (unexercised frozen entry)")
            continue
        exercised += 1
        why = exercise_problem(entry["exercise"])
        if why:
            problems.append(f"{entry['id']}: {entry['exercise']}: {why}")
    assert exercised >= 20, f"only {exercised} exercised entries: the export is not what this test expects"
    assert not problems, "\n".join(problems)
    # The resolver refuses what would not run.
    assert vitest_problem("it('a b', () => {})", "a b") is None
    for bad in ("it.skip('a b', () => {})", "test.todo('a b')", "xit('a b', () => {})",
                "// it('a b', () => {})", "describe.skip('d', () => { it('a b', () => {}) })",
                "it('a b', () => {}); it.only('c', () => {})"):
        assert vitest_problem(bad, "a b"), bad


def test_every_test_in_the_frozen_file_is_named_by_an_entry():
    export = _export()
    named = {legacy_export.parse_exercise(e["exercise"])[2].partition("[")[0]
             for e in export.entries()
             if e["exercise"] and legacy_export.parse_exercise(e["exercise"])[1] == FROZEN_FILE}
    tree = ast.parse((REPO_ROOT / FROZEN_FILE).read_text(encoding="utf-8"))
    tests = {n.name for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")}
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef) and n.name.startswith("Test")]
    imported = [a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                for a in n.names if (a.asname or a.name).startswith(("test_", "Test"))]
    assert tests, "the frozen file holds no test"
    assert not classes, f"test classes in the frozen file are named by no entry: {classes}"
    assert not imported, f"imported test functions would be collected unnamed: {imported}"
    # On the mirror the public export withholds some entries (by id); only THEIR tests may go unnamed there.
    withheld = {w for w in export.withheld if re.fullmatch(r"DR-\d+", w)} if export.is_public else set()
    unnamed = {t for t in tests - named if _entry_id_of(t) not in withheld}
    assert not unnamed, f"tests no entry names: {sorted(unnamed)}"
    if not export.is_public:  # on the mirror, the withheld entries' tests are named by no visible record
        assert named - tests == set(), f"entries naming a missing test: {sorted(named - tests)}"


#: Review rev-mistudio-p0 F1 (the miLLM T03 finding): each form keeps the named test from running, and each
#: read as a runnable exercise before. `legacyExport.test.ts` holds the TypeScript resolver to the same list.
SKIP_FORMS = [
    ('describe.skipIf', "describe.skipIf(true)('d', () => { it('the name', () => {}); });"),
    ('describe.runIf(false)', "describe.runIf(false)('d', () => { it('the name', () => {}); });"),
    ('it.skipIf', "it.skipIf(true)('the name', () => {});"),
    ('it.runIf(false)', "it.runIf(false)('the name', () => {});"),
    ('it {skip: true}', "it('the name', { skip: true }, () => {});"),
    ('it {todo: true}', "it('the name', { todo: true }, () => {});"),
    ('describe {skip: true}', "describe('d', { skip: true }, () => { it('the name', () => {}); });"),
    ('ctx.skip()', "it('the name', (ctx) => { ctx.skip(); });"),
    ('destructured skip()', "test('the name', ({ skip }) => { skip(); });"),
    ('if (false)', "if (false) { it('the name', () => {}); }"),
    ('never-called helper', "function later() { it('the name', () => {}); }"),
    ('describe.concurrent.skip', "describe.concurrent.skip('d', () => { it('the name', () => {}); });"),
    ('it.fails', "it.fails('the name', () => {});"),
]


@pytest.mark.parametrize("form,source", SKIP_FORMS, ids=[f for f, _ in SKIP_FORMS])
def test_the_resolver_refuses_every_skip_form(form, source):
    assert vitest_problem(source, "the name"), f"{form} reads as a runnable exercise"


def test_the_resolver_still_accepts_the_real_exercise_shapes():
    for ok in ("describe('d', () => { it('the name', () => { if (x) { y(); } }); });",
               "describe('d', () => it('the name', () => {}));",
               "it.concurrent('the name', () => {});",
               "const u = 'https://x/y'; describe('d', () => { it('the name', () => {}); });",
               "function helper() { return 1; }\ndescribe('d', () => { it('the name', () => {}); });"):
        assert vitest_problem(ok, "the name") is None, ok
