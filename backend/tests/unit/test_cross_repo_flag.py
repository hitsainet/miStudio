"""Test-only switches are read new-first, in one place (042 FR-042.11; FTID §9.2; FTASKS 5.3, control K-5.12).

During the rebrand an operator or CI may set either spelling of a test-only switch
(`MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS` / `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS`, the runner switch, the
real-tokenizer and live-test switches, `MISERVER_PYTHON` / `MILLM_PYTHON`). A test that read the old name
directly would silently stop honouring the new one; a test that read only the new one would stop CI's old
setting. So the only readers are `tests/support/cross_repo.py` (and its copy for the runner-host tests,
`rule_runner/tests/_switches.py`) and `tests/support/sibling_repos.py` (the sibling folder variables).

The guard is an AST walk over `backend/tests/` and `rule_runner/tests/`, never a text search: a comment
or a message may name a switch freely. **Which names are test switches is derived**, not listed: every
string constant shaped `MISTUDIO_*` / `MILLM_*` that a test names and `backend/src` does not (a name
`backend/src` names is runtime configuration, `src/core/legacy_names.py`'s concern), plus the two
switch names `cross_repo.py` declares. An old switch name may appear only where a test WRITES an
environment (`monkeypatch.setenv/delenv`, a dict-literal key, a subscript store) or as the OLD argument of
`test_env(new, old)`; anywhere else — `os.environ.get`, `os.getenv`, `os.environ[...]`, `in os.environ`,
`<mapping>.get`, any other call, a keyword argument, a module constant, a tuple — it is a read or an
indirection to one, and fails.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

from tests.support import cross_repo

REPO = Path(__file__).resolve().parents[3]
ROOTS = (REPO / "backend" / "tests", REPO / "rule_runner" / "tests")
SRC = REPO / "backend" / "src"
OLD_PREFIX = re.compile(r"^(MISTUDIO|MILLM)_[A-Z0-9_]+$")

#: The readers, each pinned to EXACTLY the old names it may hold.
READERS: dict[str, set[str]] = {
    "backend/tests/support/cross_repo.py": {cross_repo.REQUIRE_OLD, cross_repo.RUNNER_OLD},
    "rule_runner/tests/_switches.py": {cross_repo.REQUIRE_OLD, cross_repo.RUNNER_OLD},
    # ENH-009 P7 (FTASKS 18.7): the sibling helper reads no old name any more (`$MILLM_REPO` went with G4).
    "backend/tests/support/sibling_repos.py": set(),
    # The P7 inverse guard names the retired switches to look for them, and the suite switch that stays (18.1).
    "backend/tests/unit/test_legacy_retired.py": {"MILLM_REPO", "MILLM_PYTHON", "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"},
    # This guard names old switches to look for them (and plants them in its self-tests).
    "backend/tests/unit/test_cross_repo_flag.py": {"MILLM_API_URL", "MILLM_PYTHON", "MILLM_REPO",
                                                   "MISTUDIO_BYPASS_PIN", "MISTUDIO_REAL_TOKENIZERS",
                                                   "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS", "MISTUDIO_X_SWITCH"},
}
#: (file, line) -> (count, reason): one LINE each, never a whole file. Each must still match.
EXEMPT: dict[tuple[str, int], tuple[int, str]] = {}
#: Names a reader module exports for a switch name (either spelling).
NAME_ALIASES = {"REQUIRE_OLD", "RUNNER_OLD", "REQUIRE_NEW", "RUNNER_NEW"}
#: The two switches only `require_cross_repo()` / `require_runner()` may read, under EITHER spelling: a test
#: reading the new name alone would stop honouring the old one (and CI's setting).
NEW_SWITCHES = {cross_repo.REQUIRE_NEW, cross_repo.RUNNER_NEW}
WRITERS = {"setenv", "delenv"}
ENV_READS = {"get", "getenv", "pop", "setdefault", "__contains__"}


def _constants(path: Path) -> set[str]:
    return {n.value for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and OLD_PREFIX.match(n.value)}


def runtime_names() -> set[str]:
    out: set[str] = set()
    for path in SRC.rglob("*.py"):
        out |= _constants(path)
    return out


def _files() -> list[Path]:
    return sorted(p for root in ROOTS for p in root.rglob("*.py"))


def switch_names() -> set[str]:
    names = {cross_repo.REQUIRE_OLD, cross_repo.RUNNER_OLD}
    for path in _files():
        names |= _constants(path)
    return (names - runtime_names()) | {cross_repo.REQUIRE_OLD, cross_repo.RUNNER_OLD} | NEW_SWITCHES


def _call_name(call: ast.Call) -> str:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _env_read(node: ast.AST, parents: dict) -> bool:
    """`node` is the variable name in an environment READ: `x.get(node)`, `getenv(node)`, `x[node]`
    (load), `node in os.environ`."""
    par = parents.get(node)
    if isinstance(par, ast.Call) and par.args and par.args[0] is node and _call_name(par) in ENV_READS:
        return True
    if isinstance(par, ast.Subscript) and par.slice is node and isinstance(par.ctx, ast.Load):
        return True
    if isinstance(par, ast.Compare) and par.left is node and any(isinstance(o, (ast.In, ast.NotIn)) for o in par.ops):
        right = par.comparators[0]
        tail = right.attr if isinstance(right, ast.Attribute) else getattr(right, "id", "")
        return tail in ("environ", "env")
    return False


def scan(text: str, rel: str, switches: set[str]) -> list[tuple[str, int, str]]:
    """Every read of (or indirection to) a switch name in one source.

    A switch-name CONSTANT may appear only in a write context (or as `test_env`'s old argument); a NAME
    bound to one (a reader's `REQUIRE_OLD`/`REQUIRE_NEW`, or a local `FLAG = …`) may be used freely —
    in a message, an assertion, a `setenv` — except as the key of an environment read."""
    tree = ast.parse(text)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    aliases = set(NAME_ALIASES)
    for node in ast.walk(tree):  # local names bound to a switch (FLAG = REQUIRE_NEW, FLAG = "…")
        if isinstance(node, ast.Assign):
            v = node.value
            bound = (isinstance(v, ast.Constant) and v.value in switches) or \
                (isinstance(v, ast.Name) and v.id in NAME_ALIASES) or \
                (isinstance(v, ast.Attribute) and v.attr in NAME_ALIASES)
            if bound:
                aliases |= {tgt.id for tgt in node.targets if isinstance(tgt, ast.Name)}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name in aliases and _env_read(node, parents):
                out.append((rel, node.lineno, f"{ast.unparse(node)} (a switch name) read from the environment"))
            continue
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in switches):
            continue
        par = parents.get(node)
        label = node.value
        if isinstance(par, ast.Call):
            name = _call_name(par)
            if name in WRITERS and node in par.args:
                continue
            if name == "test_env" and len(par.args) >= 2 and par.args[1] is node and label not in NEW_SWITCHES:
                continue
            out.append((rel, node.lineno, f"{label} passed to {name or 'a call'}()"))
        elif isinstance(par, ast.keyword):
            call = parents.get(par)
            if isinstance(call, ast.Call) and _call_name(call) == "test_env" and par.arg == "old" \
                    and label not in NEW_SWITCHES:
                continue
            out.append((rel, node.lineno, f"{label} as keyword {par.arg}="))
        elif isinstance(par, ast.Dict) and node in par.keys:
            continue  # a fake environment written by the test
        elif isinstance(par, ast.Subscript) and par.slice is node and not isinstance(par.ctx, ast.Load):
            continue  # os.environ[...] = … / del os.environ[...]
        elif isinstance(par, ast.Expr) and isinstance(parents.get(par), (ast.Module, ast.FunctionDef,
                                                                         ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # a bare string statement
        else:
            out.append((rel, node.lineno, f"{label} read or held ({type(par).__name__})"))
    return out


# ── the tests ──────────────────────────────────────────────────────────────────────────────────────

def _switches_module():
    spec = importlib.util.spec_from_file_location("_switches_copy", REPO / "rule_runner" / "tests" / "_switches.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("spelling", ["new", "old"])
@pytest.mark.parametrize("module_name", ["cross_repo", "_switches"])
@pytest.mark.parametrize("switch", ["require_cross_repo", "require_runner"])
def test_each_spelling_alone_flips_the_flag(monkeypatch, spelling, module_name, switch):
    module = cross_repo if module_name == "cross_repo" else _switches_module()
    new, old = ((module.REQUIRE_NEW, module.REQUIRE_OLD) if switch == "require_cross_repo"
                else (module.RUNNER_NEW, module.RUNNER_OLD))
    assert new.startswith("MIWORKBENCH_") and old.startswith("MISTUDIO_")
    for name in (new, old):
        monkeypatch.delenv(name, raising=False)
    assert getattr(module, switch)() is False
    monkeypatch.setenv(new if spelling == "new" else old, "1")
    assert getattr(module, switch)() is True
    monkeypatch.setenv(new if spelling == "new" else old, "0")
    assert getattr(module, switch)() is False


def test_the_new_name_wins_over_the_old(monkeypatch):
    monkeypatch.setenv(cross_repo.REQUIRE_NEW, "0")
    monkeypatch.setenv(cross_repo.REQUIRE_OLD, "1")
    assert cross_repo.require_cross_repo() is False
    monkeypatch.setenv("ZZ_NEW", "a")
    monkeypatch.setenv("ZZ_OLD", "b")
    assert cross_repo.test_env("ZZ_NEW", "ZZ_OLD") == "a"
    monkeypatch.setenv("ZZ_NEW", "")
    assert cross_repo.test_env("ZZ_NEW", "ZZ_OLD") == "b"
    assert cross_repo.test_env("ZZ_NEW", "ZZ_OLD", environ={}) is None
    assert cross_repo.test_env("ZZ_NEW", "ZZ_OLD", "d", environ={"ZZ_OLD": "x"}) == "x"


def test_the_runner_hosts_copy_equals_cross_repo():
    copy = _switches_module()
    for name in ("REQUIRE_NEW", "REQUIRE_OLD", "RUNNER_NEW", "RUNNER_OLD"):
        assert getattr(copy, name) == getattr(cross_repo, name), name


def test_no_direct_old_prefix_read_in_tests():
    files = _files()
    assert len(files) > 500, f"only {len(files)} test files discovered"
    assert any(p.parts[-3:] == ("rule_runner", "tests", "conftest.py") for p in files), "rule_runner/tests not walked"
    runtime = runtime_names()
    assert {"MILLM_API_URL", "MISTUDIO_BYPASS_PIN"} <= runtime, f"src discovery is blind: {sorted(runtime)}"
    switches = switch_names()
    assert {cross_repo.REQUIRE_OLD, cross_repo.RUNNER_OLD, "MILLM_PYTHON", "MISTUDIO_REAL_TOKENIZERS"} <= switches, \
        f"switch discovery is blind: {sorted(switches)}"
    found, counts = [], {}
    for path in files:
        rel = str(path.relative_to(REPO))
        if rel in READERS:
            held = _constants(path)
            assert held == READERS[rel], f"{rel} holds {sorted(held)}; expected exactly {sorted(READERS[rel])}"
            continue
        for item in scan(path.read_text(encoding="utf-8"), rel, switches):
            key = (item[0], item[1])
            if key in EXEMPT:
                counts[key] = counts.get(key, 0) + 1
            else:
                found.append(f"{item[0]}:{item[1]}: {item[2]}")
    assert not found, ("a test reads an old-prefix test switch directly; use tests/support/cross_repo.py "
                       "(require_cross_repo(), require_runner(), test_env(new, old)):\n" + "\n".join(found))
    for key, (count, reason) in EXEMPT.items():
        assert counts.get(key, 0) == count, f"stale exemption {key} ({reason}): matched {counts.get(key, 0)}"
    for rel in READERS:
        assert (REPO / rel).is_file(), rel


SW = {"MISTUDIO_X_SWITCH", "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS", "MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS"}


@pytest.mark.parametrize("source", [
    'import os\nv = os.environ.get("MISTUDIO_X_SWITCH")\n',
    'import os\nv = os.getenv("MISTUDIO_X_SWITCH", "")\n',
    'import os\nv = os.environ["MISTUDIO_X_SWITCH"]\n',
    'import os\nv = "MISTUDIO_X_SWITCH" in os.environ\n',
    'def f(env):\n    return env.get("MISTUDIO_X_SWITCH")\n',
    'v = _wrapper("MISTUDIO_X_SWITCH")\n',
    'FLAG = "MISTUDIO_X_SWITCH"\n',
    'v = finder(env="MISTUDIO_X_SWITCH")\n',
    'NAMES = ("MIWORKBENCH_X", "MISTUDIO_X_SWITCH")\n',
    'import os\nfrom tests.support.cross_repo import REQUIRE_OLD\nv = os.environ.get(REQUIRE_OLD)\n',
    'import os\nfrom tests.support import cross_repo\nv = os.environ.get(cross_repo.REQUIRE_OLD)\n',
    'v = test_env("MISTUDIO_X_SWITCH", "MIWORKBENCH_X")\n',
    'import os\nv = os.environ.get("MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS") == "1"\n',
    'import os\nfrom tests.support.cross_repo import REQUIRE_NEW\nFLAG = REQUIRE_NEW\nv = os.environ.get(FLAG)\n',
    'import os\nFLAG = REQUIRE_NEW\nv = FLAG in os.environ\n',
    'import os\nv = os.environ[cross_repo.REQUIRE_NEW]\n',
])
def test_the_scanner_finds_each_read_form(source):
    assert scan(source, "planted.py", SW), source


@pytest.mark.parametrize("source", [
    'monkeypatch.setenv("MISTUDIO_X_SWITCH", "1")\n',
    'monkeypatch.delenv("MISTUDIO_X_SWITCH", raising=False)\n',
    'env = {"MISTUDIO_X_SWITCH": "1"}\n',
    'import os\nos.environ["MISTUDIO_X_SWITCH"] = "1"\n',
    'v = test_env("MIWORKBENCH_X", "MISTUDIO_X_SWITCH")\n',
    'v = test_env("MIWORKBENCH_X", old="MISTUDIO_X_SWITCH")\n',
    'pytest.skip("set MISTUDIO_X_SWITCH=1 to make this a failure")\n',
    'FLAG = REQUIRE_NEW\nm = f"set {FLAG}=1"\nmonkeypatch.setenv(FLAG, "1")\nassert FLAG in str(e.value)\n',
])
def test_writes_and_messages_pass(source):
    assert scan(source, "ok.py", SW) == []
