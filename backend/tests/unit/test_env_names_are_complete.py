"""Every old-prefix environment name this repository reads or sets is a row of the rename table (042 FR-042.8
completeness; TPD-25; TI-4; FTASKS 7.5, 7.6).

DERIVED, NEVER LISTED (memory `source-scrape-guards-fail-open`): the walk discovers names in every place an
environment variable is read or set —
* `backend/src/**/*.py` by AST: `os.environ.get(...)`, `os.getenv(...)`, `os.environ[...]`, `legacy_names.env(...)`
  / `env_flag(...)`, settings aliases (`AliasChoices(...)`, `alias=`/`validation_alias=`), and every
  MODULE-LEVEL string constant fully matching `^(MISTUDIO|MILLM)_[A-Z0-9_]+$` (TI-4: three old names are read
  through constants, `gpu_worker_recycle.py`, `gpu_worker_memory.py`, `gpu_supervisor.py`; a call-only walk
  would fail open);
* compose files: `${NAME…}` references and `environment:` keys;
* Kubernetes manifests: `env[].name` of every container;
* Dockerfiles: `ENV` keys.
An old-prefix name found anywhere must be the `old` (or `new`) of an export `env` record — the census
authority — or one of the two names renamed outright at P3 with their setter (PD-10). The real walk must
discover something of each kind it relies on, and the walker is proven on a fixture tree with one plant per
kind.
"""
from __future__ import annotations

import ast
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from tests.support import legacy_export as le

ROOT = le.REPO_ROOT
OLD_PREFIX = re.compile(r"^(MISTUDIO|MILLM)_[A-Z0-9_]+$")
#: PD-10: names set by this repository for its own children / image and renamed outright at P3 (setter and reader
#: in one commit). EMPTY since the P3 cut (042 FTASKS 12.18): `MIWORKBENCH_WORKER_GPU_UUID` and
#: `MIWORKBENCH_VERSION` replaced them, so an old-prefix name is now either an export row or a finding.
RENAMED_AT_P3: set[str] = set()
#: The planted old-prefix constant, joined at runtime so this file holds no old-prefix switch name
#: (test_cross_repo_flag.py reads every test file's string constants).
PLANT_CONST = "_".join(("MISTUDIO", "PLANT", "CONST"))

KINDS = ("environ.get", "getenv", "environ[]", "legacy_names.env", "settings_alias", "module_constant",
         "compose", "k8s", "dockerfile")
#: Kinds the REAL tree must yield (getenv and environ[] have no site under src/ today; their discovery is
#: proven on the fixture tree). `legacy_names.env` is required by its own test below (the P0-b conversion).
#: Since the P3 cut (FTASKS 12.18) no module-level old-prefix constant remains under src/ (the last one was the
#: renamed worker-card variable), so `module_constant` joins them: proven on the fixture tree only.
REAL_KINDS = ("environ.get", "settings_alias", "compose", "k8s", "dockerfile")

#: The walk's roots (git pathspecs, relative to the repository root). Control K-7.23 empties this list.
ROOTS = {
    "python": (":(glob)backend/src/**/*.py",),
    "compose": (":(glob)docker-compose*.yml",),
    "k8s": (":(glob)k8s/**/*.yaml", ":(glob)k8s/**/*.yml"),
    "dockerfile": (":(glob)**/Dockerfile", ":(glob)**/*.Dockerfile", ":(glob)Dockerfile"),
}

_COMPOSE_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")
_DOCKER_ENV = re.compile(r"^\s*ENV\s+(.+)$", re.MULTILINE)


def _ls(root: Path, specs: tuple[str, ...]) -> list[str]:
    if not specs:
        return []
    out = subprocess.run(["git", "ls-files", "-z", "--", *specs], cwd=root, capture_output=True, check=True)
    return sorted(p for p in out.stdout.decode().split("\0") if p)


def _str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _call_name(fn: ast.expr) -> str:
    if isinstance(fn, ast.Attribute):
        return f"{_call_name(fn.value)}.{fn.attr}"
    return fn.id if isinstance(fn, ast.Name) else ""


def python_names(path: str, text: str) -> list[tuple[str, str, str]]:
    """`(kind, name, where)` for every environment-name site in one module."""
    tree = ast.parse(text, filename=path)
    out: list[tuple[str, str, str]] = []
    for node in tree.body:  # module-level constants (TI-4)
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        s = _str(value) if value is not None else None
        if s and OLD_PREFIX.match(s):
            out.append(("module_constant", s, f"{path}:{node.lineno}"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            first = _str(node.args[0]) if node.args else None
            if name.endswith("environ.get") and first:
                out.append(("environ.get", first, f"{path}:{node.lineno}"))
            elif name.split(".")[-1] == "getenv" and first:
                out.append(("getenv", first, f"{path}:{node.lineno}"))
            elif name in ("legacy_names.env", "legacy_names.env_flag", "env", "env_flag") and first \
                    and re.match(r"^[A-Z][A-Z0-9_]+$", first):
                out.append(("legacy_names.env", first, f"{path}:{node.lineno}"))
            elif name.split(".")[-1] == "AliasChoices":
                for arg in node.args:
                    if _str(arg):
                        out.append(("settings_alias", _str(arg), f"{path}:{node.lineno}"))
            for kw in node.keywords:
                if kw.arg in ("alias", "validation_alias") and _str(kw.value):
                    out.append(("settings_alias", _str(kw.value), f"{path}:{node.lineno}"))
        elif isinstance(node, ast.Subscript) and _call_name(node.value).endswith("environ"):
            s = _str(node.slice)
            if s:
                out.append(("environ[]", s, f"{path}:{node.lineno}"))
    return out


def compose_names(path: str, text: str) -> list[tuple[str, str, str]]:
    out = [("compose", m.group(1), f"{path}:{text[:m.start()].count(chr(10)) + 1}")
           for m in _COMPOSE_REF.finditer(text)]
    doc = yaml.safe_load(text) or {}
    for svc_name, svc in (doc.get("services") or {}).items():
        env = (svc or {}).get("environment") or {}
        keys = env.keys() if isinstance(env, dict) else [str(e).split("=", 1)[0] for e in env]
        out += [("compose", k, f"{path} services.{svc_name}.environment") for k in keys]
    return out


def _containers(obj):
    if isinstance(obj, dict):
        for key in ("containers", "initContainers"):
            for c in obj.get(key) or []:
                if isinstance(c, dict):
                    yield c
        for v in obj.values():
            yield from _containers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _containers(v)


def k8s_names(path: str, text: str) -> list[tuple[str, str, str]]:
    out = []
    for doc in yaml.safe_load_all(text):
        for c in _containers(doc):
            for e in c.get("env") or []:
                if isinstance(e, dict) and e.get("name"):
                    out.append(("k8s", e["name"], f"{path} {c.get('name')}"))
    return out


def dockerfile_names(path: str, text: str) -> list[tuple[str, str, str]]:
    out = []
    for m in _DOCKER_ENV.finditer(text):
        line = text[:m.start()].count("\n") + 1
        body = m.group(1).strip()
        if "=" in body.split()[0]:
            keys = [tok.split("=", 1)[0] for tok in re.findall(r"(\w+=(?:\"[^\"]*\"|\S+))", body)]
        else:
            keys = [body.split()[0]]
        out += [("dockerfile", k, f"{path}:{line}") for k in keys]
    return out


READERS = {"python": python_names, "compose": compose_names, "k8s": k8s_names, "dockerfile": dockerfile_names}


def walk(root: Path = ROOT, roots: dict = ROOTS) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for group, specs in roots.items():
        for rel in _ls(root, specs):
            if group == "dockerfile" and rel.endswith(".dockerignore"):
                continue
            found += READERS[group](rel, (root / rel).read_text(encoding="utf-8", errors="replace"))
    return found


@pytest.fixture(scope="module")
def found() -> list[tuple[str, str, str]]:
    return walk()


def test_no_unlisted_old_prefix_name_in_src(found):
    export = le.load()
    listed = {r["old"] for r in export.env()} | {r["new"] for r in export.env()}
    assert export.env(), "the export lists no env record"
    unlisted = defaultdict(list)
    for kind, name, where in found:
        if OLD_PREFIX.match(name) and name not in listed and name not in RENAMED_AT_P3:
            unlisted[name].append(f"{kind} {where}")
    assert not unlisted, ("old-prefix environment names that are not rows of the export's env records "
                          "(add the row to census and to src/core/legacy_names.py, or rename at P3): "
                          + "; ".join(f"{n}: {w}" for n, w in sorted(unlisted.items())))
    # The runtime reader is only ever asked for a NEW name the table knows.
    news = {r["new"] for r in export.env()}
    wrong = [(n, w) for k, n, w in found if k == "legacy_names.env" and n not in news]
    assert not wrong, f"legacy_names.env() asked for a name that is not a new env name: {wrong}"


def test_discovery_finds_each_source_kind(found, tmp_path):
    # The real walk: something of each kind it relies on.
    real = {k for k, _n, _w in found}
    missing = [k for k in REAL_KINDS if k not in real]
    assert not missing, f"the walk discovered nothing of kind(s) {missing} (roots {sorted(ROOTS)})"
    # The walker on a fixture tree: exactly one plant per kind.
    files = {
        "backend/src/m.py": (
            "import os\nfrom pydantic import AliasChoices, Field\nfrom src.core import legacy_names\n"
            f"K = '{PLANT_CONST}'\n"
            "a = os.environ.get('PLANT_GET')\nb = os.getenv('PLANT_GETENV')\nc = os.environ['PLANT_SUB']\n"
            "d = legacy_names.env('PLANT_ENV')\n"
            "e = Field(validation_alias=AliasChoices('PLANT_ALIAS'))\n"),
        "docker-compose.yml": "services:\n  s:\n    environment:\n      PLANT_COMPOSE_KEY: x\n",
        "k8s/base/a.yaml": "kind: Deployment\nspec:\n  template:\n    spec:\n      containers:\n"
                           "        - name: c\n          env:\n            - name: PLANT_K8S\n",
        "backend/Dockerfile": "FROM x\nENV PLANT_DOCKER=1\n",
    }
    root = tmp_path / "t"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    per_kind = defaultdict(list)
    for kind, name, _w in walk(root):
        per_kind[kind].append(name)
    assert {k: sorted(v) for k, v in per_kind.items()} == {
        "module_constant": [PLANT_CONST], "environ.get": ["PLANT_GET"], "getenv": ["PLANT_GETENV"],
        "environ[]": ["PLANT_SUB"], "legacy_names.env": ["PLANT_ENV"], "settings_alias": ["PLANT_ALIAS"],
        "compose": ["PLANT_COMPOSE_KEY"], "k8s": ["PLANT_K8S"], "dockerfile": ["PLANT_DOCKER"],
    }
    assert set(per_kind) == set(KINDS)


_MODULE_CONST_LINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*(?::[^=]+)?=\s*[\"']((?:MISTUDIO|MILLM)_[A-Z0-9_]+)[\"']\s*(?:#.*)?$",
                                re.MULTILINE)


def test_runtime_reader_is_discovered(found):
    """FTASKS 3.9: the renamed direct reads go through `legacy_names.env(NEW)`; the walk must see them (and
    `test_no_unlisted_old_prefix_name_in_src` checks each asks for a new name the export lists)."""
    asked = sorted({n for k, n, _w in found if k == "legacy_names.env"})
    assert asked, "no legacy_names.env(...) read discovered under backend/src"


def test_constants_are_walked(found):
    """TI-4: every module-level old-prefix constant is walked. Two independent views must agree: the AST walk
    and a line-anchored scan of top-level assignments (a column-0 assignment cannot be a comment)."""
    walked = {(w.split(":")[0], n) for k, n, w in found if k == "module_constant"}
    scanned = set()
    for rel in _ls(ROOT, ROOTS["python"]):
        for m in _MODULE_CONST_LINE.finditer((ROOT / rel).read_text(encoding="utf-8")):
            scanned.add((rel, m.group(1)))
    # Since the P3 cut (FTASKS 12.18) the real tree may hold NO such constant (the last was the worker-card
    # variable), so an empty scan must be shown able to see one: the line scanner on a planted line, the AST walk
    # on the fixture tree (test_discovery_finds_each_source_kind).
    planted = f"K = '{PLANT_CONST}'\nK2: str = \"{PLANT_CONST}\"  # note\n    INDENTED = '{PLANT_CONST}'\n"
    assert [m.group(1) for m in _MODULE_CONST_LINE.finditer(planted)] == [PLANT_CONST, PLANT_CONST]
    assert walked == scanned, (sorted(scanned - walked), sorted(walked - scanned))
    # The P3 rename's setter constant (gpu_supervisor.py) left this set at the cut: neither old name remains.
    assert not any(n in {"_".join(("MISTUDIO", "WORKER", "GPU", "UUID")), "_".join(("MISTUDIO", "VERSION"))}
                   for _p, n in scanned)
