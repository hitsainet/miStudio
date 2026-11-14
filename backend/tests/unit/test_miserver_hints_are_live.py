"""Every `error.next` hint miServer can send names a tool THIS MCP server registers
(042 FR-042.20, BR-W07; FTDD §6.8; FTID §3.11; FTASKS 7.12, 7.13).

Cross-repo: reads miServer's checkout (`sibling_repos.miserver_repo()`, either name). Skips loudly without
it; fails under `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1` (or the suite-wide old spelling). The authority
used is printed (`-s`/`-rP`): the published `next-tool-hints.json` once 040's P0 is on miServer's main,
else the source. The functions are proven red on a temporary package by test_miserver_hints_fixture.py.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.support import miserver_hints as mh
from tests.support.cross_repo import require_cross_repo
from tests.support.next_calls import live_input_schemas
from tests.support.sibling_repos import miserver_import, miserver_repo


def test_every_published_hint_is_live():
    repo = miserver_repo()
    names, authority = mh.checked_hints(repo)
    print(f"miServer hints: {len(names)} names, authority: {authority}, repo: {repo}")
    assert names, "miServer sends no hint at all: the check would read nothing"
    live = set(live_input_schemas())
    relay = mh.built_relay_sets() if any(n not in live for n in names) else []
    problems = mh.liveness_problems(names, live, relay)
    assert problems == [], "\n".join(problems)


def test_source_hints_are_published():
    repo = miserver_repo()
    mh.checked_hints(repo)  # skip/fail rules
    published = mh.published_hints(repo)
    if published is None:
        pytest.skip("040's next-tool-hints.json is not on this miServer checkout: source is the authority")
    source = mh.source_hints(repo)
    assert source, "no hint found in miServer's source: the AST walk read nothing"
    assert source <= published, sorted(source - published)


_DUMP = (
    "import importlib, json, sys\n"
    "mod = importlib.import_module(sys.argv[1])\n"
    "names = sorted({c.__dict__['next_tool'] for c in vars(mod).values()\n"
    "                if isinstance(c, type) and issubclass(c, BaseException)\n"
    "                and isinstance(c.__dict__.get('next_tool'), str)})\n"
    "print('@@HINTS@@' + json.dumps(names))\n"
)


def test_error_module_subclasses_subset_of_ast():
    """FR-042.20 step 2: miServer's own interpreter imports its error module; every class-level hint it
    holds must be in the AST set, so the AST walk cannot miss a form the runtime uses."""
    repo = miserver_repo()
    mh.checked_hints(repo)
    python = Path(os.environ.get("MISERVER_PYTHON") or repo / "venv" / "bin" / "python")
    if not python.is_file():
        if require_cross_repo():
            pytest.fail(f"no miServer interpreter at {python} (set MISERVER_PYTHON)")
        pytest.skip(f"CROSS-REPO, SKIPPED LOUDLY: no miServer interpreter at {python}")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_URL", "PYTHONPATH"))}
    proc = subprocess.run([str(python), "-B", "-c", _DUMP, miserver_import("core.errors", repo)], cwd=repo,
                          env=env, capture_output=True, text=True, timeout=240)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@HINTS@@")]
    assert proc.returncode == 0 and lines, f"miServer's error module did not import:\n{proc.stderr[-2000:]}"
    runtime = set(json.loads(lines[0][len("@@HINTS@@"):]))
    assert runtime, "miServer's error module holds no class-level hint: the cross-check reads nothing"
    assert runtime <= mh.source_hints(repo), sorted(runtime - mh.source_hints(repo))
