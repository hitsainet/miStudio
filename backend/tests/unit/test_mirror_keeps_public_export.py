"""The mirror keeps the PUBLIC legacy-identifier export and nothing of the full one (042 CF-2, T-47;
FTASKS 7.3, 7.4; the miLLM/miForge P0 review finding on the mirror's verify step).

`docs/legacy-identifiers.md` names private hosts, database names and the private repository; only
`docs/legacy-identifiers.public.md` may reach `hitsainet/*`. Two steps of `sync-to-clean.yml` decide it,
and both are RUN here from the parsed YAML, never grepped:

* `Remove excluded files and folders` (the strip) runs on a temporary tree shaped like this repository;
  what is left of `docs/` must be exactly the schemas, the MCP contract and the public export, and the
  loader must read that tree as the public variant.
* `Verify the published mirror` (the check that runs against the PUSHED result) runs its own check lines
  on a one-commit git tree: it must refuse a tree holding the full export, and refuse one missing the
  public export, the schemas or the contract.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.support import legacy_export as le

REPO_ROOT = le.REPO_ROOT
SYNC = REPO_ROOT / ".github" / "workflows" / "sync-to-clean.yml"
GATE_SHA_EXPR = "${{ needs.gate.outputs.sha }}"


def _step(name: str) -> str:
    steps = yaml.safe_load(SYNC.read_text())["jobs"]["sync"]["steps"]
    matches = [s for s in steps if s.get("name") == name]
    assert len(matches) == 1, f"sync-to-clean.yml has {len(matches)} steps named {name!r}"
    return matches[0]["run"]


def _bash(script: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    full = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    full.update(env or {})
    return subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=cwd, env=full,
                          capture_output=True, text=True, check=False, timeout=60)


def _repo_shaped_tree(root: Path) -> None:
    for rel in ("0xcc/x.md", ".claude/x", "backups/x.dump", "scripts/k8s-helpers.sh", ".understand-anything/g.json",
                "docs/schemas/probe-definition-v1.json", "docs/mcp-contract.md", "docs/other-notes.md",
                "docs/plans/x.md", "backend/src/keep.py"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    (root / "CLAUDE.md").write_text("private\n")
    full, public = le.paths()
    # The real exports when present; the mirror holds only the public one, so plant a full-shaped stand-in.
    if public.exists():
        shutil.copy(public, root / "docs" / le.PUBLIC_NAME)
    (root / "docs" / le.FULL_NAME).write_text(full.read_text() if full.exists() else "full export\n")


def test_sync_step_keeps_only_public_export(tmp_path):
    if not le.paths()[1].exists():
        pytest.fail(f"{le.paths()[1]} is missing: the mirror's tests read only that file")
    _repo_shaped_tree(tmp_path)
    res = _bash(_step("Remove excluded files and folders"), tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert sorted(p.name for p in (tmp_path / "docs").iterdir()) == sorted(
        ["schemas", "mcp-contract.md", le.PUBLIC_NAME]), sorted(p.name for p in (tmp_path / "docs").iterdir())
    for gone in ("0xcc", ".claude", "backups", "scripts", ".understand-anything", "CLAUDE.md"):
        assert not (tmp_path / gone).exists(), gone
    assert (tmp_path / "backend" / "src" / "keep.py").exists()
    # The loader reads the stripped tree as the public variant and names what it withholds.
    export = le.load(root=tmp_path)
    assert export.is_public and export.path.name == le.PUBLIC_NAME
    assert export.withheld, "the public export withholds nothing: the full one would be needed on the mirror"
    entry_held = [n for n in export.withheld if ":" not in n]
    if entry_held:
        with pytest.raises(pytest.skip.Exception, match=entry_held[0]):
            le.require(export, "entry", entry_held[0])


# ── the Verify step, run on a published-shaped tree ─────────────────────────────────────────────────
def _verify_checks() -> str:
    """The Verify step's own check lines: everything after it moves into the clone."""
    script = _step("Verify the published mirror")
    marker = "cd /tmp/verify\n"
    assert script.count(marker) == 1, "the Verify step no longer clones into /tmp/verify: update this test"
    checks = script.split(marker, 1)[1]
    assert checks.count(GATE_SHA_EXPR) >= 1, "the Verify step no longer checks the Source-Commit trailer"
    return checks.replace(GATE_SHA_EXPR, "${EXPECTED_SHA}")


def _published(tmp_path: Path, *, full: bool, public: bool = True, schemas: bool = True,
               contract: bool = True) -> Path:
    root = tmp_path / "mirror"
    root.mkdir()
    files = {"README.md": "x\n", "backend/src/keep.py": "x\n"}
    if schemas:
        files["docs/schemas/probe-definition-v1.json"] = "{}\n"
    if contract:
        files["docs/mcp-contract.md"] = "x\n"
    if public:
        files[f"docs/{le.PUBLIC_NAME}"] = "public\n"
    if full:
        files[f"docs/{le.FULL_NAME}"] = "full\n"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    run = lambda *a: subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, env={**os.environ, **env})  # noqa: E731
    run("init", "-q", "-b", "main")
    run("add", "-A")
    run("commit", "-q", "-m", "snapshot", "--trailer", "Source-Commit: " + "a" * 40,
        "--trailer", "Private-CI: passed")
    return root


def test_verify_accepts_a_clean_snapshot(tmp_path):
    res = _bash(_verify_checks(), _published(tmp_path, full=False), {"EXPECTED_SHA": "a" * 40})
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Mirror verified" in res.stdout


def test_verify_refuses_the_full_export(tmp_path):
    res = _bash(_verify_checks(), _published(tmp_path, full=True), {"EXPECTED_SHA": "a" * 40})
    assert res.returncode != 0, res.stdout
    assert f"LEAK — 'docs/{le.FULL_NAME}'" in res.stdout, res.stdout


@pytest.mark.parametrize("missing", ["public", "schemas", "contract"])
def test_verify_requires_the_public_files(tmp_path, missing):
    res = _bash(_verify_checks(), _published(tmp_path, full=False, **{missing: False}), {"EXPECTED_SHA": "a" * 40})
    assert res.returncode != 0, res.stdout
    assert "MISSING" in res.stdout, res.stdout
