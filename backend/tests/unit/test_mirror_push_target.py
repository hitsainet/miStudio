"""Everything the sync job sends goes to the PUBLIC mirror of this repository, and nowhere else
(042 PD-16, FR-042.12; the miDataworks P0 review finding "nothing pinned the mirror push target").

A sync that pushed the orphan snapshot to the private repository would replace private `main` with a
one-commit snapshot; one that pushed to any other public name would publish there. Nothing would notice:
the job is green either way. So the target is pinned three ways, all from the PARSED workflow:

1. `MIRROR` is `<public owner>/${{ github.event.repository.name }}` exactly, where the public owner is the
   one the mirror's own image build is gated on and is not the private owner (`workflow_gates`);
2. every `git push`, `git clone` and `git ls-remote` in the sync job uses `"$REMOTE"`, and every step that
   uses it defines `REMOTE` as the token URL of `${MIRROR}` and nothing else;
3. every command line that carries the mirror token (`SYNC_TOKEN`) addresses `${MIRROR}`.
"""
from __future__ import annotations

import re
import shlex

import yaml

from tests.support import legacy_export as le
from tests.support import workflow_gates as wg

SYNC = le.REPO_ROOT / ".github" / "workflows" / "sync-to-clean.yml"
REMOTE_DEF = 'REMOTE="https://x-access-token:${SYNC_TOKEN}@github.com/${MIRROR}.git"'


def _sync_job() -> dict:
    return yaml.safe_load(SYNC.read_text())["jobs"]["sync"]


def _logical_lines(script: str) -> list[str]:
    """Shell lines with backslash continuations joined and comments dropped."""
    joined = re.sub(r"\\\n\s*", " ", script)
    out = []
    for line in joined.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _git_remote_commands(lines: list[str]) -> list[tuple[str, list[str]]]:
    found = []
    for s in lines:
        if re.search(r"\bgit\s+(push|clone|ls-remote)\b", s):
            for part in re.split(r"\$\(|\||;|&&", s):
                m = re.search(r"\bgit\s+(push|clone|ls-remote)\b(.*)", part)
                if m:
                    found.append((m.group(1), shlex.split(m.group(2).rstrip(")"), posix=True)))
    return found


def test_the_mirror_variable_is_derived_and_public():
    export = le.load()
    if export.is_public and any(n.startswith("repo:sha256:") for n in export.withheld):
        import pytest
        pytest.skip("the private repository's repo records are withheld from the public export")
    workflows = wg.load_workflows()
    private, public = wg.private_and_public_owner(workflows, export)
    mirror = _sync_job()["env"]["MIRROR"]
    assert mirror == f"{public}/${{{{ github.event.repository.name }}}}", mirror
    assert not mirror.startswith(private + "/")
    assert "github.repository }}" not in mirror, "MIRROR must never be the repository the job runs in"


def test_every_push_in_the_sync_job_goes_to_the_mirror():
    steps = _sync_job()["steps"]
    pushes = 0
    for step in steps:
        lines = _logical_lines(step.get("run") or "")
        cmds = _git_remote_commands(lines)
        if not cmds:
            continue
        defs = [s for s in lines if s.startswith("REMOTE=")]
        assert defs == [REMOTE_DEF], f"step {step.get('name')!r} defines REMOTE as {defs}"
        assert step.get("env", {}).get("SYNC_TOKEN") == "${{ secrets.HITSAINET_SYNC }}", step.get("name")
        for verb, args in cmds:
            positional = [a for a in args if not a.startswith("-")]
            assert positional and positional[0] == "$REMOTE", (
                f"step {step.get('name')!r}: git {verb} {args} does not address \"$REMOTE\"")
            pushes += verb == "push"
    # Fail closed: the main push, the release tag push and the stale-tag retarget push.
    assert pushes == 3, pushes


def test_every_use_of_the_mirror_token_addresses_the_mirror():
    uses = 0
    for step in _sync_job()["steps"]:
        for s in _logical_lines(step.get("run") or ""):
            if "SYNC_TOKEN" not in s:
                continue
            uses += 1
            assert "${MIRROR}" in s, f"step {step.get('name')!r} uses the mirror token without ${{MIRROR}}: {s}"
            assert "GITHUB_REPOSITORY" not in s, s
    assert uses >= 4, uses  # ordering read, REMOTE (push, verify), the deploy-manual dispatch
