"""This repository's old names are gone from its tracked files, outside history and the register (042
FR-042.39; BR-075; CF-13, CF-20; FTASKS 15.1, 15.8; control K-15.15).

S001's P5 gate greps every suite repository for the old PRIVATE name of this app, because at P5 that name
passes to the suite's own repository: a line still spelling it here would then point at the wrong
repository. This test runs the same grep (`rebrand_checks.py p5-pattern`: the old name, case-insensitive,
not followed by `-` or a word character) over this repository's tracked files with the census's
exclusions: the record trees `0xcc/`, `specs/`, `.claude/` and the legacy-identifier exports (CF-20,
both glob forms).

No repository name is typed here (CF-13): the old names come from the export's `repo` records for this
repository. On the public mirror the private record is withheld, so that half skips BY NAME there.

A second check covers the old PUBLIC name outside test code: it may appear only inside a frozen
identifier the export allows (DR-04, the published schema `$id`s), never as a link someone follows.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.support import brand_surfaces as bs
from tests.support import legacy_export as le
from tests.support import workflow_gates as wg

REPO = le.REPO_ROOT
#: `rebrand_checks.gate_pathspec()` (S001 FTDD §4.3) plus the root-level export form (CF-20).
GATE_PATHSPEC = (":!0xcc", ":!specs", ":!.claude", ":(exclude,glob)**/0xcc/**", *bs.EXPORT_EXCLUDES)
#: Test code may name an old repository to test the rename itself; it is not a link anyone follows.
TEST_CODE = (":(exclude,glob)backend/tests/**", ":(exclude,glob)rule_runner/tests/**",
             ":(exclude,glob)frontend/src/test/**", ":(exclude,glob)frontend/src/**/*.test.ts",
             ":(exclude,glob)frontend/src/**/*.test.tsx")


def gate_regex(old: str) -> str:
    """`rebrand_checks.P5_RE` for one old name: case-insensitive, never a longer repository name."""
    return "(?i)" + re.escape(old) + r"(?![-\w])"


def grep(root: Path, regex: str, pathspec: tuple[str, ...] = GATE_PATHSPEC) -> list[str]:
    out = subprocess.run(["git", "grep", "-n", "-I", "-P", regex, "--", ".", *pathspec], cwd=root,
                         capture_output=True, text=True)
    assert out.returncode in (0, 1), out.stderr
    return [line for line in out.stdout.splitlines() if line]


def _owners(export: le.Export) -> tuple[dict[str, tuple[str, str]], str]:
    pairs = wg.this_repo_pairs(export)
    public = wg.owners_in(wg.load_workflows(), "docker-images.yml", "build")
    assert len(public) == 1 and public <= set(pairs), (public, sorted(pairs))
    return pairs, next(iter(public))


def private_old_name(export: le.Export) -> str:
    pairs, public = _owners(export)
    private = {owner: pair for owner, pair in pairs.items() if owner != public}
    if not private:
        held = [n for n in export.withheld if n.startswith("repo:sha256:")]
        if export.is_public and held:
            pytest.skip("on the public mirror the private repository's repo records are withheld: "
                        + ", ".join(held))
        pytest.fail("the export has no repo record for this repository's private name")
    assert len(private) == 1, private
    old, new = next(iter(private.values()))
    assert old != new
    return old


def test_no_old_repo_name_outside_history_and_register():
    old = private_old_name(le.load())
    regex = gate_regex(old)
    # The grep can see: the new private name is found by the same command (never an empty tree, never a
    # broken pathspec that matches nothing).
    pairs, public = _owners(le.load())
    new = next(pair[1] for owner, pair in pairs.items() if owner != public)
    assert grep(REPO, gate_regex(new)), "the gate's grep finds nothing at all: it cannot be trusted"
    hits = grep(REPO, regex)
    assert hits == [], "old private repository name outside history and the register:\n" + "\n".join(hits)


def test_the_gate_grep_sees_a_plant_and_honours_the_exclusions(tmp_path):
    old = private_old_name(le.load())
    root = tmp_path / "tree"
    files = {
        "README.md": f"clone https://github.com/{old}.git\n",
        "sub/notes.md": f"see {old.upper()}/issues and {old}-backup (another repository)\n",
        "0xcc/record.md": f"dated: {old}\n",
        "sub/0xcc/record.md": f"dated: {old}\n",
        ".claude/x.md": f"{old}\n",
        "specs/x.md": f"{old}\n",
        "docs/legacy-identifiers.md": f"{old}\n",
        "docs/legacy-identifiers.public.md": f"{old}\n",
        "sub/docs/legacy-identifiers.md": f"{old}\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    hits = grep(root, gate_regex(old))
    assert sorted(h.split(":", 1)[0] for h in hits) == ["README.md", "sub/notes.md"], hits
    # `<old>-backup` alone is another repository's name, not a hit: only the upper-case spelling counted.
    notes = [h for h in hits if h.startswith("sub/notes.md")]
    assert len(re.findall(gate_regex(old), notes[0])) == 1, notes


def test_old_public_repo_name_only_in_frozen_schema_ids():
    """Outside test code the old PUBLIC name survives only where the export freezes it (DR-04: the `$id`
    of each published schema, which never changes). A clone URL, an issue link or a card's schema link
    names the new repository."""
    export = le.load()
    pairs, public = _owners(export)
    old, new = pairs[public]
    frozen = [re.compile(e["pattern"]) for e in export.entries() if e["status"] != "retired"]
    stray = []
    for hit in grep(REPO, gate_regex(old), (*GATE_PATHSPEC, *TEST_CODE)):
        path, line, text = hit.split(":", 2)
        spans = [m.span() for p in frozen for m in p.finditer(text)]
        for m in re.finditer(gate_regex(old), text):
            if not any(s <= m.start() and m.end() <= e for s, e in spans):
                stray.append(f"{path}:{line}: {text.strip()[:160]}")
    assert stray == [], "old public repository name outside a frozen identifier:\n" + "\n".join(stray)
    # Fail closed both ways: the frozen ids are still found (the grep sees), and the new name is in use.
    assert grep(REPO, gate_regex(old), (*GATE_PATHSPEC, *TEST_CODE)), "no frozen schema id found: nothing was read"
    assert grep(REPO, gate_regex(new), (*GATE_PATHSPEC, *TEST_CODE))
