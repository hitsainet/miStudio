"""The old GitHub Pages path of the manual is linked from nowhere in the suite (042 FR-042.4; FTDD T-03).

ENH-009 P1 renames `hitsainet/miStudio`, which ends its project Pages site at once; the manual moves to
https://docs-miworkbench.hitsai.net/ first (FR-042.1). So before P1 no tracked file of any of the six suite
repositories may still link the old path. The scan is `git grep -i` for `OLD_PATH` (the Pages host + `/miStudio`) over TRACKED files,
excluding only the dated-record paths census excludes (`EXCLUDE` in miSuite's
`specs/ENH-009-rebrand/census.py`: `0xcc/`, `specs/`, `.claude/`, lock files, the export).

The six (FPRD FR-042.4): this repository, miServer (miLLM), miDataworks, miForge, hitsai.net, and
the marketing folder of the suite repository (`miStudioMarket` since P5; `miSuiteMarket` before).
- THIS repository is always asserted.
- A sibling is asserted for real under the cross-repo switch (`MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1` or the
  old spelling, FTASKS 2.6 runs it that way). Without it, a missing sibling or a sibling that still links the
  old path SKIPS LOUDLY, naming every hit: the siblings move their links in their own features (miServer 040,
  miDataworks 020, miForge 015, S001 for hitsai.net and miSuiteMarket), and a local run must not go red on
  their schedule. CI has no siblings, so there these always skip by name.

Controls: K-1.12a plants the old link in README.md (this repository's case red); K-1.12b widens this file's
exclusions (`test_exclusions_equal_the_census_exclude` red).
"""
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import REPO_ROOT, midataworks_repo, miserver_repo, sibling_repo, suite_root

#: `github.io/` + `mistudio`, matched case-insensitively (git grep -i). Joined at runtime, and every planted link
#: below is built from it, so this file never matches its own scan of this repository.
OLD_PATH = "".join(("github.io/", "mistudio"))
HOST = "".join(("hitsainet.", OLD_PATH[:len("github.io/")]))  # hitsainet.github.io/
OLD_URL = "https://" + HOST + "miStudio"

#: A COPY of census's EXCLUDE, pinned equal to it by `test_exclusions_equal_the_census_exclude` when the suite
#: checkout is present (it is not in CI or on the mirror, where this copy is what runs).
EXCLUSIONS = (
    ":!0xcc", ":!specs", ":!.claude",
    ":(exclude,glob)**/0xcc/**", ":(exclude,glob)**/docs/legacy-identifiers*.md",
    ":(exclude,glob)**/package-lock.json", ":(exclude,glob)**/*.lock", ":(exclude,glob)**/poetry.lock",
    ":(exclude,glob)**/uv.lock", ":(exclude,glob)**/CHANGELOG*",
)

#: Test folders are not links (2026-11-09, ENH-009 S001 0.97 made the same call for its own grep). A sibling's
#: tests NAME the old path on purpose: the frozen-identifier tests exercise DR-29's literal
#: (the old Pages host plus `/miStudio`), and census finds that literal in them and pins it in the repo's baseline, so
#: they cannot stop naming it; the manual-link guards (miServer, miDataworks) name it to forbid it. Kept apart
#: from EXCLUSIONS so that copy stays equal to census's EXCLUDE.
TEST_EXCLUSIONS = (":(exclude,glob)tests/**", ":(exclude,glob)**/tests/**")

CENSUS = Path("specs") / "ENH-009-rebrand" / "census.py"


def old_path_hits(repo: Path, pathspec: str = ".") -> list[str]:
    """`path:line: text` of every tracked, non-excluded line naming the old Pages path."""
    out = subprocess.run(
        ["git", "-C", str(repo), "grep", "-n", "-I", "-i", "-F", OLD_PATH, "--", pathspec, *EXCLUSIONS,
         *TEST_EXCLUSIONS],
        capture_output=True, text=True)
    if out.returncode not in (0, 1):
        raise RuntimeError(f"git grep failed in {repo}: {out.stderr.strip()}")
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _hitsai_repo() -> Path:
    override = os.environ.get("HITSAI_REPO")
    return Path(override) if override else suite_root() / "hitsai.net"


def _market_folder() -> str:
    """The marketing folder of the suite repository: `miStudioMarket` since ENH-009 P5 (2026-11-12) moved it,
    `miSuiteMarket` before. The new name first; the old one only where the move has not happened."""
    return "miStudioMarket" if (suite_root() / "miStudioMarket").is_dir() or not (
        suite_root() / "miSuiteMarket").is_dir() else "miSuiteMarket"


#: (name, repository root, pathspec). The marketing folder is a folder of the suite repository.
def _repos() -> list[tuple[str, Path, str]]:
    return [
        ("this-repo", REPO_ROOT, "."),
        ("miServer", miserver_repo(), "."),
        ("miDataworks", midataworks_repo(), "."),
        ("miForge", sibling_repo("miForge", "MIFORGE_REPO"), "."),
        ("hitsai.net", _hitsai_repo(), "."),
        ("miStudioMarket", suite_root(), _market_folder()),
    ]


def _is_checkout(repo: Path, pathspec: str) -> bool:
    if not (repo / ".git").exists():
        return False
    return pathspec == "." or (repo / pathspec).is_dir()


@pytest.mark.parametrize("name,repo,pathspec", _repos(), ids=[r[0] for r in _repos()])
def test_no_tracked_link_to_the_old_pages_path_in_any_suite_repo(name, repo, pathspec):
    if name != "this-repo" and not _is_checkout(repo, pathspec):
        message = f"{name}: no checkout at {repo / pathspec} (set its env var; FR-042.4 needs all six repos)"
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip("CROSS-REPO CHECK SKIPPED LOUDLY: " + message)
    hits = old_path_hits(repo, pathspec)
    if hits and name != "this-repo" and not require_cross_repo():
        pytest.skip(f"CROSS-REPO CHECK SKIPPED LOUDLY: {name} still links the old Pages path in "
                    f"{len(hits)} line(s) (its own feature moves them; P1 waits for zero):\n" + "\n".join(hits))
    assert not hits, (f"{name} links the old Pages path ({OLD_PATH}); move each to "
                      "https://docs-miworkbench.hitsai.net/<same path>:\n" + "\n".join(hits))


def _census_exclude(census: Path) -> list[str]:
    """census.py's `EXCLUDE`, read by AST (nothing from the suite checkout is executed)."""
    tree = ast.parse(census.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        if any(isinstance(t, ast.Name) and t.id == "EXCLUDE" for t in targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError(f"{census} assigns no literal EXCLUDE")


def test_exclusions_equal_the_census_exclude():
    census = suite_root() / CENSUS
    if not census.is_file():
        message = f"no census at {census} (set MISUITE_REPO to the miSuite checkout)"
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip("CROSS-REPO CHECK SKIPPED LOUDLY: " + message)
    assert list(EXCLUSIONS) == _census_exclude(census)


def test_a_planted_link_is_found(tmp_path):
    """The scan finds a planted link in a tracked file, ignores an untracked one, and honours exactly the
    excluded paths (a dated record under 0xcc/ and the generated export)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    files = {
        "README.md": f"see {OLD_URL.upper()}/getting-started/introduction\n",
        "manual/docs/a.md": f"clean line\n{OLD_URL}/x\n",
        "0xcc/reviews/old.md": f"{OLD_URL}/ (a dated record)\n",
        "docs/legacy-identifiers.md": f"{HOST}miStudio\n",
        "notes/ok.md": "https://docs-miworkbench.hitsai.net/x\n",
        "tests/unit/test_frozen.py": f'LITERAL = "{HOST}miStudio"\n',
        "backend/tests/integration/test_x.py": f'LITERAL = "{HOST}miStudio"\n',
        "docs/tests-and-more.md": f"{OLD_URL}/still-a-link\n",
    }
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    (repo / "untracked.md").write_text(f"{OLD_URL}/\n")
    hits = old_path_hits(repo)
    assert sorted(h.split(":", 2)[0] + ":" + h.split(":", 2)[1] for h in hits) == [
        "README.md:1", "docs/tests-and-more.md:1", "manual/docs/a.md:2"], hits
