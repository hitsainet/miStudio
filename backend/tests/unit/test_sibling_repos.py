"""The cross-repo tests find their siblings from this checkout, not from a home directory.

See `tests/support/sibling_repos.py` for why: the suite moved directories on 2026-10-16 and a
hardcoded default would have turned every byte-identity check into a silent skip.
"""
from __future__ import annotations

import re

import pytest

from tests.support import sibling_repos
from tests.support.sibling_repos import REPO_ROOT, miserver_repo, sibling_repo

# A suite checkout directly under a home directory's app/. Paths on the k8s node (e.g.
# `/home/sean/app/k8s-millm...`) are remote and not checkouts, so only the repo names match.
# Assembled so this file does not match its own scan.
_HOME_APP = re.compile("/home/" + r"[^/\s]+" + "/app/" + r"mi(Studio|LLM|Dataworks|Forge)\b")


def test_repo_root_is_this_checkout():
    assert (REPO_ROOT / "backend" / "tests" / "support" / "sibling_repos.py").is_file()
    assert (REPO_ROOT / "docs" / "schemas").is_dir()


def _expected_default() -> object:
    # miServer's folder (the only candidate since ENH-009 P7; the old `miLLM` folder is not consulted).
    return REPO_ROOT.parent / "miServer"


def test_default_is_the_sibling_directory(monkeypatch):
    monkeypatch.delenv("MISERVER_REPO", raising=False)
    assert miserver_repo() == _expected_default()
    assert sibling_repo("miDataworks", "MIDATAWORKS_REPO_UNSET_FOR_TEST") == (
        REPO_ROOT.parent / "miDataworks"
    )


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MISERVER_REPO", str(tmp_path))
    assert miserver_repo() == tmp_path


def test_empty_override_falls_back_to_the_sibling(monkeypatch):
    monkeypatch.setenv("MISERVER_REPO", "")
    assert miserver_repo() == _expected_default()


# ── 042 FR-042.10 (FTASKS 5.1, 18.7): miServer found under its new name only since P7 ────────────────────
@pytest.fixture
def suite_parent(tmp_path, monkeypatch):
    """A temporary parent directory holding this checkout; no sibling env var set."""
    me = tmp_path / "miWorkbench"
    me.mkdir()
    monkeypatch.setattr(sibling_repos, "REPO_ROOT", me)
    monkeypatch.delenv("MISERVER_REPO", raising=False)
    return tmp_path


def test_new_folder_found(suite_parent):
    (suite_parent / "miLLM").mkdir()
    (suite_parent / "miServer").mkdir()
    assert sibling_repos.miserver_repo() == suite_parent / "miServer"


def test_old_folder_not_consulted(suite_parent):
    """INVERTED at P7 (was `test_old_folder_found`): `test_legacy_retired.py::test_millm_repo_alias_gone`."""
    (suite_parent / "miLLM").mkdir()
    assert sibling_repos.miserver_repo() == suite_parent / "miServer"


def test_missing_sibling_names_the_new_folder(suite_parent):
    assert sibling_repos.miserver_repo() == suite_parent / "miServer"


def test_env_before_folders(suite_parent, monkeypatch, tmp_path):
    (suite_parent / "miServer").mkdir()
    monkeypatch.setenv("MISERVER_REPO", str(tmp_path / "pinned"))
    assert sibling_repos.miserver_repo() == tmp_path / "pinned"


def test_no_test_or_script_hardcodes_a_home_app_path():
    """A checkout path under a home directory breaks the next time the suite moves.

    systemd units are exempt: systemd requires absolute paths. Everything else derives its
    location from its own file.
    """
    offenders = []
    candidates = list((REPO_ROOT / "backend" / "tests").rglob("*.py"))
    candidates += list(REPO_ROOT.glob("*.sh")) + list((REPO_ROOT / "scripts").rglob("*.sh"))
    for path in candidates:
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if _HOME_APP.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)

