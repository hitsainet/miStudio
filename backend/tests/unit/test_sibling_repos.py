"""The cross-repo tests find their siblings from this checkout, not from a home directory.

See `tests/support/sibling_repos.py` for why: the suite moved directories on 2026-10-16 and a
hardcoded default would have turned every byte-identity check into a silent skip.
"""
from __future__ import annotations

import re

from tests.support.sibling_repos import REPO_ROOT, millm_repo, sibling_repo

# A suite checkout directly under a home directory's app/. Paths on the k8s node (e.g.
# `/home/sean/app/k8s-millm...`) are remote and not checkouts, so only the repo names match.
# Assembled so this file does not match its own scan.
_HOME_APP = re.compile("/home/" + r"[^/\s]+" + "/app/" + r"mi(Studio|LLM|Dataworks|Forge)\b")


def test_repo_root_is_this_checkout():
    assert (REPO_ROOT / "backend" / "tests" / "support" / "sibling_repos.py").is_file()
    assert (REPO_ROOT / "docs" / "schemas").is_dir()


def test_default_is_the_sibling_directory(monkeypatch):
    monkeypatch.delenv("MILLM_REPO", raising=False)
    assert millm_repo() == REPO_ROOT.parent / "miLLM"
    assert sibling_repo("miDataworks", "MIDATAWORKS_REPO_UNSET_FOR_TEST") == (
        REPO_ROOT.parent / "miDataworks"
    )


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MILLM_REPO", str(tmp_path))
    assert millm_repo() == tmp_path


def test_empty_override_falls_back_to_the_sibling(monkeypatch):
    monkeypatch.setenv("MILLM_REPO", "")
    assert millm_repo() == REPO_ROOT.parent / "miLLM"


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

