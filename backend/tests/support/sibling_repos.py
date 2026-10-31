"""Where a sibling suite repo is checked out, for the cross-repo tests.

The suite repos (miStudio, miLLM, miDataworks, miForge) are checked out side by
side under one parent directory. A cross-repo test compares a file here with its
vendored twin over there, so it needs the sibling's path — and it must find it
from THIS checkout, never from an absolute home-directory path. On 2026-10-16
every repo moved from `~/app/<repo>` to `~/app/miSuite/<repo>` with no symlink
left behind; a hardcoded default would have pointed at a directory that no
longer exists, and every byte-identity check would have SKIPPED rather than
compared — green while checking nothing.

The env override stays: `MILLM_REPO` points a test at a miLLM worktree, which is
how a branch is checked against the miLLM it will co-release with.
"""
from __future__ import annotations

import os
from pathlib import Path

#: This checkout's root: `backend/tests/support/sibling_repos.py` → parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]


def sibling_repo(name: str, env_var: str) -> Path:
    """`$env_var` when it is set and non-empty, else `REPO_ROOT.parent / name`."""
    override = os.environ.get(env_var)
    return Path(override) if override else REPO_ROOT.parent / name


def millm_repo() -> Path:
    """The miLLM checkout: `$MILLM_REPO`, else the `miLLM` sibling of this repo."""
    return sibling_repo("miLLM", "MILLM_REPO")


def suite_root() -> Path:
    """The miSuite planning repo the app repos sit in: `$MISUITE_REPO`, else `REPO_ROOT.parent`.

    Since 2026-10-16 the suite-wide agent instructions (the GPU-node SSH route, the
    certificate procedure, the password-auth directive) live in `miSuite/CLAUDE.md`,
    not in each app's. A public mirror has no suite directory around it.
    """
    override = os.environ.get("MISUITE_REPO")
    return Path(override) if override else REPO_ROOT.parent


def midataworks_repo() -> Path:
    """The miDataworks checkout: `$MIDATAWORKS_REPO`, else the `miDataworks` sibling."""
    return sibling_repo("miDataworks", "MIDATAWORKS_REPO")
