"""Where a sibling suite repo is checked out, for the cross-repo tests.

The suite repos (miStudio, miLLM, miDataworks, miForge) are checked out side by
side under one parent directory. A cross-repo test compares a file here with its
vendored twin over there, so it needs the sibling's path — and it must find it
from THIS checkout, never from an absolute home-directory path. On 2026-10-16
every repo moved from `~/app/<repo>` to `~/app/miSuite/<repo>` with no symlink
left behind; a hardcoded default would have pointed at a directory that no
longer exists, and every byte-identity check would have SKIPPED rather than
compared — green while checking nothing.

The env override stays: `MISERVER_REPO` points a test at a miServer worktree, which is
how a branch is checked against the miServer it will co-release with. (The old `MILLM_REPO`
and the old `miLLM` folder name were accepted until ENH-009 P7.)
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

#: This checkout's root: `backend/tests/support/sibling_repos.py` → parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]


def sibling_repo(name: str, env_var: str) -> Path:
    """`$env_var` when it is set and non-empty, else `REPO_ROOT.parent / name`."""
    override = os.environ.get(env_var)
    return Path(override) if override else REPO_ROOT.parent / name


def sibling_repo_of(candidates: Sequence[tuple[str, str]]) -> Path:
    """The first sibling found among `(folder, env_var)` candidates (042 FTID §3.12, FR-042.10).

    Every candidate's env var is tried first (the first non-empty wins), then every folder
    `REPO_ROOT.parent / folder` that exists; with neither, the FIRST candidate's folder, so a
    missing sibling still names the expected path in the skip message. During the rebrand a
    sibling may be checked out under its new or its old name (ENH-009 P1 renames the repos).
    """
    if not candidates:
        raise ValueError("sibling_repo_of needs at least one candidate")
    for _folder, env_var in candidates:
        override = os.environ.get(env_var)
        if override:
            return Path(override)
    for folder, _env_var in candidates:
        if (REPO_ROOT.parent / folder).is_dir():
            return REPO_ROOT.parent / folder
    return REPO_ROOT.parent / candidates[0][0]


#: miServer (formerly miLLM). The old candidate (`$MILLM_REPO`, the `miLLM` folder) went at ENH-009 P7 (FTASKS 18.7).
MISERVER_CANDIDATES: tuple[tuple[str, str], ...] = (("miServer", "MISERVER_REPO"),)


def miserver_repo() -> Path:
    """The miServer checkout: `$MISERVER_REPO`, else the `miServer` sibling."""
    return sibling_repo_of(MISERVER_CANDIDATES)


#: miServer's package names, the new one first (a tree holding both resolves to the new one).
MISERVER_PACKAGES: tuple[str, str] = ("miserver", "millm")


def miserver_package(repo: Path) -> str | None:
    """Which package miServer's tree holds: `"miserver"` (after its P8 rename), else `"millm"`, else None.

    A tree holding both resolves to `"miserver"`; `both_packages()` lets a caller say so (FR-042.36).
    """
    for pkg in MISERVER_PACKAGES:
        if (Path(repo) / pkg / "main.py").is_file():
            return pkg
    return None


def both_packages(repo: Path) -> bool:
    return all((Path(repo) / pkg / "main.py").is_file() for pkg in MISERVER_PACKAGES)


def miserver_path(repo: Path, *parts: str) -> Path:
    """`<repo>/<package>/<parts…>` for whichever package the tree holds; raises when it holds neither."""
    pkg = miserver_package(repo)
    if pkg is None:
        raise FileNotFoundError(f"{repo} holds neither miserver/main.py nor millm/main.py")
    return Path(repo).joinpath(pkg, *parts)


def miserver_path_or_none(repo: Path, *parts: str) -> Path | None:
    """`miserver_path(repo, *parts)`, or None when the tree holds neither package (a missing sibling,
    which the caller skips loudly or fails on under the cross-repo switch)."""
    return None if miserver_package(repo) is None else miserver_path(repo, *parts)


def miserver_import(module: str, repo: Path | None = None) -> str:
    """The dotted module name in miServer's package: `"core.errors"` → `"millm.core.errors"` (or
    `"miserver.…"`). A name already starting with `millm.`/`miserver.` is re-rooted; `"main"` is the
    package's own main module."""
    pkg = miserver_package(miserver_repo() if repo is None else repo)
    if pkg is None:
        raise FileNotFoundError("miServer's checkout holds neither package")
    for name in MISERVER_PACKAGES:
        if module.startswith(name + "."):
            module = module[len(name) + 1:]
            break
    if module in MISERVER_PACKAGES:
        return pkg
    return f"{pkg}.{module}"


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
