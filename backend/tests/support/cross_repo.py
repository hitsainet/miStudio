"""The test-only switches, read under their new and old names (042 FR-042.11; FTID §9.2; TI-3).

`require_cross_repo()` is the one reader of the cross-repo switch: a cross-repo test skips loudly
when its sibling is absent, and FAILS instead when the switch is on. During the rebrand either
spelling turns it on: `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1` or the suite-wide
`MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1` (040 TPD-18 keeps the latter permanently; 042 TI-19).
`require_runner()` does the same for the rule-runner switch the CI job sets.

`test_env(new, old)` reads any other test-only switch new-first. These switches are NOT operator
configuration, so they are not `env` records of the export and not rows of `src/core/legacy_names.py`
(TI-3): S001's sampler never sees a test run.

`test_cross_repo_flag.py` keeps every test from reading an old-prefix switch directly (AST), and keeps
`rule_runner/tests/_switches.py` (the runner-host tests cannot import this package) equal to this file.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

REQUIRE_NEW = "MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS"
REQUIRE_OLD = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"
RUNNER_NEW = "MIWORKBENCH_REQUIRE_RUNNER"
RUNNER_OLD = "MISTUDIO_REQUIRE_RUNNER"
#: Both spellings of each switch, new first (for a reader of a workflow's `env:` mapping).
REQUIRE_SWITCH = (REQUIRE_NEW, REQUIRE_OLD)
RUNNER_SWITCH = (RUNNER_NEW, RUNNER_OLD)


def test_env(new: str, old: str, default: str | None = None, *,
             environ: Mapping[str, str] | None = None) -> str | None:
    """`$new` when set and non-empty, else `$old` when set and non-empty, else `default`.
    `environ` replaces `os.environ` for a reader handed an environment mapping."""
    env = os.environ if environ is None else environ
    value = env.get(new)
    if value:
        return value
    value = env.get(old)
    if value:
        return value
    return default


# Not a test, despite the name pytest collects by (`test_*`).
test_env.__test__ = False  # type: ignore[attr-defined]


def require_cross_repo() -> bool:
    """True when either spelling of the cross-repo switch is `1`."""
    return test_env(REQUIRE_NEW, REQUIRE_OLD) == "1"


def require_runner() -> bool:
    """True when either spelling of the rule-runner switch is `1` (CI's `rule-runner-tests` job)."""
    return test_env(RUNNER_NEW, RUNNER_OLD) == "1"
