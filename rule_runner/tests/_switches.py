"""The two test-only switches the runner-host tests read, new name first (042 FR-042.11; FTID §9.2).

The runner-host tests run from `rule_runner/` with their own interpreter and cannot import the backend's
`tests/support/cross_repo.py`; this is its copy for them. `backend/tests/unit/test_cross_repo_flag.py`
pins the names and the reading rule of both files equal.
"""
from __future__ import annotations

import os

REQUIRE_NEW = "MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS"
REQUIRE_OLD = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"
RUNNER_NEW = "MIWORKBENCH_REQUIRE_RUNNER"
RUNNER_OLD = "MISTUDIO_REQUIRE_RUNNER"


def _read(new: str, old: str) -> str | None:
    return os.environ.get(new) or os.environ.get(old) or None


def require_cross_repo() -> bool:
    """True when either spelling of the cross-repo switch is `1` (the new one read first)."""
    return _read(REQUIRE_NEW, REQUIRE_OLD) == "1"


def require_runner() -> bool:
    """True when either spelling of the rule-runner switch is `1` (CI's `rule-runner-tests` job)."""
    return _read(RUNNER_NEW, RUNNER_OLD) == "1"
