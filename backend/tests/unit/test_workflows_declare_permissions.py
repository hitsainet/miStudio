"""Every GitHub Actions job runs with an explicitly declared GITHUB_TOKEN scope.

CodeQL actions/missing-workflow-permissions #30 (security alerts 2026-10-17): the `detect`
job of `docker-images.yml` declared none, so it ran with the repository's default token
scope — which on the public mirror may be read/write to contents and packages. A job
inherits the workflow-level block, so either level satisfies the rule; neither does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"


def _workflows():
    if not WORKFLOWS.is_dir():
        pytest.skip(f"no workflows directory at {WORKFLOWS}")
    found = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert found, f"{WORKFLOWS} holds no workflow files"
    return found


def test_every_job_has_a_permissions_block_or_inherits_one():
    unscoped = []
    for path in _workflows():
        doc = yaml.safe_load(path.read_text())
        if "permissions" in doc:
            continue
        unscoped += [
            f"{path.name}:{name}"
            for name, job in (doc.get("jobs") or {}).items()
            if "permissions" not in job
        ]
    assert not unscoped, f"jobs running with the default token scope: {unscoped}"


def test_the_image_build_detect_job_is_read_only():
    doc = yaml.safe_load((WORKFLOWS / "docker-images.yml").read_text())
    assert doc["permissions"] == {"contents": "read"}
    assert "permissions" not in doc["jobs"]["detect"], "detect should inherit the read-only scope"
