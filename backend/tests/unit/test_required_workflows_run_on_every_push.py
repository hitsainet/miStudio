"""Every workflow the mirror gate requires runs on EVERY push and pull request: no path filter
(042 P0; the miLLM 040 P0 review's HIGH finding, applied to this repository).

`sync-to-clean.yml`'s gate (`.github/scripts/private_ci_gate.py`) passes a commit for which a required
workflow created NO run when that workflow has a push path filter and an ancestor passed it with the
filtered paths unchanged. That is sound for the code a filter names and blind to everything else: with
Backend Tests filtered to `backend/**`, a push changing only `.github/`, `k8s/`, `docs/` or `manual/` ran no
test at all — not the workflow gates, not the mirror-export test, not the frozen-identifier test that
reads `docs/legacy-identifiers.md` — and still reached the mirror. One such push could publish the full
export. So no required workflow carries `paths`/`paths-ignore` on push or pull request; the gate's own
parser is asked, so this test and the gate cannot disagree about what a filter is.
"""
from __future__ import annotations

import importlib.util

import pytest
import yaml

from tests.support import legacy_export as le

WORKFLOWS = le.REPO_ROOT / ".github" / "workflows"
GATE_SCRIPT = le.REPO_ROOT / ".github" / "scripts" / "private_ci_gate.py"


def _load(name: str) -> dict:
    doc = yaml.safe_load((WORKFLOWS / name).read_text())
    doc["on"] = doc.pop(True, doc.get("on"))
    return doc


def _required() -> list[str]:
    gate = _load("sync-to-clean.yml")["jobs"]["gate"]
    names = sorted({w.strip() for w in gate["env"]["REQUIRED_WORKFLOWS"].split() if w.strip()})
    assert names, "REQUIRED_WORKFLOWS is empty: this guard would check nothing"
    return names


def _gate():
    spec = importlib.util.spec_from_file_location("private_ci_gate_for_filter_test", GATE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_required_set_is_the_ci_this_guard_expects():
    assert {"backend-tests.yml", "frontend-ci.yml"} <= set(_required())


@pytest.mark.parametrize("workflow", ["backend-tests.yml", "frontend-ci.yml"])
def test_no_required_workflow_has_a_path_filter(workflow):
    assert workflow in _required()
    doc = _load(workflow)
    on = doc["on"]
    for event in ("push", "pull_request"):
        assert event in on, f"{workflow} does not run on {event}"
        spec = on[event] or {}
        assert "main" in (spec.get("branches") or ["main"])
        for key in ("paths", "paths-ignore"):
            assert key not in spec, (
                f"{workflow} {event} has `{key}`: a push outside it runs no guard and the gate's ancestor "
                f"fallback still lets it reach the mirror")
    assert _gate().push_pathspecs(doc) is None, "the gate itself reads a push filter on " + workflow


def test_every_required_workflow_is_checked_here():
    """A workflow added to REQUIRED_WORKFLOWS later is covered by the same rule."""
    for workflow in _required():
        doc = _load(workflow)
        for event in ("push", "pull_request"):
            spec = doc["on"].get(event) or {}
            assert not ({"paths", "paths-ignore"} & set(spec)), f"{workflow} {event}"
        assert _gate().push_pathspecs(doc) is None, workflow


def test_the_filter_check_sees_a_planted_filter():
    """Self-test: the gate's parser reports a filter where one exists (so `None` above means something)."""
    doc = yaml.safe_load("on:\n  push:\n    branches: [main]\n    paths: ['backend/**']\njobs: {}\n")
    doc["on"] = doc.pop(True)
    assert _gate().push_pathspecs(doc) == [":(glob)backend/**"]
