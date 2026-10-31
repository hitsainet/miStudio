"""The miLLM session read API names miStudio's spot-check client depends on (038 §7; miLLM 036 §7).

Read from miLLM's OWN `app.openapi()` (never `app.routes`) through the sibling checkout. miLLM owns these
names and pins the same ones in its `tests/unit/api/test_probe_session_api_names.py`. Skips LOUDLY until the
sibling serves them (miLLM 036 is built in parallel) and fails under MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1.
Control C-NAMEPATH: rename a path constant in `millm_read` → the in-repo pin below goes red.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from src.services import millm_read
from tests.support.sibling_repos import millm_repo

#: The names as miLLM 036 FPRD §7 states them (path-parameter names normalised).
EXPECTED = {
    "SAMPLE": ("post", "/api/probes/sessions/sample"),
    "TRANSCRIPT": ("get", "/api/probes/sessions/{}/transcript"),
    "EVENTS": ("get", "/api/probes/events"),
    "PROBES": ("get", "/api/probes"),
}


def test_the_client_constants_are_036s_names():
    for constant, (_method, path) in EXPECTED.items():
        assert millm_read.normalise_path(getattr(millm_read, constant)) == path, constant


def _millm_paths():
    repo = millm_repo()
    if not (repo / "millm" / "main.py").is_file():
        message = (f"miLLM is not checked out at {repo} (set MILLM_REPO). Expected on the public mirror; "
                   f"MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1 makes it fail.")
        if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    code = ("import json; from millm.main import create_app; app = create_app(); "
            "print(json.dumps({p: sorted(v) for p, v in app.openapi()['paths'].items()}))")
    # miLLM's own environment when the checkout has one (its dependencies are not miStudio's).
    python = repo / "venv" / "bin" / "python"
    interpreter = str(python) if python.is_file() else sys.executable
    out = subprocess.run([interpreter, "-c", code], cwd=repo, capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        message = f"miLLM's app could not be imported here ({out.stderr.strip()[-300:]})"
        if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    import json

    return {millm_read.normalise_path(p): set(m) for p, m in json.loads(out.stdout.splitlines()[-1]).items()}


def test_millm_serves_them():
    served = _millm_paths()
    missing = [f"{m.upper()} {p}" for m, p in EXPECTED.values() if m not in served.get(p, set())]
    if missing and os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") != "1":
        pytest.skip(f"miLLM at {millm_repo()} does not serve {missing} yet (miLLM Feature 036 is not merged there)")
    assert not missing, missing
