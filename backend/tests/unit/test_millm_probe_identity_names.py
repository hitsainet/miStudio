"""The miLLM identity names `millm_recalibrate_probe` depends on (Feature 039 FTASKS 6.9; miLLM 038 FR-38.7).

Read from miLLM's OWN `create_app().openapi()` through the sibling checkout (the
`test_millm_session_api_names.py` pattern): `GET /api/probes/{id}` documents `detector_digest`,
`detector_digest_source` and `identity_algorithm`, and `ProbeRecalibrationRequest` takes
`decision_digest` and `decision_revision`. The tool reads exactly these names; miLLM pins its side.
Skips LOUDLY naming `MILLM_REPO` when miLLM is not checked out (the public mirror) or does not serve
them yet; fails under `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.support.sibling_repos import millm_repo

SUMMARY_FIELDS = ("detector_digest", "detector_digest_source", "identity_algorithm")
REQUEST_FIELDS = ("decision_digest", "decision_revision")
REQUIRE = os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1"


def _absent(message: str) -> None:
    if REQUIRE:
        pytest.fail(message)
    pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + message)


def _openapi() -> dict:
    repo = millm_repo()
    if not (repo / "millm" / "main.py").is_file():
        _absent(f"miLLM is not checked out at {repo} (set MILLM_REPO)")
    code = ("import json; from millm.main import create_app; "
            "print('@@OPENAPI@@' + json.dumps(create_app().openapi()))")
    python = repo / "venv" / "bin" / "python"
    interpreter = str(python) if python.is_file() else sys.executable
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_URL", "PYTHONPATH"))}
    out = subprocess.run([interpreter, "-B", "-c", code], cwd=repo, env=env, capture_output=True,
                         text=True, timeout=240)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("@@OPENAPI@@")]
    if out.returncode != 0 or not lines:
        _absent(f"miLLM's app could not be imported at {repo} ({out.stderr.strip()[-300:]})")
    return json.loads(lines[0][len("@@OPENAPI@@"):])


def _resolve(document: dict, schema: dict) -> dict:
    while "$ref" in schema:
        schema = document["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    return schema


def _properties_reachable(document: dict, schema: dict, depth: int = 0) -> set:
    """Every property name reachable from a response schema (through refs, allOf/anyOf, data)."""
    if depth > 6 or not isinstance(schema, dict):
        return set()
    schema = _resolve(document, schema)
    names = set((schema.get("properties") or {}).keys())
    for child in list((schema.get("properties") or {}).values()) + [
            *schema.get("allOf", []), *schema.get("anyOf", []), *schema.get("oneOf", [])]:
        names |= _properties_reachable(document, child, depth + 1)
    return names


def test_get_probe_documents_the_identity_names():
    document = _openapi()
    route = (document.get("paths") or {}).get("/api/probes/{probe_id}", {}).get("get")
    if route is None:
        _absent("miLLM does not serve GET /api/probes/{probe_id} at "
                f"{millm_repo()} (set MILLM_REPO)")
    schema = route["responses"]["200"]["content"]["application/json"]["schema"]
    names = _properties_reachable(document, schema)
    missing = [name for name in SUMMARY_FIELDS if name not in names]
    if missing and not REQUIRE:
        pytest.skip(f"miLLM at {millm_repo()} does not document {missing} yet (miLLM 038 S1)")
    assert not missing, missing


def test_the_recalibrate_request_takes_the_identity_fields():
    document = _openapi()
    schema = (document.get("components") or {}).get("schemas", {}).get("ProbeRecalibrationRequest")
    if schema is None:
        _absent(f"miLLM at {millm_repo()} has no ProbeRecalibrationRequest (set MILLM_REPO)")
    missing = [name for name in REQUEST_FIELDS if name not in (schema.get("properties") or {})]
    if missing and not REQUIRE:
        pytest.skip(f"miLLM at {millm_repo()} does not take {missing} yet (miLLM 038 S1)")
    assert not missing, missing

