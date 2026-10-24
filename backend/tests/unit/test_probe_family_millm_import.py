"""Each member a family export ships is a definition miLLM accepts (Feature 041 FPRD §7; FTASKS 14.5).

Cross-repo: the family is exported through the REAL route, and every exported file is validated in miLLM's
own venv by miLLM's own contract model (`tests/support/millm_definition_driver.py`, a subprocess — nothing
of miLLM is imported here). Skips loudly without a miLLM checkout; fails under
`MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1`. Per-probe serving PARITY needs the served model and is the live
acceptance (FTASKS 15.3/15.4), not this test.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from src.core.config import settings
from tests.support import family_fixtures as ff
from tests.support.random_behaviour import rng_for
from tests.support.sibling_repos import millm_repo

API = "/api/v1/probe-monitors"
FLAG = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"
DRIVER = Path(__file__).resolve().parents[1] / "support" / "millm_definition_driver.py"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "probe_definition_pre_identity.json"


def _millm_or_skip() -> Path:
    root = millm_repo()
    python = root / "venv" / "bin" / "python"
    if not (root / "millm" / "api" / "schemas" / "probe.py").is_file() or not python.is_file():
        message = (f"miLLM (with its venv) is not at {root}; set MILLM_REPO. Skipping is expected on the public "
                   f"mirror; set {FLAG}=1 to make this a failure.")
        if os.environ.get(FLAG) == "1":
            pytest.fail(message)
        pytest.skip(message)
    return root


async def test_every_exported_member_imports_into_millm(client, async_session, request, tmp_path, monkeypatch):
    root = _millm_or_skip()
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    key = await ff.catalogue(client, rng_for(request.node.nodeid))
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    base = json.loads(FIXTURE.read_text())
    for i in range(3):
        probe = await ff.member(async_session, model, ds, key=key, role="primary" if i == 0 else None, layer=i)
        document = {**base, "name": f"{base['name']}-member-{i}"}
        path = tmp_path / "defs" / f"{probe.id}.probe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
        probe.definition_path = str(path)
    await async_session.commit()
    archive = await client.get(f"{API}/behaviours/{key}/family/export")
    assert archive.status_code == 200, archive.text
    out_dir = tmp_path / "unzipped"
    with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
        zf.extractall(out_dir)
    files = sorted(str(p) for p in out_dir.iterdir())
    assert len(files) == 3
    proc = subprocess.run([str(root / "venv" / "bin" / "python"), "-c", DRIVER.read_text(), *files], cwd=root,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    verdicts = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdicts == {f: "ok" for f in files}, verdicts
