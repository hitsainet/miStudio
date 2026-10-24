"""Family export: all or nothing, untouched bytes, deterministic archive (Feature 041 FR-23; T13; FTASKS 13.1 – 13.2)."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest

from src.core.config import settings
from tests.support import family_fixtures as ff
from tests.support.random_behaviour import rng_for

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    return tmp_path


async def family(client, async_session, data_dir, request, *, n=3, rungs=None):
    key = await ff.catalogue(client, rng_for(request.node.nodeid))
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    probes = []
    for i in range(n):
        probe = await ff.member(async_session, model, ds, key=key, role="primary" if i == 0 else None)
        path = data_dir / "probe_definitions" / f"{probe.id}.probe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rung = (rungs or [2] * n)[i]
        path.write_bytes(json.dumps({"name": f"probe-{probe.id}", "evidence": {"rung": rung}}, indent=1).encode()
                         + b"\n")
        probe.definition_path = str(path)
        probes.append(probe)
    await async_session.commit()
    return key, model, probes


async def test_bundle_entries_hash_equal_single_export(client, async_session, data_dir, request):
    key, model, probes = await family(client, async_session, data_dir, request)
    index = (await client.get(f"{API}/behaviours/{key}/family/export-index")).json()
    assert [m["probe_id"] for m in index["members"]] == sorted(p.id for p in probes)
    archive = await client.get(f"{API}/behaviours/{key}/family/export")
    assert archive.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
        assert zf.namelist() == [f"{m['probe_id']}.probe.json" for m in index["members"]]
        for member in index["members"]:
            payload = zf.read(member["filename"])
            on_disk = (data_dir / "probe_definitions" / f"{member['probe_id']}.probe.json").read_bytes()
            assert payload == on_disk and hashlib.sha256(payload).hexdigest() == member["sha256"]
            assert member["size"] == len(on_disk)
            assert member["export_route"] == f"/api/v1/probe-monitors/probes/{member['probe_id']}/export"
            assert zf.getinfo(member["filename"]).compress_type == zipfile.ZIP_STORED
    assert index["archive_route"].startswith(f"/api/v1/probe-monitors/behaviours/{key}/family/export?")


async def test_archive_bytes_deterministic(client, async_session, data_dir, request):
    key, _model, _probes = await family(client, async_session, data_dir, request)
    first = (await client.get(f"{API}/behaviours/{key}/family/export")).content
    second = (await client.get(f"{API}/behaviours/{key}/family/export")).content
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as zf:
        assert {i.date_time for i in zf.infolist()} == {(1980, 1, 1, 0, 0, 0)}


async def test_two_blockers_both_listed(client, async_session, data_dir, request):
    key, _model, probes = await family(client, async_session, data_dir, request, rungs=[2, 1, 2])
    probes[2].definition_path = None                                      # not built
    await async_session.commit()
    response = await client.get(f"{API}/behaviours/{key}/family/export-index")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "family_export_blocked"
    assert {(b["probe_id"], b["reason"]) for b in detail["blockers"]} == {
        (probes[1].id, "below_rung2_unacknowledged"), (probes[2].id, "no_current_definition")}
    assert {n["arguments"]["probe_id"] for n in detail["next"]} == {probes[1].id, probes[2].id}
    archive = await client.get(f"{API}/behaviours/{key}/family/export")
    assert archive.status_code == 409


async def test_a_missing_file_and_a_name_collision_block(client, async_session, data_dir, request):
    key, _model, probes = await family(client, async_session, data_dir, request)
    (data_dir / "probe_definitions" / f"{probes[0].id}.probe.json").unlink()
    for probe in probes[1:]:
        (data_dir / "probe_definitions" / f"{probe.id}.probe.json").write_text(json.dumps(
            {"name": "same-name", "evidence": {"rung": 2}}))
    detail = (await client.get(f"{API}/behaviours/{key}/family/export-index")).json()["detail"]
    reasons = {b["reason"] for b in detail["blockers"]}
    assert reasons == {"file_missing", "member_filename_collision"}
