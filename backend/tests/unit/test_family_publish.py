"""Family publish and the repository README (Feature 041 FR-24; T15, PT-7; FTASKS 13.3 – 13.4).

A mocked `HfApi` records what would be sent: ONE `upload_folder` call holding every member's untouched
bytes, the manifest merged once per member, and a README with a section per probe in the merged manifest —
also for the single publish (the pre-existing one-probe card, fixed). The token reaches nothing it should not.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from src.core.config import settings
from src.services import probe_definition_publisher as publisher
from tests.support import family_fixtures as ff
from tests.support.random_behaviour import rng_for
from tests.unit.test_probe_definition_publisher import _FakeApi

API = "/api/v1/probe-monitors"
TOKEN = "hf_" + "s3cr3t" * 6


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    return tmp_path


@pytest.fixture
def sync_db(async_engine):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", "")))
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def definition(name, layer):
    return {"name": name, "kind": "mistudio.probe-definition/v1", "model": {"hf_id": "org/base", "revision": "r" * 40},
            "read": {"layer": layer, "hook_point": "resid_post"}, "aggregation": {"rule": "mean"},
            "evidence": {"rung": 2, "rung_language": "detects on unseen tasks", "evaluations": []},
            "decision": {"threshold": 0.5}, "test_vectors": {"vectors": [], "tolerance": 0.001}, "basis": "dense"}


async def family(client, async_session, data_dir, request, n=3, names=None):
    key = await ff.catalogue(client, rng_for(request.node.nodeid))
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    probes = []
    for i in range(n):
        probe = await ff.member(async_session, model, ds, key=key, role="primary" if i == 0 else None, layer=i)
        path = data_dir / "probe_definitions" / f"{probe.id}.probe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        name = (names or [f"probe-{i}-{probe.id[-4:]}" for i in range(n)])[i]
        path.write_bytes(json.dumps(definition(name, i), indent=1).encode() + b"\n")
        probe.definition_path = str(path)
        probes.append(probe)
    await async_session.commit()
    return key, probes


async def test_one_upload_folder_call_with_every_file(client, async_session, data_dir, request, sync_db):
    _key, probes = await family(client, async_session, data_dir, request)
    api = _FakeApi()
    publisher.publish_family(sync_db, [p.id for p in probes], repo_id="owner/fam", api=api, token=TOKEN)
    assert [c[0] for c in api.calls].count("upload_folder") == 1
    folder = Path(api.uploaded["folder_path"])
    files = sorted(p.name for p in folder.iterdir())
    names = [json.loads(Path(p.definition_path).read_bytes())["name"] for p in probes]
    assert files == sorted([f"{n}.probe.json" for n in names] + [publisher.MANIFEST_NAME, publisher.README_NAME])
    for probe, name in zip(probes, names):
        assert (folder / f"{name}.probe.json").read_bytes() == Path(probe.definition_path).read_bytes()
    for probe in probes:
        sync_db.refresh(sync_db.get(type(probe), probe.id))
        row = sync_db.get(type(probe), probe.id)
        assert row.published[-1]["repo_id"] == "owner/fam"


async def test_manifest_equals_merge_once_per_member(client, async_session, data_dir, request, sync_db):
    _key, probes = await family(client, async_session, data_dir, request)
    existing = {"kind": "mistudio.probe-manifest/v1", "probes": [{"name": "older-probe", "file": "older-probe.probe.json",
                                                                   "sha256": "f" * 64}]}
    api = _FakeApi(existing_manifest=existing)
    publisher.publish_family(sync_db, [p.id for p in probes], repo_id="owner/fam", api=api, token=TOKEN)
    manifest = json.loads((Path(api.uploaded["folder_path"]) / publisher.MANIFEST_NAME).read_text())
    by_name = {e["name"]: e for e in manifest["probes"]}
    assert "older-probe" in by_name and len(manifest["probes"]) == 4
    for probe in probes:
        payload = Path(probe.definition_path).read_bytes()
        name = json.loads(payload)["name"]
        assert by_name[name]["sha256"] == hashlib.sha256(payload).hexdigest()


async def test_readme_has_a_section_per_probe(client, async_session, data_dir, request, sync_db):
    _key, probes = await family(client, async_session, data_dir, request)
    existing = {"kind": "mistudio.probe-manifest/v1", "probes": [{"name": "older-probe", "file": "older-probe.probe.json",
                                                                   "sha256": "f" * 64, "rung": 2}]}
    api = _FakeApi(existing_manifest=existing)
    publisher.publish_family(sync_db, [p.id for p in probes], repo_id="owner/fam", api=api, token=TOKEN)
    readme = (Path(api.uploaded["folder_path"]) / publisher.README_NAME).read_text()
    for probe in probes:
        assert f"## {json.loads(Path(probe.definition_path).read_bytes())['name']}" in readme
    assert "## older-probe" in readme and "Published earlier" in readme
    assert readme.count("mistudio-probe-definition") == 1                 # the front matter, once
    # The SINGLE publish uses the same renderer: a repo already holding another probe describes both.
    api2 = _FakeApi(existing_manifest=existing)
    publisher.publish(sync_db, probes[0].id, repo_id="owner/one", api=api2, token=TOKEN)
    single = (Path(api2.uploaded["folder_path"]) / publisher.README_NAME).read_text()
    assert "## older-probe" in single and f"## {json.loads(Path(probes[0].definition_path).read_bytes())['name']}" in single


async def test_collision_refused_before_any_hugging_face_call(client, async_session, data_dir, request, sync_db):
    _key, probes = await family(client, async_session, data_dir, request, names=["dup", "dup", "other"])
    api = _FakeApi()
    with pytest.raises(publisher.FamilyPublishRefused) as caught:
        publisher.publish_family(sync_db, [p.id for p in probes], repo_id="owner/fam", api=api, token=TOKEN)
    assert caught.value.code == "member_filename_collision" and api.calls == []


async def test_token_never_logged_or_stored(client, async_session, data_dir, request, sync_db):
    _key, probes = await family(client, async_session, data_dir, request)
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    logger = logging.getLogger("src.services.probe_definition_publisher")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        api = _FakeApi()
        result = publisher.publish_family(sync_db, [p.id for p in probes], repo_id="owner/fam", api=api, token=TOKEN)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    assert records, "the publisher logged nothing, so this check could not fail"
    assert all(TOKEN not in r for r in records)
    assert TOKEN not in json.dumps(result)
    for path in Path(api.uploaded["folder_path"]).iterdir():
        assert TOKEN not in path.read_text()
    for probe in probes:
        assert TOKEN not in json.dumps(sync_db.get(type(probe), probe.id).published)


async def test_the_route_checks_the_token_before_queueing(client, async_session, data_dir, request, monkeypatch):
    from src.api.v1.endpoints import probe_monitors
    from src.workers import probe_monitor_tasks

    key, probes = await family(client, async_session, data_dir, request)
    sent = []
    monkeypatch.setattr(probe_monitor_tasks.publish_probe_family, "delay",
                        lambda **kw: sent.append(kw) or type("T", (), {"id": "t-pub"})())
    monkeypatch.setattr("src.services.huggingface_sae_service.resolve_hf_token", lambda t: t or TOKEN)
    monkeypatch.setattr(probe_monitors, "_huggingface_rejects", lambda token: "rejected" if token == "bad" else None)
    bad = await client.post(f"{API}/behaviours/{key}/family/publish", json={"repo_id": "o/r", "token": "bad"})
    assert bad.status_code == 401 and sent == []
    ok = await client.post(f"{API}/behaviours/{key}/family/publish", json={"repo_id": "o/r", "private": False})
    assert ok.status_code == 202, ok.text
    assert sent == [{"behaviour_key": key, "member_ids": [p.id for p in probes], "repo_id": "o/r", "private": False,
                     "token": TOKEN}]
    assert TOKEN not in ok.text
