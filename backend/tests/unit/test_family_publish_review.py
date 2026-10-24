"""Family publish — the route's all-or-nothing gate (Feature 041 FR-24; controls record § Review R1).

B-05: deleting `export.refuse_if_blocked` from the PUBLISH route survived the S3 suite. The task re-checks only
that a definition file exists, so a member below rung 2 without its acknowledgement, or one whose built
definition no longer states the decision in force, would have been published to the Hub.
"""
from __future__ import annotations

import json

from tests.unit.test_family_publish import API, TOKEN, data_dir, family, sync_db  # noqa: F401 (fixtures)


async def test_a_blocked_family_is_refused_before_anything_is_queued(client, async_session, data_dir, request,
                                                                    monkeypatch):
    from src.api.v1.endpoints import probe_monitors
    from src.workers import probe_monitor_tasks

    key, probes = await family(client, async_session, data_dir, request)
    path = data_dir / "probe_definitions" / f"{probes[1].id}.probe.json"
    document = json.loads(path.read_bytes())
    document["evidence"]["rung"] = 1                                       # below rung 2, no acknowledgement
    path.write_bytes(json.dumps(document, indent=1).encode() + b"\n")
    sent = []
    monkeypatch.setattr(probe_monitor_tasks.publish_probe_family, "delay",
                        lambda **kw: sent.append(kw) or type("T", (), {"id": "t-pub"})())
    monkeypatch.setattr("src.services.huggingface_sae_service.resolve_hf_token", lambda t: t or TOKEN)
    monkeypatch.setattr(probe_monitors, "_huggingface_rejects", lambda token: None)
    response = await client.post(f"{API}/behaviours/{key}/family/publish", json={"repo_id": "o/r"})
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "family_export_blocked"
    assert [(b["probe_id"], b["reason"]) for b in detail["blockers"]] == [(probes[1].id, "below_rung2_unacknowledged")]
    assert sent == []


async def test_the_repository_is_created_with_the_requested_visibility(client, async_session, data_dir, request,
                                                                       sync_db):
    """B-07: the visibility the operator chose reaches `create_repo` (and the `published` record), both ways —
    the route test pinned the task's arguments, nothing pinned what the task sends to the Hub."""
    from src.services import probe_definition_publisher as publisher
    from tests.unit.test_probe_definition_publisher import _FakeApi

    _key, probes = await family(client, async_session, data_dir, request)
    for private in (True, False):
        api = _FakeApi()
        publisher.publish_family(sync_db, [p.id for p in probes], repo_id=f"owner/fam-{private}", private=private,
                                 api=api, token=TOKEN)
        assert api.created["private"] is private
        for probe in probes:
            sync_db.expire_all()
            assert sync_db.get(type(probe), probe.id).published[-1]["private"] is private


async def test_the_task_rechecks_the_blockers_before_any_hugging_face_call(client, async_session, data_dir, request,
                                                                          sync_db):
    """A member that became blocked while the publish waited in the queue stops the whole commit at upload time."""
    import pytest

    from src.services import probe_definition_publisher as publisher
    from tests.unit.test_probe_definition_publisher import _FakeApi

    _key, probes = await family(client, async_session, data_dir, request)
    path = data_dir / "probe_definitions" / f"{probes[2].id}.probe.json"
    document = json.loads(path.read_bytes())
    document["evidence"]["rung"] = 0
    path.write_bytes(json.dumps(document, indent=1).encode() + b"\n")
    api = _FakeApi()
    with pytest.raises(publisher.FamilyPublishRefused) as refused:
        publisher.publish_family(sync_db, [p.id for p in probes], repo_id="owner/fam", api=api, token=TOKEN)
    assert refused.value.code == "family_export_blocked"
    assert [(b["probe_id"], b["reason"]) for b in refused.value.details["blockers"]] == [
        (probes[2].id, "below_rung2_unacknowledged")]
    assert api.calls == []
