"""Feature 041 works without its siblings (FR-30; FTASKS 2.11, 14.4; suite rule `midataworks-standalone-first`).

* No miLLM: the family reads, and the checklist's `armed` step is `unknown` with the reason — never
  `not_done`.
* No miDataworks: views registered by hand (no dataset-version manifest) carry a sub-question and drive the
  family's coverage exactly as sent ones do.
"""
from __future__ import annotations

import pytest

from src.core.config import settings
from tests.support import family_fixtures as ff
from tests.support.detector_fixtures import dataset_row, write_split
from tests.support.next_calls import assert_next_calls_are_live
from tests.support.random_behaviour import rng_for

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def alone(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    monkeypatch.setattr(settings, "midataworks_api_url", "", raising=False)
    return tmp_path


async def test_family_without_millm_reads_armed_unknown(client, async_session, request):
    rng = rng_for(request.node.nodeid)
    key = await ff.catalogue(client, rng)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    probe = await ff.member(async_session, model, ds, key=key, role="primary")
    await async_session.commit()
    family = await client.get(f"{API}/behaviours/{key}/family")
    assert family.status_code == 200 and [m["probe_id"] for m in family.json()["members"]] == [probe.id]
    steps = {s["id"]: s for s in (await client.get(f"{API}/behaviours/{key}/checklist")).json()["steps"]}
    assert steps["armed"]["state"] == "unknown" and steps["armed"]["reason"]


async def test_family_from_hand_registered_views(client, async_session, alone, request):
    rng = rng_for(request.node.nodeid)
    a, b = ff.sub_keys(rng, 2)
    key = await ff.catalogue(client, rng, sub_questions=[a, b])
    model = await ff.model(async_session)
    ds = await dataset_row(async_session, write_split(alone, {"text": [f"r{i}" for i in range(48)],
                                                              "label": [str(i % 2) for i in range(48)]}))
    await async_session.commit()
    view = await client.post(f"{API}/datasets", json={
        "name": "hand", "dataset_id": str(ds.id), "input_column": "text", "label_column": "label",
        "label_mapping": {"1": "positive", "0": "negative"}, "role": "train", "behaviour": key, "sub_question": b})
    assert view.status_code == 201, view.text
    assert view.json()["dataset_version"] is None                     # registered by hand, no manifest
    await ff.member(async_session, model, ds, key=key, sub_question=a)
    await async_session.commit()
    body = (await client.get(f"{API}/behaviours/{key}/family")).json()
    missing = next(c for c in body["coverage"] if c["sub_question"] == b)
    assert missing["next"][0]["arguments"] == {"model_id": model.id, "train_dataset_id": view.json()["id"],
                                               "behaviour": key}
    assert_next_calls_are_live(missing["next"])


async def test_blend_score_and_export_without_siblings(client, async_session, alone, request, monkeypatch):
    """No miDataworks (the blend views are registered by hand, no manifest) and no miLLM: a blend report is
    built, and the family exports (FR-30 for S2 – S3)."""
    import json

    from tests.support import blend_world as bw

    world = await bw.build(client, async_session, seed=request.node.nodeid, dataset_version=None)
    for probe in world.members:
        path = alone / "defs" / f"{probe.id}.probe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": f"d-{probe.id}", "evidence": {"rung": 2}}))
        probe.definition_path = str(path)
    await async_session.commit()
    views = {"blend_fit": world.blend_fit.id, "tests": [v.id for v in world.tests], "calibration": world.calibration.id}
    report_id = await bw.done_report(client, monkeypatch, world, views=views, dataset_version=None)
    assert (await client.get(f"{API}/blend-reports/{report_id}")).json()["status"] == "done"
    exported = await client.get(f"{API}/behaviours/{world.key}/family/export-index")
    assert exported.status_code == 200 and len(exported.json()["members"]) == 3
