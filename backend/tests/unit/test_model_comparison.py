"""Models side by side (Feature 040 S2 FR-26; FTASKS 11.4 – 11.5)."""
from __future__ import annotations

import pytest

from src.core.config import settings
from tests.support.random_behaviour import rng_for
from tests.unit.test_detector_success_report import build_world, held

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


async def test_three_models_render_three_rows(client, async_session, data_dir, request):
    w = await build_world(client, async_session, data_dir, rng_for(request.node.nodeid), models=3,
                          criteria=lambda w: [held(w)])
    response = await client.get(f"{API}/behaviours/{w.key}/model-comparison")
    assert response.status_code == 200, response.text
    body = response.json()
    assert sorted(r["model_id"] for r in body["models"]) == sorted(m.id for m in w.models)
    assert body["total"] == 3 and body["next_cursor"] is None
    for row in body["models"]:
        assert row["overall_verdict"] == "met" and row["criteria"][0]["id"] == "crit-held"
        assert row["blend"] == {"verdict": "not_assessable", "reasons": ["no_blend_report"]}
        assert row["report_id"].startswith("dsr_")


async def test_the_definitions_models_are_the_ones_compared(client, async_session, data_dir, request):
    w = await build_world(client, async_session, data_dir, rng_for(request.node.nodeid), models=3,
                          criteria=lambda w: [held(w)])
    await client.post(f"{API}/behaviours/{w.key}/versions", json={
        "base_version": 1, "models": [{"model_id": w.models[2].id}, {"model_id": w.models[0].id}]})
    body = (await client.get(f"{API}/behaviours/{w.key}/model-comparison")).json()
    assert [r["model_id"] for r in body["models"]] == [w.models[2].id, w.models[0].id]


async def test_models_trained_on_different_dataset_versions_are_refused(client, async_session, data_dir, request):
    w = await build_world(client, async_session, data_dir, rng_for(request.node.nodeid), models=2,
                          criteria=lambda w: [held(w)])
    # the second model's probe was trained on another dataset version
    from tests.support.detector_fixtures import run_and_probe, view_row

    other = await view_row(async_session, w.train_ds, role="train", behaviour=w.key,
                           dataset_version={"version_id": "dv-other"})
    w.train.dataset_version = {"version_id": "dv-one"}
    for probe in w.probes:
        probe.behaviour = None  # retire both, then train one per version
    await run_and_probe(async_session, w.models[0], w.train, eval_views=[w.ood], calibration_view=w.cal,
                        behaviour=w.key, target_fpr=0.03, n_negatives=5000)
    await run_and_probe(async_session, w.models[1], other, eval_views=[w.ood], calibration_view=w.cal,
                        behaviour=w.key, target_fpr=0.03, n_negatives=5000)
    await async_session.commit()
    response = await client.get(f"{API}/behaviours/{w.key}/model-comparison")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "dataset_identity_mismatch"
    assert detail["difference"] == {w.models[0].id: "dv-one", w.models[1].id: "dv-other"}


async def test_a_refused_comparison_stores_no_report(client, async_session, data_dir, request):
    from sqlalchemy import select

    from src.models.detector_report import DetectorSuccessReport

    await test_models_trained_on_different_dataset_versions_are_refused(client, async_session, data_dir, request)
    rows = (await async_session.execute(select(DetectorSuccessReport))).scalars().all()
    assert rows == []
