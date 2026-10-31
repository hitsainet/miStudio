"""The behaviour-detector routes are served, and work with no sibling (Feature 040 S2; FTASKS 6.6, 11.4, 12.6).

Read from `app.openapi()['paths']`, never `app.routes` (miStudio memory `app-routes-is-not-a-route-list`):
removing the `include_router` line in `api/v1/router.py` turns `test_routes_in_openapi` red.
"""
from __future__ import annotations

import pytest

from src.core.config import settings

PATHS = {
    "/api/v1/probe-monitors/datasets/{dataset_id}/columns": {"patch"},
    "/api/v1/probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/breakdown": {"get"},
    "/api/v1/probe-monitors/behaviours/{key}/success-reports": {"post", "get"},
    "/api/v1/probe-monitors/success-reports/{report_id}": {"get"},
    "/api/v1/probe-monitors/behaviours/{key}/model-comparison": {"get"},
}


def test_routes_in_openapi():
    from src.main import app

    paths = app.openapi()["paths"]
    for path, methods in PATHS.items():
        assert path in paths, path
        assert methods <= set(paths[path]), (path, set(paths[path]))


@pytest.fixture
def standalone(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "midataworks_api_url", "", raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


async def test_every_s2_route_works_with_no_sibling(client, async_session, standalone, request):
    """FR-45: miDataworks and miLLM both absent — the breakdown, report, list, read and comparison all
    answer, and the checklist's armed step reads `unknown` with its reason, never `not_done`."""
    from tests.support.random_behaviour import rng_for
    from tests.unit.test_detector_success_report import build_world, held

    w = await build_world(client, async_session, standalone, rng_for(request.node.nodeid), criteria=lambda w: [held(w)])
    api = "/api/v1/probe-monitors"
    breakdown = await client.get(f"{api}/probes/{w.probes[0].id}/evaluations/{w.ood_eval.id}/breakdown",
                                 params={"by_column": w.column})
    assert breakdown.status_code == 200, breakdown.text
    made = await client.post(f"{api}/behaviours/{w.key}/success-reports", json={})
    assert made.status_code == 200, made.text
    assert (await client.get(f"{api}/success-reports/{made.json()['id']}")).status_code == 200
    assert (await client.get(f"{api}/behaviours/{w.key}/success-reports")).status_code == 200
    assert (await client.get(f"{api}/behaviours/{w.key}/model-comparison")).status_code == 200
    patched = await client.patch(f"{api}/datasets/{w.ood.id}/columns", json={"group_columns": [w.column, "label"]})
    assert patched.status_code == 200, patched.text
    # make the probe primary so the armed step is reached
    from src.models.probe_behaviour import ProbeBehaviourProbeRole

    async_session.add(ProbeBehaviourProbeRole(probe_id=w.probes[0].id, key=w.key, role="primary", updated_by="operator"))
    await async_session.commit()
    steps = {s["id"]: s for s in (await client.get(f"{api}/behaviours/{w.key}/checklist")).json()["steps"]}
    assert steps["armed"]["state"] == "unknown" and steps["armed"]["reason"]
