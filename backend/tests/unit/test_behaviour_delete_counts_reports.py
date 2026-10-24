"""A behaviour with blend reports is in use (Feature 041 PT-9; FTASKS 7.9)."""
from __future__ import annotations

from src.core.config import settings
from tests.support import family_fixtures as ff
from tests.support.random_behaviour import rng_for

API = "/api/v1/probe-monitors"


async def test_a_report_keeps_the_behaviour_from_deletion(client, async_session, request, monkeypatch):
    from src.models.probe_blend import ProbeBlendReport

    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    key = await ff.catalogue(client, rng_for(request.node.nodeid))
    async_session.add(ProbeBlendReport(behaviour_key=key, behaviour_version=1, model_id="m", status="done",
                                       request={}, resolved={}, created_by="operator"))
    await async_session.commit()
    response = await client.delete(f"{API}/behaviours/{key}")
    assert response.status_code == 409
    assert response.json()["detail"]["details"]["in_use"] == {"blend_reports": 1}


async def test_without_a_report_it_deletes(client, request, monkeypatch):
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    key = await ff.catalogue(client, rng_for(request.node.nodeid))
    assert (await client.delete(f"{API}/behaviours/{key}")).status_code == 204
