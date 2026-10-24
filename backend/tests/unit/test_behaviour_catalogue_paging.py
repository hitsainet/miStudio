"""The behaviour catalogue in pages (Feature 040 FR-6; FTASKS 3.1 – 3.3).

A catalogue of 1,000 random behaviours is listed page by page through the REAL route: the union of the
pages is the whole set, in key order, with no repeat; `q` and `status` filter; a cursor carries its
filters and refuses others. The route-level query count is in `test_behaviour_list_route_queries.py`.
"""
from __future__ import annotations

import pytest
from sqlalchemy import insert

from src.models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion
from src.models.probe_monitor import ProbeMonitorRun  # noqa: F401 - mapped for the usage counts
from tests.support.next_calls import assert_next_calls_are_live
from tests.support.random_behaviour import rng_for, slug

API = "/api/v1/probe-monitors"


@pytest.fixture
def no_millm(monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "millm_api_url", "")


async def seed(session, keys, *, names=None, archived=()):
    """Head + version 1 rows for `keys`, inserted in two statements."""
    from src.core.clock import utc_now

    now = utc_now()
    await session.execute(insert(ProbeBehaviour), [
        {"key": k, "current_version": 1, "created_by": "operator", "created_at": now, "updated_at": now,
         "archived_at": now if k in archived else None}
        for k in keys
    ])
    await session.execute(insert(ProbeBehaviourVersion), [
        {"key": k, "version": 1, "name": (names or {}).get(k, f"name {k}"), "positive_meaning": "p",
         "created_by": "operator", "created_at": now}
        for k in keys
    ])
    await session.commit()


async def all_pages(client, **params):
    pages, cursor = [], None
    while True:
        query = dict(params, **({"cursor": cursor} if cursor else {}))
        response = await client.get(f"{API}/behaviours", params=query)
        assert response.status_code == 200, response.text
        body = response.json()
        pages.append(body)
        cursor = body["next_cursor"]
        if cursor is None:
            return pages
        assert len(pages) < 100, "the listing does not end"


async def test_a_thousand_behaviours_list_completely_in_order(client, async_session, no_millm, request):
    rng = rng_for(request.node.nodeid)
    keys = sorted({slug(rng) for _ in range(1000)})
    await seed(async_session, keys)
    pages = await all_pages(client, limit=137)
    listed = [row["key"] for page in pages for row in page["behaviours"]]
    assert listed == keys                                    # union = the set, stable order, no repeat
    assert all(page["total"] == len(keys) and page["total_exact"] for page in pages)
    assert [len(p["behaviours"]) for p in pages[:-1]] == [137] * (len(pages) - 1)
    assert all(row["status"] == "draft" for page in pages for row in page["behaviours"])


async def test_the_default_page_is_the_runtime_setting(client, async_session, no_millm, request):
    from src.core import runtime_settings

    rng = rng_for(request.node.nodeid)
    keys = sorted({slug(rng) for _ in range(12)})
    await seed(async_session, keys)
    try:
        runtime_settings.apply({"behaviour_list_page_default": 5})
        body = (await client.get(f"{API}/behaviours")).json()
        assert [r["key"] for r in body["behaviours"]] == keys[:5] and body["limit"] == 5
        runtime_settings.apply({"behaviour_list_page_max": 7})
        over = await client.get(f"{API}/behaviours", params={"limit": 8})
        assert over.status_code == 422
        assert (over.json()["detail"]["code"], over.json()["detail"]["setting"]) == ("over_cap", "behaviour_list_page_max")
    finally:
        runtime_settings.reset()


async def test_q_matches_key_or_current_name(client, async_session, no_millm, request):
    rng = rng_for(request.node.nodeid)
    needle = slug(rng, words=1)
    by_key = f"{slug(rng, words=1)}-{needle}"
    by_name = slug(rng)
    other = slug(rng)
    await seed(async_session, [by_key, by_name, other], names={by_name: f"About {needle.upper()} things"})
    pages = await all_pages(client, q=needle, limit=1)
    assert sorted(r["key"] for p in pages for r in p["behaviours"]) == sorted([by_key, by_name])
    assert pages[0]["total"] == 2


async def test_q_is_literal_not_a_pattern(client, async_session, no_millm, request):
    rng = rng_for(request.node.nodeid)
    await seed(async_session, [slug(rng), slug(rng)])
    body = (await client.get(f"{API}/behaviours", params={"q": "%"})).json()
    assert body["behaviours"] == [] and body["total"] == 0


async def test_archived_are_hidden_unless_asked(client, async_session, no_millm, request):
    rng = rng_for(request.node.nodeid)
    keys = sorted({slug(rng) for _ in range(6)})
    await seed(async_session, keys, archived={keys[1], keys[4]})
    shown = [r["key"] for p in await all_pages(client, limit=2) for r in p["behaviours"]]
    assert shown == [k for k in keys if k not in (keys[1], keys[4])]
    every = [r["key"] for p in await all_pages(client, limit=2, include_archived=True) for r in p["behaviours"]]
    assert every == keys


async def test_status_filter_scans_and_says_when_the_total_is_exact(client, async_session, no_millm, request):
    rng = rng_for(request.node.nodeid)
    keys = sorted({slug(rng) for _ in range(9)})
    await seed(async_session, keys, archived={keys[2], keys[7]})
    pages = await all_pages(client, status="archived", include_archived=True, limit=1)
    assert [r["key"] for p in pages for r in p["behaviours"]] == [keys[2], keys[7]]
    assert pages[0]["total_exact"] is False and pages[0]["total"] is None
    one_page = (await client.get(f"{API}/behaviours", params={"status": "archived", "include_archived": True,
                                                               "limit": 50})).json()
    assert (one_page["total"], one_page["total_exact"], one_page["next_cursor"]) == (2, True, None)


async def test_uncatalogued_keys_page_by_their_own_position(client, async_session, sync_free_keys, no_millm):
    pages = await all_pages(client, limit=2)
    listed = [u["key"] for p in pages for u in p["uncatalogued"]]
    assert listed == sorted(sync_free_keys)
    assert pages[0]["uncatalogued_total"] == len(sync_free_keys)
    assert all(u["runs"] == 1 for p in pages for u in p["uncatalogued"])


@pytest.fixture
async def sync_free_keys(async_session, request):
    """Five keys written on probe runs, none catalogued."""
    from tests.unit.test_behaviour_catalogue import _base, sync_db  # noqa: F401

    rng = rng_for(request.node.nodeid)
    keys = [slug(rng) for _ in range(5)]
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset
    import uuid

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/nonexistent", architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="local", status=DatasetStatus.READY,
                      raw_path="/nonexistent", extra_metadata={})
    async_session.add_all([model, dataset])
    await async_session.flush()
    view = ProbeMonitorDataset(name="v", dataset_id=dataset.id, input_column="i", label_column="l",
                               label_mapping={}, role="train", counts={})
    async_session.add(view)
    await async_session.flush()
    for key in keys:
        async_session.add(ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, status="completed",
                                          config={}, behaviour=key))
    await async_session.commit()
    return keys


async def test_a_cursor_refuses_other_filters(client, async_session, no_millm, request):
    rng = rng_for(request.node.nodeid)
    await seed(async_session, sorted({slug(rng) for _ in range(5)}))
    first = (await client.get(f"{API}/behaviours", params={"limit": 2})).json()
    changed = await client.get(f"{API}/behaviours", params={"limit": 2, "cursor": first["next_cursor"], "q": "x"})
    assert changed.status_code == 422
    detail = changed.json()["detail"]
    assert (detail["code"], detail["field"], detail["rule"]) == ("invalid_field", "cursor", "filter_changed")
    assert_next_calls_are_live(detail["next"])
    garbage = await client.get(f"{API}/behaviours", params={"cursor": "not-a-cursor"})
    assert garbage.status_code == 422 and garbage.json()["detail"]["rule"] == "malformed"
