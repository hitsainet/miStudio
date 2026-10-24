"""A catalogue page costs the same at any catalogue size, measured at the ROUTE (Feature 040 FR-6; FTID D-7).

037's list route ran `current_version` once per head and the full checklist once per row, so its cost grew
with the catalogue. The count here is every SQL statement the request sends (`tests/support/
query_counter.py`), for a page of the same size over catalogues of 100 and 1,000 behaviours, and must be
EQUAL — not merely small. Each behaviour has a primary probe, so the batched primary read and the miLLM
read are on the path, and miLLM is asserted read once per request.

The compatibility test freezes the 037 list algorithm (per-head loop, `head_out` per row) and requires the
paged route's `behaviours` and `uncatalogued` to equal it byte for byte for a catalogue under one page.
"""
from __future__ import annotations

import json

import httpx
import pytest
from sqlalchemy import insert, select

from src.models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion
from src.services import millm_read
from tests.support.query_counter import count_queries
from tests.support.random_behaviour import rng_for, slug
from tests.unit.test_behaviour_catalogue_paging import seed

API = "/api/v1/probe-monitors"


class FakeMiLLM:
    def __init__(self):
        self.requests = []

    def handler(self, request):
        self.requests.append(request)
        return httpx.Response(200, json={"success": True, "data": []})


@pytest.fixture
def millm(monkeypatch):
    from src.core.config import settings

    fake = FakeMiLLM()
    monkeypatch.setattr(settings, "millm_api_url", "http://millm.test")
    monkeypatch.setattr(millm_read, "_transport_factory", lambda: httpx.MockTransport(fake.handler))
    return fake


async def primaries(session, keys):
    """Give each behaviour a primary probe (its own key), so the batched primary read is exercised."""
    import uuid

    from src.core.clock import utc_now
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_behaviour import ProbeBehaviourProbeRole
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/nonexistent", architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="local", status=DatasetStatus.READY,
                      raw_path="/nonexistent", extra_metadata={})
    session.add_all([model, dataset])
    await session.flush()
    view = ProbeMonitorDataset(name="v", dataset_id=dataset.id, input_column="i", label_column="l",
                               label_mapping={}, role="train", counts={})
    session.add(view)
    await session.flush()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, status="completed", config={})
    session.add(run)
    await session.flush()
    now = utc_now()
    probe_rows = [{"id": f"pm_{uuid.uuid4().hex[:12]}", "run_id": run.id, "layer": 1, "rule": "mean",
                   "variant": "dense", "behaviour": key, "created_at": now} for key in keys]
    await session.execute(insert(ProbeMonitor), probe_rows)
    await session.execute(insert(ProbeBehaviourProbeRole), [
        {"probe_id": row["id"], "key": row["behaviour"], "role": "primary", "updated_by": "operator",
         "updated_at": now} for row in probe_rows])
    await session.commit()


async def _count_first_page(client, async_engine, **params):
    with count_queries(async_engine) as counter:
        response = await client.get(f"{API}/behaviours", params=params)
    assert response.status_code == 200, response.text
    return counter.count, response.json()


@pytest.mark.parametrize("size", [100, 1000])
async def test_page_query_count_is_independent_of_catalogue_size(client, async_session, async_engine, millm,
                                                                 request, size):
    rng = rng_for(request.node.nodeid)
    keys = sorted({slug(rng) for _ in range(size)})
    await seed(async_session, keys)
    await primaries(async_session, keys)
    count, body = await _count_first_page(client, async_engine, limit=50)
    assert len(body["behaviours"]) == 50
    assert len(millm.requests) == 1                      # miLLM once per request, not per row
    if size == 1000:
        # page two of the large catalogue costs the same as page one
        millm.requests.clear()
        with count_queries(async_engine) as again:
            second = await client.get(f"{API}/behaviours", params={"limit": 50, "cursor": body["next_cursor"]})
        # Page two reads the same statements except the uncatalogued page, whose side ended on page one.
        assert second.status_code == 200 and again.count == count - 1, (count, again.statements)


async def test_query_count_equal_within_one_run(client, async_session, async_engine, millm, request):
    """The same comparison inside one test (independent of parametrise order): grow the catalogue ten-fold
    between two identical requests; the statement count must not move."""
    rng = rng_for(request.node.nodeid)
    first_keys = sorted({slug(rng) for _ in range(100)})
    await seed(async_session, first_keys)
    await primaries(async_session, first_keys)
    small, _ = await _count_first_page(client, async_engine, limit=50)
    more = sorted({slug(rng) for _ in range(900)} - set(first_keys))
    await seed(async_session, more)
    await primaries(async_session, more)
    large, body = await _count_first_page(client, async_engine, limit=50)
    assert body["total"] == len(first_keys) + len(more)
    assert small == large, (small, large)


# ── 037 compatibility: under one page and no new parameters, the arrays are 037's ──────────────


async def frozen_037_list(db):
    """The 037 `list_behaviours` algorithm, frozen here as the reference (per-head loop)."""
    from src.services.behaviour_catalogue import current_version, head_out, uncatalogued_keys, usage_counts

    heads = list((await db.execute(
        select(ProbeBehaviour).where(ProbeBehaviour.archived_at.is_(None)).order_by(ProbeBehaviour.key.asc())
    )).scalars().all())
    counts = await usage_counts(db, [h.key for h in heads]) if heads else {}
    rows = []
    for head in heads:
        out = head_out(head, await current_version(db, head))
        out["counts"] = counts.get(head.key, {"prompts": 0, "runs": 0, "probes": 0, "views": 0})
        rows.append(out)
    return rows, await uncatalogued_keys(db)


async def test_under_one_page_the_arrays_are_037s(client, async_session, millm, request):
    from src.services import behaviour_checklist

    rng = rng_for(request.node.nodeid)
    keys = sorted({slug(rng) for _ in range(7)})
    await seed(async_session, keys)
    await primaries(async_session, keys[:3])
    body = (await client.get(f"{API}/behaviours")).json()
    rows, uncatalogued = await frozen_037_list(async_session)
    for row in rows:
        row["status"] = (await behaviour_checklist.checklist(async_session, row["key"], lambda: set()))["status"]
    assert json.dumps(body["behaviours"]) == json.dumps(rows)
    assert json.dumps(body["uncatalogued"]) == json.dumps(uncatalogued)
    assert body["next_cursor"] is None and body["total"] == 7
    assert set(body) == {"behaviours", "uncatalogued", "next_cursor", "total", "total_exact",
                         "uncatalogued_total", "limit"}


async def test_every_listed_row_is_the_detail_serialiser(client, async_session, millm, request):
    """037's one-serialiser rule holds for the paged list: a row equals the detail's head fields."""
    rng = rng_for(request.node.nodeid)
    keys = sorted({slug(rng) for _ in range(3)})
    await seed(async_session, keys)
    body = (await client.get(f"{API}/behaviours")).json()
    for row in body["behaviours"]:
        detail = (await client.get(f"{API}/behaviours/{row['key']}")).json()
        shared = {k: v for k, v in detail.items() if k not in ("adopted", "counts")}
        assert {k: v for k, v in row.items() if k not in ("counts", "status")} == shared


async def test_page_query_count_is_independent_of_page_size(client, async_session, async_engine, millm, request):
    """ONE batched read per page, not one per row: a page of 5 and a page of 60 send the same number of
    statements (controls P1/P5: a per-head `current_version` or per-row checklist grows with the page).
    Found as surviving mutations in the S1 review — the size comparison alone held the page fixed."""
    rng = rng_for(request.node.nodeid)
    keys = sorted({slug(rng) for _ in range(80)})
    await seed(async_session, keys)
    await primaries(async_session, keys)
    small, body_small = await _count_first_page(client, async_engine, limit=5)
    large, body_large = await _count_first_page(client, async_engine, limit=60)
    assert (len(body_small["behaviours"]), len(body_large["behaviours"])) == (5, 60)
    assert small == large, (small, large)


async def test_millm_is_not_asked_when_no_behaviour_has_a_primary(client, async_session, millm, request):
    """The armed step is read only for a primary probe; a page without one costs miLLM nothing (P7)."""
    rng = rng_for(request.node.nodeid)
    await seed(async_session, sorted({slug(rng) for _ in range(4)}))
    body = (await client.get(f"{API}/behaviours")).json()
    assert len(body["behaviours"]) == 4 and millm.requests == []
