"""Feature 041's caps are runtime settings (FR-27; T12/T16; FTASKS 2.10, 5.5).

Each cap: one over is refused NAMING the setting, and a runtime change (`runtime_settings.apply`, the
refresh point the environment route uses) moves the cap on the next request — so a reader that reads a
literal, or `settings.<key>` instead of `effective()`, turns its test red.
"""
from __future__ import annotations

import pytest

from src.core import runtime_settings
from src.core.config import settings
from tests.support import family_fixtures as ff
from tests.support.random_behaviour import rng_for

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield
    runtime_settings.reset()


async def test_family_page_one_over_refused_naming_setting(client, async_session, request):
    rng = rng_for(request.node.nodeid)
    key = await ff.catalogue(client, rng)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    for _ in range(4):
        await ff.member(async_session, model, ds, key=key)
    await async_session.commit()

    runtime_settings.apply({"blend_family_page_size": 3})
    default = (await client.get(f"{API}/behaviours/{key}/family")).json()
    assert (default["limit"], len(default["members"]), default["next_offset"]) == (3, 3, 3)
    over = await client.get(f"{API}/behaviours/{key}/family", params={"limit": 4})
    assert over.status_code == 422
    detail = over.json()["detail"]
    assert (detail["code"], detail["setting"], detail["value"], detail["got"]) == (
        "cap_exceeded", "blend_family_page_size", 3, 4)

    runtime_settings.apply({"blend_family_page_size": 4})
    moved = await client.get(f"{API}/behaviours/{key}/family", params={"limit": 4})
    assert moved.status_code == 200 and len(moved.json()["members"]) == 4


# ── S2: the blend caps (5.5) ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dispatched(monkeypatch):
    from src.workers import probe_blend_tasks

    sent = []
    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay",
                        lambda rid: sent.append(rid) or type("T", (), {"id": "t"})())
    return sent


async def test_folds_are_the_runtime_settings(client, async_session, request, dispatched):
    from tests.support import blend_world as bw

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    over = await client.post(f"{API}/blend-reports", json=bw.start_body(world, folds=11))
    assert over.status_code == 422
    assert (over.json()["detail"]["setting"], over.json()["detail"]["value"]) == ("blend_folds_max", 10)
    runtime_settings.apply({"blend_folds_max": 12, "blend_folds_default": 7})
    assert (await client.post(f"{API}/blend-reports", json=bw.start_body(world, folds=11))).status_code == 202
    default = (await client.post(f"{API}/blend-reports", json=bw.start_body(world))).json()
    assert default["resolved"]["folds"] == 7
    runtime_settings.apply({"blend_folds_min": 4})
    under = await client.post(f"{API}/blend-reports", json=bw.start_body(world, folds=3))
    assert (under.status_code, under.json()["detail"]["setting"]) == (422, "blend_folds_min")


def test_the_fold_bounds_must_hold_their_default():
    with pytest.raises(ValueError, match="blend_folds_min"):
        runtime_settings.validated_value("blend_folds_default", 20)


async def test_the_aligned_fraction_is_the_runtime_setting(client, async_session, request, dispatched):
    from tests.support import blend_world as bw

    world = await bw.build(client, async_session, seed=request.node.nodeid, keep=0.9)
    refused = await client.post(f"{API}/blend-reports", json=bw.start_body(world))
    assert refused.status_code == 409 and refused.json()["detail"]["code"] == "members_misaligned"
    runtime_settings.apply({"blend_min_aligned_fraction": 0.7})   # three members keeping 90% share ~81%
    assert (await client.post(f"{API}/blend-reports", json=bw.start_body(world))).status_code == 202


async def test_the_report_records_the_runtime_settings(client, async_session, request, dispatched):
    """`blend_cv_min_rows_per_class` and the caps reach the report's fixed inputs at start."""
    from src.models.probe_blend import ProbeBlendReport
    from tests.support import blend_world as bw

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    runtime_settings.apply({"blend_cv_min_rows_per_class": 37})
    report_id = (await client.post(f"{API}/blend-reports", json=bw.start_body(world))).json()["report_id"]
    row = await async_session.get(ProbeBlendReport, report_id)
    await async_session.refresh(row)
    assert row.resolved["min_rows_per_class"] == 37
