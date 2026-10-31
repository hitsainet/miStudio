"""Every `next` this feature can write names a live tool with arguments it accepts (Feature 040 FR-40; 4.3).

Derived from `detector_next.CONSTRUCTORS` (filled by the constructors themselves, TI-6), never from a hand
list: each constructor's example is checked against the REAL server's registry and the tool's input
schema (`tests/support/next_calls.assert_next_calls_are_live`). The S1 refusals that carry a `next` are
then produced through the real routes and their `next` checked the same way.
"""
from __future__ import annotations

import pytest

from src.services import detector_next
from tests.support.next_calls import assert_next_calls_are_live, live_input_schemas
from tests.support.random_behaviour import behaviour_body, rng_for, slug

API = "/api/v1/probe-monitors"


def test_the_registry_is_filled_by_the_constructors():
    assert len(detector_next.CONSTRUCTORS) >= 8
    assert detector_next.next_tools() == {c.tool for c in detector_next.CONSTRUCTORS.values()}


@pytest.mark.parametrize("name", sorted(detector_next.CONSTRUCTORS))
def test_each_constructor_names_a_live_tool_with_valid_arguments(name):
    constructor = detector_next.CONSTRUCTORS[name]
    entry = constructor.example()
    assert entry["tool"] == constructor.tool
    assert constructor.tool in live_input_schemas(), f"{name} names {constructor.tool!r}, not in the live registry"
    assert_next_calls_are_live([entry])


def test_a_refusal_carries_its_next_on_the_wire():
    refusal = detector_next.DetectorRefused(409, "x", "m", {"a": 1}, next=[detector_next.next_list_models()])
    exc = refusal.to_http()
    assert exc.status_code == 409
    assert exc.detail == {"code": "x", "message": "m", "a": 1, "next": [detector_next.next_list_models()]}


# ── every S1 refusal and warning that carries a `next`, produced through the real routes ──────────


@pytest.fixture
def rng(request):
    return rng_for(request.node.nodeid)


@pytest.fixture
def standalone(monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "midataworks_api_url", "")
    monkeypatch.setattr(settings, "millm_api_url", "")


async def test_every_s1_next_on_the_wire_is_live(client, rng, standalone):
    collected = []
    body = behaviour_body(rng, held_out_columns=[slug(rng)], models=[{"model_id": f"{slug(rng)}/{slug(rng)}"},
                                                                    {"model_id": slug(rng)}])
    made = (await client.post(f"{API}/behaviours", json=body)).json()
    collected += [n for w in made["warnings"] for n in w["next"]]
    assert {w["code"] for w in made["warnings"]} == {"model_not_downloaded", "column_not_on_any_view"}
    key = body["key"]
    await client.post(f"{API}/behaviours/{key}/versions", json={"base_version": 1, "calibration_floor": 3})
    stale = (await client.post(f"{API}/behaviours/{key}/versions", json={"base_version": 1, "calibration_floor": 4})).json()
    collected += stale["detail"]["next"]
    unresolvable = (await client.post(f"{API}/behaviours/{key}/versions", json={
        "base_version": 2, "rule_set": {"name": "n", "version": 1, "digest": "a" * 64}})).json()
    collected += unresolvable["detail"]["next"]
    checklist = (await client.get(f"{API}/behaviours/{key}/checklist")).json()
    step = next(s for s in checklist["steps"] if s["id"] == "definition_complete")
    collected += step["next"]
    first = (await client.get(f"{API}/behaviours", params={"limit": 1})).json()
    await client.post(f"{API}/behaviours", json=behaviour_body(rng))
    changed = (await client.get(f"{API}/behaviours", params={"limit": 1, "q": "x", "cursor": first["next_cursor"] or "x"})).json()
    collected += changed["detail"]["next"]
    tools = {entry["tool"] for entry in collected}
    assert {"download_model", "list_models", "list_probe_datasets", "list_behaviour_versions",
            "new_behaviour_version", "list_behaviours"} <= tools, tools
    assert_next_calls_are_live(collected)
