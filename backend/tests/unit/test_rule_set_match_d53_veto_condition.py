"""D53 review (2026-11-08): an EXPLICIT match naming a veto condition is flagged on every surface, not only on the
compiled set — the single-text evaluate `notes` and the stored result `reasons`. Controls A4 / A5 (each line gated
to `match_source == "inferred"`) go red here."""
from __future__ import annotations

import pytest

from src.core.config import settings
from tests.support import rule_set_fixtures as rf
from tests.support import rules_world as rw

API = rw.API


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


async def test_evaluate_notes_an_explicit_match_naming_a_veto_condition(client, request):
    doc, names = rf.cued_set(request.node.nodeid, match=["cued", "masked"])
    answer = await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": "x"})
    assert answer.status_code == 200, answer.text
    out = answer.json()
    assert out["match_source"] == "explicit"
    assert "match_includes_veto_condition" in out.get("notes", []), out


async def test_stored_result_reasons_an_explicit_match_naming_a_veto_condition(client, async_session, data_dir,
                                                                               monkeypatch, request):
    doc = rf.masking_set(request.node.nodeid)
    when = doc["vetoes"][0]["when"]
    hit = doc["outputs"][0]["value"]["fired"]
    doc["match"] = {"version": 1, "any_of": [hit, when]}
    world = await rw.build(client, async_session, data_dir, request.node.nodeid, rule_set=doc)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    body = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]
    assert body["match_source"] == "explicit", body
    assert "match_includes_veto_condition" in body["reasons"], body["reasons"]
