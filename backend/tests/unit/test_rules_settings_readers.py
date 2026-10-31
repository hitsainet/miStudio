"""Each Feature 040 S3 runtime key reaches its reader (FTDD §5.6; `test_environment_readers.py` maps each key to the
test that drives it). An override is applied the way a refresh point applies it (`runtime_settings.apply`) and the
REAL route or task is driven; with the override absent the environment's value would answer differently.
"""
from __future__ import annotations

import pytest

from src.core import runtime_settings
from src.core.config import settings
from tests.support import rule_set_fixtures as rf
from tests.support import rules_world as rw

API = rw.API


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


@pytest.mark.parametrize("key,many", [("rule_set_max_signals", "signals"), ("rule_set_max_cases", "cases")])
async def test_the_rule_set_caps_are_the_runtime_settings(client, request, key, many):
    doc = rf.masking_set(request.node.nodeid)  # two signals, two cases
    assert (await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})).json()["valid"] is True
    runtime_settings.apply({key: 1})
    out = (await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})).json()
    assert (out["valid"], out["code"], out["setting"], out["value"]) == (False, "over_cap", key, 1)


async def test_the_row_cap_is_the_runtime_setting(client, async_session, fresh, request):
    world = await rw.build(client, async_session, fresh, request.node.nodeid)
    runtime_settings.apply({"rules_evaluation_max_rows": 10})
    response = await client.post(f"{API}/probes/{world.probe.id}/rules-evaluations",
                                 json={"view_ids": [world.eval_view.id], "calibration": False})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert (detail["setting"], detail["value"], detail["got"]) == ("rules_evaluation_max_rows", 10, 24)


async def test_the_unknown_share_and_body_cap_are_read_at_request(client, async_session, fresh, monkeypatch, request):
    world = await rw.build(client, async_session, fresh, request.node.nodeid)
    runtime_settings.apply({"rule_unknown_share_max": 0.07})
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    assert view["unknown"]["share_max"] == 0.07
    runtime_settings.apply({"rules_evaluation_max_bytes": 1024})
    failed = await rw.run_one(client, monkeypatch, world, calibration=True)
    read = (await client.get(f"{API}/rules-evaluations/{failed['evaluation_id']}")).json()
    assert read["status"] == "failed" and read["error"] == {
        "code": "over_cap", "setting": "rules_evaluation_max_bytes", "value": 1024, "got": read["error"]["got"]}
    assert read["error"]["got"] > 1024
