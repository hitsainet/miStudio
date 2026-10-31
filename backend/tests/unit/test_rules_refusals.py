"""Every S3 refusal is data with a live `next` where one exists (Feature 040 S3; FR-40; FTASKS 22.5).

Through the REAL routes. Codes covered here; the others are pinned where their behaviour is:
`rule_set_cases_failed`, `pattern_refused`, `pattern_too_large`, `over_cap`, `no_rule_set`
(`test_rule_sets_api.py`, `test_rules_settings_readers.py`); `calibration_positions_not_recorded`
(`test_rules_evaluation_task.py`); `rule_set_not_listed`, `column_absent`, `column_not_additive`, `split_changed`
(`test_view_rule_columns.py`); `rules_spend_budget` (`test_union_calibration.py`); `union_over_budget`,
`missing_evaluation` (rules) in a report (`test_success_report_rule_forms.py`).
"""
from __future__ import annotations

import pytest

from src.core.config import settings
from tests.support import rules_world as rw
from tests.support.next_calls import assert_next_calls_are_live

API = rw.API


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


async def _post(client, probe_id, **body):
    return await client.post(f"{API}/probes/{probe_id}/rules-evaluations", json=body)


async def test_an_unknown_probe(client):
    response = await _post(client, "pm_nope", view_ids=["pmd_x"])
    assert response.status_code == 404 and response.json()["detail"]["code"] == "probe_not_found"


async def test_a_view_the_probe_was_not_evaluated_on_names_the_evaluation(client, async_session, data_dir, request):
    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    response = await _post(client, world.probe.id, view_ids=[world.cal_view.id], calibration=False)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "missing_evaluation" and detail["view_id"] == world.cal_view.id
    assert detail["next"][0]["tool"] == "evaluate_probe_monitor"
    assert_next_calls_are_live(detail["next"])


async def test_calibration_without_a_calibration_view(client, async_session, data_dir, request):
    from sqlalchemy import update

    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorRun

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    await async_session.execute(update(ProbeMonitor).where(ProbeMonitor.id == world.probe.id).values(calibration_dataset_id=None))
    await async_session.execute(update(ProbeMonitorRun).where(ProbeMonitorRun.id == world.run.id).values(calibration_dataset_id=None))
    await async_session.commit()
    response = await _post(client, world.probe.id, view_ids=[world.eval_view.id], calibration=True)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "no_calibration_view"


async def test_a_probe_without_a_behaviour(client, async_session, data_dir, request):
    from sqlalchemy import update

    from src.models.probe_monitor import ProbeMonitor

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    await async_session.execute(update(ProbeMonitor).where(ProbeMonitor.id == world.probe.id).values(behaviour=None))
    await async_session.commit()
    response = await _post(client, world.probe.id, view_ids=[world.eval_view.id], calibration=False)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "no_behaviour"


async def test_an_unknown_scope_is_refused_naming_the_field(client, request):
    from tests.support import rule_set_fixtures as rf

    response = await client.post(f"{API}/rule-signals/evaluate", json={
        "rule_set": rf.masking_set(request.node.nodeid), "text": "x", "scope": "everything"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert (detail["code"], detail["field"], detail["rule"]) == ("invalid_field", "scope", "unknown_scope")


async def test_an_unknown_evaluation(client):
    for path in (f"{API}/rules-evaluations/pre_nope",):
        response = await client.get(path)
        assert response.status_code == 404 and response.json()["detail"]["code"] == "not_found"
    response = await client.post(f"{API}/rules-evaluations/pre_nope/cancel")
    assert response.status_code == 404


async def test_an_archived_view_is_refused(client, async_session, data_dir, request):
    """Review R1 (S3): a rules evaluation reads only unarchived views (`refuse_archived_inputs`), as every
    other evaluation path does."""
    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    archived = await client.post(f"{API}/datasets/{world.eval_view.id}/archive", json={})
    assert archived.status_code in (200, 202), archived.text
    response = await _post(client, world.probe.id, view_ids=[world.eval_view.id], calibration=False)
    assert response.status_code == 409, response.text
    assert world.eval_view.id in str(response.json()["detail"])
