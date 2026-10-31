"""Family scoring, the cap move and the score-result fix (Feature 041 FR-22; T18, T19, PT-8; FTASKS 12.x).

The model forward is not run here: `score_one` is spied (payload and count) where the claim is about
WIRING — one load per input, every member through `score_one(…, loaded=…)` with the SAME loaded triple —
and the candidate value is checked against the report's recorded parameters applied by hand.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.core import runtime_settings
from src.core.config import settings
from src.services import probe_monitor_run
from tests.support import blend_world as bw
from tests.support import family_fixtures as ff
from tests.support.random_behaviour import rng_for

API = "/api/v1/probe-monitors"
BACKEND = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


@pytest.fixture
def sync_db(async_engine):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", "")))
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


# ── the score_one seam (12.1) ────────────────────────────────────────────────────────────────


async def test_loaded_skips_the_load(client, async_session, request, sync_db, monkeypatch):
    model = await ff.model(async_session)
    probe = await ff.member(async_session, model, await ff.source(async_session), key=None)
    await async_session.commit()
    loads = []

    def load(run):
        loads.append(run.id)
        raise RuntimeError("the model was loaded")

    monkeypatch.setattr(probe_monitor_run, "_load_model_for_run", load)
    monkeypatch.setattr(probe_monitor_run, "probe_render", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("after load")))
    with pytest.raises(RuntimeError, match="the model was loaded"):
        probe_monitor_run.score_one(sync_db, probe.id, text="x")
    assert len(loads) == 1
    with pytest.raises(RuntimeError, match="after load"):
        probe_monitor_run.score_one(sync_db, probe.id, text="x", loaded=(object(), object(), "Llama"))
    assert len(loads) == 1, "a given `loaded` must skip the load"


def test_no_existing_caller_passes_loaded():
    """Every `score_one(` call in `src/` other than family scoring is byte-for-byte the old call (AST)."""
    for path in (BACKEND / "src").rglob("*.py"):
        if path.name == "probe_family_scoring.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "score_one":
                assert "loaded" not in {k.arg for k in node.keywords}, path


# ── score_family (12.2) ──────────────────────────────────────────────────────────────────────


def fake_score_one(calls):
    def score_one(db, probe_id, *, text=None, messages=None, loaded=None):
        calls.append((probe_id, text, id(loaded)))
        seed = sum(map(ord, probe_id)) % 97
        return {"probe_id": probe_id, "aggregate": seed / 10.0, "n_scored": 7, "fires": seed % 2 == 0,
                "threshold": 0.5, "tokens": ["a"], "token_scores": [0.1]}

    return score_one


async def test_member_scores_equal_score_one_and_load_once(client, async_session, request, sync_db, monkeypatch):
    from src.services.probe_family_scoring import score_family

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    calls, loads = [], []
    loaded = (object(), object(), "LlamaForCausalLM")
    monkeypatch.setattr(probe_monitor_run, "score_one", fake_score_one(calls))
    out = score_family(sync_db, [m.id for m in world.members], text="hello",
                       loader=lambda run: loads.append(run.id) or loaded)
    assert len(loads) == 1                                                   # ONE load per input
    assert [c[0] for c in calls] == [m.id for m in world.members]
    assert {c[2] for c in calls} == {id(loaded)}                             # every member on that load
    alone = []
    expected = [{k: v for k, v in fake_score_one(alone)(None, m.id, text="hello").items()
                 if k not in ("tokens", "token_scores")} for m in world.members]
    assert out["members"] == expected
    assert out["rules"] == {"available": False, "reason": "no_rule_set"}      # the world's version names none


async def test_candidate_value_equals_applied_params(client, async_session, request, sync_db, monkeypatch):
    from src.models.probe_blend import ProbeBlendReport
    from src.services.probe_family_scoring import score_family

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    report = json.loads((await client.get(f"{API}/blend-reports/{report_id}/report.json")).content)
    row = sync_db.get(ProbeBlendReport, report_id)
    monkeypatch.setattr(probe_monitor_run, "score_one", fake_score_one([]))
    out = score_family(sync_db, [m.id for m in world.members], text="t", report=report, resolved=row.resolved,
                       candidates=["stack", "max", "k_of_n:2"], loader=lambda run: (None, None, "x"))
    by = {c["name"]: c for c in out["candidates"]}
    scores = np.array([sum(map(ord, m.id)) % 97 / 10.0 for m in world.members])
    stack = next(c for c in report["candidates"] if c["name"] == "stack")
    z = (scores - np.array(stack["params"]["mean"])) / np.array(stack["params"]["std"])
    value = float(z @ np.array(stack["params"]["weights"]) + stack["params"]["intercept"])
    assert by["stack"]["value"] == pytest.approx(value, rel=1e-9)
    assert by["stack"]["fires"] == (value >= stack["calibration"]["threshold"])
    assert by["max"]["value"] == pytest.approx(float(z.max()), rel=1e-9)
    fired = int((scores >= np.array([m.threshold for m in world.members])).sum())
    assert by["k_of_n:2"]["value"] == fired and by["k_of_n:2"]["fires"] == (fired >= 2)


# ── routes (12.3 – 12.6) ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def queued(monkeypatch):
    sent = []

    class Task:
        def __init__(self, i):
            self.id = f"t{i}"

    def gpu_delay(task, gpu):
        return lambda **kw: sent.append((task.name, gpu, kw)) or Task(len(sent))

    monkeypatch.setattr("src.services.gpu_dispatch.gpu_delay", gpu_delay)
    monkeypatch.setattr("src.api.v1.endpoints.probe_monitors.gpu_delay", gpu_delay)
    return sent


async def test_family_score_dispatches_one_task_per_input(client, async_session, request, queued):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    response = await client.post(f"{API}/behaviours/{world.key}/family/score-batch",
                                 json={"model": world.model.id, "inputs": [{"text": "a"}, {"messages": [
                                     {"role": "user", "content": "b"}]}]})
    assert response.status_code == 202, response.text
    ids = [m.id for m in world.members]
    assert queued == [
        ("src.workers.probe_monitor_tasks.score_probe_family", "auto",
         {"behaviour_key": world.key, "member_ids": ids, "text": "a", "messages": None, "blend_report_id": None,
          "candidates": None, "include_traces": False}),
        ("src.workers.probe_monitor_tasks.score_probe_family", "auto",
         {"behaviour_key": world.key, "member_ids": ids, "text": None,
          "messages": [{"role": "user", "content": "b"}], "blend_report_id": None, "candidates": None,
          "include_traces": False}),
    ]
    assert [t["task_id"] for t in response.json()["tasks"]] == ["t1", "t2"]


async def test_3x6_over_cap_16_names_18_and_16(client, async_session, request, queued):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    runtime_settings.apply({"probe_score_batch_cap": 16})
    response = await client.post(f"{API}/behaviours/{world.key}/family/score-batch",
                                 json={"model": world.model.id, "inputs": [{"text": str(i)} for i in range(6)]})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert (detail["members"], detail["inputs"], detail["product"], detail["setting"], detail["value"]) == (
        3, 6, 18, "probe_score_batch_cap", 16)
    assert queued == []


async def test_single_and_family_caps_read_one_setting(client, async_session, request, queued):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    probe = world.members[0].id
    runtime_settings.apply({"probe_score_batch_cap": 3})
    four = {"inputs": [{"text": str(i)} for i in range(4)]}
    single = await client.post(f"{API}/probes/{probe}/score-batch", json=four)
    assert single.status_code == 422 and single.json()["detail"]["setting"] == "probe_score_batch_cap"
    family = await client.post(f"{API}/behaviours/{world.key}/family/score-batch",
                               json={"model": world.model.id, "inputs": [{"text": "x"}, {"text": "y"}]})
    assert family.status_code == 422                                          # 3 x 2 = 6 > 3
    runtime_settings.apply({"probe_score_batch_cap": 6})
    assert (await client.post(f"{API}/probes/{probe}/score-batch", json=four)).status_code == 202
    assert (await client.post(f"{API}/behaviours/{world.key}/family/score-batch",
                              json={"model": world.model.id, "inputs": [{"text": "x"}, {"text": "y"}]})).status_code == 202
    single_calls = [q for q in queued if q[0].endswith("score_probe_monitor")]
    assert [q[2] for q in single_calls] == [{"probe_id": probe, "text": str(i), "messages": None} for i in range(4)]


def test_mcp_literal_gone():
    """T19: the MCP batch tool carries no integer cap: no `MAX_SCORE_BATCH`, no comparison with a literal."""
    from src.mcp_server.tools import probe_monitors

    assert not hasattr(probe_monitors, "MAX_SCORE_BATCH")
    tree = ast.parse(inspect.getsource(probe_monitors))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "score_probe_monitor_batch")
    compares = [n for n in ast.walk(fn) if isinstance(n, ast.Compare)
                and any(isinstance(c, ast.Constant) and isinstance(c.value, int) and not isinstance(c.value, bool)
                        for c in [n.left, *n.comparators])]
    assert compares == []


class _Result:
    def __init__(self, state, info):
        self.state, self.info = state, info


async def test_score_result_of_another_probe_is_404(client, async_session, request, monkeypatch):
    from src.core.celery_app import celery_app

    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    mine = await ff.member(async_session, model, ds, key=None)
    await async_session.commit()
    monkeypatch.setattr(celery_app, "AsyncResult", lambda tid: _Result("SUCCESS", {"probe_id": "pm_someone_else"}))
    response = await client.get(f"{API}/probes/{mine.id}/score/t1")
    assert response.status_code == 404 and response.json()["detail"]["code"] == "not_this_probes_task"
    monkeypatch.setattr(celery_app, "AsyncResult", lambda tid: _Result("SUCCESS", {"probe_id": mine.id, "aggregate": 1}))
    assert (await client.get(f"{API}/probes/{mine.id}/score/t1")).json()["result"]["aggregate"] == 1


async def test_family_score_result_of_another_behaviour_is_404(client, monkeypatch, request):
    from src.core.celery_app import celery_app

    key = await ff.catalogue(client, rng_for(request.node.nodeid))
    monkeypatch.setattr(celery_app, "AsyncResult", lambda tid: _Result("SUCCESS", {"behaviour_key": "other-key"}))
    assert (await client.get(f"{API}/behaviours/{key}/family/score/t1")).status_code == 404
    monkeypatch.setattr(celery_app, "AsyncResult", lambda tid: _Result("SUCCESS", {"behaviour_key": key, "members": []}))
    assert (await client.get(f"{API}/behaviours/{key}/family/score/t1")).json()["status"] == "SUCCESS"


def test_the_family_tasks_are_routed():
    from src.core.celery_app import celery_app
    from src.services.gpu_dispatch import gpu_job_spec
    from src.workers import probe_monitor_tasks

    routes = celery_app.conf.task_routes
    assert routes["src.workers.probe_monitor_tasks.score_probe_family"] == {"queue": "extraction"}
    assert routes["src.workers.probe_monitor_tasks.publish_probe_family"] == {"queue": "processing"}
    assert gpu_job_spec(probe_monitor_tasks.score_probe_family) is not None
    assert gpu_job_spec(probe_monitor_tasks.publish_probe_family) is None


async def test_the_task_scores_with_the_reports_resolved_decisions(client, async_session, request, monkeypatch):
    """The task body hands `score_family` the report and its `resolved` (payload, count 1)."""
    from src.services import probe_family_scoring
    from src.workers import probe_monitor_tasks

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    seen = []
    monkeypatch.setattr(probe_family_scoring, "score_family",
                        lambda db, ids, **kw: seen.append((ids, kw)) or {"members": [], "candidates": []})
    out = await asyncio.to_thread(probe_monitor_tasks.score_probe_family.run, world.key, [m.id for m in world.members],
                                  text="x", blend_report_id=report_id, candidates=["stack"])
    assert out["behaviour_key"] == world.key and len(seen) == 1
    ids, kw = seen[0]
    assert ids == [m.id for m in world.members] and kw["candidates"] == ["stack"] and kw["text"] == "x"
    assert kw["report"]["report_id"] == report_id and kw["resolved"]["members"][0]["probe_id"] == world.members[0].id


# ── the rule-set verdict through 040's single-text evaluator (12.2, rules part) ─────────────────────────


async def rules_family(client, async_session, request):
    from tests.support import rule_set_fixtures as rf

    seed = request.node.nodeid
    return await bw.build(client, async_session, seed=seed, rule_set=lambda key: rf.masking_set(seed, key))


async def test_rule_verdict_comes_from_040s_single_text_evaluator(client, async_session, request, sync_db,
                                                                   monkeypatch):
    """`rules` is 040's verdict on THIS input, never a constant. Control: replace the evaluator call in
    `score_family` with a constant → the unmasked input reads no match → red."""
    from src.services import rules_evaluation
    from src.services.probe_family_scoring import score_family
    from tests.support import rule_set_fixtures as rf

    world = await rules_family(client, async_session, request)
    monkeypatch.setattr(probe_monitor_run, "score_one", fake_score_one([]))
    calls = []
    real = rules_evaluation.evaluate_input

    def spy(compiled, limits, **kw):
        calls.append(kw["text"])
        return real(compiled, limits, **kw)

    monkeypatch.setattr(rules_evaluation, "evaluate_input", spy)
    ids = [m.id for m in world.members]
    hit = score_family(sync_db, ids, text=f"id {rf.UNMASKED} end", loader=lambda run: (None, None, "x"),
                       behaviour_key=world.key)["rules"]
    masked = score_family(sync_db, ids, text=f"id {rf.MASKED} end", loader=lambda run: (None, None, "x"),
                          behaviour_key=world.key)["rules"]
    plain = score_family(sync_db, ids, text="nothing here", loader=lambda run: (None, None, "x"),
                         behaviour_key=world.key)["rules"]
    assert calls == [f"id {rf.UNMASKED} end", f"id {rf.MASKED} end", "nothing here"]       # once per input
    assert (hit["available"], hit["rule_match"], hit["rule_veto"], hit["unknown"]) == (True, True, False, False)
    assert (masked["rule_match"], masked["rule_veto"]) == (False, True)
    assert (plain["rule_match"], plain["rule_veto"]) == (False, False)
    assert hit["rule_set_digest"] and hit["engine_digest"]


async def test_rules_or_candidate_fires_on_a_rule_match_at_the_union_bar(client, async_session, request, sync_db,
                                                                         monkeypatch):
    from src.models.probe_blend import ProbeBlendReport
    from src.services.probe_family_scoring import score_family
    from tests.support import rule_set_fixtures as rf

    world = await rules_family(client, async_session, request)
    await bw.seed_rule_verdicts(async_session, world, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world, candidates=["stack", "rules_or_stack"])
    report = json.loads((await client.get(f"{API}/blend-reports/{report_id}/report.json")).content)
    row = sync_db.get(ProbeBlendReport, report_id)

    def low(db, probe_id, *, text=None, messages=None, loaded=None):      # every member scores far below its bar
        return {"probe_id": probe_id, "aggregate": -5.0, "n_scored": 7, "fires": False, "threshold": 0.8}

    monkeypatch.setattr(probe_monitor_run, "score_one", low)
    union = next(c for c in report["candidates"] if c["name"] == "rules_or_stack")
    for text, ruled in ((f"id {rf.UNMASKED} end", True), ("nothing here", False)):
        out = score_family(sync_db, [m.id for m in world.members], text=text, report=report, resolved=row.resolved,
                           candidates=["stack", "rules_or_stack"], loader=lambda run: (None, None, "x"),
                           behaviour_key=world.key)
        by = {c["name"]: c for c in out["candidates"]}
        assert by["rules_or_stack"]["value"] == pytest.approx(by["stack"]["value"])      # the base, applied as recorded
        at_union_bar = by["stack"]["value"] >= union["calibration"]["threshold"]
        assert by["rules_or_stack"]["fires"] == (ruled or at_union_bar)
        assert not at_union_bar, "the fixture must leave the stack below the union bar, so only the rule can fire"
