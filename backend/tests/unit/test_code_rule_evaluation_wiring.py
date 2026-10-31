"""Operator code rules reach the runner, and their answers reach the verdicts (Feature 040 S4; FTDD Appendix C.3, C.9).

The runner is the STUB (`tests/support/stub_code_runner.py`: the runner's wire shape over an `httpx.MockTransport`,
recording every request; it never reads source). What is pinned, through the REAL routes and the REAL task:

* **Wiring (payload and call count).** A bulk rules evaluation sends each code-rule signal to the runner ONCE PER
  CHUNK of `code_rule_batch_texts` reduced texts — the view's texts exactly, the static params, the document's
  canonical bytes, the limits in force — and the single-text route once. Falling back to `UnavailableRunner` (the
  control) turns these red.
* **`spans: null`.** A fired code rule reports its count and `spans: None`, never `[]` or a guessed span.
* **Failure is not "did not fire"** (018 FR-018.21): a row whose code rule failed is unknown with the contract reason
  and counted (`by_reason`, the per-code-rule `reasons`).
* **Cache honesty** (040-A6): a run with a host fault (`rule_worker_unavailable`) caches NOTHING; a code-rule limit
  change moves the engine digest.
* **Review R1-F8.** Verdicts cached while no runner existed (pending, `runner_unavailable`) are never reused once
  one does: the engine digest names the runner.
* **The fresh approval read per chunk** (C.6): a revocation between two chunks stops the second (`rule_not_approved`).
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from src.core import runtime_settings
from src.core.config import settings
from src.models.rules_evaluation import ProbeRulesEvaluation, RuleVerdictCache
from src.rule_runner import contract as runner_contract
from tests.support import code_rule_world as cw
from tests.support import rule_set_fixtures as rf
from tests.support import rules_world as rw
from tests.support import stub_code_runner as stub
from tests.support.random_behaviour import rng_for

API = rw.API
N_EVAL_TEXTS = 2 * len(rw.EVAL_ROWS)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


async def _world(client, session, data_dir, request, **params):
    seed = request.node.nodeid
    key = rf.key(rng_for(seed + ":key"))
    rule_set, doc, sid, expr = cw.code_set(seed, key, **params)
    world = await rw.build(client, session, data_dir, seed, rule_set=rule_set, key=key)
    return world, doc, sid, expr


async def _body(client, out):
    read = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()
    return read["result"]


async def test_a_bulk_evaluation_sends_each_code_rule_once_per_chunk(client, async_session, data_dir, monkeypatch,
                                                                     request):
    runner = stub.install(monkeypatch, code_rule_batch_texts=10)
    world, doc, sid, expr = await _world(client, async_session, data_dir, request)
    await cw.approve(client, doc)
    before = len(runner.runs())
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert out["outcome"]["status"] == "completed", out
    calls = runner.runs()[before:]
    # 24 split rows in chunks of 10 → 3 requests, for the ONE code signal
    assert [len(c["body"]["calls"][0]["texts"]) for c in calls] == [10, 10, 4]
    sent = [t for c in calls for t in c["body"]["calls"][0]["texts"]]
    assert sent == rw.eval_split(world.group_column, ["a", "b"])["text"], "exactly the view's reduced texts, in order"
    first = calls[0]["body"]
    assert first["mode"] == "stored" and first["lane"] == "batch" and first["contract"] == runner_contract.FORMAT
    assert first["calls"][0]["function"] == doc["functions"][0]["name"]
    assert first["calls"][0]["params"] == {"needle": rf.UNMASKED}
    assert first["document"] == {"code_sha256": runner_contract.code_sha256(doc),
                                 "canonical_utf8": runner_contract.canonical_document_bytes(doc).decode("utf-8")}
    floor = runner_contract.floor()
    assert first["limits"] == {**{k: floor[k] for k in runner_contract.request_limits() if k in floor},
                               "wall_ms": settings.code_rule_wall_ms}
    result = await _body(client, out)
    view = result["views"][0]
    # The stub counts the unmasked value: U and B rows match — the masking set's MATCH (rules recall 3/6, FPR 1/6)
    assert view["detectors"]["rules_alone"]["positives"]["flagged"] == 3
    assert view["detectors"]["rules_alone"]["negatives"]["flagged"] == 1
    assert view["unknown"]["n"] == 0
    assert view["code_rules"] == [{
        "signal_id": sid, "code_sha256": runner_contract.code_sha256(doc), "runner_contract": runner_contract.FORMAT,
        "contract_version": runner_contract.identity()["contract_version"],
        "bundle_digest": runner_contract.identity()["bundle_digest"], "approval_row_id": view["code_rules"][0]["approval_row_id"],
        "rows_fired": 16, "reasons": {}}]
    assert isinstance(view["code_rules"][0]["approval_row_id"], int)
    assert result["runner"]["bundle_digest"] == runner_contract.identity()["bundle_digest"]
    # Control S4o survived without this: nothing moved, so the result is NOT stale (the read recomputes the SAME
    # runner-aware digest the request recorded).
    read = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()
    assert read["stale"] == {"stale": False, "moved": [], "next": []}


async def test_the_single_text_route_reports_a_count_and_spans_null(client, async_session, data_dir, monkeypatch,
                                                                    request):
    runner = stub.install(monkeypatch)
    world, doc, sid, expr = await _world(client, async_session, data_dir, request)
    await cw.approve(client, doc)
    before = len(runner.runs())
    text = f"a {rf.UNMASKED} and {rf.UNMASKED}"
    out = await client.post(f"{API}/rule-signals/evaluate", json={"behaviour_key": world.key, "text": text})
    assert out.status_code == 200, out.text
    body = out.json()
    calls = runner.runs()[before:]
    assert len(calls) == 1 and calls[0]["body"]["calls"][0]["texts"] == [text]
    assert calls[0]["body"]["lane"] == "interactive"
    assert body["rule_match"] is True and body["unknown"] is False
    assert body["fired"] == [{"signal_id": sid, "kind": "code_rule", "pending": False, "spans": None, "count": 2}]
    assert body["expressions"][expr] == {"fires": True, "pending": False, "count": 2, "form": "code_rule"}
    assert body["code"] == {sid: {"count": 2}}
    assert body["code_rules"][runner_contract.code_sha256(doc)]["state"] == "approved"


async def test_a_failed_code_rule_is_unknown_with_its_reason_never_did_not_fire(client, async_session, data_dir,
                                                                                monkeypatch, request):
    stub.install(monkeypatch)
    # fail on the masked value's rows; count the unmasked value elsewhere
    world, doc, sid, expr = await _world(client, async_session, data_dir, request, params={
        "needle": rf.UNMASKED, "fail_on": rf.MASKED, "reason": "rule_resource_limit"})
    await cw.approve(client, doc)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await _body(client, out))["views"][0]
    # evaluated rows with the masked value: p3, p6 (B), n2 → unknown, counted, reason recorded
    assert view["unknown"]["n"] == 3
    assert view["unknown"]["by_reason"]["rule_resource_limit"] == 3
    assert view["code_rules"][0]["reasons"] == {"rule_resource_limit": 3}  # split rows p3, p6, n2 (decoys are U)
    # p6 (B) is unknown now, so rules recall drops from 3/6 to 2/6 — the failure is not read as "no match" elsewhere
    assert view["detectors"]["rules_alone"]["positives"]["flagged"] == 2
    # a verdict failure (not a host fault) IS cached
    rows = (await async_session.execute(select(RuleVerdictCache))).scalars().all()
    assert len(rows) == 1


async def test_a_host_fault_is_never_cached(client, async_session, data_dir, monkeypatch, request):
    runner = stub.install(monkeypatch)
    world, doc, sid, expr = await _world(client, async_session, data_dir, request)
    await cw.approve(client, doc)
    runner.status = 503  # the runner is down for the whole evaluation
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await _body(client, out))["views"][0]
    assert view["unknown"]["by_reason"]["rule_worker_unavailable"] == len(rw.EVAL_ROWS)
    assert (await async_session.execute(select(RuleVerdictCache))).scalars().all() == []
    # the client retried (bounded) before giving up: 3 attempts for the one chunk
    assert len([r for r in runner.runs()]) >= 3


async def test_a_code_rule_limit_change_moves_the_engine_digest(client, async_session, data_dir, monkeypatch, request):
    stub.install(monkeypatch)
    world, doc, sid, expr = await _world(client, async_session, data_dir, request)
    await cw.approve(client, doc)
    first = await rw.run_one(client, monkeypatch, world, calibration=False)
    runtime_settings.apply({"code_rules_enabled": True,
                            "code_rule_call_fuel": runner_contract.floor()["call_fuel"] * 2})
    second = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert second["evaluation_id"] != first["evaluation_id"]
    assert second["engine_digest"] != first["engine_digest"]
    stale = (await client.get(f"{API}/rules-evaluations/{first['evaluation_id']}")).json()["stale"]
    assert stale["stale"] is True and stale["moved"][0]["input"] == "engine_digest"


async def test_review_r1_f8_pending_verdicts_are_not_reused_once_a_runner_exists(client, async_session, data_dir,
                                                                                 monkeypatch, request):
    runner = stub.install(monkeypatch, enabled=False)
    world, doc, sid, expr = await _world(client, async_session, data_dir, request)
    pending = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await _body(client, pending))["views"][0]
    assert view["unknown"]["by_reason"]["runner_unavailable"] == len(rw.EVAL_ROWS)
    assert len((await async_session.execute(select(RuleVerdictCache))).scalars().all()) == 1
    assert runner.runs() == []
    # the runner appears, the operator turns code rules on and approves the hash
    runner = stub.install(monkeypatch, enabled=True)
    await cw.approve(client, doc)
    live = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert live["engine_digest"] != pending["engine_digest"]
    result = (await _body(client, live))["views"][0]
    assert result["cached"] is False and result["unknown"]["n"] == 0
    assert len(runner.runs()) >= 1, "the verdicts were recomputed through the runner, not read from the pending cache"
    stale = (await client.get(f"{API}/rules-evaluations/{pending['evaluation_id']}")).json()["stale"]
    assert stale["stale"] is True


async def test_a_revocation_between_two_chunks_stops_the_second(client, async_session, data_dir, monkeypatch, request):
    from sqlalchemy.orm import Session

    from src.models.code_rule import CodeRuleApproval

    runner = stub.install(monkeypatch, code_rule_batch_texts=12)
    world, doc, sid, expr = await _world(client, async_session, data_dir, request)
    sha, _ = await cw.approve(client, doc)
    from sqlalchemy import create_engine

    from tests.support.test_databases import sync_test_url

    engine = create_engine(sync_test_url())
    start = len(runner.runs())

    def revoke_after_first(n):
        if n == start + 1:  # the first chunk's request is being answered: revoke before the second is asked
            with Session(bind=engine) as s:
                s.add(CodeRuleApproval(code_sha256=sha, action="revoke", reason="test", changed_by="operator",
                                       origin="operator"))
                s.commit()

    runner.before_run = revoke_after_first
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert len(runner.runs()) - start == 1, "the second chunk must not reach the runner after a revocation"
    view = (await _body(client, out))["views"][0]
    assert view["unknown"]["by_reason"]["rule_not_approved"] == len(rw.EVAL_ROWS) // 2
    assert (await async_session.execute(select(RuleVerdictCache))).scalars().all() == []
    row = (await async_session.execute(select(ProbeRulesEvaluation).where(
        ProbeRulesEvaluation.id == out["evaluation_id"]))).scalar_one()
    assert row.status == "completed"
    engine.dispose()


async def test_no_text_reaches_a_log_record_across_evaluate_a_chunk_and_a_runner_fault(client, async_session, data_dir,
                                                                                       monkeypatch, request, caplog):
    """FTDD C.9 "Logs" (with the stub; the real runner's own output is checked by `test_code_rule_hostile.py`)."""
    import logging
    import random
    import string

    caplog.set_level(logging.DEBUG)
    sentinel = "".join(random.SystemRandom().choice(string.ascii_letters) for _ in range(24))
    runner = stub.install(monkeypatch)
    seed = request.node.nodeid
    key = rf.key(rng_for(seed + ":key"))
    rule_set, doc, sid, expr = cw.code_set(seed, key)
    text = {k: f"{v} {sentinel}" for k, v in rw.TEXT.items()}
    world = await rw.build(client, async_session, data_dir, seed, rule_set=rule_set, key=key, text=text)
    await cw.approve(client, doc)
    single = await client.post(f"{API}/rule-signals/evaluate", json={"behaviour_key": key, "text": f"x {sentinel}"})
    assert single.status_code == 200
    await rw.run_one(client, monkeypatch, world, calibration=False)
    runner.status = 500
    fault = await client.post(f"{API}/rule-signals/evaluate", json={"behaviour_key": key, "text": f"y {sentinel}"})
    assert fault.json()["reason"] == "rule_worker_unavailable"
    from tests.unit.test_code_rule_hostile import logged as production_logged

    logged = production_logged(caplog)
    assert not any(sentinel[i:i + 12] in logged for i in range(len(sentinel) - 11))


async def test_lone_surrogates_are_replaced_before_the_runner_sees_a_text(client, async_session, data_dir, monkeypatch,
                                                                         request):
    """The caller's half of the contract's `text_rules`: U+D800 – U+DFFF become U+FFFD before encoding."""
    import json as _json

    runner = stub.install(monkeypatch)
    world, doc, sid, expr = await _world(client, async_session, data_dir, request)
    await cw.approve(client, doc)
    before = len(runner.runs())
    raw = '{"behaviour_key": "%s", "text": "a \\ud800 b"}' % world.key
    out = await client.post(f"{API}/rule-signals/evaluate", content=raw, headers={"content-type": "application/json"})
    assert out.status_code == 200, out.text
    assert runner.runs()[before]["body"]["calls"][0]["texts"] == ["a \ufffd b"]
    assert _json.loads(raw)["text"] == "a \ud800 b"


async def test_a_runner_serving_another_bundle_runs_nothing_here(client, async_session, data_dir, monkeypatch, request):
    """A runner whose bundle is not the vendored contract's must not produce verdicts under this miStudio's identity:
    every row is a host fault (unknown, never cached)."""
    runner = stub.install(monkeypatch)
    world, doc, sid, expr = await _world(client, async_session, data_dir, request)
    await cw.approve(client, doc)
    runner.bundle_digest = "0" * 64
    out = await client.post(f"{API}/rule-signals/evaluate", json={"behaviour_key": world.key, "text": rf.UNMASKED})
    assert out.json()["reason"] == "rule_worker_unavailable" and out.json()["rule_match"] is False


async def test_a_bulk_request_with_an_unapproved_hash_is_refused_before_anything_is_queued(client, async_session,
                                                                                          data_dir, monkeypatch,
                                                                                          request):
    runner = stub.install(monkeypatch)
    world, doc, sid, expr = await _world(client, async_session, data_dir, request)
    await cw.file_request(client, doc)  # stored and pending: not approved
    dispatched = []
    from src.services import rules_evaluation

    monkeypatch.setattr(rules_evaluation, "_dispatch", lambda rid: dispatched.append(rid) or "x")
    out = await client.post(f"{API}/probes/{world.probe.id}/rules-evaluations",
                            json={"view_ids": [world.eval_view.id], "calibration": False})
    assert out.status_code == 409 and out.json()["detail"]["code"] == "code_rule_not_approved"
    assert out.json()["detail"]["state"] == "pending"
    assert dispatched == [] and (await async_session.execute(select(ProbeRulesEvaluation))).scalars().all() == []
