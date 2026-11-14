"""Per-hash operator approval of operator code, and who may decide (Feature 040 S4; FTDD Appendix C.5, C.6, C.9;
FTASKS 25.7, 25.11).

Through the REAL routes, with the runner STUBBED at its HTTP boundary (`tests/support/stub_code_runner.py`):

* **Approval gate.** With a runner, an unapproved hash is refused `code_rule_not_approved` (409, `next` =
  request_code_rule_approval) on `evaluate_rule_signals`, `validate_rule_set` (an answer, `valid: false`) and a rules
  evaluation's request. With NO runner (or `code_rules_enabled` off) the same set is accepted and its expressions are
  PENDING (D41/D44) — never refused.
* **Requesting.** `POST /code-rules/{sha}/approval-requests` runs the document's own cases FIRST in the runner at the
  contract's floor limits, each on two fresh instances (`repeat: 2`), then files EXACTLY ONE `agent_approval_requests`
  row (tool `code_rule_approve`, payload asserted). A failing or non-deterministic case files nothing.
* **The RECORD branch.** The operator's approve writes exactly one `code_rule_approvals` row (`origin='operator'`,
  the request id) and marks the request approved in one transaction.
* **Operator only.** Revoke and re-approve with `X-MiStudio-Agent` → 403 (and approve/deny of the request: OSD-56,
  `test_mcp_approvals_operator_only.py`); no live MCP tool reaches revoke or re-approve (derived from the registry);
  an agent cannot switch `code_rules_enabled`; a `code_rule_*` limit below the contract floor is refused.
* **Documents.** A referenced hash with no document → `code_rule_document_missing`; an inline document no signal
  names → `code_rule_unreferenced`; a document hashing to another sha → `code_rule_digest_mismatch`.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from src.core import runtime_settings
from src.core.config import settings
from src.models.agent_approval import AgentApprovalRequest
from src.models.code_rule import CodeRule, CodeRuleApproval, CodeRuleIsAppendOnly
from src.rule_runner import contract as runner_contract
from tests.support import code_rule_world as cw
from tests.support import rule_set_fixtures as rf
from tests.support import stub_code_runner as stub
from tests.support.random_behaviour import rng_for

API = "/api/v1"
AGENT = {"X-MiStudio-Agent": "mcp"}


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    runtime_settings.reset()
    yield
    runtime_settings.reset()


def _set(request):
    seed = request.node.nodeid
    return cw.code_set(seed, rf.key(rng_for(seed + ":key")))


def test_routes_in_openapi():
    """Removing the router's include line turns this red (FTDD §10.2)."""
    from src.main import app

    paths = app.openapi()["paths"]
    for suffix, method in (("", "get"), ("/approval-requests", "post"), ("/revoke", "post"), ("/approve", "post")):
        assert method in paths["/api/v1/code-rules/{code_sha256}" + suffix], suffix


async def test_without_a_runner_a_code_set_is_accepted_and_pending(client, monkeypatch, request):
    runner = stub.install(monkeypatch, enabled=False)
    rule_set, doc, sid, expr = _set(request)
    out = await client.post(f"{API}/probe-monitors/rule-signals/evaluate",
                            json={"rule_set": rule_set, "code_rules": [{"document": doc}], "text": "x"})
    assert out.status_code == 200, out.text
    body = out.json()
    assert (body["unknown"], body["reason"], body["runner"]) == (True, "runner_unavailable", "unavailable")
    assert body["expressions"][expr]["fires"] is None and body["expressions"][expr]["pending"] is True
    assert runner.requests == []


async def test_with_a_runner_an_unapproved_hash_is_refused_everywhere(client, async_session, monkeypatch, request):
    runner = stub.install(monkeypatch)
    rule_set, doc, sid, expr = _set(request)
    sha = runner_contract.code_sha256(doc)
    body = {"rule_set": rule_set, "code_rules": [{"document": doc}], "text": "x"}
    out = await client.post(f"{API}/probe-monitors/rule-signals/evaluate", json=body)
    assert out.status_code == 409, out.text
    detail = out.json()["detail"]
    assert (detail["code"], detail["code_sha256"], detail["state"]) == ("code_rule_not_approved", sha, "unknown")
    assert [n["tool"] for n in detail["next"]] == ["request_code_rule_approval"]
    assert detail["next"][0]["arguments"] == {"code_sha256": sha}
    valid = (await client.post(f"{API}/probe-monitors/rule-sets/validate",
                               json={"rule_set": rule_set, "code_rules": [{"document": doc}]})).json()
    assert (valid["valid"], valid["code"]) == (False, "code_rule_not_approved")
    assert valid["next"][0]["tool"] == "request_code_rule_approval"
    assert runner.runs() == [], "nothing runs before the operator approves"
    # the inline document was stored (content-addressed) all the same
    stored = (await async_session.execute(select(CodeRule).where(CodeRule.code_sha256 == sha))).scalar_one()
    assert bytes(stored.canonical) == runner_contract.canonical_document_bytes(doc)


async def test_a_request_runs_the_cases_twice_at_the_floor_then_files_one_row(client, async_session, monkeypatch,
                                                                              request):
    runner = stub.install(monkeypatch)
    rule_set, doc, sid, expr = _set(request)
    sha = runner_contract.code_sha256(doc)
    filed = await cw.file_request(client, doc, headers=AGENT)
    assert filed.status_code == 202, filed.text
    assert filed.json()["status"] == "pending_approval"
    runs = runner.runs()
    assert len(runs) == 1
    sent = runs[0]["body"]
    assert sent["mode"] == "save_check" and sent["lane"] == "interactive"
    floor = runner_contract.floor()
    assert sent["limits"] == {**{k: floor[k] for k in runner_contract.request_limits() if k in floor}, "wall_ms": 10_000}
    assert sent["calls"] == [{"function": doc["functions"][0]["name"], "params": {"needle": "x"}, "texts": ["x x"],
                              "repeat": 2}]
    rows = (await async_session.execute(select(AgentApprovalRequest).where(
        AgentApprovalRequest.tool_name == "code_rule_approve"))).scalars().all()
    assert len(rows) == 1
    payload = rows[0].payload
    assert payload["code_sha256"] == sha and payload["document"] == doc
    assert payload["facts"]["case_results"] == [{"id": "c1", "expect": {"count": 2}, "got": {"count": 2},
                                                 "same_twice": True}]
    assert payload["facts"]["requested_by"] == "agent:mcp"
    # filing again while it waits returns the same request, files nothing
    again = await cw.file_request(client, doc, headers=AGENT)
    assert again.json()["approval_request_id"] == rows[0].id and again.json()["duplicate"] is True
    assert len((await async_session.execute(select(AgentApprovalRequest))).scalars().all()) == 1


@pytest.mark.parametrize("pairs,code", [([[{"count": 1}, {"count": 1}]], "code_rule_cases_failed"),
                                        ([[{"count": 2}, {"count": 3}]], "code_rule_nondeterministic")])
async def test_a_failing_or_nondeterministic_case_files_nothing(client, async_session, monkeypatch, request, pairs,
                                                                code):
    runner = stub.install(monkeypatch)
    runner.pairs = pairs
    rule_set, doc, sid, expr = _set(request)
    filed = await cw.file_request(client, doc, headers=AGENT)
    assert filed.status_code == 422 and filed.json()["detail"]["code"] == code
    assert (await async_session.execute(select(AgentApprovalRequest))).scalars().all() == []


async def test_the_operator_approve_records_one_operator_row_in_one_transaction(client, async_session, monkeypatch,
                                                                                request):
    runner = stub.install(monkeypatch)
    rule_set, doc, sid, expr = _set(request)
    sha = runner_contract.code_sha256(doc)
    filed = (await cw.file_request(client, doc, headers=AGENT)).json()
    done = await client.post(f"{API}/mcp/approvals/{filed['approval_request_id']}/approve")
    assert done.status_code == 200 and done.json()["status"] == "approved"
    rows = (await async_session.execute(select(CodeRuleApproval).where(CodeRuleApproval.code_sha256 == sha))
            ).scalars().all()
    assert [(r.action, r.origin, r.approval_request_id) for r in rows] == [
        ("approve", "operator", filed["approval_request_id"])]
    # now it runs, through the runner
    out = await client.post(f"{API}/probe-monitors/rule-signals/evaluate",
                            json={"rule_set": rule_set, "text": f"{rf.UNMASKED}"})
    assert out.status_code == 200 and out.json()["rule_match"] is True
    state = (await client.get(f"{API}/code-rules/{sha}")).json()
    assert state["state"] == "approved" and state["approval_row_id"] == rows[0].id


async def test_a_tampered_request_document_is_not_approved(client, async_session, monkeypatch, request):
    stub.install(monkeypatch)
    rule_set, doc, sid, expr = _set(request)
    filed = (await cw.file_request(client, doc, headers=AGENT)).json()
    row = (await async_session.execute(select(AgentApprovalRequest).where(
        AgentApprovalRequest.id == filed["approval_request_id"]))).scalar_one()
    row.payload = {**row.payload, "document": {**doc, "source": doc["source"] + "\n# changed"}}
    await async_session.commit()
    done = await client.post(f"{API}/mcp/approvals/{filed['approval_request_id']}/approve")
    assert done.status_code == 409 and done.json()["detail"]["code"] == "code_rule_digest_mismatch"
    assert (await async_session.execute(select(CodeRuleApproval))).scalars().all() == []


@pytest.mark.parametrize("action", ["revoke", "approve"])
async def test_revoke_and_reapprove_are_the_operators_alone(client, async_session, monkeypatch, request, action):
    stub.install(monkeypatch)
    rule_set, doc, sid, expr = _set(request)
    sha, _ = await cw.approve(client, doc)
    for header in ("mcp", ""):
        refused = await client.post(f"{API}/code-rules/{sha}/{action}", json={"reason": "r"},
                                    headers={"X-MiStudio-Agent": header})
        assert refused.status_code == 403 and refused.json()["detail"]["code"] == "code_rule_operator_only"
    count = len((await async_session.execute(select(CodeRuleApproval))).scalars().all())
    assert count == 1
    done = await client.post(f"{API}/code-rules/{sha}/{action}", json={"reason": "operator decision"})
    assert done.status_code == 200, done.text
    assert done.json()["state"] == ("revoked" if action == "revoke" else "approved")
    rows = (await async_session.execute(select(CodeRuleApproval).order_by(CodeRuleApproval.id))).scalars().all()
    assert [(r.action, r.origin, r.changed_by) for r in rows][-1] == (action, "operator", "operator")


async def test_a_revoked_hash_is_refused_at_the_next_use(client, monkeypatch, request):
    runner = stub.install(monkeypatch)
    rule_set, doc, sid, expr = _set(request)
    sha, _ = await cw.approve(client, doc)
    assert (await client.post(f"{API}/code-rules/{sha}/revoke", json={"reason": "no"})).status_code == 200
    before = len(runner.runs())
    out = await client.post(f"{API}/probe-monitors/rule-signals/evaluate", json={"rule_set": rule_set, "text": "x"})
    assert out.status_code == 409 and out.json()["detail"]["state"] == "revoked"
    assert len(runner.runs()) == before


def test_no_live_mcp_tool_reaches_revoke_or_reapprove(monkeypatch):
    from tests.support.mcp_route_map import called_routes, normalise, registered_tools

    monkeypatch.setenv("MCP_STEERING_APPROVAL", "true")
    routes = called_routes(registered_tools(monkeypatch), "mistudio", API)
    reached = {path for (_method, path) in routes}
    assert normalise(f"{API}/code-rules/{{sha}}/approval-requests") in reached, "the probe must see the request tool"
    for decision in ("revoke", "approve"):
        assert normalise(f"{API}/code-rules/{{sha}}/{decision}") not in reached
    assert not any(path.startswith(normalise(f"{API}/mcp/approvals/{{id}}/")) and path.endswith(("/approve", "/deny"))
                   for path in reached)


async def test_documents_must_exist_match_and_be_referenced(client, async_session, monkeypatch, request):
    stub.install(monkeypatch)
    rule_set, doc, sid, expr = _set(request)
    sha = runner_contract.code_sha256(doc)
    missing = await client.post(f"{API}/probe-monitors/rule-signals/evaluate", json={"rule_set": rule_set, "text": "x"})
    assert missing.status_code == 409
    assert (missing.json()["detail"]["code"], missing.json()["detail"]["code_sha256"]) == (
        "code_rule_document_missing", sha)
    other = stub.document("other_fn")
    unreferenced = await client.post(f"{API}/probe-monitors/rule-signals/evaluate",
                                     json={"rule_set": rule_set, "code_rules": [{"document": other}], "text": "x"})
    assert unreferenced.status_code == 422 and unreferenced.json()["detail"]["code"] == "code_rule_unreferenced"
    wrong = await client.post(f"{API}/code-rules/{sha}/approval-requests",
                              json={"document": other, "cases": [cw.case_for(other)]})
    assert wrong.status_code == 422 and wrong.json()["detail"]["code"] == "code_rule_digest_mismatch"
    invalid = await client.post(f"{API}/probe-monitors/rule-signals/evaluate",
                                json={"rule_set": rule_set, "code_rules": [{"document": {"source": "x"}}], "text": "x"})
    assert invalid.status_code == 422 and invalid.json()["detail"]["code"] == "code_rule_invalid"


async def test_stored_documents_and_decisions_are_append_only(client, async_session, monkeypatch, request):
    stub.install(monkeypatch)
    rule_set, doc, sid, expr = _set(request)
    sha, _ = await cw.approve(client, doc)
    row = (await async_session.execute(select(CodeRule).where(CodeRule.code_sha256 == sha))).scalar_one()
    row.created_by = "someone else"
    with pytest.raises(CodeRuleIsAppendOnly):
        await async_session.commit()
    await async_session.rollback()


async def test_an_agent_cannot_switch_code_rules_on_and_a_limit_below_the_floor_is_refused(client):
    refused = await client.put(f"{API}/environment/code_rules_enabled", json={"value": True}, headers=AGENT)
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "environment_operator_only"
    also = await client.delete(f"{API}/environment/code_rules_enabled", headers=AGENT)
    assert also.status_code == 403
    floor = runner_contract.floor()["call_fuel"]
    low = await client.put(f"{API}/environment/code_rule_call_fuel", json={"value": floor - 1})
    assert low.status_code == 422 and "floor" in low.json()["detail"]
    raised = await client.put(f"{API}/environment/code_rule_call_fuel", json={"value": floor * 2})
    assert raised.status_code == 200, raised.text


async def test_the_switch_and_the_limits_reach_the_runner(client, monkeypatch, request):
    """Reader wiring (`test_environment_readers.py`): `code_rules_enabled` gates the runner; each limit is sent."""
    runner = stub.install(monkeypatch)
    rule_set, doc, sid, expr = _set(request)
    await cw.approve(client, doc)
    raised = {"code_rules_enabled": True, "code_rule_call_fuel": runner_contract.floor()["call_fuel"] * 2,
              "code_rule_load_fuel": runner_contract.floor()["load_fuel"] * 2,
              "code_rule_prelude_fuel": runner_contract.floor()["prelude_fuel"] * 2,
              "code_rule_memory_bytes": runner_contract.floor()["memory_bytes"] * 2,
              "code_rule_input_code_points": runner_contract.floor()["input_code_points"] * 2,
              "code_rule_output_bytes": runner_contract.floor()["output_bytes"] * 2, "code_rule_wall_ms": 5000}
    runtime_settings.apply(raised)
    before = len(runner.runs())
    await client.post(f"{API}/probe-monitors/rule-signals/evaluate", json={"rule_set": rule_set, "text": "x"})
    sent = runner.runs()[before]["body"]["limits"]
    assert sent == {"call_fuel": raised["code_rule_call_fuel"], "load_fuel": raised["code_rule_load_fuel"],
                    "prelude_fuel": raised["code_rule_prelude_fuel"], "memory_bytes": raised["code_rule_memory_bytes"],
                    "input_code_points": raised["code_rule_input_code_points"],
                    "output_bytes": raised["code_rule_output_bytes"], "wall_ms": 5000}
    runtime_settings.apply({**raised, "code_rules_enabled": False})
    before = len(runner.runs())
    off = (await client.post(f"{API}/probe-monitors/rule-signals/evaluate",
                             json={"rule_set": rule_set, "text": "x"})).json()
    assert off["reason"] == "runner_unavailable" and len(runner.runs()) == before
