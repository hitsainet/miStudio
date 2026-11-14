"""ENH-006 D81 (operator, 2026-10-29): approve every waiting CODE-RULE request with one operator call.

`POST /api/v1/mcp/approvals/approve-batch` with `{"kind": "code_rules", "approval_ids": [...]}`:

* operator-only — an `X-MiStudio-Agent` caller is refused 403 like a single approve, and nothing is approved;
* the WHOLE batch is checked first: any id that is missing, not pending, or not a code-rule request (a steering or
  probe-run request, never mixed in) is named in a 409 and NOTHING is approved;
* each id is then approved through the same path as `POST /{id}/approve` — its own `approved` row with
  `decided_by='operator'`, its own `code_rule_approvals` row (origin 'operator', its request id) — plus a shared
  `batch_id`; one failing item (a tampered document) does not stop the others;
* `kind: "settings"` is refused 422: miStudio's queue has no settings-change request (D79);
* History lists each request with the batch id.

Negative controls (0xcc/reviews/approve_all_settings_controls_2026-10-29.md): remove the route decorator → the
OpenAPI and approve tests red; skip the pre-check → the mixed-kind and not-pending tests red; approve only the first
id → the every-id test red; drop the agent refusal → the agent test red.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from src.core import runtime_settings
from src.core.config import settings
from src.models.agent_approval import AgentApprovalRequest
from src.models.code_rule import CodeRuleApproval
from src.rule_runner import contract as runner_contract
from tests.support import code_rule_world as cw
from tests.support import rule_set_fixtures as rf
from tests.support import stub_code_runner as stub
from tests.support.random_behaviour import rng_for

API = "/api/v1"
AGENT = {"X-MiStudio-Agent": "mcp"}
BATCH = f"{API}/mcp/approvals/approve-batch"


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    runtime_settings.reset()
    yield
    runtime_settings.reset()


async def _file(client, request, n: int) -> list[tuple[str, str]]:
    """File `n` distinct code-rule requests; return `(request id, code sha256)` each."""
    out = []
    for i in range(n):
        seed = f"{request.node.nodeid}:{i}"
        _rule_set, doc, _sid, _expr = cw.code_set(seed, rf.key(rng_for(seed + ":key")))
        filed = await cw.file_request(client, doc, headers=AGENT)
        assert filed.status_code == 202, filed.text
        out.append((filed.json()["approval_request_id"], runner_contract.code_sha256(doc)))
    return out


async def _rows(session, ids):
    session.expire_all()
    rows = (await session.execute(select(AgentApprovalRequest).where(AgentApprovalRequest.id.in_(ids)))).scalars()
    return {r.id: r for r in rows}


async def _decisions(session):
    session.expire_all()
    return (await session.execute(select(CodeRuleApproval).order_by(CodeRuleApproval.id))).scalars().all()


async def _steering(session) -> str:
    row = AgentApprovalRequest(tool_name="steer_compare", payload={"sae_id": "s", "prompt": "p"}, status="pending",
                               requested_by="mcp")
    session.add(row)
    await session.commit()
    return row.id


def test_the_route_is_served():
    from src.main import app

    assert "post" in app.openapi()["paths"][f"{API}/mcp/approvals/approve-batch"]


async def test_every_id_is_approved_as_a_single_approve_would_with_one_batch_id(client, async_session, monkeypatch,
                                                                                 request):
    stub.install(monkeypatch)
    filed = await _file(client, request, 3)
    ids = [i for i, _ in filed]
    out = await client.post(BATCH, json={"kind": "code_rules", "approval_ids": ids})
    assert out.status_code == 200, out.text
    body = out.json()
    batch_id = body["batch_id"]
    assert batch_id and body["kind"] == "code_rules"
    assert [(r["id"], r["status"]) for r in body["results"]] == [(i, "approved") for i in ids]
    assert {r["approval"]["batch_id"] for r in body["results"]} == {batch_id}

    rows = await _rows(async_session, ids)
    for rid in ids:
        assert (rows[rid].status, rows[rid].decided_by, rows[rid].batch_id) == ("approved", "operator", batch_id)
        assert rows[rid].resolved_at is not None
    # The per-hash record a single approve writes — one per request, each naming its own request.
    decisions = await _decisions(async_session)
    assert sorted((d.code_sha256, d.action, d.origin, d.approval_request_id) for d in decisions) == sorted(
        (sha, "approve", "operator", rid) for rid, sha in filed)
    for _rid, sha in filed:
        assert (await client.get(f"{API}/code-rules/{sha}")).json()["state"] == "approved"

    # History shows each, with the shared batch id.
    history = (await client.get(f"{API}/mcp/approvals", params={"decided": "true"})).json()["approvals"]
    assert {h["id"]: h["batch_id"] for h in history if h["id"] in ids} == {i: batch_id for i in ids}


async def test_a_mixed_kind_batch_is_refused_whole_before_anything_is_approved(client, async_session, monkeypatch,
                                                                               request):
    stub.install(monkeypatch)
    filed = await _file(client, request, 2)
    steer = await _steering(async_session)
    ids = [filed[0][0], steer, filed[1][0]]
    out = await client.post(BATCH, json={"kind": "code_rules", "approval_ids": ids})
    assert out.status_code == 409, out.text
    detail = out.json()["detail"]
    assert detail["code"] == "approve_batch_refused"
    assert [r["id"] for r in detail["refused"]] == [steer]
    assert "steer_compare" in detail["refused"][0]["why"]
    rows = await _rows(async_session, ids)
    assert {r.status for r in rows.values()} == {"pending"}
    assert all(r.batch_id is None for r in rows.values())
    assert await _decisions(async_session) == []


async def test_a_decided_or_unknown_id_is_named_and_nothing_is_approved(client, async_session, monkeypatch, request):
    stub.install(monkeypatch)
    filed = await _file(client, request, 3)
    decided = filed[1][0]
    assert (await client.post(f"{API}/mcp/approvals/{decided}/deny", json={"reason": "no"})).status_code == 200
    ids = [filed[0][0], decided, "no-such-request", filed[2][0]]
    out = await client.post(BATCH, json={"kind": "code_rules", "approval_ids": ids})
    assert out.status_code == 409, out.text
    refused = {r["id"]: r["why"] for r in out.json()["detail"]["refused"]}
    assert set(refused) == {decided, "no-such-request"}
    assert "denied" in refused[decided] and "not found" in refused["no-such-request"]
    rows = await _rows(async_session, [filed[0][0], filed[2][0]])
    assert {r.status for r in rows.values()} == {"pending"}
    assert await _decisions(async_session) == []


async def test_one_failing_item_does_not_stop_the_others(client, async_session, monkeypatch, request):
    stub.install(monkeypatch)
    filed = await _file(client, request, 3)
    ids = [i for i, _ in filed]
    bad = await async_session.get(AgentApprovalRequest, ids[1])
    doc = bad.payload["document"]
    bad.payload = {**bad.payload, "document": {**doc, "source": doc["source"] + "\n# changed"}}
    await async_session.commit()

    out = await client.post(BATCH, json={"kind": "code_rules", "approval_ids": ids})
    assert out.status_code == 200, out.text
    results = {r["id"]: r for r in out.json()["results"]}
    assert [results[i]["status"] for i in ids] == ["approved", "failed", "approved"]
    assert "no longer hashes" in results[ids[1]]["error"]
    rows = await _rows(async_session, ids)
    assert [rows[i].status for i in ids] == ["approved", "pending", "approved"]
    assert rows[ids[1]].batch_id is None
    assert sorted(d.approval_request_id for d in await _decisions(async_session)) == sorted([ids[0], ids[2]])


async def test_an_agent_cannot_batch_approve(client, async_session, monkeypatch, request):
    stub.install(monkeypatch)
    ids = [i for i, _ in await _file(client, request, 2)]
    for header in ("mcp", ""):
        out = await client.post(BATCH, json={"kind": "code_rules", "approval_ids": ids},
                                headers={"X-MiStudio-Agent": header})
        assert out.status_code == 403 and out.json()["detail"]["code"] == "approval_operator_only"
    assert {r.status for r in (await _rows(async_session, ids)).values()} == {"pending"}
    assert await _decisions(async_session) == []


@pytest.mark.parametrize("body,code", [
    ({"kind": "settings", "approval_ids": ["x"]}, "approve_batch_no_settings_requests"),
    ({"kind": "everything", "approval_ids": ["x"]}, "approve_batch_unknown_kind"),
    ({"kind": "code_rules", "approval_ids": ["a", "a"]}, "approve_batch_duplicate_ids"),
])
async def test_the_kind_and_ids_are_checked(client, body, code):
    out = await client.post(BATCH, json=body)
    assert out.status_code == 422 and out.json()["detail"]["code"] == code, out.text


async def test_an_empty_batch_is_refused(client):
    out = await client.post(BATCH, json={"kind": "code_rules", "approval_ids": []})
    assert out.status_code == 422
