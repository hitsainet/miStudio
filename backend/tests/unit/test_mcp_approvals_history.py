"""The fields and the list the Approvals page's History reads (ENH-008 S4, S6; spec §3.2 steps 5 and 7).

* S4 — `POST /mcp/approvals` REFUSES every `millm_*` tool with 422: miLLM decides its own requests now.
* S6 — each row records who asked (`requested_by`, from `X-MiStudio-Agent`; "operator" with no header) and who
  decided (`decided_by`: "operator" on approve and deny, "system" when a pending request expired); `expires_at` is
  computed on read (created + TTL while pending; approved + TTL while a caller request is unspent; null for
  steering, for an approved record tool and for every decided row).
* History — `decided=true` lists every row that is no longer pending, the most recent decision first, with
  `offset` for "Show older" and `limit` up to 500. A decided request appears there; a pending one does not.

Mutation controls (0xcc/reviews/enh008_controls_2026-10-28.md): the History filter `!= pending` → `== pending`;
drop `decided_by=OPERATOR` from deny; drop `requested_by=` from create.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update

from src.core import runtime_settings
from src.models.agent_approval import AgentApprovalRequest

API = "/api/v1"
AGENT = {"X-MiStudio-Agent": "mcp"}
PROBE_RUN = {"tool_name": "start_probe_training_run", "payload": {"model_id": 7, "train_dataset_id": "pmd_1"}}
STEER = {"tool_name": "steer_compare", "payload": {"sae_id": "s", "prompt": "hi", "selected_features": []}}


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    from src.api.v1.endpoints import mcp_approvals

    monkeypatch.setattr(mcp_approvals, "emit_progress", lambda **_: None)
    runtime_settings.reset()


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.mark.parametrize("tool", ["millm_load_model", "millm_request_probe_limits", "millm_measure_probe_capacity",
                                  "millm_import_rule_set", "millm_unload_model", "millm_delete_model"])
async def test_S4_a_millm_tool_cannot_be_filed_here(client, async_session, tool):
    response = await client.post(f"{API}/mcp/approvals", json={"tool_name": tool, "payload": {"model_id": 1}},
                                 headers=AGENT)
    assert response.status_code == 422, response.text
    assert "not an approvable tool" in response.text


async def test_S6_who_asked_is_recorded_from_the_agent_header(client):
    filed = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()
    assert filed["requested_by"] == "mcp"
    by_ui = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN)).json()
    assert by_ui["requested_by"] == "operator"
    listed = {a["id"]: a for a in (await client.get(f"{API}/mcp/approvals")).json()["approvals"]}
    assert listed[filed["id"]]["requested_by"] == "mcp"


async def test_S6_a_long_agent_name_is_cut_to_the_column(client):
    filed = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers={"X-MiStudio-Agent": "a" * 200})).json()
    assert filed["requested_by"] == "a" * 64


async def test_S6_who_decided_approve_and_deny(client):
    approve_id = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()["id"]
    deny_id = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()["id"]
    assert (await client.get(f"{API}/mcp/approvals/{approve_id}")).json()["decided_by"] is None

    approved = (await client.post(f"{API}/mcp/approvals/{approve_id}/approve")).json()
    denied = (await client.post(f"{API}/mcp/approvals/{deny_id}/deny", json={"reason": "not now"})).json()
    assert approved["decided_by"] == "operator" and approved["status"] == "approved"
    assert denied["decided_by"] == "operator" and denied["reason"] == "not now"


async def test_S6_an_expiry_is_decided_by_the_system(client, async_session):
    rid = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()["id"]
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    await async_session.execute(update(AgentApprovalRequest).values(created_at=long_ago))
    await async_session.commit()
    row = (await client.get(f"{API}/mcp/approvals/{rid}")).json()
    assert row["status"] == "expired" and row["decided_by"] == "system"


async def test_S6_expires_at_is_computed_per_kind(client, async_session):
    ttl = runtime_settings.effective().mcp_approval_ttl_hours
    pending = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()
    assert _ts(pending["expires_at"]) - _ts(pending["created_at"]) == timedelta(hours=ttl)

    approved = (await client.post(f"{API}/mcp/approvals/{pending['id']}/approve")).json()
    assert _ts(approved["expires_at"]) - _ts(approved["resolved_at"]) == timedelta(hours=ttl), (
        "an approved caller request lapses TTL after it was approved, unless it is used")

    steering = (await client.post(f"{API}/mcp/approvals", json=STEER, headers=AGENT)).json()
    assert steering["expires_at"] is None, "steering never expires"

    denied_id = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()["id"]
    denied = (await client.post(f"{API}/mcp/approvals/{denied_id}/deny", json={})).json()
    assert denied["expires_at"] is None


def test_S6_an_approved_record_tool_never_lapses():
    from src.api.v1.endpoints.mcp_approvals import expires_at_of

    now = datetime.now(timezone.utc)
    row = AgentApprovalRequest(tool_name="code_rule_approve", status="approved", created_at=now, resolved_at=now)
    assert expires_at_of(row, 24) is None
    row.status = "pending"
    assert expires_at_of(row, 24) == now + timedelta(hours=24)


async def test_History_lists_decided_requests_newest_decision_first_and_never_a_pending_one(client, async_session):
    first = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()["id"]
    second = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()["id"]
    waiting = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()["id"]
    await client.post(f"{API}/mcp/approvals/{second}/deny", json={"reason": "x"})
    await client.post(f"{API}/mcp/approvals/{first}/approve")   # decided LAST

    history = (await client.get(f"{API}/mcp/approvals", params={"decided": "true"})).json()["approvals"]
    assert [a["id"] for a in history] == [first, second], "most recent decision first, pending excluded"
    assert waiting not in [a["id"] for a in history]
    assert history[1]["status"] == "denied" and history[1]["reason"] == "x"

    older = (await client.get(f"{API}/mcp/approvals", params={"decided": "true", "offset": 1, "limit": 1})).json()
    assert [a["id"] for a in older["approvals"]] == [second]

    everything = (await client.get(f"{API}/mcp/approvals")).json()["approvals"]
    assert waiting in [a["id"] for a in everything]


async def test_History_takes_up_to_500(client):
    assert (await client.get(f"{API}/mcp/approvals", params={"decided": "true", "limit": 500})).status_code == 200
    assert (await client.get(f"{API}/mcp/approvals", params={"limit": 501})).status_code == 422


async def test_a_deny_reason_longer_than_the_field_is_refused(client):
    rid = (await client.post(f"{API}/mcp/approvals", json=PROBE_RUN, headers=AGENT)).json()["id"]
    assert (await client.post(f"{API}/mcp/approvals/{rid}/deny", json={"reason": "r" * 2001})).status_code == 422
    assert (await client.get(f"{API}/mcp/approvals/{rid}")).json()["status"] == "pending"


HOSTILE = '<img src=x onerror=alert(1)><script>alert(2)</script>'


async def test_S10_hostile_text_round_trips_exactly_as_text(client):
    """The backend never interprets or "sanitises" request text: the page renders the exact characters as text
    (the render test in the frontend proves no element is created). A server that stripped tags would hide what an
    agent actually sent from the operator who approves it."""
    payload = {HOSTILE: HOSTILE, "model_id": 7, "train_dataset_id": HOSTILE}
    filed = (await client.post(f"{API}/mcp/approvals",
                               json={"tool_name": "start_probe_training_run", "payload": payload},
                               headers={"X-MiStudio-Agent": "<b>x</b>"})).json()
    denied = (await client.post(f"{API}/mcp/approvals/{filed['id']}/deny", json={"reason": HOSTILE})).json()
    history = (await client.get(f"{API}/mcp/approvals", params={"decided": "true"})).json()["approvals"]
    row = next(a for a in history if a["id"] == filed["id"])
    assert row["payload"] == payload and row["reason"] == HOSTILE and denied["reason"] == HOSTILE
    assert row["requested_by"] == "<b>x</b>"
