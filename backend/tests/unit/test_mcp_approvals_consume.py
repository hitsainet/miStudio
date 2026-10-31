"""The approvals queue learns caller-executed tools: approve, consume once, expire (Feature 034).

Routes under test: `POST /mcp/approvals` (now any `APPROVABLE_TOOLS` key), `/approve` (a
caller-executed tool is only MARKED approved — no self-call), `/consume` (new: one conditional
UPDATE), and lazy expiry on every route.

Runs against PostgreSQL (`DATABASE_URL`, the per-agent test database) with ONE SESSION PER REQUEST,
as production has — the conftest `client` fixture shares one session across requests, which would
hide both the identity-map staleness `populate_existing` guards and the concurrency of two consumes.
SQLite is not exercised: the backend venv has no async SQLite driver and production is PostgreSQL
(recorded deviation from FTASKS 2.4's "SQLite and PostgreSQL").

Controls (recorded 2026-10-16, tasks 2.4/2.5; each red, each restore verified):
  * drop `model.status == ApprovalStatus.APPROVED.value` from the consume UPDATE →
    `test_a_consume_succeeds_ONCE` and `test_two_CONCURRENT_consumes_get_one_success` red;
  * drop `if request.payload != approval.payload` →
    `test_a_different_payload_is_refused_and_spends_nothing` red;
  * delete `await expire_stale(db)` from `consume_request` →
    `test_every_route_expires_first` (AST) and
    `test_an_approved_but_unconsumed_load_EXPIRES` red;
  * drop `model.tool_name.not_in(sorted(STEERING_TOOLS))` from the pending statement →
    `test_a_STEERING_approval_never_expires` red.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.v1.endpoints import mcp_approvals
from src.core.deps import get_db
from src.main import app
from src.models.agent_approval import AgentApprovalRequest

#: ENH-008: miLLM decides its own model loads now, so the canonical CALLER-executed request here is a probe run.
LOAD = {"tool_name": "start_probe_training_run", "payload": {"model_id": 7, "train_dataset_id": "pmd_1"}}
STEER = {"tool_name": "steer_compare",
         "payload": {"sae_id": "s", "prompt": "hi", "selected_features": []}}


@pytest.fixture
def events(monkeypatch):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        mcp_approvals, "emit_progress",
        lambda channel, event, data: seen.append((event, data)),
    )
    return seen


@pytest.fixture
def no_self_call(monkeypatch):
    """A caller-executed approve must issue NO HTTP request (the backend cannot load)."""
    class _Refuse:
        def __init__(self, *a, **k):
            raise AssertionError("approve issued an HTTP self-call for a caller-executed tool")

    monkeypatch.setattr(mcp_approvals.httpx, "AsyncClient", _Refuse)


@pytest_asyncio.fixture
async def api(async_engine):
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def per_request_session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db] = per_request_session
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test/api/v1") as client:
        yield client, maker
    app.dependency_overrides.clear()


async def _create(client, body=LOAD) -> str:
    response = await client.post("/mcp/approvals", json=body)
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def _backdate(maker, request_id: str, column: str, hours: float) -> None:
    async with maker() as session:
        await session.execute(
            update(AgentApprovalRequest)
            .where(AgentApprovalRequest.id == request_id)
            .values({column: datetime.now(timezone.utc) - timedelta(hours=hours)})
        )
        await session.commit()


class TestCreateAndApprove:
    @pytest.mark.asyncio
    async def test_a_load_request_is_accepted_and_summarised(self, api, events):
        client, _ = api
        await _create(client)
        created = [d for e, d in events if e == "approval:created"]
        assert created[-1]["summary"] == (
            "Train probes on model 7 from probe dataset pmd_1 — a GPU run (minutes to hours)")

    @pytest.mark.asyncio
    async def test_an_unknown_tool_is_refused(self, api, events):
        client, _ = api
        response = await client.post("/mcp/approvals",
                                     json={"tool_name": "delete_everything", "payload": {}})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_approving_a_load_issues_NO_self_call(self, api, events, no_self_call):
        client, _ = api
        request_id = await _create(client)
        response = await client.post(f"/mcp/approvals/{request_id}/approve")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "approved"
        assert response.json()["consumed_at"] is None
        assert ("approval:resolved", {"request_id": request_id, "status": "approved"}) in events


class TestConsume:
    async def _approved(self, client) -> str:
        request_id = await _create(client)
        assert (await client.post(f"/mcp/approvals/{request_id}/approve")).status_code == 200
        return request_id

    @pytest.mark.asyncio
    async def test_a_consume_succeeds_ONCE(self, api, events, no_self_call):
        client, _ = api
        request_id = await self._approved(client)
        first = await client.post(f"/mcp/approvals/{request_id}/consume", json=LOAD)
        assert first.status_code == 200, first.text
        assert first.json()["status"] == "consumed" and first.json()["consumed_at"]
        second = await client.post(f"/mcp/approvals/{request_id}/consume", json=LOAD)
        assert second.status_code == 409
        assert second.json()["detail"]["status"] == "consumed"

    @pytest.mark.asyncio
    async def test_two_CONCURRENT_consumes_get_one_success(self, api, events, no_self_call):
        client, _ = api
        request_id = await self._approved(client)
        responses = await asyncio.gather(*(
            client.post(f"/mcp/approvals/{request_id}/consume", json=LOAD) for _ in range(2)
        ))
        assert sorted(r.status_code for r in responses) == [200, 409]

    @pytest.mark.asyncio
    async def test_a_different_payload_is_refused_and_spends_nothing(
        self, api, events, no_self_call
    ):
        client, _ = api
        request_id = await self._approved(client)
        other = {"tool_name": "start_probe_training_run", "payload": {"model_id": 8, "train_dataset_id": "pmd_1"}}
        refused = await client.post(f"/mcp/approvals/{request_id}/consume", json=other)
        assert refused.status_code == 422
        still = await client.get(f"/mcp/approvals/{request_id}")
        assert still.json()["status"] == "approved"

    @pytest.mark.asyncio
    async def test_a_STEERING_approval_cannot_be_consumed(self, api, events):
        client, _ = api
        request_id = await _create(client, STEER)
        refused = await client.post(f"/mcp/approvals/{request_id}/consume", json=STEER)
        assert refused.status_code == 422

    @pytest.mark.asyncio
    async def test_a_PENDING_approval_cannot_be_consumed(self, api, events):
        client, _ = api
        request_id = await _create(client)
        refused = await client.post(f"/mcp/approvals/{request_id}/consume", json=LOAD)
        assert refused.status_code == 409 and refused.json()["detail"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_a_DENIED_approval_cannot_be_consumed(self, api, events):
        client, _ = api
        request_id = await _create(client)
        await client.post(f"/mcp/approvals/{request_id}/deny", json={"reason": "no"})
        refused = await client.post(f"/mcp/approvals/{request_id}/consume", json=LOAD)
        assert refused.status_code == 409
        assert refused.json()["detail"] == {"status": "denied", "reason": "no"}


class TestExpiry:
    @pytest.mark.asyncio
    async def test_a_stale_PENDING_load_reads_expired(self, api, events):
        client, maker = api
        request_id = await _create(client)
        await _backdate(maker, request_id, "created_at", 25)
        got = await client.get(f"/mcp/approvals/{request_id}")
        assert got.json()["status"] == "expired"
        assert ("approval:resolved", {"request_id": request_id, "status": "expired"}) in events
        listing = await client.get("/mcp/approvals", params={"status": "pending"})
        assert request_id not in {a["id"] for a in listing.json()["approvals"]}

    @pytest.mark.asyncio
    async def test_an_expired_request_cannot_be_approved(self, api, events, no_self_call):
        client, maker = api
        request_id = await _create(client)
        await _backdate(maker, request_id, "created_at", 25)
        response = await client.post(f"/mcp/approvals/{request_id}/approve")
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_an_approved_but_unconsumed_load_EXPIRES(self, api, events, no_self_call):
        client, maker = api
        request_id = await _create(client)
        await client.post(f"/mcp/approvals/{request_id}/approve")
        await _backdate(maker, request_id, "resolved_at", 25)
        refused = await client.post(f"/mcp/approvals/{request_id}/consume", json=LOAD)
        assert refused.status_code == 409
        assert refused.json()["detail"]["status"] == "expired"

    @pytest.mark.asyncio
    async def test_a_fresh_load_does_NOT_expire(self, api, events):
        client, maker = api
        request_id = await _create(client)
        await _backdate(maker, request_id, "created_at", 23)
        assert (await client.get(f"/mcp/approvals/{request_id}")).json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_a_STEERING_approval_never_expires(self, api, events):
        """S3-03 (2026-10-06): steering approvals do not expire."""
        client, maker = api
        request_id = await _create(client, STEER)
        await _backdate(maker, request_id, "created_at", 24 * 30)
        assert (await client.get(f"/mcp/approvals/{request_id}")).json()["status"] == "pending"


class TestTheWiring:
    def test_every_route_expires_first(self):
        """FTID §11: `expire_stale` is called FIRST by list, get, approve and consume (and deny).
        Asserted on the AST of each route, not by grep — a call in a comment or a dead branch
        must not satisfy it."""
        routes = ("list_approval_requests", "get_approval_request", "approve_request",
                  "consume_request", "deny_request")
        for name in routes:
            fn = ast.parse(inspect.getsource(getattr(mcp_approvals, name))).body[0]
            body = [n for n in fn.body
                    if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            first = body[0]
            assert (
                isinstance(first, ast.Expr) and isinstance(first.value, ast.Await)
                and isinstance(first.value.value, ast.Call)
                and getattr(first.value.value.func, "id", None) == "expire_stale"
            ), f"{name} does not call expire_stale first"

    def test_steering_endpoints_is_a_VIEW_of_the_backend_rows(self):
        assert mcp_approvals.STEERING_ENDPOINTS == {
            "steer_compare": "/api/v1/steering/async/compare",
            "steer_sweep": "/api/v1/steering/async/sweep",
            "steer_combined": "/api/v1/steering/async/combined",
        }
        assert set(mcp_approvals.STEERING_TOOLS) == set(mcp_approvals.STEERING_ENDPOINTS)
        assert mcp_approvals.CALLER_TOOLS == {
            "start_probe_training_run", "evaluate_probe_monitor",
            "start_probe_judge_run", "start_probe_spot_check",
            # Feature 040 S3 (review R1-5): the GPU arm of a recalibration.
            "recalibrate_probe_monitor",
        }
        # ENH-008 (D1): no millm_* tool files here any more — miLLM decides its own requests.
        assert not [t for t in mcp_approvals.APPROVABLE_TOOLS if t.startswith("millm_")]
        # Feature 040 S4: the one RECORD tool (a durable per-hash approval, never consumed).
        assert {t for t, (kind, _) in mcp_approvals.APPROVABLE_TOOLS.items()
                if kind is mcp_approvals.Execution.RECORD} == {"code_rule_approve"}

    def test_the_steering_summary_counts_features(self):
        assert "feature(s)" in mcp_approvals._summarize("steer_sweep", {"prompt": "x"})

    def test_the_probe_GPU_summaries_name_what_holds_the_card(self):
        """The operator approves what the line says; a probe run read as "0 feature(s)" (the
        steering fallback) would be approved blind."""
        run = mcp_approvals._summarize(
            "start_probe_training_run", {"model_id": "m_1", "train_dataset_id": "pmd_1"}
        )
        assert "m_1" in run and "pmd_1" in run and "GPU" in run
        evaluation = mcp_approvals._summarize(
            "evaluate_probe_monitor", {"probe_id": "pm_1", "dataset_ids": ["a", "b"]}
        )
        assert "pm_1" in evaluation and "2 set(s)" in evaluation
