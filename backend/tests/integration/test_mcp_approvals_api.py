"""Integration tests for the agent approval endpoints (Feature 010)."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.database import SyncSessionLocal
from src.main import app
from src.models.agent_approval import AgentApprovalRequest

pytestmark = pytest.mark.integration


@pytest.fixture()
def api_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test/api/v1")


@pytest.fixture()
def cleanup_approvals():
    created: list[str] = []
    yield created
    db = SyncSessionLocal()
    try:
        if created:
            db.query(AgentApprovalRequest).filter(
                AgentApprovalRequest.id.in_(created)
            ).delete(synchronize_session=False)
            db.commit()
    finally:
        db.close()


class TestApprovalFlow:
    @pytest.mark.asyncio
    async def test_create_list_deny(self, api_client, cleanup_approvals):
        async with api_client as client:
            create = await client.post(
                "/mcp/approvals",
                json={
                    "tool_name": "steer_compare",
                    "payload": {"sae_id": "sae_x", "prompt": "hello", "selected_features": []},
                },
            )
            assert create.status_code == 201
            request_id = create.json()["id"]
            cleanup_approvals.append(request_id)
            assert create.json()["status"] == "pending"

            listing = await client.get("/mcp/approvals", params={"status": "pending"})
            assert request_id in {a["id"] for a in listing.json()["approvals"]}

            deny = await client.post(
                f"/mcp/approvals/{request_id}/deny", json={"reason": "not now"}
            )
            assert deny.status_code == 200
            assert deny.json()["status"] == "denied"
            assert deny.json()["reason"] == "not now"

            # Deny is terminal — second resolution attempt conflicts
            again = await client.post(f"/mcp/approvals/{request_id}/approve")
            assert again.status_code == 409

    @pytest.mark.asyncio
    async def test_invalid_tool_name_rejected(self, api_client):
        async with api_client as client:
            response = await client.post(
                "/mcp/approvals",
                json={"tool_name": "delete_everything", "payload": {}},
            )
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_unknown_request_404(self, api_client):
        async with api_client as client:
            response = await client.get("/mcp/approvals/does-not-exist")
            assert response.status_code == 404
