"""`DataworksClient` — the one path to miDataworks (Feature 034, FR-5, FR-11, FR-12).

Controls (recorded 2026-10-16, task 7.2; each red, each restore verified):
  * remove the agent-header stamp (`AGENT_HEADER: self._identity` out of `sent`) →
    `test_EVERY_request_carries_the_agent_header` and
    `test_a_caller_header_CANNOT_override_the_stamp` red;
  * stamp first and caller headers second (`{AGENT_HEADER: …, **headers}`) →
    `test_a_caller_header_CANNOT_override_the_stamp` red;
  * remove the non-JSON 2xx refusal → `test_an_HTML_200_RAISES` red.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.dataworks_client import AGENT_HEADER, DataworksClient


def _client(handler, identity="agent:mistudio-mcp") -> DataworksClient:
    client = DataworksClient("http://dw:8000", identity=identity)
    client._http = httpx.AsyncClient(  # noqa: SLF001
        base_url="http://dw:8000", transport=httpx.MockTransport(handler)
    )
    return client


def test_the_header_name_is_miDataworks():
    """`core/agent_origin.py` AGENT_HEADER at miDataworks origin/main 4898e66."""
    assert AGENT_HEADER == "X-Dataworks-Agent"


@pytest.mark.parametrize("method", ["get", "post", "delete", "patch"])
def test_EVERY_request_carries_the_agent_header(method):
    seen = {}

    def handler(request):
        seen["agent"] = request.headers.get("x-dataworks-agent")
        return httpx.Response(200, json={})

    client = _client(handler)
    call = getattr(client, method)
    asyncio.run(call("/api/v1/x", json_body={}) if method in ("post", "patch", "delete")
                else call("/api/v1/x"))
    assert seen["agent"] == "agent:mistudio-mcp"


def test_a_caller_header_CANNOT_override_the_stamp():
    seen = {}

    def handler(request):
        seen["agent"] = request.headers.get("x-dataworks-agent")
        return httpx.Response(200, json={})

    asyncio.run(_client(handler).request(
        "POST", "/api/v1/label-runs", json_body={},
        headers={"X-Dataworks-Agent": "agent:someone-else", "X-Other": "1"}))
    assert seen["agent"] == "agent:mistudio-mcp"


def test_a_202_pending_body_passes_through_UNCHANGED():
    pending = {"approval_id": "ap1", "status": "pending", "action": "hub_push",
               "request_digest": "sha256:" + "0" * 64,
               "expires_at": "2026-10-17T00:00:00+00:00",
               "hint": "Waiting for the operator to approve this in miDataworks."}
    result = asyncio.run(_client(lambda r: httpx.Response(202, json=pending))
                         .post("/api/v1/publishes", json_body={}))
    assert result == pending


def test_the_error_envelope_is_unwrapped_VERBATIM():
    body = {"error": {"code": "VERSION_NOT_FOUND", "message": "No version v9.",
                      "details": {"version_id": "v9"}}}
    with pytest.raises(BackendError) as exc:
        asyncio.run(_client(lambda r: httpx.Response(404, json=body)).get("/api/v1/versions/v9"))
    assert exc.value.status_code == 404 and exc.value.detail == body["error"]


def test_an_HTML_200_RAISES():
    client = _client(lambda r: httpx.Response(200, text="<html>splash</html>"))
    with pytest.raises(BackendError, match="non-JSON"):
        asyncio.run(client.get("/api/v1/datasets"))


def test_a_read_timeout_says_it_may_already_have_been_applied():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(BackendError) as exc:
        asyncio.run(_client(handler).post("/api/v1/publishes", json_body={}))
    assert exc.value.status_code == 0 and "may already have been applied" in str(exc.value)
    assert "miDataworks" in str(exc.value)


def test_unreachable_is_status_zero():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(BackendError) as exc:
        asyncio.run(_client(handler).get("/api/v1/datasets"))
    assert exc.value.status_code == 0


def test_DELETE_carries_its_JSON_body():
    seen = {}

    def handler(request):
        seen["method"], seen["body"] = request.method, request.content
        return httpx.Response(200, json={"id": "v1"})

    asyncio.run(_client(handler).delete("/api/v1/versions/v1", json_body={"reason": "old"}))
    assert seen["method"] == "DELETE" and b'"reason"' in seen["body"]


def test_the_client_takes_no_credential():
    import inspect

    params = set(inspect.signature(DataworksClient.__init__).parameters)
    assert not {p for p in params if any(k in p for k in ("token", "secret", "password", "key"))}


def test_the_server_builds_it_WITH_the_product_identity(monkeypatch):
    """The server constructs the client from the product row; the identity must arrive."""
    from src.mcp_server.products import DATAWORKS
    from src.mcp_server.server import _client_for

    client = _client_for(DATAWORKS, "http://dw:8000")
    assert isinstance(client, DataworksClient)
    assert client._identity == DATAWORKS.identity == "agent:mistudio-mcp"  # noqa: SLF001
