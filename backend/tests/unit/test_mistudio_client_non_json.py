"""The miStudio MCP client refuses a 2xx that is not JSON (Feature 034, task 2.7).

Moved out of `test_millm_load_approval.py` when ENH-008 retired that file (miLLM decides its own model loads now;
the relay is asserted in `test_millm_approvals_relay.py`).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.mcp_server.client import BackendError, MiStudioClient


class TestTheMiStudioClientRefusesNonJson:
    """Control (recorded 2026-10-16, task 2.7): replacing `if unparseable:` with `if False:` in
    `MiStudioClient.request` → `test_an_HTML_200_RAISES` red."""

    def _client(self, handler) -> MiStudioClient:
        client = MiStudioClient("http://backend")
        client._http = httpx.AsyncClient(  # noqa: SLF001
            base_url="http://backend/api/v1", transport=httpx.MockTransport(handler)
        )
        return client

    def test_an_HTML_200_RAISES(self):
        client = self._client(lambda r: httpx.Response(
            200, text="<html>splash</html>", headers={"content-type": "text/html"}))
        with pytest.raises(BackendError) as exc:
            asyncio.run(client.post("/mcp/approvals", json_body={}))
        assert "non-JSON" in str(exc.value) and "NOT an empty result" in str(exc.value)

    def test_an_empty_200_is_still_an_empty_dict(self):
        client = self._client(lambda r: httpx.Response(204))
        assert asyncio.run(client.get("/x")) == {}

    def test_JSON_still_parses(self):
        client = self._client(lambda r: httpx.Response(200, json={"id": "ap_1"}))
        assert asyncio.run(client.post("/mcp/approvals", json_body={})) == {"id": "ap_1"}
