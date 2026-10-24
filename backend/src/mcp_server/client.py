"""
REST client for the miStudio backend.

The ONLY path from MCP tools to the backend (BR-1.3): every tool goes through
``/api/v1`` so provenance, validation, and job-queue behavior are identical to
UI-initiated actions. Backend error details (labeling 503s, steering
circuit-breaker messages, protected-label 409s) are passed through verbatim so
agents can react to them (BR-6.2).
"""

import json
import logging
from typing import Any, Optional

import httpx
from mcp.server.mcpserver.exceptions import ToolError

from . import http_common

logger = logging.getLogger(__name__)


class BackendError(ToolError):
    """Backend returned an error; message carries the structured detail verbatim.

    A `ToolError` (mcp 2.x), so the detail still reaches the agent. mcp 2.x treats any other
    exception escaping a tool as a crash and answers only "Error executing tool <name>" — the
    backend's 409/422/503 detail, the thing BR-6.2 promises verbatim, would be dropped. Under
    mcp 1.x every exception's text was passed on. Pinned by
    tests/unit/test_mcp_errors_reach_the_agent.py.
    """

    def __init__(self, status_code: int, detail: Any,
                 headers: Optional[dict[str, str]] = None):
        self.status_code = status_code
        self.detail = detail
        #: Response headers, when the client that raised kept them (Feature 034:
        #: `MiLLMClient.request_full`), so a 503's `Retry-After` reaches the agent.
        self.headers = headers or {}
        detail_text = detail if isinstance(detail, str) else json.dumps(detail)
        super().__init__(f"backend {status_code}: {detail_text}")


#: The header and value the backend reads (`services/probe_dataset_lifecycle.AGENT_HEADER`).
#: Duplicated as a literal rather than imported: the MCP server stays on HTTP and imports no
#: backend service (`test_mcp_tools_stay_on_http.py`). `test_probe_dataset_lifecycle.py` pins
#: the two spellings equal.
AGENT_HEADER = "X-MiStudio-Agent"
AGENT_NAME = "mcp"


class MiStudioClient:
    """Thin async wrapper over the backend's /api/v1 REST API."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        #: ⚠ EVERY REQUEST SAYS IT IS AN AGENT (2026-10-21). The backend reads `X-MiStudio-Agent`
        #: to record who archived a probe dataset view ("agent:mcp") and to REFUSE every purge
        #: route with 403 `purge_operator_only` — permanent delete is the operator's alone. Set on
        #: the client rather than per call, so no tool can forget it.
        self._http = httpx.AsyncClient(
            base_url=f"{base_url}/api/v1",
            timeout=timeout,
            headers={AGENT_HEADER: AGENT_NAME},
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        try:
            response = await self._http.request(method, path, params=params, json=json_body)
        except httpx.HTTPError as e:
            raise BackendError(0, f"miStudio backend unreachable: {e}") from e

        if response.status_code >= 400:
            content_type = response.headers.get("content-type", "")
            if content_type.startswith("application/json"):
                try:
                    detail = response.json().get("detail", response.text)
                except json.JSONDecodeError:
                    detail = response.text
            else:
                detail = response.text
            raise BackendError(response.status_code, detail)

        # Feature 034 (task 2.7; miDataworks 010 §6.1 recorded it): a 2xx whose body is not JSON
        # raised a bare JSONDecodeError here — an exception no tool classifies — and the
        # MiLLMClient fix for the same defect (F20 R3-17) was never carried across. The
        # load-approval path reads `id` from this client's answer, so it now refuses the same
        # way: a structured BackendError naming a misrouted ingress, never `{}`.
        body, unparseable = http_common.parse_body(response)
        if unparseable:
            raise BackendError(
                response.status_code, http_common.non_json_2xx_detail("miStudio", path)
            )
        return body if body is not None else {}

    async def get(self, path: str, **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("GET", path, params=clean)

    async def post(self, path: str, json_body: Optional[dict[str, Any]] = None, **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("POST", path, params=clean or None, json_body=json_body)

    async def patch(self, path: str, json_body: dict[str, Any]) -> Any:
        return await self.request("PATCH", path, json_body=json_body)

    async def put(self, path: str, json_body: dict[str, Any]) -> Any:
        return await self.request("PUT", path, json_body=json_body)

    async def delete(self, path: str) -> Any:
        return await self.request("DELETE", path)
