"""
DataworksClient — the ONLY path from miStudio's MCP server to miDataworks (Feature 034, FR-5).

A thin async client over miDataworks' REST API (contract: miDataworks `docs/mcp-contract.md`,
"REST contract for proxies"):

* **Success** is plain JSON, returned as is — including a `202` pending-approval body
  (`{approval_id, status, action, request_digest, expires_at, hint}`), which the proxy passes
  through unchanged (FR-11).
* **Errors** arrive as `{"error": {"code", "message", "details"}}` and are unwrapped here, once,
  into `BackendError(status, {code, message, details})` verbatim.
* **A 2xx that is not JSON is refused**, never an empty success (the misrouted-ingress lesson).
* **A read timeout** says the request may already have been applied.
* **Every request carries `X-Dataworks-Agent: <identity>`**, stamped inside `request()` AFTER any
  caller headers, so no tool can omit or override it. That header is how miDataworks knows the
  call is an agent's and applies its approval gate and its "who" (FR-11, FR-12; miDataworks
  FR-010.8, C5). No tool takes a "who" argument.
* No credential of its own: miDataworks' REST API is unauthenticated on the LAN (miDataworks
  ADR-014), and the Hugging Face token stays in miDataworks' worker (FR-13).
"""

from typing import Any, Optional

import httpx

from . import http_common
from .client import BackendError

#: miDataworks' agent-identity header (`core/agent_origin.py` AGENT_HEADER, read at miDataworks
#: origin/main `4898e66`, 2026-10-16; contract "REST contract for proxies").
AGENT_HEADER = "X-Dataworks-Agent"


class DataworksClient:
    """Async client for a miDataworks deployment's `/api` surface."""

    def __init__(self, base_url: str, identity: str = "agent:mistudio-mcp",
                 timeout: float = 60.0) -> None:
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)
        self._identity = identity

    async def close(self) -> None:
        await self._http.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Any:
        # The stamp is applied LAST: a caller's headers cannot remove or replace it.
        sent = {**(headers or {}), AGENT_HEADER: self._identity}
        try:
            response = await self._http.request(
                method, path, params=params, json=json_body, headers=sent
            )
        except httpx.TimeoutException as e:
            raise BackendError(0, http_common.timeout_detail("miDataworks", e, method, path))
        except httpx.HTTPError as e:
            raise BackendError(0, f"miDataworks backend unreachable: {e}")

        body, unparseable = http_common.parse_body(response)
        if response.status_code >= 400:
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                error = body["error"]
                raise BackendError(response.status_code, {
                    "code": error.get("code"),
                    "message": error.get("message"),
                    "details": error.get("details"),
                })
            raise BackendError(response.status_code,
                               body if body is not None else response.text)
        if unparseable:
            raise BackendError(
                response.status_code,
                http_common.non_json_2xx_detail("miDataworks", path),
            )
        return body if body is not None else {}

    async def get(self, path: str, **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("GET", path, params=clean or None)

    async def post(self, path: str, json_body: Optional[Any] = None, **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("POST", path, params=clean or None, json_body=json_body)

    async def patch(self, path: str, json_body: Any) -> Any:
        return await self.request("PATCH", path, json_body=json_body)

    async def delete(self, path: str, json_body: Optional[Any] = None, **params: Any) -> Any:
        """DELETE with an optional JSON body: `DELETE /versions/{id}` takes `{reason}`
        (miDataworks 002 FTDD §5)."""
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("DELETE", path, params=clean or None, json_body=json_body)
