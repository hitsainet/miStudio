"""
MiLLMClient — thin async wrapper over miLLM's management API (Feature 9,
Unified MCP; contract: miLLM repo docs/mcp-contract.md).

The ONE place the miLLM ApiResponse envelope
({success, data, error:{code,message,details}}) is unwrapped — tools must
never see or leak the wrapper. Raw (non-envelope) endpoints — cluster export —
use raw_get.
"""

import json
from typing import Any, Optional

import httpx

from .client import BackendError



def _timeout_detail(e: Exception, method: str, path: str) -> str:
    """F20 R3-18. `str(e)` on an httpx timeout is frequently EMPTY, so the
    message read "miLLM request timed out: " with no diagnostic at all.

    The PHASE is what matters: a connect timeout means miLLM was never reached,
    while a read timeout on a POST means the request may have ALREADY COMMITTED
    server-side — an agent that retries the second one can double-import or
    double-activate. Shared by both request paths, because they had the same
    defect in the same words.
    """
    detail = (
        f"miLLM request timed out ({type(e).__name__}) on {method} {path}"
    )
    if str(e):
        detail += f": {e}"
    if isinstance(e, httpx.ReadTimeout):
        detail += (
            " — a read timeout means the request may already have been "
            "applied; check state before retrying."
        )
    return detail


class MiLLMClient:
    """Async client for a miLLM deployment's /api management surface."""

    def __init__(self, base_url: str, timeout: float = 60.0,
                 bearer_token: str = "") -> None:
        headers = {}
        if bearer_token:
            # Forward-compatible: miLLM is unauthenticated today (contract §6);
            # a future additive bearer requirement slots in here.
            headers["Authorization"] = f"Bearer {bearer_token}"
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
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
            response = await self._http.request(
                method, path, params=params, json=json_body
            )
        except httpx.TimeoutException as e:
            # 009 R3: a reachable-but-slow miLLM was reported 'unreachable',
            # sending agents down a connectivity rabbit hole (and looping
            # retries on heavy imports).
            raise BackendError(0, _timeout_detail(e, method, path))
        except httpx.HTTPError as e:
            raise BackendError(0, f"miLLM backend unreachable: {e}")

        # F20 R3-17: `unparseable` distinguishes "the response had no body" from
        # "the body was there and was not JSON". Both used to collapse to
        # `body = None`, and on a 2xx that meant `return {}` — so a 200 HTML
        # splash page from a misrouted ingress reached the agent as an EMPTY
        # SUCCESSFUL RESULT.
        #
        # The concrete harm: `millm_circuit_status` returns `{}`, the agent
        # concludes NOTHING IS STEERING, and activates a circuit that contends
        # with one already serving that layer. The JSONDecodeError was the only
        # evidence the response had not come from miLLM at all, and it was
        # discarded.
        #
        # `raw_get` already guarded this (009 R2) and the guard was never
        # carried across to the enveloped path — the same one-representative
        # pattern R3 found in the reachability harness and both copy audits.
        body: Any = None
        unparseable = False
        if response.content:
            try:
                body = response.json()
            except json.JSONDecodeError:
                body = None
                unparseable = True

        # Envelope unwrap (contract §2). The envelope is authoritative even
        # on 4xx/5xx — its error.message is written to be agent-safe.
        if isinstance(body, dict) and "success" in body:
            if body.get("success"):
                return body.get("data")
            error = body.get("error") or {}
            raise BackendError(
                response.status_code,
                {"code": error.get("code", "MILLM_ERROR"),
                 "message": error.get("message", "miLLM request failed"),
                 "details": error.get("details")},
            )

        if response.status_code >= 400:
            raise BackendError(response.status_code,
                               body if body is not None else response.text)

        # R3-17: a 2xx whose body would not parse is NOT an empty success.
        if unparseable:
            raise BackendError(
                response.status_code,
                f"miLLM returned non-JSON content from {path} — the response "
                "did not come from the API (a misrouted ingress or a proxy "
                "error page). This is NOT an empty result: do not read it as "
                "'nothing is configured'.",
            )
        return body if body is not None else {}

    async def get(self, path: str, **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("GET", path, params=clean or None)

    async def post(self, path: str, json_body: Optional[dict[str, Any]] = None,
                   **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("POST", path, params=clean or None,
                                  json_body=json_body)

    async def put(self, path: str, json_body: dict[str, Any]) -> Any:
        return await self.request("PUT", path, json_body=json_body)

    async def delete(self, path: str, **params: Any) -> Any:
        """DELETE with optional query params (Feature 20).

        Added for the circuit surface: circuit deletion and scoped
        edge-observation clearing both need it, and neither can be expressed as
        a POST without lying about the verb.
        """
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("DELETE", path, params=clean or None)

    async def raw_get(self, path: str) -> Any:
        """GET an endpoint that returns a RAW document (no envelope) — the
        cluster export IS the portable artifact (contract §2)."""
        try:
            response = await self._http.get(path)
        except httpx.TimeoutException as e:
            raise BackendError(0, _timeout_detail(e, "GET", path))
        except httpx.HTTPError as e:
            raise BackendError(0, f"miLLM backend unreachable: {e}")
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = None
        if response.status_code >= 400:
            # Export failures still arrive as envelope JSON — surface the
            # structured {code, message, details} like every other path
            # (009 R1: agents got the envelope as an escaped string blob).
            if isinstance(body, dict) and body.get("error"):
                error = body["error"]
                raise BackendError(response.status_code,
                                   {"code": error.get("code", "MILLM_ERROR"),
                                    "message": error.get("message", ""),
                                    "details": error.get("details")})
            # Non-envelope JSON (e.g. FastAPI 422 detail) stays structured
            # rather than re-stringified (009 R2).
            raise BackendError(response.status_code,
                               body if body is not None else response.text)
        if body is None:
            # 2xx non-JSON (misrouted ingress, splash page) must be a
            # structured BackendError, not a bare JSONDecodeError (009 R2).
            raise BackendError(
                response.status_code,
                f"export returned non-JSON content from {path}")
        return body
