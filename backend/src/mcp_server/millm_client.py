"""
MiLLMClient — thin async wrapper over miLLM's management API (Feature 9,
Unified MCP; contract: miLLM repo docs/mcp-contract.md).

The ONE place the miLLM ApiResponse envelope
({success, data, error:{code,message,details}}) is unwrapped — tools must
never see or leak the wrapper. Raw (non-envelope) endpoints — cluster export —
use raw_get.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from . import http_common
from .client import BackendError



def _timeout_detail(e: Exception, method: str, path: str) -> str:
    """F20 R3-18 — now the shared `http_common.timeout_detail`, labelled for miLLM."""
    return http_common.timeout_detail("miLLM", e, method, path)


#: The content type miLLM serves batch files with (contract §4f; 026 FTDD TD6). Confirmed at the
#: 034 phase-6 precondition read (miLLM origin/main `6d7212b`, `routes/openai/files.py`).
JSONL_CONTENT_TYPE = "application/jsonl"

#: ENH-008 (D1): miLLM's agent-identity header. miLLM gates an agent's model loads, unloads, deletes,
#: probe-limit writes, capacity measurements and operator-code imports in ITS OWN approval queue, and
#: it tells an agent from the operator by this header alone (miLLM `api/approval_gate.py`). A request
#: without it is treated as the OPERATOR's and runs at once — so every request this client sends
#: carries it, stamped on each of the three ways out (`request_full`, `_get_jsonl_full`, `raw_get`)
#: and LAST, so no caller-supplied header can drop it. Pinned per call by
#: `test_millm_client_agent_header.py`.
AGENT_HEADER = "X-miLLM-Agent"
#: The identity this MCP path presents (`products.MILLM.identity`; `server._client_for` passes it).
DEFAULT_IDENTITY = "agent:mistudio-mcp"


#: Longest stretch of a non-JSON error body quoted back to an agent (an HTML proxy page can be
#: kilobytes; the start says what it is).
ERROR_TEXT_MAX_CHARS = 500
#: miLLM's own code for a request that fails validation (`millm.core.errors`, VALIDATION_ERROR). A
#: FastAPI `{"detail": [...]}` 422 is that refusal before miLLM's handlers ran.
VALIDATION_CODE = "VALIDATION_ERROR"


def _trimmed(text: str) -> str:
    text = text.strip()
    if len(text) <= ERROR_TEXT_MAX_CHARS:
        return text
    return text[:ERROR_TEXT_MAX_CHARS] + f"… ({len(text)} characters in all)"


def _validation_message(errors: list[Any]) -> str:
    parts = []
    for err in errors:
        if not isinstance(err, dict):
            parts.append(str(err))
            continue
        loc = ".".join(str(p) for p in (err.get("loc") or []) if p != "body")
        msg = str(err.get("msg") or err.get("message") or err.get("type") or "invalid")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def error_fields(status_code: int, detail: Any) -> dict[str, Any]:
    """`{code, message, details}` from a miLLM error, whatever shape it came in. F-3 (2026-10-29).

    A `BackendError` from miLLM carries one of: the unwrapped management envelope
    (`{code, message, details}`, built by `request_full`/`raw_get`); a raw envelope or an
    OpenAI-style body (`{"error": {...}}`); FastAPI's own 422 (`{"detail": [...]}` or
    `{"detail": "text"}`, which miLLM's management routes answer for body validation); or the body
    text of a non-JSON answer. Reading only the first shape turned miLLM's 422 for an over-long
    `reason` into `code: null, message: null` — the agent learned nothing.

    `message` is NEVER empty: without one from miLLM it names the HTTP status and quotes the body,
    trimmed. `code` is miLLM's when it sent one, `VALIDATION_ERROR` for a FastAPI validation list,
    otherwise `None`."""
    code: Any = None
    message: Any = None
    details: Any = None
    if isinstance(detail, dict):
        inner = detail.get("error") if isinstance(detail.get("error"), dict) else detail
        code = inner.get("code")
        message = inner.get("message")
        details = inner.get("details")
        if message is None and "detail" in inner:
            fastapi = inner["detail"]
            if isinstance(fastapi, list):
                code = code or VALIDATION_CODE
                message = _validation_message(fastapi)
                details = fastapi if details is None else details
            elif fastapi is not None:
                message = fastapi if isinstance(fastapi, str) else json.dumps(fastapi)
        if not message:
            message = (f"miLLM answered HTTP {status_code} without a message: "
                       f"{_trimmed(json.dumps(detail, default=str))}")
    elif isinstance(detail, list):
        code = VALIDATION_CODE
        message = _validation_message(detail)
        details = detail
    else:
        text = "" if detail is None else str(detail)
        message = (f"miLLM answered HTTP {status_code}: {_trimmed(text)}" if text.strip()
                   else f"miLLM answered HTTP {status_code} with an empty body")
    if not message:
        message = f"miLLM answered HTTP {status_code}"
    return {"code": code, "message": str(message), "details": details}


@dataclass(frozen=True)
class MiLLMResponse:
    """A successful miLLM answer with what `request()` throws away (FR-21)."""

    status: int
    body: Any
    headers: dict[str, str]


class MiLLMClient:
    """Async client for a miLLM deployment's /api management surface."""

    def __init__(self, base_url: str, timeout: float = 60.0,
                 bearer_token: str = "", identity: str = DEFAULT_IDENTITY) -> None:
        #: ENH-008: the value of `X-miLLM-Agent` on every request (never empty: an empty header
        #: would still be an agent to miLLM, but a name says WHICH agent asked).
        self._identity = identity or DEFAULT_IDENTITY
        headers = {}
        if bearer_token:
            # Forward-compatible: miLLM is unauthenticated today (contract §6);
            # a future additive bearer requirement slots in here.
            headers["Authorization"] = f"Bearer {bearer_token}"
        #: The deployment this client talks to — a cache key for per-miLLM facts (039: the OpenAPI
        #: field detection of `millm_recalibrate_probe`).
        self.base_url = base_url.rstrip("/")
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
        """The unwrapped body. Every pre-034 caller uses this, unchanged."""
        response = await self.request_full(
            method, path, params=params, json_body=json_body
        )
        return response.body

    async def request_full(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        files: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> "MiLLMResponse":
        """Status, unwrapped body AND response headers (Feature 034, FR-21).

        miLLM reports steering state, the seed it used and fields it ignored in RESPONSE
        HEADERS (`X-miLLM-Steering`, `X-miLLM-Seed`, …) and a 503's back-off in `Retry-After`.
        `request()` returned the body only, so none of that could reach an agent. Errors carry
        the headers too (`BackendError.headers`), because `Retry-After` arrives on a 503.
        """
        # ENH-008: stamped LAST, so a caller's headers can never drop or rename the agent.
        sent = {**(headers or {}), AGENT_HEADER: self._identity}
        try:
            response = await self._http.request(
                method, path, params=params, json=json_body, headers=sent,
                files=files, data=data,
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
        body, unparseable = http_common.parse_body(response)
        response_headers = dict(response.headers)

        # Envelope unwrap (contract §2). The envelope is authoritative even
        # on 4xx/5xx — its error.message is written to be agent-safe.
        if isinstance(body, dict) and "success" in body:
            if body.get("success"):
                return MiLLMResponse(response.status_code, body.get("data"),
                                     response_headers)
            error = body.get("error") or {}
            raise BackendError(
                response.status_code,
                {"code": error.get("code", "MILLM_ERROR"),
                 "message": error.get("message", "miLLM request failed"),
                 "details": error.get("details")},
                headers=response_headers,
            )

        if response.status_code >= 400:
            raise BackendError(response.status_code,
                               body if body is not None else response.text,
                               headers=response_headers)

        # R3-17: a 2xx whose body would not parse is NOT an empty success.
        if unparseable:
            raise BackendError(
                response.status_code,
                http_common.non_json_2xx_detail("miLLM", path),
                headers=response_headers,
            )
        return MiLLMResponse(response.status_code,
                             body if body is not None else {}, response_headers)

    # ── Verb methods ──────────────────────────────────────────────────────────────────────
    #
    # ⚠ TOOLS CALL THESE AS `millm.<verb>("<literal path>", …)` AND NOTHING ELSE. miLLM's
    # `tests/unit/test_mcp_tool_paths_are_real.py` reads miStudio's tool modules by regex, maps
    # every `millm.<x>(` call site to an HTTP verb and checks its literal path against miLLM's
    # routers; an unknown verb (`request_full`, `upload`, …) FAILS that guard. So the Feature 034
    # capabilities — sent headers, response headers, multipart, JSON Lines — are keyword options
    # here, not new call shapes. `test_millm_client_extensions.py` pins the same rule locally.
    #
    # `headers=` sends request headers; `full=True` returns a `MiLLMResponse` (status, body,
    # headers) instead of the body. Every other keyword is a query parameter, as before.

    async def get(self, path: str, *, headers: Optional[dict[str, str]] = None,
                  full: bool = False, jsonl: bool = False, **params: Any) -> Any:
        """GET. `jsonl=True` reads a JSON Lines body (Feature 034, FR-22) via `get_jsonl`."""
        clean = {k: v for k, v in params.items() if v is not None}
        if jsonl:
            response = await self._get_jsonl_full(path, headers=headers)
            return response if full else response.body
        response = await self.request_full("GET", path, params=clean or None, headers=headers)
        return response if full else response.body

    async def post(self, path: str, json_body: Optional[dict[str, Any]] = None, *,
                   headers: Optional[dict[str, str]] = None, full: bool = False,
                   upload: Optional[dict[str, Any]] = None, **params: Any) -> Any:
        """POST. `upload={"filename", "content", "purpose"}` sends multipart via `upload()`."""
        clean = {k: v for k, v in params.items() if v is not None}
        if upload is not None:
            response = await self.upload(path, headers=headers, **upload)
        else:
            response = await self.request_full("POST", path, params=clean or None,
                                               json_body=json_body, headers=headers)
        return response if full else response.body

    async def put(self, path: str, json_body: dict[str, Any], *,
                  headers: Optional[dict[str, str]] = None, full: bool = False) -> Any:
        response = await self.request_full("PUT", path, json_body=json_body, headers=headers)
        return response if full else response.body

    async def delete(self, path: str, *, headers: Optional[dict[str, str]] = None,
                     full: bool = False, **params: Any) -> Any:
        """DELETE with optional query params (Feature 20) and headers (Feature 034: the lease
        release sends its id in `X-miLLM-Lease`).

        Added for the circuit surface: circuit deletion and scoped
        edge-observation clearing both need it, and neither can be expressed as
        a POST without lying about the verb.
        """
        clean = {k: v for k, v in params.items() if v is not None}
        response = await self.request_full("DELETE", path, params=clean or None,
                                           headers=headers)
        return response if full else response.body

    async def upload(self, path: str, *, filename: str, content: bytes, purpose: str,
                     content_type: str = JSONL_CONTENT_TYPE,
                     headers: Optional[dict[str, str]] = None) -> "MiLLMResponse":
        """Multipart upload (`POST /v1/files`, miLLM FR-26.1.1): `file` and `purpose` fields."""
        return await self.request_full(
            "POST", path, headers=headers,
            files={"file": (filename, content, content_type)},
            data={"purpose": purpose},
        )

    async def get_jsonl(self, path: str, *,
                        headers: Optional[dict[str, str]] = None) -> list[Any]:
        """A JSON Lines body, parsed line by line (Feature 034, FR-22)."""
        return (await self._get_jsonl_full(path, headers=headers)).body

    async def _get_jsonl_full(self, path: str, *,
                              headers: Optional[dict[str, str]] = None) -> "MiLLMResponse":
        """`get_jsonl` with the status and response headers kept.

        The content type is CHECKED against the one miLLM documents (`application/jsonl`, miLLM
        contract §4f) before anything is parsed: the default path refuses a 2xx that is not JSON,
        and this path refuses one that is not JSON Lines — an HTML error page must never be
        "zero rows".
        """
        try:
            response = await self._http.request(
                "GET", path, headers={**(headers or {}), AGENT_HEADER: self._identity})
        except httpx.TimeoutException as e:
            raise BackendError(0, _timeout_detail(e, "GET", path))
        except httpx.HTTPError as e:
            raise BackendError(0, f"miLLM backend unreachable: {e}")
        response_headers = dict(response.headers)
        if response.status_code >= 400:
            body, _ = http_common.parse_body(response)
            raise BackendError(response.status_code,
                               body if body is not None else response.text,
                               headers=response_headers)
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type != JSONL_CONTENT_TYPE:
            raise BackendError(
                response.status_code,
                f"miLLM returned {content_type or 'no content type'} from {path}, not "
                f"{JSONL_CONTENT_TYPE} — the response did not come from the file store (a "
                "misrouted ingress or a proxy page). This is NOT an empty result.",
                headers=response_headers,
            )
        rows = []
        for number, line in enumerate(response.text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise BackendError(response.status_code,
                                   f"line {number} of {path} is not JSON: {e}")
        return MiLLMResponse(response.status_code, rows, response_headers)

    async def raw_get(self, path: str) -> Any:
        """GET an endpoint that returns a RAW document (no envelope) — the
        cluster export IS the portable artifact (contract §2)."""
        try:
            response = await self._http.get(path, headers={AGENT_HEADER: self._identity})
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
