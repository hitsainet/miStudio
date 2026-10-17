"""One owner of what every miLLM scoring, generation and batch request sends and returns
(Feature 034, FR-19, FR-21, FR-23).

**Sent.** `X-miLLM-Load-Policy: refuse` — scoring and generation must never swap the served model
(R-04.41) — and `X-miLLM-Strict: true` — a field miLLM would ignore becomes a visible 400 instead
of a silent drop (T-100). Plus `X-miLLM-Lease` when the agent holds the lease. Both headers are
inert on routes that never load (`/api/probes/score`, the batch routes) and are sent anyway, so
one rule covers every tool (FR-19).

**Returned.** `{"result": <body>, "millm_headers": {...}}`, with the provenance miLLM reports in
response headers (`X-miLLM-Steering`, `-Seed`, `-Constrained`, `-Ignored-Fields`) — keys only
when the header was sent, values verbatim.

**Refused.** A fourth result shape, distinct from unavailable, argument error and backend error:
`{"refused": <code>, "message", "next", "retry_after_s", "status_code"}` for `409
model_not_resident` / `model_leased` (next: `millm_load_model`) and every `503` (with miLLM's
`Retry-After`).

⚠ HOW TOOLS CALL IT, AND WHY IT IS SHAPED THIS WAY. A tool writes
`await send(gate, millm.post("/v1/chat/completions", json_body=…, headers=sent_headers(lease_id),
full=True))`. The call stays a literal `millm.<verb>("<path>")` because miLLM's
`test_mcp_tool_paths_are_real.py` reads that form by regex from miStudio's modules and checks the
path against miLLM's routers; a helper taking `(method, path)` would hide every path from it. The
headers still have one owner (`sent_headers`), and `test_millm_inference_tools.py` asserts by AST
that every `millm.*` call in these modules is wrapped in `send(...)` and passes
`headers=sent_headers(...)`.
"""

from __future__ import annotations

from typing import Any, Awaitable, Optional

from .client import BackendError

SENT_HEADERS: dict[str, str] = {"X-miLLM-Load-Policy": "refuse", "X-miLLM-Strict": "true"}
LEASE_HEADER = "X-miLLM-Lease"

#: response header → key in `millm_headers` (FR-21).
PROVENANCE_HEADERS: dict[str, str] = {
    "x-millm-steering": "steering",
    "x-millm-seed": "seed",
    "x-millm-constrained": "constrained",
    "x-millm-ignored-fields": "ignored_fields",
}

#: 409 codes that mean "the model you named is not the one serving, and this request will not
#: change that" — the agent's next step is a (approved) load.
LOAD_REFUSALS = frozenset({"model_not_resident", "model_leased"})


def sent_headers(lease_id: Optional[str] = None) -> dict[str, str]:
    """The headers every scoring, generation and batch request sends."""
    headers = dict(SENT_HEADERS)
    if lease_id:
        headers[LEASE_HEADER] = lease_id
    return headers


def provenance(headers: dict[str, str]) -> dict[str, str]:
    """The `X-miLLM-*` provenance a response carried — verbatim, keys only when present."""
    lowered = {k.lower(): v for k, v in (headers or {}).items()}
    return {key: lowered[name] for name, key in PROVENANCE_HEADERS.items() if name in lowered}


def steering_state(raw: Optional[str]) -> str:
    """What `X-miLLM-Steering` says, coarsely (FR-21).

    The header is an RFC 8941 List of kind tokens (miLLM 028 FTDD §5.2). ONLY a List whose sole
    member is `none` means unsteered. `unknown` (with its `reason`) is NOT unsteered, and an
    absent header is a miLLM that predates the feature — "not_reported", never "unsteered".
    """
    if raw is None:
        return "not_reported"
    kinds = [member.split(";", 1)[0].strip().lower() for member in raw.split(",")]
    kinds = [k for k in kinds if k]
    if kinds == ["none"]:
        return "unsteered"
    if "unknown" in kinds or not kinds:
        return "unknown"
    return "steered"


def _code_and_message(detail: Any) -> tuple[Optional[str], str]:
    if isinstance(detail, dict):
        error = detail.get("error") if isinstance(detail.get("error"), dict) else detail
        code = error.get("code")
        message = error.get("message") or ""
        return (str(code).lower() if code else None), str(message)
    return None, str(detail)


def retry_after_seconds(headers: dict[str, str]) -> Optional[int]:
    lowered = {k.lower(): v for k, v in (headers or {}).items()}
    raw = lowered.get("retry-after")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def refusal(exc: BackendError) -> Optional[dict[str, Any]]:
    """The structured refusal for a miLLM 409 load refusal or any 503; `None` otherwise."""
    code, message = _code_and_message(exc.detail)
    if exc.status_code == 409 and code in LOAD_REFUSALS:
        return {"refused": code, "message": message, "next": "millm_load_model",
                "retry_after_s": None, "status_code": 409}
    if exc.status_code == 503:
        return {"refused": code or "unavailable", "message": message, "next": None,
                "retry_after_s": retry_after_seconds(exc.headers), "status_code": 503}
    return None


async def send(gate: Any, call: Awaitable[Any]) -> dict[str, Any]:
    """Await one miLLM call made with `full=True`, and shape what comes back.

    A connection failure or read timeout (`BackendError(0)`) invalidates the gate and reads as
    unavailable, with the timeout's "may already have been applied" message intact (FR-23).
    """
    try:
        response = await call
    except BackendError as exc:
        if exc.status_code == 0:
            gate.invalidate("millm")
            return {"unavailable": "millm", "reason": str(exc)}
        refused = refusal(exc)
        if refused is not None:
            return refused
        raise
    return {"result": response.body, "millm_headers": provenance(response.headers)}
