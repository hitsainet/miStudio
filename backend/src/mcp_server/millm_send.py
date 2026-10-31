"""One owner of what every miLLM scoring, generation and batch request sends and returns
(Feature 034, FR-19, FR-21, FR-23).

**Sent.** `X-miLLM-Load-Policy: refuse` — scoring and generation must never swap the served model
(R-04.41) — and `X-miLLM-Strict: true` — a field miLLM would ignore becomes a visible 400 instead
of a silent drop (T-100). Plus `X-miLLM-Lease` when the agent holds the lease. Both headers are
inert on routes that never load (`/api/probes/score`, the batch routes) and are sent anyway, so
one rule covers every tool (FR-19).

**Returned.** `{"result": <body>, "millm_headers": {...}}`, with the provenance miLLM reports in
response headers (`X-miLLM-Steering`, `-Seed`, `-Constrained`, `-Ignored-Fields`,
`-Scoring-Precision`) — keys only
when the header was sent, values verbatim.

**Refused.** A fourth result shape, distinct from unavailable, argument error and backend error:
`{"refused": <code>, "message", "next", "retry_after_s", "status_code"}` for `409
model_not_resident` / `model_leased` (next: `millm_load_model`), `409
scoring_precision_unavailable` (with its `reason`, from `X-miLLM-Scoring-Precision-Reason`),
`400 model_not_generative` / `400 ambiguous_model_name` / `409 model_name_taken` (with a `hint`
naming the remedy) and every `503` (with miLLM's `Retry-After`).

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
from .millm_client import error_fields

SENT_HEADERS: dict[str, str] = {"X-miLLM-Load-Policy": "refuse", "X-miLLM-Strict": "true"}
LEASE_HEADER = "X-miLLM-Lease"

#: response header → key in `millm_headers` (FR-21).
PROVENANCE_HEADERS: dict[str, str] = {
    "x-millm-steering": "steering",
    "x-millm-seed": "seed",
    "x-millm-constrained": "constrained",
    "x-millm-ignored-fields": "ignored_fields",
    # miLLM Feature 25 (FR-25.15): the precision a `/v1/completions` scoring request RAN in. Read
    # from the header, never assumed from what was asked: miLLM refuses rather than downgrades,
    # so the echo is the only evidence a float32 answer is float32.
    "x-millm-scoring-precision": "scoring_precision",
}

#: The header miLLM sends a `409 scoring_precision_unavailable` refusal's reason in — the OpenAI
#: error envelope has no field for it (miLLM `ScoringPrecisionUnavailableError.headers`).
SCORING_PRECISION_REASON_HEADER = "x-millm-scoring-precision-reason"

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


def _code_and_message(detail: Any, status_code: int = 0) -> tuple[Optional[str], str]:
    """Lower-cased code and message from any miLLM error shape (`millm_client.error_fields`)."""
    fields = error_fields(status_code, detail)
    code = fields["code"]
    return (str(code).lower() if code else None), fields["message"]


def retry_after_seconds(headers: dict[str, str]) -> Optional[int]:
    lowered = {k.lower(): v for k, v in (headers or {}).items()}
    raw = lowered.get("retry-after")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


#: Request refusals whose remedy is a DIFFERENT MODEL NAME, not a retry (2026-10-19). Each was a
#: raised tool error before: an agent read "miLLM rejected the request" and had to parse prose to
#: learn that the fix was to pick another row.
#:
#: * `400 model_not_generative` — the named model is an embedder or a classifier (miLLM
#:   `refuse_if_non_generative`); chat/completions cannot be served by it.
#: * `400 ambiguous_model_name` — a bare repo name matches several quantizations
#:   (`repo:Q4_K_M`, `repo:IQ4_XS`, or since miLLM D56 `repo:Q8` / `repo:Q4` / `repo:FP32`); the
#:   message names the tags. miLLM never picks one silently.
#: * `409 model_name_taken` — a registration's served name is held by another row (miLLM
#:   `_unique_served_name`); pass a `custom_name`.
NAME_REFUSALS: dict[tuple[int, str], str] = {
    (400, "model_not_generative"): "millm_list_models",
    (400, "ambiguous_model_name"): "millm_list_models",
    (409, "model_name_taken"): "millm_download_model",
}

NAME_REFUSAL_HINTS: dict[str, str] = {
    "model_not_generative": (
        "That model cannot generate text (an embedder or classifier). Pick a text-generation "
        "model from millm_list_models; embedders are served by millm_embeddings."
    ),
    "ambiguous_model_name": (
        "The bare name matches several quantizations. Send the tagged name (repo:QUANT) the "
        "message lists, as millm_list_models reports it."
    ),
    "model_name_taken": (
        "Another model row already serves under that name. Re-call with a custom_name of "
        "your own."
    ),
}


def _header(headers: dict[str, str], name: str) -> Optional[str]:
    lowered = {k.lower(): v for k, v in (headers or {}).items()}
    return lowered.get(name)


def refusal(exc: BackendError) -> Optional[dict[str, Any]]:
    """The structured refusal for a miLLM 409 load refusal, a 409 scoring-precision refusal, a
    model-name refusal (400/409) or any 503; `None` otherwise."""
    code, message = _code_and_message(exc.detail, exc.status_code)
    if exc.status_code == 409 and code in LOAD_REFUSALS:
        return {"refused": code, "message": message, "next": "millm_load_model",
                "retry_after_s": None, "status_code": 409}
    if exc.status_code == 409 and code == "scoring_precision_unavailable":
        # REFUSED, NEVER DOWNGRADED (miLLM FR-25.15): the reason (memory | device |
        # model_not_float32_castable | busy) travels in a header because the OpenAI envelope has
        # no field for it; a `details.reason` is read when a future envelope carries one.
        details = exc.detail.get("details") if isinstance(exc.detail, dict) else None
        reason = _header(exc.headers, SCORING_PRECISION_REASON_HEADER) or (
            details.get("reason") if isinstance(details, dict) else None
        )
        return {"refused": code, "reason": reason, "message": message,
                "next": "retry later when reason is busy; otherwise score with "
                        "scoring_precision='native' (and compare against a native reference)",
                "retry_after_s": retry_after_seconds(exc.headers), "status_code": 409}
    if code is not None and (exc.status_code, code) in NAME_REFUSALS:
        return {"refused": code, "message": message,
                "next": NAME_REFUSALS[(exc.status_code, code)],
                "hint": NAME_REFUSAL_HINTS[code],
                "retry_after_s": None, "status_code": exc.status_code}
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
