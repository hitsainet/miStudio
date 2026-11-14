"""The backend's reads of miServer (Feature 037 checklist, Feature 038 spot-checks; ENH-005 D16, D28).

miWorkbench PULLS; miServer never calls miWorkbench (D16). Everything here is a GET or a read-only POST (the
seeded sample) against the base URL in `settings.miserver_api_url`. Empty means miServer is not configured,
which is a normal state (BR-011, memory: midataworks-standalone-first applies to every sibling): the
catalogue reports "unknown", and a spot-check refuses with 503 and a sentence naming miServer.

⚠ "UNKNOWN" IS NEVER "FALSE" (037 D-T5). A failed read of miServer says nothing about whether a probe is
armed. Every reader here returns `Unknown(reason)` on any failure rather than an empty answer a caller
could mistake for "nothing is armed".

⚠ THE ROUTE NAMES ARE A CROSS-REPO CONTRACT owned by miServer (036 FPRD §7). They are module constants
pinned against miServer's own `app.openapi()` by `tests/unit/test_millm_session_api_names.py` (skips loudly
without the sibling checkout; fails under MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

import httpx

logger = logging.getLogger(__name__)

#: miLLM's probe list (exists since miLLM Feature 24). Each summary carries `source_probe_id` — the
#: miStudio probe id from the definition's provenance — from miLLM 036 FR-36.14 on.
PROBES = "/api/probes"
#: The event list (exists), filtered by `request_id`: every probe's verdicts on one turn.
EVENTS = "/api/probes/events"
#: miLLM 036 FR-36.12: the seeded, stratified sample of recorded turns.
SAMPLE = "/api/probes/sessions/sample"
#: miLLM 036 FR-36.9: one turn's newest-turn transcript (404 when none was recorded).
TRANSCRIPT = "/api/probes/sessions/{session_id}/transcript"
#: miLLM's OpenAPI document, read to detect which of the routes above it serves.
OPENAPI = "/openapi.json"

#: The two routes a spot-check cannot run without (038 TI4).
SPOT_CHECK_ROUTES = (SAMPLE, TRANSCRIPT)

READ_TIMEOUT = httpx.Timeout(3.0, connect=2.0)
RUN_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


@dataclass(frozen=True)
class Unknown:
    """A read of miServer that could not be answered, with the reason in plain words."""

    reason: str


def normalise_path(path: str) -> str:
    """`/a/{session_id}/b` → `/a/{}/b`: a route's shape without its parameter names."""
    import re

    return re.sub(r"\{[^}]*\}", "{}", path)


def base_url() -> str:
    from ..core.config import settings

    return (settings.miserver_api_url or "").rstrip("/")


def _unwrap(body: Any) -> Any:
    """miServer wraps every response as ``{success, data}`` (its `ApiResponse`)."""
    if isinstance(body, dict) and "data" in body and "success" in body:
        return body["data"]
    return body


def make_client(timeout: httpx.Timeout = READ_TIMEOUT, transport: Optional[httpx.BaseTransport] = None) -> Optional[httpx.Client]:
    url = base_url()
    if not url:
        return None
    return httpx.Client(base_url=url, timeout=timeout, transport=transport)


#: Injected by tests (an `httpx.MockTransport`), so every read goes through the real client code.
_transport_factory: Optional[Callable[[], httpx.BaseTransport]] = None


def _client(timeout: httpx.Timeout) -> Optional[httpx.Client]:
    transport = _transport_factory() if _transport_factory is not None else None
    return make_client(timeout, transport)


def armed_source_probe_ids() -> Union[Set[str], Unknown]:
    """The miWorkbench ids of every probe miServer has ARMED — one `GET /api/probes?armed=true`.

    `Unknown` when miServer is not configured, unreachable, answers an error, or serves summaries that do
    not carry `source_probe_id` (miServer before 036 FR-36.14): then it cannot say which miWorkbench probe a
    served probe came from, and guessing would be the one wrong answer.
    """
    client = _client(READ_TIMEOUT)
    if client is None:
        return Unknown("miServer is not connected (MISERVER_API_URL is not set on the miWorkbench backend)")
    try:
        with client:
            response = client.get(PROBES, params={"armed": "true"})
    except httpx.HTTPError as exc:
        return Unknown(f"miServer at {base_url()} could not be reached ({type(exc).__name__})")
    if response.status_code >= 400:
        return Unknown(f"miServer answered {response.status_code} when asked for its armed probes")
    try:
        rows = _unwrap(response.json())
    except ValueError:
        return Unknown("miServer's probe list was not JSON")
    if not isinstance(rows, list):
        return Unknown("miServer's probe list had an unexpected shape")
    if rows and not all(isinstance(r, dict) and "source_probe_id" in r for r in rows):
        return Unknown(
            "miServer does not report which miWorkbench probe each served probe came from "
            "(source_probe_id; it needs miServer Feature 036)"
        )
    return {str(r["source_probe_id"]) for r in rows if r.get("source_probe_id")}


@dataclass(frozen=True)
class NotConfigured:
    """miServer is not connected at all (`MISERVER_API_URL` unset) — a normal, complete state.

    ⚠ DISTINCT FROM `Unknown` ON PURPOSE (Feature 039 TI8, X5). After a re-cut, "miServer could not be
    reached" means copies may still serve the old bar and the agent should look
    (`miserver_list_probes`); "miServer is not configured" means there are none to look for. One type for
    both would either send a standalone miWorkbench's agent to a tool that cannot answer, or tell a
    connected one with an outage that nothing needs pushing.
    """

    reason: str


def imported_copies(source_probe_id: str) -> Union[List[str], NotConfigured, Unknown]:
    """The ids of miServer's imported copies of one miWorkbench probe — armed or not (Feature 039 FR-14).

    One `GET /api/probes` with no `armed` filter, matched on `source_probe_id` (miServer 036
    FR-36.14). A miServer whose summaries do not carry that field cannot say which copies came from
    this probe: `Unknown`, never an empty list.
    """
    client = _client(READ_TIMEOUT)
    if client is None:
        return NotConfigured("miServer is not connected (MISERVER_API_URL is not set on the miWorkbench backend)")
    try:
        with client:
            response = client.get(PROBES)
    except httpx.HTTPError as exc:
        return Unknown(f"miServer at {base_url()} could not be reached ({type(exc).__name__})")
    if response.status_code >= 400:
        return Unknown(f"miServer answered {response.status_code} when asked for its probes")
    try:
        rows = _unwrap(response.json())
    except ValueError:
        return Unknown("miServer's probe list was not JSON")
    if not isinstance(rows, list):
        return Unknown("miServer's probe list had an unexpected shape")
    if rows and not all(isinstance(r, dict) and "source_probe_id" in r for r in rows):
        return Unknown(
            "miServer does not report which miWorkbench probe each of its probes came from "
            "(source_probe_id; it needs miServer Feature 036)"
        )
    return sorted(str(r["id"]) for r in rows if r.get("source_probe_id") == source_probe_id and r.get("id"))


def armed_probes() -> Union[List[Dict[str, Any]], Unknown]:
    """miServer's armed probe summaries (each with `id` and `source_probe_id`), for the spot-check."""
    client = _client(READ_TIMEOUT)
    if client is None:
        return Unknown("miServer is not connected (MISERVER_API_URL is not set on the miWorkbench backend)")
    try:
        with client:
            response = client.get(PROBES, params={"armed": "true"})
    except httpx.HTTPError as exc:
        return Unknown(f"miServer at {base_url()} could not be reached ({type(exc).__name__})")
    if response.status_code >= 400:
        return Unknown(f"miServer answered {response.status_code} when asked for its armed probes")
    rows = _unwrap(response.json())
    if not isinstance(rows, list):
        return Unknown("miServer's probe list had an unexpected shape")
    return [r for r in rows if isinstance(r, dict)]


def spot_check_availability() -> Dict[str, Any]:
    """Whether miServer serves what a spot-check needs (038 FR-17, TI4): ``{available, reason, url}``.

    Read from miServer's own OpenAPI document, so a miServer that predates Feature 036 is refused with a
    sentence that says so, not with a 404 halfway through a run.
    """
    url = base_url()
    if not url:
        return {"available": False, "url": None,
                "reason": "miServer is not connected: set MISERVER_API_URL on the miWorkbench backend."}
    client = _client(READ_TIMEOUT)
    try:
        with client:
            response = client.get(OPENAPI)
    except httpx.HTTPError as exc:
        return {"available": False, "url": url,
                "reason": f"miServer at {url} could not be reached ({type(exc).__name__})."}
    if response.status_code >= 400:
        return {"available": False, "url": url,
                "reason": f"miServer at {url} answered {response.status_code} for its API description."}
    try:
        paths = set((response.json() or {}).get("paths") or {})
    except ValueError:
        return {"available": False, "url": url, "reason": f"miServer at {url} sent an unreadable API description."}
    # Path-parameter NAMES are miLLM's business (`{id}` or `{session_id}`); compare the shape.
    served = {normalise_path(p) for p in paths}
    missing = [route for route in SPOT_CHECK_ROUTES if normalise_path(route) not in served]
    if missing:
        return {
            "available": False,
            "url": url,
            "reason": (
                f"miServer at {url} does not serve {', '.join(missing)} yet — spot-checks need miServer "
                f"Feature 036 (probe sessions and recorded turns)."
            ),
        }
    return {"available": True, "url": url, "reason": None}


# ── Feature 038: the session reads a spot-check makes ─────────────────────────────────────────


class MiLLMUnavailable(RuntimeError):
    """miServer could not answer a read a running spot-check depends on (job-level, not per turn)."""


class SessionReader:
    """The three reads of a spot-check, on one `httpx.Client` (FTDD 038 §5; FTID 038 TI1–TI3).

    * `sample(body)` — ONE `POST /api/probes/sessions/sample` (miServer 036 FR-36.12);
    * `turn_text(session_id, turn_index)` — `GET /api/probes/sessions/{id}/transcript?turn_index=n`,
      `None` on 404 (no transcript was recorded: D5 makes recording opt-in; D52 reports the shortfall);
    * `events_for_request(request_id)` — the existing `GET /api/probes/events?request_id=`, every probe's
      verdicts on that turn (D53: every armed probe of the behaviour is compared).
    """

    def __init__(self) -> None:
        client = _client(RUN_TIMEOUT)
        if client is None:
            raise MiLLMUnavailable("miServer is not connected (MISERVER_API_URL is not set on the miWorkbench backend)")
        self._client = client

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SessionReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _call(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise MiLLMUnavailable(f"miServer could not be reached ({type(exc).__name__})") from exc
        if response.status_code >= 500:
            raise MiLLMUnavailable(f"miServer answered {response.status_code} for {path}")
        return response

    def sample(self, body: Dict[str, Any]) -> Dict[str, Any]:
        response = self._call("POST", SAMPLE, json=body)
        if response.status_code >= 400:
            raise MiLLMUnavailable(f"miServer refused the sample ({response.status_code}): {response.text[:300]}")
        data = _unwrap(response.json()) or {}
        turns = data.get("turns", data.get("rows")) or []
        return {"turns": [t for t in turns if isinstance(t, dict)], "strata": data.get("strata") or {}}

    def turn_text(self, session_id: str, turn_index: Optional[int]) -> Optional[Dict[str, Any]]:
        params = {"turn_index": turn_index} if turn_index is not None else None
        response = self._call("GET", TRANSCRIPT.format(session_id=session_id), params=params)
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise MiLLMUnavailable(f"miServer refused a transcript read ({response.status_code})")
        data = _unwrap(response.json())
        if isinstance(data, dict) and isinstance(data.get("turns"), list):
            matches = [t for t in data["turns"] if isinstance(t, dict)
                       and (turn_index is None or t.get("turn_index") == turn_index)]
            data = matches[-1] if matches else None
        if not isinstance(data, dict) or data.get("user_text") is None:
            return None
        return data

    def events_for_request(self, request_id: str) -> List[Dict[str, Any]]:
        response = self._call("GET", EVENTS, params={"request_id": request_id, "limit": 500})
        if response.status_code >= 400:
            raise MiLLMUnavailable(f"miServer refused the event list ({response.status_code})")
        rows = _unwrap(response.json())
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
