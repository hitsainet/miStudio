"""
HealthGate — per-product availability cache for the unified MCP server
(Feature 9).

Tools stay registered through outages (MCP clients cache tool lists;
unregistering churns agents — contract §3): when a product is down, gated
tools return a structured {"unavailable": <product>, "reason": …} result
instead. `degraded` (any 2xx) is AVAILABLE — miLLM with no model loaded must
still accept imports and report status.
"""

import asyncio
import time
from typing import Callable, Optional

import httpx

PROBE_TIMEOUT_S = 3.0

# product -> health path relative to that product's base URL
_HEALTH_PATHS = {"millm": "/api/health", "mistudio": "/api/v1/system/health"}


def _retrieve_exception(task: "asyncio.Task") -> None:
    """Consume a background task's result so asyncio does not warn about it.

    MIS-E2E-118: this was `lambda t: t.exception()`. Calling `.exception()` on
    a CANCELLED task re-raises `CancelledError` — inside a done callback, where
    asyncio catches it and logs `ERROR asyncio: Exception in callback`. So every
    graceful shutdown, which cancels the in-flight refresh, printed an
    ERROR-level traceback for a task behaving exactly as intended.

    Cosmetic on its own, and that is the problem: it is precisely the noise
    that teaches an operator to skim shutdown logs.
    """
    if task.cancelled():
        return
    task.exception()


class HealthGate:
    """TTL-cached availability probes, one entry per product."""

    def __init__(self, millm_url: str = "", ttl_s: float = 10.0,
                 mistudio_url: str = "") -> None:
        self._urls = {"millm": millm_url.rstrip("/"),
                      "mistudio": mistudio_url.rstrip("/")}
        self._ttl = ttl_s
        # product -> (checked_at_monotonic, available, reason)
        self._cache: dict[str, tuple[float, bool, str]] = {}
        # Lazily built long-lived client (009 R2: eager construction leaked
        # a client per instantiation in tests; per-probe construction cost a
        # TCP setup every TTL window).
        self._http: Optional[httpx.AsyncClient] = None
        # Single-flight per product: concurrent tool calls at TTL expiry
        # must not stampede an already-struggling backend (009 R2).
        # NOTE single-loop object: locks, the lazy client, and background
        # tasks bind to the first loop that uses them.
        self._locks: dict[str, asyncio.Lock] = {}
        # Strong refs to background refreshes (009 R3: create_task results
        # were discarded — collectable under GC pressure, and exceptions
        # were never retrieved). Also dedups task churn per product.
        self._refreshing: dict[str, asyncio.Task] = {}

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=PROBE_TIMEOUT_S,
                                           follow_redirects=False)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def check(self, product: str) -> tuple[bool, str]:
        """(available, reason). reason is agent-readable on refusal."""
        hit = self._cache.get(product)
        if hit is not None and time.monotonic() - hit[0] < self._ttl:
            return hit[1], hit[2]
        lock = self._locks.setdefault(product, asyncio.Lock())
        async with lock:
            # Re-check under the lock — a concurrent caller may have probed.
            hit = self._cache.get(product)
            if hit is not None and time.monotonic() - hit[0] < self._ttl:
                return hit[1], hit[2]
            available, reason = await self._probe(product)
            # Stamp AFTER the probe: stamping before it burned up to the
            # probe timeout out of every TTL window (009 R2).
            self._cache[product] = (time.monotonic(), available, reason)
        return available, reason

    def snapshot(self, product: str) -> tuple[Optional[bool], str]:
        """Last known state WITHOUT probing — never blocks (the /health
        route runs on orchestrator probe timeouts; a hung backend must not
        take the MCP server out of rotation: 009 R2). Kicks off a background
        refresh when the entry is stale. (None, ...) = never probed yet."""
        import asyncio

        hit = self._cache.get(product)
        stale = hit is None or time.monotonic() - hit[0] >= self._ttl
        pending = self._refreshing.get(product)
        if stale and (pending is None or pending.done()):
            try:
                task = asyncio.get_running_loop().create_task(
                    self.check(product))
                task.add_done_callback(_retrieve_exception)
                self._refreshing[product] = task
            except RuntimeError:
                pass  # no loop (sync test context) — next check() will probe
        if hit is None:
            return None, "not probed yet"
        return hit[1], hit[2]

    def invalidate(self, product: Optional[str] = None) -> None:
        if product is None:
            self._cache.clear()
        else:
            self._cache.pop(product, None)

    @staticmethod
    def public_reason(reason: str) -> str:
        """Coarse category safe for the UNAUTHENTICATED /health route —
        detailed reasons carry internal URLs and exception internals
        (009 R2 topology leak) and stay in logs + authenticated tools."""
        if reason in ("ok", "not probed yet"):
            return reason
        # Anchored checks (009 R3: substring matching miscategorized an
        # 'unhealthy' hostname inside an HTTP-status reason)
        if reason.startswith("HTTP "):
            return "error response"
        if reason.startswith("status 'unhealthy'"):
            return "unhealthy"
        if reason.startswith("unknown product"):
            return "unknown product"
        if "not configured" in reason:
            return "not configured"
        if reason.startswith("timed out"):
            return "timed out"
        return "unreachable"

    async def _probe(self, product: str) -> tuple[bool, str]:
        path = _HEALTH_PATHS.get(product)
        if path is None:
            return False, f"unknown product '{product}'"
        base = self._urls.get(product, "")
        if not base:
            return False, ("MILLM_API_URL is not configured"
                           if product == "millm"
                           else f"{product} URL is not configured")
        url = f"{base}{path}"
        try:
            response = await self._client().get(url)
        except httpx.TimeoutException as e:
            return False, f"timed out after {PROBE_TIMEOUT_S}s ({url})"
        except httpx.HTTPError as e:
            return False, f"{type(e).__name__}: {e} ({url})"
        except httpx.InvalidURL as e:
            # MIS-E2E-116: `InvalidURL` derives from `Exception`, NOT from
            # `httpx.HTTPError` — its MRO is (InvalidURL, Exception, ...). So a
            # malformed configured URL escaped both handlers above and
            # propagated out of the gate, breaking the one promise the gate
            # makes: the server instructions tell agents an unavailable product
            # returns {"unavailable": …, "reason": …} and never raises. A
            # typo'd MILLM_API_URL turned every gated tool into a traceback.
            return False, f"malformed URL: {e} ({url!r})"
        except Exception as e:  # noqa: BLE001 - the gate's contract is to answer
            # Anything else is still an answer, not an exception. This gate is
            # the boundary that turns "cannot reach it" into a value; if it
            # raises, every caller downstream has to re-implement that.
            return False, f"probe failed, {type(e).__name__}: {e} ({url})"
        # Strictly 2xx: a 3xx from an ingress fronting a dead backend must
        # not count as available (009 R1; redirects are not followed).
        if not 200 <= response.status_code < 300:
            return False, f"HTTP {response.status_code} from {url}"
        # Contract: available <=> 2xx AND status != 'unhealthy'. degraded IS
        # available (miLLM with no model loaded still accepts imports).
        # NOTE: today's liveness endpoints hardcode healthy statuses — this
        # branch is defensive for the contract's reserved semantics.
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("status") == "unhealthy":
                return False, f"status 'unhealthy' from {url}"
        except ValueError:
            pass  # non-JSON body on 2xx — treat as available
        return True, "ok"


def gated(gate: HealthGate, product: str) -> Callable:
    """Decorator: run the gate before the tool body; on refusal return the
    uniform structured-unavailable result (never raise — pitfall 1/3)."""
    def wrap(fn: Callable) -> Callable:
        import functools

        from .client import BackendError

        @functools.wraps(fn)
        async def inner(*args, **kwargs):
            ok, reason = await gate.check(product)
            if not ok:
                return {"unavailable": product, "reason": reason}
            try:
                return await fn(*args, **kwargs)
            except BackendError as exc:
                if exc.status_code == 0:
                    # Mid-TTL outage (EC-9.3): the gate said ok seconds ago
                    # but the call itself couldn't reach the backend — make
                    # THIS call structured and expire the stale entry so
                    # the next one probes.
                    gate.invalidate(product)
                    return {"unavailable": product, "reason": str(exc)}
                raise

        return inner

    return wrap
