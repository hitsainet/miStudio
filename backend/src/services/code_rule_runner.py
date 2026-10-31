"""The backend's client of the isolated operator-code runner (Feature 040 S4; FTDD Appendix C.3, C.4; FR-31).

miStudio never runs operator code in its own processes (FR-31): the API and the `rules` worker hand each code rule's
document hash, function, params and the REDUCED texts to `mistudio-rule-runner` (`src/rule_runner/`, its own
Deployment) over HTTP and read back a count or a contract reason per text. This module never imports Wasmtime
(`test_rule_runner_is_self_contained.py` proves the backend never loads it).

* `enabled(settings)` — a runner is configured (`CODE_RULE_RUNNER_URL`) AND the operator turned `code_rules_enabled`
  on. Otherwise operator code is PENDING (D41/D44): `runner_for` answers `UnavailableRunner`.
* `code_limits(settings)` — the seven request limits from the runtime settings, refused below the vendored contract's
  floor (`code_rule_limit_below_floor`; D47: raising is allowed, lowering never).
* `identity(settings)` — what a verdict computed through the runner depends on: the contract version, bundle digest
  and runtime major (from the VENDORED contract the image is built from) and the limits in force. It enters
  `rule_signals.engine_digest` for a set with code rules (Review R1-F8; 040-A6).
* `HttpCodeRuleRunner.run_batch(signal, texts)` — one request per (signal, chunk), split further only when the runner
  showed it cannot finish a whole chunk. Before each chunk it re-reads the hash's approval (`approved`, a fresh query
  per chunk, C.6): revoked → every text `rule_not_approved`, no call. A runner that cannot be reached, answers an
  error or serves another bundle → every text `rule_worker_unavailable` after a bounded retry: a host fault, never
  cached (C.3), never "did not fire".

Debt D1 – D4 of Review R2 (S4), in the 040 controls record:

* **D1 — the lane backstop.** The runner bounds a request's lane wait by ONE deadline (`wall_ms` from the request's
  start), so a chunk whose texts need more than `slots × wall_ms` in total answers its tail `rule_worker_unavailable`
  while its head ran. Those texts are RE-SENT (only them) for as long as a request makes progress, and the number of
  texts the runner did finish becomes the most this client sends of that code rule in one request from then on (the
  measured cost, per code rule). A request that runs nothing is not re-sent: the host fault stands.
* **D3 — the API path.** On the `interactive` lane a request makes ONE attempt (no retry, no sleep) and every request
  of one API call shares `INTERACTIVE_BUDGET_S` from the runner's creation (miLLM 039 R2-3's stateless budget): no
  request starts past it, none waits longer than what is left of it; what it did not reach reads
  `rule_worker_unavailable` (pending, never cached).
* **D4 — a runner down for a whole bulk evaluation.** After one chunk exhausted its attempts the runner is marked down
  for the rest of THIS evaluation (one `HttpCodeRuleRunner` per task): later chunks answer `rule_worker_unavailable`
  without a call or a sleep, instead of `ATTEMPTS` calls and two sleeps per chunk.

⚠ PRIVACY: nothing here logs a text, a source, a param or a reply byte — a log line names the code hash prefix, an
HTTP status or an error class at most.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..rule_runner import contract as runner_contract
from ..rule_signals.runner import UnavailableRunner
from .detector_next import DetectorRefused

logger = logging.getLogger(__name__)

WORKER_UNAVAILABLE = runner_contract.Reason.RULE_WORKER_UNAVAILABLE.value
NOT_APPROVED = runner_contract.Reason.RULE_NOT_APPROVED.value
INPUT_TOO_LARGE = runner_contract.Reason.RULE_INPUT_TOO_LARGE.value
#: Setting name → the contract's request-limit name (`request_limits` of the vendored contract).
LIMIT_SETTINGS = {
    "call_fuel": "code_rule_call_fuel", "load_fuel": "code_rule_load_fuel", "prelude_fuel": "code_rule_prelude_fuel",
    "memory_bytes": "code_rule_memory_bytes", "input_code_points": "code_rule_input_code_points",
    "output_bytes": "code_rule_output_bytes", "wall_ms": "code_rule_wall_ms",
}
#: Attempts per request when the runner cannot be reached (C.10: in-flight evaluations see a bounded retry during a
#: runner rollout, then `rule_worker_unavailable`).
ATTEMPTS = 3
RETRY_SLEEP_S = 1.0
#: Debt D3: the whole runner time one API request may spend (every request on the `interactive` lane, one attempt
#: each). miLLM 039 Review R2-3's `STATELESS_CODE_BUDGET_S`: a constant, not a setting.
INTERACTIVE_BUDGET_S = 30.0

#: Tests set this to an `httpx.MockTransport` factory (the `rule_set_store._transport_factory` pattern) or leave it
#: None; production talks to `CODE_RULE_RUNNER_URL`.
_transport_factory: Optional[Callable[[], Any]] = None


def _clock() -> float:
    return time.monotonic()


def http_client(timeout_s: float) -> Any:
    import httpx

    transport = _transport_factory() if _transport_factory is not None else None
    return httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0), transport=transport)


def _effective() -> Any:
    from ..core.runtime_settings import effective

    return effective()


def enabled(settings: Any = None) -> bool:
    settings = settings if settings is not None else _effective()
    return bool(str(settings.code_rule_runner_url or "").strip()) and bool(settings.code_rules_enabled)


def code_limits(settings: Any = None) -> Dict[str, int]:
    """The request limits in force, refused below the contract floor (an environment value can be lower; the
    environment route refuses a lower override at write)."""
    settings = settings if settings is not None else _effective()
    floor = runner_contract.floor()
    names = runner_contract.request_limits()
    # Explicit reads, one per setting (`test_settings_readers.py` finds each).
    values = {"call_fuel": settings.code_rule_call_fuel, "load_fuel": settings.code_rule_load_fuel,
              "prelude_fuel": settings.code_rule_prelude_fuel, "memory_bytes": settings.code_rule_memory_bytes,
              "input_code_points": settings.code_rule_input_code_points,
              "output_bytes": settings.code_rule_output_bytes, "wall_ms": settings.code_rule_wall_ms}
    assert set(values) == set(LIMIT_SETTINGS)
    out = {name: int(values[name]) for name in names}
    low = {name: (out[name], floor[name]) for name in names if name in floor and out[name] < floor[name]}
    if low:
        name, (value, minimum) = sorted(low.items())[0]
        raise DetectorRefused(409, "code_rule_limit_below_floor",
                              f"{LIMIT_SETTINGS[name]} is {value}, below the runner contract's floor {minimum}; raise "
                              f"it to at least the floor (a floor may be raised, never lowered)",
                              {"setting": LIMIT_SETTINGS[name], "value": value, "floor": minimum})
    return out


def identity(settings: Any = None) -> Dict[str, Any]:
    settings = settings if settings is not None else _effective()
    if not enabled(settings):
        return {"available": False}
    return {**runner_contract.identity(), "limits": code_limits(settings)}


class HttpCodeRuleRunner:
    """The production `CodeRuleRunner` (`rule_signals.runner`)."""

    available = True

    def __init__(self, base_url: str, *, limits: Mapping[str, int], documents: Mapping[str, bytes],
                 approved: Optional[Callable[[str], bool]] = None, lane: str = "batch",
                 client: Any = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.limits = dict(limits)
        self.documents = dict(documents)
        self.approved = approved
        self.lane = lane
        self._client = client
        self._identity = {**runner_contract.identity(), "limits": dict(limits)}
        self.calls = 0
        # D3: the API path makes one attempt per request, all of them inside one budget.
        self.attempts = 1 if lane == "interactive" else ATTEMPTS
        self.deadline: Optional[float] = _clock() + INTERACTIVE_BUDGET_S if lane == "interactive" else None
        # D4: set once a request exhausted its attempts; the rest of this runner's work calls nothing.
        self.down = False
        # D1: per code rule, the most texts one request is known to finish inside the runner's lane backstop.
        self.fit: Dict[str, int] = {}

    def identity(self) -> Dict[str, Any]:
        return dict(self._identity)

    def _http(self) -> Any:
        if self._client is None:
            self._client = http_client(2 * self.limits["wall_ms"] / 1000 + 30)
        return self._client

    def run_batch(self, signal: Mapping[str, Any], texts: Sequence[str]) -> List[Dict[str, Any]]:
        sha = str(signal["code_sha256"])
        n = len(texts)
        if self.approved is not None and not self.approved(sha):
            return [{"reason": NOT_APPROVED} for _ in range(n)]
        canonical = self.documents.get(sha)
        if canonical is None:  # checked at load (`check_code_rules`); a host-side gap, never a verdict
            logger.error("code rule %s… has no document in this request", sha[:12])
            return [{"reason": WORKER_UNAVAILABLE} for _ in range(n)]
        if self.down:  # D4: no call, no sleep
            return [{"reason": WORKER_UNAVAILABLE} for _ in range(n)]
        out: List[Dict[str, Any]] = [{"reason": WORKER_UNAVAILABLE} for _ in range(n)]
        # D5 (the request body): a text longer than `input_code_points` is answered here exactly as the runner answers
        # it (`len(text) > limits.input_code_points` → `rule_input_too_large`; a lone surrogate's replacement keeps the
        # length), so it never travels: the runner's memory holds the instances plus bounded bodies.
        todo = []
        for i, text in enumerate(texts):
            if len(text) > self.limits["input_code_points"]:
                out[i] = {"reason": INPUT_TOO_LARGE}
            else:
                todo.append(i)
        while todo:
            size = min(len(todo), self.fit.get(sha, len(todo)))
            piece, todo = todo[:size], todo[size:]
            answer = self._request(signal, sha, canonical, [texts[i] for i in piece])
            if answer is None:  # unreachable after the retry, or the API budget is spent: all that is left is pending
                break
            starved = []
            for i, result in zip(piece, answer):
                out[i] = result
                if result.get("reason") == WORKER_UNAVAILABLE:
                    starved.append(i)
            ran = len(piece) - len(starved)
            if starved and ran:
                # D1: the runner's lane backstop cut this request short. Re-send only what it did not run, and never
                # again send this code rule more texts than it just finished.
                self.fit[sha] = ran
                todo = starved + todo
        return out

    def _request(self, signal: Mapping[str, Any], sha: str, canonical: bytes,
                 texts: Sequence[str]) -> Optional[List[Dict[str, Any]]]:
        body = {
            "contract": runner_contract.FORMAT, "mode": "stored", "lane": self.lane, "limits": self.limits,
            "document": {"code_sha256": sha, "canonical_utf8": canonical.decode("utf-8")},
            "calls": [{"function": signal["function"], "params": dict(signal.get("params") or {}),
                       "texts": [runner_contract.replace_lone_surrogates(t) for t in texts]}],
        }
        for attempt in range(self.attempts):
            left = None if self.deadline is None else self.deadline - _clock()
            if left is not None and left <= 0:
                logger.warning("code_rule_interactive_budget_spent: code rule %s… not run (%.0f s budget)", sha[:12],
                               INTERACTIVE_BUDGET_S)
                return None
            self.calls += 1
            answer = self._post(body, sha, left)
            if answer is not None:
                return self._results(answer, len(texts), sha)
            if attempt + 1 < self.attempts:
                time.sleep(RETRY_SLEEP_S)
        self.down = True
        logger.error("rule runner unreachable after %d attempt(s) for code rule %s…; the rest of this run reads "
                     "rule_worker_unavailable without calling it", self.attempts, sha[:12])
        return None

    def _post(self, body: Dict[str, Any], sha: str, left: Optional[float] = None) -> Optional[Dict[str, Any]]:
        extra: Dict[str, Any] = {}
        if left is not None:  # D3: never wait longer than what is left of the API budget
            import httpx

            read = min(2 * self.limits["wall_ms"] / 1000 + 30, left)
            extra["timeout"] = httpx.Timeout(read, connect=min(5.0, read))
        try:
            response = self._http().post(self.base_url + "/v1/run", json=body, **extra)
        except Exception as exc:  # noqa: BLE001 - logged by class and hash only
            logger.warning("rule runner unreachable (%s) for code rule %s…", type(exc).__name__, sha[:12])
            return None
        if response.status_code != 200:
            code = None
            try:
                code = response.json().get("error", {}).get("code")
            except Exception:  # noqa: BLE001
                pass
            logger.warning("rule runner answered %s (%s) for code rule %s…", response.status_code, code, sha[:12])
            if 400 <= response.status_code < 500 and response.status_code not in (408, 409, 429):
                return {"refused": code or response.status_code}
            return None
        try:
            return dict(response.json())
        except Exception:  # noqa: BLE001
            return None

    def _results(self, answer: Dict[str, Any], n: int, sha: str) -> List[Dict[str, Any]]:
        if "refused" in answer:
            return [{"reason": WORKER_UNAVAILABLE} for _ in range(n)]
        if answer.get("bundle_digest") != self._identity["bundle_digest"] or \
                answer.get("contract_version") != self._identity["contract_version"]:
            logger.error("rule runner serves another bundle/contract than this miStudio's; code rule %s… not run",
                         sha[:12])
            return [{"reason": WORKER_UNAVAILABLE} for _ in range(n)]
        try:
            rows = answer["results"][0]
        except (KeyError, IndexError, TypeError):
            return [{"reason": WORKER_UNAVAILABLE} for _ in range(n)]
        if not isinstance(rows, list) or len(rows) != n:
            return [{"reason": WORKER_UNAVAILABLE} for _ in range(n)]
        out: List[Dict[str, Any]] = []
        for row in rows:
            count, reason = (row or {}).get("count"), (row or {}).get("reason")
            if type(count) is int and count >= 0 and reason is None:
                out.append({"count": count})
            elif isinstance(reason, str) and reason in runner_contract.REASONS and count is None:
                out.append({"reason": reason})
            else:  # the runner broke its own grammar: a host fault, never a verdict
                out.append({"reason": WORKER_UNAVAILABLE})
        return out


def runner_for(settings: Any, *, documents: Mapping[str, bytes], approved: Optional[Callable[[str], bool]],
               lane: str = "batch") -> Any:
    """The runner a request or a task uses: `UnavailableRunner` (code pending) unless `enabled`."""
    if not enabled(settings):
        return UnavailableRunner()
    return HttpCodeRuleRunner(str(settings.code_rule_runner_url), limits=code_limits(settings), documents=documents,
                              approved=approved, lane=lane)
