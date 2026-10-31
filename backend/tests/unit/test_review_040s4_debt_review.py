"""040 S4 debt D1 – D5, independent review (2026-10-25): the regressions for the mutations the builder's tests let
survive, and for the approval-request route's API-path bound (record: `0xcc/reviews/040_S4_debt_review_2026-10-25.md`).

* **RV-1 (mutation M5 survived)** — only `rule_worker_unavailable` is re-sent. A contract reason the guest produced
  (`rule_timeout`, `rule_error`, …) is a verdict about that text (miLLM D64: `rule_timeout` reported by the runner is
  final); re-sending it would run the guest again and shrink the measured fit for nothing.
* **RV-2 (mutation M6 survived)** — once a piece of a split chunk exhausted its attempts, the rest of THAT chunk is not
  sent (`break`, not `continue`): a runner that goes down mid-chunk costs `ATTEMPTS` calls, not `ATTEMPTS` per piece.
* **RV-3 (mutation M18 survived)** — on the interactive lane the CONNECT timeout is capped by what is left of the
  budget too, not a flat 5 s.
* **RV-4 (pre-existing, fixed here)** — `POST /code-rules/{sha}/approval-requests` (an MCP tool) waited up to 120 s for
  the case run plus 60 s for the inspect on a runner that accepts and never answers; both now share ONE
  `INTERACTIVE_BUDGET_S` budget.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from src.core import runtime_settings
from src.core.config import settings
from src.rule_runner import contract as runner_contract
from tests.support import code_rule_world as cw
from tests.support import stub_code_runner as stub

UNAVAILABLE = runner_contract.Reason.RULE_WORKER_UNAVAILABLE.value
TIMEOUT = runner_contract.Reason.RULE_TIMEOUT.value
ERROR = runner_contract.Reason.RULE_ERROR.value


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


def _reply(results: List[Dict[str, Any]]) -> httpx.Response:
    identity = runner_contract.identity()
    return httpx.Response(200, json={"contract_version": identity["contract_version"],
                                     "bundle_digest": identity["bundle_digest"], "results": [results]})


def _runner(handler, *, lane: str = "batch", **limit_overrides: int):
    from src.services import code_rule_runner

    doc = stub.document("review_fn")
    sha = runner_contract.code_sha256(doc)
    limits = {name: 1 for name in runner_contract.request_limits()}
    limits.update({"input_code_points": 1_000, "wall_ms": 10_000, **limit_overrides})
    runner = code_rule_runner.HttpCodeRuleRunner(
        "http://runner", limits=limits, documents={sha: runner_contract.canonical_document_bytes(doc)}, lane=lane,
        client=httpx.Client(transport=httpx.MockTransport(handler)))
    return runner, stub.code_signal("s", doc, params={"needle": "x"})


# ── RV-1 ──────────────────────────────────────────────────────────────────────────────────────────


def test_rv1_only_worker_unavailable_is_resent_never_a_guest_reason():
    sent: List[List[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["calls"][0]["texts"]
        sent.append(texts)
        if len(sent) == 1:  # head ran (a count, a guest timeout, a guest error), tail starved by the backstop
            return _reply([{"count": 1}, {"reason": TIMEOUT}, {"reason": ERROR},
                           {"reason": UNAVAILABLE}, {"reason": UNAVAILABLE}])
        return _reply([{"count": 7} for _ in texts])

    runner, signal = _runner(handler)
    out = runner.run_batch(signal, ["a", "b", "c", "d", "e"])
    assert out == [{"count": 1}, {"reason": TIMEOUT}, {"reason": ERROR}, {"count": 7}, {"count": 7}]
    assert sent == [["a", "b", "c", "d", "e"], ["d", "e"]], "only the starved texts travel again"
    assert runner.fit == {signal["code_sha256"]: 3}, "the runner finished three texts (a reason is a finished text)"


# ── RV-2 ──────────────────────────────────────────────────────────────────────────────────────────


def test_rv2_a_runner_that_goes_down_mid_chunk_costs_one_exhausted_piece(monkeypatch):
    from src.services import code_rule_runner

    monkeypatch.setattr(code_rule_runner.time, "sleep", lambda s: None)
    calls: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["calls"][0]["texts"]
        calls.append(len(texts))
        if len(calls) == 1:  # finishes two, starves four: fit = 2, so the tail goes as two pieces of two
            return _reply([{"count": 1} if i < 2 else {"reason": UNAVAILABLE} for i in range(len(texts))])
        return httpx.Response(503, json={"error": {"code": "runner_internal"}})

    runner, signal = _runner(handler)
    out = runner.run_batch(signal, list("abcdef"))
    assert out == [{"count": 1}] * 2 + [{"reason": UNAVAILABLE}] * 4
    assert calls == [6] + [2] * code_rule_runner.ATTEMPTS, "the second piece is never sent once the first exhausted"
    assert runner.down is True


# ── RV-3 ──────────────────────────────────────────────────────────────────────────────────────────


def test_rv3_the_connect_timeout_is_capped_by_the_budget_too(monkeypatch):
    from src.services import code_rule_runner

    now = [500.0]
    monkeypatch.setattr(code_rule_runner, "_clock", lambda: now[0])
    seen: List[Dict[str, float]] = []

    def ok(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        return _reply([{"count": 1}])

    runner, signal = _runner(ok, lane="interactive")
    now[0] += code_rule_runner.INTERACTIVE_BUDGET_S - 2.0
    assert runner.run_batch(signal, ["a"]) == [{"count": 1}]
    (timeout,) = seen
    assert timeout["read"] == pytest.approx(2.0) and timeout["connect"] == pytest.approx(2.0), timeout


# ── RV-4 ──────────────────────────────────────────────────────────────────────────────────────────


async def test_rv4_the_approval_request_route_waits_one_interactive_budget(client, monkeypatch):
    """The case run waits at most `INTERACTIVE_BUDGET_S` (it was 120 s) and the inspect at most what is LEFT of the
    same budget (it was another 60 s): here the case run took 25 s of a 30 s budget, so the inspect waits 5 s."""
    from src.services import code_rule_runner

    runner = stub.install(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(code_rule_runner, "_clock", lambda: now[0])
    waits: Dict[str, Dict[str, float]] = {}

    def timed(request: httpx.Request) -> httpx.Response:
        waits[request.url.path] = dict(request.extensions["timeout"])
        if request.url.path == "/v1/run":
            now[0] += 25.0
        return runner.handle(request)

    monkeypatch.setattr(code_rule_runner, "_transport_factory", lambda: httpx.MockTransport(timed))
    doc = stub.document("budget_fn")
    filed = await cw.file_request(client, doc)
    assert filed.status_code == 202, filed.text
    budget = code_rule_runner.INTERACTIVE_BUDGET_S
    assert waits["/v1/run"]["read"] == pytest.approx(budget), waits
    assert waits["/v1/inspect"]["read"] == pytest.approx(budget - 25.0), waits
