"""040 S4 Review R2 regressions (2026-10-24): each test was written against a defect or an unpinned line the review
found, and each was run RED against the line it guards (the negative controls are in the 040 controls record,
"Review R2 (S4)").

* **R2-F1 — a host fault never reaches the verdict cache through a second code rule.** A row whose FIRST code rule
  failed with a cacheable reason (`rule_error`) and whose SECOND hit a host fault (`rule_worker_unavailable`) is
  unknown with `rule_error` — but the view's per-code-rule tally also holds the host fault, so the view must not be
  cached (040-A6: an outage is never frozen into a cache).
* **R2-F2 — only its own route files a `code_rule_approve` request.** `POST /mcp/approvals` would otherwise accept a
  forged payload (no case run, invented "case results" on the operator's card, a document never stored).
* **R2-F3 — an approved operator-code request does not "expire".** Approving it WRITES the per-hash approval; it is
  never consumed, so the caller-executed "approved but not used" expiry must not relabel it.
* **R2-F4 — the runner's own host fault in a case run is "unavailable", not "your case failed".** A case answered
  `rule_worker_unavailable` (the lane backstop) files nothing and says the runner could not run it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from sqlalchemy import select, update

from src.core import runtime_settings
from src.core.config import settings
from src.models.agent_approval import AgentApprovalRequest
from src.rule_runner import contract as runner_contract
from tests.support import code_rule_world as cw
from tests.support import rule_set_fixtures as rf
from tests.support import stub_code_runner as stub
from tests.support.random_behaviour import rng_for

API = "/api/v1"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


# ── R2-F1 ─────────────────────────────────────────────────────────────────────────────────────────


class _PerSignalRunner:
    """A runner whose answer depends only on the SIGNAL (never on code): `{signal id: result}`."""

    available = True

    def __init__(self, answers: Dict[str, Dict[str, Any]]) -> None:
        self.answers = answers
        self.calls: List[str] = []

    def run_batch(self, signal: Any, texts: Any) -> List[Dict[str, Any]]:
        self.calls.append(signal["id"])
        return [dict(self.answers[signal["id"]]) for _ in texts]

    def identity(self) -> Dict[str, Any]:
        return {"available": True}


def _two_code_rules(request) -> Any:
    from src import rule_signals
    from src.services.rule_set_store import engine_limits

    rng = rng_for(request.node.nodeid)
    first, second = stub.document("first_fn"), stub.document("second_fn", statement="second fixture")
    a, b, expr, col = rf.ident(rng, "a"), rf.ident(rng, "b"), rf.ident(rng, "e"), rf.ident(rng, "c")
    document = rf.document(rf.key(rng), rng, signals=[stub.code_signal(a, first), stub.code_signal(b, second)],
                           expressions={expr: {"any": [{"signal": a}, {"signal": b}]}},
                           outputs=[rf.fired(col, expr)])
    limits = engine_limits()
    return rule_signals.load(document, limits), limits, a, b


def _verdicts(compiled, limits, runner, n_rows: int = 3):
    from src.services import rules_evaluation

    class _Split:  # the slice of a `datasets.Dataset` the function reads: column names, length, a column
        column_names = ["t"]

        def __len__(self) -> int:
            return n_rows

        def __getitem__(self, column: str) -> List[str]:
            return [f"row {i}" for i in range(n_rows)]

    view = SimpleNamespace(id="view_r2", text_column="t", input_column="t")
    data = _Split()
    return rules_evaluation.compute_view_verdicts(view, data, compiled, limits, "all", should_stop=lambda: None,
                                                  beat=lambda _d, _n: None, runner=runner)


def test_r2_f1_a_host_fault_on_a_second_code_rule_makes_the_view_uncacheable(request):
    from src.services import rules_evaluation

    compiled, limits, a, b = _two_code_rules(request)
    runner = _PerSignalRunner({a: {"reason": "rule_error"}, b: {"reason": "rule_worker_unavailable"}})
    out = _verdicts(compiled, limits, runner)
    assert sorted(runner.calls) == sorted([a, b])  # one request per signal for the one chunk
    # the row is unknown with the FIRST failed rule's (cacheable) reason …
    assert set(out.unknown.tolist()) == {rules_evaluation.CODE_REASON_CODES["rule_error"]}
    # … but the tally carries the host fault, so nothing may be cached
    assert out.code[b]["reasons"] == {"rule_worker_unavailable": 3}
    assert out.cacheable is False


@pytest.mark.parametrize("transient", sorted(["rule_worker_unavailable", "rule_timeout", "rule_not_approved"]))
def test_r2_f1_every_uncacheable_reason_on_any_code_rule_blocks_the_cache(request, transient):
    compiled, limits, a, b = _two_code_rules(request)
    assert _verdicts(compiled, limits, _PerSignalRunner({a: {"reason": "rule_resource_limit"},
                                                         b: {"reason": transient}})).cacheable is False


def test_r2_f1_deterministic_reasons_alone_stay_cacheable(request):
    compiled, limits, a, b = _two_code_rules(request)
    out = _verdicts(compiled, limits, _PerSignalRunner({a: {"reason": "rule_error"}, b: {"count": 0}}))
    assert out.cacheable is True
    out = _verdicts(compiled, limits, _PerSignalRunner({a: {"count": 2}, b: {"count": 0}}))
    assert out.cacheable is True and out.rule_match.all()


# ── R2-F2 ─────────────────────────────────────────────────────────────────────────────────────────


async def test_r2_f2_the_generic_create_route_refuses_a_code_rule_request(client, async_session):
    doc = stub.document("forged_fn")
    sha = runner_contract.code_sha256(doc)
    forged = {"code_sha256": sha, "document": doc,
              "facts": {"case_results": [{"id": "c1", "got": {"count": 1}, "same_twice": True}]}}
    for headers in ({"X-MiStudio-Agent": "mcp"}, {}):
        r = await client.post(f"{API}/mcp/approvals", json={"tool_name": "code_rule_approve", "payload": forged},
                              headers=headers)
        assert r.status_code == 422, r.text
        assert r.json()["detail"]["code"] == "approval_filed_by_its_route"
    rows = (await async_session.execute(select(AgentApprovalRequest))).scalars().all()
    assert rows == []
    # every OTHER approvable tool is still filed through the generic route (the MCP tools' path)
    ok = await client.post(f"{API}/mcp/approvals", json={"tool_name": "start_probe_training_run",
                                                         "payload": {"model_id": 1}},
                           headers={"X-MiStudio-Agent": "mcp"})
    assert ok.status_code == 201, ok.text


# ── R2-F3 ─────────────────────────────────────────────────────────────────────────────────────────


async def test_r2_f3_an_approved_code_rule_request_never_expires(client, async_session, monkeypatch):
    stub.install(monkeypatch)
    doc = stub.document("kept_fn")
    _sha, request_id = await cw.approve(client, doc)
    # pending a caller-executed request beside it, both far older than the TTL
    other = await client.post(f"{API}/mcp/approvals", json={"tool_name": "start_probe_training_run",
                                                            "payload": {"model_id": 7}})
    other_id = other.json()["id"]
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    await async_session.execute(update(AgentApprovalRequest).values(created_at=long_ago, resolved_at=long_ago))
    await async_session.commit()
    listed = {a["id"]: a["status"] for a in (await client.get(f"{API}/mcp/approvals")).json()["approvals"]}
    assert listed[request_id] == "approved"   # the operator's per-hash approval: a record, not a spendable grant
    assert listed[other_id] == "expired"      # the TTL still applies to every caller-executed request
    state = (await client.get(f"{API}/code-rules/{_sha}")).json()
    assert state["state"] == "approved"


# ── R2-F4 ─────────────────────────────────────────────────────────────────────────────────────────


async def test_r2_f4_a_host_fault_in_a_case_run_is_unavailable_not_a_failing_case(client, async_session,
                                                                                  monkeypatch):
    runner = stub.install(monkeypatch)
    doc = stub.document("slow_fn")
    # the first instance answered, the second hit the lane backstop: NOT non-determinism, NOT a failing case
    runner.pairs = [[{"count": 2}, {"reason": "rule_worker_unavailable"}]]
    r = await cw.file_request(client, doc, headers={"X-MiStudio-Agent": "mcp"})
    assert r.status_code == 503, r.text
    assert r.json()["detail"]["code"] == "runner_refused_cases"
    assert (await async_session.execute(select(AgentApprovalRequest))).scalars().all() == []
    # a case that EXPECTS a reason the guest can produce is still judged on its answer
    runner.pairs = [[{"reason": "rule_timeout"}, {"reason": "rule_timeout"}]]
    case = {**cw.case_for(doc), "expect": {"reason": "rule_timeout"}}
    r = await cw.file_request(client, doc, headers={"X-MiStudio-Agent": "mcp"}, cases=[case])
    assert r.status_code == 202, r.text


# ── R2-M14 (a surviving mutation; miLLM 039 S3 R1's "client wait shorter than worker wait + run") ───────────────


@pytest.mark.parametrize("wall_ms", [100, 10_000, 60_000])
def test_r2_m14_the_client_waits_longer_than_the_runner_can_take(monkeypatch, wall_ms):
    """The runner answers a request within its lane backstop (`wall_ms` to start every unit) plus one unit's own
    epoch deadline (`wall_ms`): 2 × wall_ms. A client read timeout shorter than that abandons answers the runner is
    still computing and RETRIES them (up to `ATTEMPTS` copies of a chunk running at once)."""
    from src.services import code_rule_runner

    seen: List[float] = []
    monkeypatch.setattr(code_rule_runner, "http_client", lambda timeout_s: seen.append(timeout_s) or object())
    limits = {name: 1 for name in runner_contract.request_limits()}
    limits["wall_ms"] = wall_ms
    runner = code_rule_runner.HttpCodeRuleRunner("http://runner", limits=limits, documents={})
    runner._http()
    assert len(seen) == 1
    assert seen[0] >= 2 * wall_ms / 1000 + 5, "the client must outwait the runner's backstop plus one call"


# ── R2-M4 (surviving mutations: the client's reading of the runner's grammar) ────────────────────────────────


@pytest.mark.parametrize("row", [{"count": -1}, {"count": 2, "reason": "rule_error"}, {"reason": "made_up_reason"},
                                 {"count": True}, {"count": 1.0}, {}, None])
def test_r2_m4_a_reply_outside_the_runner_grammar_is_a_host_fault_never_a_verdict(row):
    """A runner that breaks its own grammar (a negative count, an unknown reason, both, neither) is a host fault:
    `rule_worker_unavailable` — never cached, never a count or a reason the operator's code did not produce."""
    import httpx

    from src.services import code_rule_runner

    identity = runner_contract.identity()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "contract_version": identity["contract_version"], "bundle_digest": identity["bundle_digest"],
        "results": [[row, {"count": 3}]]}))
    doc = stub.document("grammar_fn")
    sha = runner_contract.code_sha256(doc)
    limits = {name: 1 for name in runner_contract.request_limits()}
    runner = code_rule_runner.HttpCodeRuleRunner("http://runner", limits=limits,
                                                 documents={sha: runner_contract.canonical_document_bytes(doc)},
                                                 client=httpx.Client(transport=transport))
    out = runner.run_batch(stub.code_signal("s", doc), ["a", "b"])
    assert out == [{"reason": "rule_worker_unavailable"}, {"count": 3}]


# ── R2-M42 / R2-M44 (surviving mutations: the task's runner comes from what its REQUEST recorded) ───────────────


def _code_compiled(request):
    compiled, _limits, _a, _b = _two_code_rules(request)
    return compiled


def _enable(monkeypatch, **overrides):
    from src.services import code_rule_runner

    monkeypatch.setattr(settings, "code_rule_runner_url", "http://stub-rule-runner:8080")
    monkeypatch.setattr(code_rule_runner, "_transport_factory", None)
    runtime_settings.apply({"code_rules_enabled": True, **overrides})


def test_r2_m42_the_task_runs_under_the_limits_its_request_recorded(monkeypatch, request):
    """The cache key and the stored `engine_digest` are the REQUEST's (`row.engine_digest`, limits included). A task
    that ran with the limits in force NOW would file verdicts computed under other limits under that key."""
    from src.services import code_rule_runner, code_rules, rules_evaluation

    _enable(monkeypatch, code_rule_wall_ms=20_000)
    monkeypatch.setattr(code_rules, "documents_for_sync", lambda db, shas: {})
    recorded = {**code_rule_runner.identity(), "limits": {**code_rule_runner.code_limits(), "wall_ms": 7_000}}
    runner, batch = rules_evaluation._task_runner(SimpleNamespace(), {"runner": recorded}, _code_compiled(request))
    assert isinstance(runner, code_rule_runner.HttpCodeRuleRunner)
    assert runner.limits == recorded["limits"] and runner.limits["wall_ms"] == 7_000
    assert runner.identity()["limits"] == recorded["limits"]


def test_r2_m44_switched_off_after_the_request_is_a_host_fault_never_cached_pending(monkeypatch, request):
    """Recorded WITH a runner, run after the operator switched code rules off: every text is a host fault (unknown,
    never cached). Answering `runner_unavailable` (pending) instead would cache a no-runner verdict under the
    request's runner identity — the inverse of Review R1-F8."""
    from src.services import code_rule_runner, rules_evaluation

    _enable(monkeypatch)
    recorded = {**code_rule_runner.identity(), "limits": code_rule_runner.code_limits()}
    runtime_settings.apply({"code_rules_enabled": False})
    compiled = _code_compiled(request)
    runner, _batch = rules_evaluation._task_runner(SimpleNamespace(), {"runner": recorded}, compiled)
    assert runner.available is True
    from src.services.rule_set_store import engine_limits

    out = _verdicts(compiled, engine_limits(), runner)
    assert set(out.unknown.tolist()) == {rules_evaluation.CODE_REASON_CODES["rule_worker_unavailable"]}
    assert out.cacheable is False


# ── R2-M30 / R2-M46 (surviving mutations: the switch ships off, and no agent header turns it on) ──────────────


def test_r2_m30_code_rules_ship_off_with_no_runner():
    """A fresh deployment runs no operator code: `code_rules_enabled` defaults to False and no runner URL is set
    (k8s sets the URL; the switch stays the operator's, after the post-deploy runner check)."""
    from src.core.config import Settings
    from src.services import code_rule_runner

    fresh = Settings(_env_file=None)
    assert fresh.code_rules_enabled is False
    assert code_rule_runner.enabled(Settings(_env_file=None, code_rule_runner_url="http://runner:8080")) is False


@pytest.mark.parametrize("value", ["", "mcp", "agent:someone"])
async def test_r2_m46_any_agent_header_value_is_refused_on_environment_writes(client, value):
    """D-21: `X-MiStudio-Agent` with ANY value — an empty one included — is refused on both environment writes."""
    headers = {"X-MiStudio-Agent": value}
    put = await client.put(f"{API}/environment/code_rules_enabled", json={"value": True}, headers=headers)
    assert put.status_code == 403 and put.json()["detail"]["code"] == "environment_operator_only"
    delete = await client.delete(f"{API}/environment/code_rules_enabled", headers=headers)
    assert delete.status_code == 403


# ── R2-M45 (surviving mutation: the runner cannot rewrite its own code or guest) ─────────────────────────────


def test_r2_m45_the_runner_user_owns_nothing_in_the_image():
    """`/app` (the package, the contract, the fetched and precompiled guest) is root-owned and only readable by the
    runner's uid: even with the read-only root filesystem removed, the runner process could not replace its own
    bundle or code. Pinned on the Dockerfile's instructions, in order."""
    from tests.unit.test_rule_runner_image import _instructions

    steps = _instructions()
    user_at = next(i for i, s in enumerate(steps) if s.startswith("USER "))
    assert steps[user_at] == "USER 10001"
    chowns = [s for s in steps if "chown" in s]
    assert chowns and all("chown -R root:root /app" in s for s in chowns), chowns
    assert all("10001" not in s and "runner:" not in s for s in chowns)
    assert any("chown -R root:root /app" in s for s in steps[:user_at])
    assert not any(s.startswith(("COPY", "ADD")) and "--chown" in s for s in steps)
