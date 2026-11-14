"""040 S4 Review R2 debt D1 – D5 (2026-10-24): each test was run RED against the line it guards (the negative
controls are in the 040 controls record, "Debt D1 – D5").

* **D1 — the lane backstop.** The runner bounds a request's lane wait by one deadline (`wall_ms` from the request's
  start), so a chunk needing more than `slots × wall_ms` answers its TAIL `rule_worker_unavailable` every time. The
  client re-sends only those texts while a request makes progress, and sends that code rule no more texts per
  request than the runner just finished.
* **D2 — a limit above the runner's ceiling.** Refused at write and at start, naming the limit, the ceiling and the
  runner's variable; the backend's ceiling table is DERIVED here from the runner's own `RunnerConfig` read with the
  manifest's environment, so the two cannot drift.
* **D3 — the API path.** One attempt per request, no sleep, inside one `INTERACTIVE_BUDGET_S` budget; no request
  starts past it and none waits longer than what is left of it.
* **D4 — a runner down for a whole bulk evaluation.** After one chunk exhausted its attempts, later chunks call
  nothing and sleep nothing.
* **D5 — memory.** Every lane slot at the manifest's memory ceiling, plus a reserve for the process, fits the pod's
  memory limit; a text longer than `input_code_points` never travels (answered here as the runner answers it).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
import yaml

from src.core import runtime_settings
from src.core.config import CODE_RULE_RUNNER_CEILINGS, Settings, settings
from src.rule_runner import contract as runner_contract
from tests.support import code_rule_world as cw
from tests.support import rule_set_fixtures as rf
from tests.support import rules_world as rw
from tests.support import stub_code_runner as stub
from tests.support.random_behaviour import rng_for

API = "/api/v1"
REPO = Path(__file__).resolve().parents[3]
UNAVAILABLE = runner_contract.Reason.RULE_WORKER_UNAVAILABLE.value


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


def _limits(**overrides: int) -> Dict[str, int]:
    out = {name: 1 for name in runner_contract.request_limits()}
    out["input_code_points"] = 1_000
    out["wall_ms"] = 10_000
    out.update(overrides)
    return out


def _doc_and_signal(name: str = "debt_fn"):
    doc = stub.document(name)
    sha = runner_contract.code_sha256(doc)
    return {sha: runner_contract.canonical_document_bytes(doc)}, stub.code_signal("s", doc, params={"needle": "x"})


def _reply(results: List[Dict[str, Any]]) -> httpx.Response:
    identity = runner_contract.identity()
    return httpx.Response(200, json={"contract_version": identity["contract_version"],
                                     "bundle_digest": identity["bundle_digest"], "results": [results]})


def _runner(handler, *, lane: str = "batch", **limit_overrides: int):
    from src.services import code_rule_runner

    documents, signal = _doc_and_signal()
    runner = code_rule_runner.HttpCodeRuleRunner("http://runner", limits=_limits(**limit_overrides),
                                                 documents=documents, lane=lane,
                                                 client=httpx.Client(transport=httpx.MockTransport(handler)))
    return runner, signal


class _Capacity:
    """A runner that finishes the first ``k`` texts of every request and starves the rest (the lane backstop's
    shape, deterministic): records the texts each request carried."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.sent: List[List[str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["calls"][0]["texts"]
        self.sent.append(texts)
        return _reply([{"count": len(t)} if i < self.k else {"reason": UNAVAILABLE} for i, t in enumerate(texts)])


# ── D1 ────────────────────────────────────────────────────────────────────────────────────────────


def test_d1_a_starved_tail_is_resent_and_the_next_chunk_is_sized_to_what_ran():
    capacity = _Capacity(40)
    runner, signal = _runner(capacity)
    texts = [f"t{i:03d}" + "x" * (i % 7) for i in range(100)]
    out = runner.run_batch(signal, texts)
    assert out == [{"count": len(t)} for t in texts], "every text of the chunk ran: none left on the backstop"
    # the first request carries the whole chunk; only the starved tail is re-sent, at most what ran (40) at a time
    assert [len(s) for s in capacity.sent] == [100, 40, 20]
    assert [t for s in capacity.sent for t in s[:40]] == texts, "each text ran exactly once, in order"
    capacity.sent.clear()
    again = runner.run_batch(signal, texts[:90])
    assert again == [{"count": len(t)} for t in texts[:90]]
    assert [len(s) for s in capacity.sent] == [40, 40, 10], "the measured fit sizes the next chunk up front"
    assert runner.fit == {signal["code_sha256"]: 40}


def test_d1_a_request_that_runs_nothing_is_not_resent():
    """No progress is a host fault that stands (pending, never cached) — never an endless re-send."""
    capacity = _Capacity(0)
    runner, signal = _runner(capacity)
    out = runner.run_batch(signal, ["a", "b", "c"])
    assert out == [{"reason": UNAVAILABLE}] * 3
    assert len(capacity.sent) == 1 and runner.calls == 1


def test_d1_against_the_runners_own_lane_and_backstop():
    """The runner's REAL `Lane` with the backstop `Runner.execute` applies (one deadline per request, `wall_ms`
    from its start; a unit with no slot by then answers `rule_worker_unavailable`), slow units on two slots: the
    first request starves its tail, and every text still gets a count through `rule_signals.evaluate_many`."""
    from src import rule_signals
    from src.rule_runner.lanes import Lane, LaneFull
    from src.services.rule_set_store import engine_limits

    lane = Lane("batch", 2)
    executed: List[str] = []
    lock = threading.Lock()

    def unit(text: str) -> Dict[str, Any]:
        time.sleep(0.01)
        with lock:
            executed.append(text)
        return {"count": text.count("x")}

    sizes: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        texts = body["calls"][0]["texts"]
        sizes.append(len(texts))
        deadline = time.monotonic() + body["limits"]["wall_ms"] / 1000  # as `Runner.execute`
        pending: List[Any] = []
        for text in texts:
            try:
                pending.append(lane.submit(lambda t=text: unit(t), deadline - time.monotonic()))
            except LaneFull:
                pending.append({"reason": UNAVAILABLE})
        return _reply([p.result() if isinstance(p, Future) else p for p in pending])

    try:
        rng = rng_for("d1-real-lane")
        doc = stub.document("lane_fn")
        sid, expr, col = rf.ident(rng, "s"), rf.ident(rng, "e"), rf.ident(rng, "c")
        document = rf.document(rf.key(rng), rng, signals=[stub.code_signal(sid, doc)],
                               expressions={expr: {"signal": sid}}, outputs=[rf.fired(col, expr)])
        compiled = rule_signals.load(document, engine_limits())
        runner, _signal = _runner(handler, wall_ms=100)
        runner.documents = {runner_contract.code_sha256(doc): runner_contract.canonical_document_bytes(doc)}
        rows = [[{"role": "assistant", "content": "x" * (1 + i % 3) + f" row {i}"}] for i in range(60)]
        verdicts = rule_signals.evaluate_many(compiled, engine_limits(), rows, runner=runner, batch_texts=256)
    finally:
        lane.shutdown()
    assert sizes[0] == 60 and len(sizes) >= 2, f"the backstop must have cut the first request short: {sizes}"
    assert len(executed) == len(rows) == len(set(executed)), "each text ran once, none twice, none lost"
    assert not any(v.unknown for v in verdicts), [v.reason for v in verdicts if v.unknown]


async def test_d1_through_the_task_no_row_is_left_on_the_backstop(client, async_session, data_dir, monkeypatch,
                                                                   request):
    """The REAL route and task: a runner that finishes 3 texts per request. Before D1 every text past the third read
    `rule_worker_unavailable` (unknown, never cached) on every run."""
    from src.services import code_rule_runner

    runner = stub.install(monkeypatch)
    seen: List[int] = []

    def starving(req: httpx.Request) -> httpx.Response:
        response = runner.handle(req)
        if req.url.path != "/v1/run":
            return response
        body = response.json()
        seen.append(len(body["results"][0]))
        body["results"][0] = [r if i < 3 else {"reason": UNAVAILABLE} for i, r in enumerate(body["results"][0])]
        return httpx.Response(200, json=body)

    monkeypatch.setattr(code_rule_runner, "_transport_factory", lambda: httpx.MockTransport(starving))
    seed = request.node.nodeid
    key = rf.key(rng_for(seed + ":key"))
    rule_set, doc, _sid, _expr = cw.code_set(seed, key)
    world = await rw.build(client, async_session, data_dir, seed, rule_set=rule_set, key=key)
    await cw.approve(client, doc)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    read = (await client.get(f"{rw.API}/rules-evaluations/{out['evaluation_id']}")).json()
    assert "result" in read, (out, read)
    view = read["result"]["views"][0]
    assert UNAVAILABLE not in view["unknown"]["by_reason"], view["unknown"]
    big = next(i for i, n in enumerate(seen) if n > 3)  # the view's chunk, starved past its third text
    assert seen[big + 1:] and max(seen[big + 1:]) == 3, seen


# ── D2 ────────────────────────────────────────────────────────────────────────────────────────────


def _manifest_runner_env() -> Dict[str, str]:
    docs = [d for d in yaml.safe_load_all((REPO / "k8s" / "base" / "rule-runner.yaml").read_text()) if d]
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    (container,) = deployment["spec"]["template"]["spec"]["containers"]
    return {e["name"]: str(e["value"]) for e in container.get("env") or []}


def _deployed_config():
    from src.rule_runner.config import RunnerConfig

    return RunnerConfig.from_env(_manifest_runner_env())


def test_d2_the_ceiling_table_is_the_deployed_runners_own():
    """Derived from the authority: the runner's `RunnerConfig` read with the environment the manifest gives it. A
    manifest ceiling changed without the table (or the table without the manifest) is red; so is a variable name the
    runner does not read."""
    from src.rule_runner.config import RunnerConfig
    from src.services.code_rule_runner import LIMIT_SETTINGS

    deployed = _deployed_config()
    assert set(CODE_RULE_RUNNER_CEILINGS) == set(LIMIT_SETTINGS.values())
    for name, setting in LIMIT_SETTINGS.items():
        ceiling, variable = CODE_RULE_RUNNER_CEILINGS[setting]
        assert getattr(deployed, f"max_{name}") == ceiling, setting
        assert getattr(RunnerConfig.from_env({variable: "12345"}), f"max_{name}") == 12345, variable


@pytest.mark.parametrize("setting", sorted(CODE_RULE_RUNNER_CEILINGS))
async def test_d2_a_limit_above_the_ceiling_is_refused_at_write_naming_it(client, setting):
    ceiling, variable = CODE_RULE_RUNNER_CEILINGS[setting]
    over = await client.put(f"{API}/environment/{setting}", json={"value": ceiling + 1})
    assert over.status_code == 422, over.text
    detail = over.json()["detail"]
    assert setting in detail and str(ceiling) in detail and variable in detail, detail
    at = await client.put(f"{API}/environment/{setting}", json={"value": ceiling})
    assert at.status_code == 200, at.text


@pytest.mark.parametrize("setting", sorted(CODE_RULE_RUNNER_CEILINGS))
def test_d2_a_limit_above_the_ceiling_is_refused_at_start(setting):
    ceiling, variable = CODE_RULE_RUNNER_CEILINGS[setting]
    with pytest.raises(ValueError) as exc:
        Settings(_env_file=None, **{setting: ceiling + 1})
    assert variable in str(exc.value) and str(ceiling) in str(exc.value)
    assert getattr(Settings(_env_file=None, **{setting: ceiling}), setting) == ceiling


# ── D3 ────────────────────────────────────────────────────────────────────────────────────────────


def test_d3_the_interactive_lane_makes_one_attempt_and_never_sleeps(monkeypatch):
    from src.services import code_rule_runner

    slept: List[float] = []
    monkeypatch.setattr(code_rule_runner.time, "sleep", lambda s: slept.append(s))
    calls: List[int] = []

    def down(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    runner, signal = _runner(down, lane="interactive")
    assert runner.run_batch(signal, ["a", "b"]) == [{"reason": UNAVAILABLE}] * 2
    assert len(calls) == 1 and slept == []
    batch, signal = _runner(down, lane="batch")
    calls.clear()
    batch.run_batch(signal, ["a"])
    assert len(calls) == code_rule_runner.ATTEMPTS, "the bulk lane keeps its bounded retry"


def test_d3_one_budget_bounds_every_request_of_an_api_call(monkeypatch):
    from src.services import code_rule_runner

    now = [1000.0]
    monkeypatch.setattr(code_rule_runner, "_clock", lambda: now[0])
    timeouts: List[float] = []

    def ok(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"]["read"])
        texts = json.loads(request.content)["calls"][0]["texts"]
        return _reply([{"count": 1} for _ in texts])

    runner, signal = _runner(ok, lane="interactive", wall_ms=60_000)
    now[0] += code_rule_runner.INTERACTIVE_BUDGET_S - 4.0
    assert runner.run_batch(signal, ["a"]) == [{"count": 1}]
    assert timeouts == [pytest.approx(4.0)], "a request waits no longer than what is left of the budget"
    now[0] += 4.0
    assert runner.run_batch(signal, ["b"]) == [{"reason": UNAVAILABLE}]
    assert len(timeouts) == 1, "no request starts once the budget is spent"


async def test_d3_the_single_text_route_calls_a_down_runner_once(client, async_session, data_dir, monkeypatch,
                                                                 request):
    """The API path end to end: `POST /rule-signals/evaluate` with the runner answering 503 makes ONE request (it
    made `ATTEMPTS`, each up to 2 × wall + 30 s, in an API threadpool thread) and reads pending."""
    runner = stub.install(monkeypatch)
    seed = request.node.nodeid
    key = rf.key(rng_for(seed + ":key"))
    rule_set, doc, _sid, _expr = cw.code_set(seed, key)
    world = await rw.build(client, async_session, data_dir, seed, rule_set=rule_set, key=key)
    await cw.approve(client, doc)
    runner.status = 503
    before = len(runner.runs())
    out = await client.post(f"{rw.API}/rule-signals/evaluate", json={"behaviour_key": world.key, "text": "x x"})
    assert out.status_code == 200, out.text
    assert out.json()["unknown"] is True and out.json()["reason"] == UNAVAILABLE
    assert len(runner.runs()) - before == 1


# ── D4 ────────────────────────────────────────────────────────────────────────────────────────────


def test_d4_after_one_exhausted_chunk_the_rest_call_and_sleep_nothing(monkeypatch):
    from src.services import code_rule_runner

    slept: List[float] = []
    monkeypatch.setattr(code_rule_runner.time, "sleep", lambda s: slept.append(s))
    calls: List[int] = []

    def down(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": {"code": "runner_internal"}})

    runner, signal = _runner(down)
    for _chunk in range(5):
        assert runner.run_batch(signal, ["a", "b"]) == [{"reason": UNAVAILABLE}] * 2
    assert len(calls) == code_rule_runner.ATTEMPTS
    assert len(slept) == code_rule_runner.ATTEMPTS - 1
    assert runner.down is True


async def test_d4_through_the_task_a_down_runner_costs_one_chunk(client, async_session, data_dir, monkeypatch,
                                                                 request):
    """The REAL task with 2 texts per runner request (several chunks): a runner down throughout is called for the
    first chunk only (`ATTEMPTS`), not `ATTEMPTS` per chunk."""
    from src.services import code_rule_runner

    runner = stub.install(monkeypatch, code_rule_batch_texts=2)
    seed = request.node.nodeid
    key = rf.key(rng_for(seed + ":key"))
    rule_set, doc, _sid, _expr = cw.code_set(seed, key)
    world = await rw.build(client, async_session, data_dir, seed, rule_set=rule_set, key=key)
    await cw.approve(client, doc)
    runner.status = 503
    before = len(runner.runs())
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await client.get(f"{rw.API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    assert view["unknown"]["by_reason"][UNAVAILABLE] == len(rw.EVAL_ROWS)
    assert len(runner.runs()) - before == code_rule_runner.ATTEMPTS


# ── D5 ────────────────────────────────────────────────────────────────────────────────────────────

#: The runner process's own memory beside its instances: miDataworks 018 R2 Q2 measured a 337 MiB peak; reserved.
PROCESS_RESERVE_BYTES = 512 << 20
_UNITS = {"Ki": 1 << 10, "Mi": 1 << 20, "Gi": 1 << 30}


def _bytes(quantity: str) -> int:
    for suffix, scale in _UNITS.items():
        if quantity.endswith(suffix):
            return int(quantity[: -len(suffix)]) * scale
    return int(quantity)


def test_d5_every_slot_at_the_memory_ceiling_fits_the_pod_limit():
    docs = [d for d in yaml.safe_load_all((REPO / "k8s" / "base" / "rule-runner.yaml").read_text()) if d]
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    (container,) = deployment["spec"]["template"]["spec"]["containers"]
    limit = _bytes(container["resources"]["limits"]["memory"])
    deployed = _deployed_config()
    instances = deployed.interactive_slots + deployed.batch_slots
    assert instances * deployed.max_memory_bytes + PROCESS_RESERVE_BYTES <= limit, (
        f"{instances} × {deployed.max_memory_bytes} + reserve exceeds the pod limit {limit}")
    assert deployed.max_memory_bytes >= runner_contract.floor()["memory_bytes"], "the ceiling must admit the floor"


def test_d5_compose_runs_the_runner_under_the_same_memory_ceiling():
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    env = compose["services"]["rule-runner"]["environment"]
    assert str(env["RUNNER_MAX_MEMORY_BYTES"]) == _manifest_runner_env()["RUNNER_MAX_MEMORY_BYTES"]


def test_d5_a_text_over_the_input_limit_never_travels():
    capacity = _Capacity(10_000)
    runner, signal = _runner(capacity, input_code_points=8)
    out = runner.run_batch(signal, ["short", "x" * 9, "x" * 8, "\ud800" * 9])
    too_large = {"reason": runner_contract.Reason.RULE_INPUT_TOO_LARGE.value}
    assert out == [{"count": 5}, too_large, {"count": 8}, too_large]
    assert capacity.sent == [["short", "x" * 8]]
