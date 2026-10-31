"""FR-31 hostile tests through miStudio's DEPLOYED path and the REAL runner (Feature 040 S4; FTDD Appendix C.9;
FTASKS 25.9; ENH-006 D48).

Every vendored hostile vector (`runner-cases-v1.json` `hostile`, 17 vectors over classes 5 – 8 and 10 – 16) is
filed for approval through `POST /code-rules/{sha}/approval-requests` (its own expectation as its case, run twice
at the floor), approved by the operator, and evaluated through `POST /probe-monitors/rule-signals/evaluate` against
a runner SUBPROCESS started like its image (no database, no secret). Each must end as the vector says: the reason
(unknown, never "did not fire") or the clamped count (fired, ``spans: None``).

SENTINELS: the runner process is started holding a fresh random value in its environment and in a file in its
working directory, and a code rule whose SOURCE carries a third one is run (and cached in the runner) before the
vectors; the evaluated text carries a fourth. None — nor any 12-character piece of one — may appear in any reply,
any stored row or any log record (`caplog`, and the runner's own stdout/stderr), across a single-text evaluation, a
bulk evaluation's chunk and a runner fault.

⚠ D48 — NOT WRITTEN HERE: the escape forms FR-31 names (a socket; a host file such as /etc/passwd or the sentinel
file; another behaviour's data — the cached rule or the environment sentinel) are guest code reaching files, the
environment, sockets or the clock past the prelude: hostile classes 1, 2, 3, 4, 9 (and the guest forms of 17 – 19).
The operator writes them; until then each is reported PENDING below (a named skip), mirrored by
`rule_runner.contract.PENDING_HOSTILE_CLASSES`, and NOTHING here reports them covered. The host defences they will
exercise are pinned at host level by `rule_runner/tests/test_wasi_table.py` (no preopened directory, an empty
environment, a fixed clock, trapping `sock_*` / `proc_raise`, markers once and in order) and the deployment's by
`test_rule_runner_manifest.py` (Property 1 and the NetworkPolicy of Property 2).

Runs in CI's `rule-runner-tests` job (`MISTUDIO_REQUIRE_RUNNER=1`); elsewhere it SKIPS LOUDLY without the bundle.
"""
from __future__ import annotations

import logging
import random
import string

import pytest
from sqlalchemy import select

from src.core import runtime_settings
from src.core.config import settings
from src.rule_runner import contract as runner_contract
from tests.support import code_rule_world as cw
from tests.support import real_rule_runner as real
from tests.support import rule_set_fixtures as rf
from tests.support.random_behaviour import rng_for

API = "/api/v1"
CASES = runner_contract.load_cases()
HOSTILE = CASES["hostile"]


def _sentinel() -> str:
    return "".join(random.SystemRandom().choice(string.ascii_letters) for _ in range(24))


SENTINELS = {"env": _sentinel(), "file": _sentinel(), "cached_rule": _sentinel(), "text": _sentinel()}


def _clean(blob: str) -> bool:
    for value in SENTINELS.values():
        if any(value[i:i + 12] in blob for i in range(len(value) - 11)):
            return False
    return True


#: SQL echo (`core/database.py`: `echo=settings.is_development`) logs statement PARAMETERS — a stored document's
#: source among them — in a development process only; every deployed container sets ENVIRONMENT=production
#: (`test_sql_echo_is_off_in_every_deployed_container`). Those records are excluded here, nothing else is.
DEV_ONLY_LOGGERS = ("sqlalchemy.engine",)


def logged(caplog) -> str:
    return " ".join(r.getMessage() for r in caplog.records if not r.name.startswith(DEV_ONLY_LOGGERS))


def test_sql_echo_is_off_in_every_deployed_container():
    from pathlib import Path

    import yaml

    from src.core.config import Settings

    assert Settings(_env_file=None, environment="production").is_development is False
    for doc in yaml.safe_load_all((Path(__file__).resolve().parents[3] / "k8s" / "base" / "backend.yaml").read_text()):
        if not doc or doc.get("kind") != "Deployment":
            continue
        for container in doc["spec"]["template"]["spec"]["containers"]:
            env = {e["name"]: e.get("value") for e in container.get("env") or []}
            assert env.get("ENVIRONMENT") == "production", container["name"]


def vector_holds(expect: dict, got: dict) -> bool:
    """THE hostile assertion (shared with `test_harness_sees_breaches.py`): what the runner answered for one text."""
    if "count" in expect:
        return got.get("count") == expect["count"] and "reason" not in got
    return got.get("reason") == expect.get("reason") and "count" not in got


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    runtime_settings.reset()
    yield
    runtime_settings.reset()


@pytest.fixture
def runner(monkeypatch):
    with real.RealRunner(env={"RUNNER_SENTINEL": SENTINELS["env"]},
                         files={"sentinel.txt": SENTINELS["file"]}) as running:
        real.point_backend_at(monkeypatch, running)
        yield running
    assert _clean(running.output[0].decode("utf-8", "replace") + running.output[1].decode("utf-8", "replace")), \
        "a sentinel reached the runner's own output"


def _rule_set(request, document: dict, function: str, params: dict) -> tuple:
    seed = request.node.nodeid + function
    rng = rng_for(seed)
    sid, expr, col = rf.ident(rng, "k"), rf.ident(rng, "e"), rf.ident(rng, "c")
    signal = {"id": sid, "kind": "code_rule", "code_sha256": runner_contract.code_sha256(document),
              "runner_contract": runner_contract.FORMAT, "function": function, "params": dict(params),
              "outputs": [{"name": "hits", "type": "int"}]}
    doc = rf.document(rf.key(rng), rng, signals=[signal], expressions={expr: {"signal": sid}},
                      outputs=[rf.fired(col, expr)])
    return doc, sid, expr


async def _approve(client, document: dict, case: dict) -> None:
    sha = runner_contract.code_sha256(document)
    filed = await client.post(f"{API}/code-rules/{sha}/approval-requests", headers={"X-MiStudio-Agent": "mcp"},
                              json={"document": document, "cases": [case]})
    assert filed.status_code == 202, filed.text
    if filed.json()["status"] == "approved":  # this document's hash is already approved here
        return
    done = await client.post(f"{API}/mcp/approvals/{filed.json()['approval_request_id']}/approve")
    assert done.status_code == 200, done.text


async def _cache_a_sentinel_rule(client, request) -> None:
    """A rule whose SOURCE carries a sentinel, run (so the runner caches its document) before the vectors."""
    document = {"runner_contract": runner_contract.FORMAT,
                "source": f"_S = '{SENTINELS['cached_rule']}'\ndef f(text, params):\n    return 0\n",
                "functions": [{"name": "f", "output": {"type": "count"}}], "params_schema": {"type": "object"},
                "derived_from": {"sources": ["tests"], "statement": "sentinel"}}
    await _approve(client, document, {"id": "s", "function": "f", "params": {}, "text": "x",
                                      "expect": {"count": 0}})
    doc, _, _ = _rule_set(request, document, "f", {})
    out = await client.post(f"{API}/probe-monitors/rule-signals/evaluate", json={"rule_set": doc, "text": "x"})
    assert out.status_code == 200 and out.json()["unknown"] is False


async def test_every_vendored_hostile_vector_through_evaluate_rule_signals(client, async_session, runner, request,
                                                                          caplog):
    caplog.set_level(logging.DEBUG)
    await _cache_a_sentinel_rule(client, request)
    covered = set()
    for case in HOSTILE:
        document = CASES["documents"][case["doc"]]
        await _approve(client, document, {"id": case["id"], "function": case["function"], "params": case["params"],
                                          "text": case["text"], "expect": case["expect"]})
        doc, sid, expr = _rule_set(request, document, case["function"], case["params"])
        text = case["text"] + " " + SENTINELS["text"] if case["number"] == 8 else case["text"]
        out = await client.post(f"{API}/probe-monitors/rule-signals/evaluate", json={"rule_set": doc, "text": text})
        assert out.status_code == 200, (case["id"], out.text)
        body = out.json()
        got = body["code"][sid]
        assert vector_holds(case["expect"], got), (case["id"], got)
        if "reason" in case["expect"]:
            assert (body["unknown"], body["reason"], body["rule_match"]) == (True, case["expect"]["reason"], False)
        else:
            assert body["expressions"][expr]["fires"] is (case["expect"]["count"] >= 1)
            fired = [f for f in body["fired"] if f["signal_id"] == sid]
            assert all(f["spans"] is None for f in fired), "a fired code rule carries spans: null"
        assert _clean(out.text), case["id"]
        covered.add(case["number"])
    assert covered == set(range(1, 20)) - runner_contract.PENDING_HOSTILE_CLASSES
    assert _clean(logged(caplog))


async def test_the_runner_fault_path_leaks_nothing(client, async_session, monkeypatch, request, caplog):
    """A runner that dies: per-row `rule_worker_unavailable` (a host fault), no sentinel in any log."""
    caplog.set_level(logging.DEBUG)
    with real.RealRunner(env={"RUNNER_SENTINEL": SENTINELS["env"]}) as running:
        real.point_backend_at(monkeypatch, running)
        document = CASES["documents"]["count_tokens"]
        await _approve(client, document, {"id": "t", "function": "f", "params": {}, "text": "one two",
                                          "expect": {"count": 2}})
        doc, sid, _ = _rule_set(request, document, "f", {})
        running.stop()
        out = await client.post(f"{API}/probe-monitors/rule-signals/evaluate",
                                json={"rule_set": doc, "text": "x " + SENTINELS["text"]})
    assert out.status_code == 200
    assert out.json()["reason"] == "rule_worker_unavailable" and out.json()["unknown"] is True
    assert _clean(out.text) and _clean(logged(caplog))


async def test_a_bulk_chunk_leaks_nothing_and_counts_like_the_vectors(client, async_session, monkeypatch, request,
                                                                      runner, caplog):
    """A rules evaluation whose rows all carry the text sentinel, through one real chunk."""
    from tests.support import rules_world as rw

    caplog.set_level(logging.DEBUG)
    seed = request.node.nodeid
    document = CASES["documents"]["count_tokens"]
    await _approve(client, document, {"id": "t", "function": "f", "params": {}, "text": "one two",
                                      "expect": {"count": 2}})
    key = rf.key(rng_for(seed + ":key"))
    rng = rng_for(seed)
    sid, expr, col = rf.ident(rng, "k"), rf.ident(rng, "e"), rf.ident(rng, "c")
    signal = {"id": sid, "kind": "code_rule", "code_sha256": runner_contract.code_sha256(document),
              "runner_contract": runner_contract.FORMAT, "function": "f", "params": {},
              "outputs": [{"name": "hits", "type": "int"}]}
    rule_set = rf.document(key, rng, signals=[signal], expressions={expr: {"signal": sid}},
                           outputs=[rf.fired(col, expr)])
    text = {k: f"{v} {SENTINELS['text']}" for k, v in rw.TEXT.items()}
    world = await rw.build(client, async_session, settings.data_dir, seed, rule_set=rule_set, key=key, text=text)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert out["outcome"]["status"] == "completed"
    read = await client.get(f"{API}/probe-monitors/rules-evaluations/{out['evaluation_id']}")
    view = read.json()["result"]["views"][0]
    assert view["unknown"]["n"] == 0 and view["code_rules"][0]["rows_fired"] == 2 * len(rw.EVAL_ROWS)
    assert _clean(read.text) and _clean(logged(caplog))


async def test_the_deployed_path_counts_exactly_the_text_sent(client, async_session, runner, request):
    """Control HB3 survived the vectors alone: a client appending to the text still met every vector. The vendored
    conformance documents through the deployed path: the count is the one the contract states for exactly that text."""
    for case in [c for c in CASES["conformance"] if c["id"] in ("conf.count_tokens", "conf.params", "conf.emoji")]:
        document = CASES["documents"][case["doc"]]
        await _approve(client, document, {"id": case["id"], "function": case["function"], "params": case["params"],
                                          "text": case["text"], "expect": case["expect"]})
        doc, sid, expr = _rule_set(request, document, case["function"], case["params"])
        body = (await client.post(f"{API}/probe-monitors/rule-signals/evaluate",
                                  json={"rule_set": doc, "text": case["text"]})).json()
        assert body["code"][sid] == case["expect"], case["id"]
        assert body["expressions"][expr]["count"] == case["expect"]["count"], case["id"]


def test_fixture_1_luhn_through_the_runner_equals_the_vendored_016_luhn_vectors(runner):
    """Equivalence (C.9): 018's fixture 1 (Luhn by algorithm, on runs of 12+ digits) through miStudio's runner agrees
    with miStudio's engine's `luhn@1` vectors wherever the fixture reads the value."""
    import json
    import re

    import httpx

    from src.rule_signals import document as rs_document

    vendored = json.loads(rs_document.CASES_PATH.read_text(encoding="utf-8"))
    vectors = [c for c in vendored["validators"]["luhn@1"]["cases"] if "value" in c["input"]]
    usable = [c for c in vectors if re.fullmatch(r"[0-9][0-9 -]{10,30}[0-9]", c["input"]["value"])
              and sum(ch.isdigit() for ch in c["input"]["value"]) >= 12]
    assert len(usable) >= 2, "the probe must see both a valid and an invalid vector"
    document = CASES["documents"]["fixture1_luhn"]
    floor = runner_contract.floor()
    body = {"contract": runner_contract.FORMAT, "mode": "stored", "lane": "batch",
            "limits": {**{k: floor[k] for k in runner_contract.request_limits() if k in floor}, "wall_ms": 10_000},
            "document": {"code_sha256": runner_contract.code_sha256(document),
                         "canonical_utf8": runner_contract.canonical_document_bytes(document).decode("utf-8")},
            "calls": [{"function": "check", "params": {}, "texts": [c["input"]["value"] for c in usable]}]}
    results = httpx.post(runner.url + "/v1/run", json=body, timeout=60).json()["results"][0]
    assert [r.get("count") for r in results] == [1 if c["expect"]["valid"] else 0 for c in usable]


# ── D48: the escape forms FR-31 names are the OPERATOR's to write; reported PENDING, never covered ────────────────


@pytest.mark.parametrize("form,classes", [
    ("socket", {1}), ("host_file", {2}), ("another_behaviours_data", {2, 3}), ("process", {4}), ("clock", {9}),
    ("replayed_or_reordered_marker", {17, 18}), ("forged_reply_line", {19})])
def test_escape_form_is_pending_for_the_operator(form, classes):
    assert classes <= runner_contract.PENDING_HOSTILE_CLASSES
    assert not (classes & runner_contract.covered_hostile_classes(CASES))
    pytest.skip(f"PENDING (ENH-006 D48): the {form} escape form (hostile class {sorted(classes)}) is written by the "
                f"operator; the host defence is pinned by rule_runner/tests/test_wasi_table.py")
