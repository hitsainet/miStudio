"""Session spot-checks end to end (Feature 038 FR-1 – FR-9, FR-12, FR-13, FR-17).

miLLM 036's session routes do not exist yet, so miLLM is a CONTRACT-FAITHFUL FAKE: an `httpx.MockTransport`
serving exactly the paths and fields 036 FPRD §7 / FR-36.9 / FR-36.12 / FR-36.14 name, behind the REAL
`millm_read` client. The judge is a SEPARATE fake behind the real OpenAI client made by Part A's
`_client_for`, so a call to the wrong one fails, and every judge request's headers are visible.

Fixtures differ in the field under test: the three probes of the window test differ only in `window`;
the two prompts of FR-7 differ only in `behaviour`.

Controls: 0xcc/reviews/037_038_implementation_controls_2026-10-21.md.
"""
from __future__ import annotations

import json
import logging

import httpx
import pytest

from src.services import millm_read
from src.services import probe_spot_check as spot
from tests.unit.test_behaviour_catalogue import API, _create, _probe, _prompt, sync_db  # noqa: F401

SENTINEL = "SENTINEL-8f3a"


class FakeMiLLM:
    """miLLM as 036 specifies it: probe list (with source_probe_id), events, SAMPLE, TURN_TEXT, OpenAPI."""

    def __init__(self, *, armed, turns, transcripts, events=None, serves_sessions=True):
        self.armed = armed
        self.turns = turns
        self.transcripts = transcripts           # {(session_id, turn_index): {user_text, reply_text}}
        self.events = events or {}               # {request_id: [event, ...]}
        self.serves_sessions = serves_sessions
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        ok = lambda data: httpx.Response(200, json={"success": True, "data": data})  # noqa: E731
        if path == "/openapi.json":
            paths = {"/api/probes": {}, "/api/probes/events": {}}
            if self.serves_sessions:
                paths.update({"/api/probes/sessions/sample": {}, "/api/probes/sessions/{id}/transcript": {}})
            return httpx.Response(200, json={"paths": paths})
        if path == "/api/probes":
            return ok(self.armed)
        if path == "/api/probes/sessions/sample" and request.method == "POST":
            body = json.loads(request.content)
            strata = {
                name: {"requested": body["strata"][name],
                       "returned": sum(1 for t in self.turns if t["stratum"] == name),
                       "short": max(0, body["strata"][name] - sum(1 for t in self.turns if t["stratum"] == name))}
                for name in ("fired", "near_threshold", "not_fired")
            }
            return ok({"turns": self.turns, "strata": strata})
        if path.startswith("/api/probes/sessions/") and path.endswith("/transcript"):
            session = path.split("/")[4]
            turn = int(request.url.params["turn_index"])
            text = self.transcripts.get((session, turn))
            if text is None:
                return httpx.Response(404, json={"success": False})
            return ok({"session_id": session, "turn_index": turn, **text})
        if path == "/api/probes/events":
            return ok(self.events.get(request.url.params.get("request_id"), []))
        return httpx.Response(404, json={"success": False, "path": path})

    def calls(self, method, path):
        return [r for r in self.requests if r.method == method and r.url.path == path]


class FakeJudge:
    """An OpenAI-compatible judge: answers from `answers` in turn, recording every request."""

    def __init__(self, answer=lambda messages: '{"answer": "yes"}'):
        self.answer = answer
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        content = self.answer(body["messages"])
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
        })

    def client(self):
        from src.services.probe_monitor_judge import _client_for

        return _client_for("http://judge.test/v1", None, http_client=httpx.Client(transport=httpx.MockTransport(self.handler)))


@pytest.fixture
def wire(monkeypatch):
    """Point the backend at a fake miLLM through the real client."""
    from src.core.config import settings

    state = {}

    def install(fake: FakeMiLLM):
        state["fake"] = fake
        monkeypatch.setattr(settings, "millm_api_url", "http://millm.test")
        monkeypatch.setattr(millm_read, "_transport_factory", lambda: httpx.MockTransport(fake.handler))
        return fake

    return install


@pytest.fixture
def queued(monkeypatch):
    from src.workers import probe_spot_check_tasks

    sent = []

    class _Task:
        id = "celery-1"

    monkeypatch.setattr(probe_spot_check_tasks.run_probe_spot_check, "delay",
                        lambda **kw: sent.append(kw) or _Task())
    return sent


def _turn(i, stratum, *, verdict, score, window="all"):
    return {"session_id": f"ps_{i}", "request_id": f"chatcmpl-{i}", "turn_index": 0, "event_id": 100 + i,
            "score": score, "threshold": 0.5, "verdict": verdict, "stratum": stratum, "window": window,
            "threshold_revision": 1}


async def _setup(client, sync_db, *, prompt_behaviour="pii"):
    probe = _probe(sync_db, behaviour="pii", confirmed="pii", run_behaviour=None, rung=2)
    prompt = _prompt(sync_db, behaviour=prompt_behaviour)
    await _create(client)
    response = await client.put(f"{API}/behaviours/pii/probes/{probe.id}/role", json={"role": "primary"})
    assert response.status_code == 200, response.text
    return probe, prompt


def _body(prompt, **over):
    return {"behaviour": "pii", "prompt_id": prompt.id, "endpoint": "http://judge.test/v1", "model": "judge",
            "size": 3, "seed": 42, **over}


def _fake(probe, turns=None, transcripts=None, **kw):
    turns = turns if turns is not None else [
        _turn(1, "fired", verdict=True, score=0.9),
        _turn(2, "near_threshold", verdict=False, score=0.48),
        _turn(3, "not_fired", verdict=False, score=0.1),
    ]
    transcripts = transcripts if transcripts is not None else {
        (t["session_id"], 0): {"user_text": f"user {t['session_id']}", "reply_text": f"reply {t['session_id']}"}
        for t in turns
    }
    return FakeMiLLM(armed=[{"id": "pr_1", "source_probe_id": probe.id}], turns=turns, transcripts=transcripts, **kw)


# ── FR-17: standalone ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_millm_is_503_naming_millm(client, sync_db, monkeypatch, queued):
    from src.core.config import settings

    _, prompt = await _setup(client, sync_db)
    monkeypatch.setattr(settings, "millm_api_url", "")
    refused = await client.post(f"{API}/spot-checks", json=_body(prompt))
    assert refused.status_code == 503
    assert "miLLM" in refused.json()["detail"]["message"]
    assert queued == []
    assert (await client.get(f"{API}/spot-checks/availability")).json()["available"] is False


@pytest.mark.asyncio
async def test_unreachable_millm_is_503(client, sync_db, monkeypatch, queued):
    from src.core.config import settings

    _, prompt = await _setup(client, sync_db)
    monkeypatch.setattr(settings, "millm_api_url", "http://millm.test")

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(millm_read, "_transport_factory", lambda: httpx.MockTransport(down))
    refused = await client.post(f"{API}/spot-checks", json=_body(prompt))
    assert refused.status_code == 503 and "could not be reached" in refused.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_a_millm_without_the_session_routes_is_503_saying_so(client, sync_db, wire, queued):
    probe, prompt = await _setup(client, sync_db)
    wire(_fake(probe, serves_sessions=False))
    refused = await client.post(f"{API}/spot-checks", json=_body(prompt))
    assert refused.status_code == 503
    assert "Feature 036" in refused.json()["detail"]["message"]


# ── FR-1, FR-7, start ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_round_trip(client, sync_db, wire, queued):
    """`prompt_key` and `behaviour` hold distinct values, so a swapped column is visible."""
    probe, prompt = await _setup(client, sync_db)
    wire(_fake(probe))
    started = await client.post(f"{API}/spot-checks", json=_body(prompt, window="last_user"))
    assert started.status_code == 202, started.text
    assert queued == [{"spot_check_id": started.json()["id"]}]
    got = (await client.get(f"{API}/spot-checks/{started.json()['id']}")).json()
    assert got["behaviour"] == "pii" and got["prompt_key"] == f"{prompt.name}/v1"
    assert got["prompt_hash"] == prompt.content_hash and got["prompt_id"] == prompt.id
    assert (got["endpoint"], got["model"], got["status"]) == ("http://judge.test/v1", "judge", "pending")
    spec = got["sample_spec"]
    assert spec["seed"] == 42 and spec["window"] == "last_user"
    assert spec["strata"] == {"fired": 1, "near_threshold": 1, "not_fired": 1}
    assert spec["primary"] == {"mistudio_probe_id": probe.id, "millm_probe_id": "pr_1"}
    assert got["excerpts_expire_at"] and got["pinned"] is False


@pytest.mark.asyncio
async def test_prompt_must_judge_the_behaviour(client, sync_db, wire, queued):
    """Two prompts that differ only in their behaviour."""
    probe, prompt = await _setup(client, sync_db)
    other = _prompt(sync_db, behaviour="high-stakes")
    wire(_fake(probe))
    refused = await client.post(f"{API}/spot-checks", json=_body(other))
    assert refused.status_code == 422 and refused.json()["detail"]["code"] == "prompt_behaviour_mismatch"
    assert (await client.post(f"{API}/spot-checks", json=_body(prompt))).status_code == 202


@pytest.mark.asyncio
async def test_size_cap_and_one_running_check_per_behaviour(client, sync_db, wire, queued):
    probe, prompt = await _setup(client, sync_db)
    wire(_fake(probe))
    assert (await client.post(f"{API}/spot-checks", json=_body(prompt, size=501))).status_code == 422
    assert (await client.post(f"{API}/spot-checks", json=_body(prompt))).status_code == 202
    again = await client.post(f"{API}/spot-checks", json=_body(prompt))
    assert again.status_code == 409 and again.json()["detail"]["code"] == "check_running"


@pytest.mark.asyncio
async def test_the_primary_must_be_armed_in_millm(client, sync_db, wire, queued):
    probe, prompt = await _setup(client, sync_db)
    fake = _fake(probe)
    fake.armed = [{"id": "pr_9", "source_probe_id": "pm_somebody_else"}]
    wire(fake)
    refused = await client.post(f"{API}/spot-checks", json=_body(prompt))
    assert refused.status_code == 422 and refused.json()["detail"]["code"] == "primary_not_armed"


# ── execute: FR-3, FR-4, FR-6, FR-9 ──────────────────────────────────────────────────────────


async def _started(client, sync_db, wire, queued, fake_kwargs=None, **body):
    probe, prompt = await _setup(client, sync_db)
    fake = wire(_fake(probe, **(fake_kwargs or {})))
    started = await client.post(f"{API}/spot-checks", json=_body(prompt, **body))
    assert started.status_code == 202, started.text
    return started.json()["id"], probe, prompt, fake


def _run(sync_db, check_id, judge):
    sync_db.expire_all()
    return spot.execute(sync_db, check_id, judge_client=judge.client())


@pytest.mark.asyncio
async def test_reads_the_sample_once_with_the_spec(client, sync_db, wire, queued):
    check_id, probe, _prompt_row, fake = await _started(client, sync_db, wire, queued, window="last_user")
    fake.requests.clear()
    _run(sync_db, check_id, FakeJudge())
    samples = fake.calls("POST", "/api/probes/sessions/sample")
    assert len(samples) == 1
    sent = json.loads(samples[0].content)
    assert sent == {"probe_id": "pr_1", "window": "last_user",
                    "strata": {"fired": 1, "near_threshold": 1, "not_fired": 1}, "near_band": 0.1,
                    "since": sent["since"], "until": None, "seed": 42, "require_transcript": True}
    transcripts = [r for r in fake.requests if r.url.path.endswith("/transcript")]
    assert sorted(r.url.path for r in transcripts) == [f"/api/probes/sessions/ps_{i}/transcript" for i in (1, 2, 3)]
    assert all(r.url.params["turn_index"] == "0" for r in transcripts)


@pytest.mark.asyncio
async def test_same_seed_same_request(client, sync_db, wire, queued):
    check_id, _, _, fake = await _started(client, sync_db, wire, queued)
    _run(sync_db, check_id, FakeJudge())
    first = json.loads(fake.calls("POST", "/api/probes/sessions/sample")[0].content)
    from src.models.probe_spot_check import ProbeSpotCheck

    row = sync_db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == check_id).one()
    assert spot.sample_request(row) == {**first, "since": row.sample_spec["since"]}


@pytest.mark.asyncio
async def test_window_shapes_the_judged_text(client, sync_db, wire, queued):
    """One turn, three compared probes that differ ONLY in the window their event scored."""
    from src.models.probe_spot_check import ProbeSpotCheckItem

    probe, prompt = await _setup(client, sync_db)
    p2 = _probe(sync_db, behaviour="pii", run_behaviour=None)
    p3 = _probe(sync_db, behaviour="pii", run_behaviour=None)
    turn = _turn(1, "fired", verdict=True, score=0.9, window="last_user")
    fake = wire(FakeMiLLM(
        armed=[{"id": "pr_1", "source_probe_id": probe.id}, {"id": "pr_2", "source_probe_id": p2.id},
               {"id": "pr_3", "source_probe_id": p3.id}],
        turns=[turn],
        transcripts={("ps_1", 0): {"user_text": "my name is Ada", "reply_text": "hello Ada"}},
        events={"chatcmpl-1": [
            {"id": 201, "probe_id": "pr_2", "window": "response", "score": 0.7, "threshold": 0.5, "verdict": True, "scored": True},
            {"id": 202, "probe_id": "pr_3", "window": "all", "score": 0.2, "threshold": 0.5, "verdict": False, "scored": True},
        ]},
    ))
    started = await client.post(f"{API}/spot-checks", json=_body(prompt, window="last_user", size=1))
    judge = FakeJudge()
    _run(sync_db, started.json()["id"], judge)
    items = {i.probe_id: i for i in sync_db.query(ProbeSpotCheckItem).filter(
        ProbeSpotCheckItem.spot_check_id == started.json()["id"]).all()}
    user = {"role": "user", "content": "my name is Ada"}
    reply = {"role": "assistant", "content": "hello Ada"}
    assert (items[probe.id].window, items[probe.id].judge_saw, items[probe.id].judged_text) == ("last_user", "same_window", [user])
    assert (items[p2.id].window, items[p2.id].judge_saw, items[p2.id].judged_text) == ("response", "user_and_reply", [user, reply])
    assert (items[p3.id].window, items[p3.id].judge_saw, items[p3.id].judged_text) == ("all", "newest_turn", [user, reply])
    # The judge received exactly the rendered user message for each distinct excerpt (two distinct).
    sent = [json.loads(r.content)["messages"][1]["content"] for r in judge.requests]
    assert len(sent) == 2
    assert len(fake.calls("GET", "/api/probes/events")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("form,reply,positive", [
    ({"kind": "binary"}, '{"answer": "yes"}', True),
    ({"kind": "binary"}, '{"answer": "no"}', False),
    ({"kind": "rating", "min": 1, "max": 10, "positive_from": 6, "json_key": "rating"}, '{"rating": 7}', True),
    ({"kind": "rating", "min": 1, "max": 10, "positive_from": 6, "json_key": "rating"}, '{"rating": 3}', False),
    ({"kind": "labels", "labels": ["leak", "clean"], "positive_labels": ["leak"], "json_key": "label"}, '{"label": "leak"}', True),
])
async def test_answer_forms_map_to_judge_positive(client, sync_db, wire, queued, form, reply, positive):
    from src.models.probe_judge_prompt import ProbeJudgePrompt
    from src.models.probe_spot_check import ProbeSpotCheckItem
    from src.services import judge_prompt_library as library
    from src.services.judge_prompt_forms import content_hash

    probe = _probe(sync_db, behaviour="pii", confirmed="pii", run_behaviour=None)
    content = dict(system_prompt="judge it", user_template="{text}",
                   answer_form=library.normalise_answer_form(form), max_interaction_chars=8000)
    prompt = ProbeJudgePrompt(name="form-test", version=1, title="t", behaviour="pii",
                              content_hash=content_hash(name="form-test", version=1, **content),
                              source="operator", created_by="operator", **content)
    sync_db.add(prompt)
    sync_db.commit()
    await _create(client)
    await client.put(f"{API}/behaviours/pii/probes/{probe.id}/role", json={"role": "primary"})
    wire(_fake(probe, turns=[_turn(1, "fired", verdict=True, score=0.9)]))
    started = await client.post(f"{API}/spot-checks", json=_body(prompt, size=1))
    _run(sync_db, started.json()["id"], FakeJudge(lambda m: reply))
    item = sync_db.query(ProbeSpotCheckItem).filter(ProbeSpotCheckItem.spot_check_id == started.json()["id"]).one()
    assert (item.parse_failed, item.judge_positive) == (False, positive)


@pytest.mark.asyncio
async def test_an_unreadable_reply_is_a_parse_failure_and_over_the_limit_fails(client, sync_db, wire, queued):
    from src.models.probe_spot_check import ProbeSpotCheck

    check_id, *_ = await _started(client, sync_db, wire, queued)
    _run(sync_db, check_id, FakeJudge(lambda m: "maybe?"))
    row = sync_db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == check_id).one()
    assert row.status == "failed" and row.parse_failures == 3
    assert row.error_message.startswith("JudgeUnreliable")
    probe_report = next(iter(row.metrics["probes"].values()), None) or next(iter(row.metrics["newest_turn_probes"].values()))
    assert probe_report["n_parse_failed"] == 3 and probe_report["overall"]["n"] == 0


@pytest.mark.asyncio
async def test_short_stratum_is_reported_not_padded(client, sync_db, wire, queued):
    """Only one turn exists; three are asked for. The items are exactly what came back."""
    from src.models.probe_spot_check import ProbeSpotCheck, ProbeSpotCheckItem

    check_id, *_ = await _started(
        client, sync_db, wire, queued, fake_kwargs={"turns": [_turn(1, "fired", verdict=True, score=0.9)]}, size=6,
    )
    _run(sync_db, check_id, FakeJudge())
    row = sync_db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == check_id).one()
    assert row.metrics["shortfall"]["near_threshold"] == {"requested": 2, "returned": 0, "short": 2}
    assert row.metrics["shortfall"]["fired"]["returned"] == 1
    assert sync_db.query(ProbeSpotCheckItem).filter(ProbeSpotCheckItem.spot_check_id == check_id).count() == 1


@pytest.mark.asyncio
async def test_a_turn_without_a_transcript_is_counted_never_judged(client, sync_db, wire, queued):
    from src.models.probe_spot_check import ProbeSpotCheckItem

    check_id, probe, _, fake = await _started(client, sync_db, wire, queued, fake_kwargs={"transcripts": {}})
    judge = FakeJudge()
    _run(sync_db, check_id, judge)
    items = sync_db.query(ProbeSpotCheckItem).filter(ProbeSpotCheckItem.spot_check_id == check_id).all()
    assert len(items) == 3 and all(i.not_recorded and i.judged_text is None for i in items)
    assert judge.requests == []


@pytest.mark.asyncio
async def test_a_failing_judge_endpoint_fails_the_run_and_keeps_partial_items(client, sync_db, wire, queued):
    from src.models.probe_spot_check import ProbeSpotCheck

    check_id, *_ = await _started(client, sync_db, wire, queued)
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] > 1:
            raise httpx.ConnectError("connection refused", request=request)
        return FakeJudge().handler(request)

    from src.services.probe_monitor_judge import _client_for

    sync_db.expire_all()
    spot.execute(sync_db, check_id, judge_client=_client_for(
        "http://judge.test/v1", None, http_client=httpx.Client(transport=httpx.MockTransport(flaky))))
    row = sync_db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == check_id).one()
    assert row.status == "failed" and row.partial is True
    assert "JudgeUnavailable" in row.error_message


# ── FR-12: retention and privacy ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deleting_the_run_leaves_zero_items(client, sync_db, wire, queued):
    from src.models.probe_spot_check import ProbeSpotCheckItem

    check_id, *_ = await _started(client, sync_db, wire, queued)
    _run(sync_db, check_id, FakeJudge())
    assert sync_db.query(ProbeSpotCheckItem).filter(ProbeSpotCheckItem.spot_check_id == check_id).count() == 3
    assert (await client.delete(f"{API}/spot-checks/{check_id}")).status_code == 204
    sync_db.expire_all()
    assert sync_db.query(ProbeSpotCheckItem).filter(ProbeSpotCheckItem.spot_check_id == check_id).count() == 0


@pytest.mark.asyncio
async def test_sentinel_never_leaves_the_items_table(client, sync_db, wire, queued, caplog):
    """A turn carrying a sentinel: absent from logs (every level), the task arguments and the result."""
    from src.models.probe_spot_check import ProbeSpotCheck, ProbeSpotCheckItem

    caplog.set_level(logging.DEBUG)
    transcripts = {(f"ps_{i}", 0): {"user_text": f"{SENTINEL} user {i}", "reply_text": f"{SENTINEL} reply"} for i in (1, 2, 3)}
    check_id, *_ = await _started(client, sync_db, wire, queued, fake_kwargs={"transcripts": transcripts})
    result = _run(sync_db, check_id, FakeJudge(lambda m: '{"answer": "yes", "reason": "' + SENTINEL + '"}'))
    assert SENTINEL not in json.dumps(result)
    assert SENTINEL not in json.dumps(queued)
    # SQLAlchemy's statement echo is excluded: it is on only under ENVIRONMENT=development (this suite);
    # the deployed backend sets ENVIRONMENT so echo is off (k8s/base/backend.yaml). Every other logger counts.
    logged = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith("sqlalchemy"))
    assert logged, "positive control: the run does log"
    assert SENTINEL not in logged
    row = sync_db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == check_id).one()
    assert SENTINEL not in json.dumps(row.metrics) + json.dumps(row.sample_returned)
    stored = sync_db.query(ProbeSpotCheckItem).filter(ProbeSpotCheckItem.spot_check_id == check_id).all()
    assert all(SENTINEL in json.dumps(i.judged_text) for i in stored), "positive control: the text IS stored"


@pytest.mark.asyncio
async def test_excerpts_expire_unless_pinned(client, sync_db, wire, queued):
    from datetime import timedelta

    from src.core.clock import utc_now
    from src.models.probe_spot_check import ProbeSpotCheck, ProbeSpotCheckItem

    check_id, *_ = await _started(client, sync_db, wire, queued)
    _run(sync_db, check_id, FakeJudge())
    row = sync_db.query(ProbeSpotCheck).filter(ProbeSpotCheck.id == check_id).one()
    row.excerpts_expire_at = utc_now() - timedelta(minutes=1)
    row.pinned = True
    sync_db.commit()
    assert spot.expire_excerpts(sync_db) == 0
    row.pinned = False
    sync_db.commit()
    assert spot.expire_excerpts(sync_db) == 3
    items = sync_db.query(ProbeSpotCheckItem).filter(ProbeSpotCheckItem.spot_check_id == check_id).all()
    assert all(i.judged_text is None and i.judge_reply is None for i in items)
    assert all(i.probe_score is not None for i in items), "counts, scores and ids stay"


def test_excerpt_expiry_is_on_the_beat():
    from src.core.celery_app import celery_app

    tasks = {entry["task"] for entry in celery_app.conf.beat_schedule.values()}
    assert "src.workers.probe_spot_check_tasks.expire_spot_check_excerpts" in tasks
    assert "src.workers.probe_spot_check_tasks.expire_spot_check_excerpts" in celery_app.tasks


@pytest.mark.asyncio
async def test_pinning_clears_the_expiry_and_unpinning_restarts_it(client, sync_db, wire, queued):
    check_id, *_ = await _started(client, sync_db, wire, queued)
    pinned = (await client.post(f"{API}/spot-checks/{check_id}/pin", json={"pinned": True})).json()
    assert pinned["pinned"] is True and pinned["excerpts_expire_at"] is None
    unpinned = (await client.post(f"{API}/spot-checks/{check_id}/pin", json={"pinned": False})).json()
    assert unpinned["pinned"] is False and unpinned["excerpts_expire_at"] is not None


# ── FR-13: wiring ────────────────────────────────────────────────────────────────────────────


def test_spot_check_routes_in_openapi():
    from src.main import app

    paths = app.openapi()["paths"]
    expected = {
        "/api/v1/probe-monitors/spot-checks": {"get", "post"},
        "/api/v1/probe-monitors/spot-checks/availability": {"get"},
        "/api/v1/probe-monitors/spot-checks/{check_id}": {"get", "delete"},
        "/api/v1/probe-monitors/spot-checks/{check_id}/items": {"get"},
        "/api/v1/probe-monitors/spot-checks/{check_id}/cancel": {"post"},
        "/api/v1/probe-monitors/spot-checks/{check_id}/pin": {"post"},
        "/api/v1/probe-monitors/spot-checks/{check_id}/disagreements.jsonl": {"get"},
    }
    missing = {p: sorted(m - set(paths.get(p, {}))) for p, m in expected.items() if m - set(paths.get(p, {}))}
    assert not missing, missing


def test_the_task_routes_to_processing():
    from src.core.celery_app import celery_app

    route = celery_app.amqp.router.route({}, "src.workers.probe_spot_check_tasks.run_probe_spot_check")
    assert route["queue"].name == "processing"

