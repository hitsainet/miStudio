"""The state seen live on 2026-11-02, and work on the GPU queues with no row (ENH-006 D116, G13).

Everything here is REAL BROKER STRUCTURE: Celery publishes the message, kombu's own `basic_get` moves it to `unacked` and
`unacked_index`, exactly as a worker's delivery does. An imitation written by hand would agree with the reader by
construction, which is how a suite stays green over the bug.

The property that must never break: a healthy long-running task never reads as stuck.

Controls: `0xcc/reviews/mcp_gaps_controls_2026-11-02.md`.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from celery import Celery
from sqlalchemy import select

from src.core.celery_app import celery_app
from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus
from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun
from src.models.task_queue import TaskQueue
from src.services import active_jobs, gpu_queue_state as state
from src.workers import gpu_worker_watch as watch
from src.workers import task_completion, task_liveness
from tests.support.real_broker import real_broker  # noqa: F401 - a fixture

celery_app.loader.import_default_modules()

EVALUATE = "src.workers.probe_monitor_tasks.evaluate_probe_monitor"
RUN = "src.workers.probe_monitor_tasks.run_probe_monitor"


class Broker:
    """A real Redis driven the way a producer and a solo worker drive it."""

    def __init__(self, client, url):
        self.client, self.url = client, url
        self.app = Celery("test", broker=url, backend=url)
        assert self.app.conf.broker_url == url, "this test must not reach any other broker"
        self.held = []

    def publish(self, task, queue="gpu.auto", task_id=None, **kwargs):
        task_id = task_id or str(uuid.uuid4())
        self.app.send_task(task, kwargs=kwargs, task_id=task_id, queue=queue)
        return task_id

    def deliver(self, queue="gpu.auto"):
        """What a worker's prefetch does: the message moves to `unacked`. The channel stays open, as a wedged worker's."""
        connection = self.app.connection_for_read()
        channel = connection.channel()
        # Kept, and told not to restore: a channel that is garbage collected puts its unacked messages back.
        channel.do_restore = False
        message = channel.basic_get(queue, no_ack=False)
        assert message is not None
        self.held.append((connection, channel, message))
        return message

    def age_delivery(self, seconds):
        for tag, score in self.client.zrange(state.UNACKED_INDEX_KEY, 0, -1, withscores=True):
            self.client.zadd(state.UNACKED_INDEX_KEY, {tag: score - seconds})

    def close(self):
        self.app.close()


@pytest.fixture
def broker(real_broker, monkeypatch):  # noqa: F811
    client, url = real_broker
    # Celery prefers these over the constructor's `broker=`, so unset they would publish to the dev Redis.
    monkeypatch.setenv("CELERY_BROKER_URL", url)
    monkeypatch.setenv("CELERY_RESULT_BACKEND", url)
    monkeypatch.setattr(state, "_client", lambda: client)
    monkeypatch.setattr(task_completion, "_client", lambda: client)
    monkeypatch.setattr(task_liveness, "_client", lambda: client)
    made = Broker(client, url)
    yield made
    made.close()


def stamp_completed(client, task_id, seconds_ago, task=EVALUATE):
    when = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    client.set(task_completion.DONE_PREFIX + task_id, json.dumps({"completed_at": when, "task": task}))


def result_success(client, task_id, seconds_ago):
    when = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    client.set(f"celery-task-meta-{task_id}", json.dumps({"status": "SUCCESS", "date_done": when, "result": {}}))


def assess(**kwargs):
    kwargs.setdefault("cards_held", 0)
    return state.assess(**kwargs)


class TestTheStateSeenLive:
    def test_a_finished_task_never_acked_with_work_waiting_and_no_card_is_wedged(self, broker):
        finished = broker.publish(RUN, run_id="pmr_1")
        broker.deliver()
        result_success(broker.client, finished, 600)
        waiting = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])

        first = assess()
        assert first["state"] == "suspect" and first["condition_holds"] is True, first
        assert [s["task_id"] for s in first["stuck"]] == [finished]
        assert first["waiting"] == 1 and first["cards_held"] == 0

        broker.client.set(state.FIRST_SEEN_KEY, str(time.time() - state.WEDGE_CONFIRM_S - 1))
        confirmed = assess()
        assert confirmed["state"] == "wedged", confirmed
        assert finished in confirmed["message"] and "1 message" in confirmed["message"]
        assert "kill -USR2" in confirmed["message"] and "Nothing restarts it" in confirmed["message"]
        assert waiting not in [s["task_id"] for s in confirmed["stuck"]]

    @pytest.mark.parametrize("age,expected", [(0.0, "suspect"), (state.WEDGE_CONFIRM_S - 30, "suspect"),
                                              (state.WEDGE_CONFIRM_S + 30, "wedged")])
    def test_the_state_must_have_lasted_before_it_is_called_wedged(self, broker, age, expected):
        """A worker between tasks can show this for one pass. Every other test here ages the mark past the window, so
        dropping the window from the check went unnoticed (ENH-006 D116)."""
        finished = broker.publish(RUN, run_id="pmr_1")
        broker.deliver()
        result_success(broker.client, finished, 600)
        broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.client.set(state.FIRST_SEEN_KEY, str(time.time() - age))
        answer = assess()
        assert answer["condition_holds"] is True
        assert answer["state"] == expected, answer

    def test_the_completed_stamp_is_evidence_after_the_result_has_expired(self, broker):
        finished = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.deliver()
        stamp_completed(broker.client, finished, 7200)
        assert assess()["stuck"][0]["evidence"] == "completed"


class TestAHealthyWorker:
    """These are the false alarms that would teach an operator to ignore it."""

    def test_a_long_running_task_is_ok(self, broker):
        """No stamp and no result until it returns, however long it has been running, with work waiting."""
        running = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.deliver()
        broker.age_delivery(30_000)
        broker.client.set(task_liveness.LIVENESS_PREFIX + running, "1")
        broker.publish(EVALUATE, probe_id="pm_2", dataset_ids=["d"])
        answer = assess(cards_held=0)
        assert answer["state"] == "ok" and answer["stuck"] == []

    def test_a_task_that_just_finished_is_within_its_ack(self, broker):
        finished = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.deliver()
        result_success(broker.client, finished, 3)
        broker.publish(EVALUATE, probe_id="pm_2", dataset_ids=["d"])
        assert assess()["state"] == "ok"

    @pytest.mark.asyncio
    async def test_a_run_cancelled_under_a_live_task_is_on_its_way_to_a_checkpoint(self, broker, async_session):
        """The row is terminal at once and the worker stops at its next checkpoint. It is still stamping."""
        run_id = await _run_row(async_session, "cancelled")
        task_id = broker.publish(RUN, run_id=run_id)
        broker.deliver()
        broker.age_delivery(30_000)
        broker.client.set(task_liveness.LIVENESS_PREFIX + task_id, "1")
        assert assess()["state"] == "ok"

    def test_an_acked_message_is_gone(self, broker):
        finished = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        message = broker.deliver()
        result_success(broker.client, finished, 600)
        message.ack()
        assert assess()["state"] == "ok"
        assert broker.client.hlen(state.UNACKED_KEY) == 0, "the reader's premise: an ack empties the hash"


class TestWhatIsNotYetWedged:
    def test_nothing_waiting_is_suspect(self, broker):
        finished = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.deliver()
        result_success(broker.client, finished, 600)
        broker.client.set(state.FIRST_SEEN_KEY, "1")
        answer = assess()
        assert answer["state"] == "suspect" and answer["condition_holds"] is False and "nothing is waiting" in answer["message"]

    def test_a_card_held_is_suspect(self, broker):
        """Another worker has work: the queue may just be waiting for it."""
        finished = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.deliver()
        result_success(broker.client, finished, 600)
        broker.publish(EVALUATE, probe_id="pm_2", dataset_ids=["d"])
        broker.client.set(state.FIRST_SEEN_KEY, "1")
        answer = assess(cards_held=1)
        assert answer["state"] == "suspect" and "a card is held" in answer["message"]

    def test_an_unreadable_broker_is_unknown_never_ok(self):
        def down():
            raise ConnectionError("no broker")

        answer = state.assess(client=MagicMock(zrange=MagicMock(side_effect=ConnectionError("no broker"))))
        assert answer["state"] == "unknown" and "no broker" in answer["reason"]


class TestARowIsEvidence:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status,stuck", [("completed", True), ("cancelled", True), ("running", False)])
    async def test_the_row_found_through_the_tasks_own_declaration(self, broker, async_session, status, stuck):
        """Its result has expired and it has no stamp. Dead liveness and a terminal row say it ended."""
        run_id = await _run_row(async_session, status)
        broker.publish(RUN, run_id=run_id)
        broker.deliver()
        broker.age_delivery(600)
        answer = assess()
        assert bool(answer["stuck"]) is stuck, answer
        if stuck:
            assert answer["stuck"][0]["evidence"] == f"row {status}"


async def _run_row(session, status):
    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="t", status=ModelStatus.READY, file_path="/x",
                  architecture="L", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="c", source="local", status=DatasetStatus.READY, raw_path="/x", extra_metadata={})
    session.add_all([model, dataset])
    await session.commit()
    view = ProbeMonitorDataset(name="t", dataset_id=dataset.id, input_column="t", label_column="l",
                               label_mapping={"1": "positive", "0": "negative"}, role="train", counts={})
    session.add(view)
    await session.commit()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[], config={}, status=status,
                          environment={})
    session.add(run)
    await session.commit()
    return run.id


class TestTheBeatCheck:
    def test_the_beat_entry_is_wired(self):
        entry = celery_app.conf.beat_schedule["watch-gpu-workers"]
        assert entry["task"] == "watch_gpu_workers" and "watch_gpu_workers" in celery_app.tasks
        assert entry["schedule"] <= 60 and entry["options"]["queue"] == "low_priority"

    def test_a_fresh_worker_finds_it_with_nothing_else_imported(self):
        """The registry here already holds the task, because this file imports its module. A worker has only autodiscovery."""
        import subprocess
        import sys
        from pathlib import Path

        program = ("from src.core.celery_app import celery_app\n"
                   "celery_app.loader.import_default_modules()\n"
                   "print('registered' if 'watch_gpu_workers' in celery_app.tasks else 'missing')\n")
        done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=240,
                              cwd=Path(__file__).resolve().parents[2])
        assert done.returncode == 0, done.stderr[-1500:]
        assert "registered" in done.stdout, done.stdout[-500:]

    def test_it_runs_on_a_queue_the_stuck_worker_does_not_consume(self):
        routed = celery_app.amqp.router.route({}, "watch_gpu_workers", (), {})["queue"].name
        assert routed == "low_priority"

    @pytest.fixture
    def tables(self, async_session):
        return async_session

    @pytest.mark.asyncio
    async def test_it_confirms_over_passes_reports_once_and_clears(self, broker, tables, caplog, monkeypatch):
        monkeypatch.setattr(state, "_cards_held", lambda: 0)
        finished = broker.publish(RUN, run_id="pmr_x")
        broker.deliver()
        result_success(broker.client, finished, 600)
        broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])

        first = watch.watch_gpu_workers(client=broker.client)
        assert first["state"] == "suspect"
        assert broker.client.get(state.FIRST_SEEN_KEY) is not None
        assert await self._rows(tables) == []

        aged = str(time.time() - state.WEDGE_CONFIRM_S - 1)
        broker.client.set(state.FIRST_SEEN_KEY, aged)
        with caplog.at_level("ERROR", logger="src.workers.gpu_worker_watch"):
            second = watch.watch_gpu_workers(client=broker.client)
        assert second["state"] == "wedged"
        # Rewritten every pass, the mark would never reach the confirmation window.
        assert broker.client.get(state.FIRST_SEEN_KEY).decode() == aged
        assert "WORKER WEDGED" in caplog.text and finished in caplog.text
        rows = await self._rows(tables)
        assert [(r.status, r.entity_id) for r in rows] == [("failed", finished)]
        assert finished in rows[0].error_message

        watch.watch_gpu_workers(client=broker.client)
        assert len(await self._rows(tables)) == 1, "one row per stuck task, however many passes"

        broker.client.delete(state.UNACKED_KEY, state.UNACKED_INDEX_KEY)
        assert watch.watch_gpu_workers(client=broker.client)["state"] == "ok"
        assert broker.client.get(state.FIRST_SEEN_KEY) is None
        assert (await self._rows(tables, refresh=True))[0].status == "completed"

    async def _rows(self, session, refresh=False):
        session.expire_all()
        return list((await session.execute(
            select(TaskQueue).where(TaskQueue.task_type == watch.WEDGE_TASK_TYPE))).scalars())

    @pytest.mark.asyncio
    async def test_a_healthy_pass_marks_and_reports_nothing(self, broker, tables, monkeypatch):
        task_id = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.deliver()
        broker.client.set(task_liveness.LIVENESS_PREFIX + task_id, "1")
        assert watch.watch_gpu_workers(client=broker.client)["state"] == "ok"
        assert broker.client.get(state.FIRST_SEEN_KEY) is None
        assert await self._rows(tables) == []

    @pytest.mark.asyncio
    async def test_it_restarts_nothing(self, broker, tables, monkeypatch):
        """Reads, a mark, a log line and a row. The message stays; no signal is sent; nothing is revoked."""
        monkeypatch.setattr(state, "_cards_held", lambda: 0)
        finished = broker.publish(RUN, run_id="pmr_x")
        broker.deliver()
        result_success(broker.client, finished, 600)
        waiting = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.client.set(state.FIRST_SEEN_KEY, str(time.time() - state.WEDGE_CONFIRM_S - 1))
        kill = MagicMock()
        revoke = MagicMock()
        monkeypatch.setattr("os.kill", kill)
        monkeypatch.setattr(celery_app.control, "revoke", revoke)
        assert watch.watch_gpu_workers(client=broker.client)["state"] == "wedged"
        kill.assert_not_called()
        revoke.assert_not_called()
        assert broker.client.hlen(state.UNACKED_KEY) == 1
        assert broker.client.llen("gpu.auto") == 1
        assert waiting in broker.client.lindex("gpu.auto", 0).decode()

    @pytest.mark.asyncio
    async def test_the_operator_sees_the_row_and_cannot_be_offered_a_retry(self, broker, client, tables, monkeypatch):
        monkeypatch.setattr(state, "_cards_held", lambda: 0)
        finished = broker.publish(RUN, run_id="pmr_x")
        broker.deliver()
        result_success(broker.client, finished, 600)
        broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.client.set(state.FIRST_SEEN_KEY, str(time.time() - state.WEDGE_CONFIRM_S - 1))
        watch.watch_gpu_workers(client=broker.client)
        listed = (await client.get("/api/v1/task-queue/failed")).json()
        rows = listed["data"] if isinstance(listed, dict) and "data" in listed else listed
        mine = [r for r in rows if r["task_type"] == watch.WEDGE_TASK_TYPE]
        assert len(mine) == 1 and mine[0]["can_retry"] is False and finished in mine[0]["error_message"]


class TestG13TheQueuesAreVisible:
    def test_queued_work_with_no_row_is_named_from_the_message(self, broker):
        """Nothing here lists EVALUATE: the name comes from the message, so tomorrow's task shows too."""
        evaluation = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        invented = broker.publish("src.workers.something_added_tomorrow.new_gpu_task", queue="gpu.gpu-abc")
        messages = {m["task_id"]: m for m in state.read_gpu_messages()}
        assert messages[evaluation]["task"] == EVALUATE and messages[evaluation]["state"] == "queued"
        assert messages[invented]["task"].endswith("new_gpu_task") and messages[invented]["queue"] == "gpu.gpu-abc"

    def test_a_delivered_message_is_visible_too(self, broker):
        running = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.deliver()
        assert [(m["task_id"], m["state"]) for m in state.read_gpu_messages()] == [(running, "delivered")]

    def test_a_parked_hand_off_is_visible(self, broker):
        from src.services.gpu_worker_queues import park_job

        task_id = str(uuid.uuid4())
        park_job(EVALUATE, ["pm"], {}, {"queue": "gpu.auto", "task_id": task_id}, due_at=time.time() + 60, client=broker.client)
        assert [(m["task_id"], m["state"], m["queue"]) for m in state.read_gpu_messages()] == [(task_id, "parked", "gpu.auto")]

    def test_a_message_on_another_queue_is_not_a_gpu_message(self, broker):
        broker.publish("x", queue="datasets")
        assert state.read_gpu_messages() == []

    def test_arguments_never_leave(self, broker):
        """A score's arguments are the text being scored, and this answer leaves the box."""
        broker.publish("src.workers.probe_monitor_tasks.score_probe_monitor", probe_id="pm", text="my secret text")
        assert "my secret text" not in json.dumps(state.read_gpu_messages())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", ["/api/v1/operations/active-jobs", "/api/v1/operations/gpu-status"])
    async def test_no_answer_carries_a_message_or_its_arguments(self, broker, client, monkeypatch, route):
        """Queued, delivered, parked and stuck, read through what leaves the box. A Celery body is base64, so looking
        for the text alone finds nothing even when the whole body is in the answer: look for the body as well, and
        hold the row to its fields. Mutating `read_gpu_messages` and `gpu_queue_jobs` to pass the body on went unnoticed
        by `test_arguments_never_leave` above (ENH-006 D116)."""
        from src.services.gpu_worker_queues import park_job

        secret = "my secret text"
        monkeypatch.setattr(state, "_cards_held", lambda: 0)
        delivered = broker.publish(EVALUATE, probe_id="pm", text=secret)
        broker.publish(EVALUATE, probe_id="pm", text=secret)
        bodies = [json.loads(raw)["body"] for raw in broker.client.lrange("gpu.auto", 0, -1)]
        assert len(bodies) == 2 and secret not in bodies[0], "the bodies are encoded, which is why this test reads them"
        broker.deliver()
        park_job(EVALUATE, [secret], {"text": secret}, {"queue": "gpu.auto", "task_id": str(uuid.uuid4())},
                 due_at=time.time() + 60, client=broker.client)
        result_success(broker.client, delivered, 600)
        broker.client.set(state.FIRST_SEEN_KEY, str(time.time() - state.WEDGE_CONFIRM_S - 1))

        response = await client.get(route)
        assert response.status_code == 200
        body = response.json()
        assert body["worker_watch"]["state"] == "wedged", "the stuck message must be in the answer this test reads"
        for leak in (secret, *bodies):
            assert leak not in response.text

        if route.endswith("active-jobs"):
            rows = [j for j in body["jobs"] if j["kind"] == active_jobs.GPU_QUEUE_KIND]
            assert sorted(r["state"] for r in rows) == ["delivered", "parked", "queued"]
            for row in rows:
                # ENH-006 G41 added `orphan_reason` (a code such as `row_missing`, never an argument).
                assert set(row) == {"kind", "id", "status", "state", "progress", "gpu_uuid", "gpu_uuids", "task_id",
                                    "task", "queue", "created_at", "source_table", "orphan_reason"}
                assert row["orphan_reason"] in (None, "row_missing") or row["orphan_reason"].startswith("row_")

    @pytest.mark.asyncio
    async def test_the_route_lists_it_and_errors_stay_empty(self, broker, client):
        evaluation = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        body = (await client.get("/api/v1/operations/active-jobs")).json()
        assert body["errors"] == []
        mine = [j for j in body["jobs"] if j["task_id"] == evaluation]
        assert len(mine) == 1
        assert (mine[0]["kind"], mine[0]["status"], mine[0]["task"], mine[0]["queue"]) == (
            active_jobs.GPU_QUEUE_KIND, "queued", EVALUATE, "gpu.auto")
        assert active_jobs.GPU_QUEUE_KIND in body["kinds"]
        assert body["worker_watch"]["state"] == "ok"

    @pytest.mark.asyncio
    async def test_a_kind_filter_and_a_card_filter_reach_it(self, broker, client):
        on_card = broker.publish(EVALUATE, queue="gpu.gpu-abc", probe_id="pm", dataset_ids=["d"])
        broker.publish(EVALUATE, probe_id="pm", dataset_ids=["d"])
        narrowed = (await client.get("/api/v1/operations/active-jobs", params={"kind": active_jobs.GPU_QUEUE_KIND})).json()
        assert narrowed["count"] == 2
        card = (await client.get("/api/v1/operations/active-jobs", params={"gpu": "GPU-ABC"})).json()
        assert [j["task_id"] for j in card["jobs"]] == [on_card]

    @pytest.mark.asyncio
    async def test_a_job_with_its_own_row_is_not_listed_twice(self, broker, client, async_session):
        task_id = broker.publish(RUN, run_id="pmr_x")
        run_id = await _run_row(async_session, "pending")
        (await async_session.get(ProbeMonitorRun, run_id)).celery_task_id = task_id
        await async_session.commit()
        body = (await client.get("/api/v1/operations/active-jobs")).json()
        assert [j["kind"] for j in body["jobs"] if j["task_id"] == task_id] == ["probe_training"]

    @pytest.mark.asyncio
    async def test_an_unreadable_broker_is_an_error_not_an_empty_queue(self, client, monkeypatch):
        def down():
            raise ConnectionError("broker down")

        monkeypatch.setattr(state, "_client", down)
        body = (await client.get("/api/v1/operations/active-jobs")).json()
        assert [e["kind"] for e in body["errors"]] == [active_jobs.GPU_QUEUE_KIND]
        assert "broker down" in body["errors"][0]["error"]
        assert body["worker_watch"]["state"] == "unknown"

    @pytest.mark.asyncio
    async def test_gpu_status_carries_the_watch(self, broker, client, monkeypatch):
        monkeypatch.setattr("src.services.gpu_status.read_cards", lambda: ([], "no cards here"))
        monkeypatch.setattr(state, "_cards_held", lambda: 0)
        finished = broker.publish(RUN, run_id="pmr_x")
        broker.deliver()
        result_success(broker.client, finished, 600)
        broker.publish(EVALUATE, probe_id="pm", dataset_ids=["d"])
        broker.client.set(state.FIRST_SEEN_KEY, str(time.time() - state.WEDGE_CONFIRM_S - 1))
        body = (await client.get("/api/v1/operations/gpu-status")).json()
        assert body["worker_watch"]["state"] == "wedged" and body["worker_watch"]["stuck"][0]["task_id"] == finished


class TestThePreflight:
    def _verdict(self, watch_answer):
        from src.mcp_server.tools.gpu_ops import verdict

        card = {"in_use_by": []}
        return verdict(card, None, [], {"configured": False}, {"configured": False}, watch_answer)

    def test_wedged_is_not_clear(self):
        result, busy, uncertain = self._verdict({"state": "wedged", "message": "A GPU worker looks wedged"})
        assert result == "uncertain" and uncertain == ["A GPU worker looks wedged"] and busy == []

    def test_unreadable_is_not_clear(self):
        result, _busy, uncertain = self._verdict({"state": "unknown", "reason": "broker down"})
        assert result == "uncertain" and "broker down" in uncertain[0]

    @pytest.mark.parametrize("answer", [{"state": "ok"}, {"state": "suspect", "message": "m"}, None])
    def test_ok_and_suspect_stay_clear(self, answer):
        """A killed worker's message reads suspect for up to 12 hours; it does not close a free card."""
        assert self._verdict(answer)[0] == "clear"

    def test_the_tool_passes_it_on_from_the_route(self, monkeypatch):
        import asyncio

        monkeypatch.delenv("MILLM_API_URL", raising=False)
        monkeypatch.delenv("MIFORGE_API_URL", raising=False)

        from mcp.server.mcpserver import MCPServer

        from src.mcp_server.config import MCPSettings
        from src.mcp_server.tools import gpu_ops

        calls = []

        class Studio:
            async def get(self, path, **params):
                calls.append(path)
                if path == "/operations/gpu-status":
                    return {"cards": [{"in_use_by": []}], "worker_watch": {"state": "wedged", "message": "stuck"}}
                return {"jobs": [], "errors": [], "read_at": "now"}

        mcp = MCPServer("t")
        gpu_ops.register(mcp, Studio(), MCPSettings(auth_token="x" * 32))
        answer = asyncio.run(mcp._tool_manager._tools["gpu_preflight"].fn(gpu="GPU-abc"))  # noqa: SLF001
        assert answer["verdict"] == "uncertain" and answer["uncertain_reasons"] == ["stuck"]
        assert answer["mistudio"]["worker_watch"]["state"] == "wedged"
        assert calls == ["/operations/gpu-status", "/operations/active-jobs"]
