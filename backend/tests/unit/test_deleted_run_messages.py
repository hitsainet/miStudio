"""A deleted run's queued work is shown as gone, and its message is revoked (ENH-006 G41).

On 2026-11-07 at about 9:10 PM ET four queued D128 runs were cancelled (MCP) and then deleted (UI,
`DELETE /probe-monitors/runs/{id}`). Their rows went; their four `run_probe_monitor` messages stayed
on `gpu.auto`, and `list_active_jobs` showed them as queued work for runs that no longer existed.
The worker refuses a missing row (`missing_row="cancelled"`), so no GPU work ran — but an operator
(and `gpu_preflight`) read four jobs waiting for a card.

Now: deleting an ENDED run still revokes its message (the cancel branch wrote nothing for a
terminal row, so nothing was sent), and a waiting message whose own row is gone or has ended reads
`status: orphaned` with `orphan_reason`, found through the task's own `refuses_terminal_row`
declaration in the live registry. Over a real broker (kombu's own structures) and the real routes.
"""
from __future__ import annotations

import json
import time
import uuid

import pytest

from src.models.probe_monitor import ProbeMonitorRun
from src.services import active_jobs
from tests.support.real_broker import real_broker  # noqa: F401 - a fixture
from tests.unit.test_gpu_worker_watch import EVALUATE, RUN, _run_row, broker  # noqa: F401 - fixtures

ACTIVE = "/api/v1/operations/active-jobs"


async def _gpu_rows(client):
    body = (await client.get(ACTIVE)).json()
    assert body["errors"] == []
    return {j["task_id"]: j for j in body["jobs"] if j["kind"] == active_jobs.GPU_QUEUE_KIND}


class TestOrphanedMessages:
    @pytest.mark.asyncio
    async def test_a_deleted_runs_message_reads_orphaned(self, broker, client, async_session):
        """The G41 case: the message names a run whose row no longer exists."""
        gone = broker.publish(RUN, run_id=f"pmr_{uuid.uuid4().hex[:12]}")
        rows = await _gpu_rows(client)
        assert (rows[gone]["status"], rows[gone]["orphan_reason"], rows[gone]["state"]) == (
            "orphaned", "row_missing", "queued")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["cancelled", "completed", "failed"])
    async def test_an_ended_runs_message_reads_orphaned(self, broker, client, async_session, status):
        run_id = await _run_row(async_session, status)
        task_id = broker.publish(RUN, run_id=run_id)
        rows = await _gpu_rows(client)
        assert (rows[task_id]["status"], rows[task_id]["orphan_reason"]) == ("orphaned", f"row_{status}")

    @pytest.mark.asyncio
    async def test_a_live_runs_redelivered_message_stays_queued(self, broker, client, async_session):
        """Its row is pending under ANOTHER task id (so it is not deduplicated): real waiting work."""
        run_id = await _run_row(async_session, "pending")
        (await async_session.get(ProbeMonitorRun, run_id)).celery_task_id = "the-first-dispatch"
        await async_session.commit()
        task_id = broker.publish(RUN, run_id=run_id)
        rows = await _gpu_rows(client)
        assert (rows[task_id]["status"], rows[task_id]["orphan_reason"]) == ("queued", None)

    @pytest.mark.asyncio
    async def test_a_task_without_a_row_declaration_is_never_called_orphaned(self, broker, client):
        """An evaluation has no row of its own: queued, never orphaned on a guess."""
        task_id = broker.publish(EVALUATE, probe_id="pm_gone", dataset_ids=["d"])
        rows = await _gpu_rows(client)
        assert (rows[task_id]["status"], rows[task_id]["orphan_reason"]) == ("queued", None)

    @pytest.mark.asyncio
    async def test_a_missing_row_where_missing_means_not_yet_written_stays_queued(self, broker, client):
        """`jlens_task` rows are keyed by the Celery id and opened AFTER dispatch, so a missing row there
        means "not written yet" (`missing_row="continue"`), never "deleted": not an orphan."""
        task_id = broker.publish("src.workers.jlens_band_tasks.compute_band_report", artifact_id="ja_1")
        rows = await _gpu_rows(client)
        assert (rows[task_id]["status"], rows[task_id]["orphan_reason"]) == ("queued", None)

    @pytest.mark.asyncio
    async def test_a_parked_message_for_a_deleted_run_reads_orphaned(self, broker, client):
        from src.services.gpu_worker_queues import park_job

        task_id = str(uuid.uuid4())
        park_job(RUN, [], {"run_id": "pmr_deleted00000"}, {"queue": "gpu.auto", "task_id": task_id},
                 due_at=time.time() + 60, client=broker.client)
        rows = await _gpu_rows(client)
        assert (rows[task_id]["state"], rows[task_id]["status"], rows[task_id]["orphan_reason"]) == (
            "parked", "orphaned", "row_missing")

    @pytest.mark.asyncio
    async def test_a_delivered_message_is_never_called_orphaned(self, broker, client, async_session):
        """A row cancelled under a RUNNING task is a task on its way to a checkpoint."""
        run_id = await _run_row(async_session, "cancelled")
        task_id = broker.publish(RUN, run_id=run_id)
        broker.deliver()
        rows = await _gpu_rows(client)
        assert (rows[task_id]["status"], rows[task_id]["orphan_reason"]) == ("running", None)

    @pytest.mark.asyncio
    async def test_preflight_does_not_count_an_orphan_as_waiting(self, broker, client, async_session):
        """`gpu_preflight` counts queued work with no card as a reason for doubt; an orphan is not one."""
        gone = broker.publish(RUN, run_id="pmr_deleted00000")
        body = (await client.get(ACTIVE)).json()
        waiting = [j for j in body["jobs"]
                   if j.get("status") in ("queued", "pending") and not j.get("gpu_uuids")]
        assert gone not in {j["task_id"] for j in waiting}

    @pytest.mark.asyncio
    async def test_an_unreadable_row_never_reads_orphaned(self, broker, client, monkeypatch):
        """`_orphan_of`: a row that cannot be read is not an orphan. Calling it one would hide real
        waiting work from preflight and the watch on the strength of a database error."""
        import src.core.cancellation as cancellation

        def down(kind, target):
            raise RuntimeError("database unavailable")

        monkeypatch.setattr(cancellation, "row_state_of", down)
        task_id = broker.publish(RUN, run_id="pmr_whatever0000")
        body = (await client.get(ACTIVE)).json()
        row = next(j for j in body["jobs"] if j.get("task_id") == task_id)
        assert (row["status"], row["orphan_reason"]) == ("queued", None)

    @pytest.mark.asyncio
    async def test_an_orphan_on_a_card_queue_does_not_hold_the_card(self, broker, client, async_session):
        """MAJOR 1: an orphan on `gpu.<uuid>` keeps its `gpu_uuids`; `assemble` must not count it."""
        from src.services import gpu_status

        gone = broker.publish(RUN, queue="gpu.gpu-abc", run_id="pmr_deleted00000")
        body = (await client.get(ACTIVE)).json()
        row = next(j for j in body["jobs"] if j.get("task_id") == gone)
        assert row["status"] == "orphaned" and row["gpu_uuids"] == ["gpu-abc"]
        cards = gpu_status.assemble([{"uuid": "GPU-ABC", "processes": []}], {}, body["jobs"])
        assert cards[0]["in_use"] is False, cards[0]["in_use_by"]
        assert [j["task_id"] for j in cards[0]["jobs"]] == [gone], "still listed on the card, as orphaned"

    @pytest.mark.asyncio
    async def test_live_work_on_a_card_queue_still_holds_the_card(self, broker, client, async_session):
        """The other side of the line: a delivered (running) message on the card queue does hold it."""
        from src.services import gpu_status

        live = broker.publish(EVALUATE, queue="gpu.gpu-abc", probe_id="pm_1", dataset_ids=["d"])
        broker.deliver(queue="gpu.gpu-abc")
        body = (await client.get(ACTIVE)).json()
        cards = gpu_status.assemble([{"uuid": "GPU-ABC", "processes": []}], {}, body["jobs"])
        assert cards[0]["in_use"] is True and live in cards[0]["in_use_by"][0], cards[0]["in_use_by"]

    @pytest.mark.asyncio
    async def test_preflight_does_not_read_busy_for_an_orphan_on_a_card_queue(
            self, broker, client, async_session, monkeypatch):
        """End to end: the real routes behind `gpu_preflight`, the real tool. Before the fix it answered
        `busy` with "miStudio gpu_queue_message ... (orphaned)" for work the worker will refuse."""
        from mcp.server.mcpserver import MCPServer

        from src.mcp_server.config import MCPSettings
        from src.mcp_server.tools import gpu_ops

        monkeypatch.delenv("MILLM_API_URL", raising=False)
        monkeypatch.delenv("MIFORGE_API_URL", raising=False)
        card = {"uuid": "GPU-ABC", "index": 0, "name": "test card",
                "memory": {"total_mib": 1.0, "used_mib": 0.0, "free_mib": 1.0},
                "utilization": {"gpu_percent": 0, "memory_percent": 0}, "processes": []}
        monkeypatch.setattr("src.services.gpu_status.read_cards", lambda: ([card], None))
        gone = broker.publish(RUN, queue="gpu.gpu-abc", run_id="pmr_deleted00000")

        class Studio:
            async def get(self, path, **params):
                r = await client.get(f"/api/v1{path}", params={k: v for k, v in params.items() if v is not None})
                assert r.status_code == 200, r.text
                return r.json()

        mcp = MCPServer("t")
        gpu_ops.register(mcp, Studio(), MCPSettings(auth_token="x" * 32))
        tool = mcp._tool_manager._tools["gpu_preflight"].fn  # noqa: SLF001
        answer = await tool(gpu="GPU-ABC")
        assert [j["task_id"] for j in answer["mistudio"]["card"]["jobs"]] == [gone], "the orphan is on this card"
        assert (answer["verdict"], answer["busy_reasons"], answer["uncertain_reasons"]) == ("clear", [], []), answer

    @pytest.mark.asyncio
    async def test_an_orphan_is_not_work_waiting_behind_a_stuck_worker(self, broker, async_session):
        """The worker watch's `waiting` applies the same rule: a stuck worker with only orphans behind it
        blocks nothing, so it is not called wedged however long it lasts."""
        from src.services import gpu_queue_state as state
        from tests.unit.test_gpu_worker_watch import result_success

        finished = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
        broker.deliver()
        result_success(broker.client, finished, 600)
        broker.publish(RUN, run_id="pmr_deleted00000")
        broker.client.set(state.FIRST_SEEN_KEY, str(time.time() - state.WEDGE_CONFIRM_S - 1))
        answer = state.assess(cards_held=0)
        assert (answer["state"], answer["waiting"], answer["orphaned"], answer["condition_holds"]) == (
            "suspect", 0, 1, False), answer
        assert "1 orphaned message(s)" in answer["message"]

    def test_the_arguments_still_never_leave(self, broker):
        from src.services import gpu_queue_state as state

        broker.publish("src.workers.probe_monitor_tasks.score_probe_monitor", probe_id="pm", text="my secret text")
        assert "my secret text" not in json.dumps(state.read_gpu_messages(with_rows=True))


class TestDeleteRevokes:
    @pytest.fixture
    def revoked(self, monkeypatch):
        from src.core.celery_app import celery_app

        sent = []
        monkeypatch.setattr(celery_app.control, "revoke", lambda task_id, **kw: sent.append((task_id, kw)))
        return sent

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["cancelled", "completed", "failed"])
    async def test_deleting_an_ended_run_revokes_its_message(self, client, async_session, revoked, status):
        run_id = await _run_row(async_session, status)
        (await async_session.get(ProbeMonitorRun, run_id)).celery_task_id = "celery-ended-1"
        await async_session.commit()
        response = await client.delete(f"/api/v1/probe-monitors/runs/{run_id}")
        assert response.status_code == 204, response.text
        assert revoked == [("celery-ended-1", {})], "once, by id, and never with terminate"

    @pytest.mark.asyncio
    async def test_deleting_a_pending_run_cancels_and_revokes_once(self, client, async_session, revoked, monkeypatch):
        import src.core.cancellation as cancellation

        asked = []
        real = cancellation.request_cancel

        def spy(kind, target_id, **kw):
            asked.append((kind, target_id, kw.get("celery_task_id")))
            return real(kind, target_id, **kw)

        monkeypatch.setattr(cancellation, "request_cancel", spy)
        run_id = await _run_row(async_session, "pending")
        (await async_session.get(ProbeMonitorRun, run_id)).celery_task_id = "celery-pending-1"
        await async_session.commit()
        assert (await client.delete(f"/api/v1/probe-monitors/runs/{run_id}")).status_code == 204
        assert asked == [("probe_monitor_run", run_id, "celery-pending-1")]
        assert revoked == [("celery-pending-1", {})]

    @pytest.mark.asyncio
    async def test_a_run_with_no_task_id_sends_nothing(self, client, async_session, revoked):
        run_id = await _run_row(async_session, "failed")
        assert (await client.delete(f"/api/v1/probe-monitors/runs/{run_id}")).status_code == 204
        assert revoked == []
