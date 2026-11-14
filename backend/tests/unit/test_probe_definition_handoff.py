"""A built probe definition reaches miLLM without passing through the agent (ENH-006 G17).

Three gaps, found moving the D123 Llama probe (504,520 bytes) into miLLM on 2026-11-03:

(a) `miserver_import_probe` took the document inline or from a HuggingFace repo, so an agent had to
    paste half a megabyte or publish a DLP probe publicly. It now takes `mistudio_probe_id`: the MCP
    server reads miStudio's `/probe-monitors/probes/{id}/definition` and posts it to miLLM's
    `/api/probes/import` itself (over HTTP, like `miserver_recalibrate_probe`).
(b) `build_probe_definition` answered a Celery id, and `get_task_status` reads the task queue,
    which had no row: 404. The route now opens a `task_queue` row and the task moves it
    (running → completed, or failed with a refusal's reason or the exception).
(c) The import's answer carried miLLM's id as `id`, so `.probe_id` read null. A copy is added.

Payload and call counts are asserted for the two-call path; the one-call inline path is pinned
in `test_reachability.EXPECTED_CALLS`.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.tools import miserver_probes as probe_tools
from tests.support.external_ctx import make_ctx

PM = "pm_8b2c68df722a"
DOC = {"kind": "mistudio.probe-definition/v1", "head": {"weights": [0.1] * 64},
       "provenance": {"probe_id": PM}}
ROW = {"id": "pr_d9a97e20b7a4", "name": "ssn-l12", "layer": 12}


def _register(*, mistudio_answer=None, mistudio_error=None, millm_answer=None):
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("test")
    millm = MagicMock()
    millm.base_url = "http://millm.test"
    millm.get = AsyncMock(side_effect=AssertionError("no miLLM GET expected"))
    millm.post = AsyncMock(return_value=dict(millm_answer if millm_answer is not None else ROW))
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, None))
    mistudio = MagicMock()
    mistudio.get = AsyncMock(side_effect=mistudio_error if mistudio_error is not None
                             else (lambda *a, **k: mistudio_answer if mistudio_answer is not None else DOC))
    probe_tools.register(mcp, millm, gate, make_ctx(mistudio))
    return mcp, millm, mistudio, gate


def _call(registered, **kwargs):
    mcp = registered[0]
    fn = mcp._tool_manager._tools["miserver_import_probe"].fn
    return asyncio.run(fn(**kwargs))


class TestImportByMiStudioId:
    def test_one_miStudio_read_and_ONE_miLLM_post_carrying_the_document(self):
        registered = _register()
        _mcp, millm, mistudio, _gate = registered
        out = _call(registered, mistudio_probe_id=PM)
        assert mistudio.get.await_count == 1
        assert mistudio.get.await_args.args == (f"/probe-monitors/probes/{PM}/definition",)
        assert millm.post.await_count == 1
        assert millm.post.await_args.args == ("/api/probes/import",)
        assert millm.post.await_args.kwargs == {"json_body": DOC, "on_conflict": "rename"}
        assert out["probe_id"] == out["id"] == ROW["id"]
        assert "head" not in out, "the document must not come back through the agent"

    def test_on_conflict_travels(self):
        registered = _register()
        _call(registered, mistudio_probe_id=PM, on_conflict="fail")
        assert registered[1].post.await_args.kwargs["on_conflict"] == "fail"

    def test_not_built_is_a_refusal_with_the_build_call_and_posts_nothing(self):
        registered = _register(mistudio_error=BackendError(
            409, f"probe {PM} has no built definition; POST to this path to build one"))
        out = _call(registered, mistudio_probe_id=PM)
        assert out["refused"] == "definition_not_built"
        assert "no built definition" in out["reason"]
        assert [n["tool"] for n in out["next"]] == ["build_probe_definition"]
        assert out["next"][0]["arguments"] == {"probe_id": PM}
        assert registered[1].post.await_count == 0

    def test_miStudio_down_is_unavailable_and_posts_nothing(self):
        registered = _register(mistudio_error=BackendError(503, "down"))
        out = _call(registered, mistudio_probe_id=PM)
        assert out["unavailable"] == "miworkbench" and "nothing was imported" in out["reason"]
        assert registered[1].post.await_count == 0

    def test_miStudio_unreachable_status_0_is_unavailable_and_posts_nothing(self):
        """`MiWorkbenchClient` raises `BackendError(0, …)` on a refused connection, the real shape of
        "miStudio down". Read as the 404 branch it told the agent the probe does not exist."""
        registered = _register(mistudio_error=BackendError(0, "miStudio backend unreachable: refused"))
        out = _call(registered, mistudio_probe_id=PM)
        assert out.get("unavailable") == "miworkbench", out
        assert "error" not in out and "nothing was imported" in out["reason"]
        assert registered[1].post.await_count == 0

    @pytest.mark.parametrize("bad", ["../probe-monitors/runs", "pm_8b2c/../x", "pm_1?detail=full", "PM_8B2C",
                                     "pm_8B2C68DF722A", "pm_", "pmr_8b2c68df722a", " pm_8b2c68df722a",
                                     "pm_8b2c68df722a\n", "pm_8b2c%2F", "", "   "])
    def test_an_invalid_id_is_refused_by_name_before_any_request(self, bad):
        """NIT 7: the id goes into a URL path, so it is checked against what ids look like first."""
        registered = _register()
        out = _call(registered, mistudio_probe_id=bad)
        assert "pm_ and lowercase hex" in out["error"].replace("`", ""), out
        assert "nothing was sent" in out["error"]
        assert registered[2].get.await_count == 0 and registered[1].post.await_count == 0
        assert registered[3].check.await_count == 0, "an argument error must not consult the gate"

    def test_the_pattern_accepts_every_id_the_generator_makes(self):
        """The check must never refuse a real id: pinned against the generator, not a hand-written example."""
        import re

        from src.mcp_server.patterns import PROBE_MONITOR_ID
        from src.models.probe_monitor import _pm_id

        made = [_pm_id() for _ in range(200)]
        assert all(re.fullmatch(PROBE_MONITOR_ID, pid) for pid in made), made
        registered = _register()
        _call(registered, mistudio_probe_id=made[0])
        assert registered[2].get.await_args.args == (f"/probe-monitors/probes/{made[0]}/definition",)

    def test_an_unknown_probe_is_an_error_and_posts_nothing(self):
        registered = _register(mistudio_error=BackendError(404, f"Probe {PM} not found"))
        out = _call(registered, mistudio_probe_id=PM)
        assert "error" in out and registered[1].post.await_count == 0

    def test_a_non_document_answer_posts_nothing(self):
        registered = _register(mistudio_answer=["not", "a", "document"])
        out = _call(registered, mistudio_probe_id=PM)
        assert out["unavailable"] == "miworkbench" and registered[1].post.await_count == 0

    @pytest.mark.parametrize("extra", [{"definition": DOC}, {"repo_id": "o/r", "filename": "a.probe.json"}])
    def test_two_sources_are_refused_before_anything_is_sent(self, extra):
        registered = _register()
        out = _call(registered, mistudio_probe_id=PM, **extra)
        assert "exactly ONE" in out["error"]
        assert registered[2].get.await_count == 0 and registered[1].post.await_count == 0
        assert registered[3].check.await_count == 0, "an argument error must not consult the gate"

    def test_no_source_names_all_three(self):
        out = _call(_register())
        assert "mistudio_probe_id" in out["error"] and "definition" in out["error"]

    def test_the_parameter_is_in_the_live_registry(self, monkeypatch):
        """Through the real built server, not this module's `register`."""
        from tests.unit.test_mcp_millm_probe_results import MiLLMStub, _server

        mcp = _server(MiLLMStub(), monkeypatch)
        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
        assert "mistudio_probe_id" in tools["miserver_import_probe"].input_schema["properties"]


class TestTheAnswerCarriesProbeId:
    def test_inline_import(self):
        out = _call(_register(), definition=DOC)
        assert out["probe_id"] == ROW["id"] and out["id"] == ROW["id"]

    def test_hub_import(self):
        out = _call(_register(), repo_id="o/r", filename="a.probe.json")
        assert out["probe_id"] == ROW["id"]

    def test_a_present_probe_id_is_not_overwritten_and_a_non_row_is_untouched(self):
        assert probe_tools.with_probe_id({"id": "a", "probe_id": "b"}) == {"id": "a", "probe_id": "b"}
        assert probe_tools.with_probe_id({"refused": "x"}) == {"refused": "x"}
        assert probe_tools.with_probe_id(["x"]) == ["x"]


# ── (b) the build is pollable ──────────────────────────────────────────────────────────────────


def _quiet_route(monkeypatch):
    from src.api.v1.endpoints import probe_monitors as ep

    async def no_refusal(_pid):
        return None

    monkeypatch.setattr(ep, "refuse_precision_mismatch", no_refusal)
    monkeypatch.setattr(ep, "refuse_render_mismatch", no_refusal)
    return ep


@pytest.mark.asyncio
async def test_the_build_route_answers_a_task_queue_id_get_task_status_reads(
        client, async_session, monkeypatch):
    from tests.unit.test_probe_report_summary import _world

    probe_id = await _world(async_session)
    ep = _quiet_route(monkeypatch)
    sent = []

    def dispatch(task, *, gpu_request, kwargs, task_id):
        sent.append((task.name, gpu_request, kwargs, task_id))
        return MagicMock(id=task_id)

    monkeypatch.setattr(ep, "dispatch_gpu_task", dispatch)
    r = await client.post(f"/api/v1/probe-monitors/probes/{probe_id}/definition", json={})
    assert r.status_code == 202, r.text
    body = r.json()
    assert len(sent) == 1
    name, gpu_request, kwargs, celery_id = sent[0]
    assert (name, gpu_request) == ("src.workers.probe_monitor_tasks.build_probe_definition", "auto")
    assert kwargs == {"probe_id": probe_id, "acknowledge_below_rung2": None, "vector_count": 16, "seed": 1337}
    assert body["task_id"] == celery_id
    assert body["task_queue_id"] and body["task_queue_id"].startswith("tq_")
    row = await client.get(f"/api/v1/task-queue/{body['task_queue_id']}")
    assert row.status_code == 200, row.text
    data = row.json()["data"]
    assert (data["task_type"], data["entity_id"], data["entity_type"], data["status"], data["task_id"]) == (
        "probe_definition_build", probe_id, "probe_monitor", "queued", celery_id)


@pytest.mark.asyncio
async def test_the_row_exists_before_any_worker_can_see_the_task(client, async_session, monkeypatch):
    """MINOR 5: dispatched first, a build refused in its first milliseconds could write its terminal
    state before the row was inserted, and `update_row` does not retry: "queued" forever. At the
    moment of dispatch the row must already be committed, under the id the task is sent with."""
    from tests.unit.test_probe_report_summary import _world

    probe_id = await _world(async_session)
    ep = _quiet_route(monkeypatch)
    seen_at_dispatch = []

    def dispatch(task, *, gpu_request, kwargs, task_id):
        seen_at_dispatch.append(_row(task_id))      # a separate connection: only committed rows
        return MagicMock(id=task_id)

    monkeypatch.setattr(ep, "dispatch_gpu_task", dispatch)
    r = await client.post(f"/api/v1/probe-monitors/probes/{probe_id}/definition", json={})
    assert r.status_code == 202, r.text
    assert seen_at_dispatch == [("queued", None, False)]


@pytest.mark.asyncio
async def test_a_row_that_cannot_be_written_refuses_and_dispatches_nothing(client, async_session, monkeypatch):
    """Never `task_queue_id: null` silently: an unpollable build is refused before it is sent."""
    from src.workers import jlens_progress
    from tests.unit.test_probe_report_summary import _world

    probe_id = await _world(async_session)
    ep = _quiet_route(monkeypatch)
    sent = []
    monkeypatch.setattr(ep, "dispatch_gpu_task", lambda *a, **k: sent.append(k) or MagicMock(id="x"))
    monkeypatch.setattr(jlens_progress, "open_row", lambda *a, **k: None)
    r = await client.post(f"/api/v1/probe-monitors/probes/{probe_id}/definition", json={})
    assert r.status_code == 503, r.text
    assert "nothing was dispatched" in r.json()["detail"]
    assert sent == []


@pytest.mark.asyncio
async def test_a_dispatch_that_fails_fails_its_row(client, async_session, monkeypatch):
    """The row was opened first, so a broker that refuses the send must not leave it waiting."""
    from tests.unit.test_probe_report_summary import _world

    probe_id = await _world(async_session)
    ep = _quiet_route(monkeypatch)
    ids = []

    def dispatch(task, *, gpu_request, kwargs, task_id):
        ids.append(task_id)
        raise ConnectionError("broker refused")

    monkeypatch.setattr(ep, "dispatch_gpu_task", dispatch)
    with pytest.raises(ConnectionError):
        await client.post(f"/api/v1/probe-monitors/probes/{probe_id}/definition", json={})
    status, error, _ = _row(ids[0])
    assert status == "failed" and error == "not dispatched: ConnectionError: broker refused"


def _row(task_id):
    from src.core.database import get_sync_db
    from src.models.task_queue import TaskQueue

    with get_sync_db() as db:
        row = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).one()
        return row.status, row.error_message, row.started_at is not None


def _open(task_id):
    from src.workers import jlens_progress

    assert jlens_progress.open_row(jlens_progress.PROBE_DEFINITION_BUILD, PM, task_id, None,
                                   "probe_monitor")


def _run_task(monkeypatch, task_id, outcome):
    """Run the real task wrapper with a request id; the build itself is replaced."""
    from src.workers import probe_monitor_tasks as tasks

    seen = []

    def fake_build(self, probe_id, **kw):
        seen.append(probe_id)
        assert _row(task_id)[0] == "running", "the row must read running while the build works"
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(tasks, "_build_definition", fake_build)
    task = tasks.build_probe_definition
    task.push_request(id=task_id)
    try:
        return task.run(probe_id=PM), seen
    finally:
        task.pop_request()


@pytest.mark.asyncio
async def test_a_completed_build_completes_its_row(async_session, monkeypatch):
    _open("celery-ok")
    result, seen = _run_task(monkeypatch, "celery-ok", {"status": "completed", "probe_id": PM})
    assert seen == [PM] and result["status"] == "completed"
    assert _row("celery-ok") == ("completed", None, True)


@pytest.mark.asyncio
async def test_a_refused_build_fails_its_row_with_the_reason(async_session, monkeypatch):
    _open("celery-refused")
    _run_task(monkeypatch, "celery-refused", {"status": "refused", "http_status": 422,
                                              "reason": "probe is below rung 2"})
    status, error, _ = _row("celery-refused")
    assert status == "failed" and error == "refused (422): probe is below rung 2"


@pytest.mark.asyncio
async def test_a_crashed_build_fails_its_row_and_still_raises(async_session, monkeypatch):
    _open("celery-crash")
    with pytest.raises(RuntimeError):
        _run_task(monkeypatch, "celery-crash", RuntimeError("CUDA out of memory"))
    status, error, _ = _row("celery-crash")
    assert status == "failed" and error == "RuntimeError: CUDA out of memory"


@pytest.mark.asyncio
async def test_an_operator_cancel_still_fails_the_row(async_session, monkeypatch):
    """MINOR 3: `OperatorCancelled` is a BaseException, deliberately. Caught as `Exception` it escaped
    the handler and left the row `running` until the janitor, over an hour later."""
    from src.core.cancellation import OperatorCancelled

    _open("celery-base")
    with pytest.raises(OperatorCancelled):
        _run_task(monkeypatch, "celery-base", OperatorCancelled("jlens_task", "celery-base"))
    status, error, _ = _row("celery-base")
    assert status == "failed" and error == "OperatorCancelled: jlens_task:celery-base cancelled"
