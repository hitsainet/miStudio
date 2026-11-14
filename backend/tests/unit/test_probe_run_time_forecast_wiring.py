"""ENH-006 G15 wiring: the time forecast reaches the forecast answer, the agent's start answer, the
approval request the operator decides from, and the card that shows it.

Each test here goes RED when its wiring line is removed (controls recorded in
`0xcc/reviews/g15_time_forecast_2026-11-09.md`). The model itself is tested in
`test_probe_run_time_forecast.py`.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.config import MCPSettings
from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun
from tests.unit.test_probe_dataset_manifest import arrow_dict, dataset_id  # noqa: F401 - fixtures
from tests.unit.test_probe_monitor_pipeline import tiny_tokenizer  # noqa: F401 - fixture
from tests.unit.test_probe_run_memory_forecast import preflight_rows  # noqa: F401 - fixture

FORECAST = "/probe-monitors/runs/forecast"


# ── the route ────────────────────────────────────────────────────────────────────────────────


class TestTheForecastRouteCarriesIt:
    @pytest.mark.asyncio
    async def test_the_route_calls_the_time_forecast_once_with_the_request_and_the_memory_answer(
        self, client, preflight_rows, monkeypatch  # noqa: F811
    ):
        from src.services import probe_run_time_forecast as tf

        seen = []

        def spy(request, memory):
            seen.append((request, memory))
            return {"forecast": True, "summary": "sentinel"}

        monkeypatch.setattr(tf, "time_forecast_for_request", spy)
        model_id, view_id = preflight_rows
        response = await client.post(f"/api/v1{FORECAST}", json={
            "model_id": model_id, "train_dataset_id": view_id, "eval_dataset_ids": ["pmd_e1", "pmd_e2"],
            "calibration_dataset_id": "pmd_c1", "gpu": "auto",
            "config": {"scope": "all", "max_length": 64, "stride": 5, "eval_max_rows": 5000}})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["time_forecast"] == {"forecast": True, "summary": "sentinel"}
        assert len(seen) == 1
        request, memory = seen[0]
        assert request["model_id"] == model_id and request["train_dataset_id"] == view_id
        assert request["eval_dataset_ids"] == ["pmd_e1", "pmd_e2"]
        assert request["calibration_dataset_id"] == "pmd_c1"
        assert request["config"]["eval_max_rows"] == 5000 and request["gpu"] == "auto"
        # The memory answer it reads its rows and tokens from — the one the route returns beside it.
        assert memory["checked"] is True and memory["rows"] == body["rows"] == 80
        assert memory["real_tokens"] == body["real_tokens"]

    @pytest.mark.asyncio
    async def test_for_real_from_a_completed_run_in_the_database(
        self, client, async_session, preflight_rows  # noqa: F811
    ):
        """No stub: the route reads a completed run's stage timings from Postgres and forecasts from them."""
        model_id, view_id = preflight_rows
        eval_view = ProbeMonitorDataset(
            name="eval", dataset_id=(await async_session.get(ProbeMonitorDataset, view_id)).dataset_id,
            split="test", input_column="text", label_column="label",
            label_mapping={"yes": "positive", "no": "negative"}, role="eval",
            counts={"positive": 300, "negative": 9700})
        async_session.add(eval_view)
        await async_session.flush()
        done = datetime.now(timezone.utc) - timedelta(hours=1)
        past = ProbeMonitorRun(
            id=f"pmr_{uuid.uuid4().hex[:12]}", model_id=model_id, train_dataset_id=view_id,
            eval_dataset_ids=[eval_view.id], status="completed", completed_at=done,
            config={"scope": "all", "rules": ["mean"], "top_n_layers": 1, "eval_max_rows": 1000},
            layer_selection={"chosen": [5]},
            environment={
                "render": {"rendered": 80, "scored_tokens": 4000},
                "layers_swept": [0, 5, 10],
                "stage_seconds": {"rendering": 8.0, "pooled_capture": 40.0, "layer_selection": 2.4,
                                  "token_capture": 20.0, "training": 16.0, "calibrating": 0.0,
                                  "evaluating": 100.0, "rung": 0.1},
            })
        async_session.add(past)
        await async_session.commit()

        response = await client.post(f"/api/v1{FORECAST}", json={
            "model_id": model_id, "train_dataset_id": view_id, "eval_dataset_ids": [eval_view.id],
            "config": {"scope": "all", "max_length": 64, "stride": 5, "rules": ["mean"], "eval_max_rows": 2000}})
        assert response.status_code == 200, response.text
        body = response.json()
        time = body["time_forecast"]
        assert time["forecast"] is True, time
        assert [r["run_id"] for r in time["basis"]["runs"]] == [past.id]
        # The past run scored 1,000 rows in 100 s; this one scores 2,000 of the same set.
        assert time["eval_rows"] == 2000
        assert time["stages"]["evaluating"]["seconds"]["typical"] == pytest.approx(200.0)
        # Rendering: 8 s for 80 rows, and this request renders the same 80.
        assert time["stages"]["rendering"]["seconds"]["typical"] == pytest.approx(8.0)
        assert time["summary"].startswith("About ")

    @staticmethod
    async def _past_run(async_session, model_id, view_id, *, eval_ids, hours_ago=1.0, gpu_uuid=None,
                        evaluating=100.0):
        past = ProbeMonitorRun(
            id=f"pmr_{uuid.uuid4().hex[:12]}", model_id=model_id, train_dataset_id=view_id,
            eval_dataset_ids=eval_ids, status="completed", gpu_uuid=gpu_uuid,
            completed_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
            config={"scope": "all", "rules": ["mean"], "top_n_layers": 1, "eval_max_rows": 1000},
            layer_selection={"chosen": [5]},
            environment={"render": {"rendered": 80, "scored_tokens": 4000}, "layers_swept": [0, 5, 10],
                         "stage_seconds": {"rendering": 8.0, "evaluating": evaluating}})
        async_session.add(past)
        await async_session.commit()
        return past

    @staticmethod
    async def _eval_view(async_session, view_id, rows):
        view = ProbeMonitorDataset(
            name="eval", dataset_id=(await async_session.get(ProbeMonitorDataset, view_id)).dataset_id,
            split="test", input_column="text", label_column="label",
            label_mapping={"yes": "positive", "no": "negative"}, role="eval",
            counts={"positive": 0, "negative": rows})
        async_session.add(view)
        await async_session.commit()
        return view

    @pytest.mark.asyncio
    async def test_a_past_runs_own_sets_are_sized_even_when_this_run_names_others(
        self, async_session, preflight_rows  # noqa: F811
    ):
        """The rate is the PAST run's seconds over the PAST run's rows: its sets must be looked up too."""
        from src.core.database import SyncSessionLocal
        from src.services import probe_run_time_forecast as tf

        model_id, view_id = preflight_rows
        theirs = await self._eval_view(async_session, view_id, 5000)
        mine = await self._eval_view(async_session, view_id, 3000)
        await self._past_run(async_session, model_id, view_id, eval_ids=[theirs.id])  # 1,000 rows in 100 s
        session = SyncSessionLocal()
        try:
            answer = tf.time_forecast_from_db(
                session, {"model_id": model_id, "eval_dataset_ids": [mine.id], "config": {"eval_max_rows": 2000}},
                {"checked": True, "rows": 80, "real_tokens": 4000, "swept_layers": 3})
        finally:
            session.close()
        assert answer["stages"]["evaluating"]["seconds"]["typical"] == pytest.approx(200.0)

    @pytest.mark.asyncio
    async def test_the_history_is_this_models_even_when_others_ran_since(
        self, async_session, preflight_rows, monkeypatch  # noqa: F811
    ):
        """The query reads at most HISTORY_LIMIT runs: another model's newer runs must not crowd this one out."""
        from src.core.database import SyncSessionLocal
        from src.models.model import Model, ModelStatus
        from src.services import probe_run_time_forecast as tf

        model_id, view_id = preflight_rows
        other = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="other", status=ModelStatus.READY, file_path="/x",
                      architecture="LlamaForCausalLM", params_count=1, architecture_config={"hidden_size": 8})
        async_session.add(other)
        await async_session.commit()
        mine = await self._past_run(async_session, model_id, view_id, eval_ids=[], hours_ago=5)
        await self._past_run(async_session, other.id, view_id, eval_ids=[], hours_ago=1)
        monkeypatch.setattr(tf, "HISTORY_LIMIT", 1)
        session = SyncSessionLocal()
        try:
            answer = tf.time_forecast_from_db(session, {"model_id": model_id, "config": {}},
                                              {"checked": True, "rows": 80, "real_tokens": 4000, "swept_layers": 3})
        finally:
            session.close()
        assert answer["forecast"] is True and [r["run_id"] for r in answer["basis"]["runs"]] == [mine.id]

    @pytest.mark.asyncio
    async def test_a_card_named_by_index_narrows_to_that_cards_runs(
        self, async_session, preflight_rows, monkeypatch  # noqa: F811
    ):
        """An index is resolved to its UUID as submit resolves it, so the runs measured on that card are used."""
        from src.core.database import SyncSessionLocal
        from src.services import gpu_placement
        from src.services import probe_run_time_forecast as tf

        model_id, view_id = preflight_rows
        on_a = await self._past_run(async_session, model_id, view_id, eval_ids=[], gpu_uuid="GPU-aaa", hours_ago=3)
        await self._past_run(async_session, model_id, view_id, eval_ids=[], gpu_uuid="GPU-bbb", hours_ago=1)
        resolved = []

        def resolve(requested, cards=None):
            resolved.append(requested)
            return {"0": "GPU-aaa", "1": "GPU-bbb"}.get(str(requested), "auto")

        monkeypatch.setattr(gpu_placement, "resolve_request", resolve)
        session = SyncSessionLocal()
        try:
            answer = tf.time_forecast_from_db(session, {"model_id": model_id, "config": {}, "gpu": "0"},
                                              {"checked": True, "rows": 80, "real_tokens": 4000, "swept_layers": 3})
        finally:
            session.close()
        assert resolved == ["0"]
        assert answer["basis"]["card"] == {"pinned": True, "cards": ["GPU-aaa"]}
        assert [r["run_id"] for r in answer["basis"]["runs"]] == [on_a.id]

    @pytest.mark.asyncio
    async def test_with_nothing_measured_it_says_so_and_the_memory_answer_stands(
        self, client, preflight_rows  # noqa: F811
    ):
        model_id, view_id = preflight_rows
        body = (await client.post(f"/api/v1{FORECAST}", json={
            "model_id": model_id, "train_dataset_id": view_id})).json()
        assert body["checked"] is True
        assert body["time_forecast"]["forecast"] is False
        assert "seconds" not in body["time_forecast"]
        assert body["time_forecast"]["summary"].startswith("No time forecast:")

    @pytest.mark.asyncio
    async def test_a_time_forecast_that_raises_costs_nothing_else(self, client, preflight_rows, monkeypatch):  # noqa: F811
        from src.services import probe_run_time_forecast as tf

        def boom(*args, **kwargs):
            raise RuntimeError("no history")

        monkeypatch.setattr(tf, "time_forecast_from_db", boom)
        model_id, view_id = preflight_rows
        body = (await client.post(f"/api/v1{FORECAST}", json={
            "model_id": model_id, "train_dataset_id": view_id})).json()
        assert body["checked"] is True
        assert body["time_forecast"] == {
            "forecast": False, "reason": "RuntimeError: no history",
            "summary": "No time forecast: it could not run (RuntimeError: no history)."}


# ── the MCP tool ─────────────────────────────────────────────────────────────────────────────


TIME = {
    "forecast": True, "complete": True, "summary": "About 5.6 h (3.8 h to 7.7 h) by the measured rates ...",
    "seconds": {"low": 13680.0, "typical": 20253.0, "high": 27684.0}, "eval_rows": 15895, "eval_max_rows": 5000,
    "eval_rows_by_set": {"pmd_2": 15895}, "stages": {"evaluating": {"rates": ["many"]}},
    "basis": {"runs": [{"run_id": "pmr_x"}]},
}
FITS = {"checked": True, "fits": True, "needed_gib": 12.0, "message": "fits"}


class _Studio:
    def __init__(self, forecast=None, fail=None):
        self.forecast, self.fail = forecast, fail
        self.calls = []

    async def post(self, path, json_body=None, **params):
        self.calls.append((path, json_body))
        if path == FORECAST:
            if self.fail is not None:
                raise self.fail
            return self.forecast
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        return {"id": "pmr_9", "task_id": "t_9", "status": "queued"}


def _tool(studio, name, *, approval=True, **arguments):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import probe_monitors

    mcp = MCPServer("t")
    probe_monitors.register(mcp, studio, MCPSettings(auth_token="x" * 32, probe_gpu_approval=approval))
    return asyncio.run(mcp._tool_manager._tools[name].fn(**arguments))  # noqa: SLF001


RUN = {"model_id": "m_1", "train_dataset_id": "pmd_1", "eval_dataset_ids": ["pmd_2"],
       "config": {"eval_max_rows": 5000}}
PAYLOAD = {"model_id": "m_1", "train_dataset_id": "pmd_1", "eval_dataset_ids": ["pmd_2"], "gpu": "auto",
           "config": {"eval_max_rows": 5000}}
CARD = {"forecast": True, "summary": TIME["summary"], "eval_rows": 15895, "eval_max_rows": 5000,
        "eval_rows_by_set": {"pmd_2": 15895}, "seconds": TIME["seconds"], "complete": True}


class TestTheStartToolCarriesIt:
    def test_the_first_answer_has_it_beside_the_memory_forecast(self):
        studio = _Studio({**FITS, "time_forecast": TIME})
        result = _tool(studio, "start_probe_training_run", **RUN)
        assert result["status"] == "pending_approval"
        assert result["time_forecast"] == TIME
        assert result["memory_forecast"] == FITS, "the time forecast is beside it, not inside it"

    def test_the_approval_request_carries_the_summary_and_the_eval_rows_once(self):
        studio = _Studio({**FITS, "time_forecast": TIME})
        _tool(studio, "start_probe_training_run", **RUN)
        filed = [body for path, body in studio.calls if path == "/mcp/approvals"]
        assert filed == [{"tool_name": "start_probe_training_run", "payload": PAYLOAD,
                          "forecast": {"time": CARD}}]

    def test_the_payload_approved_is_unchanged_so_consume_still_matches(self):
        studio = _Studio({**FITS, "time_forecast": TIME})
        _tool(studio, "start_probe_training_run", **RUN)
        forecast_body = studio.calls[0][1]
        filed = [body for path, body in studio.calls if path == "/mcp/approvals"][0]
        assert filed["payload"] == forecast_body == PAYLOAD
        assert "forecast" not in filed["payload"] and "time_forecast" not in filed["payload"]

    def test_a_forecast_route_without_one_is_said_on_the_card(self):
        studio = _Studio(FITS)
        result = _tool(studio, "start_probe_training_run", **RUN)
        assert result["time_forecast"]["forecast"] is False
        filed = [body for path, body in studio.calls if path == "/mcp/approvals"][0]
        assert filed["forecast"]["time"]["forecast"] is False
        assert filed["forecast"]["time"]["summary"].startswith("No time forecast")

    def test_a_failed_forecast_route_is_said_on_the_card(self):
        studio = _Studio(fail=BackendError(0, "unreachable"))
        result = _tool(studio, "start_probe_training_run", **RUN)
        assert result["status"] == "pending_approval"
        card = [body for path, body in studio.calls if path == "/mcp/approvals"][0]["forecast"]["time"]
        assert card["forecast"] is False and "unreachable" in card["summary"]

    def test_the_too_large_refusal_says_the_time_too(self):
        studio = _Studio({"checked": True, "fits": False, "message": "99 GiB", "time_forecast": TIME})
        result = _tool(studio, "start_probe_training_run", **RUN)
        assert result["refused"] == "probe_training_too_large" and result["time_forecast"] == TIME

    def test_the_approved_recall_files_nothing_and_forecasts_nothing(self):
        studio = _Studio({**FITS, "time_forecast": TIME})
        result = _tool(studio, "start_probe_training_run", approval_request_id="apr_1", **RUN)
        assert result["id"] == "pmr_9" and "time_forecast" not in result
        assert [path for path, _ in studio.calls] == ["/mcp/approvals/apr_1/consume", "/probe-monitors/runs"]

    def test_with_approval_off_the_answer_still_carries_it(self):
        studio = _Studio({**FITS, "time_forecast": TIME})
        result = _tool(studio, "start_probe_training_run", approval=False, **RUN)
        assert result["id"] == "pmr_9" and result["time_forecast"] == TIME


class TestTheForecastToolSendsTheEvaluationSets:
    def test_eval_calibration_and_card_reach_the_route(self):
        studio = _Studio({**FITS, "time_forecast": TIME})
        result = _tool(studio, "forecast_probe_training_run", model_id="m_1", train_dataset_id="pmd_1",
                       eval_dataset_ids=["pmd_2", "pmd_3"], calibration_dataset_id="pmd_c", gpu="GPU-x",
                       config={"eval_max_rows": 5000})
        assert studio.calls == [(FORECAST, {"model_id": "m_1", "train_dataset_id": "pmd_1",
                                            "config": {"eval_max_rows": 5000},
                                            "eval_dataset_ids": ["pmd_2", "pmd_3"],
                                            "calibration_dataset_id": "pmd_c", "gpu": "GPU-x"})]
        assert result["time_forecast"] == TIME

    def test_omitted_they_are_not_sent(self):
        studio = _Studio(FITS)
        _tool(studio, "forecast_probe_training_run", model_id="m_1", train_dataset_id="pmd_1")
        assert studio.calls == [(FORECAST, {"model_id": "m_1", "train_dataset_id": "pmd_1"})]


# ── the approval store ───────────────────────────────────────────────────────────────────────


class TestTheApprovalKeepsIt:
    @pytest.mark.asyncio
    async def test_filed_forecast_is_stored_and_listed_and_consume_ignores_it(self, client):
        payload = {"model_id": "m_1", "train_dataset_id": "pmd_1", "eval_dataset_ids": [], "gpu": "auto"}
        created = await client.post("/api/v1/mcp/approvals", json={
            "tool_name": "start_probe_training_run", "payload": payload, "forecast": {"time": CARD}})
        assert created.status_code == 201, created.text
        approval_id = created.json()["id"]
        assert created.json()["forecast"] == {"time": CARD}
        listed = (await client.get("/api/v1/mcp/approvals", params={"status": "pending"})).json()["approvals"]
        assert [a["forecast"] for a in listed if a["id"] == approval_id] == [{"time": CARD}]
        assert (await client.post(f"/api/v1/mcp/approvals/{approval_id}/approve")).status_code == 200
        consumed = await client.post(f"/api/v1/mcp/approvals/{approval_id}/consume", json={
            "tool_name": "start_probe_training_run", "payload": payload})
        assert consumed.status_code == 200, consumed.text

    @pytest.mark.asyncio
    async def test_a_request_without_one_reads_null(self, client):
        created = await client.post("/api/v1/mcp/approvals", json={
            "tool_name": "start_probe_training_run", "payload": {"model_id": "m_2"}})
        assert created.status_code == 201 and created.json()["forecast"] is None

    @pytest.mark.asyncio
    async def test_an_oversized_forecast_is_refused(self, client):
        created = await client.post("/api/v1/mcp/approvals", json={
            "tool_name": "start_probe_training_run", "payload": {"model_id": "m_3"},
            "forecast": {"time": {"summary": "x" * 20_000}}})
        assert created.status_code == 422
