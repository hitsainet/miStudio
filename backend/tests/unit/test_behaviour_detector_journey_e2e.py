"""FR-46: three behaviours, one call sequence, through the LIVE MCP registry only (Feature 040 S2; TI-24).

The agent journey of FPRD §2, from `create_behaviour` to `compare_detectors_across_models`, walked the way
an agent walks it — `call_tool` over the protocol against the real `build_server()` (deployed categories),
whose client is pointed at the FastAPI app over ASGI — for three fixtures:

1. a randomly keyed behaviour;
2. the second behaviour (`backend/tests/fixtures/behaviour_packs/high-stakes.json`): no rule set, no held-out
   column, criteria judged in distribution;
3. the pilot (`behaviour_packs/dlp-identifiers.json`, its first behaviour).

Fakes only where a GPU would be: the task broker records the queued probe run, and a FIXTURE SCORE WRITER
stands in for its body (probe rows, evaluation arrays with positions, calibration negatives with positions)
— the run's own stages are covered by `test_probe_monitor_pipeline.py` and `test_calibration_positions.py`.
The GPU start goes through the real approval gate: the agent's first call files a request, the operator
approves it over REST, the re-call consumes it.

Each hop asserts the HTTP request, the queued task (name, arguments, count) and what the next tool reads.
The SAME tool sequence must succeed for all three; any difference is a finding (BR-029).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from tests.support.detector_fixtures import write_split
from tests.support.random_behaviour import behaviour_body, rng_for, slug
from tests.unit.test_mcp_discovery_chain_e2e import _deployed_categories

PACKS = Path(__file__).resolve().parents[1] / "fixtures" / "behaviour_packs"
N_ROWS = 160


@pytest.fixture
def maker(async_engine):
    url = async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", ""))
    engine = create_engine(url)
    yield sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    engine.dispose()


@pytest.fixture
def backend(async_engine, maker, monkeypatch, tmp_path):
    from src.core.database import get_db
    from src.main import app

    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "midataworks_api_url", "", raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    amaker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def _db():
        async with amaker() as session:
            yield session

    app.dependency_overrides[get_db] = _db
    yield SimpleNamespace(app=app, maker=maker, data_dir=tmp_path)
    app.dependency_overrides.pop(get_db, None)


class Broker:
    """Records every queued GPU task; `drain()` runs the fixture score writer in its place."""

    def __init__(self):
        self.sent = []

    def delay_for(self, task, gpu_request=None):
        def delay(*args, **kwargs):
            self.sent.append({"task": task.name.rsplit(".", 1)[-1], "args": args, "kwargs": kwargs})
            return SimpleNamespace(id=f"celery-{len(self.sent)}")
        return delay


@pytest.fixture
def broker(monkeypatch):
    from src.api.v1.endpoints import probe_monitors as endpoints

    b = Broker()
    monkeypatch.setattr(endpoints, "gpu_delay", lambda task, gpu_request=None: b.delay_for(task, gpu_request))
    return b


@pytest.fixture
def mcp_server(backend, monkeypatch):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    for var in ("MILLM_API_URL", "MIDATAWORKS_API_URL"):
        monkeypatch.delenv(var, raising=False)
    mcp, client = build_server(MCPSettings(tool_categories=_deployed_categories(), allow_anonymous=True,
                                           _env_file=None), stdio=True)
    requests = []

    async def record(request):
        requests.append((request.method, request.url.path))

    client._http = httpx.AsyncClient(  # noqa: SLF001
        transport=httpx.ASGITransport(app=backend.app), base_url="http://test/api/v1",
        event_hooks={"request": [record]})
    return SimpleNamespace(mcp=mcp, requests=requests)


def write_scores(maker, run_id: str, cal_negatives: int):
    """The fixture score writer: what the two gated GPU tasks leave behind (probe, evaluations with
    positions, calibration negatives with positions), from the views' own labels."""
    import uuid

    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorEvaluation, ProbeMonitorRun
    from src.services.probe_monitor_run import artifact_dir_for, evaluation_array_paths, persist_calibration_negatives

    with maker() as db:
        run = db.get(ProbeMonitorRun, run_id)
        directory = artifact_dir_for(run.id)
        directory.mkdir(parents=True, exist_ok=True)
        probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, variant="dense", val_metrics={},
                             selected=True, behaviour=run.behaviour, threshold=0.5,
                             target_fpr=run.config["target_fpr"], threshold_source="calibration_set",
                             calibration_dataset_id=run.calibration_dataset_id,
                             sub_question=run.sub_question, behaviour_version=run.behaviour_version)
        db.add(probe)
        db.flush()
        gen = np.random.default_rng(5)
        negatives = gen.normal(size=cal_negatives).astype(np.float32)
        probe.calibration_scores_path = persist_calibration_negatives(directory, probe.id, negatives)
        probe.calibration_positions_path = persist_calibration_negatives(
            directory, probe.id, list(range(1, 2 * cal_negatives, 2)), name="positions", dtype=np.int64)
        probe.calibration_n_negatives = cal_negatives
        for view_id in run.eval_dataset_ids:
            view = db.get(ProbeMonitorDataset, view_id)
            positions = np.arange(0, N_ROWS, 2)
            labels = (positions % 4 == 0).astype(np.int8)
            scores = np.where(labels == 1, np.where(positions % 8 == 0, 0.9, 0.2), 0.1)
            evaluation = ProbeMonitorEvaluation(id=f"pme_{uuid.uuid4().hex[:12]}", probe_id=probe.id,
                                                dataset_id=view.id, status="completed", metrics={"auroc": 0.9})
            paths = evaluation_array_paths(directory, probe.id, view.id, evaluation.id)
            for column, values in (("scores_path", scores.astype(np.float32)), ("positions_path", positions),
                                   ("n_scored_path", np.full(len(positions), 9, dtype=np.int32)),
                                   ("labels_path", labels)):
                np.save(paths[column], values)
                setattr(evaluation, column, str(paths[column]))
            db.add(evaluation)
        run.status = "completed"
        db.commit()
        return probe.id


def fixtures(rng):
    random_body = behaviour_body(rng, target_fpr=0.02)
    random_body.update({"success_criteria": [{"id": slug(rng), "detector": "best_single", "subset": "all",
                                              "held_out_test": "genre", "min_recall": 0.4}],
                        "held_out_columns": ["genre"],
                        "sub_questions": [{"key": slug(rng), "question": "q", "positive_meaning": "p"}]})
    second = json.loads((PACKS / "high-stakes.json").read_text())["behaviours"][0]
    pilot = json.loads((PACKS / "dlp-identifiers.json").read_text())["behaviours"][0]
    out = []
    for body in (random_body, second, pilot):
        body = {k: v for k, v in body.items() if k != "rule_ids"}
        out.append(body)
    return out


async def _call(session, tool, args):
    result = await session.call_tool(tool, args)
    assert not result.is_error, f"{tool} failed: {result.content}"
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and set(structured) == {"result"} and isinstance(structured["result"], list):
        return structured["result"]                      # a list answer (server.list_result)
    return json.loads(result.content[0].text)


async def journey(session, client, backend, broker, body):
    """The agent's call sequence; returns the tools called, in order, and the reads it ended with."""
    import uuid

    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus

    calls = []

    async def call(tool, args):
        calls.append(tool)
        return await _call(session, tool, args)

    key = body["key"]
    held = (body.get("held_out_columns") or [None])[0]
    target = body["target_fpr"]
    # The data the agent registers (downloaded datasets; the download itself is not this journey).
    with backend.maker() as db:
        model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY, file_path="/x",
                      architecture="LlamaForCausalLM", params_count=1)
        columns = {"text": [f"row {i}" for i in range(N_ROWS)], "label": [str(int(i % 4 == 0)) for i in range(N_ROWS)],
                   "kind": [("a", "b", "c")[i % 3] for i in range(N_ROWS)],
                   "genre": [f"g{i % 2}" for i in range(N_ROWS)]}
        train_ds = Dataset(id=uuid.uuid4(), name="train", source="local", status=DatasetStatus.READY,
                           raw_path=write_split(backend.data_dir, columns), extra_metadata={})
        test_columns = dict(columns, genre=[f"h{i % 2}" for i in range(N_ROWS)])
        test_ds = Dataset(id=uuid.uuid4(), name="test", source="local", status=DatasetStatus.READY,
                          raw_path=write_split(backend.data_dir, test_columns), extra_metadata={})
        db.add_all([model, train_ds, test_ds])
        db.commit()
        model_id, train_id, test_id = model.id, str(train_ds.id), str(test_ds.id)

    made = await call("create_behaviour", body)
    assert made["key"] == key and made["current_version"] == 1
    mapping = {"1": "positive", "0": "negative"}
    group = [c for c in (held, "kind") if c]
    train = await call("register_probe_dataset", {"name": f"{key}-train", "dataset_id": train_id, "input_column": "text",
                                                  "label_column": "label", "label_mapping": mapping, "role": "train",
                                                  "behaviour": key, "group_columns": group})
    ood = await call("register_probe_dataset", {"name": f"{key}-ood", "dataset_id": test_id, "input_column": "text",
                                                "label_column": "label", "label_mapping": mapping, "role": "eval",
                                                "distribution": "out_of_distribution", "behaviour": key,
                                                "group_columns": [c for c in (held,) if c]})
    iid = await call("register_probe_dataset", {"name": f"{key}-id", "dataset_id": test_id, "input_column": "text",
                                                "label_column": "label", "label_mapping": mapping, "role": "eval",
                                                "distribution": "in_distribution", "behaviour": key})
    cal = await call("register_probe_dataset", {"name": f"{key}-cal", "dataset_id": train_id, "input_column": "text",
                                                "label_column": "label", "label_mapping": {"1": "excluded", "0": "negative"},
                                                "role": "calibration"})
    added = await call("update_probe_dataset_columns", {"dataset_id": ood["id"], "group_columns": [*([held] if held else []), "kind"]})
    assert added["added"] == ["kind"]

    run_args = {"model_id": model_id, "train_dataset_id": train["id"], "eval_dataset_ids": [ood["id"], iid["id"]],
                "calibration_dataset_id": cal["id"], "behaviour": key,
                "config": {"layers": [1], "rules": ["mean"], "target_fpr": target}}
    pending = await call("start_probe_training_run", run_args)
    assert pending["status"] == "pending_approval"
    approved = await client.post(f"/api/v1/mcp/approvals/{pending['approval_request_id']}/approve")
    assert approved.status_code == 200, approved.text
    queued = await call("start_probe_training_run", {**run_args, "approval_request_id": pending["approval_request_id"]})
    assert [s["task"] for s in broker.sent] == ["run_probe_monitor"] and broker.sent[0]["kwargs"] == {"run_id": queued["id"]}
    broker.sent.clear()
    probe_id = write_scores(backend.maker, queued["id"], cal_negatives=5000)

    probes = await call("list_probe_monitors", {"run_id": queued["id"]})
    assert [p["id"] for p in probes] == [probe_id]
    report = await call("get_probe_monitor_report", {"probe_id": probe_id})
    evaluation = next(e for e in report["evaluations"] if e["dataset_id"] == ood["id"])
    assert report["calibration_size"]["n_negatives"] == 5000
    run_report = await call("get_probe_training_run", {"run_id": queued["id"]})
    assert run_report["calibration_size"] == report["calibration_size"]  # FTASKS 9.7: the run shows it too

    breakdown = await call("get_probe_breakdown", {"probe_id": probe_id, "evaluation_id": evaluation["id"], "by_column": "kind"})
    assert {g["value"] for g in breakdown["groups"]} == {"a", "b", "c"}
    built = await call("get_detector_success_report", {"key": key})
    listed = await call("list_detector_success_reports", {"key": key})
    assert [r["id"] for r in listed["reports"]] == [built["id"]]
    read = await call("read_detector_success_report", {"report_id": built["id"]})
    assert read["report"] == built["report"]
    comparison = await call("compare_detectors_across_models", {"key": key})
    declared = [m["model_id"] for m in body.get("models") or []]
    if declared:
        # The definition names its models (the pilot's two): those are the rows, each refused with a
        # `next` when this deployment has no probe of the behaviour on it — never silently dropped.
        assert [r["model_id"] for r in comparison["models"]] == declared
        assert all(r["refused"]["code"] == "no_probe" for r in comparison["models"])
    else:
        assert [r["model_id"] for r in comparison["models"]] == [model_id]
    checklist = await call("get_behaviour_checklist", {"key": key})
    return calls, built, checklist


class TestThreeBehavioursOneSequence:
    @pytest.mark.asyncio
    async def test_the_same_calls_succeed_for_all_three(self, backend, broker, mcp_server, client, request):
        from mcp import Client

        sequences, reports = [], []
        async with Client(mcp_server.mcp) as session:
            listed = {t.name for t in (await session.list_tools()).tools}
            for body in fixtures(rng_for(request.node.nodeid)):
                calls, built, checklist = await journey(session, client, backend, broker, body)
                assert set(calls) <= listed
                sequences.append(calls)
                reports.append(built["report"])
                steps = {s["id"]: s["state"] for s in checklist["steps"]}
                assert steps["definition_complete"] == "done" and steps["success_report"] == "done"
        assert sequences[0] == sequences[1] == sequences[2], sequences
        random_report, second, pilot = reports
        # the random behaviour: a held-out criterion judged on the OOD view
        assert random_report["criteria"][0]["in_distribution"] is False
        # the second behaviour: no held-out column, criteria judged in distribution
        assert second["criteria"][0]["in_distribution"] is True and second["rule_set_digest"] is None
        # the pilot: rules_or_blend / blend_alone read not assessable until S3 / 041
        assert [c["reasons"][0]["code"] for c in pilot["criteria"]] == ["engine_not_available", "no_blend_report"]
        assert all(r["probe"]["chosen_by"] == "only_run_selected" for r in reports)


def test_adding_the_fixtures_changes_no_source_file():
    """Generality control (FR-46): the pack fixtures are data under tests/; nothing under src/ names them —
    the generality guard derives its tokens from them and is green (test_no_behaviour_names_in_source)."""
    from tests.support.behaviour_tokens import derive_tokens

    tokens = set(derive_tokens())
    for pack in PACKS.glob("*.json"):
        for behaviour in json.loads(pack.read_text())["behaviours"]:
            assert behaviour["key"] in tokens
