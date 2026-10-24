"""Sampled evaluations: every member scores THE SAME rows of a calibration view (Feature 041 T20, T21, PT-11;
FTASKS 4.1 – 4.6).

The sample's seed is the VIEW and the SIZE only (`sample_seed_for`), never the run — two members trained by
different runs sample identical rows. Positions recorded after a sample are positions in the SPLIT (T21,
040 S2's `keep[example.index]` rewrite, pinned here as 041's dependency). A calibration view evaluated
without a sample is refused before anything is queued (the unguarded million-row GPU pass, PT-11).

`render_all` (the chat render) is replaced by an identity so the sampler, the builder and the writer run for
real without a tokenizer; the model forward is replaced in the writer test only.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import settings
from src.services import probe_monitor_render
from src.services import probe_monitor_run as run_module
from tests.support import family_fixtures as ff
from tests.support.detector_fixtures import dataset_row, write_split

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    return tmp_path


@pytest.fixture
def no_render(monkeypatch):
    def render_all(tokenizer, conversations, **kwargs):
        return [SimpleNamespace(i=i) for i, _ in enumerate(conversations)], SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr(probe_monitor_render, "render_all", render_all)


@pytest.fixture
def sync_db(async_engine):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", "")))
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


async def big_view(async_session, data_dir, n=10_000, role="calibration"):
    labels = ["0"] * n if role == "calibration" else [str(i % 2) for i in range(n)]
    ds = await dataset_row(async_session, write_split(data_dir, {"text": [f"row {i}" for i in range(n)], "label": labels}))
    view = await ff.view(async_session, ds, key=None, role=role)
    await async_session.commit()
    return view


def context(seed):
    return SimpleNamespace(seed=seed, scope="all", max_length=64)


def indices(sync_db, view_id, ctx, *, max_rows, sample_seed=None):
    from src.models.probe_monitor import ProbeMonitorDataset

    view = sync_db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == view_id).one()
    examples, _r, _s, _l = run_module._prepare_examples(sync_db, view, ctx, None, role=view.role, max_rows=max_rows,
                                                        sample_seed=sample_seed)
    return examples


def test_the_seed_is_the_view_and_the_size():
    a = run_module.sample_seed_for("pmd_a", 500)
    assert a == run_module.sample_seed_for("pmd_a", 500)
    assert a != run_module.sample_seed_for("pmd_b", 500) and a != run_module.sample_seed_for("pmd_a", 501)


async def test_two_members_sample_the_same_rows(async_session, data_dir, sync_db, no_render):
    view = await big_view(async_session, data_dir)
    seed = run_module.sample_seed_for(view.id, 500)
    first = [e.index for e in indices(sync_db, view.id, context(11), max_rows=500, sample_seed=seed)]
    second = [e.index for e in indices(sync_db, view.id, context(97), max_rows=500, sample_seed=seed)]
    assert len(first) == 500 and first == second


async def test_run_calibration_stage_unchanged(async_session, data_dir, sync_db, no_render):
    """No `sample_seed` → the run's own seed draws the rows, exactly as before."""
    view = await big_view(async_session, data_dir)
    got = [e.index for e in indices(sync_db, view.id, context(11), max_rows=500)]
    assert got == run_module.subsample_indices(10_000, 500, 11)


async def test_positions_are_split_rows_after_a_sample(async_session, data_dir, sync_db, no_render):
    """T21: each sampled example's `index` is its row in the SPLIT — the row's own text says so."""
    view = await big_view(async_session, data_dir, role="eval")
    seed = run_module.sample_seed_for(view.id, 500)
    for example in indices(sync_db, view.id, context(3), max_rows=500, sample_seed=seed):
        assert f"row {example.index}" in str(example.messages)


# ── the route (PT-11, 4.2 – 4.3) ─────────────────────────────────────────────────────────────


@pytest.fixture
def dispatched(monkeypatch):
    sent = []

    class Task:
        id = "task-eval"

    monkeypatch.setattr("src.api.v1.endpoints.probe_monitors.gpu_delay",
                        lambda task, gpu: (lambda **kw: sent.append((task.name, kw)) or Task()))
    monkeypatch.setattr("src.api.v1.endpoints.probe_monitors.refuse_precision_mismatch", _ok)
    monkeypatch.setattr("src.api.v1.endpoints.probe_monitors.refuse_render_mismatch", _ok)
    return sent


async def _ok(probe_id):
    return None


async def probe_and_views(async_session, request):
    from tests.support.random_behaviour import rng_for

    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    probe = await ff.member(async_session, model, ds, key=None)
    cal = await ff.view(async_session, ds, key=None, role="calibration")
    test = await ff.view(async_session, ds, key=None, role="eval")
    await async_session.commit()
    _ = rng_for(request.node.nodeid)
    return probe, cal, test


async def test_calibration_view_without_sample_refused(client, async_session, request, dispatched):
    from src.schemas.probe_monitor import DEFAULT_CALIBRATION_MAX_ROWS

    probe, cal, test = await probe_and_views(async_session, request)
    response = await client.post(f"{API}/probes/{probe.id}/evaluate", json=[test.id, cal.id])
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert (detail["code"], detail["view"], detail["default_sample_rows"]) == (
        "calibration_view_needs_sample", cal.id, DEFAULT_CALIBRATION_MAX_ROWS)
    assert detail["next"][0]["arguments"] == {"probe_id": probe.id, "dataset_ids": [test.id, cal.id],
                                              "sample_rows": DEFAULT_CALIBRATION_MAX_ROWS}
    assert dispatched == []


async def test_sample_rows_reach_the_task(client, async_session, request, dispatched):
    probe, cal, test = await probe_and_views(async_session, request)
    response = await client.post(f"{API}/probes/{probe.id}/evaluate", params={"sample_rows": 700},
                                 json=[test.id, cal.id])
    assert response.status_code == 202, response.text
    assert dispatched == [("src.workers.probe_monitor_tasks.evaluate_probe_monitor",
                           {"probe_id": probe.id, "dataset_ids": [test.id, cal.id], "sample_rows": 700})]
    # Without a calibration view nothing changes: no sample, the old call.
    dispatched.clear()
    await client.post(f"{API}/probes/{probe.id}/evaluate", json=[test.id])
    assert dispatched == [("src.workers.probe_monitor_tasks.evaluate_probe_monitor",
                           {"probe_id": probe.id, "dataset_ids": [test.id], "sample_rows": None})]


def test_the_task_threads_sample_rows_to_evaluate_probe(monkeypatch):
    from src.workers import probe_monitor_tasks

    seen = []
    monkeypatch.setattr("src.services.probe_monitor_run.evaluate_probe",
                        lambda db, probe_id, dataset_ids, sample_rows=None: seen.append((probe_id, dataset_ids, sample_rows))
                        or {"ok": True})
    monkeypatch.setattr(probe_monitor_tasks, "_require_row", lambda *a: None)
    result = probe_monitor_tasks.evaluate_probe_monitor.run("pm_1", ["pmd_1"], sample_rows=321)
    assert result == {"ok": True} and seen == [("pm_1", ["pmd_1"], 321)]


# ── the writer (4.3 metrics.sample, 4.4 refused calibration keeps its arrays) ──────────────


async def test_metrics_sample_and_refused_calibration_keeps_arrays(async_session, data_dir, sync_db, no_render,
                                                                   monkeypatch):
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorEvaluation
    from src.services import probe_monitor_capture

    view = await big_view(async_session, data_dir, n=3000)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    probe = await ff.member(async_session, model, ds, key=None)
    await async_session.commit()

    def scores(model_, rendered, specs, **kwargs):
        return [[SimpleNamespace(aggregate=float(i % 7) / 7.0, n_scored=5) for i, _ in enumerate(rendered)]
                for _ in specs]

    monkeypatch.setattr(probe_monitor_capture, "forward_scores_many", scores)
    row = sync_db.query(ProbeMonitor).filter(ProbeMonitor.id == probe.id).one()
    trained = SimpleNamespace(rule="mean", rule_params={})
    ctx = SimpleNamespace(seed=5, scope="all", max_length=64, target_fpr=0.01, artifact_dir=data_dir / "run")
    ctx.artifact_dir.mkdir()
    written = run_module._evaluate_probes_on_sets(sync_db, [(row, None, trained, None, None)], ctx, None, None,
                                                  "LlamaForCausalLM", [view.id], sample_rows=400)
    assert written == 1
    evaluation = sync_db.query(ProbeMonitorEvaluation).filter(ProbeMonitorEvaluation.probe_id == probe.id).one()
    assert evaluation.status == "refused"                      # no positives on a calibration view
    assert evaluation.metrics["sample"] == {"rows": 400, "seed_basis": "view+rows", "source_rows": 3000}
    for column in ("scores_path", "positions_path", "n_scored_path", "labels_path"):
        assert np.load(getattr(evaluation, column)).shape == (400,)
    positions = np.load(evaluation.positions_path)
    assert positions.tolist() == run_module.subsample_indices(3000, 400, run_module.sample_seed_for(view.id, 400))


# ── MCP (4.5): the approval binds the sample ─────────────────────────────────────────────────


class _Client:
    def __init__(self):
        self.calls = []

    async def post(self, path, json_body=None, **params):
        self.calls.append(("POST", path, json_body, params))
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        return {"task_id": "t", "status": "queued"}


def _tool(approval):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import probe_monitors

    client = _Client()
    mcp = MCPServer("t")
    probe_monitors.register(mcp, client, MCPSettings(auth_token="x" * 32, probe_gpu_approval=approval))
    return mcp._tool_manager._tools["evaluate_probe_monitor"].fn, client  # noqa: SLF001


def test_the_approval_payload_carries_the_sample():
    fn, client = _tool(True)
    asyncio.run(fn(probe_id="pm_1", dataset_ids=["pmd_c"], sample_rows=2500))
    assert client.calls == [("POST", "/mcp/approvals", {"tool_name": "evaluate_probe_monitor", "payload": {
        "probe_id": "pm_1", "dataset_ids": ["pmd_c"], "sample_rows": 2500}}, {})]


def test_the_sample_reaches_the_route_as_a_query():
    fn, client = _tool(False)
    asyncio.run(fn(probe_id="pm_1", dataset_ids=["pmd_c"], sample_rows=2500))
    assert client.calls == [("POST", "/probe-monitors/probes/pm_1/evaluate", ["pmd_c"], {"sample_rows": 2500})]


def test_the_approval_summary_names_the_sample():
    from src.api.v1.endpoints.mcp_approvals import _summarize

    text = _summarize("evaluate_probe_monitor", {"probe_id": "pm_1", "dataset_ids": ["a"], "sample_rows": 2500})
    assert "a sample of 2500 rows" in text
    assert "sample" not in _summarize("evaluate_probe_monitor", {"probe_id": "pm_1", "dataset_ids": ["a"]})
