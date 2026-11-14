"""A corpus-sized evaluation view is never scored whole (ENH-006 G15, 2026-11-03).

Run `pmr_b63a7e7c94ca` listed a 1,000,000-row everyday-text view among its evaluation sets and scored it
whole: 150,730 batches, an ETA of 347 hours, ~95 GB of worker memory, killed by the kernel — and the six
sets it had finished were lost with it, because their rows were committed only after the last set.

Pinned here: the refusal at both submit points, the run's sample reaching the loop, and the per-set
commit (a set that fails leaves the sets before it, and nothing of its own).
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import settings
from src.schemas.probe_monitor import MAX_UNSAMPLED_EVAL_ROWS, ProbeRunConfig
from src.services import eval_sample_size, probe_monitor_render
from src.services import probe_monitor_run as run_module
from tests.support import family_fixtures as ff
from tests.support.detector_fixtures import dataset_row, write_split

API = "/api/v1/probe-monitors"
BIG = {"positive": 0, "negative": 1_000_000}


def _view(counts, role="everyday_text"):
    return SimpleNamespace(id=f"pmd_{uuid.uuid4().hex[:6]}", name="v", role=role, counts=counts)


# ── the pure decision ─────────────────────────────────────────────────────────────────────────


def test_a_view_above_the_limit_is_named_with_its_size():
    big, small = _view(BIG), _view({"positive": 248, "negative": 1136}, role="eval")
    assert eval_sample_size.too_large_to_score_whole([small, big], None) == [(big, 1_000_000)]


def test_the_limit_itself_is_scored_whole():
    at = _view({"positive": 0, "negative": MAX_UNSAMPLED_EVAL_ROWS})
    over = _view({"positive": 1, "negative": MAX_UNSAMPLED_EVAL_ROWS})
    assert eval_sample_size.too_large_to_score_whole([at, over], None) == [(over, MAX_UNSAMPLED_EVAL_ROWS + 1)]


def test_a_sample_bounds_every_view():
    assert eval_sample_size.too_large_to_score_whole([_view(BIG)], 5000) == []


def test_unrecorded_counts_are_not_guessed():
    assert eval_sample_size.scored_rows(_view({})) is None
    assert eval_sample_size.too_large_to_score_whole([_view({}), _view(None)], None) == []


def test_the_run_config_takes_a_sample_and_defaults_to_none():
    assert ProbeRunConfig().eval_max_rows is None
    assert ProbeRunConfig(eval_max_rows=5000).eval_max_rows == 5000
    with pytest.raises(ValueError):
        ProbeRunConfig(eval_max_rows=99)


# ── the two submit points ─────────────────────────────────────────────────────────────────────


async def _world(session):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/x", architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="d", source="local", status=DatasetStatus.READY,
                      raw_path="/x", extra_metadata={})
    session.add_all([model, dataset])
    await session.flush()

    def view(name, role, counts):
        return ProbeMonitorDataset(name=name, dataset_id=dataset.id, input_column="i", label_column="l",
                                   label_mapping={"1": "positive"}, role=role, counts=counts)

    views = {"train": view("train view", "train", {"positive": 30, "negative": 30}),
             "eval": view("eval view", "eval", {"positive": 30, "negative": 30}),
             "everyday": view("everyday-text-openwebtext", "everyday_text", BIG)}
    session.add_all(views.values())
    await session.commit()
    return model, {k: v.id for k, v in views.items()}


async def test_a_run_naming_a_corpus_sized_view_is_refused_before_anything_is_queued(
        client, async_session, monkeypatch):
    from src.api.v1.endpoints import probe_monitors as ep

    model, views = await _world(async_session)
    queued = []
    monkeypatch.setattr(ep, "gpu_delay", lambda *a, **k: queued.append(a) or (lambda **kw: None))
    r = await client.post(f"{API}/runs", json={
        "model_id": model.id, "train_dataset_id": views["train"],
        "eval_dataset_ids": [views["eval"], views["everyday"]],
    })
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert views["everyday"] in detail and "1,000,000 rows" in detail and "config.eval_max_rows" in detail
    assert views["eval"] not in detail
    assert queued == []


async def test_a_run_with_a_sample_passes_the_size_check(client, async_session, monkeypatch):
    from src.api.v1.endpoints import probe_monitors as ep

    model, views = await _world(async_session)
    monkeypatch.setattr(ep, "gpu_delay", lambda *a, **k: (lambda **kw: SimpleNamespace(id="t")))
    r = await client.post(f"{API}/runs", json={
        "model_id": model.id, "train_dataset_id": views["train"],
        "eval_dataset_ids": [views["eval"], views["everyday"]], "config": {"eval_max_rows": 5000},
    })
    assert "eval_max_rows" not in r.text, r.text


async def test_an_evaluation_of_a_corpus_sized_view_needs_a_sample(client, async_session, monkeypatch):
    from src.api.v1.endpoints import probe_monitors as ep
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorRun

    model, views = await _world(async_session)
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=views["train"], eval_dataset_ids=[],
                          config={}, status="completed")
    async_session.add(run)
    await async_session.flush()
    probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, val_metrics={})
    async_session.add(probe)
    await async_session.commit()

    async def ok(*_a, **_k):
        return None

    monkeypatch.setattr(ep, "refuse_precision_mismatch", ok)
    monkeypatch.setattr(ep, "refuse_render_mismatch", ok)
    queued = []
    monkeypatch.setattr(ep, "gpu_delay",
                        lambda *a, **k: (lambda **kw: queued.append(kw) or SimpleNamespace(id="t")))
    r = await client.post(f"{API}/probes/{probe.id}/evaluate", json=[views["eval"], views["everyday"]])
    assert r.status_code == 422, r.text
    assert "sample_rows" in r.json()["detail"] and queued == []

    r = await client.post(f"{API}/probes/{probe.id}/evaluate", params={"sample_rows": 5000},
                          json=[views["eval"], views["everyday"]])
    assert r.status_code == 202, r.text
    assert queued == [{"probe_id": probe.id, "dataset_ids": [views["eval"], views["everyday"]],
                       "sample_rows": 5000}]


# ── the run threads its sample into the loop ──────────────────────────────────────────────────


def test_the_run_passes_eval_max_rows_to_the_evaluation_loop():
    """By AST: the stage's call to `_evaluate_probes_on_sets` passes `sample_rows` from the run config."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(run_module))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "_evaluate_probes_on_sets"]
    run_calls = [c for c in calls if any(isinstance(a, ast.Call) and getattr(a.func, "id", None) == "list"
                                         for a in c.args)]
    assert len(run_calls) == 1, "the run stage's evaluation call moved"
    keyword = {k.arg: k.value for k in run_calls[0].keywords}.get("sample_rows")
    assert keyword is not None, "the run stage does not pass its evaluation sample"
    assert ast.unparse(keyword) == "context.config.get('eval_max_rows')"


# ── per-set commit ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "miserver_api_url", "", raising=False)
    return tmp_path


@pytest.fixture
def sync_db(async_engine):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", "")))
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


async def test_a_failed_set_keeps_the_sets_before_it_and_nothing_of_its_own(
        async_session, data_dir, sync_db, monkeypatch):
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorEvaluation
    from src.services import probe_monitor_capture

    def render_all(tokenizer, conversations, **kwargs):
        return [SimpleNamespace(i=i) for i, _ in enumerate(conversations)], SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr(probe_monitor_render, "render_all", render_all)
    views = []
    for n in (40, 40):
        ds = await dataset_row(async_session, write_split(
            data_dir, {"text": [f"row {i}" for i in range(n)], "label": [str(i % 2) for i in range(n)]}))
        views.append(await ff.view(async_session, ds, key=None, role="eval"))
    model = await ff.model(async_session)
    source = await ff.source(async_session)
    first = await ff.member(async_session, model, source, key=None)
    second = await ff.member(async_session, model, source, key=None)
    await async_session.commit()

    calls = []

    def scores(model_, rendered, specs, **kwargs):
        calls.append(len(rendered))
        per = [[SimpleNamespace(aggregate=float(i % 2), n_scored=3) for i, _ in enumerate(rendered)]
               for _ in specs]
        if len(calls) == 2:
            per[1] = per[1][:-1]           # the second set's arrays misalign for the SECOND probe only
        return per

    monkeypatch.setattr(probe_monitor_capture, "forward_scores_many", scores)
    trained = SimpleNamespace(rule="mean", rule_params={})
    rows = [sync_db.query(ProbeMonitor).filter(ProbeMonitor.id == p.id).one() for p in (first, second)]
    ctx = SimpleNamespace(seed=5, scope="all", max_length=64, skip_template_prefix=False, target_fpr=0.01, artifact_dir=data_dir / "run")
    ctx.artifact_dir.mkdir()
    group = [(row, None, trained, None, None) for row in rows]

    with pytest.raises(ValueError, match="39 scores against 40 labels"):
        run_module._evaluate_probes_on_sets(sync_db, group, ctx, None, None, "LlamaForCausalLM",
                                            [views[0].id, views[1].id])
    # The failure handler commits (it stamps the run failed): nothing of the failed set may land with it.
    sync_db.commit()

    kept = sync_db.query(ProbeMonitorEvaluation).all()
    assert sorted((e.probe_id, e.dataset_id) for e in kept) == sorted(
        (row.id, views[0].id) for row in rows)
    for evaluation in kept:
        assert np.load(evaluation.scores_path).shape == (40,)
    on_disk = sorted(p.name for p in ctx.artifact_dir.iterdir())
    assert on_disk and all(views[1].id not in name for name in on_disk)
