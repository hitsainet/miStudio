"""`GET /probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/scores` (035 FR-13 – FR-16).

Driven through the REAL app. The arrays are written by `evaluation_array_paths` — the writer's own
naming — under `<data_dir>/probe_monitors/<run>`, so the route's root check sees what production
writes.

MUTATION CONTROLS (recorded in 0xcc/reviews/035_implementation_controls_2026-10-18.md):
  C17 unregister the route (decorator removed)       → every 200 test gets a 404
  C18 drop the `positions_path` (409) check           → the legacy test fails
  C19 drop the length-mismatch check                  → the mismatch test fails
  C22 serve rows from a positions slice off by one    → the concatenation test fails
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from src.core.config import settings
from src.services.probe_monitor_metrics import auroc
from src.services.probe_monitor_run import artifact_dir_for, evaluation_array_paths

pytestmark = pytest.mark.asyncio
COMMIT = "d" * 40
N = 25


async def _seed(session, *, revision=COMMIT, status="completed", auroc_value=0.75):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import (
        ProbeMonitor, ProbeMonitorDataset, ProbeMonitorEvaluation, ProbeMonitorRun,
    )

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/nonexistent", architecture="LlamaForCausalLM", params_count=1_000)
    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="HuggingFace", hf_repo_id="org/ds",
                      status=DatasetStatus.READY, raw_path="/nonexistent", extra_metadata={},
                      revision=revision)
    session.add_all([model, dataset])
    await session.flush()
    view = ProbeMonitorDataset(name="eval", dataset_id=dataset.id, input_column="i", label_column="l",
                               label_mapping={"1": "positive", "0": "negative"}, role="eval", counts={})
    session.add(view)
    await session.flush()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[view.id],
                          config={}, status="completed", environment={})
    session.add(run)
    await session.flush()
    probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, variant="dense",
                         val_metrics={}, selected=True)
    other = ProbeMonitor(run_id=run.id, layer=1, rule="max", rule_params={}, variant="dense",
                         val_metrics={}, selected=False)
    session.add_all([probe, other])
    await session.flush()

    rng = np.random.default_rng(3)
    labels = np.array([i % 2 for i in range(N)], dtype=np.int8)
    scores = (rng.normal(size=N) + labels).astype(np.float32)
    positions = np.array([i * 3 + 1 for i in range(N)], dtype=np.int64)  # filtered rows → gaps
    n_scored = rng.integers(1, 50, size=N).astype(np.int32)
    evaluation = ProbeMonitorEvaluation(
        id=f"pme_{uuid.uuid4().hex[:12]}", probe_id=probe.id, dataset_id=view.id, status=status,
        metrics={"auroc": auroc_value} if auroc_value is not None else {"scored": False, "reason": "x"},
    )
    directory = artifact_dir_for(run.id)
    directory.mkdir(parents=True, exist_ok=True)
    paths = evaluation_array_paths(directory, probe.id, view.id, evaluation.id)
    for column, values in (("scores_path", scores), ("positions_path", positions),
                           ("n_scored_path", n_scored), ("labels_path", labels)):
        np.save(paths[column], values)
        setattr(evaluation, column, str(paths[column]))
    session.add(evaluation)
    await session.commit()
    return {"probe": probe.id, "other": other.id, "evaluation": evaluation, "view": view.id,
            "dataset": str(dataset.id), "scores": scores, "positions": positions,
            "n_scored": n_scored, "labels": labels, "paths": paths}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def _url(probe, evaluation):
    return f"/api/v1/probe-monitors/probes/{probe}/evaluations/{evaluation}/scores"


async def _all_pages(client, probe, evaluation, limit):
    rows, offset, pages = [], 0, 0
    while offset is not None:
        r = await client.get(_url(probe, evaluation), params={"offset": offset, "limit": limit})
        assert r.status_code == 200, r.text
        body = r.json()
        rows += body["rows"]
        offset = body["next_offset"]
        pages += 1
    return rows, body, pages


async def test_pages_concatenate_to_EXACTLY_the_stored_arrays(client, async_session):
    seeded = await _seed(async_session)
    ev = seeded["evaluation"].id
    rows, last, pages = await _all_pages(client, seeded["probe"], ev, limit=7)
    assert pages == 4  # 7+7+7+4
    assert [r["position"] for r in rows] == seeded["positions"].tolist()
    assert [r["n_scored"] for r in rows] == seeded["n_scored"].tolist()
    assert [r["label"] for r in rows] == seeded["labels"].tolist()
    assert np.allclose([r["score"] for r in rows], seeded["scores"])
    assert last["total"] == N
    # The consumer's use: the AUROC recomputed from served rows.
    assert abs(auroc([r["score"] for r in rows], [r["label"] for r in rows])
               - auroc(seeded["scores"].tolist(), seeded["labels"].tolist())) < 5e-4


async def test_the_envelope_names_its_sources(client, async_session):
    seeded = await _seed(async_session)
    r = await client.get(_url(seeded["probe"], seeded["evaluation"].id))
    body = r.json()
    assert body["evaluation_id"] == seeded["evaluation"].id
    assert body["probe_id"] == seeded["probe"]
    assert body["dataset_id"] == seeded["view"]
    assert body["source_dataset_id"] == seeded["dataset"]
    assert body["revision"] == COMMIT
    assert body["status"] == "completed" and body["reported_auroc"] == 0.75
    assert body["offset"] == 0 and body["next_offset"] is None and len(body["rows"]) == N
    assert set(body["rows"][0]) == {"position", "score", "label", "n_scored"}


async def test_the_defaults_and_limits(client, async_session):
    seeded = await _seed(async_session)
    url = _url(seeded["probe"], seeded["evaluation"].id)
    assert (await client.get(url, params={"limit": 50_001})).status_code == 422
    assert (await client.get(url, params={"limit": 0})).status_code == 422
    assert (await client.get(url, params={"offset": -1})).status_code == 422
    from src.main import app

    params = {p["name"]: p for p in app.openapi()["paths"][
        "/api/v1/probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/scores"]["get"]["parameters"]}
    assert params["limit"]["schema"]["default"] == 10_000
    assert params["limit"]["schema"]["maximum"] == 50_000


async def test_an_offset_past_the_end_is_an_empty_last_page(client, async_session):
    seeded = await _seed(async_session)
    r = await client.get(_url(seeded["probe"], seeded["evaluation"].id), params={"offset": N + 5})
    assert r.status_code == 200
    assert r.json()["rows"] == [] and r.json()["next_offset"] is None and r.json()["total"] == N


async def test_a_legacy_revision_reads_null(client, async_session):
    seeded = await _seed(async_session, revision=None)
    r = await client.get(_url(seeded["probe"], seeded["evaluation"].id))
    assert r.json()["revision"] is None


async def test_a_refused_evaluation_is_served_with_a_null_auroc(client, async_session):
    seeded = await _seed(async_session, status="refused", auroc_value=None)
    r = await client.get(_url(seeded["probe"], seeded["evaluation"].id))
    assert r.status_code == 200
    assert r.json()["status"] == "refused" and r.json()["reported_auroc"] is None


async def test_another_probes_evaluation_is_404(client, async_session):
    seeded = await _seed(async_session)
    r = await client.get(_url(seeded["other"], seeded["evaluation"].id))
    assert r.status_code == 404
    assert (await client.get(_url(seeded["probe"], "pme_nope"))).status_code == 404


async def test_a_LEGACY_evaluation_is_refused_naming_the_reevaluation(client, async_session):
    seeded = await _seed(async_session)
    seeded["evaluation"].positions_path = None
    await async_session.commit()
    r = await client.get(_url(seeded["probe"], seeded["evaluation"].id))
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "positions_not_recorded"
    assert detail["evaluate"] == f"POST /api/v1/probe-monitors/probes/{seeded['probe']}/evaluate"
    assert detail["message"]
    assert "rows" not in r.json()


@pytest.mark.parametrize("breakage, check", [
    ("missing_file", "file_missing"),
    ("missing_column", "path_not_recorded"),
    ("mismatch", "length_mismatch"),
    ("outside_root", "path_outside_artifact_root"),
    ("corrupt", "file_unreadable"),
])
async def test_a_broken_artifact_is_a_500_naming_the_check(client, async_session, tmp_path, breakage, check):
    seeded = await _seed(async_session)
    ev = seeded["evaluation"]
    if breakage == "missing_file":
        seeded["paths"]["n_scored_path"].unlink()
    elif breakage == "missing_column":
        ev.labels_path = None
    elif breakage == "mismatch":
        np.save(seeded["paths"]["labels_path"], seeded["labels"][:-1])
    elif breakage == "outside_root":
        outside = tmp_path.parent / f"elsewhere_{uuid.uuid4().hex}.npy"
        np.save(outside, seeded["scores"])
        ev.scores_path = str(outside)
    elif breakage == "corrupt":
        seeded["paths"]["positions_path"].write_bytes(b"not an array")
    await async_session.commit()
    r = await client.get(_url(seeded["probe"], ev.id))
    assert r.status_code == 500, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "evaluation_artifacts_broken"
    assert detail["check"] == check and detail["evaluation_id"] == ev.id
    # No internals in a 5xx body: the paths and the numpy error go to the log only.
    assert str(settings.data_dir) not in r.text and ".npy" not in r.text
