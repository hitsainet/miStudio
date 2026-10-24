"""`GET /probe-monitors/runs?train_dataset_id=` (035 FR-9, FR-10; D3, D17).

MUTATION CONTROL: C13 remove the `.where(train_dataset_id == …)` clause → the oldest-run test fails.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.asyncio


async def _seed(session):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/x", architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="c", source="local", status=DatasetStatus.READY,
                      raw_path="/x", extra_metadata={})
    session.add_all([model, dataset])
    await session.flush()

    def view(name, role="train"):
        return ProbeMonitorDataset(name=name, dataset_id=dataset.id, input_column="i", label_column="l",
                                   label_mapping={}, role=role, counts={})

    target, other, evalset, calib = view("V"), view("W"), view("E", "eval"), view("C", "calibration")
    session.add_all([target, other, evalset, calib])
    await session.flush()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runs = []
    # The OLDEST run trains on V; 204 newer runs train on W — 2 of them use V for evaluation
    # and calibration only, which must NOT count (D3).
    runs.append(ProbeMonitorRun(model_id=model.id, train_dataset_id=target.id, eval_dataset_ids=[],
                                config={}, status="completed", environment={}, created_at=t0))
    for i in range(1, 205):
        runs.append(ProbeMonitorRun(
            model_id=model.id, train_dataset_id=other.id,
            eval_dataset_ids=[target.id] if i == 1 else [evalset.id],
            calibration_dataset_id=target.id if i == 2 else None,
            config={}, status="completed" if i % 2 else "failed", environment={},
            created_at=t0 + timedelta(minutes=i),
        ))
    # One newer V run that FAILED, for the status intersection.
    runs.append(ProbeMonitorRun(model_id=model.id, train_dataset_id=target.id, eval_dataset_ids=[],
                                config={}, status="failed", environment={},
                                created_at=t0 + timedelta(days=1)))
    session.add_all(runs)
    await session.commit()
    return target.id, other.id, runs[0].id, runs[-1].id


async def test_the_oldest_run_trained_on_the_set_is_found(client, async_session):
    target, _, oldest, newest_failed = await _seed(async_session)
    r = await client.get("/api/v1/probe-monitors/runs", params={"train_dataset_id": target, "limit": 200})
    assert r.status_code == 200, r.text
    ids = [run["id"] for run in r.json()]
    assert ids == [newest_failed, oldest], "newest first, training set only"
    assert all(run["train_dataset_id"] == target for run in r.json())


async def test_without_the_filter_the_oldest_is_beyond_the_200_window(client, async_session):
    """The defect XR-4 closes: miDataworks' scan of the newest 200 misses it."""
    _, _, oldest, _ = await _seed(async_session)
    r = await client.get("/api/v1/probe-monitors/runs", params={"limit": 200})
    assert oldest not in [run["id"] for run in r.json()]


async def test_the_filter_ANDs_with_status(client, async_session):
    target, _, oldest, newest_failed = await _seed(async_session)
    r = await client.get("/api/v1/probe-monitors/runs",
                         params={"train_dataset_id": target, "status": "completed"})
    assert [run["id"] for run in r.json()] == [oldest]
    r = await client.get("/api/v1/probe-monitors/runs",
                         params={"train_dataset_id": target, "status": "failed"})
    assert [run["id"] for run in r.json()] == [newest_failed]


async def test_the_limit_is_still_capped_at_200(client, async_session):
    r = await client.get("/api/v1/probe-monitors/runs", params={"train_dataset_id": "x", "limit": 201})
    assert r.status_code == 422
