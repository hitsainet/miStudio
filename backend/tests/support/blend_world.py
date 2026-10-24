"""A catalogued behaviour with a family, its blend views and every member's stored evaluations — on a real
database and real files (Feature 041 S2 route and task tests).

Arrays are written through 035's writer names (`detector_fixtures.evaluation_row`) under the test's data
root; positions are non-contiguous and members keep different subsets; the calibration evaluation is
`refused` with `metrics.sample.rows` as a real sampled calibration evaluation is. Keys are random.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tests.support import family_fixtures as ff
from tests.support.detector_fixtures import Arrays, evaluation_row
from tests.support.random_behaviour import rng_for

API = "/api/v1/probe-monitors"


@dataclass
class BlendWorld:
    key: str
    model: Any
    members: List[Any]
    runs: Dict[str, Any]
    blend_fit: Any
    tests: List[Any]
    calibration: Any
    evaluations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sample_rows: int = 2000


def _arrays(rng, positions, labels, signal):
    scores = (signal * labels + rng.normal(0.0, 1.0, labels.shape[0])).astype(np.float32)
    return scores


async def build(client, session, *, seed: str, signals: Sequence[float] = (2.0, 1.0, 0.0), n: int = 240,
                keep: float = 0.99, sample_rows: int = 2000, skip: Sequence[tuple] = (), threshold: float = 0.8,
                target_fpr: float = 0.05, dataset_version: Optional[str] = "dv-blend", criteria=None,
                held_out_columns=None, calibration_status: str = "refused") -> BlendWorld:
    """`skip`: (member index, view role) pairs left without an evaluation."""
    import hashlib

    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big"))
    key = await ff.catalogue(client, rng_for(seed))
    if criteria is not None or held_out_columns is not None or target_fpr is not None:
        body: Dict[str, Any] = {"base_version": 1, "target_fpr": target_fpr}
        if criteria is not None:
            body["success_criteria"] = criteria
        if held_out_columns is not None:
            body["held_out_columns"] = held_out_columns
        made = await client.post(f"{API}/behaviours/{key}/versions", json=body)
        assert made.status_code == 201, made.text
    model = await ff.model(session)
    ds = await ff.source(session)
    fit = await ff.view(session, ds, key=key, role="blend_fit", dataset_version=dataset_version)
    ood = await ff.view(session, ds, key=key, role="eval", distribution="out_of_distribution",
                        dataset_version=dataset_version)
    ind = await ff.view(session, ds, key=key, role="eval", distribution="in_distribution",
                        dataset_version=dataset_version)
    cal = await ff.view(session, ds, key=key, role="calibration", dataset_version=dataset_version)
    members, runs = [], {}
    for j, _signal in enumerate(signals):
        probe = await ff.member(session, model, ds, key=key, role="primary" if j == 0 else None)
        probe.threshold = threshold
        members.append(probe)
    await session.flush()
    from src.models.probe_monitor import ProbeMonitorRun

    for probe in members:
        runs[probe.id] = await session.get(ProbeMonitorRun, probe.run_id)
    world = BlendWorld(key=key, model=model, members=members, runs=runs, blend_fit=fit, tests=[ood, ind],
                       calibration=cal, sample_rows=sample_rows)
    for role, view in (("blend_fit", fit), ("ood", ood), ("ind", ind), ("calibration", cal)):
        size = n * 2 if role == "calibration" else n
        positions = np.sort(rng.choice(size * 3, size=size, replace=False)).astype(np.int64)
        labels = np.zeros(size, dtype=np.int8) if role == "calibration" else (rng.random(size) < 0.5).astype(np.int8)
        for j, probe in enumerate(members):
            if (j, role) in skip:
                continue
            idx = np.sort(rng.choice(size, size=int(round(size * keep)), replace=False))
            scores = _arrays(rng, positions, labels.astype(np.float64), signals[j])
            arrays = Arrays(scores=scores[idx], positions=positions[idx],
                            n_scored=rng.integers(5, 200, size=idx.size).astype(np.int32), labels=labels[idx])
            status = calibration_status if role == "calibration" else "completed"
            evaluation = await evaluation_row(session, probe, runs[probe.id], view, arrays, status=status)
            metrics = {"auroc": 0.8, "scored": role != "calibration"}
            if role == "calibration":
                metrics["sample"] = {"rows": sample_rows, "seed_basis": "view+rows", "source_rows": size * 10}
            evaluation.metrics = metrics
            world.evaluations.setdefault(probe.id, {})[view.id] = evaluation
    await session.commit()
    return world


def start_body(world: BlendWorld, **extra) -> Dict[str, Any]:
    body = {"behaviour_key": world.key, "model": world.model.id, "dataset_version": "dv-blend",
            "calibration_sample_rows": world.sample_rows}
    body.update(extra)
    return {k: v for k, v in body.items() if v is not None}


async def done_report(client, monkeypatch, world: BlendWorld, **extra) -> str:
    """Start a report through the route (dispatch recorded, not queued) and run the REAL task in-process."""
    import asyncio

    from src.workers import probe_blend_tasks

    class _Task:
        id = "task-done"

    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay", lambda rid: _Task())
    response = await client.post(f"{API}/blend-reports", json=start_body(world, **extra))
    assert response.status_code == 202, response.text
    report_id = response.json()["report_id"]
    outcome = await asyncio.to_thread(probe_blend_tasks.run_blend_evaluation, report_id)
    assert outcome["status"] == "done", outcome
    return report_id
