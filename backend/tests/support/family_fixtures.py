"""Probe-family fixtures (Feature 041; FTID §8).

Keys, sub-question keys and names are RANDOM (`random_behaviour`); nothing names a behaviour. A member is
built field by field so a flag fixture differs from its siblings ONLY in the field under test (the "agree
by construction" trap: every default here is shared, and a test overrides exactly one).
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence

from tests.support.random_behaviour import behaviour_body, slug

API = "/api/v1/probe-monitors"

#: The render form every member shares unless a test overrides it (`served_bos_unrecorded`).
SHARED_RENDER = {"generation_prompt": True, "add_special_tokens": False}
SHARED_PRECISION = "bfloat16"


async def catalogue(client, rng, *, sub_questions: Sequence[str] = (), models: Optional[List[str]] = None,
                    key: Optional[str] = None) -> str:
    """Catalogue a random behaviour declaring `sub_questions` (keys) and `models`; returns the key."""
    body = behaviour_body(rng, **({"key": key} if key else {}))
    if sub_questions:
        body["sub_questions"] = [{"key": s, "question": f"q {s}", "positive_meaning": f"p {s}"}
                                 for s in sub_questions]
    if models:
        body["models"] = [{"model_id": m} for m in models]
    made = await client.post(f"{API}/behaviours", json=body)
    assert made.status_code == 201, made.text
    return body["key"]


def sub_keys(rng, n: int) -> List[str]:
    return [slug(rng, words=2) for _ in range(n)]


async def model(session, model_id: Optional[str] = None):
    from tests.support.detector_fixtures import model_row

    return await model_row(session, model_id)


async def source(session):
    from src.models.dataset import Dataset, DatasetStatus

    row = Dataset(id=uuid.uuid4(), name=f"corpus-{uuid.uuid4().hex[:6]}", source="local",
                  status=DatasetStatus.READY, raw_path="/nonexistent", extra_metadata={})
    session.add(row)
    await session.flush()
    return row


async def view(session, dataset, *, key: Optional[str], role: str = "train", sub_question: Optional[str] = None,
               behaviour_version: Optional[int] = None, dataset_version: Optional[str] = "dv-shared",
               distribution: Optional[str] = None, name: Optional[str] = None):
    from src.models.probe_monitor import ProbeMonitorDataset

    row = ProbeMonitorDataset(
        name=name or f"view-{uuid.uuid4().hex[:6]}", dataset_id=dataset.id, input_column="text",
        label_column="label", label_mapping={"1": "positive", "0": "negative"}, role=role,
        distribution=distribution, counts={}, behaviour=key, sub_question=sub_question,
        behaviour_version=behaviour_version,
        dataset_version={"version_id": dataset_version} if dataset_version else None,
    )
    session.add(row)
    await session.flush()
    return row


async def member(session, model_row, dataset, *, key: str, sub_question: Optional[str] = None,
                 role: Optional[str] = None, precision: Optional[str] = SHARED_PRECISION,
                 render: Any = "shared", scope: str = "all", dataset_version: Optional[str] = "dv-shared",
                 built: bool = True, rung: int = 2, train_view=None, layer: int = 1, rule: str = "mean"):
    """One family member: its own training view (unless given), a completed run and one probe.
    `role` None leaves no role row (an implicit candidate)."""
    from src.models.probe_behaviour import ProbeBehaviourProbeRole
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorRun

    if train_view is None:
        train_view = await view(session, dataset, key=key, sub_question=sub_question, dataset_version=dataset_version)
    environment: Dict[str, Any] = {}
    if precision is not None:
        environment["model_dtype"] = precision
    run = ProbeMonitorRun(model_id=model_row.id, train_dataset_id=train_view.id, eval_dataset_ids=[],
                          config={"scope": scope}, status="completed", environment=environment, behaviour=key,
                          sub_question=sub_question)
    session.add(run)
    await session.flush()
    probe = ProbeMonitor(run_id=run.id, layer=layer, rule=rule, rule_params={}, variant="dense", val_metrics={},
                         behaviour=key, rung=rung, rung_reasons=[], threshold=0.5, sub_question=sub_question,
                         render_form=SHARED_RENDER if render == "shared" else render)
    if built:
        probe.definition_path = f"/nonexistent/{uuid.uuid4().hex}.probe.json"
        probe.definition_sha256 = "a" * 64
    session.add(probe)
    await session.flush()
    if role is not None:
        session.add(ProbeBehaviourProbeRole(probe_id=probe.id, key=key, role=role, updated_by="operator"))
        await session.flush()
    return probe
