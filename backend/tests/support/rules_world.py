"""A real-file world for rules ∪ probe evaluations (Feature 040 S3; FTDD §10.1, FTID §8.2).

Rows of the evaluated view carry text of three kinds, chosen so every figure is HAND-COMPUTABLE:

* ``U`` — an unmasked value: the rule set MATCHES;
* ``M`` — a masked value: the rule set VETOES;
* ``B`` — both, masked and unmasked: a veto fires AND a match fires, so the row matches (no veto);
* ``P`` — plain words: nothing fires.

The evaluation's positions are the ODD split rows (non-contiguous); every EVEN split row is a ``U`` decoy, so
reading verdicts by array index instead of by position gives other figures (the FTID fixture rule).

The evaluated rows (positives then negatives), with the probe's score (bar 0.5, global, scope ``all``):

| row | label | kind | score | rules | probe | union |
|---|---|---|---|---|---|---|
| p1 | 1 | U | 0.9 | T | T | T |
| p2 | 1 | U | 0.1 | T | F | T |
| p3 | 1 | M | 0.9 | F | T | **F** (vetoed above the bar) |
| p4 | 1 | P | 0.9 | F | T | T |
| p5 | 1 | P | 0.1 | F | F | F |
| p6 | 1 | B | 0.1 | T | F | T (a match beats the veto) |
| n1 | 0 | P | 0.1 | F | F | F |
| n2 | 0 | M | 0.9 | F | T | F |
| n3 | 0 | U | 0.1 | T | F | T |
| n4 | 0 | P | 0.9 | F | T | T |
| n5 | 0 | P | 0.1 | F | F | F |
| n6 | 0 | P | 0.1 | F | F | F |

Recall: rules 3/6, probe 3/6, union 4/6. FPR: rules 1/6, probe 2/6, union 2/6.

Calibration (FR-36's acceptance fixture): 1,000 negatives on the ODD rows of a 2,000-row split (EVEN rows are
``U`` decoys), target 1% (budget 10). Four rows MATCH (0.4%, low scores), three are MASKED with the three
highest scores (vetoed: removed first), the rest plain with distinct scores. The union bar lets the probe spend
the remaining six (0.6%) and the union realises exactly 1%. Calibrating on all negatives instead would let the
probe spend ten, and the union 1.4%.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tests.support import rule_set_fixtures as rf
from tests.support.detector_fixtures import (
    Arrays,
    dataset_row,
    declare,
    evaluation_row,
    model_row,
    run_and_probe,
    view_row,
    write_split,
)
from tests.support.random_behaviour import behaviour_body, rng_for, slug

API = "/api/v1/probe-monitors"

TEXT = {"U": f"my number is {rf.UNMASKED} thanks", "M": f"my number is {rf.MASKED} thanks",
        "B": f"old {rf.MASKED} new {rf.UNMASKED}", "P": "nothing to see in these plain words"}
EVAL_ROWS = [("p1", 1, "U", 0.9), ("p2", 1, "U", 0.1), ("p3", 1, "M", 0.9), ("p4", 1, "P", 0.9),
             ("p5", 1, "P", 0.1), ("p6", 1, "B", 0.1), ("n1", 0, "P", 0.1), ("n2", 0, "M", 0.9),
             ("n3", 0, "U", 0.1), ("n4", 0, "P", 0.9), ("n5", 0, "P", 0.1), ("n6", 0, "P", 0.1)]
CAL_N = 1000
CAL_MATCH = 4
CAL_MASKED = 3


@dataclass
class RulesWorld:
    key: str
    rule_set: Dict[str, Any]
    group_column: str
    eval_view: Any = None
    cal_view: Any = None
    probe: Any = None
    run: Any = None
    evaluation: Any = None
    data_dir: Path = None
    extra: Dict[str, Any] = field(default_factory=dict)


def eval_split(group_column: str, groups: List[str], text: Dict[str, str] = None) -> Dict[str, List[Any]]:
    text = text or TEXT
    texts, labels, group = [], [], []
    for i, (_name, label, kind, _score) in enumerate(EVAL_ROWS):
        texts += [text["U"], text[kind]]          # even: decoy; odd: the evaluated row
        labels += [str(1 - label), str(label)]
        group += [groups[0], groups[i % 2]]
    return {"text": texts, "label": labels, group_column: group}


def eval_arrays(scores: Dict[str, float] = None) -> Arrays:
    n = len(EVAL_ROWS)
    scores = scores or {}
    return Arrays(scores=np.array([scores.get(r[0], r[3]) for r in EVAL_ROWS], dtype=np.float64),
                  positions=np.arange(1, 2 * n, 2), n_scored=np.full(n, 12),
                  labels=np.array([r[1] for r in EVAL_ROWS], dtype=np.int8))


def calibration_rows() -> List[str]:
    """Kinds of the 1,000 calibration negatives, in score order (the scores are `calibration_scores`)."""
    kinds = ["M"] * CAL_MASKED + ["P"] * (CAL_N - CAL_MASKED - CAL_MATCH) + ["U"] * CAL_MATCH
    return kinds


def calibration_scores() -> np.ndarray:
    # Strictly descending: the three masked rows highest, the four matching rows lowest.
    return np.linspace(0.99, 0.01, CAL_N)


async def build(client, session, data_dir: Path, seed: str, *, rule_set: Any = "masking",
                with_positions: bool = True, behaviour_extra: Dict[str, Any] = None,
                distribution: str = "out_of_distribution", threshold: float = 0.5,
                scores: Dict[str, float] = None, key: str = None, text: Dict[str, str] = None) -> RulesWorld:
    rng = rng_for(seed)
    key = key or slug(rng)
    doc = rf.masking_set(seed, key) if rule_set == "masking" else rule_set
    world = RulesWorld(key=key, rule_set=doc, group_column=slug(rng, words=1), data_dir=data_dir)
    groups = [slug(rng, words=1), slug(rng, words=1)]
    eval_ds = await dataset_row(session, write_split(data_dir, eval_split(world.group_column, groups, text)))
    cal_kinds = calibration_rows()
    cal_texts: List[str] = []
    for kind in cal_kinds:
        cal_texts += [(text or TEXT)["U"], (text or TEXT)[kind]]
    cal_ds = await dataset_row(session, write_split(data_dir, {"text": cal_texts, "label": ["0"] * (2 * CAL_N)}))
    model = await model_row(session)
    train = await view_row(session, eval_ds, role="train", behaviour=key)
    # counts sum to the split's rows, so the drift check (`split_changed`) runs on every read
    world.eval_view = await view_row(session, eval_ds, role="eval", distribution=distribution, behaviour=key,
                                     counts={"positive": len(EVAL_ROWS), "negative": len(EVAL_ROWS)})
    world.cal_view = await view_row(session, cal_ds, role="calibration", counts={"negative": 2 * CAL_N})
    await declare(session, world.eval_view, [world.group_column])
    world.run, world.probe = await run_and_probe(
        session, model, train, eval_views=[world.eval_view], calibration_view=world.cal_view, behaviour=key,
        threshold=threshold, target_fpr=0.01, calibration_scores=calibration_scores().tolist(), n_negatives=CAL_N)
    if with_positions:
        directory = Path(world.probe.calibration_scores_path).parent
        path = directory / f"{world.probe.id}__calibration_positions.npy"
        np.save(path, np.arange(1, 2 * CAL_N, 2, dtype=np.int64))
        world.probe.calibration_positions_path = str(path)
    world.evaluation = await evaluation_row(session, world.probe, world.run, world.eval_view, eval_arrays(scores))
    await session.commit()
    body = behaviour_body(rng, key=key, target_fpr=0.01, **(behaviour_extra or {}))
    if doc is not None:
        body["rule_set"] = {"document": doc}
    made = await client.post(f"{API}/behaviours", json=body)
    assert made.status_code == 201, made.text
    return world


async def run_one(client, monkeypatch, world: RulesWorld, **body) -> Dict[str, Any]:
    """Request through the route (dispatch recorded, never queued) and run the REAL task in-process."""
    import asyncio

    from src.services import rules_evaluation
    from src.workers.rules_evaluation_tasks import run_rules_evaluation

    dispatched: List[str] = []
    monkeypatch.setattr(rules_evaluation, "_dispatch", lambda rid: dispatched.append(rid) or f"celery-{uuid.uuid4().hex[:6]}")
    payload = {"view_ids": [world.eval_view.id], **body}
    response = await client.post(f"{API}/probes/{world.probe.id}/rules-evaluations", json=payload)
    assert response.status_code in (200, 202), response.text
    out = response.json()
    if dispatched:
        assert dispatched == [out["evaluation_id"]]
        outcome = await asyncio.to_thread(run_rules_evaluation, out["evaluation_id"])
        out["outcome"] = outcome
    out["dispatched"] = dispatched
    return out
