"""A run ranks its probes by recall at the calibrated decision (Feature 040 S2 FR-21; 040-T28; FTASKS 10.x).

Two probes ranked in OPPOSITE orders by AUROC and by recall at their decisions: probe A separates the
classes well but its bar sits above most positives (high AUROC, low recall); probe B separates them less
cleanly but its bar is low enough to catch them (lower AUROC, high recall). A stamped run picks B; an
`auroc`-stamped run and an unstamped (pre-S2) run pick A.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.services import probe_monitor_run as run_mod


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


@pytest.fixture
def sync_db(async_engine):
    url = async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", ""))
    engine = create_engine(url)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def _world(db, rule):
    import uuid

    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorEvaluation, ProbeMonitorRun

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="t", status=ModelStatus.READY, file_path="/x",
                  architecture="LlamaForCausalLM", params_count=1)
    ds = Dataset(id=uuid.uuid4(), name="d", source="local", status=DatasetStatus.READY, raw_path="/x", extra_metadata={})
    db.add_all([model, ds])
    db.flush()
    train = ProbeMonitorDataset(name="t", dataset_id=ds.id, input_column="i", label_column="l", label_mapping={},
                                role="train", counts={})
    ood = ProbeMonitorDataset(name="o", dataset_id=ds.id, input_column="i", label_column="l", label_mapping={},
                              role="eval", distribution="out_of_distribution", counts={})
    db.add_all([train, ood])
    db.flush()
    config = {"scope": "all"} if rule is None else {"scope": "all", "selection_rule": rule}
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=train.id, eval_dataset_ids=[ood.id], config=config,
                          status="completed", environment={})
    db.add(run)
    db.flush()
    labels = np.array([1] * 50 + [0] * 50, dtype=np.int8)
    # A: positives 2.0..2.49, negatives 0.0..0.49 (AUROC 1.0) — bar 2.4 catches 10 of 50.
    # B: positives 1.0..1.49, negatives 0.8..1.29 (AUROC < 1) — bar 0.9 catches all 50.
    scores = {"A": np.r_[np.linspace(2.0, 2.49, 50), np.linspace(0.0, 0.49, 50)],
              "B": np.r_[np.linspace(1.0, 1.49, 50), np.linspace(0.8, 1.29, 50)]}
    bars = {"A": 2.4, "B": 0.9}
    aurocs = {"A": 1.0, "B": 0.84}
    probes = {}
    for name in ("A", "B"):
        probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, variant="dense",
                             val_metrics={"val_auroc": aurocs[name]}, threshold=bars[name])
        db.add(probe)
        db.flush()
        directory = run_mod.artifact_dir_for(run.id)
        directory.mkdir(parents=True, exist_ok=True)
        evaluation = ProbeMonitorEvaluation(id=f"pme_{uuid.uuid4().hex[:12]}", probe_id=probe.id, dataset_id=ood.id,
                                            status="completed", metrics={"auroc": aurocs[name]})
        paths = run_mod.evaluation_array_paths(directory, probe.id, ood.id, evaluation.id)
        for column, values in (("scores_path", scores[name].astype(np.float32)),
                               ("positions_path", np.arange(100, dtype=np.int64)),
                               ("n_scored_path", np.full(100, 10, dtype=np.int32)),
                               ("labels_path", labels)):
            np.save(paths[column], values)
            setattr(evaluation, column, str(paths[column]))
        db.add(evaluation)
        probes[name] = probe
    db.commit()
    return run, probes


@pytest.mark.parametrize("rule,winner", [("recall_at_decision", "B"), ("auroc", "A"), (None, "A")])
def test_the_stamp_chooses_the_ranking(sync_db, rule, winner):
    run, probes = _world(sync_db, rule)
    best = run_mod.select_best_probe(sync_db, [probes["A"], probes["B"]])
    assert best.id == probes[winner].id


def test_both_rankings_are_reported(sync_db):
    run, probes = _world(sync_db, "recall_at_decision")
    rankings = run_mod.selection_rankings(sync_db, [probes["A"], probes["B"]], run)
    assert [r["probe_id"] for r in rankings["recall_at_decision"]] == [probes["B"].id, probes["A"].id]
    assert [r["probe_id"] for r in rankings["auroc"]] == [probes["A"].id, probes["B"].id]
    assert rankings["basis"] == "out_of_distribution"
    assert rankings["recall_at_decision"][0]["key"][0] == pytest.approx(1.0)
    assert rankings["recall_at_decision"][1]["key"][0] == pytest.approx(0.2)


def test_the_key_orders_mean_then_worst_then_auroc():
    k = run_mod.recall_selection_key
    assert k([0.8, 0.8], [0.5]) > k([0.9, 0.6], [0.99])            # mean decides first
    assert k([0.7, 0.7], [0.5]) > k([0.9, 0.5], [0.99])            # equal mean: worst set second
    assert k([0.7, 0.7], [0.9]) > k([0.7, 0.7], [0.8])             # then AUROC
    assert k([], []) == (0.0, 0.0, 0.0)


def test_the_branch_calls_the_recall_ranking():
    tree = ast.parse(inspect.getsource(run_mod.select_best_probe))
    calls = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "selection_rankings" in calls and "probe_selection_key" in calls


@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
def test_with_no_evaluation_evidence_the_validation_auroc_still_chooses(sync_db, order):
    """Review R1 (040 S1-S2): a stamped run whose probes have no evaluation at all (a run submitted with
    no eval sets — `eval_dataset_ids` defaults to empty) ties every recall key at (0, 0, 0). The ranking
    then kept list order, so the run selected whichever probe was trained first; before S2 the validation
    AUROC broke the tie. Both list orders, so a list-order pick fails one of them."""
    from src.models.probe_monitor import ProbeMonitorEvaluation

    run, probes = _world(sync_db, "recall_at_decision")
    sync_db.query(ProbeMonitorEvaluation).filter(
        ProbeMonitorEvaluation.probe_id.in_([p.id for p in probes.values()])).delete(synchronize_session=False)
    sync_db.commit()
    best = run_mod.select_best_probe(sync_db, [probes[name] for name in order])
    assert best.id == probes["A"].id                       # val_auroc 1.0 against 0.84
    rankings = run_mod.selection_rankings(sync_db, [probes[name] for name in order], run)
    assert [r["probe_id"] for r in rankings["recall_at_decision"]] == [probes["A"].id, probes["B"].id]
    assert [r["val_auroc"] for r in rankings["recall_at_decision"]] == [1.0, 0.84]
