"""G40 (operator, 2026-11-08): a run never selects a probe it cannot trust.

Run `pmr_55715c1c87e2` selected its `attention` probe by recall at the decision bar: validation AUROC 0.4998,
threshold −35.10, firing on 100% of positives, all 5,000 everyday rows and all 2,000 nonsense rows. Perfect
recall, because the bar decided nothing. Selection now excludes a probe whose validation AUROC is below 0.55,
or whose bar fires on more than 10x its target false-positive rate on the calibration negatives, records the
reason in the rankings, and selects none (saying why) when nothing qualifies.

Each world below makes the degenerate probe the WINNER of both rankings had it been eligible (perfect recall,
the higher OOD AUROC), so a selection that ignores the exclusion picks it and the test goes red.
"""
from __future__ import annotations

import ast
import inspect
import uuid

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


LABELS = np.array([1] * 50 + [0] * 50, dtype=np.int8)
# D: fires on every row (bar far below every score): recall 1.0. OOD AUROC recorded 0.99 so the AUROC
# ranking prefers it too. G: an honest probe, bar 1.2 catches 30 of 50 positives and no negative.
SCORES = {"D": np.r_[np.linspace(2.0, 2.49, 50), np.linspace(0.0, 0.49, 50)],
          "G": np.r_[np.linspace(1.0, 1.49, 50), np.linspace(0.0, 0.49, 50)]}
BARS = {"D": -35.10, "G": 1.2}
OOD_AUROC = {"D": 0.99, "G": 0.90}


def _world(db, tmp_path, rule, spec):
    """`spec[name]`: probe columns (val_auroc, target_fpr, calibration negatives/lengths, bands)."""
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
    probes = {}
    for name, cols in spec.items():
        probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, variant="dense",
                             val_metrics={"val_auroc": cols.get("val_auroc", 0.95)}, threshold=BARS[name],
                             target_fpr=cols.get("target_fpr"), realised_fpr=cols.get("realised_fpr"),
                             length_bands=cols.get("bands"))
        for column, values in (("calibration_scores_path", cols.get("negatives")),
                               ("calibration_lengths_path", cols.get("lengths"))):
            if values is not None:
                path = tmp_path / f"{name}_{column}_{uuid.uuid4().hex[:6]}.npy"
                np.save(path, np.asarray(values))
                setattr(probe, column, str(path))
        db.add(probe)
        db.flush()
        directory = run_mod.artifact_dir_for(run.id)
        directory.mkdir(parents=True, exist_ok=True)
        evaluation = ProbeMonitorEvaluation(id=f"pme_{uuid.uuid4().hex[:12]}", probe_id=probe.id, dataset_id=ood.id,
                                            status="completed", metrics={"auroc": OOD_AUROC[name]})
        paths = run_mod.evaluation_array_paths(directory, probe.id, ood.id, evaluation.id)
        for column, values in (("scores_path", SCORES[name].astype(np.float32)),
                               ("positions_path", np.arange(100, dtype=np.int64)),
                               ("n_scored_path", np.full(100, 10, dtype=np.int32)),
                               ("labels_path", LABELS)):
            np.save(paths[column], values)
            setattr(evaluation, column, str(paths[column]))
        db.add(evaluation)
        probes[name] = probe
    db.commit()
    return run, probes


RULES = ["recall_at_decision", "auroc", None]
HONEST = {"val_auroc": 0.93, "target_fpr": 0.01, "negatives": np.linspace(-5.0, 0.0, 1000),
          "lengths": np.full(1000, 12)}


@pytest.mark.parametrize("rule", RULES)
def test_the_degenerate_probe_would_win_without_the_exclusion(sync_db, tmp_path, rule, monkeypatch):
    """The control for every test below: with exclusions switched off, D wins under every rule."""
    run, probes = _world(sync_db, tmp_path, rule, {"D": {"val_auroc": 0.4998}, "G": HONEST})
    best = run_mod.select_best_probe(sync_db, [probes["D"], probes["G"]], exclusions={})
    assert best.id == probes["D"].id


@pytest.mark.parametrize("rule", RULES)
def test_a_chance_level_probe_is_never_selected(sync_db, tmp_path, rule):
    run, probes = _world(sync_db, tmp_path, rule, {"D": {"val_auroc": 0.4998}, "G": HONEST})
    best = run_mod.select_best_probe(sync_db, [probes["D"], probes["G"]])
    assert best.id == probes["G"].id
    rankings = run_mod.selection_rankings(sync_db, [probes["D"], probes["G"]], run)
    for ranking in ("recall_at_decision", "auroc"):
        last = rankings[ranking][-1]
        assert last["probe_id"] == probes["D"].id
        assert [r["code"] for r in last["excluded"]] == ["chance_level_validation_auroc"]
        assert last["excluded"][0]["val_auroc"] == pytest.approx(0.4998)
        assert rankings[ranking][0]["excluded"] == []


def test_just_below_the_floor_is_excluded(sync_db, tmp_path):
    # (A NaN validation AUROC cannot be stored: JSONB refuses the token, so it is not a case here.)
    run, probes = _world(sync_db, tmp_path, "recall_at_decision", {"D": {"val_auroc": 0.5499}, "G": HONEST})
    assert run_mod.select_best_probe(sync_db, [probes["D"], probes["G"]]).id == probes["G"].id


def test_exactly_the_floor_qualifies(sync_db, tmp_path):
    run, probes = _world(sync_db, tmp_path, "recall_at_decision", {"D": {"val_auroc": 0.55}, "G": HONEST})
    assert run_mod.select_best_probe(sync_db, [probes["D"], probes["G"]]).id == probes["D"].id


@pytest.mark.parametrize("rule", RULES)
def test_a_bar_firing_on_every_calibration_negative_is_never_selected(sync_db, tmp_path, rule):
    """The constant-floor shape: every negative scores the bar exactly, so `>=` fires on all of them."""
    floor = {"val_auroc": 0.96, "target_fpr": 0.01, "negatives": np.full(1000, BARS["D"]),
             "lengths": np.full(1000, 12)}
    run, probes = _world(sync_db, tmp_path, rule, {"D": floor, "G": HONEST})
    assert run_mod.select_best_probe(sync_db, [probes["D"], probes["G"]]).id == probes["G"].id
    reasons = run_mod.selection_exclusions([probes["D"]], run)[probes["D"].id]
    assert reasons == [{"code": "bar_fires_too_often", "calibration_fpr": 1.0, "basis": "recorded_decision",
                        "target_fpr": 0.01, "maximum": pytest.approx(0.1)}]


@pytest.mark.parametrize("fires,excluded", [(100, False), (101, True)])
def test_ten_times_the_target_is_the_limit(sync_db, tmp_path, fires, excluded):
    """1,000 negatives at a 1% target: 100 firing (10x) qualifies, 101 does not."""
    negatives = np.r_[np.full(fires, 0.0), np.full(1000 - fires, -10.0)]
    cols = {"val_auroc": 0.9, "target_fpr": 0.01, "negatives": negatives, "lengths": np.full(1000, 12)}
    run, probes = _world(sync_db, tmp_path, "recall_at_decision", {"D": cols})
    probes["D"].threshold = 0.0
    reasons = run_mod.selection_exclusions([probes["D"]], run)[probes["D"].id]
    assert bool(reasons) is excluded


def test_the_check_reads_the_recorded_decision_not_only_the_global_bar(sync_db, tmp_path):
    """A global bar above every negative, but a length band (what is actually served) below all of them."""
    cols = {"val_auroc": 0.9, "target_fpr": 0.01, "negatives": np.zeros(1000), "lengths": np.full(1000, 12),
            "bands": [{"min_tokens": 0, "max_tokens": None, "threshold": -1.0}]}
    run, probes = _world(sync_db, tmp_path, "recall_at_decision", {"D": cols})
    probes["D"].threshold = 5.0
    from src.services.probe_breakdown import recorded_decisions

    served = recorded_decisions(probes["D"], "all", np.full(1000, 12)).fires(np.zeros(1000))
    assert served.all(), "precondition: the band, not the global bar, decides these rows"
    reasons = run_mod.selection_exclusions([probes["D"]], run)[probes["D"].id]
    assert reasons and reasons[0]["basis"] == "recorded_decision" and reasons[0]["calibration_fpr"] == 1.0


def test_without_lengths_the_global_bar_is_measured(sync_db, tmp_path):
    cols = {"val_auroc": 0.9, "target_fpr": 0.01, "negatives": np.full(1000, BARS["D"])}
    run, probes = _world(sync_db, tmp_path, "recall_at_decision", {"D": cols})
    assert run_mod.calibration_firing_rate(probes["D"], run) == (1.0, "global_bar")


def test_without_stored_negatives_the_recorded_rate_is_used(sync_db, tmp_path):
    run, probes = _world(sync_db, tmp_path, "recall_at_decision",
                         {"D": {"val_auroc": 0.9, "target_fpr": 0.01, "realised_fpr": 0.5}})
    assert run_mod.calibration_firing_rate(probes["D"], run) == (0.5, "realised_fpr")
    assert [r["code"] for r in run_mod.selection_exclusions([probes["D"]], run)[probes["D"].id]] == [
        "bar_fires_too_often"]


def test_an_unmeasured_bar_excludes_nothing(sync_db, tmp_path):
    run, probes = _world(sync_db, tmp_path, "recall_at_decision", {"D": {"val_auroc": 0.9, "target_fpr": 0.01}})
    assert run_mod.calibration_firing_rate(probes["D"], run) is None
    assert run_mod.selection_exclusions([probes["D"]], run)[probes["D"].id] == []


@pytest.mark.parametrize("rule", RULES)
def test_when_nothing_qualifies_the_run_selects_none_and_says_why(sync_db, tmp_path, rule):
    floor = {"val_auroc": 0.96, "target_fpr": 0.01, "negatives": np.full(1000, BARS["D"]),
             "lengths": np.full(1000, 12)}
    run, probes = _world(sync_db, tmp_path, rule, {"D": floor, "G": {"val_auroc": 0.5}})
    environment = {}
    best = run_mod.record_selection(sync_db, run, [probes["D"], probes["G"]], environment)
    assert best is None
    assert not probes["D"].selected and not probes["G"].selected
    selection = environment["selection"]
    assert selection["selected"] is None
    assert "no probe qualified: all 2 were excluded" in selection["none_selected"]
    assert "bar_fires_too_often" in selection["none_selected"]
    assert "chance_level_validation_auroc" in selection["none_selected"]
    assert {r["probe_id"]: [x["code"] for x in r["excluded"]] for r in selection["recall_at_decision"]} == {
        probes["D"].id: ["bar_fires_too_often"], probes["G"].id: ["chance_level_validation_auroc"]}


def test_record_selection_marks_and_records_the_eligible_winner(sync_db, tmp_path):
    run, probes = _world(sync_db, tmp_path, "recall_at_decision", {"D": {"val_auroc": 0.4998}, "G": HONEST})
    environment = {}
    best = run_mod.record_selection(sync_db, run, [probes["D"], probes["G"]], environment)
    assert best.id == probes["G"].id and probes["G"].selected is True and not probes["D"].selected
    assert environment["selection"]["selected"] == probes["G"].id
    assert environment["selection"]["none_selected"] is None
    assert environment["selection"]["rule"] == "recall_at_decision"


def test_the_run_completes_through_record_selection():
    """The CALLER (memory: mutation survivors cluster on callers). `execute_probe_run`'s own body must call
    `record_selection` with the run row, the trained probes and the environment it then writes."""
    source = inspect.getsource(run_mod)
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "record_selection"]
    assert len(calls) == 1, "exactly one production call"
    call = calls[0]
    assert [getattr(a, "id", None) for a in call.args[:2]] == ["db", "row"]
    assert getattr(call.args[3], "id", None) == "environment"
    # …and the environment it fills is the one the run stores.
    owner = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                 and any(n is call for n in ast.walk(f)))
    stores = [n for n in ast.walk(owner) if isinstance(n, ast.Assign)
              and any(getattr(t, "attr", None) == "environment" for t in n.targets)]
    assert any(getattr(s.value, "id", None) == "environment" for s in stores)
