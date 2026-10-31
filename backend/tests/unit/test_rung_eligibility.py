"""Which evaluations a probe's rung is computed from (Feature 041 T22, FR-1/OQ-1 (a); D36 answers OI-1).

⚠ Collecting blend inputs must never move a rung: rung 2 needs EVERY out-of-distribution set to clear 0.5,
and a rung change invalidates the built definition — so a `blend_fit` or calibration evaluation, or a
specialist scored on the main question, would silently un-export a family. One pure predicate
(`rung_eligibility_of`) decides; `recompute_rung` and the two serializers call it (asserted by spy).

Fixtures differ only in the view field under test (role, or sub-question). Keys are random.
"""
from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.services import probe_monitor_rung
from src.services.probe_monitor_rung import rung_eligibility_of
from tests.support.random_behaviour import rng_for, slug

API = "/api/v1/probe-monitors"


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


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


# ── the predicate ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("role, view_sub, probe_sub, counts, label", [
    ("eval", None, None, True, "own"),
    ("eval", "s1", "s1", True, "own"),
    ("eval", None, "s1", False, "main"),           # OQ-1 (a): a specialist on the main question
    ("eval", "s2", "s1", False, "other"),
    ("eval", "s1", None, False, "other"),           # D36 / OI-1: the symmetric rule
    ("blend_fit", None, None, False, "own"),
    ("calibration", None, None, False, "own"),
    ("train", None, None, False, "own"),
    ("everyday_text", None, None, True, None),      # ENH-006 D82: counts towards the everyday-text check
    ("nonsense_text", None, None, True, None),      # ENH-006 D84: counts towards the nonsense-text check
    (None, None, None, False, None),                # a purged view
])
def test_the_predicate(role, view_sub, probe_sub, counts, label):
    verdict = rung_eligibility_of(role, view_sub, probe_sub)
    assert (verdict.counts, verdict.label_question) == (counts, label)
    assert (verdict.note is None) == counts


# ── recompute_rung ───────────────────────────────────────────────────────────────────────────


def _world(db, *, probe_sub=None, built=False):
    """A probe at rung 2 on one passing out-of-distribution eval view of its own question."""
    import uuid

    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun

    key = slug(rng_for(uuid.uuid4().hex))
    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY, file_path="/x",
                  architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="c", source="local", status=DatasetStatus.READY, raw_path="/x",
                      extra_metadata={})
    db.add_all([model, dataset])
    db.flush()

    def view(role, distribution=None, sub=None):
        row = ProbeMonitorDataset(name=f"v-{uuid.uuid4().hex[:6]}", dataset_id=dataset.id, input_column="i",
                                  label_column="l", label_mapping={}, role=role, distribution=distribution,
                                  counts={}, behaviour=key, sub_question=sub)
        db.add(row)
        db.flush()
        return row

    train = view("train", sub=probe_sub)
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=train.id, eval_dataset_ids=[], config={},
                          status="completed", environment={}, behaviour=key, sub_question=probe_sub)
    db.add(run)
    db.flush()
    probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, variant="dense", val_metrics={},
                         behaviour=key, rung=2, rung_reasons=[], sub_question=probe_sub,
                         threshold=1.0, target_fpr=0.01)
    if built:
        probe.definition_path = "/nonexistent/x.probe.json"
        probe.definition_sha256 = "c" * 64
        probe.definition_build = {"resolved": {"detector_digest": "d" * 64}}
    db.add(probe)
    db.flush()
    own = view("eval", "out_of_distribution", sub=probe_sub)
    _evaluation(db, probe, own, ci_low=0.9)
    # ENH-006 D82: rung 2 also needs a passed everyday-text check — 0 false alarms in 1,000 at the shipped bar.
    from src.models.probe_monitor import ProbeMonitorEvaluation

    everyday = view("everyday_text")
    db.add(ProbeMonitorEvaluation(probe_id=probe.id, dataset_id=everyday.id, status="refused",
                                  metrics={"scored": False, "everyday_text": {
                                      "threshold": 1.0, "rows": 1000, "false_alarms": 0}}))
    # ENH-006 D84: and a passed nonsense-text check, and a catch rate at the shipped bar on the OOD set.
    nonsense = view("nonsense_text")
    db.add(ProbeMonitorEvaluation(probe_id=probe.id, dataset_id=nonsense.id, status="refused",
                                  metrics={"scored": False, "nonsense_text": {
                                      "threshold": 1.0, "rows": 1000, "false_alarms": 0}}))
    db.commit()
    return probe, view


def _evaluation(db, probe, view, *, ci_low):
    from src.models.probe_monitor import ProbeMonitorEvaluation

    row = ProbeMonitorEvaluation(probe_id=probe.id, dataset_id=view.id, status="completed",
                                 metrics={"scored": True, "auroc": ci_low, "ci": {"low": ci_low, "high": 1.0},
                                          # ENH-006 D84: a full catch rate recorded at the shipped bar (1.0).
                                          "catch_rate": {"threshold": 1.0, "positives": 100, "caught": 100}})
    db.add(row)
    db.flush()
    return row


def _recompute(db, probe):
    from src.services.probe_monitor_run import recompute_rung

    outcome = recompute_rung(db, probe.id)
    db.refresh(probe)
    return outcome


async def test_blend_fit_and_calibration_evaluations_do_not_move_rungs(sync_db):
    probe, view = _world(sync_db)
    assert _recompute(sync_db, probe)["rung"] == 2
    blend = view("blend_fit")
    calibration = view("calibration")
    _evaluation(sync_db, probe, blend, ci_low=0.1)          # would fail rung 1 and 2 if it counted
    _evaluation(sync_db, probe, calibration, ci_low=0.1)
    sync_db.commit()
    outcome = _recompute(sync_db, probe)
    assert outcome["rung"] == 2 and probe.rung == 2
    assert any(f"{blend.id} does not count towards the rung: role blend_fit" in r for r in outcome["reasons"])
    assert any(f"{calibration.id} does not count towards the rung: role calibration" in r for r in outcome["reasons"])


async def test_an_eval_view_with_the_same_failing_ci_DOES_move_the_rung(sync_db):
    """The negative half of the fixture above: the same failing CI on an `eval` OOD view demotes —
    so the test above holds because of the role, not because the failing rows are ignored anyway."""
    probe, view = _world(sync_db)
    _evaluation(sync_db, probe, view("eval", "out_of_distribution"), ci_low=0.1)
    sync_db.commit()
    assert _recompute(sync_db, probe)["rung"] < 2


async def test_specialist_on_main_question_does_not_count(sync_db):
    probe, view = _world(sync_db, probe_sub="sq-own")
    main = view("eval", "out_of_distribution", sub=None)
    _evaluation(sync_db, probe, main, ci_low=0.1)
    sync_db.commit()
    outcome = _recompute(sync_db, probe)
    assert outcome["rung"] == 2
    assert any(f"{main.id} does not count towards the rung: answers the main question" in r
               for r in outcome["reasons"])


async def test_specialist_on_its_own_sub_question_counts(sync_db):
    probe, view = _world(sync_db, probe_sub="sq-own")
    _evaluation(sync_db, probe, view("eval", "out_of_distribution", sub="sq-own"), ci_low=0.1)
    sync_db.commit()
    assert _recompute(sync_db, probe)["rung"] < 2


async def test_main_probe_on_sub_question_does_not_count(sync_db):
    """D36 (OI-1): symmetric — a main-question probe's evaluation on a sub-question view is labelled
    `other` and does not move its rung."""
    probe, view = _world(sync_db, probe_sub=None)
    other = view("eval", "out_of_distribution", sub="sq-some")
    _evaluation(sync_db, probe, other, ci_low=0.1)
    sync_db.commit()
    outcome = _recompute(sync_db, probe)
    assert outcome["rung"] == 2
    assert any(f"{other.id} does not count towards the rung: answers sub-question sq-some" in r
               for r in outcome["reasons"])


async def test_no_definition_invalidated_by_blend_inputs(sync_db):
    probe, view = _world(sync_db, built=True)
    assert _recompute(sync_db, probe)["rung"] == 2
    _evaluation(sync_db, probe, view("blend_fit"), ci_low=0.1)
    _evaluation(sync_db, probe, view("calibration"), ci_low=0.1)
    sync_db.commit()
    _recompute(sync_db, probe)
    assert probe.definition_path is not None and probe.definition_sha256 == "c" * 64
    assert "invalidated" not in (probe.definition_build or {})


async def test_recompute_rung_CALLS_the_predicate_once_per_evaluation(sync_db, monkeypatch):
    """The wiring: `recompute_rung` asks the shared predicate (payload and count), so an inlined copy that
    ignores the role or the sub-question cannot pass."""
    probe, view = _world(sync_db, probe_sub="sq-own")
    blend = view("blend_fit")
    _evaluation(sync_db, probe, blend, ci_low=0.9)
    sync_db.commit()
    calls = []
    real = probe_monitor_rung.rung_eligibility_of

    def spy(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(probe_monitor_rung, "rung_eligibility_of", spy)
    _recompute(sync_db, probe)
    assert sorted(calls, key=str) == sorted([("eval", "sq-own", "sq-own"), ("blend_fit", None, "sq-own")], key=str)


# ── the serializers (FR-1 label, T22 exposure) ───────────────────────────────────────────────


async def _scored_world(async_session, data_dir, *, probe_sub, view_sub, role="eval"):
    from tests.support.detector_fixtures import Arrays, dataset_row, evaluation_row, model_row, run_and_probe, view_row

    ds = await dataset_row(async_session, "/nonexistent")
    model = await model_row(async_session)
    train = await view_row(async_session, ds, role="train", sub_question=probe_sub)
    target = await view_row(async_session, ds, role=role, distribution="out_of_distribution", sub_question=view_sub)
    run, probe = await run_and_probe(async_session, model, train)
    probe.sub_question = probe_sub
    arrays = Arrays(scores=np.array([0.1, 0.9, 0.4]), positions=np.array([3, 7, 11]),
                    n_scored=np.array([5, 5, 5]), labels=np.array([0, 1, 0]))
    evaluation = await evaluation_row(async_session, probe, run, target, arrays)
    await async_session.commit()
    return probe, evaluation


@pytest.mark.parametrize("probe_sub, view_sub, role, label, counts", [
    (None, None, "eval", "own", True),
    ("sq-a", None, "eval", "main", False),
    ("sq-a", "sq-a", "eval", "own", True),
    (None, "sq-a", "eval", "other", False),
    (None, None, "blend_fit", "own", False),
])
async def test_scores_and_report_carry_the_label_and_count(client, async_session, data_dir, probe_sub, view_sub,
                                                            role, label, counts, monkeypatch):
    probe, evaluation = await _scored_world(async_session, data_dir, probe_sub=probe_sub, view_sub=view_sub, role=role)
    calls = []
    real = probe_monitor_rung.rung_eligibility_of
    monkeypatch.setattr(probe_monitor_rung, "rung_eligibility_of", lambda *a: calls.append(a) or real(*a))
    scores = (await client.get(f"{API}/probes/{probe.id}/evaluations/{evaluation.id}/scores")).json()
    assert (scores["label_question"], scores["counts_towards_rung"]) == (label, counts)
    report = (await client.get(f"{API}/probes/{probe.id}")).json()
    entry = next(e for e in report["evaluations"] if e["id"] == evaluation.id)
    assert (entry["label_question"], entry["counts_towards_rung"]) == (label, counts)
    # Both serializers ask the ONE predicate, with the view's role and sub-question and the probe's.
    assert calls == [(role, view_sub, probe_sub), (role, view_sub, probe_sub)]


async def test_every_other_scores_field_is_unchanged(client, async_session, data_dir):
    """On a fixture with no sub-question the scores response is 035's plus the two additive fields."""
    probe, evaluation = await _scored_world(async_session, data_dir, probe_sub=None, view_sub=None)
    body = (await client.get(f"{API}/probes/{probe.id}/evaluations/{evaluation.id}/scores")).json()
    assert set(body) == {"evaluation_id", "probe_id", "dataset_id", "source_dataset_id", "revision", "status",
                         "reported_auroc", "total", "offset", "next_offset", "rows", "label_question",
                         "counts_towards_rung"}
    assert body["rows"] == [{"position": 3, "score": pytest.approx(0.1), "label": 0, "n_scored": 5},
                            {"position": 7, "score": pytest.approx(0.9), "label": 1, "n_scored": 5},
                            {"position": 11, "score": pytest.approx(0.4), "label": 0, "n_scored": 5}]
    assert (body["status"], body["total"], body["reported_auroc"]) == ("completed", 3, pytest.approx(0.8))
