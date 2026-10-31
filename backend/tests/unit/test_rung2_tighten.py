"""ENH-006 D84 (2026-10-29): rung 2 also needs a CATCH RATE at the shipped bar and a NONSENSE-TEXT check.

Why ("I think the probe sucks." — operator): the everyday-text check (D82) on `pmr_bbe9a3608802` failed the
run's pick and three others, but PASSED the three inert `max` probes (0% false alarms: they barely fire on
anything; in-distribution AUROC ~0.63, recall at their bar <= 2.5%) and the layer-0 `mean` probe (0.35%),
which fired on 280/280 of miLLM's random-word benchmark chats. A false-alarm check alone promotes a probe that
never fires, and prose is not word salad.

The rule: rung 2 needs, besides every out-of-distribution AUROC interval above 0.5,
  (a) on EVERY counted out-of-distribution set, recall AT THE SHIPPED BAR (`score >= probe.threshold`) of at
      least `rung2_min_recall` (0.5), re-taken from the stored per-row scores at the bar in force; and
  (b) a completed evaluation on a `nonsense_text` view of at least `nonsense_text_min_rows` (1,000) rows with
      false alarms at the shipped bar at most `nonsense_text_fpr_factor` (2) x the target rate.

Negative controls (0xcc/reviews/rung2_tighten_controls_2026-10-29.md):
  K1 drop the catch-rate requirement in `compute`                 -> red
  K2 drop the nonsense-text requirement in `compute`              -> red
  K3 measure recall at the set's own 1% operating point           -> red (TestRecompute::test_..._own_bar)
  K4 `>` instead of `>=` in the needed-count comparison            -> red (the 50% boundary)
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core import runtime_settings
from src.core.config import settings
from src.schemas.evidence_ladder import ProbeRung
from src.services.probe_monitor_rung import (
    NONSENSE_NOT_CHECKED,
    EverydaySet,
    RecallSet,
    SetResult,
    catch_rate_check,
    compute,
    everyday_check,
    nonsense_check,
)

API = "/api/v1/probe-monitors"
NOT_CHECKED = "not yet checked on nonsense text — evaluate on a nonsense-text set"
OLD_RUNG_2 = [SetResult("held_out", False, 0.9), SetResult("ood", True, 0.9)]
EVERYDAY_OK = everyday_check([EverydaySet("prose", 2000, 0, 1.0)], target_fpr=0.01)
NONSENSE_OK = nonsense_check([EverydaySet("salad", 2000, 0, 1.0)], target_fpr=0.01)


def _catch(positives, caught):
    return catch_rate_check([RecallSet("ood", positives, caught, 1.0)])


@pytest.fixture(autouse=True)
def _fresh_settings():
    runtime_settings.reset()
    yield
    runtime_settings.reset()


def test_the_phrases_are_the_operator_decision():
    assert NONSENSE_NOT_CHECKED == NOT_CHECKED
    reason = _catch(410, 41).reason
    assert reason.startswith("catches 10% of positives on unseen tasks at the shipped bar (needs ≥ 50%)")


# ── the pure rule ─────────────────────────────────────────────────────────────────────────────


class TestTheCatchRate:
    def test_an_inert_probe_with_no_false_alarms_is_rung_1_with_the_catch_rate_reason(self):
        """The three `max` probes: 0% everywhere, recall 2% at the bar."""
        result = compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=NONSENSE_OK, catch_rate=_catch(1000, 20))
        assert result.rung is ProbeRung.HELD_OUT
        reason = next(r for r in result.reasons if r.startswith("rung 2 not reached"))
        assert "catches 2% of positives on unseen tasks at the shipped bar (needs ≥ 50%)" in reason
        assert "20 of 1,000 positives" in reason
        assert result.next_step.startswith("catches 2% of positives on unseen tasks at the shipped bar (needs ≥ 50%)")
        assert result.catch_rate["status"] == "too_few_caught"
        assert result.catch_rate["sets"][0]["sentence"] == (
            "On unseen tasks (ood): catches 2% of positives at the shipped bar (20 of 1,000; needs ≥ 50%)")

    def test_fifty_percent_exactly_passes_and_49_9_fails(self):
        assert compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=NONSENSE_OK, catch_rate=_catch(1000, 500)).rung == 2
        assert compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=NONSENSE_OK, catch_rate=_catch(1000, 499)).rung == 1
        assert compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=NONSENSE_OK, catch_rate=_catch(3, 2)).rung == 2  # 66.7%

    def test_every_out_of_distribution_set_must_catch_enough(self):
        check = catch_rate_check([RecallSet("a", 100, 90), RecallSet("b", 100, 10)])
        assert compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=NONSENSE_OK, catch_rate=check).rung == 1
        assert "on b: 10 of 100 positives" in check.reason

    def test_an_unmeasured_set_is_not_a_pass(self):
        check = catch_rate_check([RecallSet("a", 100, 90), RecallSet("b", None, None, 1.0, "evaluate it again")])
        assert check.status == "not_measured"
        assert compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=NONSENSE_OK, catch_rate=check).rung == 1

    def test_no_catch_rate_supplied_is_not_a_pass(self):
        assert compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=NONSENSE_OK).rung is ProbeRung.HELD_OUT

    def test_the_share_is_a_parameter(self):
        assert catch_rate_check([RecallSet("a", 100, 10)], min_recall=0.1).passed
        assert not catch_rate_check([RecallSet("a", 100, 9)], min_recall=0.1).passed


class TestTheNonsenseCheck:
    def test_no_nonsense_set_is_rung_1_and_says_so(self):
        result = compute(OLD_RUNG_2, EVERYDAY_OK, catch_rate=_catch(100, 90))
        assert result.rung is ProbeRung.HELD_OUT
        assert f"rung 2 not reached: {NOT_CHECKED}" in result.reasons
        assert result.next_step == NOT_CHECKED
        assert result.nonsense_text["status"] == "not_checked" and result.nonsense_text["kind"] == "nonsense_text"

    def test_false_alarms_over_the_limit_is_rung_1_with_the_numbers(self):
        check = nonsense_check([EverydaySet("salad", 2000, 41, 1.0)], target_fpr=0.01)
        result = compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=check, catch_rate=_catch(100, 90))
        assert result.rung is ProbeRung.HELD_OUT
        reason = next(r for r in result.reasons if r.startswith("rung 2 not reached"))
        assert "false alarms on nonsense text 2.05% at the shipped bar (allowed ≤ 2%)" in reason
        assert "41 of 2,000 rows, allowed up to 40" in reason
        assert check.sets[0]["sentence"] == "On nonsense text: 41 false alarms in 2,000 (allowed up to 40)"
        ok = nonsense_check([EverydaySet("salad", 2000, 40, 1.0)], target_fpr=0.01)
        assert compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=ok, catch_rate=_catch(100, 90)).rung == 2

    def test_firing_on_all_of_it_says_so(self):
        check = nonsense_check([EverydaySet("salad", 1000, 1000, 1.0)], target_fpr=0.01)
        assert check.sets[0]["sentence"] == "Fired on 100% of nonsense text (1,000 rows)"

    def test_a_small_set_does_not_count(self):
        check = nonsense_check([EverydaySet("salad", 999, 0, 1.0)], target_fpr=0.01)
        assert check.status == "not_checked"
        assert compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=check, catch_rate=_catch(100, 90)).rung == 1

    def test_all_four_together_reach_rung_2_and_3(self):
        assert compute(OLD_RUNG_2, EVERYDAY_OK, nonsense=NONSENSE_OK, catch_rate=_catch(100, 90)).rung == 2
        judged = [SetResult("held_out", False, 0.9), SetResult("ood", True, 0.9, judge_completed=True)]
        assert compute(judged, EVERYDAY_OK, nonsense=NONSENSE_OK, catch_rate=_catch(100, 90)).rung == 3

    def test_the_catch_rate_is_named_first_when_several_fail(self):
        result = compute(OLD_RUNG_2, everyday_check([], target_fpr=0.01), catch_rate=_catch(100, 5))
        failed = [r for r in result.reasons if r.startswith("rung 2 not reached")]
        assert len(failed) == 3 and "catches 5%" in failed[0]
        assert result.next_step.startswith("catches 5%")

    def test_the_wording_keeps_the_forbidden_words_out(self):
        from src.schemas.evidence_ladder import PROBE_FORBIDDEN_WORDS

        bad = nonsense_check([EverydaySet("p", 1200, 1200)], target_fpr=0.01)
        text = " ".join([NOT_CHECKED, bad.reason, bad.next_step, _catch(100, 1).reason,
                         _catch(100, 1).next_step]).lower()
        assert not [w for w in PROBE_FORBIDDEN_WORDS if w in text]


# ── recompute_rung against the database and the stored scores ───────────────────────────────


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


def _world(db, tmp_path, *, threshold, ood_positive_scores, ood_negative_scores=None, nonsense_scores=None,
           store_scores=True):
    """A probe with an in-distribution pass, an OOD set whose per-row scores are ON DISK (as the writer stores
    them, with the metrics `evaluate` really produces), a passed everyday-text check, and — unless
    `nonsense_scores` is None — a nonsense-text evaluation with its scores on disk."""
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorEvaluation, ProbeMonitorRun
    from src.services.probe_monitor_metrics import evaluate
    from src.services.probe_monitor_rung import everyday_false_alarms, shipped_bar_recall
    from tests.support.everyday import add_everyday_pass

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY, file_path="/x",
                  architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="c", source="local", status=DatasetStatus.READY, raw_path="/x",
                      extra_metadata={})
    db.add_all([model, dataset])
    db.flush()

    def view(role, distribution=None, name=None):
        row = ProbeMonitorDataset(name=name or f"v-{uuid.uuid4().hex[:6]}", dataset_id=dataset.id,
                                  input_column="i", label_column="l", label_mapping={}, role=role,
                                  distribution=distribution, counts={})
        db.add(row)
        db.flush()
        return row

    train = view("train")
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=train.id, eval_dataset_ids=[], config={},
                          status="completed", environment={})
    db.add(run)
    db.flush()
    probe = ProbeMonitor(run_id=run.id, layer=0, rule="max", rule_params={}, variant="dense", val_metrics={},
                         rung=2, rung_reasons=[], threshold=threshold, target_fpr=0.01,
                         weights_path="/nonexistent/w.pt")
    db.add(probe)
    db.flush()
    db.add(ProbeMonitorEvaluation(probe_id=probe.id, dataset_id=view("eval", "in_distribution").id,
                                  status="completed",
                                  metrics={"scored": True, "auroc": 0.95, "ci": {"low": 0.9, "high": 1.0}}))
    rng = np.random.default_rng(3)
    negatives = (np.asarray(ood_negative_scores, dtype=np.float32) if ood_negative_scores is not None
                 else rng.normal(0.0, 1.0, 400).astype(np.float32))
    positives = np.asarray(ood_positive_scores, dtype=np.float32)
    scores = np.concatenate([positives, negatives])
    labels = np.concatenate([np.ones(len(positives), np.int8), np.zeros(len(negatives), np.int8)])
    ood = view("eval", "out_of_distribution", name="carriers")
    metrics = evaluate(scores.tolist(), labels.tolist(), name="carriers", out_of_distribution=True,
                       target_fpr=0.01)
    metrics["catch_rate"] = shipped_bar_recall(scores.tolist(), labels.tolist(), threshold)
    paths = {}
    if store_scores:
        stem = tmp_path / f"{probe.id}__{ood.id}"
        np.save(f"{stem}.npy", scores)
        np.save(f"{stem}__labels.npy", labels)
        paths = {"scores_path": f"{stem}.npy", "labels_path": f"{stem}__labels.npy"}
    db.add(ProbeMonitorEvaluation(probe_id=probe.id, dataset_id=ood.id, status="completed", metrics=metrics,
                                  n_positive=len(positives), n_negative=len(negatives), **paths))
    add_everyday_pass(db, probe, dataset.id, rows=2000, directory=tmp_path)
    nonsense = None
    if nonsense_scores is not None:
        nonsense = view("nonsense_text", name="salad")
        values = np.asarray(nonsense_scores, dtype=np.float32)
        zeros = np.zeros(len(values), dtype=np.int8)
        stem = tmp_path / f"{probe.id}__{nonsense.id}"
        np.save(f"{stem}.npy", values)
        np.save(f"{stem}__labels.npy", zeros)
        db.add(ProbeMonitorEvaluation(
            probe_id=probe.id, dataset_id=nonsense.id, status="refused", n_positive=0, n_negative=len(values),
            metrics={"scored": False, "nonsense_text": everyday_false_alarms(values.tolist(), zeros.tolist(),
                                                                              threshold)},
            scores_path=f"{stem}.npy", labels_path=f"{stem}__labels.npy"))
    db.commit()
    return SimpleNamespace(probe=probe, run=run, ood=ood, dataset=dataset, nonsense=nonsense)


def _recompute(db, probe):
    from src.services.probe_monitor_run import recompute_rung

    outcome = recompute_rung(db, probe.id)
    db.refresh(probe)
    return outcome


QUIET = np.full(2000, -5.0)              # a nonsense set the probe never fires on


class TestRecompute:
    async def test_an_inert_probe_with_zero_false_alarms_is_rung_1(self, sync_db, tmp_path):
        """Recall 2% at the bar: 4 of 200 positives at or above 10.0, nothing else fires anywhere."""
        positives = np.concatenate([np.full(4, 11.0), np.full(196, 0.5)])
        world = _world(sync_db, tmp_path, threshold=10.0, ood_positive_scores=positives, nonsense_scores=QUIET)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1
        reason = next(r for r in world.probe.rung_reasons if r.startswith("rung 2 not reached"))
        assert "catches 2% of positives on unseen tasks at the shipped bar (needs ≥ 50%)" in reason
        detail = world.probe.rung_detail
        assert detail["everyday_text"]["passed"] and detail["nonsense_text"]["passed"]
        assert [(s["positives"], s["caught"]) for s in detail["catch_rate"]["sets"]] == [(200, 4)]
        assert detail["next_step"].startswith("catches 2% of positives")

    async def test_recall_is_measured_at_the_shipped_bar_not_the_sets_own(self, sync_db, tmp_path):
        """Positives ~N(3,1), negatives ~N(0,1): the set's own 1%-FPR bar (~2.3) catches ~75%; the shipped
        bar of 4.3 catches ~10%. Measured at its own bar (K3) this probe would pass."""
        rng = np.random.default_rng(11)
        world = _world(sync_db, tmp_path, threshold=4.3, ood_positive_scores=rng.normal(3.0, 1.0, 400),
                       nonsense_scores=QUIET)
        own = next(p for p in world_metrics(sync_db, world)["operating_points"] if p["target_fpr"] == 0.01)
        assert own["recall"] > 0.5                       # the trap: the set's own bar reads fine
        _recompute(sync_db, world.probe)
        sets = world.probe.rung_detail["catch_rate"]["sets"]
        assert sets[0]["recall"] < 0.2 and sets[0]["threshold"] == pytest.approx(4.3)
        assert world.probe.rung == 1

    async def test_enough_caught_and_quiet_on_nonsense_is_rung_2(self, sync_db, tmp_path):
        world = _world(sync_db, tmp_path, threshold=2.0, ood_positive_scores=np.full(100, 3.0),
                       nonsense_scores=QUIET)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 2
        assert world.probe.rung_detail["catch_rate"]["sets"][0]["sentence"] == (
            "On unseen tasks (carriers ({})): catches 100% of positives at the shipped bar (100 of 100; "
            "needs ≥ 50%)".format(world.ood.id))
        assert world.probe.rung_detail["nonsense_text"]["sets"][0]["sentence"] == (
            "On nonsense text: 0 false alarms in 2,000 (allowed up to 40)")

    async def test_no_nonsense_set_is_rung_1(self, sync_db, tmp_path):
        world = _world(sync_db, tmp_path, threshold=2.0, ood_positive_scores=np.full(100, 3.0))
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1 and world.probe.rung_detail["next_step"] == NOT_CHECKED

    async def test_nonsense_false_alarms_over_the_limit_is_rung_1(self, sync_db, tmp_path):
        noisy = np.concatenate([np.full(41, 2.5), np.full(1959, -5.0)])     # 41 of 2,000 at or above 2.0
        world = _world(sync_db, tmp_path, threshold=2.0, ood_positive_scores=np.full(100, 3.0),
                       nonsense_scores=noisy)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1
        reason = next(r for r in world.probe.rung_reasons if r.startswith("rung 2 not reached"))
        assert "false alarms on nonsense text 2.05% at the shipped bar (allowed ≤ 2%)" in reason
        assert "41 of 2,000 rows, allowed up to 40" in reason

    async def test_the_catch_rate_is_retaken_at_the_bar_in_force(self, sync_db, tmp_path):
        """A re-cut raised the bar from 2.0 to 3.5: the stored scores are re-read there, and nobody fires."""
        world = _world(sync_db, tmp_path, threshold=2.0, ood_positive_scores=np.full(100, 3.0),
                       nonsense_scores=QUIET)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 2
        world.probe.threshold = 3.5
        sync_db.commit()
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1
        assert world.probe.rung_detail["catch_rate"]["sets"][0]["caught"] == 0

    async def test_without_score_files_the_recorded_count_counts_only_at_the_same_bar(self, sync_db, tmp_path):
        world = _world(sync_db, tmp_path, threshold=2.0, ood_positive_scores=np.full(100, 3.0),
                       nonsense_scores=QUIET, store_scores=False)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 2                     # recorded at 2.0, bar is 2.0
        world.probe.threshold = 2.5
        sync_db.commit()
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1
        assert world.probe.rung_detail["catch_rate"]["status"] == "not_measured"

    async def test_the_three_knobs_are_the_runtime_settings(self, sync_db, tmp_path):
        """READER test for `rung2_min_recall`, `nonsense_text_fpr_factor` and `nonsense_text_min_rows`."""
        positives = np.concatenate([np.full(30, 3.0), np.full(70, 0.0)])          # 30% caught at 2.0
        noisy = np.concatenate([np.full(100, 2.5), np.full(1900, -5.0)])           # 5% false alarms
        world = _world(sync_db, tmp_path, threshold=2.0, ood_positive_scores=positives, nonsense_scores=noisy)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1
        runtime_settings.apply({"rung2_min_recall": 0.3, "nonsense_text_fpr_factor": 5.0})
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 2
        assert world.probe.rung_detail["catch_rate"]["min_recall"] == 0.3
        assert world.probe.rung_detail["nonsense_text"]["factor"] == 5.0
        runtime_settings.apply({"rung2_min_recall": 0.3, "nonsense_text_fpr_factor": 5.0,
                                "nonsense_text_min_rows": 2001})
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1 and world.probe.rung_detail["nonsense_text"]["min_rows"] == 2001

    async def test_the_report_and_list_carry_both_blocks(self, sync_db, client, tmp_path):
        positives = np.concatenate([np.full(4, 11.0), np.full(196, 0.5)])
        world = _world(sync_db, tmp_path, threshold=10.0, ood_positive_scores=positives, nonsense_scores=QUIET)
        _recompute(sync_db, world.probe)
        report = (await client.get(f"{API}/probes/{world.probe.id}")).json()
        assert report["probe"]["rung"] == 1
        assert report["rung_next_step"].startswith("catches 2% of positives on unseen tasks at the shipped bar")
        assert report["catch_rate"]["status"] == "too_few_caught"
        assert report["nonsense_text"]["sets"][0]["sentence"] == "On nonsense text: 0 false alarms in 2,000 (allowed up to 40)"
        nonsense_eval = next(e for e in report["evaluations"] if e["dataset_id"] == world.nonsense.id)
        assert nonsense_eval["counts_towards_rung"] is True
        listed = (await client.get(f"{API}/probes", params={"run_id": world.run.id})).json()
        assert [p["rung_next_step"] for p in listed] == [report["rung_next_step"]]


def world_metrics(db, world):
    from src.models.probe_monitor import ProbeMonitorEvaluation

    row = db.query(ProbeMonitorEvaluation).filter(ProbeMonitorEvaluation.dataset_id == world.ood.id).one()
    return row.metrics


# ── registration ─────────────────────────────────────────────────────────────────────────────


class TestRegistration:
    def _body(self, **over):
        body = dict(name="salad", dataset_id=str(uuid.uuid4()), input_column="text", label_column="kind",
                    label_mapping={"*": "negative"}, role="nonsense_text")
        body.update(over)
        return body

    def test_a_nonsense_set_is_negatives_only_and_behaviour_agnostic(self):
        from pydantic import ValidationError

        from src.schemas.probe_monitor import ProbeDatasetCreate

        assert ProbeDatasetCreate(**self._body()).role == "nonsense_text"
        with pytest.raises(ValidationError, match="must not map any value to 'positive'"):
            ProbeDatasetCreate(**self._body(label_mapping={"a": "negative", "b": "positive"}))
        with pytest.raises(ValidationError, match="behaviour-agnostic"):
            ProbeDatasetCreate(**self._body(behaviour="some-key"))


# ── the D84 migration ───────────────────────────────────────────────────────────────────────


def test_the_migration_restates_cached_rungs_and_round_trips():
    """`e010d84a7c31` on a scratch table: rung 2 with an in-distribution pass -> 1, rung 3 without -> 0, rung 1
    untouched; reason appended; next step stored; definition invalidated; the downgrade restores rung,
    reasons and detail."""
    import importlib.util
    import json
    import os
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import text

    path = (Path(__file__).resolve().parents[2] / "alembic" / "versions"
            / "e010d84a7c31_rung2_catch_rate_and_nonsense_text.py")
    spec = importlib.util.spec_from_file_location("rev_e010d84a7c31", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.revision, module.down_revision) == ("e010d84a7c31", "e009a1d82e70")
    assert module.NOT_CHECKED == NONSENSE_NOT_CHECKED

    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio_test"
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_d84_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    engine = create_engine(server + "/" + name)
    in_passed = ["CI lower bound above 0.5 in-distribution on: a", "every out-of-distribution set run (1) cleared 0.5"]
    ood_only = ["no in-distribution set cleared 0.5, so rung 1 is not claimed", "every ..."]
    before_detail = {"next_step": "x", "everyday_text": {"status": "passed"}}
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS probe_monitors"))
            conn.execute(text(
                "CREATE TABLE probe_monitors (id varchar(36) primary key, rung integer not null, "
                "rung_reasons jsonb not null, target_fpr double precision, definition_path varchar(1000), "
                "definition_built_at timestamptz, definition_sha256 varchar(64), definition_build jsonb, "
                "rung_detail jsonb)"))
            conn.execute(text(
                "INSERT INTO probe_monitors VALUES "
                "('a', 2, CAST(:ip AS jsonb), 0.01, '/d/a.json', now(), :sha, "
                "  CAST('{\"resolved\": {\"detector_digest\": \"dd\", \"decision_digest\": \"ee\"}}' AS jsonb), "
                "  CAST(:det AS jsonb)),"
                "('b', 3, CAST(:oo AS jsonb), 0.01, NULL, NULL, NULL, NULL, NULL),"
                "('c', 1, CAST(:ip AS jsonb), 0.01, '/d/c.json', now(), :sha, NULL, NULL)"),
                {"ip": json.dumps(in_passed), "oo": json.dumps(ood_only), "sha": "c" * 64,
                 "det": json.dumps(before_detail)})
            with Operations.context(MigrationContext.configure(conn)):
                module.upgrade()
            rows = {r.id: r for r in conn.execute(text("SELECT * FROM probe_monitors"))}
        assert (rows["a"].rung, rows["b"].rung, rows["c"].rung) == (1, 0, 1)
        assert rows["a"].rung_reasons[-1] == f"rung 2 not reached: {NOT_CHECKED}"
        assert rows["a"].rung_detail["next_step"] == NOT_CHECKED
        assert rows["a"].rung_detail["everyday_text"] == {"status": "passed"}       # kept
        assert rows["a"].rung_detail["nonsense_text"]["allowed_rate"] == pytest.approx(0.02)
        assert rows["a"].definition_path is None
        inv = rows["a"].definition_build["invalidated"]
        assert inv["cause"] == "rung" and "2 -> 1" in inv["reason"] and inv["was_identity"]["detector_digest"] == "dd"
        assert rows["c"].rung_detail is None and rows["c"].definition_path == "/d/c.json"   # untouched
        with engine.begin() as conn:
            with Operations.context(MigrationContext.configure(conn)):
                module.downgrade()
            back = {r.id: r for r in conn.execute(text("SELECT * FROM probe_monitors"))}
        assert (back["a"].rung, back["b"].rung, back["c"].rung) == (2, 3, 1)
        assert back["a"].rung_reasons == in_passed and back["b"].rung_reasons == ood_only
        assert back["a"].rung_detail == before_detail and back["b"].rung_detail is None
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS probe_monitors"))
        engine.dispose()
