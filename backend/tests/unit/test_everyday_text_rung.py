"""ENH-006 D82 (2026-10-29): rung 2 needs few false alarms on EVERYDAY TEXT at the SHIPPED bar.

Why: miLLM's OpenTelemetry benchmark sent random everyday words through LFM2.5 with the nine probes of
`pmr_bbe9a3608802` armed; the right answer was "absent" every time. The layer-0 `mean` probe fired on
280/280 and the layer-1 `attention` probe on 140/280, both at rung 2 ("detects on unseen tasks") — their
out-of-distribution sets were other carriers of the concept, never plain prose.

The rule: a probe reaches rung 2 only with a completed evaluation on at least one everyday-text set
(view role `everyday_text`) of at least `everyday_text_min_rows` rows (1,000), on which its false-alarm
rate AT THE SHIPPED BAR is at most `everyday_text_fpr_factor` (2) x its target rate. Otherwise at most
rung 1, and the reasons and next step say which.

Negative controls (0xcc/reviews/everyday_text_check_controls_2026-10-29.md):
  C1 drop the requirement in `compute` -> red (TestThePureRule, TestRecompute)
  C2 measure at the set's own target-FPR bar instead of the shipped bar -> red (TestRecompute)
  C3 ignore the minimum rows -> red (TestThePureRule::test_a_set_below_the_minimum_does_not_count)
  C4 allowance = target instead of factor x target -> red (test_the_allowance_is_twice_the_target)
  C5 drop `recompute_rung` from the recalibration commit -> red (test_a_recut_recomputes_the_rung)
  C6 report reads the generic step instead of the stored one -> red (test_the_report_and_list_carry_it)
"""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core import runtime_settings
from src.core.config import settings
from src.schemas.evidence_ladder import ProbeRung
from src.services.probe_monitor_rung import (
    EVERYDAY_NOT_CHECKED,
    EverydaySet,
    RecallSet,
    SetResult,
    catch_rate_check,
    compute as _compute,
    everyday_check,
    nonsense_check,
)

#: ENH-006 D84: rung 2 also needs passed nonsense-text and catch-rate checks. This file tests the
#: everyday-text half, so the other two are given passing (tested in `test_rung2_tighten.py`).
NONSENSE_PASSED = nonsense_check([EverydaySet("nonsense", 1000, 0, 0.0)], target_fpr=0.01)
CATCH_PASSED = catch_rate_check([RecallSet("ood", 100, 100, 0.0)])


def compute(results, everyday=None):
    return _compute(results, everyday, nonsense=NONSENSE_PASSED, catch_rate=CATCH_PASSED)

API = "/api/v1/probe-monitors"
NOT_CHECKED = "not yet checked on everyday text — evaluate on an everyday-text set"
OLD_RUNG_2 = [SetResult("held_out", False, 0.9), SetResult("ood", True, 0.9)]
OLD_RUNG_3 = [SetResult("held_out", False, 0.9), SetResult("ood", True, 0.9, judge_completed=True)]


@pytest.fixture(autouse=True)
def _fresh_settings():
    runtime_settings.reset()
    yield
    runtime_settings.reset()


def test_the_phrase_is_the_operator_decision():
    assert EVERYDAY_NOT_CHECKED == NOT_CHECKED


# ── the pure rule ─────────────────────────────────────────────────────────────────────────────


class TestThePureRule:
    def test_old_rung_2_evidence_without_an_everyday_set_is_rung_1(self):
        result = compute(OLD_RUNG_2, everyday_check([], target_fpr=0.01))
        assert result.rung is ProbeRung.HELD_OUT
        assert f"rung 2 not reached: {NOT_CHECKED}" in result.reasons
        assert result.next_step == NOT_CHECKED
        assert result.everyday_text["status"] == "not_checked"

    def test_no_check_supplied_is_not_a_pass(self):
        """`None` means not checked — absence of evidence never promotes."""
        assert compute(OLD_RUNG_2).rung is ProbeRung.HELD_OUT
        assert compute(OLD_RUNG_3).rung is ProbeRung.HELD_OUT

    def test_old_rung_3_evidence_is_capped_too(self):
        assert compute(OLD_RUNG_3, everyday_check([], target_fpr=0.01)).rung is ProbeRung.HELD_OUT

    def test_without_an_in_distribution_pass_the_cap_is_rung_0(self):
        result = compute([SetResult("ood", True, 0.9)], everyday_check([], target_fpr=0.01))
        assert result.rung is ProbeRung.TRAINED

    def test_firing_on_all_everyday_text_is_rung_1_with_the_numbers(self):
        check = everyday_check([EverydaySet("prose", 1200, 1200, 0.5)], target_fpr=0.01)
        result = compute(OLD_RUNG_2, check)
        assert result.rung is ProbeRung.HELD_OUT
        reason = next(r for r in result.reasons if r.startswith("rung 2 not reached"))
        assert "false alarms on everyday text 100% at the shipped bar (allowed ≤ 2%)" in reason
        assert "1,200 of 1,200 rows, allowed up to 24" in reason
        assert result.next_step.startswith("false alarms on everyday text 100% at the shipped bar (allowed ≤ 2%)")
        assert check.sets[0]["sentence"] == "Fired on 100% of everyday text (1,200 rows)"

    def test_within_the_allowance_with_enough_rows_is_rung_2(self):
        check = everyday_check([EverydaySet("prose", 1200, 0, 0.5)], target_fpr=0.01)
        result = compute(OLD_RUNG_2, check)
        assert result.rung is ProbeRung.UNSEEN_TASKS
        assert check.sets[0]["sentence"] == "On everyday text: 0 false alarms in 1,200 (allowed up to 24)"
        assert compute(OLD_RUNG_3, check).rung is ProbeRung.JUDGE_COMPARED

    def test_the_allowance_is_twice_the_target_and_inclusive(self):
        """24 of 1,200 at a 1% target is exactly 2x and passes; 25 does not; 13 (above 1x) passes."""
        assert compute(OLD_RUNG_2, everyday_check([EverydaySet("p", 1200, 24)], target_fpr=0.01)).rung == 2
        assert compute(OLD_RUNG_2, everyday_check([EverydaySet("p", 1200, 13)], target_fpr=0.01)).rung == 2
        assert compute(OLD_RUNG_2, everyday_check([EverydaySet("p", 1200, 25)], target_fpr=0.01)).rung == 1

    def test_a_set_below_the_minimum_does_not_count(self):
        check = everyday_check([EverydaySet("small", 999, 0, 0.5)], target_fpr=0.01)
        result = compute(OLD_RUNG_2, check)
        assert result.rung is ProbeRung.HELD_OUT and check.status == "not_checked"
        assert "999 rows, fewer than the 1,000 the check needs" in check.sets[0]["sentence"]
        # 1,000 exactly counts
        assert compute(OLD_RUNG_2, everyday_check([EverydaySet("p", 1000, 0)], target_fpr=0.01)).rung == 2

    def test_every_counted_set_must_pass(self):
        check = everyday_check([EverydaySet("a", 1000, 0), EverydaySet("b", 1000, 500)], target_fpr=0.01)
        assert compute(OLD_RUNG_2, check).rung is ProbeRung.HELD_OUT

    def test_a_probe_with_no_target_rate_cannot_pass(self):
        assert everyday_check([EverydaySet("a", 5000, 0)], target_fpr=None).status == "not_checked"

    def test_the_wording_keeps_the_forbidden_words_out(self):
        from src.schemas.evidence_ladder import PROBE_FORBIDDEN_WORDS

        check = everyday_check([EverydaySet("p", 1200, 1200)], target_fpr=0.01)
        text = " ".join([NOT_CHECKED, check.reason, check.next_step]).lower()
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


def _world(db, *, threshold=0.5, target_fpr=0.01, built=False):
    """A probe that MEETS the old rung-2 evidence: a passing in-distribution and out-of-distribution set."""
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorEvaluation, ProbeMonitorRun

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY, file_path="/x",
                  architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="c", source="local", status=DatasetStatus.READY, raw_path="/x",
                      extra_metadata={})
    db.add_all([model, dataset])
    db.flush()

    def view(role, distribution=None):
        row = ProbeMonitorDataset(name=f"v-{uuid.uuid4().hex[:6]}", dataset_id=dataset.id, input_column="i",
                                  label_column="l", label_mapping={}, role=role, distribution=distribution,
                                  counts={})
        db.add(row)
        db.flush()
        return row

    train = view("train")
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=train.id, eval_dataset_ids=[], config={},
                          status="completed", environment={})
    db.add(run)
    db.flush()
    probe = ProbeMonitor(run_id=run.id, layer=0, rule="mean", rule_params={}, variant="dense", val_metrics={},
                         rung=2, rung_reasons=[], threshold=threshold, target_fpr=target_fpr,
                         weights_path="/nonexistent/w.pt")
    if built:
        probe.definition_path = "/nonexistent/x.probe.json"
        probe.definition_sha256 = "c" * 64
        probe.definition_build = {"resolved": {"detector_digest": "d" * 64}}
    db.add(probe)
    db.flush()
    for distribution in ("in_distribution", "out_of_distribution"):
        db.add(ProbeMonitorEvaluation(probe_id=probe.id, dataset_id=view("eval", distribution).id,
                                      status="completed",
                                      metrics={"scored": True, "auroc": 0.95, "ci": {"low": 0.9, "high": 1.0},
                                               # ENH-006 D84: a full catch rate, recorded at this bar.
                                               "catch_rate": {"threshold": threshold, "positives": 100,
                                                              "caught": 100}}))
    # ENH-006 D84: and a passed nonsense-text check — this file tests the everyday-text half.
    from tests.support.everyday import add_nonsense_pass

    add_nonsense_pass(db, probe, dataset.id)
    db.commit()
    return SimpleNamespace(probe=probe, run=run, view=view)


def _everyday(db, world, scores, tmp_path: Path, *, recorded_at=None):
    """An everyday-text view and its evaluation, with the per-row scores ON DISK as the writer stores them."""
    from src.models.probe_monitor import ProbeMonitorEvaluation
    from src.services.probe_monitor_rung import everyday_false_alarms

    view = world.view("everyday_text")
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.zeros(len(scores), dtype=np.int8)
    stem = tmp_path / f"{world.probe.id}__{view.id}__{uuid.uuid4().hex[:6]}"
    np.save(f"{stem}.npy", scores)
    np.save(f"{stem}__labels.npy", labels)
    bar = world.probe.threshold if recorded_at is None else recorded_at
    db.add(ProbeMonitorEvaluation(
        probe_id=world.probe.id, dataset_id=view.id, status="refused", n_positive=0, n_negative=len(scores),
        metrics={"scored": False, "everyday_text": everyday_false_alarms(scores.tolist(), labels.tolist(), bar)},
        scores_path=f"{stem}.npy", labels_path=f"{stem}__labels.npy"))
    db.commit()
    return view


def _recompute(db, probe):
    from src.services.probe_monitor_run import recompute_rung

    outcome = recompute_rung(db, probe.id)
    db.refresh(probe)
    return outcome


class TestRecompute:
    async def test_no_everyday_set_drops_old_rung_2_to_1_and_says_why(self, sync_db):
        world = _world(sync_db)
        outcome = _recompute(sync_db, world.probe)
        assert world.probe.rung == 1 and outcome["rung"] == 1
        assert f"rung 2 not reached: {NOT_CHECKED}" in world.probe.rung_reasons
        assert world.probe.rung_detail["next_step"] == NOT_CHECKED
        assert world.probe.rung_detail["everyday_text"]["status"] == "not_checked"

    async def test_a_drop_invalidates_the_cached_definition_like_any_rung_change(self, sync_db):
        world = _world(sync_db, built=True)
        _recompute(sync_db, world.probe)
        assert world.probe.definition_path is None
        assert world.probe.definition_build["invalidated"]["cause"] == "rung"

    async def test_firing_on_every_everyday_row_at_the_shipped_bar_is_rung_1(self, sync_db, tmp_path):
        world = _world(sync_db, threshold=0.5)
        # Scores spread 0.6..1.6: every one is above the shipped bar of 0.5, so 100% false alarms there —
        # while the set's OWN 1% bar would sit near 1.59 and fire on ~1% (C2's mutation).
        _everyday(sync_db, world, np.linspace(0.6, 1.6, 1200), tmp_path)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1
        reason = next(r for r in world.probe.rung_reasons if r.startswith("rung 2 not reached"))
        assert "false alarms on everyday text 100% at the shipped bar (allowed ≤ 2%)" in reason
        sets = world.probe.rung_detail["everyday_text"]["sets"]
        assert [(s["rows"], s["false_alarms"], s["allowed"]) for s in sets] == [(1200, 1200, 24)]

    async def test_few_false_alarms_with_enough_rows_reaches_rung_2(self, sync_db, tmp_path):
        world = _world(sync_db, threshold=2.0)
        _everyday(sync_db, world, np.linspace(0.6, 1.6, 1200), tmp_path)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 2
        assert world.probe.rung_detail["everyday_text"]["sets"][0]["sentence"] == (
            "On everyday text: 0 false alarms in 1,200 (allowed up to 24)")

    async def test_the_count_is_retaken_at_the_bar_in_force(self, sync_db, tmp_path):
        """Recorded at 2.0 (0 alarms) but the bar is now 0.5: the stored scores are re-read at 0.5."""
        world = _world(sync_db, threshold=0.5)
        _everyday(sync_db, world, np.linspace(0.6, 1.6, 1200), tmp_path, recorded_at=2.0)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1

    async def test_a_set_below_the_minimum_rows_does_not_count(self, sync_db, tmp_path):
        world = _world(sync_db, threshold=2.0)
        _everyday(sync_db, world, np.linspace(0.6, 1.6, 999), tmp_path)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1
        assert world.probe.rung_detail["next_step"] == NOT_CHECKED

    async def test_the_two_knobs_are_the_runtime_settings(self, sync_db, tmp_path):
        """READER test for `everyday_text_min_rows` and `everyday_text_fpr_factor`."""
        world = _world(sync_db, threshold=1.5)
        # 0.6..1.6 over 1,000 rows: ~100 rows (10%) at or above 1.5.
        _everyday(sync_db, world, np.linspace(0.6, 1.6, 1000), tmp_path)
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1                                   # 10% > 2% at the defaults
        runtime_settings.apply({"everyday_text_fpr_factor": 20.0})     # allowance 20%
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 2
        assert world.probe.rung_detail["everyday_text"]["factor"] == 20.0
        runtime_settings.apply({"everyday_text_fpr_factor": 20.0, "everyday_text_min_rows": 1001})
        _recompute(sync_db, world.probe)
        assert world.probe.rung == 1 and world.probe.rung_detail["everyday_text"]["min_rows"] == 1001


# ── the export gate, the recalibration commit, the report and list ────────────────────────────


class TestSurfaces:
    async def test_the_export_gate_refuses_below_rung_2_without_acknowledgement(self, sync_db):
        """Through `build`, which recomputes the rung before it gates: a cached rung 2 is not trusted."""
        from src.services.probe_definition_builder import ProbeExportRefused, build

        world = _world(sync_db)
        assert world.probe.rung == 2  # the cache says 2
        with pytest.raises(ProbeExportRefused) as refused:
            build(sync_db, world.probe.id, model_loader=lambda *a, **k: pytest.fail("gate passed"))
        assert "at rung 1" in str(refused.value) and NOT_CHECKED in str(refused.value)

    async def test_the_export_gate_takes_an_acknowledgement(self, sync_db):
        from src.services.probe_definition_builder import check_export_gate
        from src.services.probe_monitor_run import recompute_rung

        world = _world(sync_db)
        recompute_rung(sync_db, world.probe.id)
        sync_db.refresh(world.probe)
        ack = check_export_gate(world.probe, world.run,
                                acknowledge_below_rung2={"reason": "pilot only, everyday text pending"})
        assert ack is not None and ack.reason.startswith("pilot only")

    async def test_the_report_and_list_carry_it(self, sync_db, client, tmp_path):
        world = _world(sync_db, threshold=0.5)
        _everyday(sync_db, world, np.linspace(0.6, 1.6, 1200), tmp_path)
        _recompute(sync_db, world.probe)
        report = (await client.get(f"{API}/probes/{world.probe.id}")).json()
        assert report["probe"]["rung"] == 1
        assert report["rung_next_step"].startswith("false alarms on everyday text 100% at the shipped bar")
        assert report["probe"]["rung_next_step"] == report["rung_next_step"]
        assert report["everyday_text"]["sets"][0]["sentence"] == "Fired on 100% of everyday text (1,200 rows)"
        everyday_eval = next(e for e in report["evaluations"] if e["status"] == "refused")
        assert everyday_eval["counts_towards_rung"] is True
        listed = (await client.get(f"{API}/probes", params={"run_id": world.run.id})).json()
        assert [p["rung_next_step"] for p in listed] == [report["rung_next_step"]]

    def test_a_recut_recomputes_the_rung(self, tmp_path, monkeypatch):
        """The bar moved, so the false-alarm count at the shipped bar moved: `apply` asks the ladder again."""
        from src.services import probe_monitor_run
        from src.services import probe_recalibration as R
        from tests.unit.test_probe_recalibration import _FakeDb, _probe

        calls = []
        monkeypatch.setattr(probe_monitor_run, "recompute_rung", lambda db, pid: calls.append(pid) or {})
        monkeypatch.setattr(R, "_lock_row", lambda db, probe: probe)   # the stub session's lock, as there
        rng = np.random.default_rng(7)
        probe = _probe(tmp_path, negatives=sorted(rng.normal(0, 5, 2000).tolist(), reverse=True), threshold=5.0)
        R.apply(_FakeDb(), probe, R.propose(probe, [], target_fpr=0.005), reason="tighter")
        assert calls == [probe.id]


# ── registration ─────────────────────────────────────────────────────────────────────────────


class TestRegistration:
    def _body(self, **over):
        body = dict(name="prose", dataset_id=str(uuid.uuid4()), input_column="text", label_column="label",
                    label_mapping={"0": "negative"}, role="everyday_text")
        body.update(over)
        return body

    def test_an_everyday_set_is_negatives_only(self):
        from pydantic import ValidationError

        from src.schemas.probe_monitor import ProbeDatasetCreate

        assert ProbeDatasetCreate(**self._body()).role == "everyday_text"
        with pytest.raises(ValidationError, match="must not map any value to 'positive'"):
            ProbeDatasetCreate(**self._body(label_mapping={"0": "negative", "1": "positive"}))

    @pytest.mark.parametrize("field, value", [("behaviour", "some-key"), ("sub_question", "sq-a"),
                                              ("positive_meaning", "x"), ("distribution", "out_of_distribution")])
    def test_an_everyday_set_is_behaviour_agnostic(self, field, value):
        from pydantic import ValidationError

        from src.schemas.probe_monitor import ProbeDatasetCreate

        with pytest.raises(ValidationError):
            ProbeDatasetCreate(**self._body(**{field: value}))


    def test_every_row_negative_is_the_wildcard_and_only_here(self):
        from pydantic import ValidationError

        from src.schemas.probe_monitor import ProbeDatasetCreate
        from src.services.probe_monitor_inputs import map_label

        assert ProbeDatasetCreate(**self._body(label_mapping={"*": "negative"})).label_mapping == {"*": "negative"}
        with pytest.raises(ValidationError, match="only on an everyday-text or nonsense-text set"):
            ProbeDatasetCreate(**self._body(role="eval", label_mapping={"*": "negative", "1": "positive"}))
        with pytest.raises(ValidationError, match="only on an everyday-text or nonsense-text set"):
            ProbeDatasetCreate(**self._body(role="calibration", label_mapping={"*": "negative"}))
        assert map_label("anything at all", {"*": "negative"}) == "negative"
        assert map_label("*", {"*": "excluded", "x": "negative"}) == "excluded"   # exact key first
        assert map_label("y", {"x": "negative"}) is None                          # no wildcard, unmapped

    async def test_a_text_only_corpus_registers_through_the_route(self, client, async_session, data_dir):
        """An OpenWebText-like split with ONE column: the text is the label column, every row negative."""
        from tests.support.detector_fixtures import dataset_row, write_split

        raw = write_split(data_dir, {"text": [f"an ordinary sentence number {i}." for i in range(40)]})
        ds = await dataset_row(async_session, raw)
        await async_session.commit()
        response = await client.post(f"{API}/datasets", json=self._body(
            dataset_id=str(ds.id), label_column="text", label_mapping={"*": "negative"}))
        assert response.status_code in (200, 201), response.text
        body = response.json()
        assert body["role"] == "everyday_text" and body["counts"]["negative"] == 40
        assert body["counts"].get("positive", 0) == 0


# ── the D82 migration ───────────────────────────────────────────────────────────────────────


def test_the_migration_restates_cached_rungs_and_round_trips():
    """`e009a1d82e70` on a scratch table with the columns it touches: rung 2 with an in-distribution pass
    -> 1, rung 3 without -> 0, rung 1 untouched; reasons appended; next step stored; a cached definition
    invalidated (cause "rung"); downgrade drops the column."""
    import importlib.util
    import json
    import os

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import text

    path = Path(__file__).resolve().parents[2] / "alembic" / "versions" / "e009a1d82e70_everyday_text_rung_check.py"
    spec = importlib.util.spec_from_file_location("rev_e009a1d82e70", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.revision, module.down_revision) == ("e009a1d82e70", "e008c3d5f7a9")
    assert module.NOT_CHECKED == EVERYDAY_NOT_CHECKED

    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio_test"
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_d82_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    engine = create_engine(server + "/" + name)
    in_passed = json.dumps(["CI lower bound above 0.5 in-distribution on: a", "every out-of-distribution set run (1) cleared 0.5"])
    ood_only = json.dumps(["no in-distribution set cleared 0.5, so rung 1 is not claimed", "every ..."])
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS probe_monitors"))
            conn.execute(text(
                "CREATE TABLE probe_monitors (id varchar(36) primary key, rung integer not null, "
                "rung_reasons jsonb not null, target_fpr double precision, definition_path varchar(1000), "
                "definition_built_at timestamptz, definition_sha256 varchar(64), definition_build jsonb)"))
            conn.execute(text(
                "INSERT INTO probe_monitors VALUES "
                "('a', 2, CAST(:ip AS jsonb), 0.01, '/d/a.json', now(), :sha, "
                "  CAST('{\"resolved\": {\"detector_digest\": \"dd\", \"decision_digest\": \"ee\"}}' AS jsonb)),"
                "('b', 3, CAST(:oo AS jsonb), 0.01, NULL, NULL, NULL, NULL),"
                "('c', 1, CAST(:ip AS jsonb), 0.01, '/d/c.json', now(), :sha, NULL)"),
                {"ip": in_passed, "oo": ood_only, "sha": "c" * 64})
            with Operations.context(MigrationContext.configure(conn)):
                module.upgrade()
            rows = {r.id: r for r in conn.execute(text("SELECT * FROM probe_monitors"))}
        assert (rows["a"].rung, rows["b"].rung, rows["c"].rung) == (1, 0, 1)
        assert rows["a"].rung_reasons[-1] == f"rung 2 not reached: {NOT_CHECKED}"
        assert rows["a"].rung_detail["next_step"] == NOT_CHECKED
        assert rows["a"].rung_detail["everyday_text"]["allowed_rate"] == pytest.approx(0.02)
        assert rows["a"].definition_path is None and rows["a"].definition_sha256 is None
        inv = rows["a"].definition_build["invalidated"]
        assert (inv["cause"], inv["was"], inv["was_sha256"]) == ("rung", "/d/a.json", "c" * 64)
        assert inv["was_identity"]["detector_digest"] == "dd" and "2 -> 1" in inv["reason"]
        assert rows["b"].definition_build is None                      # no definition, nothing to invalidate
        assert rows["c"].rung_detail is None and rows["c"].definition_path == "/d/c.json"   # untouched
        with engine.begin() as conn:
            with Operations.context(MigrationContext.configure(conn)):
                module.downgrade()
            columns = {r[0] for r in conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'probe_monitors'"))}
        assert "rung_detail" not in columns
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS probe_monitors"))
        engine.dispose()
