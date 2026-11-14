"""The ONE assembly of the decision block and its stored snapshot (039 FR-8; TI5/X4; D-T4; B2).

What must hold:
* `decision_from_values` is the one assembly — the builder's helpers are THE SAME objects now;
* every bar writer stores the contract decision and its digest with the bar: `apply` (the re-cut)
  and `recut_probe_windows_on_gpu` (through `apply`) here; the run's calibrating stage in
  `test_probe_monitor_pipeline.py::TestTheDefinitionCarriesItsIdentity`;
* the calibration reference is the snapshot's FROZEN one while the source is unchanged, and is
  re-derived when the source changes (a NULL source becoming `calibration_set`, X4);
* a build copies the snapshot and refuses `decision_snapshot_stale` when the row's bar no longer
  forms it; a not-formable bar is recorded with its reason and re-tried, never guessed;
* the lazy formation of a pre-039 probe's snapshot is a compare-and-set.

Real rows in `mistudio_test` (a sync session over the shared async engine), behaviour keys random.
"""
from __future__ import annotations

import secrets
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.schemas.probe_definition import DatasetRef
from src.schemas.probe_identity import decision_digest_of
from src.services import probe_decision as P


@pytest.fixture
def sync_db(async_engine):
    url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", "")
    )
    engine = create_engine(url)
    session = sessionmaker(bind=engine, autoflush=False)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


@pytest.fixture
def second_session(async_engine):
    url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", "")
    )
    engine = create_engine(url)
    session = sessionmaker(bind=engine, autoflush=False)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def _negatives(tmp_path: Path, name: str, n: int = 2000) -> str:
    path = tmp_path / name
    values = np.linspace(-10.0, 30.0, n, dtype=np.float32)
    np.save(path, values)
    return str(path)


def make_rows(db, tmp_path: Path, *, threshold_source="calibration_set", hf_repo_id="org/cal-set",
              windows=False):
    """A model, a calibration dataset + view, a run, and one probe with negatives on disk whose bar
    was cut by `calibrate` at 1% — what the run's own stage would have written."""
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun
    from src.services.probe_monitor_trainer import calibrate

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/nonexistent", architecture="LlamaForCausalLM", params_count=1000)
    dataset = Dataset(id=uuid.uuid4(), name="cal", source="local", status=DatasetStatus.READY,
                      raw_path="/nonexistent", extra_metadata={}, hf_repo_id=hf_repo_id)
    db.add_all([model, dataset])
    db.flush()
    train = ProbeMonitorDataset(name="train", dataset_id=dataset.id, input_column="i",
                                label_column="l", label_mapping={}, role="train", counts={})
    cal = ProbeMonitorDataset(name="cal", dataset_id=dataset.id, input_column="i", label_column="l",
                              label_mapping={}, role="calibration", counts={},
                              config="default", split="test")
    db.add_all([train, cal])
    db.flush()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=train.id, eval_dataset_ids=[],
                          calibration_dataset_id=cal.id,
                          config={"layers": [1], "rules": ["mean"], "scope": "all",
                                  "target_fpr": 0.01},
                          behaviour="b-" + secrets.token_hex(4), status="completed",
                          environment={})
    db.add(run)
    db.flush()
    scores_path = _negatives(tmp_path, f"neg_{uuid.uuid4().hex[:6]}.npy")
    values = [float(v) for v in np.load(scores_path)]
    bar = calibrate(values, target_fpr=0.01, source=threshold_source or "calibration_set")
    window_decisions = None
    if windows:
        window_decisions = {
            name: {"threshold": bar.threshold, "target_fpr": 0.01, "realised_fpr": bar.realised_fpr,
                   "n_negatives": len(values), "scope": scope,
                   "scores_path": _negatives(tmp_path, f"w_{name}_{uuid.uuid4().hex[:6]}.npy"),
                   "length_bands": None}
            for name, scope in (("prompt", "user"), ("response", "assistant"))
        }
    probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, variant="dense",
                         val_metrics={}, selected=True, streamable=True, rung=2, rung_reasons=[],
                         calibration_scores_path=scores_path, threshold=bar.threshold,
                         target_fpr=0.01, realised_fpr=bar.realised_fpr,
                         threshold_source=threshold_source, window_decisions=window_decisions,
                         behaviour=run.behaviour)
    db.add(probe)
    db.commit()
    return SimpleNamespace(model=model, dataset=dataset, cal=cal, run=run, probe=probe)


def _expected(db, probe):
    """What the snapshot of the row's CURRENT bar must be, recomputed independently of the writer."""
    ref = (
        DatasetRef(hf_id=probe.decision_snapshot["decision"]["calibration"]["hf_id"],
                   config="default", split="test", revision=None)
        if probe.threshold_source == "calibration_set" else None
    )
    decision = P.decision_for_row(probe, ref)
    return decision, decision_digest_of(decision)


# ── the one assembly ──────────────────────────────────────────────────────────


class TestTheOneAssembly:
    def test_the_builder_helpers_ARE_these(self):
        """One home for "drop entries without a threshold" and "whole table or none"."""
        from src.services import probe_definition_builder as builder

        assert builder.bands_or_none is P.bands_or_none
        stub = SimpleNamespace(window_decisions={"prompt": {"threshold": 1.0}})
        assert builder._window_decisions(stub) == P.window_decisions_or_none(stub.window_decisions)

    def test_pure_assembly_over_a_row_stub(self):
        row = SimpleNamespace(
            threshold=2.5, target_fpr=0.01, realised_fpr=0.009, threshold_source="calibration_set",
            window_decisions={
                "prompt": {"threshold": 1.5, "target_fpr": 0.01, "scores_path": "/data/x.npy",
                           "lengths_path": "/data/y.npy", "length_bands": [
                               {"min_tokens": 0, "max_tokens": 9, "threshold": 1.0},
                               {"min_tokens": 10, "max_tokens": None, "threshold": None}]},
                "response": {"threshold": None},
            },
            length_bands=[{"min_tokens": 0, "max_tokens": None, "threshold": 2.0}],
        )
        ref = DatasetRef(hf_id="org/set")
        decision = P.decision_for_row(row, ref)
        dumped = decision.model_dump(mode="json")
        assert dumped["calibration"]["hf_id"] == "org/set"
        assert set(dumped["windows"]) == {"prompt"}, "a window with no bar must be dropped"
        assert "scores_path" not in dumped["windows"]["prompt"], "bookkeeping reached the contract"
        assert dumped["windows"]["prompt"]["length_bands"] is None, "a torn band table was kept"
        assert dumped["length_bands"][0]["threshold"] == 2.0


# ── TI5 / X4: the calibration reference ───────────────────────────────────────


class TestTheCalibrationReference:
    def test_threshold_source_change_re_derives_calibration(self, sync_db, tmp_path):
        """X4: a NULL source becomes `calibration_set` on re-cut, and the snapshot's `null` reference
        must not be copied across that change."""
        rows = make_rows(sync_db, tmp_path, threshold_source=None)
        probe = rows.probe
        P.write_decision_snapshot(probe, None)          # formed under the NULL source
        sync_db.commit()
        assert probe.decision_snapshot["decision"]["calibration"] is None
        same = P.calibration_for_bar(sync_db, probe, None)
        assert same is None
        changed = P.calibration_for_bar(sync_db, probe, "calibration_set")
        assert isinstance(changed, DatasetRef) and changed.hf_id == "org/cal-set", (
            "a calibration-set bar would be published naming no calibration corpus"
        )

    def test_the_frozen_reference_survives_an_edited_dataset_row(self, sync_db, tmp_path):
        """While the source is unchanged the snapshot's reference is used, so a dataset row edited
        after the cut cannot move the digest of a bar nobody re-cut."""
        rows = make_rows(sync_db, tmp_path)
        P.write_decision_snapshot(rows.probe, P.calibration_from_run(sync_db, rows.probe,
                                                                     "calibration_set"))
        sync_db.commit()
        rows.dataset.hf_repo_id = "org/renamed"
        sync_db.commit()
        ref = P.calibration_for_bar(sync_db, rows.probe, "calibration_set")
        assert ref.hf_id == "org/cal-set"

    def test_a_local_only_calibration_set_is_not_formable_with_its_reason(self, sync_db, tmp_path):
        rows = make_rows(sync_db, tmp_path, hf_repo_id=None)
        snapshot = P.write_decision_snapshot(
            rows.probe, P.calibration_from_run(sync_db, rows.probe, "calibration_set")
        )
        assert snapshot["decision"] is None and snapshot["digest"] is None
        assert snapshot["stale_reason"] == "decision_not_formable"
        assert "HuggingFace identifier" in snapshot["detail"]
        assert rows.probe.decision_digest is None


# ── FR-8: every bar writer stores the snapshot ────────────────────────────────


class TestEveryBarWriterWritesTheSnapshot:
    def test_every_bar_writer_writes_the_snapshot_apply(self, sync_db, tmp_path):
        from src.services.probe_recalibration import recalibrate_probe

        rows = make_rows(sync_db, tmp_path)
        out = recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.02, commit=True)
        assert out["committed"] is True
        sync_db.expire_all()
        probe = rows.probe
        decision, digest = _expected(sync_db, probe)
        assert probe.threshold == out["proposed"]["threshold"]
        assert probe.decision_snapshot["decision"] == decision.model_dump(mode="json")
        assert probe.decision_digest == digest == probe.decision_snapshot["digest"]
        assert probe.decision_snapshot["decision"]["target_fpr"] == 0.02

    def test_every_bar_writer_writes_the_snapshot_gpu_recut(self, sync_db, tmp_path, monkeypatch):
        """The GPU window re-cut commits through `apply`, with the re-score stubbed."""
        from src.core import config as config_module
        from src.services import probe_monitor_run as run_mod

        monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
        rows = make_rows(sync_db, tmp_path, windows=True)
        probe = rows.probe
        # The run's artifact directory exists for any run that trained (it holds the weights).
        run_mod.artifact_dir_for(rows.run.id).mkdir(parents=True)
        negatives = [float(v) for v in np.linspace(-8.0, 28.0, 2000, dtype=np.float32)]
        lengths = [20 + (i % 300) for i in range(2000)]

        def fake_group(db, prepared, group, scopes, *a, **k):
            return [{scope: run_mod.CalibrationScores(negatives=negatives, n_rows=2000,
                                                      lengths=lengths if scope == "all" else [])
                     for scope in scopes}]

        monkeypatch.setattr(run_mod, "_load_model_for_run", lambda run: (None, None, "llama"))
        monkeypatch.setattr(run_mod, "load_probe", lambda db, pid: (
            SimpleNamespace(layer=1), SimpleNamespace(rule="mean", rule_params={})))
        monkeypatch.setattr(run_mod, "_prepare_calibration_rows", lambda *a, **k: ([], []))
        monkeypatch.setattr(run_mod, "_score_calibration_group", fake_group)
        monkeypatch.setattr(run_mod, "probe_precision", lambda db, run: {"refusal": None})
        monkeypatch.setattr(run_mod, "probe_render", lambda run, probe: {"refusal": None})

        out = run_mod.recut_probe_windows_on_gpu(sync_db, probe.id, target_fpr=0.02, reason="t")
        assert out["committed"] is True
        sync_db.expire_all()
        decision, digest = _expected(sync_db, probe)
        assert probe.window_decisions and all(
            entry["target_fpr"] == 0.02 for entry in probe.window_decisions.values()
        )
        assert probe.decision_snapshot["decision"] == decision.model_dump(mode="json")
        assert probe.decision_digest == digest

    def test_the_run_stage_names_the_calibration_set_it_cut_from(
        self, sync_db, tmp_path, monkeypatch
    ):
        """Review R1: the run's calibrating stage on the CALIBRATION-SET path stores a snapshot that
        names the run's calibration corpus. The pipeline test runs the validation-negatives path
        (calibration `None` by construction), so a stage that passed no calibration survived it —
        and the build's stale check re-forms with the snapshot's own frozen reference, so it would
        then publish a `calibration_set` bar naming no corpus, silently."""
        from src.core import config as config_module
        from src.services import probe_monitor_run as run_mod

        monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
        rows = make_rows(sync_db, tmp_path, threshold_source=None)
        probe = rows.probe
        run_mod.artifact_dir_for(rows.run.id).mkdir(parents=True)
        negatives = [float(v) for v in np.linspace(-8.0, 28.0, 2000, dtype=np.float32)]
        lengths = [20 + (i % 300) for i in range(2000)]

        def fake_group(db, prepared, group, scopes, *a, **k):
            return [
                {scope: run_mod.CalibrationScores(negatives=negatives, n_rows=2000,
                                                  lengths=lengths if scope == "all" else [])
                 for scope in scopes}
                for _ in group
            ]

        monkeypatch.setattr(run_mod, "_prepare_calibration_rows", lambda *a, **k: ([], []))
        monkeypatch.setattr(run_mod, "_score_calibration_group", fake_group)
        run_mod._calibrate_probes(
            sync_db,
            [(probe, SimpleNamespace(layer=1), SimpleNamespace(rule="mean", rule_params={}), [], [])],
            rows.cal.id, run_mod.context_from_row(rows.run), None, None, "llama",
        )
        sync_db.commit()
        sync_db.expire_all()
        assert probe.threshold_source == "calibration_set"
        stored = probe.decision_snapshot["decision"]
        assert stored is not None and stored["calibration"] is not None, (
            "a calibration-set bar was stored naming no calibration corpus"
        )
        assert (stored["calibration"]["hf_id"], stored["calibration"]["config"],
                stored["calibration"]["split"]) == ("org/cal-set", "default", "test")
        expected = P.decision_for_row(probe, DatasetRef(**stored["calibration"]))
        assert probe.decision_digest == decision_digest_of(expected)


# ── D-T4: the build copies the snapshot or refuses ────────────────────────────


class TestTheBuildCopiesTheSnapshot:
    def test_build_refuses_a_stale_snapshot(self, sync_db, tmp_path):
        from src.services.probe_definition_builder import ProbeExportRefused

        rows = make_rows(sync_db, tmp_path)
        P.ensure_decision_snapshot(sync_db, rows.probe)
        assert P.decision_for_build(sync_db, rows.probe).threshold == rows.probe.threshold
        rows.probe.threshold = rows.probe.threshold + 0.5      # a bar writer that forgot it
        sync_db.commit()
        with pytest.raises(ProbeExportRefused) as refused:
            P.decision_for_build(sync_db, rows.probe)
        assert refused.value.status == 409
        assert str(refused.value).startswith("decision_snapshot_stale: fields threshold")

    def test_a_window_written_without_its_snapshot_is_refused_too(self, sync_db, tmp_path):
        from src.services.probe_definition_builder import ProbeExportRefused

        rows = make_rows(sync_db, tmp_path, windows=True)
        P.ensure_decision_snapshot(sync_db, rows.probe)
        changed = dict(rows.probe.window_decisions)
        changed["prompt"] = {**changed["prompt"], "threshold": changed["prompt"]["threshold"] + 1}
        rows.probe.window_decisions = changed
        sync_db.commit()
        with pytest.raises(ProbeExportRefused) as refused:
            P.decision_for_build(sync_db, rows.probe)
        assert "windows" in str(refused.value)

    def test_recut_then_build_agree(self, sync_db, tmp_path):
        """The block a re-cut stores is the block the build publishes."""
        from src.services.probe_recalibration import recalibrate_probe

        rows = make_rows(sync_db, tmp_path)
        recalibrate_probe(sync_db, rows.probe.id, target_fpr=0.005, commit=True)
        decision = P.decision_for_build(sync_db, rows.probe)
        assert decision_digest_of(decision) == rows.probe.decision_digest

    @pytest.mark.parametrize("window_only", [False, True], ids=["global", "window-only"])
    def test_recut_then_build_digest_agrees(self, sync_db, tmp_path, monkeypatch, window_only):
        """FR-8 end to end (4.2): the digest the PREVIEW promises is the digest the commit stores and
        the digest the build publishes — for a global re-cut and for one that moves only a window."""
        from src.services import millm_read
        from src.services.probe_recalibration import recalibrate_probe

        monkeypatch.setattr(millm_read, "imported_copies",
                            lambda pid: millm_read.NotConfigured("not connected"))
        rows = make_rows(sync_db, tmp_path, windows=window_only)
        target = 0.02
        if window_only:
            # The windows' bars sit away from what their arrays give at 1%, the global bar on it:
            # a re-cut at 1% then moves ONLY the windows.
            rows.probe.window_decisions = {
                name: {**entry, "threshold": entry["threshold"] + 3.0}
                for name, entry in rows.probe.window_decisions.items()
            }
            sync_db.commit()
            target = 0.01
        preview = recalibrate_probe(sync_db, rows.probe.id, target_fpr=target, commit=False)
        if window_only:
            assert preview["proposed"]["threshold"] == rows.probe.threshold
        committed = recalibrate_probe(sync_db, rows.probe.id, target_fpr=target, commit=True)
        assert committed["committed"] is True
        built = decision_digest_of(P.decision_for_build(sync_db, rows.probe))
        assert preview["proposed"]["decision_digest"] == committed["current"]["decision_digest"] == (
            built
        ) == rows.probe.decision_digest

    def test_a_not_formable_bar_is_refused_with_its_reason_and_retried(self, sync_db, tmp_path):
        from src.services.probe_definition_builder import ProbeExportRefused

        rows = make_rows(sync_db, tmp_path, hf_repo_id=None)
        P.ensure_decision_snapshot(sync_db, rows.probe)
        assert rows.probe.decision_snapshot["decision"] is None
        with pytest.raises(ProbeExportRefused) as refused:
            P.decision_for_build(sync_db, rows.probe)
        assert "HuggingFace identifier" in str(refused.value) and refused.value.status == 422
        # The dataset row gains its HuggingFace id: the build must now succeed, as it always did.
        rows.dataset.hf_repo_id = "org/cal-set"
        sync_db.commit()
        decision = P.decision_for_build(sync_db, rows.probe)
        assert decision.calibration.hf_id == "org/cal-set"
        assert rows.probe.decision_digest == decision_digest_of(decision)

    def test_the_retry_never_overwrites_a_recut_committed_meanwhile(
        self, sync_db, second_session, tmp_path, monkeypatch
    ):
        """Review R1: the not-formable retry is a compare-and-set. A re-cut that commits a new bar
        and its snapshot while the build re-forms the OLD bar must keep its snapshot, and the build
        must publish the re-cut's decision — not overwrite it and leave every later build refusing
        `decision_snapshot_stale`."""
        from src.services.probe_recalibration import recalibrate_probe

        rows = make_rows(sync_db, tmp_path, hf_repo_id=None)
        P.ensure_decision_snapshot(sync_db, rows.probe)
        assert rows.probe.decision_snapshot["decision"] is None
        rows.dataset.hf_repo_id = "org/cal-set"          # now formable
        sync_db.commit()
        old_threshold = rows.probe.threshold

        real = P.calibration_from_run
        fired = []

        def recut_meanwhile(db, probe, threshold_source):
            if not fired:
                fired.append(True)
                recalibrate_probe(second_session, probe.id, target_fpr=0.02, commit=True)
            return real(db, probe, threshold_source)

        monkeypatch.setattr(P, "calibration_from_run", recut_meanwhile)
        decision = P.decision_for_build(sync_db, rows.probe)
        assert fired, "the interleaving never happened, so this proves nothing"
        sync_db.expire_all()
        probe = rows.probe
        assert probe.threshold != old_threshold
        assert probe.decision_snapshot["decision"]["threshold"] == probe.threshold, (
            "the retry overwrote the re-cut's snapshot with one formed from the old bar"
        )
        assert decision.threshold == probe.threshold
        assert decision_digest_of(decision) == probe.decision_digest


# ── lazy formation is a compare-and-set ───────────────────────────────────────


class TestLazyFormation:
    def test_ensure_forms_a_missing_snapshot_once(self, sync_db, tmp_path):
        rows = make_rows(sync_db, tmp_path)
        assert rows.probe.decision_snapshot is None
        P.ensure_decision_snapshot(sync_db, rows.probe)
        first = rows.probe.decision_snapshot
        assert first["digest"] == rows.probe.decision_digest and first["decision"] is not None
        P.ensure_decision_snapshot(sync_db, rows.probe)
        assert rows.probe.decision_snapshot == first

    def test_it_never_overwrites_a_snapshot_written_concurrently(
        self, sync_db, second_session, tmp_path
    ):
        from src.models.probe_monitor import ProbeMonitor

        rows = make_rows(sync_db, tmp_path)
        assert rows.probe.decision_snapshot is None          # read before the other writer
        other = second_session.query(ProbeMonitor).filter(ProbeMonitor.id == rows.probe.id).one()
        written = P.write_decision_snapshot(other, None)
        written_digest = other.decision_digest
        second_session.commit()
        P.ensure_decision_snapshot(sync_db, rows.probe)       # holds the stale "None" in memory
        assert rows.probe.decision_digest == written_digest
        assert rows.probe.decision_snapshot["formed_at"] == written["formed_at"]


def test_decision_snapshot_stale_is_logged(sync_db, tmp_path, caplog):
    """8.4: the build guard's refusal leaves one structured line with the probe id and both digests."""
    import logging

    from src.services.probe_definition_builder import ProbeExportRefused

    rows = make_rows(sync_db, tmp_path)
    P.ensure_decision_snapshot(sync_db, rows.probe)
    stored = rows.probe.decision_digest
    rows.probe.threshold = rows.probe.threshold + 0.5
    sync_db.commit()
    with caplog.at_level(logging.INFO, logger="src.services.probe_decision"):
        with pytest.raises(ProbeExportRefused):
            P.decision_for_build(sync_db, rows.probe)
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("decision_snapshot_stale")]
    assert len(lines) == 1
    assert f"probe={rows.probe.id}" in lines[0] and f"stored_digest={stored}" in lines[0]
    assert "fields=['threshold']" in lines[0]
