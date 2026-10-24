"""The GPU re-cut arm keeps calibration positions (Feature 040 S2 FR-20; FTID D-1, TI-10; FTASKS 9.3).

The arm scores only the contract-window scopes and never rewrites the base negatives, so positions are
taken from the pass whose scope equals the run's and written only when their count equals the stored base
negatives. A run scoped `assistant`/`user`/`content` has no such pass: a caller that needs positions is
refused `positions_not_recoverable` BEFORE the model loads. The model and the re-score are stubbed (the
`test_probe_recalibration` harness); the arm, `_calibrate_windows`, `propose` and `apply` run for real.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.services import probe_monitor_run as run_mod
from src.services.probe_recalibration import RecalibrationRefused
from tests.unit.test_probe_decision import make_rows, sync_db  # noqa: F401
from tests.unit.test_probe_recalibration import no_millm  # noqa: F401


def _fresh(db, probe_id):
    from src.models.probe_monitor import ProbeMonitor

    db.expire_all()
    return db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).one()


def _stubs(monkeypatch, rows, positions, loads):
    run_mod.artifact_dir_for(rows.run.id).mkdir(parents=True, exist_ok=True)
    negatives = [float(v) for v in np.linspace(-8.0, 28.0, 2000, dtype=np.float32)]
    lengths = [20 + (i % 300) for i in range(2000)]

    def fake_group(db, prepared, group, scopes, *a, **k):
        return [{scope: run_mod.CalibrationScores(
            negatives=negatives, n_rows=2000, lengths=lengths if scope == "all" else [],
            positions=list(positions) if scope == "all" else [])
            for scope in scopes}]

    def load(run):
        loads.append(run.id)
        return None, None, "llama"

    monkeypatch.setattr(run_mod, "_load_model_for_run", load)
    monkeypatch.setattr(run_mod, "load_probe", lambda db, pid: (
        SimpleNamespace(layer=1), SimpleNamespace(rule="mean", rule_params={})))
    monkeypatch.setattr(run_mod, "_prepare_calibration_rows", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(run_mod, "_score_calibration_group", fake_group)
    monkeypatch.setattr(run_mod, "probe_precision", lambda db, run: {"refusal": None})
    monkeypatch.setattr(run_mod, "probe_render", lambda run, probe: {"refusal": None})


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    from src.core import config as config_module

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def test_the_recut_writes_positions_from_the_base_pass(sync_db, tmp_path, monkeypatch, no_millm):  # noqa: F811
    rows = make_rows(sync_db, tmp_path, windows=True)
    positions = [3 * i + 7 for i in range(2000)]                  # split positions, not 0..n-1
    _stubs(monkeypatch, rows, positions, [])
    run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.02, reason="t")
    probe = _fresh(sync_db, rows.probe.id)
    stored = np.load(probe.calibration_positions_path)
    assert stored.dtype == np.int64 and stored.tolist() == positions
    assert probe.calibration_n_negatives == 2000
    assert probe.calibration_positions_path.endswith("__calibration_positions.npy")
    # 039's lengths, written by the same pass, survive `apply`'s locked re-read too (the flush; was lost)
    assert probe.calibration_lengths_path and np.load(probe.calibration_lengths_path).shape == (2000,)


def test_positions_that_do_not_match_the_stored_negatives_are_not_written(sync_db, tmp_path, monkeypatch, no_millm):  # noqa: F811
    rows = make_rows(sync_db, tmp_path, windows=True)
    _stubs(monkeypatch, rows, list(range(1999)), [])
    run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.02, reason="t")
    assert _fresh(sync_db, rows.probe.id).calibration_positions_path is None


@pytest.mark.parametrize("scope", ["assistant", "user", "content"])
def test_unrecoverable_positions_are_refused_before_the_model_loads(sync_db, tmp_path, monkeypatch, no_millm, scope):  # noqa: F811
    rows = make_rows(sync_db, tmp_path, windows=True)
    rows.run.config = {**rows.run.config, "scope": scope}
    sync_db.commit()
    loads = []
    _stubs(monkeypatch, rows, list(range(2000)), loads)
    with pytest.raises(RecalibrationRefused) as caught:
        run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.02, reason="t",
                                           require_positions=True)
    assert caught.value.code == "positions_not_recoverable"
    assert loads == [], "the model was loaded before a refusal that needed no GPU"


def test_the_backup_glob_covers_the_positions_file(tmp_path):
    """The arm's array backup/restore (`{probe}__calibration_*.npy`) holds the positions file too."""
    (tmp_path / "pm_x__calibration_positions.npy").write_bytes(b"old")
    backup = run_mod._backup_calibration_arrays(tmp_path, "pm_x")
    (tmp_path / "pm_x__calibration_positions.npy").write_bytes(b"new")
    run_mod._restore_calibration_arrays(backup, tmp_path, "pm_x")
    assert (tmp_path / "pm_x__calibration_positions.npy").read_bytes() == b"old"
