"""Calibration negatives keep their split positions (Feature 040 S2 FR-20; FTDD §4.6; FTASKS 9.1 – 9.2, 9.4).

THE SUBSAMPLE TRAP, as a junction test: a 10,000-row calibration split is sampled at 500 rows; some rows
are excluded by the label mapping and some fail to render; the real `_prepare_calibration_rows`,
`_score_calibration_group` and `_calibrate_probes` run in sequence, with only the render and the model's
forward faked. The fake forward scores each row AS ITS OWN SPLIT ROW NUMBER, so a stored negative equals
the split position it came from — and the positions file must say exactly that position, index for
index. A position taken from the subsample (0..499) or from the list (0..n-1) cannot pass.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import settings
from src.models.dataset import Dataset
from src.models.probe_monitor import ProbeMonitorDataset
from src.services import probe_monitor_capture, probe_monitor_render, probe_monitor_run as run_mod

N_SPLIT = 10_000


def _label(i):
    return "skip" if i % 17 == 3 else "neg"


@pytest.fixture
def world(tmp_path, monkeypatch):
    from datasets import Dataset as HFDataset

    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    raw = tmp_path / "datasets" / "cal"
    HFDataset.from_dict({
        "text": [f"row {i}" + (" BROKEN" if i % 23 == 4 else "") for i in range(N_SPLIT)],
        "kind": [_label(i) for i in range(N_SPLIT)],
    }).save_to_disk(str(raw))

    def fake_render_all(tokenizer, conversations, **kwargs):
        out = []
        for conversation in conversations:
            text = " ".join(m["content"] for m in conversation)
            out.append(None if "BROKEN" in text else SimpleNamespace(text=text))
        return out, SimpleNamespace(as_dict=lambda: {})

    def fake_by_scope(model, rendered, specs, *, scopes, **kwargs):
        def row(r):
            number = int(r.text.split()[1])
            # n_scored 0 on every 29th row: not a negative of the window (filtered with the scores)
            return SimpleNamespace(aggregate=float(number), n_scored=0 if number % 29 == 0 else 5 + number % 7)
        return {scope: [[row(r) for r in rendered] for _ in specs] for scope in scopes}

    monkeypatch.setattr(probe_monitor_render, "render_all", fake_render_all)
    monkeypatch.setattr(probe_monitor_capture, "forward_scores_by_scope", fake_by_scope)
    dataset = SimpleNamespace(id="ds-cal", raw_path=str(raw))
    view = SimpleNamespace(id="pmd_cal", dataset_id="ds-cal", input_column="text", label_column="kind",
                           pair_column=None, label_mapping={"neg": "negative", "skip": "excluded"},
                           keyword_filter=None, split=None, name="cal")
    directory = tmp_path / "probe_monitors" / "pmr_x"
    directory.mkdir(parents=True)

    class DB:
        def query(self, model):
            target = {Dataset: dataset, ProbeMonitorDataset: view}[model]
            return SimpleNamespace(filter=lambda *a, **k: SimpleNamespace(first=lambda: target))

    return SimpleNamespace(db=DB(), dir=directory)


def context(world, config, scope="all"):
    return SimpleNamespace(artifact_dir=world.dir, scope=scope, max_length=64, target_fpr=0.01, seed=0,
                           config=config)


def test_prepared_positions_are_split_rows_of_the_sample(world):
    ctx = context(world, {"calibration_max_rows": 500})
    rendered, labels, positions = run_mod._prepare_calibration_rows(world.db, "pmd_cal", ctx, None)
    sample = run_mod.subsample_indices(N_SPLIT, 500, 0)
    assert set(positions) <= set(sample) and len(positions) < 500           # excluded / broken rows dropped
    assert [int(r.text.split()[1]) for r in rendered] == positions          # each rendered row IS that split row
    assert positions != list(range(len(positions)))


@pytest.mark.parametrize("scope", ["all", "assistant"])
def test_the_calibrating_stage_stores_positions_aligned_with_the_negatives(world, monkeypatch, scope):
    """`all` is also re-written by the window pass whose scope is the run's; `assistant` matches no
    contract window, so there the base write is the ONLY one (S2 review: a `range(n)` mutation of the
    base write survived the `all` case alone)."""
    from src.services import probe_decision

    monkeypatch.setattr(probe_decision, "calibration_or_not_formable", lambda db, cid, source: None)
    monkeypatch.setattr(probe_decision, "write_decision_snapshot", lambda probe, calibration: {})
    ctx = context(world, {"calibration_max_rows": 500}, scope)
    probe = SimpleNamespace(id="pm_cal", variant="dense", sae_id=None, sae_feature_indices=None)
    head = SimpleNamespace(layer=1)
    trained = SimpleNamespace(rule="mean", rule_params={})
    run_mod._calibrate_probes(world.db, [(probe, head, trained, [], [])], "pmd_cal", ctx, None,
                              SimpleNamespace(pad_token_id=0), "llama")
    negatives = np.load(probe.calibration_scores_path)
    positions = np.load(probe.calibration_positions_path)
    assert positions.dtype == np.int64
    assert len(positions) == len(negatives) == probe.calibration_n_negatives
    assert positions.tolist() == [int(v) for v in negatives.tolist()], "a negative is not the row its position names"
    assert all(p % 29 != 0 for p in positions.tolist())                   # n_scored 0 rows filtered with the scores
    assert positions.tolist() != list(range(len(positions)))


def test_positions_beyond_two_to_the_24_round_trip_exactly(tmp_path):
    big = [2 ** 24 + 1, 2 ** 30 + 3, 2 ** 40 + 7]
    path = run_mod.persist_calibration_negatives(tmp_path, "pm_x", big, name="positions", dtype=np.int64)
    assert np.load(path).tolist() == big
    lossy = run_mod.persist_calibration_negatives(tmp_path, "pm_y", big, name="positions")
    assert np.load(lossy).tolist() != big, "float32 would have been exact — the fixture proves nothing"


def test_a_run_without_the_key_samples_the_legacy_2000(world, monkeypatch):
    """D-2: a config from before `calibration_max_rows` (no key) re-cuts on the 2,000 rows it was cut on."""
    seen = []
    real = run_mod._prepare_examples

    def spy(db, view, context, tokenizer, *, role, max_rows=None):
        seen.append(max_rows)
        return real(db, view, context, tokenizer, role=role, max_rows=max_rows)

    monkeypatch.setattr(run_mod, "_prepare_examples", spy)
    run_mod._prepare_calibration_rows(world.db, "pmd_cal", context(world, {}), None)
    run_mod._prepare_calibration_rows(world.db, "pmd_cal", context(world, {"calibration_max_rows": 700}), None)
    assert seen == [run_mod.LEGACY_CALIBRATION_MAX_ROWS, 700] and run_mod.LEGACY_CALIBRATION_MAX_ROWS == 2000


def test_the_default_sample_is_5000_and_still_affords_its_target():
    from src.schemas.probe_monitor import DEFAULT_CALIBRATION_MAX_ROWS, ProbeRunConfig

    assert DEFAULT_CALIBRATION_MAX_ROWS == 5000
    assert ProbeRunConfig().model_dump()["calibration_max_rows"] == 5000
    with pytest.raises(ValueError, match="cannot afford"):
        ProbeRunConfig(target_fpr=0.0001, calibration_max_rows=5000)
