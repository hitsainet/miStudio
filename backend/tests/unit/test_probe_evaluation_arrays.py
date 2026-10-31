"""One evaluation, four aligned files, named by the evaluation's id (035 FR-11, FR-12; D14, D16, D44, D45).

`_evaluate_probes_on_sets` runs for real, and so does `_prepare_examples` — over a real Arrow
dataset on disk, with a label mapping that EXCLUDES rows, a keyword filter that DROPS rows and a
render that FAILS for some rows. Those three are exactly how a kept row's position stops equalling
its index in the evaluated list, so positions that "happen to be 0..n-1" cannot pass. Only the model
forward (`forward_scores_many`) and the renderer are faked; the fake renderer fails rows on a marker.

MUTATION CONTROLS (recorded in 0xcc/reviews/035_implementation_controls_2026-10-18.md):
  C14 the file name reverts to `{probe}__{dataset}.npy`   → the two-evaluations test fails
  C15 the positions save is skipped                       → the positions test fails
  C16 the unlink-on-failure is removed                    → the commit-failure test fails
  C23 positions written as list order (range(n))          → the positions test fails
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import settings
from src.models.dataset import Dataset
from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation
from src.services import probe_monitor_capture, probe_monitor_render, probe_monitor_run
from src.services.probe_monitor_metrics import auroc

N_ROWS = 160


def _row_text(i):
    # every 7th row lacks the keyword (filtered out); every 11th renders BROKEN (dropped).
    words = ["pos" if i % 2 == 0 else "neg"]
    if i % 7 != 3:
        words.append("keep")
    if i % 11 == 5:
        words.append("BROKEN")
    return " ".join(words)


def _label(i):
    if i % 13 == 6:
        return "skip"           # mapped to `excluded`
    return "high" if i % 2 == 0 else "low"


def expected_positions():
    kept = []
    for i in range(N_ROWS):
        if _label(i) == "skip" or i % 7 == 3 or i % 11 == 5:
            continue
        kept.append(i)
    return kept


@pytest.fixture
def world(tmp_path, monkeypatch):
    from datasets import Dataset as HFDataset

    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    raw = tmp_path / "datasets" / "org_ds"
    HFDataset.from_dict({
        "text": [_row_text(i) for i in range(N_ROWS)],
        "stakes": [_label(i) for i in range(N_ROWS)],
    }).save_to_disk(str(raw))

    def fake_render_all(tokenizer, conversations, **kwargs):
        rendered = []
        for conversation in conversations:
            text = " ".join(m["content"] for m in conversation)
            rendered.append(None if "BROKEN" in text else SimpleNamespace(text=text))
        return rendered, SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr(probe_monitor_render, "render_all", fake_render_all)
    state = {"bias": 0.0, "forwards": 0}

    def fake_forward_scores_many(model, rendered, specs, **kwargs):
        state["forwards"] += 1
        rng = np.random.default_rng(int(state["bias"] * 100) + 1)
        out = []
        for _spec in specs:
            out.append([
                SimpleNamespace(
                    aggregate=(1.0 if "pos" in r.text else 0.0) + float(rng.normal()) * (1.0 + state["bias"]),
                    n_scored=len(r.text) + 3,
                )
                for r in rendered
            ])
        return out

    monkeypatch.setattr(probe_monitor_capture, "forward_scores_many", fake_forward_scores_many)

    dataset = SimpleNamespace(id="ds-1", raw_path=str(raw))
    view = SimpleNamespace(
        id="pmd_eval", dataset_id="ds-1", input_column="text", label_column="stakes", pair_column=None,
        label_mapping={"high": "positive", "low": "negative", "skip": "excluded"},
        keyword_filter={"terms": ["keep"], "mode": "any"}, split=None, name="eval",
        distribution="in_distribution", role="eval",
    )
    artifact_dir = tmp_path / "probe_monitors" / "pmr_x"
    artifact_dir.mkdir(parents=True)
    return {"dataset": dataset, "view": view, "dir": artifact_dir, "state": state}


class FakeDB:
    def __init__(self, world, fail_commit=False):
        self.world, self.added, self.fail_commit, self.commits = world, [], fail_commit, 0

    def query(self, model):
        target = {Dataset: self.world["dataset"], ProbeMonitorDataset: self.world["view"]}[model]
        return SimpleNamespace(filter=lambda *a, **k: SimpleNamespace(first=lambda: target))

    def add(self, row):
        self.added.append(row)

    def commit(self):
        if self.fail_commit:
            raise RuntimeError("the database went away")
        self.commits += 1


def _evaluate(world, db):
    probe = SimpleNamespace(id="pm_abc", target_fpr=None, variant="dense", sae_id=None, threshold=None)
    trained = SimpleNamespace(rule="mean", rule_params={})
    context = SimpleNamespace(artifact_dir=world["dir"], scope="all", max_length=64,
                              target_fpr=0.1, seed=0)
    return probe_monitor_run._evaluate_probes_on_sets(
        db, [(probe, object(), trained, None, None)], context, None,
        SimpleNamespace(pad_token_id=0), "LlamaForCausalLM", ["pmd_eval"],
    )


def test_positions_are_the_KEPT_rows_source_positions_and_all_four_align(world):
    db = FakeDB(world)
    assert _evaluate(world, db) == 1
    (row,) = db.added
    positions = np.load(row.positions_path)
    want = expected_positions()
    assert positions.dtype == np.int64
    assert positions.tolist() == want, "positions are not the kept rows' source positions"
    assert positions.tolist() != list(range(len(want))), "fixture too weak: no gaps"
    scores, n_scored, labels = (np.load(row.scores_path), np.load(row.n_scored_path), np.load(row.labels_path))
    assert len(scores) == len(n_scored) == len(labels) == len(want)
    assert (scores.dtype, n_scored.dtype, labels.dtype) == (np.float32, np.int32, np.int8)
    assert labels.tolist() == [1 if i % 2 == 0 else 0 for i in want]
    # n_scored is the capture's count for that row (the fake: len(text) + 3).
    assert n_scored.tolist() == [len(_row_text(i)) + 3 for i in want]
    assert db.commits == 1


def test_every_path_is_its_own_column_named_by_the_evaluation_id(world):
    db = FakeDB(world)
    _evaluate(world, db)
    (row,) = db.added
    assert row.id.startswith("pme_")
    stem = f"pm_abc__pmd_eval__{row.id}"
    assert row.scores_path.endswith(f"{stem}.npy")
    assert row.positions_path.endswith(f"{stem}__positions.npy")
    assert row.n_scored_path.endswith(f"{stem}__n_scored.npy")
    assert row.labels_path.endswith(f"{stem}__labels.npy")
    assert "__calibration_" not in row.scores_path


def test_two_evaluations_keep_two_files_each_reproducing_ITS_auroc(world):
    """C14. The pre-035 name `{probe}__{dataset}.npy` made the second overwrite the first."""
    db = FakeDB(world)
    _evaluate(world, db)
    world["state"]["bias"] = 3.0      # a different, noisier second evaluation
    _evaluate(world, db)
    first, second = db.added
    assert first.scores_path != second.scores_path
    for row in (first, second):
        recomputed = auroc(np.load(row.scores_path).astype(float).tolist(), np.load(row.labels_path).tolist())
        assert abs(recomputed - row.metrics["auroc"]) < 5e-4, (row.id, recomputed, row.metrics["auroc"])
    assert abs(first.metrics["auroc"] - second.metrics["auroc"]) > 1e-3, "fixture too weak"


def test_a_failed_commit_leaves_no_orphan_files(world):
    """C16 / D45: files are written before the row lands; if it never lands they go."""
    db = FakeDB(world, fail_commit=True)
    with pytest.raises(RuntimeError, match="went away"):
        _evaluate(world, db)
    assert list(world["dir"].iterdir()) == []


def test_misaligned_arrays_are_refused_before_anything_is_written(world, monkeypatch):
    """TI12: a capture returning fewer results than examples must not produce a servable lie."""
    real = probe_monitor_capture.forward_scores_many

    def short(*args, **kwargs):
        return [results[:-1] for results in real(*args, **kwargs)]

    monkeypatch.setattr(probe_monitor_capture, "forward_scores_many", short)
    db = FakeDB(world)
    # `evaluate()` refuses scores≠labels first; the writer's own guard (TI12) backs it for positions.
    with pytest.raises(ValueError, match="not aligned|scores against"):
        _evaluate(world, db)
    assert list(world["dir"].iterdir()) == [] and db.commits == 0


def _evaluate_at(world, db, threshold):
    probe = SimpleNamespace(id="pm_abc", target_fpr=None, variant="dense", sae_id=None, threshold=threshold)
    context = SimpleNamespace(artifact_dir=world["dir"], scope="all", max_length=64, target_fpr=0.1, seed=0)
    return probe_monitor_run._evaluate_probes_on_sets(
        db, [(probe, object(), SimpleNamespace(rule="mean", rule_params={}), None, None)], context, None,
        SimpleNamespace(pad_token_id=0), "LlamaForCausalLM", ["pmd_eval"],
    )


def test_an_eval_view_records_its_catch_rate_at_the_shipped_bar(world):
    """ENH-006 D84: the writer records positives and how many fire at the SHIPPED bar (`>=`), from the very
    scores it saves — the count `recall_set_of` falls back on while the bar is unchanged."""
    db = FakeDB(world)
    _evaluate_at(world, db, 0.5)
    (row,) = db.added
    scores, labels = np.load(row.scores_path), np.load(row.labels_path)
    positives = scores[labels == 1]
    assert row.metrics["catch_rate"] == {"threshold": 0.5, "positives": int(len(positives)),
                                         "caught": int((positives >= np.float32(0.5)).sum())}
    assert 0 < row.metrics["catch_rate"]["caught"] < len(positives)        # the fixture discriminates


def test_a_nonsense_view_records_its_false_alarms_at_the_shipped_bar(world):
    """ENH-006 D84: a `nonsense_text` view is built negatives-only and records false alarms like everyday."""
    world["view"].role, world["view"].label_mapping = "nonsense_text", {"*": "negative"}
    world["view"].distribution = None
    db = FakeDB(world)
    _evaluate_at(world, db, 0.5)
    (row,) = db.added
    scores, labels = np.load(row.scores_path), np.load(row.labels_path)
    assert set(labels.tolist()) == {0} and "catch_rate" not in row.metrics
    assert row.metrics["nonsense_text"] == {"threshold": 0.5, "rows": int(len(scores)),
                                            "false_alarms": int((scores >= np.float32(0.5)).sum())}
