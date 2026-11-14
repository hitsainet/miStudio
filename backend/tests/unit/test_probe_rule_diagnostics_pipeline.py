"""G39 after training: a REAL run records the evidence's dilution and each probe's constant floor.

Drives `execute_probe_run` (the pipeline test's database, tiny random LM and the G38 prefix tokenizer) over a
training view with a span column. A row is `prefix (7) + n words + </turn> + <turn> assistant`, so n + 3 tokens
are scored with the prefix skipped; a positive's evidence is its first k words, k tokens. Rows differ in n and
k, and every seventh row is EXCLUDED by the mapping, so a row's place among the run's examples is not its
place in the split: a join by the wrong one, a width read with the prefix, or a span counted in characters
gives other figures than the ones computed here from the data alone. Removing the line in
`execute_probe_run` that records `environment["rule_diagnostics"]` turns this red (controls in the review
record).
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

from tests.unit.test_probe_monitor_pipeline import (  # noqa: F401 - fixtures
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
)
from tests.unit.test_template_prefix_pipeline import prefix_tokenizer  # noqa: F401 - fixture

pytest.importorskip("datasets")

N_ROWS = 84
#: Scored per row beyond its words: </turn>, <turn>, assistant (the prefix is skipped).
TAIL = 3


def _rows():
    rng = np.random.default_rng(11)
    inputs, labels, spans, expected = [], [], [], []
    for i in range(N_ROWS):
        excluded = i % 7 == 6
        positive = i % 2 == 0
        n = int(rng.integers(4, 9))
        k = int(rng.integers(1, 4)) if positive else 0
        head = [f"risk{rng.integers(0, 8)}" for _ in range(k)] if positive else [f"calm{rng.integers(0, 8)}"]
        words = head + [f"filler{rng.integers(0, 16)}" for _ in range(n - len(head))]
        text = " ".join(words)
        inputs.append(text)
        labels.append("skip" if excluded else ("high" if positive else "low"))
        evidence = " ".join(head)
        spans.append(json.dumps([[0, len(evidence)]]) if positive else "[]")
        if positive and not excluded:
            expected.append(k / (n + TAIL))
    return inputs, labels, spans, expected


@pytest.fixture(scope="module")
def span_corpus(tmp_path_factory):
    from datasets import Dataset as HFDataset

    inputs, labels, spans, _ = _rows()
    path = tmp_path_factory.mktemp("span_corpus") / "arrow"
    HFDataset.from_dict({"inputs": inputs, "stakes": labels, "spans": spans}).save_to_disk(str(path))
    return path


@pytest.fixture(scope="module")
def diagnosed(pipeline_db, span_corpus, monkeypatch_module, tiny_lm, prefix_tokenizer, tmp_path_factory):  # noqa: F811
    from src.core import config as config_module
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun
    from src.services import probe_monitor_run as run_module
    from src.services import probe_rule_recommendation as R

    monkeypatch_module.setattr(config_module.settings, "data_dir", tmp_path_factory.mktemp("g39_artifacts"),
                               raising=False)
    model_row = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny-random-g39", status=ModelStatus.READY,
                      file_path=str(span_corpus), architecture="LlamaForCausalLM", params_count=1_000)
    dataset_row = Dataset(id=uuid.uuid4(), name="probe-corpus-g39", source="local", status=DatasetStatus.READY,
                          raw_path=str(span_corpus), extra_metadata={})
    pipeline_db.add_all([model_row, dataset_row])
    pipeline_db.commit()
    from datasets import load_from_disk

    data = load_from_disk(str(span_corpus))
    profile = R.profile_spans("spans", [(i, 1 if s == "high" else 0, t, c) for i, (t, s, c)
                                        in enumerate(zip(data["inputs"], data["stakes"], data["spans"], strict=True))
                                        if s != "skip"])
    train_view = ProbeMonitorDataset(
        name="train", dataset_id=dataset_row.id, input_column="inputs", label_column="stakes",
        label_mapping={"high": "positive", "low": "negative", "skip": "excluded"}, role="train", counts={},
        span_column="spans", evidence_spans=profile,
    )
    pipeline_db.add(train_view)
    pipeline_db.commit()
    run = ProbeMonitorRun(
        model_id=model_row.id, train_dataset_id=train_view.id, eval_dataset_ids=[],
        config={"layers": [1, 2], "rules": ["max", "mean"], "scope": "all", "max_length": 64,
                "val_fraction": 0.25, "seed": 1337, "top_n_layers": 1, "target_fpr": 0.1,
                "calibration_max_rows": 100, "skip_template_prefix": True},
        status="pending", environment={},
    )
    pipeline_db.add(run)
    pipeline_db.commit()

    def loader(row):
        row.gpu_request = "auto"
        row.gpu_uuid = "GPU-00000000-1111-2222-3333-444444444444"
        return tiny_lm, prefix_tokenizer, "LlamaForCausalLM"

    run_module.execute_probe_run(pipeline_db, run.id, model_loader=loader)
    pipeline_db.refresh(run)
    return run


def test_the_run_completes(diagnosed):
    assert diagnosed.status == "completed", diagnosed.error_message


def test_dilution_is_the_evidences_tokens_over_each_rows_scored_tokens(diagnosed):
    expected = _rows()[3]
    dilution = diagnosed.environment["rule_diagnostics"]["dilution"]
    assert dilution["n"] == len(expected)
    for name, q in (("p10", 10), ("median", 50), ("p90", 90)):
        assert dilution[name] == pytest.approx(float(np.percentile(expected, q)), abs=1e-6), name
    assert len(set(expected)) > 3, "the fixture's control: the rows' figures differ"


def test_every_probe_has_its_constant_floor_on_the_validation_rows(diagnosed, pipeline_db):  # noqa: F811
    from src.models.probe_monitor import ProbeMonitor

    probes = pipeline_db.query(ProbeMonitor).filter(ProbeMonitor.run_id == diagnosed.id).all()
    rows = diagnosed.environment["rule_diagnostics"]["probes"]
    assert sorted(r["probe_id"] for r in rows) == sorted(p.id for p in probes) and len(rows) == 2
    # The validation split (val_fraction 0.25, whole pairs kept together): the same rows for every probe.
    sizes = {row["constant_floor"]["rows"] for row in rows if row["constant_floor"]}
    assert len(sizes) == 1 and N_ROWS * 0.2 <= sizes.pop() <= N_ROWS * 0.3
    for row in rows:
        floor = row["constant_floor"]
        assert floor is not None, row
        assert 1 <= floor["rows_at_min"] <= floor["rows"]
        assert set(floor) >= {"min_score", "share", "floored", "positives_at_min"}
