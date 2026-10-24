"""A live probe TRAINING run says what its percentage is counting (operator, 2026-10-21).

Within each stage `execute_probe_run`'s heartbeat now writes `progress_detail` beside `progress`:
the stage, its units done and total (capture batches, probes, evaluation SETS), the item inside the
stage (a probe's epochs, a set's batches), a rate and an ETA. Driven through the REAL run on CPU with
the pipeline test's tiny model, over TWO evaluation sets so the per-set accounting is exercised.

NEGATIVE CONTROLS (recorded in 0xcc/reviews/live_progress_2026-10-21.md):
  R1  `on_epoch=` dropped from the `train_rule` call            → the epochs test
  R2  the `segment(...)` call dropped from `_evaluate_probes_on_sets` → the sets test and the
      monotone-across-sets test (the bar rewound at every set)
  R3  `scale=1000` dropped from the calibrating heartbeat        → the calibration-units test
  R4  `progress_detail=` dropped from the heartbeat's write      → the per-stage readings test
  R5  the stage-entry write dropped from `stage()`               → the every-stage test
  R6  the unthrottled poll before completion removed             → the late-cancel test
"""
import uuid

import pytest

pytest.importorskip("datasets")

from tests.unit.test_probe_monitor_pipeline import (  # noqa: E402,F401 - fixtures by import
    _loader_that_records_its_placement,
    arrow_dataset,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
)

SET_NAMES = ("unseen-a", "unseen-b")


@pytest.fixture(scope="module")
def two_set_run(pipeline_db, arrow_dataset):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun

    model_row = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny-random", status=ModelStatus.READY,
                      file_path=str(arrow_dataset), architecture="LlamaForCausalLM",
                      params_count=1_000)
    dataset_row = Dataset(id=uuid.uuid4(), name="probe-corpus", source="local",
                          status=DatasetStatus.READY, raw_path=str(arrow_dataset), extra_metadata={})
    pipeline_db.add_all([model_row, dataset_row])
    pipeline_db.commit()
    mapping = {"high": "positive", "low": "negative"}
    train = ProbeMonitorDataset(name="train", dataset_id=dataset_row.id, input_column="inputs",
                                label_column="stakes", label_mapping=mapping, pair_column="pair",
                                role="train", counts={})
    evals = [
        ProbeMonitorDataset(name=name, dataset_id=dataset_row.id, input_column="inputs",
                            label_column="stakes", label_mapping=mapping, role="eval",
                            distribution="out_of_distribution", counts={})
        for name in SET_NAMES
    ]
    pipeline_db.add_all([train, *evals])
    pipeline_db.commit()
    run = ProbeMonitorRun(
        model_id=model_row.id, train_dataset_id=train.id,
        eval_dataset_ids=[view.id for view in evals],
        config={"layers": [1, 2], "rules": ["mean"], "scope": "all", "max_length": 64,
                "val_fraction": 0.25, "seed": 1337, "top_n_layers": 1, "target_fpr": 0.1},
        status="running", environment={},
    )
    pipeline_db.add(run)
    pipeline_db.commit()
    return run


@pytest.fixture(scope="module")
def readings(pipeline_db, two_set_run, tiny_lm, tiny_tokenizer, tmp_path_factory,
             monkeypatch_module):
    """Run the REAL `execute_probe_run` once, recording every `record_progress` it makes."""
    from src.core import config as config_module
    from src.services import probe_monitor_capture as capture_module
    from src.services import probe_monitor_run as run_module

    monkeypatch_module.setattr(config_module.settings, "data_dir",
                               tmp_path_factory.mktemp("live_artifacts"), raising=False)
    # Several batches per set, and every report written, so a rewind between sets is VISIBLE: with
    # the 10 s throttle a fast test writes only each set's first and last batch.
    real_plan = capture_module.plan_batches
    monkeypatch_module.setattr(
        capture_module, "plan_batches",
        lambda examples, *, token_budget=None: real_plan(examples, token_budget=8),
    )
    monkeypatch_module.setattr(run_module, "HEARTBEAT_SECONDS", 0.0)

    recorded = []
    real = run_module.record_progress

    def spy(kind, target_id, **kwargs):
        landed = real(kind, target_id, **kwargs)
        recorded.append({"landed": landed, **kwargs})
        return landed

    monkeypatch_module.setattr(run_module, "record_progress", spy)
    run_module.execute_probe_run(
        pipeline_db, two_set_run.id,
        model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer),
    )
    return [r for r in recorded if r.get("progress_detail") is not None]


def _in(readings, stage):
    return [r for r in readings if r["progress_detail"]["stage"] == stage]


def test_every_stage_says_it_began(readings):
    from src.services.probe_monitor_run import STAGE_NAMES

    entered = [r["progress_detail"]["stage"] for r in readings if r["progress_detail"]["done"] is None]
    assert entered == STAGE_NAMES, entered
    assert all(r["progress_detail"]["kind"] == "training" for r in readings)


def test_the_captures_count_batches(readings):
    for stage in ("pooled_capture", "token_capture"):
        counted = [r["progress_detail"] for r in _in(readings, stage) if r.get("progress") is not None]
        assert counted, f"{stage} wrote no reading inside the stage"
        assert all(d["unit"] == "batches" and 0 < d["done"] <= d["total"] for d in counted), counted
        assert max(d["total"] for d in counted) > 1, "one batch: the readings prove nothing"


def test_training_reports_EPOCHS_inside_a_probe(readings):
    epochs = [
        r["progress_detail"]["current"] for r in _in(readings, "training")
        if (r["progress_detail"]["current"] or {}).get("unit") == "epochs"
    ]
    assert epochs, "the training stage never said which epoch a probe was on"
    assert epochs[0]["name"].startswith("mean at layer")
    assert all(1 <= e["done"] <= e["total"] for e in epochs)


def test_evaluation_counts_SETS_and_names_the_one_in_progress(readings):
    sets = [r["progress_detail"] for r in _in(readings, "evaluating") if r.get("progress") is not None]
    assert sets and all(d["unit"] == "sets" and d["total"] == len(SET_NAMES) for d in sets), sets
    assert {d["current"]["name"] for d in sets} == set(SET_NAMES)


def test_evaluation_progress_does_not_rewind_between_sets(readings):
    """Each set's batches restarted the band from its low end, so the bar went BACKWARDS at every
    set boundary. They are now one item of `count` within the stage."""
    values = [r["progress"] for r in _in(readings, "evaluating") if r.get("progress") is not None]
    assert len(values) >= 4, values
    assert values == sorted(values), f"the evaluating readings rewind: {values}"


def test_calibration_reports_whole_units_not_thousandths(readings):
    counted = [r["progress_detail"] for r in _in(readings, "calibrating") if r.get("progress") is not None]
    assert counted, "calibrating wrote no reading"
    assert all(d["total"] <= 10 for d in counted), counted


def test_a_rate_is_measured(readings):
    assert any(r["progress_detail"]["rate_per_second"] for r in readings)


def test_the_reading_is_COMMITTED_on_the_row(pipeline_db, two_set_run, readings):
    from sqlalchemy.orm import sessionmaker

    from src.models.probe_monitor import ProbeMonitorRun

    other = sessionmaker(bind=pipeline_db.get_bind())()
    try:
        row = other.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == two_set_run.id).one()
        assert row.status == "completed"
        assert row.progress_detail and row.progress_detail["kind"] == "training"
    finally:
        other.close()


def test_a_cancel_in_the_last_seconds_is_NOT_overwritten_by_completion(
    pipeline_db, two_set_run, readings, tiny_lm, tiny_tokenizer, monkeypatch,
):
    """`stage()`'s cancel check is throttled on time, so a cancel landing after the last poll was
    turned into COMPLETED by the success path. The checker here never re-polls on its own (one
    hour), so only the unthrottled poll before completion can see the cancel."""
    import sqlalchemy as sa

    from src.core.cancellation import OperatorCancelled, cancel_checker
    from src.models.probe_monitor import ProbeMonitorRun
    from src.services import probe_monitor_run as run_module

    run = ProbeMonitorRun(
        model_id=two_set_run.model_id, train_dataset_id=two_set_run.train_dataset_id,
        eval_dataset_ids=list(two_set_run.eval_dataset_ids)[:1], config=dict(two_set_run.config),
        status="running", environment={},
    )
    pipeline_db.add(run)
    pipeline_db.commit()
    engine = pipeline_db.get_bind()

    monkeypatch.setattr(
        run_module, "cancel_checker_for",
        lambda run_id, db=None: cancel_checker("probe_monitor_run", run_id, db=db,
                                               min_interval_s=3600.0),
    )
    real_rung = run_module.recompute_rung

    def rung_then_cancel(db, probe_id):
        result = real_rung(db, probe_id)
        with engine.begin() as connection:  # what the API does, from another connection
            connection.execute(sa.text("UPDATE probe_monitor_runs SET status = 'cancelled' "
                                       "WHERE id = :i"), {"i": run.id})
        return result

    monkeypatch.setattr(run_module, "recompute_rung", rung_then_cancel)
    with pytest.raises(OperatorCancelled):
        run_module.execute_probe_run(
            pipeline_db, run.id,
            model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer),
        )
    with engine.connect() as connection:
        status = connection.execute(
            sa.text("SELECT status FROM probe_monitor_runs WHERE id = :i"), {"i": run.id}
        ).scalar_one()
    assert status == "cancelled", f"the cancellation was overwritten: {status}"
