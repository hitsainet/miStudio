"""The probe LIST and the probe REPORT state the same summary (2026-10-08).

`GET /probe-monitors/probes/{id}` returned `probe.length_band_count = 0` for every probe while
`GET /probe-monitors/probes?run_id=…` returned the true 4. The list derived its summary fields in
`summary_without_curve`; the report built its summary with a bare `model_validate`, so every
DERIVED field fell back to its schema default there:

* `length_band_count` → 0 ("no length bands") on a probe with four;
* `window_thresholds` → `{}`, which the schema DEFINES as "no per-window bar was placed";
* `scope` → None ("the run is gone") while the run was right there;
* `rung_language` / `rung_next_step` → "".

Each was a confident wrong statement rather than a blank. Both routes now build the summary with
`probe_summary`, and the derived fields default to None (not computed), never to a value.

These go through the real routes and a real database: a test that called `probe_summary` twice
would agree with itself and say nothing about which function each ROUTE calls.
"""

from __future__ import annotations

import uuid

import pytest

BANDS = [
    {"max_tokens": 32, "threshold": 10.0},
    {"max_tokens": 128, "threshold": 11.0},
    {"max_tokens": 512, "threshold": 12.0},
    {"max_tokens": None, "threshold": 13.0},
]
WINDOWS = {
    "all": {"threshold": 11.9},
    "prompt": {"threshold": 7.9},
    "response": {"threshold": 16.9},
}
DERIVED = ("length_band_count", "window_thresholds", "scope", "rung_language", "rung_next_step")


async def _seed(session, *, scope="input"):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun

    model = Model(
        id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
        file_path="/nonexistent", architecture="LlamaForCausalLM", params_count=1_000,
    )
    dataset = Dataset(
        id=uuid.uuid4(), name="corpus", source="local", status=DatasetStatus.READY,
        raw_path="/nonexistent", extra_metadata={},
    )
    session.add_all([model, dataset])
    await session.flush()
    view = ProbeMonitorDataset(
        name="train", dataset_id=dataset.id, input_column="inputs", label_column="stakes",
        label_mapping={"high": "positive", "low": "negative"}, role="train", counts={},
    )
    session.add(view)
    await session.flush()
    # A NON-default scope, so a summary that skipped the run lookup cannot match by accident.
    run = ProbeMonitorRun(
        model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[],
        config={"scope": scope}, status="completed", environment={},
    )
    session.add(run)
    await session.flush()

    def probe(**extra):
        return ProbeMonitor(
            run_id=run.id, layer=11, rule="mean", rule_params={}, variant="dense",
            val_metrics={"val_auroc": 0.9, "history": [{"epoch": 1, "loss": 1.0}]},
            selected=True, threshold=11.9, target_fpr=0.01, realised_fpr=0.01,
            threshold_source="calibration_set", streamable=True, rung=2, rung_reasons=[],
            **extra,
        )

    banded = probe(length_bands=BANDS, window_decisions=WINDOWS)
    plain = probe(length_bands=None, window_decisions=None)
    session.add_all([banded, plain])
    await session.flush()
    return run, banded, plain


async def _both(client, run_id, probe_id):
    listed = await client.get("/api/v1/probe-monitors/probes", params={"run_id": run_id})
    assert listed.status_code == 200, listed.text
    from_list = next(p for p in listed.json() if p["id"] == probe_id)
    report = await client.get(f"/api/v1/probe-monitors/probes/{probe_id}")
    assert report.status_code == 200, report.text
    return from_list, report.json()["probe"]


@pytest.mark.asyncio
async def test_a_probe_with_bands_reads_the_same_in_both_routes(client, async_session):
    run, banded, _ = await _seed(async_session)
    from_list, from_report = await _both(client, run.id, banded.id)

    assert from_report["length_band_count"] == 4, "the report must not say zero bands"
    assert from_report["window_thresholds"] == {"all": 11.9, "prompt": 7.9, "response": 16.9}
    assert from_report["scope"] == "input"
    assert from_report["rung_language"]
    for field in DERIVED:
        assert from_list[field] == from_report[field], field


@pytest.mark.asyncio
async def test_a_probe_with_no_bands_reads_the_same_in_both_routes(client, async_session):
    run, _, plain = await _seed(async_session)
    from_list, from_report = await _both(client, run.id, plain.id)

    # FILLED, and there are none — not "not computed".
    assert from_report["length_band_count"] == 0
    assert from_report["window_thresholds"] == {}
    for field in DERIVED:
        assert from_list[field] == from_report[field], field


@pytest.mark.asyncio
async def test_the_two_routes_differ_only_by_the_curve(client, async_session):
    """The one intended difference: the list strips the per-epoch curve, the report keeps it.
    Everything else in the summary is the same object, so it must serialise the same."""
    run, banded, _ = await _seed(async_session)
    from_list, from_report = await _both(client, run.id, banded.id)

    assert "history" in from_report["val_metrics"]
    assert "history" not in from_list["val_metrics"]
    strip = lambda s: {k: v for k, v in s.items() if k != "val_metrics"}  # noqa: E731
    assert strip(from_list) == strip(from_report)


def test_derived_fields_default_to_not_computed():
    """A summary nobody derived says so: None, never 0 or {} — those mean "filled, and none"."""
    from src.schemas.probe_monitor import ProbeMonitorSummary

    fields = ProbeMonitorSummary.model_fields
    assert fields["length_band_count"].default is None
    assert fields["window_thresholds"].default is None
