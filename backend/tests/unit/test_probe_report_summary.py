"""One probe's report, a summary by default for an agent (ENH-006 G37).

`get_probe_monitor_report` answered 1,347,273 characters for `pm_8b2c68df722a` on 2026-11-07,
775,255 of them six evaluations' stored ROC curves (about 115K each). The route now takes
`detail=summary|full`: `full` (the route's default, what the Probes screen draws) is the report as
it was; `summary` (the tool's default) drops each evaluation's and judge set's `roc` (keeping
`roc_points`, its length) and the probe's per-epoch `val_metrics.history`, and names both in
`omitted`. Every rung, catch-rate, false-alarm, threshold and transfer field is unchanged.

Through the real route over the real database. The tool's call and payload are pinned in
`test_reachability.EXPECTED_CALLS`; its default is pinned here.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest

pytestmark = pytest.mark.asyncio

API = "/api/v1/probe-monitors"

#: A stored curve the size the node's were: 2,985 points per set.
ROC_POINTS = 2985


def _roc(n: int = ROC_POINTS) -> list[dict]:
    return [{"threshold": 20.0 - i * 0.01, "fpr": i / n, "tpr": min(1.0, 2 * i / n)} for i in range(n)]


def _metrics(name: str, ood: bool) -> dict:
    return {
        "name": name, "out_of_distribution": ood, "scored": True, "auroc": 0.91,
        "ci": {"low": 0.88, "high": 0.94}, "n_positive": 410, "n_negative": 1200,
        "roc": _roc(),
        "operating_points": [{"target_fpr": 0.01, "threshold": 11.9, "realised_fpr": 0.0098,
                              "recall": 0.62}],
        "length_bands": [{"band": "0-64", "recall": 0.7}],
        "catch_rate": {"positives": 410, "fired": 41, "share": 0.1},
    }


RUNG_DETAIL = {
    "next_step": "evaluate on an everyday-text set",
    "everyday_text": {"status": "not_checked", "sets": []},
    "nonsense_text": {"status": "passed", "sets": [{"rows": 2000, "false_alarms": 3}]},
    "catch_rate": {"status": "failed", "sets": [{"positives": 410, "fired": 41,
                                                 "sentence": "catches 10% of positives"}]},
}


async def _world(session) -> str:
    """One probe of one run with six evaluations (a full stored curve each), a 300-epoch training
    curve, a stored rung detail and one judge run whose per-set result carries a curve too."""
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import (
        ProbeMonitor,
        ProbeMonitorDataset,
        ProbeMonitorEvaluation,
        ProbeMonitorJudgeRun,
        ProbeMonitorRun,
    )

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="llama", status=ModelStatus.READY,
                  file_path="/x", architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="ssn", source="local", status=DatasetStatus.READY,
                      raw_path="/x", extra_metadata={})
    session.add_all([model, dataset])
    await session.flush()
    views = [ProbeMonitorDataset(name=f"set{i}", dataset_id=dataset.id, input_column="text",
                                 label_column="label", label_mapping={"1": "positive", "0": "negative"},
                                 role="train" if i == 0 else "eval",
                                 distribution="out_of_distribution" if i else "in_distribution",
                                 counts={}) for i in range(7)]
    session.add_all(views)
    await session.flush()
    run = ProbeMonitorRun(id=f"pmr_{uuid.uuid4().hex[:12]}", model_id=model.id,
                          train_dataset_id=views[0].id, eval_dataset_ids=[v.id for v in views[1:]],
                          config={"target_fpr": 0.01}, status="completed")
    session.add(run)
    await session.flush()
    probe_id = f"pm_{uuid.uuid4().hex[:12]}"
    session.add(ProbeMonitor(
        id=probe_id, run_id=run.id, layer=12, rule="mean", selected=True, rung=1,
        rung_reasons=["not yet checked on everyday text"], rung_detail=RUNG_DETAIL,
        threshold=11.914362907409668, target_fpr=0.01, realised_fpr=0.0098,
        threshold_source="calibration_set",
        val_metrics={"auroc": 0.998, "history": [{"epoch": e, "loss": 1 / (e + 1), "val_auroc": 0.9}
                                                 for e in range(300)]},
    ))
    await session.flush()
    for view in views[1:]:
        session.add(ProbeMonitorEvaluation(
            id=f"pme_{uuid.uuid4().hex[:12]}", probe_id=probe_id, dataset_id=view.id,
            status="completed", n_positive=410, n_negative=1200,
            metrics=_metrics(view.name, ood=True)))
    session.add(ProbeMonitorJudgeRun(
        id=f"pmj_{uuid.uuid4().hex[:12]}", endpoint="http://j/v1", model="qwen",
        prompt_version="stakes-rating/v1", dataset_ids=[views[1].id], probe_id=probe_id,
        status="completed", max_rows_per_set=2000,
        metrics={"per_set": {views[1].id: {"auroc": 0.87, "roc": _roc(40),
                                           "decision": {"recall": 0.8}}},
                 "max_rows_per_set": 2000}))
    await session.commit()
    return probe_id


async def _report(client, probe_id, **params):
    r = await client.get(f"{API}/probes/{probe_id}", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _strip(value):
    """The full report as a summary should read it: every `roc` swapped for `roc_points`, the
    history gone. Written independently of the route's own helper, so the two cannot share a bug."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "roc":
                out["roc_points"] = len(item)
            elif key == "history":
                continue
            else:
                out[key] = _strip(item)
        return out
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


class TestTheRoute:
    async def test_full_is_the_default_and_carries_the_curves(self, client, async_session):
        probe_id = await _world(async_session)
        full = await _report(client, probe_id)
        assert full["detail"] == "full" and full["omitted"] == []
        assert len(full["evaluations"]) == 6
        assert all(len(e["metrics"]["roc"]) == ROC_POINTS for e in full["evaluations"])
        assert len(full["probe"]["val_metrics"]["history"]) == 300
        assert await _report(client, probe_id, detail="full") == full

    async def test_a_summary_is_the_full_report_without_the_curves(self, client, async_session):
        """Every field kept, field for field: rung, its words, next step, everyday/nonsense text,
        catch rate, thresholds, operating points, transfer, calibration size, judge decision."""
        probe_id = await _world(async_session)
        full = await _report(client, probe_id, detail="full")
        summary = await _report(client, probe_id, detail="summary")
        assert summary["detail"] == "summary"
        assert any("roc" in o for o in summary["omitted"]) and any("history" in o for o in summary["omitted"])
        expected = _strip({k: v for k, v in full.items() if k not in ("detail", "omitted")})
        assert {k: v for k, v in summary.items() if k not in ("detail", "omitted")} == expected
        for e in summary["evaluations"]:
            assert "roc" not in e["metrics"] and e["metrics"]["roc_points"] == ROC_POINTS
            assert e["metrics"]["operating_points"] and e["metrics"]["catch_rate"]
        judge_set = next(iter(summary["judge_runs"][0]["metrics"]["per_set"].values()))
        assert "roc" not in judge_set and judge_set["roc_points"] == 40
        assert "history" not in summary["probe"]["val_metrics"]
        assert summary["probe"]["val_metrics"]["auroc"] == 0.998

    async def test_the_rung_and_bar_fields_survive_by_name(self, client, async_session):
        """Named, so a summary that dropped one of them is red even if `_strip` drifted."""
        probe_id = await _world(async_session)
        summary = await _report(client, probe_id, detail="summary")
        assert summary["catch_rate"] == RUNG_DETAIL["catch_rate"]
        assert summary["everyday_text"] == RUNG_DETAIL["everyday_text"]
        assert summary["nonsense_text"] == RUNG_DETAIL["nonsense_text"]
        assert summary["rung_next_step"] == "evaluate on an everyday-text set"
        assert summary["rung_language"] and summary["probe"]["rung"] == 1
        assert summary["probe"]["threshold"] == 11.914362907409668
        assert summary["probe"]["target_fpr"] == 0.01 and summary["probe"]["realised_fpr"] == 0.0098
        assert summary["threshold_transfer"] and summary["threshold_transfer"] == (
            await _report(client, probe_id))["threshold_transfer"]

    async def test_a_summary_is_under_a_twentieth_of_the_full_report(self, client, async_session):
        probe_id = await _world(async_session)
        full = len(json.dumps(await _report(client, probe_id, detail="full")))
        summary = len(json.dumps(await _report(client, probe_id, detail="summary")))
        assert full > 600_000, full                      # six curves the size the node stored
        assert summary * 20 < full, (summary, full)

    async def test_an_unknown_detail_is_refused(self, client, async_session):
        probe_id = await _world(async_session)
        assert (await client.get(f"{API}/probes/{probe_id}", params={"detail": "tiny"})).status_code == 422


def test_the_tool_asks_for_a_summary_by_default():
    """The ledger passes every argument, so a tool default of `full` passes it."""
    from tests.unit.test_reachability import _tools_by_name

    manager, recorder = _tools_by_name()
    asyncio.run(manager.call_tool("get_probe_monitor_report", {"probe_id": "pm_1"}, context=None))
    assert recorder.calls == [("GET", "/probe-monitors/probes/pm_1", {"detail": "summary"})]
