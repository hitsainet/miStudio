"""A page of probe training runs for an agent (ENH-006 G36).

`list_probe_training_runs` read `GET /probe-monitors/runs`, every run whole: 161,922 characters for
39 runs on 2026-11-07, most of it each run's environment (memory forecast, stage timings, GPU
memory, selection) and layer grid. Queuing the D128 stack needed seven ids and their states. The
tool now reads `GET /probe-monitors/runs-page`: filtered by status, training view, model,
behaviour and a search over id or label, paged with the total that matched, and a summary per run
unless `detail=full`.

Through the real route over the real database. The tool's call and payload are pinned in
`test_reachability.EXPECTED_CALLS`; its DEFAULTS are pinned here.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.asyncio

API = "/api/v1/probe-monitors"
T0 = datetime(2026, 11, 7, 11, 34, tzinfo=timezone.utc)

BULK_ENV = {
    "memory_forecast": {"arithmetic": "needed 46.78 GiB = " + "x" * 2000},
    "stage_seconds": {"training": 9184.4, "evaluating": 5622.6},
    "gpu_memory": {"stages": {s: {"peak_reserved_mib": 22140.0} for s in "abcdefgh"}},
}
BULK_GRID = {"grid": [{"layer": layer, "pooling": "mean", "val_auroc": 0.91} for layer in range(8, 26, 2)]}


async def _world(session):
    """Seven stack runs on model A (first is running, one selected probe), one other-model run,
    one failed run of another behaviour."""
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun

    def model():
        return Model(id=f"m_{uuid.uuid4().hex[:12]}", name="llama", status=ModelStatus.READY,
                     file_path="/x", architecture="LlamaForCausalLM", params_count=1)

    a, b = model(), model()
    dataset = Dataset(id=uuid.uuid4(), name="ssn", source="local", status=DatasetStatus.READY,
                      raw_path="/x", extra_metadata={})
    session.add_all([a, b, dataset])
    await session.flush()
    views = [ProbeMonitorDataset(name=f"v{i}", dataset_id=dataset.id, input_column="text",
                                 label_column="label", label_mapping={"1": "positive"}, role="train",
                                 counts={}) for i in range(7)]
    calibration = ProbeMonitorDataset(name="cal", dataset_id=dataset.id, input_column="text",
                                      label_column="label", label_mapping={"0": "negative"},
                                      role="calibration", counts={})
    session.add_all([*views, calibration])
    await session.flush()

    def run(model_id, view, minutes, label, status="pending", behaviour="dlp-ssn-us", **kw):
        return ProbeMonitorRun(
            id=f"pmr_{uuid.uuid4().hex[:12]}", model_id=model_id, train_dataset_id=view.id,
            eval_dataset_ids=["pmd_e1", "pmd_e2"], calibration_dataset_id=calibration.id,
            config={"target_fpr": 1 / 700, "layers": list(range(8, 26, 2))}, status=status,
            behaviour=behaviour, behaviour_label=label, environment=BULK_ENV,
            layer_selection=BULK_GRID, created_at=T0 + timedelta(minutes=minutes), **kw)

    forms = ["digit forms", "spelled", "split into parts", "spaced", "mixed with words",
             "split across lines", "glued letters"]
    stack = [run(a.id, views[i], i, f"US SSN disclosure, D128 stack: {f}") for i, f in enumerate(forms)]
    stack[0].status, stack[0].stage, stack[0].progress = "running", "pooled_capture", 12.0
    stack[0].progress_detail = {"stage": "pooled_capture", "eta_seconds": 2400.0}
    stack[0].gpu_uuid = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
    other_model = run(b.id, views[0], 10, "LFM probe 100%_kept", status="completed")
    failed = run(a.id, views[1], 11, "humor probe", status="failed", behaviour="humor",
                 error_message="the batches need 52 GiB")
    session.add_all([*stack, other_model, failed])
    await session.flush()
    session.add(ProbeMonitor(id=f"pm_{uuid.uuid4().hex[:12]}", run_id=other_model.id, layer=8,
                             rule="mean", selected=True))
    session.add(ProbeMonitor(id=f"pm_{uuid.uuid4().hex[:12]}", run_id=other_model.id, layer=8,
                             rule="max", selected=False))
    await session.commit()
    return {"a": a.id, "b": b.id, "views": [v.id for v in views], "stack": [r.id for r in stack],
            "other_model": other_model.id, "failed": failed.id, "calibration": calibration.id}


async def _page(client, **params):
    r = await client.get(f"{API}/runs-page", params=params)
    assert r.status_code == 200, r.text
    return r.json()


class TestFilters:
    async def test_search_over_the_label_finds_the_stack(self, client, async_session):
        w = await _world(async_session)
        page = await _page(client, search="d128 STACK")
        assert page["total"] == 7 and {r["id"] for r in page["items"]} == set(w["stack"])

    async def test_search_over_an_id(self, client, async_session):
        w = await _world(async_session)
        page = await _page(client, search=w["stack"][4][4:].upper())
        assert [r["id"] for r in page["items"]] == [w["stack"][4]]

    async def test_percent_and_underscore_are_text(self, client, async_session):
        w = await _world(async_session)
        assert [r["id"] for r in (await _page(client, search="100%_k"))["items"]] == [w["other_model"]]
        assert (await _page(client, search="d1_8"))["total"] == 0

    async def test_status_model_behaviour_and_view_narrow(self, client, async_session):
        w = await _world(async_session)
        assert [r["id"] for r in (await _page(client, status="running"))["items"]] == [w["stack"][0]]
        assert (await _page(client, model_id=w["b"]))["total"] == 1
        assert (await _page(client, model_id=w["a"]))["total"] == 8
        assert [r["id"] for r in (await _page(client, behaviour="humor"))["items"]] == [w["failed"]]
        on_view_0 = await _page(client, train_dataset_id=w["views"][0])
        assert {r["id"] for r in on_view_0["items"]} == {w["stack"][0], w["other_model"]}


class TestPaging:
    async def test_pages_are_newest_first_disjoint_and_cover_the_total(self, client, async_session):
        w = await _world(async_session)
        seen = []
        for offset in (0, 4):
            page = await _page(client, search="d128", offset=offset, limit=4)
            assert (page["total"], page["offset"], page["limit"]) == (7, offset, 4)
            seen += [r["id"] for r in page["items"]]
        assert seen == list(reversed(w["stack"]))

    async def test_a_tie_on_created_at_is_broken_by_id(self, client, async_session):
        from sqlalchemy import update

        from src.models.probe_monitor import ProbeMonitorRun

        w = await _world(async_session)
        await async_session.execute(update(ProbeMonitorRun).values(created_at=T0))
        await async_session.commit()
        seen = []
        for offset in range(0, 7, 3):
            seen += [r["id"] for r in (await _page(client, search="d128", offset=offset, limit=3))["items"]]
        assert seen == sorted(w["stack"], reverse=True)

    async def test_the_default_page_is_fifty_and_the_cap_two_hundred(self, client, async_session):
        await _world(async_session)
        assert (await _page(client))["limit"] == 50
        assert (await client.get(f"{API}/runs-page", params={"limit": 201})).status_code == 422


class TestSummary:
    async def test_a_running_summary_says_where_it_is(self, client, async_session):
        w = await _world(async_session)
        item = (await _page(client, status="running"))["items"][0]
        assert item["stage"] == "pooled_capture" and item["progress"] == 12.0
        assert item["eta_seconds"] == 2400.0
        assert item["gpu_uuid"] == "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
        assert item["target_fpr"] == pytest.approx(1 / 700)
        assert item["eval_dataset_ids"] == ["pmd_e1", "pmd_e2"] and item["calibration_dataset_id"] == w["calibration"]
        assert item["train_dataset_id"] == w["views"][0] and item["model_id"] == w["a"]
        assert item["selected_probe_id"] is None
        for gone in ("config", "environment", "layer_selection", "progress_detail", "artifact_dir"):
            assert gone not in item
        assert "GiB" not in json.dumps(item) and "val_auroc" not in json.dumps(item)

    async def test_the_selected_probe_is_named_and_an_error_kept(self, client, async_session):
        from sqlalchemy import select

        from src.models.probe_monitor import ProbeMonitor

        w = await _world(async_session)
        chosen = (await async_session.execute(select(ProbeMonitor.id).where(
            ProbeMonitor.run_id == w["other_model"], ProbeMonitor.selected.is_(True)))).scalar_one()
        assert (await _page(client, model_id=w["b"]))["items"][0]["selected_probe_id"] == chosen
        failed = (await _page(client, status="failed"))["items"][0]
        assert failed["error_message"] == "the batches need 52 GiB"

    async def test_a_summary_is_a_fraction_of_the_whole_run(self, client, async_session):
        await _world(async_session)
        summary = await _page(client, search="d128")
        whole = await _page(client, search="d128", detail="full")
        assert whole["detail"] == "full" and "environment" in whole["items"][0]
        assert len(json.dumps(summary["items"])) * 4 < len(json.dumps(whole["items"]))


class TestTheOldListIsUnchanged:
    async def test_the_list_still_answers_whole_runs(self, client, async_session):
        await _world(async_session)
        rows = (await client.get(f"{API}/runs")).json()
        assert isinstance(rows, list) and len(rows) == 9 and all("environment" in r for r in rows)


def test_the_tool_defaults_to_a_summary_page_of_fifty():
    """The ledger passes every argument, so a default of `full` or a larger page passes it."""
    from tests.unit.test_reachability import _tools_by_name

    manager, recorder = _tools_by_name()
    asyncio.run(manager.call_tool("list_probe_training_runs", {}, context=None))
    assert recorder.calls == [("GET", "/probe-monitors/runs-page",
                               {"status": None, "train_dataset_id": None, "model_id": None,
                                "behaviour": None, "search": None, "offset": 0, "limit": 50,
                                "detail": "summary"})]
