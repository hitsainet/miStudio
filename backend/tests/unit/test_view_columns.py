"""View columns, sub-questions and `blend_fit` (Feature 040 FR-4, FR-12 – FR-14; FTASKS 6.2 – 6.7).

Splits are real `datasets` on disk read through the real `open_split`; registration and the column update
go through the REAL routes. Column names, values and keys are random.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from src.core.config import settings
from src.services import view_columns
from tests.support.detector_fixtures import dataset_row, view_row, write_split
from tests.support.random_behaviour import behaviour_body, rng_for, slug

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


@pytest.fixture
def rng(request):
    return rng_for(request.node.nodeid)


def columns(rng, n=40):
    kinds = [slug(rng, words=1) for _ in range(3)]
    return {
        "text": [f"row {i}" for i in range(n)],
        "label": [str(i % 2) for i in range(n)],
        "kind": [kinds[i % 3] for i in range(n)],
        "row_id": [f"id-{i}" for i in range(n)],
        "flag": [bool(i % 4 == 0) for i in range(n)],
    }, kinds


async def _dataset(async_session, data_dir, rng, n=40):
    cols, kinds = columns(rng, n)
    ds = await dataset_row(async_session, write_split(data_dir, cols))
    await async_session.commit()
    return ds, cols, kinds


def body(ds, **extra):
    out = {"name": f"view-{ds.id.hex[:6]}", "dataset_id": str(ds.id), "input_column": "text", "label_column": "label",
           "label_mapping": {"1": "positive", "0": "negative"}, "role": "eval"}
    out.update(extra)
    return out


# ── registration declares and snapshots (FR-12) ───────────────────────────────────────────────


async def test_registration_snapshots_declared_columns_by_split_position(client, async_session, data_dir, rng):
    ds, cols, kinds = await _dataset(async_session, data_dir, rng)
    made = await client.post(f"{API}/datasets", json=body(ds, group_columns=["kind", "flag"]))
    assert made.status_code == 201, made.text
    view = made.json()
    assert view["group_columns"] == ["kind", "flag"]
    entry = view["column_snapshots"]["kind"]
    snap = view_columns.load(entry, "kind")
    assert [snap.values[c] for c in np.asarray(snap.codes).tolist()] == cols["kind"]   # position i = row i
    assert entry["n_rows"] == 40 and entry["n_values"] == 3
    assert Path(entry["path"]).is_relative_to(data_dir / "probe_monitor_views" / view["id"])
    flag = view_columns.load(view["column_snapshots"]["flag"], "flag")
    assert sorted(flag.values) == [False, True]
    # the snapshot is position-indexed data only: no text column was copied
    assert not any("row " in json.dumps(v) for v in snap.values)


async def test_an_absent_column_is_refused_and_nothing_is_written(client, async_session, data_dir, rng):
    ds, *_ = await _dataset(async_session, data_dir, rng)
    missing = slug(rng)
    response = await client.post(f"{API}/datasets", json=body(ds, group_columns=["kind", missing]))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert (detail["code"], detail["columns"]) == ("column_absent", [missing])
    assert not (data_dir / "probe_monitor_views").exists() or not any((data_dir / "probe_monitor_views").rglob("*.npy"))


async def test_a_row_id_column_is_refused_too_many_values(client, async_session, data_dir, rng):
    from src.core import runtime_settings

    ds, *_ = await _dataset(async_session, data_dir, rng)
    try:
        runtime_settings.apply({"group_column_max_values": 10})
        detail = (await client.post(f"{API}/datasets", json=body(ds, group_columns=["row_id"]))).json()["detail"]
    finally:
        runtime_settings.reset()
    assert (detail["code"], detail["column"], detail["value"]) == ("too_many_values", "row_id", 10)


# ── additive update (FR-14) ──────────────────────────────────────────────────────────────────


async def test_columns_are_added_never_removed(client, async_session, data_dir, rng):
    ds, *_ = await _dataset(async_session, data_dir, rng)
    view = (await client.post(f"{API}/datasets", json=body(ds, group_columns=["kind"]))).json()
    first = view["column_snapshots"]["kind"]["sha256"]
    added = await client.patch(f"{API}/datasets/{view['id']}/columns", json={"group_columns": ["kind", "flag"]})
    assert added.status_code == 200, added.text
    assert added.json()["added"] == ["flag"] and added.json()["group_columns"] == ["kind", "flag"]
    assert added.json()["column_snapshots"]["kind"]["sha256"] == first           # unchanged, not re-read
    removed = await client.patch(f"{API}/datasets/{view['id']}/columns", json={"group_columns": ["flag"]})
    assert removed.status_code == 422
    assert (removed.json()["detail"]["code"], removed.json()["detail"]["removed"]) == ("column_not_additive", ["kind"])


async def test_column_history_without_a_behaviour(client, async_session, data_dir, rng):
    from src.models.detector_report import ProbeMonitorDatasetColumnChange

    ds, *_ = await _dataset(async_session, data_dir, rng)
    view = (await client.post(f"{API}/datasets", json=body(ds))).json()
    await client.patch(f"{API}/datasets/{view['id']}/columns", json={"group_columns": ["kind"]},
                       headers={"X-MiStudio-Agent": "mcp"})
    rows = (await async_session.execute(select(ProbeMonitorDatasetColumnChange).where(
        ProbeMonitorDatasetColumnChange.view_id == view["id"]))).scalars().all()
    assert [(r.field, r.old_value, r.new_value, r.actor) for r in rows] == [("group_columns", [], ["kind"], "agent:mcp")]


async def test_column_history_with_a_behaviour_is_the_behaviours(client, async_session, data_dir, rng):
    from src.models.probe_behaviour import ProbeBehaviourHistory

    ds, *_ = await _dataset(async_session, data_dir, rng)
    key = slug(rng)
    view = (await client.post(f"{API}/datasets", json=body(ds, behaviour=key))).json()
    await client.patch(f"{API}/datasets/{view['id']}/columns", json={"group_columns": ["kind"]})
    rows = (await async_session.execute(select(ProbeBehaviourHistory).where(
        ProbeBehaviourHistory.key == key, ProbeBehaviourHistory.field == "group_columns"))).scalars().all()
    assert [(r.subject_id, r.new_value) for r in rows] == [(view["id"], ["kind"])]


async def test_an_archived_view_refuses_new_columns(client, async_session, data_dir, rng):
    ds, *_ = await _dataset(async_session, data_dir, rng)
    view = (await client.post(f"{API}/datasets", json=body(ds))).json()
    await client.post(f"{API}/datasets/{view['id']}/archive", json={})
    response = await client.patch(f"{API}/datasets/{view['id']}/columns", json={"group_columns": ["kind"]})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "probe_dataset_archived"


# ── sub-questions against the CITED version (FR-4) ───────────────────────────────────────────


async def _behaviour_with_two_versions(client, rng):
    sub_a, sub_b = slug(rng), slug(rng)
    b = behaviour_body(rng, sub_questions=[{"key": sub_a, "question": "q", "positive_meaning": "p"}])
    await client.post(f"{API}/behaviours", json=b)
    await client.post(f"{API}/behaviours/{b['key']}/versions", json={
        "base_version": 1, "sub_questions": [{"key": sub_b, "question": "q", "positive_meaning": "p"}]})
    return b["key"], sub_a, sub_b


async def test_a_sub_question_is_checked_against_the_cited_version(client, async_session, data_dir, rng):
    ds, *_ = await _dataset(async_session, data_dir, rng)
    key, removed, current = await _behaviour_with_two_versions(client, rng)
    old = await client.post(f"{API}/datasets", json=body(ds, behaviour=key, sub_question=removed, behaviour_version=1))
    assert old.status_code == 201, old.text
    assert (old.json()["sub_question"], old.json()["behaviour_version"]) == (removed, 1)
    new = await client.post(f"{API}/datasets", json=body(ds, behaviour=key, sub_question=current))
    assert new.status_code == 201 and new.json()["behaviour_version"] == 2
    refused = await client.post(f"{API}/datasets", json=body(ds, behaviour=key, sub_question=removed))
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert (detail["code"], detail["declared_keys"], detail["behaviour_version"]) == ("sub_question_undeclared", [current], 2)


async def test_a_sub_question_needs_a_behaviour(client, async_session, data_dir, rng):
    ds, *_ = await _dataset(async_session, data_dir, rng)
    detail = (await client.post(f"{API}/datasets", json=body(ds, sub_question=slug(rng)))).json()["detail"]
    assert (detail["code"], detail["rule"]) == ("invalid_field", "requires_behaviour")


# ── blend_fit (FR-13) ────────────────────────────────────────────────────────────────────────


async def test_blend_fit_is_a_role_a_link_and_an_evaluation_set(client, async_session, data_dir, rng):
    ds, *_ = await _dataset(async_session, data_dir, rng)
    blend = await client.post(f"{API}/datasets", json=body(ds, role="blend_fit"))
    assert blend.status_code == 201 and blend.json()["role"] == "blend_fit"
    b = behaviour_body(rng)
    await client.post(f"{API}/behaviours", json=b)
    linked = await client.post(f"{API}/behaviours/{b['key']}/views/{blend.json()['id']}", json={})
    assert linked.status_code == 200, linked.text
    assert [v["use"] for v in linked.json()["adopted"]["views"]] == ["blend_fit"]


async def test_a_blend_fit_view_is_never_a_training_view(client, async_session, data_dir, rng):
    from tests.support.detector_fixtures import model_row

    ds, *_ = await _dataset(async_session, data_dir, rng)
    model = await model_row(async_session)
    await async_session.commit()
    blend = (await client.post(f"{API}/datasets", json=body(ds, role="blend_fit"))).json()
    response = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": blend["id"]})
    assert response.status_code == 422 and "a run must train on a 'train' view" in response.text


async def test_the_purge_plan_of_a_blend_fit_view_names_its_role(client, async_session, data_dir, rng):
    from src.services.probe_dataset_lifecycle import ROLE_WORDS

    assert ROLE_WORDS["blend_fit"]
    ds, *_ = await _dataset(async_session, data_dir, rng)
    blend = (await client.post(f"{API}/datasets", json=body(ds, role="blend_fit"))).json()
    await client.post(f"{API}/datasets/{blend['id']}/archive", json={})
    plan = await client.post(f"{API}/datasets/purge/plan", json={"ids": [blend["id"]]})
    assert plan.status_code == 200, plan.text


# ── run and probe carry the training view's sub-question (FR-13) ─────────────────────────────


async def test_a_run_copies_sub_question_and_version_from_its_training_view(client, async_session, data_dir, rng, monkeypatch):
    from src.models.probe_monitor import ProbeMonitorRun
    from tests.support.detector_fixtures import model_row

    ds, *_ = await _dataset(async_session, data_dir, rng)
    key, _old, current = await _behaviour_with_two_versions(client, rng)
    model = await model_row(async_session)
    await async_session.commit()
    train = (await client.post(f"{API}/datasets", json=body(ds, role="train", behaviour=key, sub_question=current))).json()

    class Task:
        id = "task-1"

    monkeypatch.setattr("src.api.v1.endpoints.probe_monitors.gpu_delay", lambda task, gpu: (lambda **kw: Task()))
    response = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": train["id"]})
    assert response.status_code == 202, response.text
    run = (await async_session.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == response.json()["id"]))).scalar_one()
    assert (run.sub_question, run.behaviour_version) == (current, 2)
    assert run.config["selection_rule"] == "recall_at_decision" and run.config["calibration_max_rows"] == 5000


def test_persist_probe_copies_from_the_run():
    """`_persist_probe` writes the run's sub-question and version onto the probe (AST: the keywords read
    the RUN, not a constant)."""
    import ast
    import inspect

    from src.services import probe_monitor_run

    tree = ast.parse(inspect.getsource(probe_monitor_run._persist_probe))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "ProbeMonitor")
    keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert keywords["sub_question"] == "getattr(run, 'sub_question', None)"
    assert keywords["behaviour_version"] == "getattr(run, 'behaviour_version', None)"


# ── purge removes the snapshots (6.7) ────────────────────────────────────────────────────────


async def test_purge_deletes_the_views_snapshots_and_the_tombstone_says_so(client, async_session, data_dir, rng):
    ds, *_ = await _dataset(async_session, data_dir, rng)
    view = (await client.post(f"{API}/datasets", json=body(ds, group_columns=["kind"]))).json()
    directory = data_dir / "probe_monitor_views" / view["id"]
    assert directory.is_dir()
    await client.post(f"{API}/datasets/{view['id']}/archive", json={})
    purged = await client.post(f"{API}/datasets/purge", json={"ids": [view["id"]], "reason": "x"})
    assert purged.status_code == 200, purged.text
    result = purged.json()["results"][0]
    assert result["outcome"] == "purged"
    assert result["rows_removed"]["probe_monitor_views"] == str(directory)
    assert not directory.exists()


def test_snapshot_publish_is_atomic(tmp_path, monkeypatch):
    """A failure while writing leaves no published directory and no temporary one."""
    monkeypatch.setattr(view_columns.np, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        view_columns._write_column(tmp_path / "v", "c", np.zeros(3, dtype=np.int32), ["a"])
    assert list((tmp_path / "v").iterdir()) == []


async def test_the_column_count_cap_is_the_runtime_setting(client, async_session, data_dir, rng):
    from src.core import runtime_settings

    ds, *_ = await _dataset(async_session, data_dir, rng)
    try:
        runtime_settings.apply({"view_max_group_columns": 1})
        detail = (await client.post(f"{API}/datasets", json=body(ds, group_columns=["kind", "flag"]))).json()["detail"]
    finally:
        runtime_settings.reset()
    assert (detail["code"], detail["setting"], detail["value"]) == ("over_cap", "view_max_group_columns", 1)
