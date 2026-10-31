"""The three detectors, the unknown share and producer columns (Feature 040 S3; FR-33, FR-34, FR-35; FTASKS 21.2).

Pure function tests of `detectors` (the controls' targets: drop `AND NOT rule_veto`; `OR` → `AND`; treat
`unknown` as a match), then producer columns through the REAL task: compared, never preferred (FR-34), and
FR-31's one exception (`producer_computed`, pending operator code, the SAME digest).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.config import settings
from src.services.rules_evaluation import UNKNOWN_BUDGET, UNKNOWN_NONE, UNKNOWN_PENDING, detectors
from tests.support import rule_set_fixtures as rf
from tests.support import rules_world as rw
from tests.support.random_behaviour import rng_for

API = rw.API


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def _rows():
    #            match  veto   unknown          fires
    rows = [(True, False, UNKNOWN_NONE, False),     # a rule match alone
            (False, True, UNKNOWN_NONE, True),      # a veto above the probe bar: vetoed
            (False, False, UNKNOWN_NONE, True),     # the probe alone
            (True, False, UNKNOWN_BUDGET, False),   # unknown (budget): NO match
            (False, True, UNKNOWN_PENDING, True),   # unknown (pending): NO veto, the probe decides
            (False, False, UNKNOWN_NONE, False)]
    return [np.array([r[i] for r in rows]) for i in range(4)]


def test_the_three_detectors_row_by_row():
    match, veto, unknown, fires = _rows()
    d = detectors(match, veto, unknown.astype(np.uint8), fires)
    assert d["rules_alone"].tolist() == [True, False, False, False, False, False]
    assert d["probe_alone"].tolist() == [False, True, True, False, True, False]
    assert d["rules_or_probe"].tolist() == [True, False, True, False, True, False]


async def test_producer_columns_are_compared_never_preferred(client, async_session, data_dir, monkeypatch, request):
    from src.services import view_columns

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    # A producer column that says TRUE everywhere (it disagrees with miStudio on most rows), for this digest.
    import datasets

    split = datasets.load_from_disk(str((await _raw_path(async_session, world.eval_view))))
    column = "producer_" + rf.ident(rng_for(request.node.nodeid), "m")
    split = split.add_column(column, [True] * len(split))
    path = data_dir / "datasets" / "with-producer"
    split.save_to_disk(str(path))
    await _repoint(async_session, world.eval_view, str(path))
    limits = view_columns.ColumnLimits(max_values=10, max_columns=8)
    entries = await view_columns.snapshot_columns(async_session, world.eval_view, [column], limits)
    world.eval_view.column_snapshots = {**(world.eval_view.column_snapshots or {}), **entries}
    world.eval_view.rule_columns = {"rule_match": column, "rule_set_id": "s@1",
                                    "rule_set_digest": __import__("src.rule_signals", fromlist=["x"]).rule_set_digest(world.rule_set)}
    await async_session.commit()
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    # miStudio's verdicts are used: the same hand-computed rates as without the producer column
    assert (view["detectors"]["rules_alone"]["positives"]["flagged"], view["detectors"]["rules_alone"]["negatives"]["flagged"]) == (3, 1)
    agreement = view["agreement"]["rule_match"]
    assert agreement == {"both": 4, "neither": 0, "mistudio_only": 0, "producer_only": 8}
    assert view["agreement"]["used"] == "mistudio" and "producer_computed" not in view["reasons"]


async def test_pending_code_takes_the_producer_verdict_for_the_same_digest(client, async_session, data_dir, monkeypatch,
                                                                            request):
    from src.services import view_columns

    rng = rng_for(request.node.nodeid)
    code, expr, col = rf.ident(rng, "k"), rf.ident(rng, "e"), rf.ident(rng, "c")

    world = await rw.build(client, async_session, data_dir, request.node.nodeid, rule_set=None)
    doc = rf.document(world.key, rng, signals=[rf.code_rule_signal(code)], expressions={expr: {"signal": code}},
                      outputs=[rf.fired(col, expr)])
    made = await client.post(f"{API}/behaviours/{world.key}/versions",
                             json={"base_version": 1, "rule_set": {"document": doc}})
    assert made.status_code == 201, made.text
    import datasets

    import src.rule_signals as rule_signals

    split = datasets.load_from_disk(str(await _raw_path(async_session, world.eval_view)))
    producer = [i % 4 == 1 for i in range(len(split))]  # matches rows 1, 5, 9, … (evaluated rows 0, 2, 4, …)
    column = "producer_" + rf.ident(rng, "m")
    split = split.add_column(column, producer)
    path = data_dir / "datasets" / "with-producer-code"
    split.save_to_disk(str(path))
    await _repoint(async_session, world.eval_view, str(path))
    entries = await view_columns.snapshot_columns(async_session, world.eval_view, [column],
                                                  view_columns.ColumnLimits(max_values=10, max_columns=8))
    world.eval_view.column_snapshots = {**(world.eval_view.column_snapshots or {}), **entries}
    world.eval_view.rule_columns = {"rule_match": column, "rule_set_id": "s@1",
                                    "rule_set_digest": rule_signals.rule_set_digest(doc)}
    await async_session.commit()
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    assert "producer_computed" in view["reasons"]
    assert view["unknown"]["n"] == 0  # every pending row took the producer verdict
    flagged = view["detectors"]["rules_alone"]
    # evaluated rows 0, 2, 4 of the positives and 0, 2, 4 of the negatives (array order) match
    assert (flagged["positives"]["flagged"], flagged["negatives"]["flagged"]) == (3, 3)


async def _producer_column(async_session, world, data_dir, values, name):
    """Add a producer verdict column to the evaluated view's split and snapshot it (Review R1 helper)."""
    import datasets

    from src.services import view_columns

    split = datasets.load_from_disk(str(await _raw_path(async_session, world.eval_view)))
    split = split.add_column(name, values)
    path = data_dir / "datasets" / f"with-{name}"
    split.save_to_disk(str(path))
    await _repoint(async_session, world.eval_view, str(path))
    entries = await view_columns.snapshot_columns(async_session, world.eval_view, [name],
                                                  view_columns.ColumnLimits(max_values=10, max_columns=8))
    world.eval_view.column_snapshots = {**(world.eval_view.column_snapshots or {}), **entries}


async def test_pending_code_never_takes_a_producer_verdict_for_another_digest(client, async_session, data_dir,
                                                                              monkeypatch, request):
    """Review R1 (S3), FR-31: the exception is for EXACTLY the evaluated digest. A producer column written by
    another rule set (another digest) is compared at most, never used: the pending rows stay unknown."""
    rng = rng_for(request.node.nodeid)
    code, expr, col = rf.ident(rng, "k"), rf.ident(rng, "e"), rf.ident(rng, "c")
    world = await rw.build(client, async_session, data_dir, request.node.nodeid, rule_set=None)
    doc = rf.document(world.key, rng, signals=[rf.code_rule_signal(code)], expressions={expr: {"signal": code}},
                      outputs=[rf.fired(col, expr)])
    assert (await client.post(f"{API}/behaviours/{world.key}/versions",
                              json={"base_version": 1, "rule_set": {"document": doc}})).status_code == 201
    column = "producer_" + rf.ident(rng, "m")
    await _producer_column(async_session, world, data_dir, [i % 4 == 1 for i in range(24)], column)
    world.eval_view.rule_columns = {"rule_match": column, "rule_set_id": "s@1", "rule_set_digest": "f" * 64}
    await async_session.commit()
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    assert "producer_computed" not in view["reasons"] and view["agreement"]["used"] == "mistudio"
    assert view["agreement"]["rule_set_digest_matches"] is False
    assert view["unknown"]["n"] == 12 and view["unknown"]["by_reason"]["runner_unavailable"] == 12


@pytest.mark.parametrize("truthy,falsy", [(1, 0), ("true", "false")])
async def test_producer_verdicts_written_as_ints_or_strings_are_read(client, async_session, data_dir, monkeypatch,
                                                                     request, truthy, falsy):
    """Review R1 (S3), FR-34: a producer column holding 1/0 or "true"/"false" (not JSON booleans) is read as the
    verdicts it states; reading only `True` would count every row as producer-false."""
    import src.rule_signals as rule_signals

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    column = "producer_" + rf.ident(rng_for(request.node.nodeid), "m")
    await _producer_column(async_session, world, data_dir, [truthy] * 24, column)
    world.eval_view.rule_columns = {"rule_match": column, "rule_set_id": "s@1",
                                    "rule_set_digest": rule_signals.rule_set_digest(world.rule_set)}
    await async_session.commit()
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    assert view["agreement"]["rule_match"] == {"both": 4, "neither": 0, "mistudio_only": 0, "producer_only": 8}


async def test_pending_code_without_a_producer_is_unknown_and_counted(client, async_session, data_dir, monkeypatch,
                                                                       request):
    rng = rng_for(request.node.nodeid)
    code, expr, col = rf.ident(rng, "k"), rf.ident(rng, "e"), rf.ident(rng, "c")
    world = await rw.build(client, async_session, data_dir, request.node.nodeid, rule_set=None)
    doc = rf.document(world.key, rng, signals=[rf.code_rule_signal(code)], expressions={expr: {"signal": code}},
                      outputs=[rf.fired(col, expr)])
    assert (await client.post(f"{API}/behaviours/{world.key}/versions",
                              json={"base_version": 1, "rule_set": {"document": doc}})).status_code == 201
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    view = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    assert view["unknown"]["n"] == 12 and view["unknown"]["by_reason"]["runner_unavailable"] == 12
    assert view["unknown"]["over_share"] is True
    assert "rule_unknown_share" in view["reasons"] and "runner_unavailable" in view["reasons"]
    # unknown is no match and no veto: union equals probe
    assert view["detectors"]["rules_or_probe"] == view["detectors"]["probe_alone"]


async def _raw_path(session, view):
    from sqlalchemy import select

    from src.models.dataset import Dataset

    return (await session.execute(select(Dataset.raw_path).where(Dataset.id == view.dataset_id))).scalar_one()


async def _repoint(session, view, raw_path):
    from sqlalchemy import update

    from src.models.dataset import Dataset

    await session.execute(update(Dataset).where(Dataset.id == view.dataset_id).values(raw_path=raw_path))
    await session.flush()
