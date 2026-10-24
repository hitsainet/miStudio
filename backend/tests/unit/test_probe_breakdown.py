"""The breakdown: recall at the deployed decision, by subgroup, from stored arrays (040 FR-15 – FR-18; 8.x).

Through the REAL route. The fixture's evaluation positions are NON-CONTIGUOUS (every other split row,
offset), so joining by array index instead of by position gives different groups; one group has 12 rows
(below the 30-row floor); the probe has length bands, a window bar and window bands that all differ from
its global bar. Every expected figure is computed in the test from the raw arrays with plain loops.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.config import settings
from src.services.probe_monitor_metrics import deployed_decision, wilson_interval
from src.services.probe_monitor_trainer import calibrate
from tests.support.detector_fixtures import (
    Arrays,
    dataset_row,
    declare,
    evaluation_row,
    model_row,
    run_and_probe,
    view_row,
    write_split,
)
from tests.support.next_calls import assert_next_calls_are_live
from tests.support.random_behaviour import rng_for, slug

API = "/api/v1/probe-monitors"
N_SPLIT = 200
PB = [{"min_tokens": 0, "max_tokens": 20, "threshold": 0.3}, {"min_tokens": 21, "max_tokens": None, "threshold": 0.8}]
WINDOW = {"all": {"threshold": 0.9, "length_bands": None, "target_fpr": 0.01}}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


async def world(session, data_dir, rng, *, scope="all", window_decisions=None, bands=PB, history=None,
                calibration=True, with_positions=True):
    values = [slug(rng, words=1) for _ in range(3)]
    # kind: values[2] on exactly the split rows that make a 12-row group among the evaluated rows
    kinds = []
    for i in range(N_SPLIT):
        kinds.append(values[2] if (i % 2 == 1 and i < 25) else values[(i // 2) % 2])
    ds = await dataset_row(session, write_split(data_dir, {"text": [str(i) for i in range(N_SPLIT)],
                                                          "label": [str(i % 2) for i in range(N_SPLIT)],
                                                          "kind": kinds}))
    model = await model_row(session)
    train = await view_row(session, ds, role="train")
    test = await view_row(session, ds, role="eval", distribution="out_of_distribution")
    cal = await view_row(session, ds, role="calibration") if calibration else None
    await declare(session, test, ["kind"])
    gen = np.random.default_rng(11)
    cal_scores = gen.normal(size=500) if calibration else None
    run, probe = await run_and_probe(session, model, train, eval_views=[test], calibration_view=cal, scope=scope,
                                     threshold=0.5, length_bands=bands, window_decisions=window_decisions,
                                     target_fpr=0.02, calibration_scores=cal_scores, history=history)
    positions = np.arange(1, N_SPLIT, 2)[:90]                     # 1, 3, 5, … — gaps by construction
    m = len(positions)
    labels = (gen.random(m) < 0.5).astype(np.int8)
    scores = gen.normal(size=m) + labels
    n_scored = gen.integers(1, 60, size=m)
    evaluation = await evaluation_row(session, probe, run, test, Arrays(scores, positions, n_scored, labels),
                                      with_positions=with_positions)
    await session.commit()
    return {"probe": probe, "run": run, "evaluation": evaluation, "view": test, "kinds": kinds, "values": values,
            "scores": scores, "positions": positions, "n_scored": n_scored, "labels": labels, "cal": cal_scores}


def expected(w, thresholds):
    """Per-value counts from the raw arrays, joined BY POSITION, with plain loops."""
    out = {}
    for i, position in enumerate(w["positions"].tolist()):
        value = w["kinds"][position]
        g = out.setdefault(value, {"p": 0, "pf": 0, "n": 0, "nf": 0})
        threshold = thresholds[i]
        fired = threshold is not None and float(np.float32(w["scores"][i])) >= threshold
        if w["labels"][i] == 1:
            g["p"] += 1
            g["pf"] += int(fired)
        else:
            g["n"] += 1
            g["nf"] += int(fired)
    return out


def check(body, exp, min_rows=30):
    got = {g["value"]: g for g in body["groups"]}
    assert set(got) == set(exp)
    for value, e in exp.items():
        g = got[value]
        assert (g["positives"]["n"], g["positives"]["flagged"]) == (e["p"], e["pf"])
        assert (g["negatives"]["n"], g["negatives"]["flagged"]) == (e["n"], e["nf"])
        assert g["positives"]["recall"] == pytest.approx(e["pf"] / e["p"] if e["p"] else None, abs=1e-6) if e["p"] else g["positives"]["recall"] is None
        if e["p"]:
            lo, hi = wilson_interval(e["pf"], e["p"])
            assert g["positives"]["ci"] == [pytest.approx(lo, abs=1e-6), pytest.approx(hi, abs=1e-6)]
        assert g["too_small"] == (e["p"] + e["n"] < min_rows)


def url(w):
    return f"{API}/probes/{w['probe'].id}/evaluations/{w['evaluation'].id}/breakdown"


async def test_recorded_decision_by_position_with_a_small_group_shown(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid))
    response = await client.get(url(w), params={"by_column": "kind"})
    assert response.status_code == 200, response.text
    body = response.json()
    thresholds = [deployed_decision(0.5, None, PB, "all", "all", int(n)).threshold for n in w["n_scored"]]
    exp = expected(w, thresholds)
    check(body, exp)
    small = [g for g in body["groups"] if g["too_small"]]
    assert [g["rows"] for g in small] == [12]                       # present and flagged, not dropped
    assert body["decision"]["rule_counts"]["probe_band"] == len(w["positions"])
    assert body["decision"]["window"] == "all" and body["decision"]["scope"] == "all"


async def test_a_window_bar_and_bands_apply_to_an_all_run(client, async_session, data_dir, request):
    """The `all` window has its own bar (0.9): with no window bands, an `all` probe's bands refine it."""
    w = await world(async_session, data_dir, rng_for(request.node.nodeid), window_decisions=WINDOW)
    body = (await client.get(url(w), params={"by_column": "kind"})).json()
    thresholds = [deployed_decision(0.5, WINDOW["all"], PB, "all", "all", int(n)).threshold for n in w["n_scored"]]
    check(body, expected(w, thresholds))


async def test_an_assistant_run_has_no_window_and_uses_probe_bands(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid), scope="assistant", window_decisions=WINDOW)
    body = (await client.get(url(w), params={"by_column": "kind"})).json()
    assert body["decision"]["window"] is None
    thresholds = [deployed_decision(0.5, None, PB, "response", None, int(n)).threshold for n in w["n_scored"]]
    check(body, expected(w, thresholds))


RESPONSE = {"response": {"threshold": 0.9, "target_fpr": 0.01,
                          "length_bands": [{"min_tokens": 0, "max_tokens": 30, "threshold": 0.2},
                                           {"min_tokens": 31, "max_tokens": None, "threshold": 0.7}]}}


async def test_a_last_assistant_run_is_judged_over_the_response_window(client, async_session, data_dir, request):
    """Review R1 (040 S1-S2): every other test here is scoped `all` or has no window, where the run's scope and
    the window entry cannot move a bar, so passing a constant scope or dropping the entry survived. A
    `last_assistant` run reads the `response` window, whose own bands apply."""
    w = await world(async_session, data_dir, rng_for(request.node.nodeid), scope="last_assistant", window_decisions=RESPONSE)
    body = (await client.get(url(w), params={"by_column": "kind"})).json()
    assert body["decision"]["window"] == "response"
    thresholds = [deployed_decision(0.5, RESPONSE["response"], PB, "response", "response", int(n)).threshold
                  for n in w["n_scored"]]
    check(body, expected(w, thresholds))
    assert body["decision"]["rule_counts"]["window_band"] == len(w["positions"])


async def test_a_filter_and_a_min_rows_override(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid))
    value = w["values"][0]
    body = (await client.get(url(w), params={"by_column": "kind", "filter": [f"kind={value}"], "min_rows": 1000})).json()
    assert [g["value"] for g in body["groups"]] == [value]
    assert body["groups"][0]["too_small"] is True and body["min_rows"] == 1000
    nothing = (await client.get(url(w), params={"by_column": "kind", "filter": ["kind=" + slug(rng_for("x"))]})).json()
    assert nothing["groups"] == [] and nothing["total"] == 0          # an unknown value is an empty group


async def test_pages_concatenate_to_every_group(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid))
    whole = (await client.get(url(w), params={"by_column": "kind"})).json()
    first = (await client.get(url(w), params={"by_column": "kind", "limit": 2})).json()
    assert first["next_cursor"] and first["total"] == 3
    second = (await client.get(url(w), params={"by_column": "kind", "limit": 2, "cursor": first["next_cursor"]})).json()
    assert first["groups"] + second["groups"] == whole["groups"] and second["next_cursor"] is None


async def test_the_same_request_gives_the_same_bytes(client, async_session, data_dir, request):
    from src.services import probe_breakdown

    w = await world(async_session, data_dir, rng_for(request.node.nodeid))
    a = (await client.get(url(w), params={"by_column": "kind"})).content
    probe_breakdown._memo.clear()                                     # recomputed, not memoised
    b = (await client.get(url(w), params={"by_column": "kind"})).content
    assert a == b


async def test_the_revision_is_named_and_moves_with_the_decision(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid),
                    history=[{"revision": 1}, {"revision": 2}])
    body = (await client.get(url(w), params={"by_column": "kind"})).json()
    assert body["decision"]["revision"] == 2
    w["probe"].calibration_history = [{"revision": 1}, {"revision": 2}, {"revision": 3}]
    await async_session.commit()
    again = (await client.get(url(w), params={"by_column": "kind"})).json()
    assert again["decision"]["revision"] == 3 and again["inputs_digest"] != body["inputs_digest"]


async def test_target_fpr_is_cut_on_the_calibration_negatives(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid))
    body = (await client.get(url(w), params={"by_column": "kind", "mode": "target_fpr", "target_fpr": 0.05})).json()
    cut = calibrate([float(x) for x in np.asarray(w["cal"], dtype=np.float32)], target_fpr=0.05, source="calibration_set")
    assert body["decision"]["threshold"] == pytest.approx(cut.threshold, abs=1e-6)
    check(body, expected(w, [cut.threshold] * len(w["positions"])))
    assert body["decision"]["rule_counts"] is None


async def test_target_fpr_without_a_calibration_view_is_refused(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid), calibration=False)
    response = await client.get(url(w), params={"by_column": "kind", "mode": "target_fpr", "target_fpr": 0.05})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "no_calibration_view"


async def test_a_column_not_declared_is_refused_with_the_declaration_as_next(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid))
    column = slug(rng_for(request.node.nodeid + "c"))
    response = await client.get(url(w), params={"by_column": column})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert (detail["code"], detail["column"]) == ("column_not_declared", column)
    assert detail["next"][0]["arguments"]["group_columns"] == ["kind", column]
    assert_next_calls_are_live(detail["next"])


async def test_a_pre_035_evaluation_is_refused_positions_not_recorded(client, async_session, data_dir, request):
    """035's refusal (`probe_monitors.py`), held by the breakdown too: never joined by index."""
    w = await world(async_session, data_dir, rng_for(request.node.nodeid), with_positions=False)
    response = await client.get(url(w), params={"by_column": "kind"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "positions_not_recorded"
    assert detail["next"][0]["tool"] == "evaluate_probe_monitor"
    assert_next_calls_are_live(detail["next"])


async def test_another_probes_evaluation_is_not_found(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid))
    response = await client.get(f"{API}/probes/pm_other/evaluations/{w['evaluation'].id}/breakdown", params={"by_column": "kind"})
    assert response.status_code == 404


async def test_the_memo_key_moves_with_every_setting_read(client, async_session, data_dir, request):
    from src.core import runtime_settings

    w = await world(async_session, data_dir, rng_for(request.node.nodeid))
    base = (await client.get(url(w), params={"by_column": "kind"})).json()["inputs_digest"]
    for key, value in (("breakdown_min_subgroup_rows", 7), ("calibration_floor_negatives", 3),
                       ("breakdown_page_default", 9), ("breakdown_max_filters", 2), ("breakdown_page_max", 400)):
        try:
            runtime_settings.apply({key: value})
            moved = (await client.get(url(w), params={"by_column": "kind"})).json()
        finally:
            runtime_settings.reset()
        assert moved["inputs_digest"] != base, key
        assert moved["settings"][key] == value


async def test_the_calibration_sentence(client, async_session, data_dir, request):
    w = await world(async_session, data_dir, rng_for(request.node.nodeid))
    body = (await client.get(url(w), params={"by_column": "kind"})).json()
    cal = body["calibration"]
    assert (cal["n_negatives"], cal["n_negatives_source"], cal["allowed_false_positives"]) == (500, "array_length", 10)
    assert cal["sentence"] == "2% of 500 = 10 rows" and cal["few_negatives"] is True and cal["floor_source"] == "setting"


async def test_two_behaviours_concurrently_give_their_sequential_bytes(async_session, data_dir, request):
    """FR-11: two random behaviours' breakdowns, interleaved on threads, give the bytes they give in
    sequence — the memo is keyed by the full input digest, never by anything a second behaviour shares."""
    from pathlib import Path

    from src.services import probe_breakdown
    from tests.support.concurrency import assert_concurrent_equals_sequential

    worlds = [await world(async_session, data_dir, rng_for(request.node.nodeid + str(i))) for i in range(2)]
    settings_ = probe_breakdown.BreakdownSettings(min_subgroup_rows=30, page_default=100, page_max=500,
                                                 max_filters=8, calibration_floor=2000)
    probe_breakdown._memo.clear()

    def run(w):
        return probe_breakdown.compute(probe=w["probe"], run=w["run"], evaluation=w["evaluation"], view=w["view"],
                                       request=probe_breakdown.BreakdownRequest(by_column="kind"), settings=settings_,
                                       root=Path(data_dir) / "probe_monitors")

    assert run(worlds[0])["groups"] != run(worlds[1])["groups"]          # the two really differ
    assert_concurrent_equals_sequential(run, worlds)
