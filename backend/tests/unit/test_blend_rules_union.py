"""Rules ∪ blend candidates end to end (Feature 041 FR-8, FR-10, FR-17, FR-20; FTASKS 6.6, 9.5).

A real database, the real start route and the REAL task in-process, over 040's verdict cache seeded the way a
completed rules evaluation leaves it (`blend_world.seed_rule_verdicts`: by split position, keyed by view, set digest,
engine digest and reading). The rule set is a random-named masking set (`rule_set_fixtures`), so nothing here can
pass because a name matches `src/`.

* `rules_or_*` run on the cached verdicts: the union calibrated through 040's `union_cut`, its final verdict stored
  as 0/1 in its own `c<NN>` file, the rule match stored beside the rows (`rules__<view>.npy`);
* a missing cache is refused at the call (PT-5) naming the rules evaluation that fills it;
* a rule set with operator code and no runner refuses the rules ∪ candidates with 040's `runner_unavailable`;
* 040's success report judges `rules_or_blend` on the report's rules ∪ candidate and `blend_alone` on its base,
  by the SAME function the report used (FR-20), and a stale report is refused;
* the report reads stale when the rule engine moves (040 FR-38).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.core import runtime_settings
from src.core.config import settings
from src.services import probe_breakdown, view_columns
from src.services.probe_blend.report_format import round_floats
from tests.support import blend_world as bw
from tests.support import rule_set_fixtures as rf
from tests.support.next_calls import assert_next_calls_are_live

API = "/api/v1/probe-monitors"
CANDIDATES = ["stack", "rules_or_stack", "any_of", "rules_or_any_of"]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


async def rules_world(client, async_session, request, **kwargs):
    seed = request.node.nodeid
    return await bw.build(client, async_session, seed=seed, rule_set=lambda key: rf.masking_set(seed, key), **kwargs)


async def report_of(client, report_id):
    return json.loads((await client.get(f"{API}/blend-reports/{report_id}/report.json")).content)


async def test_rules_or_candidates_run_on_the_cached_verdicts(client, async_session, request, monkeypatch, data_dir):
    world = await rules_world(client, async_session, request)
    seeded = await bw.seed_rule_verdicts(async_session, world, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world, candidates=CANDIDATES)
    report = await report_of(client, report_id)
    by = {c["name"]: c for c in report["candidates"]}
    assert list(by) == ["stack", "any_of", "rules_or_stack", "rules_or_any_of"]      # the fixed registry order
    union = by["rules_or_stack"]
    assert union["calibration"]["source"] == "union_calibration"
    assert union["calibration"]["realised_fpr"] <= 0.05                       # the world's target
    assert union["params"]["rules"] == {k: seeded[k] for k in ("rule_set_digest", "engine_digest")}
    assert report["behaviour"]["rule_set"]["engine_digest"] == seeded["engine_digest"]
    rows = data_dir / "probe_monitors" / "blends" / report_id
    cal = world.calibration.id
    positions = np.load(rows / f"positions__{cal}.npy")
    labels = np.load(rows / f"labels__{cal}.npy")
    final = np.load(rows / f"c{union['index']:02d}__{cal}.npy")
    assert set(np.unique(final).tolist()) <= {0.0, 1.0}
    assert final[labels == 0].mean() == pytest.approx(union["calibration"]["realised_fpr"], abs=1e-6)
    # ⚠ by position: the stored rule match is the seeded cache read at each aligned row's split position
    assert np.array_equal(np.load(rows / f"rules__{cal}.npy").astype(bool), seeded["match"][cal][positions])
    # every rule match is flagged by the union, on every test view
    for test in world.tests:
        seen = np.load(rows / f"rules__{test.id}.npy").astype(bool)
        assert np.all(np.load(rows / f"c{union['index']:02d}__{test.id}.npy")[seen] == 1.0)
    # the verdict kinds still share one n_fired array; the rules ∪ verdict kind has its own file
    assert (rows / f"nfired__{cal}.npy").exists() and (rows / f"c{by['rules_or_any_of']['index']:02d}__{cal}.npy").exists()


async def test_default_candidates_include_the_rules_union(client, async_session, request, monkeypatch):
    world = await rules_world(client, async_session, request)
    await bw.seed_rule_verdicts(async_session, world, seed=request.node.nodeid)
    monkeypatch.setattr("src.workers.probe_blend_tasks.run_blend_evaluation.delay", lambda rid: type("T", (), {"id": "t"})())
    started = await client.post(f"{API}/blend-reports", json=bw.start_body(world))
    assert started.status_code == 202, started.text
    names = started.json()["candidates"]
    assert "rules_or_stack" in names and names.index("rules_or_stack") > names.index("best_single")


async def test_missing_rule_verdicts_refused_naming_the_rules_evaluation(client, async_session, request, monkeypatch):
    world = await rules_world(client, async_session, request)
    await bw.seed_rule_verdicts(async_session, world, seed=request.node.nodeid, views=[world.blend_fit])
    sent = []
    monkeypatch.setattr("src.workers.probe_blend_tasks.run_blend_evaluation.delay",
                        lambda rid: sent.append(rid) or type("T", (), {"id": "t"})())
    response = await client.post(f"{API}/blend-reports", json=bw.start_body(world, candidates=["rules_or_stack"]))
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "rule_verdicts_missing"
    assert sorted(detail["views"]) == sorted([*(v.id for v in world.tests), world.calibration.id])
    call = detail["next"][0]
    assert call["tool"] == "evaluate_detector_with_rules" and sorted(call["arguments"]["view_ids"]) == sorted(detail["views"])
    assert_next_calls_are_live(detail["next"])
    assert sent == []                                                 # nothing queued for the refused call
    # probe-only candidates still run without the cache
    ok = await client.post(f"{API}/blend-reports", json=bw.start_body(world, candidates=["stack"]))
    assert ok.status_code == 202, ok.text


async def test_operator_code_without_a_runner_refuses_only_the_rules_union(client, async_session, request, monkeypatch):
    from tests.support import code_rule_world as cw

    seed = request.node.nodeid
    world = await bw.build(client, async_session, seed=seed, rule_set=lambda key: cw.code_set(seed, key)[0])
    monkeypatch.setattr("src.workers.probe_blend_tasks.run_blend_evaluation.delay", lambda rid: type("T", (), {"id": "t"})())
    response = await client.post(f"{API}/blend-reports", json=bw.start_body(world, candidates=["rules_or_max"]))
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "runner_unavailable"
    ok = await client.post(f"{API}/blend-reports", json=bw.start_body(world, candidates=["max"]))
    assert ok.status_code == 202, ok.text


async def test_breakdown_of_a_rules_candidate_equals_report_bytes(client, async_session, request, monkeypatch, data_dir):
    world = await rules_world(client, async_session, request)
    ood = world.tests[0]
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 3, 5000).astype(np.int32)
    entry = view_columns._write_column(view_columns.view_dir(ood.id), "source", codes, ["a", "b", "c"])
    ood.group_columns, ood.column_snapshots = ["source"], {"source": entry}
    await async_session.commit()
    await bw.seed_rule_verdicts(async_session, world, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world, candidates=CANDIDATES)
    report = await report_of(client, report_id)
    union = next(c for c in report["candidates"] if c["name"] == "rules_or_stack")
    rows = data_dir / "probe_monitors" / "blends" / report_id
    positions = np.load(rows / f"positions__{ood.id}.npy")
    labels = np.load(rows / f"labels__{ood.id}.npy")
    values = np.load(rows / f"c{union['index']:02d}__{ood.id}.npy")
    again = probe_breakdown.groups_of(codes[positions], labels == 1, values >= 0.5, values,
                                      np.ones(values.shape[0], bool), ["a", "b", "c"], 30)
    assert round_floats(again) == union["tests"][ood.id]["breakdown"]["source"]


def union_criteria():
    return [{"id": "crit-union", "detector": "rules_or_blend", "subset": "all", "min_recall": 0.3},
            {"id": "crit-blind", "detector": "blend_alone", "subset": "rule_invisible_positives", "min_recall": 0.3}]


async def success(client, world, candidate, report_id):
    return await client.post(f"{API}/behaviours/{world.key}/success-reports",
                             json={"model_id": world.model.id, "probe_id": world.members[0].id,
                                   "blend_report_id": report_id, "candidate": candidate})


async def test_success_report_judges_rules_or_blend_on_the_union_candidate(client, async_session, request, monkeypatch):
    """FR-20: named `stack`, the `rules_or_blend` criterion is judged on the report's `rules_or_stack` and the
    `blend_alone` criterion on `stack` — each equal to the report's own `criteria` entry (the same function).
    Control: delete the rules ∪ branch from `_judge_criterion` → the probe is judged instead (no rules
    evaluation of it exists) → red."""
    world = await rules_world(client, async_session, request, criteria=union_criteria())
    await bw.seed_rule_verdicts(async_session, world, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world, candidates=CANDIDATES)
    report = await report_of(client, report_id)
    own = {c["name"]: {k["id"]: k for k in c["criteria"]} for c in report["candidates"]}
    for named in ("stack", "rules_or_stack"):
        made = await success(client, world, named, report_id)
        assert made.status_code == 200, made.text
        by = {c["id"]: c for c in made.json()["report"]["criteria"]}
        union, blind = by["crit-union"], by["crit-blind"]
        assert union["blend"]["candidate"] == "rules_or_stack" and blind["blend"]["candidate"] == "stack"
        mine = own["rules_or_stack"]["crit-union"]
        assert (union["verdict"], round_floats(union["figures"])) == (mine["verdict"], mine["figures"])
        assert union["verdict"] in ("met", "not_met")
        assert union["union_calibration"]["source"] == "union_calibration"
        base = own["stack"]["crit-blind"]
        assert (blind["verdict"], round_floats(blind["figures"])) == (base["verdict"], base["figures"])


async def test_rule_invisible_positives_exclude_what_the_rules_see(client, async_session, request, monkeypatch, data_dir):
    world = await rules_world(client, async_session, request, criteria=union_criteria())
    await bw.seed_rule_verdicts(async_session, world, seed=request.node.nodeid, positive_share=0.5)
    report_id = await bw.done_report(client, monkeypatch, world, candidates=["stack", "rules_or_stack"])
    report = await report_of(client, report_id)
    blind = next(c for c in report["candidates"] if c["name"] == "stack")["criteria"][0]
    rows = data_dir / "probe_monitors" / "blends" / report_id
    expected = 0
    for view_id in blind["test_view_ids"]:
        labels = np.load(rows / f"labels__{view_id}.npy")
        seen = np.load(rows / f"rules__{view_id}.npy").astype(bool)
        expected += int(((labels == 1) & ~seen).sum())
    assert blind["figures"]["positives"] == expected > 0


async def test_stale_rules_candidate_refused(client, async_session, request, monkeypatch):
    world = await rules_world(client, async_session, request, criteria=union_criteria())
    await bw.seed_rule_verdicts(async_session, world, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world, candidates=CANDIDATES)
    world.members[1].decision_digest = "9" * 64
    await async_session.commit()
    response = await success(client, world, "rules_or_stack", report_id)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "blend_report_stale"


async def test_a_moved_rule_engine_makes_the_report_stale(client, async_session, request, monkeypatch):
    import src.rule_signals as rule_signals

    world = await rules_world(client, async_session, request)
    await bw.seed_rule_verdicts(async_session, world, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world, candidates=["rules_or_stack"])
    before = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert before["stale"] is False, before["stale_reasons"]
    real = rule_signals.vector_digests
    monkeypatch.setattr(rule_signals, "vector_digests", lambda: {**real(), "rule-signal-cases-v1.json": "1" * 64})
    after = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert [r["code"] for r in after["stale_reasons"]] == ["rules_engine_changed"]
