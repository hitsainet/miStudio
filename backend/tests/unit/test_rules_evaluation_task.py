"""Rules ∪ probe evaluations, end to end on real files (Feature 040 S3; FR-33 – FR-38; FTASKS 21.1 – 21.8).

Through the REAL route; the dispatch is recorded (payload and count) and the REAL task body then runs in-process.
Every figure is hand-computed from `tests/support/rules_world.py`'s table.

Pinned here:
* three detectors per row and their rates (FR-33), including a veto above the probe bar and a row where a
  match and a veto both fire;
* positions, never array indices: every even split row is a matching decoy;
* the duplicate request returns the existing row, 200 when completed (040-T29); a second identity on the
  same view reuses the cached verdicts with ZERO engine calls on it; a changed rule set recomputes (FR-37);
* the stored body is immutable: a rewrite is refused (FR-38), and moved inputs read `stale` naming them;
* the live Celery registry, the queue route, the beat entry and the cancellation scope (wiring that fails
  when the registration line is removed);
* cancellation stops between chunks and leaves `cancelled`; `get_task_status`'s `task_queue` row follows;
* no row text in any log record (`caplog` sentinel).
"""
from __future__ import annotations

import json
import logging

import pytest
from sqlalchemy import select

from src.core.config import settings
from tests.support import rules_world as rw
from tests.support.next_calls import assert_next_calls_are_live

API = rw.API


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def _rates(view, name):
    det = view["detectors"][name]
    return det["positives"]["flagged"], det["positives"]["n"], det["negatives"]["flagged"], det["negatives"]["n"]


async def test_three_detectors_hand_computed(client, async_session, data_dir, monkeypatch, request):
    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert out["outcome"]["status"] == "completed", out
    read = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()
    view = read["result"]["views"][0]
    assert _rates(view, "rules_alone") == (3, 6, 1, 6)
    assert _rates(view, "probe_alone") == (3, 6, 2, 6)
    assert _rates(view, "rules_or_probe") == (4, 6, 2, 6)
    assert view["detectors"]["rules_or_probe"]["positives"]["recall"] == pytest.approx(4 / 6, abs=1e-6)
    assert view["unknown"]["n"] == 0 and view["reasons"] == []
    # TI-25's reading is recorded (Review R1): the set's one `fired` output names the match expression
    assert read["result"]["match_expressions"] == [world.rule_set["outputs"][0]["value"]["fired"]]
    # per declared group: the two groups split the evaluated rows by position parity of their index
    groups = view["groups"][world.group_column]
    assert sum(g["rows"] for g in groups) == 12
    assert read["stale"] == {"stale": False, "moved": [], "next": []}


async def test_the_task_queue_row_is_polled_by_get_task_status(client, async_session, data_dir, monkeypatch, request):
    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    task = (await client.get(f"/api/v1/task-queue/{out['task_queue_id']}")).json()["data"]
    assert (task["entity_id"], task["entity_type"], task["task_type"], task["status"]) == (
        out["evaluation_id"], "probe_rules_evaluation", "rules_evaluation", "completed")
    # Review R1 (S3): the `running` mirror stamps started_at (get_task_status shows when it began)
    assert task.get("started_at"), task
    assert_next_calls_are_live(out["next"])


async def test_a_duplicate_returns_the_existing_row(client, async_session, data_dir, monkeypatch, request):
    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    first = await rw.run_one(client, monkeypatch, world, calibration=False)
    again = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert again["evaluation_id"] == first["evaluation_id"] and again["duplicate"] is True
    assert again["dispatched"] == [], "a duplicate must not queue a second task"


async def test_a_veto_condition_read_as_a_match_is_a_reason(client, async_session, data_dir, monkeypatch, request):
    """Review R1 (S3), TI-25: a set exposing its veto condition as a `fired` column has it read as a match; the
    stored result says so in its reasons (the operator question stays open)."""
    from tests.support import rule_set_fixtures as rf

    doc = rf.masking_set(request.node.nodeid)
    when = doc["vetoes"][0]["when"]
    doc["outputs"] = doc["outputs"] + [{"column": "masked_col", "type": "bool", "value": {"fired": when}}]
    doc["cases"] = []
    world = await rw.build(client, async_session, data_dir, request.node.nodeid, rule_set=doc)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    body = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]
    assert "match_includes_veto_condition" in body["reasons"] and when in body["match_expressions"]


async def test_a_row_over_the_text_budget_is_counted_as_budget_not_pending(client, async_session, data_dir,
                                                                            monkeypatch, request):
    """Review R1 (S3), FR-35: an over-budget row is unknown with reason `rule_budget_exceeded` — not
    `runner_unavailable`, which would name operator code the set does not have."""
    from src.core import runtime_settings

    long_plain = "nothing to see in these plain words " * 4  # 144 code points, over the floor of 100
    world = await rw.build(client, async_session, data_dir, request.node.nodeid, text={**rw.TEXT, "P": long_plain})
    runtime_settings.apply({"rule_text_max_code_points": 100})
    try:
        out = await rw.run_one(client, monkeypatch, world, calibration=False)
    finally:
        runtime_settings.reset()
    view = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    assert view["unknown"]["by_reason"] == {"rule_budget_exceeded": 6, "runner_unavailable": 0}
    assert "runner_unavailable" not in view["reasons"]


async def test_a_set_without_vetoes_says_so_and_carries_no_veto_note(client, async_session, data_dir, monkeypatch,
                                                                    request):
    """Review R1 (S3), FR-33: `no_veto_signals` is a stated reason, and the per-row veto note appears only when the
    set has vetoes (the masking set's result carries it)."""
    from tests.support import rule_set_fixtures as rf

    plain = await rw.build(client, async_session, data_dir, request.node.nodeid + "-plain",
                           rule_set=rf.masking_set(request.node.nodeid + "-plain", vetoes=False))
    body = (await client.get(f"{API}/rules-evaluations/"
                             f"{(await rw.run_one(client, monkeypatch, plain, calibration=False))['evaluation_id']}")).json()["result"]
    assert "no_veto_signals" in body["reasons"] and body["veto_note"] is None
    masked = await rw.build(client, async_session, data_dir, request.node.nodeid + "-masked")
    body = (await client.get(f"{API}/rules-evaluations/"
                             f"{(await rw.run_one(client, monkeypatch, masked, calibration=False))['evaluation_id']}")).json()["result"]
    assert "no_veto_signals" not in body["reasons"] and "per row" in body["veto_note"]


async def test_a_redelivered_task_for_a_finished_row_does_nothing(client, async_session, data_dir, monkeypatch, request):
    """Review R1 (S3), memory `cancel-must-not-be-its-own-restart`: an acks-late redelivery of a completed
    evaluation returns at once — no recompute, no status change, the body untouched."""
    import asyncio

    import src.rule_signals as rule_signals
    from src.workers.rules_evaluation_tasks import run_rules_evaluation

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    first = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()
    calls = {"n": 0}
    real = rule_signals.evaluate
    monkeypatch.setattr(rule_signals, "evaluate", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or real(*a, **k))
    again = await asyncio.to_thread(run_rules_evaluation, out["evaluation_id"])
    assert again == {"evaluation_id": out["evaluation_id"], "status": "completed"} and calls["n"] == 0
    second = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()
    assert (second["status"], second["body_sha256"], second["completed_at"]) == (
        "completed", first["body_sha256"], first["completed_at"])


async def test_a_broker_outage_leaves_no_pending_ghost(client, async_session, data_dir, monkeypatch, request):
    """Review R1 (S3): a dispatch that raises fails the row (and its `task_queue` row) and answers 503
    `broker_unavailable`. Left `pending`, the row held the partial-unique identity forever — every identical
    request came back as an in-flight duplicate that never ran — and the reaper reaps `running` rows only."""
    from src.models.rules_evaluation import ProbeRulesEvaluation
    from src.models.task_queue import TaskQueue
    from src.services import rules_evaluation

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)

    def down(rid):
        raise ConnectionError("broker down")

    monkeypatch.setattr(rules_evaluation, "_dispatch", down)
    response = await client.post(f"{API}/probes/{world.probe.id}/rules-evaluations",
                                 json={"view_ids": [world.eval_view.id], "calibration": False})
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "broker_unavailable"
    rows = (await async_session.execute(select(ProbeRulesEvaluation))).scalars().all()
    assert [(r.status, r.error["code"]) for r in rows] == [("failed", "broker_unavailable")]
    entry = await async_session.get(TaskQueue, rows[0].task_queue_id)
    await async_session.refresh(entry)
    assert entry.status == "failed"
    again = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert again["duplicate"] is False and again["evaluation_id"] != rows[0].id
    assert again["outcome"]["status"] == "completed"


async def test_verdicts_are_reused_and_a_new_rule_set_recomputes(client, async_session, data_dir, monkeypatch, request):
    import src.rule_signals as rule_signals

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    calls = {"n": 0}
    real = rule_signals.evaluate_many

    def counting(compiled, limits, inputs, **kwargs):  # rows through the engine (S4: one call per chunk of rows)
        calls["n"] += len(inputs)
        return real(compiled, limits, inputs, **kwargs)

    monkeypatch.setattr(rule_signals, "evaluate_many", counting)
    await rw.run_one(client, monkeypatch, world, calibration=False)
    eval_rows = 2 * len(rw.EVAL_ROWS)
    assert calls["n"] == eval_rows
    calls["n"] = 0
    second = await rw.run_one(client, monkeypatch, world, calibration=True)  # another identity, same view
    assert second["outcome"]["status"] == "completed"
    assert calls["n"] == 2 * rw.CAL_N, "the evaluated view's verdicts must come from the cache (zero engine calls)"
    # a changed rule set (another digest) recomputes the view
    changed = dict(world.rule_set, description="A changed rule set.")
    made = await client.post(f"{API}/behaviours/{world.key}/versions",
                             json={"base_version": 1, "rule_set": {"document": changed}})
    assert made.status_code == 201, made.text
    calls["n"] = 0
    third = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert third["evaluation_id"] != second["evaluation_id"] and calls["n"] == eval_rows


async def test_a_completed_body_is_never_rewritten_and_moved_inputs_read_stale(client, async_session, data_dir,
                                                                               monkeypatch, request):
    from src.models.rules_evaluation import CompletedResultIsImmutable, ProbeRulesEvaluation

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    row = (await async_session.execute(select(ProbeRulesEvaluation).where(
        ProbeRulesEvaluation.id == out["evaluation_id"]))).scalar_one()
    before = row.body
    row.body = before.replace("rules_alone", "rules_alonX")
    with pytest.raises(CompletedResultIsImmutable):
        await async_session.flush()
    await async_session.rollback()
    # re-vendored engine vectors: the result stays byte-identical and reads stale, naming the engine digest
    import src.rule_signals as rule_signals

    real = rule_signals.vector_digests
    monkeypatch.setattr(rule_signals, "vector_digests", lambda: {**real(), "rule-signal-cases-v1.json": "0" * 64})
    read = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()
    assert read["stale"]["stale"] is True
    assert [m["input"] for m in read["stale"]["moved"]] == ["engine_digest"]
    assert json.dumps(read["result"], sort_keys=True) == json.dumps(json.loads(before), sort_keys=True)
    assert_next_calls_are_live(read["stale"]["next"])


async def test_the_database_refuses_a_body_rewrite(async_session, client, data_dir, monkeypatch, request):
    from sqlalchemy import text

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    with pytest.raises(Exception, match="never rewritten"):
        await async_session.execute(text("UPDATE probe_rules_evaluations SET body = 'x' WHERE id = :i"),
                                    {"i": out["evaluation_id"]})
    await async_session.rollback()


async def test_no_rule_set_is_valid_and_union_is_the_probe(client, async_session, data_dir, monkeypatch, request):
    world = await rw.build(client, async_session, data_dir, request.node.nodeid, rule_set=None)
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert out["outcome"]["status"] == "completed", out
    view = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    assert view["reasons"] == ["no_rule_set"]
    assert view["detectors"]["rules_or_probe"] == view["detectors"]["probe_alone"]
    assert _rates(view, "rules_alone") == (0, 6, 0, 6)


async def test_calibration_without_positions_is_refused_with_the_recut(client, async_session, data_dir, monkeypatch,
                                                                        request):
    world = await rw.build(client, async_session, data_dir, request.node.nodeid, with_positions=False)
    response = await client.post(f"{API}/probes/{world.probe.id}/rules-evaluations",
                                 json={"view_ids": [world.eval_view.id], "calibration": True})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "calibration_positions_not_recorded"
    assert detail["next"][0]["arguments"]["require_positions"] is True
    assert detail["next"][0]["arguments"]["recut_windows"] is True
    assert_next_calls_are_live(detail["next"])


async def test_the_union_bar_on_the_calibration_negatives(client, async_session, data_dir, monkeypatch, request):
    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    out = await rw.run_one(client, monkeypatch, world, calibration=True)
    cal = (await client.get(f"{API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["calibration"]
    g = cal["global"]
    assert (g["budget"], g["rule_false_positives"], g["vetoed"], g["probe_false_positives_allowed"]) == (10, 4, 3, 6)
    assert g["realised_union_fpr"] == pytest.approx(0.01) and g["rule_share"] == pytest.approx(0.004)
    assert g["probe_false_positives"] == 6
    assert cal["aligned_by"] == "position"
    assert_next_calls_are_live(cal["next"])
    # REPORTED, NEVER WRITTEN (OQ-040-4 (a)): the probe's recorded bar is untouched.
    from sqlalchemy import select

    from src.models.probe_monitor import ProbeMonitor

    threshold = (await async_session.execute(select(ProbeMonitor.threshold).where(
        ProbeMonitor.id == world.probe.id).execution_options(populate_existing=True))).scalar_one()
    assert threshold == 0.5


async def test_no_text_reaches_a_log_record(client, async_session, data_dir, monkeypatch, request, caplog):
    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    caplog.set_level(logging.DEBUG)
    await rw.run_one(client, monkeypatch, world, calibration=True)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    for sentinel in (rw.rf.UNMASKED, rw.rf.MASKED, "plain words"):
        assert sentinel not in joined


# ── wiring read from the live registries ─────────────────────────────────────────────────────────


def test_the_task_is_in_the_live_registry_and_routed_to_rules():
    from src.core.celery_app import celery_app

    celery_app.loader.import_default_modules()
    name = "src.workers.rules_evaluation_tasks.run_rules_evaluation"
    assert name in celery_app.tasks
    assert celery_app.conf.task_routes[name] == {"queue": "rules"}
    assert celery_app.tasks[name].__wrapped__.__cooperative_cancel_scope__ == "probe_rules_evaluation"  # type: ignore[attr-defined]


def test_the_reaper_is_on_the_beat():
    from src.core.celery_app import celery_app

    celery_app.loader.import_default_modules()
    entry = celery_app.conf.beat_schedule["cleanup-stuck-rules-evaluations"]
    assert entry["task"] == "cleanup_stuck_rules_evaluations" and entry["task"] in celery_app.tasks
    assert entry["schedule"] == 300.0 and entry["options"]["queue"] == "low_priority"
    assert celery_app.conf.task_routes["cleanup_stuck_rules_evaluations"] == {"queue": "low_priority"}


def test_the_cancellation_scope_is_registered():
    from src.core import cancellation as C

    scope = C.get_scope("probe_rules_evaluation")
    assert scope.terminal_values == frozenset({"completed", "failed", "cancelled"})
    assert scope.missing_row == "cancelled"
    from src.models.rules_evaluation import ProbeRulesEvaluation

    assert scope.model() is ProbeRulesEvaluation


def test_routes_are_in_openapi():
    from src.main import app

    paths = app.openapi()["paths"]
    for path, method in (("/api/v1/probe-monitors/probes/{probe_id}/rules-evaluations", "post"),
                         ("/api/v1/probe-monitors/rules-evaluations/{evaluation_id}", "get"),
                         ("/api/v1/probe-monitors/behaviours/{key}/rules-evaluations", "get"),
                         ("/api/v1/probe-monitors/rules-evaluations/{evaluation_id}/cancel", "post")):
        assert method in paths.get(path, {}), path
