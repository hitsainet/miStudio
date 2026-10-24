"""A live judge run says how far it has got — rows, set, rate, ETA (operator, 2026-10-21).

⚠ WHY. "The probe run does not show progress." A judge run wrote `progress` once per SET, so a
two-set run read 0% for its whole first set: twenty minutes at ~2 judge calls a second with nothing
on the row. Now every row is counted (`judge_rows(on_row=...)`), a reading is written every
~5 s or 25 rows (whichever first) plus at every set boundary, and `progress_detail` carries rows
judged / planned, the set, a sliding-window rate, an ETA and the parse failures so far. The API and
the MCP tools return it.

Through the REAL `execute_judge_run`, a real database and the real routes wherever the claim is
about wiring; only the judge endpoint and the row reader are scripted.

NEGATIVE CONTROLS (each broke the named line, ran this file, saw red, restored from saved bytes
with a sha256 check) are recorded in 0xcc/reviews/live_progress_2026-10-21.md:
  L1  `on_row=tracker.row` dropped from the `judge_rows` call     → the mid-set reading tests
  L2  the row throttle always fires                               → the bounded-writes test
  L3  the progress write bypasses `record_progress`               → the cancel test
  L4  the unthrottled poll before completion removed              → the cancel test
  L5  the janitor's moved-progress branch reverted to `False`     → the janitor tests
  L6  `progress_detail` dropped from the judge-run serialiser     → the API and MCP tests
  L7  `progress_detail` dropped from `ProbeRunResponse`           → the API and MCP tests
  L8  `rows_judged` back to summing `entry["n"]`                  → the true-total test
  L9  the origin (threshold None) accepted as reachable           → the not-reachable test
"""
from __future__ import annotations

import math
import uuid
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.services.live_progress import (
    JUDGE_PROGRESS_ROWS,
    JudgeProgress,
    RateWindow,
    eta_seconds,
)

API = "/api/v1/probe-monitors"


# ── the tracker on its own (no database) ──────────────────────────────────────────────────────


class _Clock:
    def __init__(self, step: float = 0.0):
        self.now = 1000.0
        self.step = step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


def _tracker(plan, *, step=0.0, every_rows=25, every_seconds=5.0):
    writes = []
    tracker = JudgeProgress(
        lambda progress, detail: writes.append((progress, detail)) or True,
        plan,
        clock=_Clock(step),
        every_rows=every_rows,
        every_seconds=every_seconds,
    )
    return tracker, writes


class TestTheTracker:
    def test_the_denominator_is_the_PLAN_summed_over_every_set(self):
        tracker, writes = _tracker([("a", "US test", 685), ("b", "UK test", 685)])
        tracker.begin()
        assert tracker.rows_total == 1370
        assert writes[0][1]["rows_total"] == 1370 and writes[0][1]["rows_done"] == 0
        assert writes[0][1]["phase"] == "planned"

    def test_progress_is_rows_over_rows_across_sets_and_never_goes_back(self):
        tracker, writes = _tracker([("a", "A", 60), ("b", "B", 40)])
        tracker.begin()
        for index, size in enumerate((60, 40)):
            tracker.start_set(index)
            for _ in range(size):
                tracker.row(object())
            tracker.end_set()
        values = [progress for progress, _ in writes]
        assert values == sorted(values), values
        assert values[-1] == 100.0
        assert any(0 < value < 60 for value in values), "no reading inside the first set"

    def test_with_a_frozen_clock_it_writes_once_per_25_rows(self):
        tracker, writes = _tracker([("a", "A", 100)])
        tracker.begin()
        tracker.start_set(0)
        before = len(writes)
        for _ in range(100):
            tracker.row(object())
        assert len(writes) - before == 100 // JUDGE_PROGRESS_ROWS

    def test_with_a_slow_judge_it_writes_every_5_seconds_instead(self):
        """1 s per row: the time rule fires before the row rule — every 5 rows, not every 25."""
        tracker, writes = _tracker([("a", "A", 50)], step=1.0)
        tracker.begin()
        tracker.start_set(0)
        before = len(writes)
        for _ in range(50):
            tracker.row(object())
        assert 8 <= len(writes) - before <= 12, len(writes) - before

    def test_the_rate_and_eta_are_measured(self):
        """Half a second per row is 2 rows/s; 100 rows left is 50 s."""
        tracker, writes = _tracker([("a", "A", 200)], step=0.5)
        tracker.begin()
        tracker.start_set(0)
        for _ in range(100):
            tracker.row(object())
        detail = tracker.snapshot()
        assert detail["rate_per_second"] == pytest.approx(2.0, rel=0.05)
        assert detail["eta_seconds"] == pytest.approx(50, abs=3)

    def test_no_rate_is_None_never_a_guess(self):
        tracker, _ = _tracker([("a", "A", 10)])
        detail = tracker.snapshot()
        assert detail["rate_per_second"] is None and detail["eta_seconds"] is None

    def test_parse_failures_are_counted_as_they_happen(self):
        tracker, _ = _tracker([("a", "A", 10)])
        tracker.begin()
        tracker.start_set(0)
        for answer in (object(), None, object(), None):
            tracker.row(answer)
        assert tracker.snapshot()["parse_failures"] == 2

    def test_the_window_slides(self):
        window = RateWindow(window_seconds=10)
        for second in range(30):
            # 1 unit/s for 20 s, then 5 units/s.
            window.add(float(second), second if second < 20 else 20 + (second - 20) * 5)
        assert window.rate() == pytest.approx(5.0, rel=0.25)

    def test_eta_edges(self):
        assert eta_seconds(0, None) == 0.0
        assert eta_seconds(10, None) is None
        assert eta_seconds(10, 0.0) is None
        assert eta_seconds(10, 2.0) == 5.0


# ── the real run against a real database ──────────────────────────────────────────────────────


@pytest.fixture
def sync_db(async_engine):
    url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", "")
    )
    engine = create_engine(url)
    session = sessionmaker(bind=engine, autoflush=False)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def _prompt(db):
    from src.models.probe_judge_prompt import ProbeJudgePrompt
    from src.services import judge_prompt_library as library
    from src.services.judge_prompt_forms import content_hash

    content = dict(system_prompt="s answer", user_template="{text}",
                   answer_form=library.normalise_answer_form({"kind": "binary"}),
                   max_interaction_chars=8000)
    name = f"p-{uuid.uuid4().hex[:8]}"
    row = ProbeJudgePrompt(
        name=name, version=1, title="t", behaviour="pii",
        content_hash=content_hash(name=name, version=1, **content),
        source="operator", created_by="operator", **content,
    )
    db.add(row)
    db.commit()
    return row


def _views(db, names):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.probe_monitor import ProbeMonitorDataset

    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="local", status=DatasetStatus.READY,
                      raw_path="/nonexistent", extra_metadata={})
    db.add(dataset)
    db.flush()
    views = [
        ProbeMonitorDataset(name=name, dataset_id=dataset.id, input_column="i", label_column="l",
                            label_mapping={}, role="eval", counts={})
        for name in names
    ]
    db.add_all(views)
    db.commit()
    return views


def _run(db, prompt, views, **extra):
    from src.models.probe_monitor import ProbeMonitorJudgeRun

    run = ProbeMonitorJudgeRun(
        endpoint="http://127.0.0.1:9/v1", model="judge",
        prompt_version=f"{prompt.name}/v{prompt.version}", prompt_id=prompt.id,
        prompt_hash=prompt.content_hash, dataset_ids=[view.id for view in views],
        status="running", metrics={}, **extra,
    )
    db.add(run)
    db.commit()
    return run


class _Client:
    def __init__(self, reply, on_call=None):
        self.reply = reply
        self.calls = 0
        self.on_call = on_call
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.calls)
        content = self.reply(self.calls, kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


SIZES = {"US test": 60, "UK test": 40}


@pytest.fixture
def judge_env(monkeypatch, sync_db):
    """`execute_judge_run` with the rows and the endpoint scripted, everything else real; every
    `record_progress` call it makes is recorded with what it wrote and whether it landed."""
    from src.core import cancellation
    from src.services import probe_monitor_judge as judge
    from src.utils import millm_utils

    def rows_for(db, view, limit=None):
        size = SIZES[view.name]
        rows = [[{"role": "user", "content": f"{view.name} {i} {'a@b.c' if i % 2 else ''}"}]
                for i in range(size)]
        labels = [i % 2 for i in range(size)]
        if limit is not None:
            rows, labels = rows[:limit], labels[:limit]
        return rows, labels

    monkeypatch.setattr(judge, "_interactions_for", rows_for)
    monkeypatch.setattr(millm_utils, "ensure_model_loaded", lambda endpoint, model: None)

    recorded = []
    real = cancellation.record_progress

    def spy(kind, target_id, **kwargs):
        landed = real(kind, target_id, **kwargs)
        if kind == "probe_monitor_judge":
            recorded.append({"landed": landed, **kwargs})
        return landed

    monkeypatch.setattr(cancellation, "record_progress", spy)

    def install(client):
        monkeypatch.setattr(judge, "_client_for", lambda endpoint, key=None: client)

    def reply(_n, kwargs):
        return '{"answer": "yes"}' if "a@b.c" in kwargs["messages"][1]["content"] else '{"answer": "no"}'

    return SimpleNamespace(judge=judge, recorded=recorded, install=install, reply=reply)


def _readings(recorded):
    return [r for r in recorded if r.get("progress_detail") is not None]


class TestAJudgeRunReportsAsItGoes:
    def _complete(self, sync_db, judge_env, **extra):
        prompt = _prompt(sync_db)
        views = _views(sync_db, list(SIZES))
        run = _run(sync_db, prompt, views, **extra)
        client = _Client(judge_env.reply)
        judge_env.install(client)
        judge_env.judge.execute_judge_run(sync_db, run.id)
        sync_db.refresh(run)
        return run, client

    def test_progress_moves_INSIDE_a_set_not_only_between_sets(self, sync_db, judge_env):
        """The defect in one assertion: a reading part-way through the FIRST set."""
        self._complete(sync_db, judge_env)
        inside = [
            r for r in _readings(judge_env.recorded)
            if r["progress_detail"]["set_name"] == "US test"
            and 0 < r["progress_detail"]["set_rows_done"] < SIZES["US test"]
        ]
        assert inside, (
            "no reading landed while the first set was being judged, so a two-set run reads 0% "
            "for its whole first half"
        )
        assert all(r["landed"] for r in inside)

    def test_progress_is_monotone_and_ends_at_100(self, sync_db, judge_env):
        run, _ = self._complete(sync_db, judge_env)
        values = [r["progress"] for r in _readings(judge_env.recorded)]
        assert values == sorted(values), values
        done = [r["progress_detail"]["rows_done"] for r in _readings(judge_env.recorded)]
        assert done == sorted(done), done
        assert run.progress == 100.0

    def test_the_writes_are_BOUNDED_by_the_throttle(self, sync_db, judge_env):
        """At most: the plan, two per set boundary, one per 25 rows (time adds none in a fast
        test). One write per row would be 100+."""
        self._complete(sync_db, judge_env)
        total = sum(SIZES.values())
        bound = 1 + 2 * len(SIZES) + math.ceil(total / JUDGE_PROGRESS_ROWS) + len(SIZES)
        writes = len(_readings(judge_env.recorded))
        assert writes <= bound, f"{writes} progress writes for {total} rows (bound {bound})"
        assert writes >= 1 + 2 * len(SIZES) + 2, f"only {writes} writes — no per-row readings"

    def test_the_finished_row_says_what_it_did(self, sync_db, judge_env):
        run, client = self._complete(sync_db, judge_env)
        detail = run.progress_detail
        assert detail["kind"] == "judge" and detail["phase"] == "done"
        assert detail["rows_done"] == detail["rows_total"] == client.calls == 100
        assert detail["set_count"] == 2

    def test_rows_judged_is_the_TRUE_total(self, sync_db, judge_env):
        """⚠ It summed a key no per-set entry has and read 0 on every run — 0 on pmj_abc1e02877c0,
        which judged 2,520 rows."""
        run, client = self._complete(sync_db, judge_env)
        assert run.metrics["rows_judged"] == client.calls == 100

    def test_the_plan_is_min_of_the_cap_and_the_set(self, sync_db, judge_env):
        """`max_rows_per_set=50` over sets of 60 and 40 plans 50 + 40 = 90 — the bar's denominator is
        exactly the rows the judge is handed, fixed before the first one."""
        run, client = self._complete(sync_db, judge_env, max_rows_per_set=50)
        first = _readings(judge_env.recorded)[0]["progress_detail"]
        assert first["phase"] == "planned" and first["rows_total"] == 90
        assert client.calls == 90 and run.progress_detail["rows_done"] == 90
        set_totals = {
            r["progress_detail"]["set_name"]: r["progress_detail"]["set_rows_total"]
            for r in _readings(judge_env.recorded) if r["progress_detail"]["set_name"]
        }
        assert set_totals == {"US test": 50, "UK test": 40}

    def test_parse_failures_show_while_it_runs(self, sync_db, judge_env):
        def reply(n, kwargs):
            return "not json" if n in (3, 7) else judge_env.reply(n, kwargs)

        prompt = _prompt(sync_db)
        views = _views(sync_db, list(SIZES))
        run = _run(sync_db, prompt, views, parse_failure_limit=0.2)
        judge_env.install(_Client(reply))
        judge_env.judge.execute_judge_run(sync_db, run.id)
        seen = [r["progress_detail"]["parse_failures"] for r in _readings(judge_env.recorded)]
        assert 2 in seen[:-1], seen


class TestAJudgeThatCannotRunAtTheProbesRateSaysSo:
    """⚠ Seen live on pmj_abc1e02877c0: a yes/no judge has ONE operating point (4.5% and 3.7%
    false positives on its two sets), so at the probe's 1% the only threshold within budget is the
    ROC origin — "call nothing positive" — and the table read "recall 0 at 1%". That says the judge
    catches nothing; the truth is it cannot be run at that rate."""

    @staticmethod
    def _yes_no(n_pos=100, n_neg=100, caught=86, false_alarms=5):
        from src.services.judge_prompt_forms import ParsedAnswer

        yes = ParsedAnswer(value="yes", score=1.0, positive=True)
        no = ParsedAnswer(value="no", score=0.0, positive=False)
        answers = [yes] * caught + [no] * (n_pos - caught) + [yes] * false_alarms + [no] * (n_neg - false_alarms)
        labels = [1] * n_pos + [0] * n_neg
        return answers, labels

    def test_below_its_own_rate_it_is_NOT_REACHABLE_never_recall_0(self):
        from src.services.probe_monitor_judge import score_judge_answers

        answers, labels = self._yes_no()
        point = score_judge_answers(answers, labels, {"kind": "binary"}, target_fpr=0.01)["at_probe_fpr"]
        assert point["reachable"] is False
        assert point["recall"] is None, "an unreachable rate reported a recall"
        assert point["own_false_positive_rate"] == pytest.approx(0.05)
        assert point["own_recall"] == pytest.approx(0.86)
        assert point["reason"] == (
            "this judge can't run at the probe's 1% false-positive rate; at its own rate (5.0%) "
            "it catches 86%"
        )

    def test_at_or_above_its_own_rate_it_is_reachable_and_measured(self):
        from src.services.probe_monitor_judge import score_judge_answers

        answers, labels = self._yes_no()
        point = score_judge_answers(answers, labels, {"kind": "binary"}, target_fpr=0.10)["at_probe_fpr"]
        assert point["reachable"] is True
        assert point["recall"] == pytest.approx(0.86)
        assert point["realised_fpr"] == pytest.approx(0.05)

    def test_the_rung_does_not_read_it(self, sync_db):
        """Rung 3 counts a COMPLETED, SCORED judge run of the probe's behaviour; an unreachable
        operating point must neither lift nor sink it."""
        from src.services.probe_monitor_run import recompute_rung
        from tests.unit.test_judge_prompt_library import (
            _judge_run,
            _locked_prompt,
            _probe_with_ood_pass,
        )

        probe, ood = _probe_with_ood_pass(sync_db, behaviour="pii")
        run = _judge_run(sync_db, probe, ood, _locked_prompt(sync_db, behaviour="pii"))
        run.metrics = {**run.metrics, "per_set": {ood.id: {
            "scored": True,
            "at_probe_fpr": {"target_fpr": 0.01, "reachable": False, "recall": None},
        }}}
        sync_db.commit()
        assert recompute_rung(sync_db, probe.id)["rung"] == 3


class TestCancellationSurvivesTheThrottledWrites:
    def test_a_cancel_mid_run_STAYS_cancelled_and_no_later_reading_lands(self, sync_db, judge_env):
        """The operator cancels while row 25 is being judged — exactly when a throttled write is
        due. That write, every later one, and the completion must all leave CANCELLED alone."""
        from src.core.cancellation import OperatorCancelled

        prompt = _prompt(sync_db)
        views = _views(sync_db, list(SIZES))
        run = _run(sync_db, prompt, views)
        engine = sync_db.get_bind()

        def cancel_on(n):
            if n == 25:
                # What the API does, from another connection.
                with engine.begin() as connection:
                    connection.execute(
                        sa.text("UPDATE probe_monitor_judge_runs SET status = 'cancelled' "
                                "WHERE id = :i"), {"i": run.id},
                    )

        judge_env.install(_Client(judge_env.reply, on_call=cancel_on))
        with pytest.raises(OperatorCancelled):
            judge_env.judge.execute_judge_run(sync_db, run.id)

        other = sessionmaker(bind=engine)()
        try:
            stored = other.execute(
                sa.text("SELECT status, progress_detail, metrics FROM probe_monitor_judge_runs "
                        "WHERE id = :i"), {"i": run.id},
            ).one()
        finally:
            other.close()
        assert stored.status == "cancelled", f"the cancellation was overwritten: {stored.status}"
        # The cancel lands DURING row 25's call, so no stored reading may count row 25 or later.
        assert stored.progress_detail["rows_done"] < 25, (
            f"a reading landed after the cancel ({stored.progress_detail['rows_done']} rows)"
        )
        assert stored.metrics == {}, "a cancelled run recorded a result"
        refused = [r for r in _readings(judge_env.recorded) if not r["landed"]]
        assert refused, "no write was attempted after the cancel, so the guard was never exercised"


# ── the janitor reads moving progress as life ─────────────────────────────────────────────────


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _DB:
    def __init__(self, runs, judges):
        self.runs, self.judges = runs, judges
        self.commits = 0

    def query(self, model):
        from src.models.probe_monitor import ProbeMonitorJudgeRun

        return _Query(self.judges if model is ProbeMonitorJudgeRun else self.runs)

    def commit(self):
        self.commits += 1


class _Ctx:
    def __init__(self, db):
        self._db = db

    def __enter__(self):
        return self._db

    def __exit__(self, *a):
        return False


class TestTheJanitorSparesAJobWhoseProgressMoves:
    """⚠ `_looks_alive` answered False for ANY stall reading but None, so a counter that had just
    moved (0.0) was reaped whenever `task_looks_alive` said no — and a judge run, which has no
    `updated_at`, is examined on every sweep after its first hour."""

    def _sweep(self, monkeypatch, *, stalled, runs=(), judges=()):
        from unittest.mock import patch

        from src.workers import cleanup_stuck_probe_monitor_runs as janitor

        db = _DB(list(runs), list(judges))
        monkeypatch.setattr(janitor, "emit_probe_monitor_failed", lambda *a, **k: True)
        with patch.object(janitor.cleanup_stuck_probe_monitor_runs_task, "get_db",
                          lambda: _Ctx(db)), \
             patch.object(janitor, "progress_stalled_seconds", lambda *a, **k: stalled), \
             patch("src.workers.task_heartbeat.task_looks_alive", lambda *a, **k: False):
            return janitor.cleanup_stuck_probe_monitor_runs_task.run()

    @staticmethod
    def _judge():
        from datetime import datetime, timedelta, timezone

        old = datetime.now(timezone.utc) - timedelta(hours=3)
        return SimpleNamespace(id="pmj_1", status="running", celery_task_id="t1", progress=41.2,
                               created_at=old, error_message=None)

    @staticmethod
    def _run_row():
        from datetime import datetime, timedelta, timezone

        old = datetime.now(timezone.utc) - timedelta(hours=3)
        return SimpleNamespace(id="pmr_1", status="running", celery_task_id="t1", progress=61.0,
                               updated_at=old, created_at=old, stage="training",
                               artifact_dir=None, error_message=None)

    def test_a_judge_whose_progress_just_moved_is_SPARED(self, monkeypatch):
        judge = self._judge()
        reaped = self._sweep(monkeypatch, stalled=0.0, judges=[judge])
        assert judge.status == "running" and reaped["judge_runs"] == 0

    def test_a_judge_stalled_past_its_window_is_reaped(self, monkeypatch):
        from src.workers.cleanup_stuck_probe_monitor_runs import JUDGE_STUCK_THRESHOLD_MINUTES

        judge = self._judge()
        reaped = self._sweep(monkeypatch, stalled=JUDGE_STUCK_THRESHOLD_MINUTES * 60 + 1,
                             judges=[judge])
        assert judge.status == "failed" and reaped["judge_runs"] == 1

    def test_a_training_run_whose_progress_just_moved_is_SPARED(self, monkeypatch):
        run = self._run_row()
        reaped = self._sweep(monkeypatch, stalled=0.0, runs=[run])
        assert run.status == "running" and reaped["runs"] == 0

    def test_a_training_run_stalled_past_its_window_is_reaped(self, monkeypatch):
        from src.workers.cleanup_stuck_probe_monitor_runs import STUCK_THRESHOLD_MINUTES

        run = self._run_row()
        reaped = self._sweep(monkeypatch, stalled=STUCK_THRESHOLD_MINUTES * 60 + 1, runs=[run])
        assert run.status == "failed" and reaped["runs"] == 1


# ── the API and the MCP tools return the live reading ─────────────────────────────────────────

JUDGE_DETAIL = {
    "kind": "judge", "phase": "judging", "rows_done": 812, "rows_total": 1370,
    "set_index": 1, "set_count": 2, "set_id": "pmd_x", "set_name": "US test",
    "set_rows_done": 127, "set_rows_total": 685, "rate_per_second": 2.1, "eta_seconds": 266.0,
    "parse_failures": 3, "updated_at": "2026-10-21T12:00:00+00:00",
}
RUN_DETAIL = {
    "kind": "training", "stage": "token_capture", "stage_index": 3, "stage_count": 8,
    "done": 120.0, "total": 400.0, "unit": "batches", "current": None,
    "rate_per_second": 3.2, "eta_seconds": 88.0, "updated_at": "2026-10-21T12:00:00+00:00",
}


def _live_rows(db):
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorRun

    prompt = _prompt(db)
    views = _views(db, ["US test"])
    judge = _run(db, prompt, views, progress=59.3, progress_detail=JUDGE_DETAIL)
    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/nonexistent", architecture="LlamaForCausalLM", params_count=1)
    db.add(model)
    db.flush()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=views[0].id, eval_dataset_ids=[],
                          config={}, status="running", stage="token_capture", progress=48.0,
                          progress_detail=RUN_DETAIL, environment={})
    db.add(run)
    db.commit()
    return judge, run


@pytest.mark.asyncio
async def test_GET_judge_runs_returns_the_live_reading(client, sync_db):
    judge, _ = _live_rows(sync_db)
    listed = {r["id"]: r for r in (await client.get(f"{API}/judge-runs")).json()}
    assert listed[judge.id]["progress_detail"] == JUDGE_DETAIL
    assert listed[judge.id]["progress"] == pytest.approx(59.3)


@pytest.mark.asyncio
async def test_GET_a_training_run_returns_the_live_reading(client, sync_db):
    _, run = _live_rows(sync_db)
    response = await client.get(f"{API}/runs/{run.id}")
    assert response.status_code == 200, response.text
    assert response.json()["progress_detail"] == RUN_DETAIL
    listed = {r["id"]: r for r in (await client.get(f"{API}/runs")).json()}
    assert listed[run.id]["progress_detail"] == RUN_DETAIL


class _Bridge:
    """The MCP tools' HTTP client, answered by the REAL app — so a tool's result is what the
    route returns, not what a stub was told to say."""

    def __init__(self, http):
        self.http = http
        self.calls = []

    async def get(self, path, **params):
        self.calls.append(("GET", path))
        query = {k: v for k, v in params.items() if v is not None}
        response = await self.http.get(f"/api/v1{path}", params=query)
        assert response.status_code == 200, response.text
        return response.json()


async def _call_tool(http, name, arguments):
    import json

    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import probe_monitors

    mcp = MCPServer("t")
    bridge = _Bridge(http)
    probe_monitors.register(mcp, bridge, MCPSettings(auth_token="x" * 32))
    result = await mcp.call_tool(name, arguments)
    # mcp 2.x returns a CallToolResult (1.x: a list of blocks, or a (blocks, structured) tuple).
    content = result.content
    parts = [json.loads(part.text) for part in content if getattr(part, "text", None)]
    # A list result comes back one content part per element; a dict as one part.
    return (parts[0] if len(parts) == 1 and isinstance(parts[0], dict) else parts), bridge.calls


@pytest.mark.asyncio
async def test_the_MCP_judge_listing_carries_the_live_reading(client, sync_db):
    judge, _ = _live_rows(sync_db)
    listed, calls = await _call_tool(client, "list_probe_judge_runs", {})
    assert calls == [("GET", "/probe-monitors/judge-runs")]
    rows = listed if isinstance(listed, list) else [listed]
    entry = next(r for r in rows if r["id"] == judge.id)
    assert entry["progress_detail"] == JUDGE_DETAIL


@pytest.mark.asyncio
async def test_the_MCP_training_run_carries_the_live_reading(client, sync_db):
    _, run = _live_rows(sync_db)
    got, calls = await _call_tool(client, "get_probe_training_run", {"run_id": run.id})
    assert calls == [("GET", f"/probe-monitors/runs/{run.id}")]
    assert got["progress_detail"] == RUN_DETAIL


def test_the_mcp_descriptions_tell_an_agent_the_fields_exist():
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import probe_monitors

    mcp = MCPServer("t")
    probe_monitors.register(mcp, SimpleNamespace(), MCPSettings(auth_token="x" * 32))
    tools = {tool.name: tool for tool in mcp._tool_manager.list_tools()}
    for name in ("list_probe_judge_runs", "get_probe_training_run"):
        assert "progress_detail" in tools[name].description, name
    assert "eta_seconds" in tools["list_probe_judge_runs"].description
    assert "reachable: false" in tools["list_probe_judge_runs"].description
