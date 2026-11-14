"""The evaluation's lifecycle: an abandoned run is reaped, a long one heartbeats (review R1-C).

FOUND 2026-09-15 (review round 1, R1-C).

* NO JANITOR. ``trainings.evaluation.status`` is written ``pending`` by the endpoint
  and ``running`` by the job, and only the job ever moved it on. A pod roll, an
  OOM kill or a lost lease — anything that is not an ``Exception`` — left it there
  forever: the panel disabled its button and polled every ten seconds
  indefinitely, the endpoint refused a new run with 409, and the ``force=true``
  its message named was sent by nothing in the UI. The ``nlp_status`` shape.
* NO HEARTBEAT. The job wrote ``running`` once, before a pass that can take hours
  at a large token budget, so no clock could tell a live evaluation from a dead one.
* TWO EXTRACTIONS OF ONE TOKENIZATION were judged separately, so the rows the
  larger one read counted as unseen by the smaller — training data evaluated as
  text "the training never read".
* THE RE-RUN LOADED A TRANSCODER STRICTLY. Its export has no input centring bias,
  so the whole re-run was recorded FAILED where the post-run step records the same
  SAE as skipped.
* THE RESERVATION WAS FLAT. 3,072 MB whatever the vocabulary, against a measured
  5 GB at 65,536 (see test_sae_evaluation_memory.py).

MUTATION CONTROLS (2026-09-15; each applied alone, this file run, restored, sha256 verified):
  L1 evaluation_looks_abandoned ignores task_alive (always "gone")   RED  test_a_live_task_is_never_condemned_for_quiet
  L2 the age check removed                                          RED  test_a_recent_write_is_not_judged_at_all
  L3 pending judged like running                                    RED  test_pending_is_never_reaped
  L4 reap_abandoned_evaluations writes nothing                      RED  test_the_reaper_fails_the_abandoned_record_and_emits,
                                                                          test_the_reaper_selects_running_records_in_postgres
  L5 the janitor call removed from cleanup_stuck_trainings          RED  test_the_stuck_trainings_janitor_runs_the_reaper
  L6 the heartbeat never calls record                               RED  test_a_long_evaluation_rewrites_its_record_as_it_goes
  L7 merge_shared_tokenizations call removed from run_evaluation    RED  test_run_evaluation_never_reads_a_row_a_sibling_extraction_read
  L8 merge takes the SMALLER bound                                  RED  test_the_larger_bound_wins, test_run_evaluation_never_reads...
  L9 the re-run loads every SAE (no pre-load skip)                  RED  test_a_rerun_skips_a_transcoder_before_loading_it
  L10 the task reserves the flat 3,072 MB                           RED  test_the_rerun_reserves_memory_for_its_vocabulary
"""

import ast
import inspect
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import torch

from src.services import training_evaluation as te
from src.services.training_evaluation import EvalSource

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def _doc(status="running", minutes_ago=30, task_id="task-1", **extra):
    stamp = (NOW - timedelta(minutes=minutes_ago)).isoformat()
    return {"status": status, "task_id": task_id, "started_at": stamp, "updated_at": stamp, **extra}


def _gone(*_):
    return False


def _alive(*_):
    return True


# ── the rule ────────────────────────────────────────────────────────────────


class TestTheAbandonedRule:
    def test_a_stale_running_record_whose_task_is_gone_is_abandoned(self):
        assert te.evaluation_looks_abandoned(_doc(), task_alive=_gone, now=NOW)

    def test_a_live_task_is_never_condemned_for_quiet(self):
        assert not te.evaluation_looks_abandoned(_doc(), task_alive=_alive, now=NOW)

    def test_a_recent_write_is_not_judged_at_all(self):
        asked = []
        doc = _doc(minutes_ago=5)
        assert not te.evaluation_looks_abandoned(doc, task_alive=lambda *a: asked.append(a) or False, now=NOW)
        assert asked == []

    def test_the_boundary_is_the_stated_age(self):
        at = NOW - timedelta(seconds=te.ABANDONED_AFTER_SECONDS)
        doc = {"status": "running", "task_id": "t", "updated_at": at.isoformat()}
        assert not te.evaluation_looks_abandoned(doc, task_alive=_gone, now=NOW)
        assert te.evaluation_looks_abandoned(doc, task_alive=_gone, now=NOW + timedelta(seconds=1))

    def test_pending_is_never_reaped(self):
        """A pending request may be queued behind a days-long job; nothing but the operator can judge it."""
        doc = _doc(status="pending", minutes_ago=24 * 60, requested_at=(NOW - timedelta(days=1)).isoformat())
        assert not te.evaluation_looks_abandoned(doc, task_alive=_gone, now=NOW)

    @pytest.mark.parametrize("status", ["completed", "failed", "skipped"])
    def test_a_finished_record_is_never_reaped(self, status):
        assert not te.evaluation_looks_abandoned(_doc(status=status, minutes_ago=600), task_alive=_gone, now=NOW)

    def test_the_last_write_is_the_clock_not_the_start(self):
        doc = _doc(minutes_ago=120)
        doc["updated_at"] = (NOW - timedelta(minutes=1)).isoformat()
        assert not te.evaluation_looks_abandoned(doc, task_alive=_gone, now=NOW)

    def test_the_task_is_asked_about_the_last_write(self):
        asked = []
        doc = _doc(minutes_ago=45, task_id="celery-abc")
        te.evaluation_looks_abandoned(doc, task_alive=lambda *a: asked.append(a) or True, now=NOW)
        assert asked == [("celery-abc", NOW - timedelta(minutes=45))]

    @pytest.mark.parametrize("document", [None, {}, {"status": "running"}, {"status": "running", "updated_at": "?"}])
    def test_a_record_with_no_readable_time_is_left_alone(self, document):
        assert not te.evaluation_looks_abandoned(document, task_alive=_gone, now=NOW)


# ── the reaper ──────────────────────────────────────────────────────────────


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *a):
        return self

    def all(self):
        return list(self.rows)


class _Db:
    def __init__(self, rows):
        self.rows = rows
        self.commits = 0

    def query(self, model):
        return _Query(self.rows)

    def commit(self):
        self.commits += 1


def test_the_reaper_fails_the_abandoned_record_and_emits(monkeypatch):
    emitted = []
    monkeypatch.setattr(te, "_emit", lambda training_id, document: emitted.append((training_id, document)))
    dead = SimpleNamespace(id="train_dead", evaluation=_doc(minutes_ago=30, progress={"stage": "means"}))
    live = SimpleNamespace(id="train_live", evaluation=_doc(minutes_ago=2))
    db = _Db([dead, live])

    reaped = te.reap_abandoned_evaluations(db, now=NOW, task_alive=_gone)

    assert reaped == ["train_dead"]
    assert dead.evaluation["status"] == "failed"
    assert "no longer running" in dead.evaluation["reason"]
    assert dead.evaluation["completed_at"] == NOW.isoformat()
    assert dead.evaluation["progress"] == {"stage": "means"}, "what it had reached is kept"
    assert live.evaluation["status"] == "running"
    assert db.commits == 1
    assert [training_id for training_id, _ in emitted] == ["train_dead"]
    json.dumps(dead.evaluation, allow_nan=False)


@pytest.mark.asyncio
async def test_the_reaper_selects_running_records_in_postgres(async_session, monkeypatch):
    """The JSONB filter itself, on a real database: only `running` rows are candidates."""
    from src.models.model import Model
    from src.models.training import Training

    monkeypatch.setattr(te, "_emit", lambda *a: None)
    async_session.add(Model(id="m_reap", name="tiny", repo_id="org/tiny", architecture="llama",
                            params_count=1, quantization="FP16", status="ready"))
    await async_session.flush()
    for training_id, document in {
        "train_r_run": _doc(minutes_ago=30),
        "train_r_pend": _doc(status="pending", minutes_ago=30),
        "train_r_done": _doc(status="completed", minutes_ago=30),
        "train_r_none": None,
    }.items():
        async_session.add(Training(
            id=training_id, model_id="m_reap", dataset_id="", status="completed",
            hyperparameters={"total_steps": 1}, total_steps=1, evaluation=document,
        ))
    await async_session.commit()

    reaped = await async_session.run_sync(
        lambda session: te.reap_abandoned_evaluations(session, now=NOW, task_alive=_gone)
    )
    assert reaped == ["train_r_run"]
    async_session.expire_all()
    rows = {row.id: row for row in (await async_session.execute(
        __import__("sqlalchemy").select(Training).where(Training.id.like("train_r_%"))
    )).scalars()}
    assert rows["train_r_run"].evaluation["status"] == "failed"
    assert rows["train_r_pend"].evaluation["status"] == "pending"
    assert rows["train_r_done"].evaluation["status"] == "completed"
    assert rows["train_r_none"].evaluation is None


def _calls_in(module, function_name, callee):
    tree = ast.parse(inspect.getsource(module))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function_name)
    return [
        n for n in ast.walk(function)
        if isinstance(n, ast.Call) and (getattr(n.func, "id", None) == callee or getattr(n.func, "attr", None) == callee)
    ]


def test_the_stuck_trainings_janitor_runs_the_reaper(monkeypatch):
    """Wiring, by behaviour: the beat janitor with no stuck training still sweeps evaluations."""
    from src.workers import cleanup_stuck_trainings as janitor

    swept = []
    monkeypatch.setattr(te, "reap_abandoned_evaluations", lambda db, **kw: swept.append(db) or ["t"])

    class _EmptyQuery:
        def filter(self, *a, **k):
            return self

        def all(self):
            return []

    class _S:
        def query(self, *a):
            return _EmptyQuery()

        def commit(self):
            pass

        def rollback(self):
            pass

    session = _S()

    @contextmanager
    def get_db():
        yield session

    monkeypatch.setattr(janitor.cleanup_stuck_trainings_task, "get_db", get_db, raising=False)
    janitor.cleanup_stuck_trainings_task.run()
    assert swept == [session]
    assert len(_calls_in(janitor, "cleanup_stuck_trainings_task", "_sweep_abandoned_evaluations")) == 1
    assert len(_calls_in(janitor, "_sweep_abandoned_evaluations", "reap_abandoned_evaluations")) == 1


def test_a_failing_sweep_does_not_stop_the_stuck_training_sweep(monkeypatch):
    """Fail-soft, in its own session: the progress-gate fakes found the first wiring
    raising out of the janitor when the sweep failed inside the stuck-training session."""
    from src.workers import cleanup_stuck_trainings as janitor

    def explode(db, **kw):
        raise RuntimeError("the sweep broke")

    monkeypatch.setattr(te, "reap_abandoned_evaluations", explode)
    opened = []

    class _EmptyQuery:
        def filter(self, *a, **k):
            return self

        def all(self):
            return []

    @contextmanager
    def get_db():
        session = SimpleNamespace(query=lambda *a: _EmptyQuery(), commit=lambda: None)
        opened.append(session)
        yield session

    monkeypatch.setattr(janitor.cleanup_stuck_trainings_task, "get_db", get_db, raising=False)
    assert janitor.cleanup_stuck_trainings_task.run() == {"cleaned": 0}
    assert len(opened) == 2 and opened[0] is not opened[1]


def test_a_finished_evaluation_is_announced_on_the_event_the_ui_listens_for(tmp_path, monkeypatch):
    """REVIEW R1-C C22. Renaming the event survived the suite: nothing asserted the emit.
    The frontend's `useTrainingWebSocket` test pins the listener to the same literal.

    NEGATIVE CONTROL: `event="training:evaluated"` in _emit -> RED here.
    """
    from src.workers import websocket_emitter
    from test_training_evaluation import _run

    sent = []
    monkeypatch.setattr(websocket_emitter, "emit_training_progress", lambda **kw: sent.append(kw) or True)
    document, _row, _db, _loads = _run(tmp_path, token_budget=24)
    assert document["status"] == "completed", document.get("reason")
    [call] = sent
    assert call == {"training_id": "t1", "event": "training:evaluation",
                    "data": {"training_id": "t1", "evaluation": document}}


# ── the heartbeat ───────────────────────────────────────────────────────────


def test_a_long_evaluation_rewrites_its_record_as_it_goes(tmp_path, monkeypatch):
    from test_training_evaluation import _run

    monkeypatch.setattr(te, "HEARTBEAT_SECONDS", 0)
    # One 8-token row a batch, so the three rows are three batches.
    document, row, db, _loads = _run(tmp_path, token_budget=24, batch_tokens=8)
    assert document["status"] == "completed", document.get("reason")
    running = [w for w in db.writes if w["status"] == "running"]
    stages = [w.get("progress", {}).get("stage") for w in running]
    assert "loading_model" in stages and "means" in stages and "cross_entropy" in stages
    beats = [w["progress"] for w in running if w.get("progress", {}).get("stage") == "cross_entropy"]
    assert beats[-1]["batches_done"] == beats[-1]["batches"] == 3
    assert all(w["updated_at"] for w in running)


def test_a_quiet_heartbeat_does_not_write_every_batch(tmp_path, monkeypatch):
    """NEGATIVE CONTROL for the throttle: at the default interval a short run writes no beats."""
    from test_training_evaluation import _run

    document, row, db, _loads = _run(tmp_path, token_budget=24)
    stages = [w.get("progress", {}).get("stage") for w in db.writes if w["status"] == "running"]
    assert "cross_entropy" not in stages and "means" not in stages


def test_a_stop_between_batches_is_recorded_cancelled(tmp_path):
    """R1-D R1D-7, at the unit level. A stop answered after the first batch ends the
    evaluation there: recorded `cancelled` with the reason, never `completed`.

    NEGATIVE CONTROL: the `should_stop` check removed from the heartbeat -> RED here.
    """
    from test_training_evaluation import _run

    asked = []

    def should_stop():
        asked.append(True)
        return "the training was cancelled" if len(asked) >= 2 else None

    document, _row, db, _loads = _run(tmp_path, token_budget=24, batch_tokens=8, should_stop=should_stop)
    assert document["status"] == "cancelled", document
    assert "cancelled" in document["reason"]
    assert [w["status"] for w in db.writes].count("completed") == 0
    assert len(asked) == 2, "the evaluation must stop at the first batch after the stop, not run on"


class TestThePostRunStopReason:
    """`post_run_stop_reason`: what the post-run evaluation checks between batches (R1-D R1D-7).

    R1D-7's end-to-end reproduction covers an operator's Stop; these cover the rest of
    the rule, including a lost GPU lease, which nothing else exercises on this path.

    NEGATIVE CONTROL R7e: `stop_signal_for(row, step)` without `lease_lost` -> RED
    test_a_lost_lease_stops_it.
    """

    def _reason(self, monkeypatch, row, lease=None):
        from src.workers import training_tasks

        monkeypatch.setattr(training_tasks, "lease_lost_reason", lambda: lease)

        class _Query:
            def filter_by(self, **kw):
                return self

            def first(self):
                return row

        @contextmanager
        def get_db():
            yield SimpleNamespace(query=lambda model: _Query())

        return training_tasks.post_run_stop_reason(get_db, "t1", 40)

    def test_a_running_training_is_not_stopped(self, monkeypatch):
        assert self._reason(monkeypatch, SimpleNamespace(status="running")) is None

    @pytest.mark.parametrize("status", ["cancelled", "paused"])
    def test_an_operator_stop_or_pause_stops_it(self, monkeypatch, status):
        assert self._reason(monkeypatch, SimpleNamespace(status=status)) == f"the training was {status}"

    def test_a_deleted_row_stops_it(self, monkeypatch):
        assert self._reason(monkeypatch, None) == "deleted"

    def test_a_lost_lease_stops_it(self, monkeypatch):
        reason = self._reason(monkeypatch, SimpleNamespace(status="running"), lease="the lease on GPU-1 expired")
        assert reason == "the lease on GPU-1 expired"


def test_the_training_task_hands_the_evaluation_its_stop_check():
    """Wiring, by AST: the post-run call passes `should_stop` as a call to `post_run_stop_reason`.

    NEGATIVE CONTROL R7d: the keyword removed -> RED here (and R1D-7's reproduction).
    """
    from src.workers import training_tasks

    tree = ast.parse(inspect.getsource(training_tasks))
    task = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_sae_task")
    [call] = [n for n in ast.walk(task) if isinstance(n, ast.Call)
              and getattr(n.func, "id", None) == "run_post_run_evaluation"]
    [keyword] = [k for k in call.keywords if k.arg == "should_stop"]
    inner = [n for n in ast.walk(keyword.value) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "post_run_stop_reason"]
    assert len(inner) == 1


def test_the_rerun_task_stops_on_a_lost_lease(monkeypatch, tmp_path):
    from src.services import gpu_job_claim
    from src.workers import training_evaluation_tasks as task_module

    captured = {}
    monkeypatch.setattr(task_module, "load_exported_sae", lambda hp, path: torch.nn.Linear(1, 1))
    monkeypatch.setattr(task_module, "run_evaluation", lambda **kw: captured.update(kw) or {"status": "completed"})
    _patch_rerun(monkeypatch, tmp_path, task_module)
    task_module.evaluate_training_task.run(training_id="t1")

    monkeypatch.setattr(gpu_job_claim, "lease_lost_reason", lambda: "the lease on GPU-1 expired")
    assert captured["should_stop"]() == "the lease on GPU-1 expired"
    monkeypatch.setattr(gpu_job_claim, "lease_lost_reason", lambda: None)
    assert captured["should_stop"]() is None


@pytest.mark.asyncio
async def test_an_evaluation_write_moves_the_trainings_clock(async_session):
    """R1-D J1. `cleanup_stuck_trainings` reaps a RUNNING training whose `updated_at` is
    30 minutes old; the post-run evaluation runs while the training is RUNNING. Every
    evaluation write — the heartbeat included — must move that clock, on a real database.

    NEGATIVE CONTROL: `onupdate` removed from `Training.updated_at` -> RED here (the
    write lands and the clock stays two hours old).
    """
    from contextlib import contextmanager as _cm

    import sqlalchemy as sa

    from src.models.model import Model, ModelStatus, QuantizationFormat
    from src.models.training import Training

    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    async_session.add(Model(id="m_clock", name="tiny", architecture="llama", params_count=1,
                            quantization=QuantizationFormat.FP16, status=ModelStatus.READY))
    async_session.add(Training(id="train_clock", model_id="m_clock", dataset_id="", status="running",
                               hyperparameters={"total_steps": 1}, total_steps=1))
    await async_session.commit()
    await async_session.execute(sa.update(Training).where(Training.id == "train_clock").values(updated_at=stale))
    await async_session.commit()

    def write(session):
        @_cm
        def get_db():
            yield session

        return te.write_evaluation(get_db, "train_clock", {"status": "running", "updated_at": NOW.isoformat()})

    assert await async_session.run_sync(write) is True
    async_session.expire_all()
    row = (await async_session.execute(sa.select(Training).where(Training.id == "train_clock"))).scalar_one()
    assert row.evaluation["status"] == "running"
    assert row.updated_at > stale + timedelta(hours=1), f"the training's clock did not move: {row.updated_at}"


# ── shared tokenizations ────────────────────────────────────────────────────


class TestMergeSharedTokenizations:
    def test_distinct_tokenizations_are_untouched(self):
        a = EvalSource(label="a", dataset_path="/d/x", rows_read=10)
        b = EvalSource(label="b", dataset_path="/d/y", rows_read=20)
        assert te.merge_shared_tokenizations([a, b]) == [a, b]

    def test_the_larger_bound_wins(self):
        merged = te.merge_shared_tokenizations([
            EvalSource(label="small", dataset_path="/d/x", rows_read=10, rows_processed=10, weight=1.0),
            EvalSource(label="large", dataset_path="/d/./x", rows_read=50, rows_processed=40, weight=3.0),
        ])
        [source] = merged
        assert source.rows_read == 50 and source.rows_processed == 40
        assert source.weight == 4.0 and source.label == "small+large"
        assert te.first_unseen_row(source, 100) == 50

    def test_an_extraction_that_read_every_row_wins_outright(self):
        [source] = te.merge_shared_tokenizations([
            EvalSource(label="a", dataset_path="/d/x", rows_read=10),
            EvalSource(label="b", dataset_path="/d/x", rows_read=0),
        ])
        assert te.first_unseen_row(source, 100) == 100

    def test_an_unknown_bound_leaves_nothing_unseen(self):
        [source] = te.merge_shared_tokenizations([
            EvalSource(label="a", dataset_path="/d/x", rows_read=10),
            EvalSource(label="b", dataset_path="/d/x"),
        ])
        assert te._unseen_count(source, 100) == 0

    def test_named_rows_keep_only_what_every_source_names(self):
        [source] = te.merge_shared_tokenizations([
            EvalSource(label="a", dataset_path="/d/x", candidate_rows=(1, 2, 3, 9)),
            EvalSource(label="b", dataset_path="/d/x", candidate_rows=(2, 3, 4, 9)),
        ])
        assert source.candidate_rows == (2, 3, 9)

    def test_a_tokenization_read_both_ways_is_refused(self):
        with pytest.raises(ValueError, match="cannot be decided"):
            te.merge_shared_tokenizations([
                EvalSource(label="a", dataset_path="/d/x", rows_read=10),
                EvalSource(label="b", dataset_path="/d/x", candidate_rows=(20,)),
            ])


def test_run_evaluation_never_reads_a_row_a_sibling_extraction_read(tmp_path):
    """End to end: a 4-row and a 7-row extraction of one 12-row tokenization; only rows >= 7 are read."""
    from test_training_evaluation import _RecordingOpen, _run, _tokenization

    path = _tokenization(tmp_path)
    opener = _RecordingOpen()
    document, _row, _db, _loads = _run(
        tmp_path, token_budget=10_000, open_dataset=opener,
        sources=[
            EvalSource(label="small", dataset_path=path, rows_read=4, rows_processed=4),
            EvalSource(label="large", dataset_path=path, rows_read=7, rows_processed=7),
        ],
    )
    assert document["status"] == "completed", document.get("reason")
    assert opener.selected and min(opener.selected) >= 7
    [source] = document["sources"]
    assert source["rows_read_by_training"] == 7


# ── the re-run task ─────────────────────────────────────────────────────────


def test_a_rerun_skips_a_transcoder_before_loading_it(monkeypatch, tmp_path):
    from test_training_evaluation import rerun as _rerun_fixture  # noqa: F401 - fixture module import

    from src.workers import training_evaluation_tasks as task_module

    loaded = []
    monkeypatch.setattr(task_module, "load_exported_sae", lambda hp, path: loaded.append(path) or torch.nn.Linear(1, 1))
    captured = {}
    monkeypatch.setattr(task_module, "run_evaluation", lambda **kw: captured.update(kw) or {"status": "skipped"})
    _patch_rerun(monkeypatch, tmp_path, task_module,
                 hp={"architecture_type": "transcoder", "training_layers": [3], "hook_types": ["residual"]})

    result = task_module.evaluate_training_task.run(training_id="t1")

    assert result["status"] == "skipped"
    assert loaded == [], "a transcoder must not be loaded strictly just to be skipped"
    assert captured["saes"] == {}
    assert captured["skipped_saes"] == [
        {"layer": 3, "hook_type": "residual", "reason": "a transcoder reconstructs a different layer's output"}
    ]


def test_the_rerun_reserves_memory_for_its_vocabulary(monkeypatch, tmp_path):
    from src.workers import training_evaluation_tasks as task_module

    requested = []

    def place(request, required_mb=None, allow_shard=False):
        requested.append(required_mb)
        from src.services.gpu_placement import Placement

        return Placement(card=None, device=torch.device("cpu"))

    monkeypatch.setattr(task_module, "load_exported_sae", lambda hp, path: torch.nn.Linear(1, 1))
    monkeypatch.setattr(task_module, "run_evaluation", lambda **kw: {"status": "completed"})
    _patch_rerun(monkeypatch, tmp_path, task_module, place=place,
                 architecture_config={"text_config": {"vocab_size": 262_144}})
    task_module.evaluate_training_task.run(training_id="t1")

    [required] = requested
    expected = te.evaluation_working_mb(262_144)
    assert expected > task_module.EVALUATION_WORKING_MB
    assert required >= expected


def _patch_rerun(monkeypatch, tmp_path, task_module, *, hp=None, place=None, architecture_config=None):
    """The re-run task over stand-in rows and a real export directory layout."""
    from src.models.activation_extraction import ActivationExtraction
    from src.models.model import Model
    from src.models.training import Training
    from src.services.gpu_placement import Placement
    from src.workers import base_task

    hp = hp or {"architecture_type": "jumprelu", "training_layers": [3], "hook_types": ["residual"]}
    rows = {
        Training: [SimpleNamespace(id="t1", status="completed", model_id="m_x", extraction_id=None,
                                   extraction_ids=["ext_1"], hyperparameters=hp, evaluation=None)],
        Model: [SimpleNamespace(id="m_x", repo_id="org/tiny", quantization="FP16", file_path=None,
                                params_count=1000, architecture_config=architecture_config or {})],
        ActivationExtraction: [SimpleNamespace(id="ext_1", output_path=str(tmp_path / "ext"))],
    }

    class _Q:
        def __init__(self, found):
            self.found = found

        def filter(self, *a):
            return self

        def first(self):
            return self.found[0] if self.found else None

    class _S:
        def query(self, model):
            return _Q(rows.get(model, []))

        def commit(self):
            pass

    @contextmanager
    def get_sync_db():
        yield _S()

    monkeypatch.setattr(base_task, "get_sync_db", get_sync_db)
    monkeypatch.setattr(task_module, "place_job",
                        place or (lambda request, required_mb=None, allow_shard=False:
                                  Placement(card=None, device=torch.device("cpu"))))
    monkeypatch.setattr(task_module.settings, "data_dir", tmp_path)
    layer_dir = tmp_path / "trainings" / "t1" / "community_format" / "layer_3_residual"
    layer_dir.mkdir(parents=True)
    (layer_dir / "sae_weights.safetensors").write_bytes(b"\0" * 2048)
