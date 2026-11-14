"""A delivery for a row that has already ended does nothing (ENH-006 D116 G12).

Live on 2026-11-02: the GPU queue held a `run_probe_monitor` message for a run an operator had cancelled
while it was queued (the revoke lives only in the worker's memory), and `unacked` held one for a run that
had finished. `run_probe_monitor` wrote `running` unconditionally, so the next worker would have started
the cancelled run on the GPU and re-run the finished one for two hours over its own results.

These run the REAL task against REAL rows and real files. What is faked is the work the task would start,
so "no work done" is a count, and the `gpu_job` claim, so a terminal delivery that reached it fails loudly.

Controls are recorded in `0xcc/reviews/mcp_gaps_controls_2026-11-02.md`.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.core.cancellation import SCOPES, get_scope, terminal_status_of
from src.core.celery_app import celery_app
from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus
from src.models.probe_monitor import (
    ProbeMonitor,
    ProbeMonitorDataset,
    ProbeMonitorJudgeRun,
    ProbeMonitorRun,
)
from src.services import probe_monitor_run as run_module
from src.workers import probe_monitor_tasks as tasks
from src.workers.probe_spot_check_tasks import run_probe_spot_check

TERMINAL = ["completed", "failed", "cancelled"]
SNAPSHOT = ("status", "stage", "progress", "error_message", "environment", "completed_at", "updated_at",
            "started_at", "celery_task_id")


@pytest.fixture
async def run_row(async_session):
    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY, file_path="/nowhere",
                  architecture="LlamaForCausalLM", params_count=1_000)
    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="local", status=DatasetStatus.READY, raw_path="/nowhere",
                      extra_metadata={})
    async_session.add_all([model, dataset])
    await async_session.commit()
    view = ProbeMonitorDataset(name="train", dataset_id=dataset.id, input_column="t", label_column="l",
                               label_mapping={"1": "positive", "0": "negative"}, role="train", counts={})
    async_session.add(view)
    await async_session.commit()

    async def make(status, **fields):
        run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[], config={},
                              status=status, environment={"marker": "as it was"}, **fields)
        async_session.add(run)
        await async_session.commit()
        return run

    return make


@pytest.fixture
def work(monkeypatch, tmp_path):
    """The work a delivery would start, as a count; artifacts under a tmp data dir."""
    from src.core import config as config_module

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
    spy = MagicMock(return_value={"status": "completed"})
    monkeypatch.setattr(run_module, "execute_probe_run", spy)
    monkeypatch.setattr(tasks, "emit_probe_monitor_completed", MagicMock())
    return spy


async def _snapshot(session, row):
    await session.refresh(row)
    return {name: getattr(row, name, None) for name in SNAPSHOT}


class TestATerminalRunIsLeftAlone:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", TERMINAL)
    async def test_the_real_task_does_nothing_to_the_row_or_its_artifacts(
        self, async_session, run_row, work, status, monkeypatch
    ):
        # Per-card mode, as deployed. In single mode `gpu_job` is a pass-through, and the claim patch below
        # would do nothing whichever side of it the guard sits on: a control caught that.
        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: True)
        run = await run_row(status, error_message="as it was", progress=42.0, stage="rung")
        directory = run_module.artifact_dir_for(run.id)
        directory.mkdir(parents=True)
        (directory / "tokens_layer11.f16").write_bytes(b"token bytes")
        (directory / "head.pt").write_bytes(b"weights")
        files = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
        before = await _snapshot(async_session, run)

        # A claim here would mean a lease and a model for a job that must not run.
        with patch("src.workers.gpu_job.claiming", side_effect=AssertionError("it reached the GPU claim")):
            result = tasks.run_probe_monitor(run_id=run.id)

        assert result == {"status": "skipped", "reason": f"already_{status}",
                          "scope": "probe_monitor_run", "target_id": run.id}
        work.assert_not_called()
        async_session.expire_all()
        assert await _snapshot(async_session, run) == before
        assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()} == files

    @pytest.mark.asyncio
    async def test_it_is_logged(self, async_session, run_row, work, caplog):
        run = await run_row("cancelled")
        with caplog.at_level("WARNING", logger="src.core.cancellation"):
            tasks.run_probe_monitor(run_id=run.id)
        assert run.id in caplog.text and "already cancelled" in caplog.text


class TestALiveRunStillRuns:
    """A guard that refuses everything would pass every test above."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["pending", "running"])
    async def test_it_starts_and_reads_running(self, async_session, run_row, work, status, monkeypatch):
        # `running` is a run a crash interrupted: redelivered, it restarts, as it always has.
        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
        run = await run_row(status)
        seen = []
        work.side_effect = lambda db, run_id, on_stage: seen.append(
            db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == run_id).first().status) or {"status": "completed"}

        assert tasks.run_probe_monitor(run_id=run.id) == {"status": "completed"}
        assert work.call_count == 1 and work.call_args.args[1] == run.id
        assert seen == ["running"]

    @pytest.mark.asyncio
    async def test_a_deleted_run_is_still_the_tasks_refusal(self, work, monkeypatch, async_session):
        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
        with pytest.raises(ValueError, match="has no row"):
            tasks.run_probe_monitor(run_id=f"pmr_{uuid.uuid4().hex[:12]}")
        work.assert_not_called()


class TestACancelThatArrivesWhileStarting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", TERMINAL)
    async def test_it_is_not_overwritten(self, async_session, run_row, work, status, monkeypatch):
        """The window between the guard's read and the task's write: the first write is conditional."""
        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
        monkeypatch.setattr("src.core.cancellation.terminal_status_of", lambda kind, target_id: None)
        run = await run_row(status)
        result = tasks.run_probe_monitor(run_id=run.id)
        assert result["reason"] == "ended_while_starting"
        work.assert_not_called()
        async_session.expire_all()
        await async_session.refresh(run)
        assert run.status == status


class TestAnUnreadableRowDoesNotStopTheWork:
    @pytest.mark.asyncio
    async def test_the_task_goes_on_to_its_own_write(self, async_session, run_row, work, monkeypatch):
        """Its first write needs the same database, so an outage cannot pass for "go ahead"."""
        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
        run = await run_row("pending")

        def broken(kind, target_id):
            raise RuntimeError("no database")

        monkeypatch.setattr("src.core.cancellation.terminal_status_of", broken)
        assert tasks.run_probe_monitor(run_id=run.id) == {"status": "completed"}
        assert work.call_count == 1


class TestTheTaskWithNoLifecycleRowStillRuns:
    @pytest.mark.asyncio
    async def test_an_evaluation_runs_though_its_probes_run_has_finished(self, async_session, run_row, monkeypatch):
        """A probe's run is ALWAYS finished by the time anyone evaluates the probe. Guarding on it would
        have stopped every evaluation there is, including the one waiting in the live queue."""
        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
        run = await run_row("completed")
        probe = ProbeMonitor(run_id=run.id, layer=11, rule="mean", rule_params={}, val_metrics={})
        async_session.add(probe)
        await async_session.commit()
        evaluate = MagicMock(return_value={"evaluated": 1})
        monkeypatch.setattr(run_module, "evaluate_probe", evaluate)

        assert tasks.evaluate_probe_monitor(probe_id=probe.id, dataset_ids=["pmd_1"]) == {"evaluated": 1}
        evaluate.assert_called_once()
        assert evaluate.call_args.args[1:] == (probe.id, ["pmd_1"])


class TestTheSiblingsBesideIt:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", TERMINAL)
    async def test_a_terminal_judge_run_calls_no_judge(self, async_session, status, monkeypatch):
        """Its service writes `running` unconditionally too, and every row it re-judges is a paid call."""
        row = ProbeMonitorJudgeRun(endpoint="http://judge", model="m", prompt_version="v1", dataset_ids=[],
                                   status=status)
        async_session.add(row)
        await async_session.commit()
        execute = MagicMock()
        monkeypatch.setattr("src.services.probe_monitor_judge.execute_judge_run", execute)
        result = tasks.run_probe_monitor_judge(judge_run_id=row.id)
        assert result["reason"] == f"already_{status}"
        execute.assert_not_called()
        async_session.expire_all()
        await async_session.refresh(row)
        assert row.status == status

    @pytest.mark.asyncio
    async def test_a_live_judge_run_runs(self, async_session, monkeypatch):
        row = ProbeMonitorJudgeRun(endpoint="http://judge", model="m", prompt_version="v1", dataset_ids=[],
                                   status="pending")
        async_session.add(row)
        await async_session.commit()
        execute = MagicMock(return_value={"status": "completed"})
        monkeypatch.setattr("src.services.probe_monitor_judge.execute_judge_run", execute)
        assert tasks.run_probe_monitor_judge(judge_run_id=row.id) == {"status": "completed"}
        execute.assert_called_once()

    def test_a_terminal_spot_check_runs_no_check(self, monkeypatch):
        execute = MagicMock()
        monkeypatch.setattr("src.services.probe_spot_check.execute", execute)
        monkeypatch.setattr("src.core.cancellation.terminal_status_of", lambda kind, target_id: "cancelled")
        result = run_probe_spot_check(spot_check_id="psc_1")
        assert result == {"status": "skipped", "reason": "already_cancelled",
                          "scope": "probe_spot_check", "target_id": "psc_1"}
        execute.assert_not_called()


class TestTheWiringIsInTheLiveRegistry:
    """Read from `celery_app.tasks`, the object a worker dispatches, never from the source."""

    @pytest.mark.parametrize("name,expected", [
        ("src.workers.probe_monitor_tasks.run_probe_monitor", ("probe_monitor_run", "run_id")),
        ("src.workers.probe_monitor_tasks.run_probe_monitor_judge", ("probe_monitor_judge", "judge_run_id")),
        ("src.workers.probe_spot_check_tasks.run_probe_spot_check", ("probe_spot_check", "spot_check_id")),
    ])
    def test_the_task_declares_its_row(self, name, expected):
        declared = getattr(celery_app.tasks[name].run, "__refuses_terminal_row__", None)
        assert declared == expected
        kind, _argument = declared
        assert kind in SCOPES
        assert {"completed", "failed", "cancelled"} <= set(get_scope(kind).terminal_values)

    @pytest.mark.parametrize("name", ["evaluate_probe_monitor", "recut_probe_windows", "score_probe_monitor",
                                      "score_probe_family", "build_probe_definition"])
    def test_a_task_with_no_lifecycle_row_declares_none(self, name):
        task = celery_app.tasks[f"src.workers.probe_monitor_tasks.{name}"]
        assert not hasattr(task.run, "__refuses_terminal_row__")

    def test_it_sits_outside_the_gpu_claim_and_the_marker_survives(self):
        """Outside, so the row is read before any precheck hands the message on. The marker is how the
        dispatcher knows it is a GPU job."""
        from src.services.gpu_dispatch import GPU_JOB_MARKER

        run = celery_app.tasks["src.workers.probe_monitor_tasks.run_probe_monitor"].run
        assert hasattr(run, GPU_JOB_MARKER)
        assert hasattr(run, "__refuses_terminal_row__")


class TestTheStatusRead:
    @pytest.mark.asyncio
    async def test_live_terminal_and_missing(self, async_session, run_row):
        assert terminal_status_of("probe_monitor_run", (await run_row("cancelled")).id) == "cancelled"
        assert terminal_status_of("probe_monitor_run", (await run_row("completed")).id) == "completed"
        assert terminal_status_of("probe_monitor_run", (await run_row("running")).id) is None
        assert terminal_status_of("probe_monitor_run", "pmr_gone") is None
