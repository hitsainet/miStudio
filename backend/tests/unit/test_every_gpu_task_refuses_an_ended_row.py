"""A delivery for a job that has ended does nothing, for every GPU task (ENH-006 D116).

Driven from `celery_app.tasks`, the object a worker dispatches: a GPU task added tomorrow must declare its row here or
state why not, so this cannot go stale by omission. Then real rows for each lifecycle, in each status.

The rule from `refuses_terminal_row`: a route that re-dispatches a row may declare this only if it commits a live status
before dispatching. That is pinned at the moment of dispatch for a capture retry and an attribution re-run.

Controls: `0xcc/reviews/mcp_gaps_controls_2026-11-02.md`.
"""
from __future__ import annotations

import inspect
import uuid
from unittest.mock import MagicMock

import pytest

from src.core.cancellation import SCOPES, terminal_status_of
from src.core.celery_app import celery_app
from src.models.circuit import Circuit
from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from src.models.extraction_job import ExtractionStatus
from src.models.steering_record_run import SteeringRecordRun
from src.models.task_queue import TaskQueue
from src.services.gpu_dispatch import GPU_JOB_MARKER

celery_app.loader.import_default_modules()

#: A GPU task that does not declare a row, and why. A new GPU task must be in one or the other.
NO_DECLARED_ROW = {
    "src.workers.probe_monitor_tasks.evaluate_probe_monitor": "no lifecycle row; its probe's run has always ended by then",
    "src.workers.probe_monitor_tasks.recut_probe_windows": "no lifecycle row",
    "src.workers.probe_monitor_tasks.score_probe_monitor": "no lifecycle row",
    "src.workers.probe_monitor_tasks.score_probe_family": "no lifecycle row",
    "src.workers.probe_monitor_tasks.build_probe_definition": "no lifecycle row",
    "src.workers.circuit_calibration_tasks.reproduce_circuit_calibration": "keyed by manifest; no lifecycle row of its own",
    "src.workers.training_evaluation_tasks.evaluate_training": "refuses unless its training is completed; no row of its own",
    "steering.compare": "no lifecycle row", "steering.sweep": "no lifecycle row", "steering.combined": "no lifecycle row",
    "neuronpedia.compute_dashboard_data": "no lifecycle row",
    "push_to_neuronpedia_local": "its push job is created by the task's own caller; no cancel scope",
    "train_sae": "its own guard: training_tasks.start_refusal",
    "label_features": "its own guard: the pre-flight's completed and cancelled branches",
    "labeling.resume_sweep_step": "keeps its own row's state; a step re-reads the sweep",
    "src.workers.extraction_tasks.extract_features_from_sae": "keyed by SAE, not the job row; its own guard",
    "workers.model_tasks.download_and_load_model": (
        "a retry re-dispatches the row and commits after dispatching, so a guard here would refuse a deliberate retry"
    ),
}


def gpu_tasks():
    return sorted(name for name, task in celery_app.tasks.items() if hasattr(task.run, GPU_JOB_MARKER))


def declared():
    return sorted(name for name, task in celery_app.tasks.items() if hasattr(task.run, "__refuses_terminal_row__"))


class TestEveryGpuTaskIsAccountedFor:
    def test_the_registry_is_read(self):
        assert len(gpu_tasks()) >= 34, gpu_tasks()

    def test_each_declares_a_row_or_says_why_not(self):
        unaccounted = set(gpu_tasks()) - set(declared()) - set(NO_DECLARED_ROW)
        assert not unaccounted, sorted(unaccounted)

    def test_no_exemption_is_stale_or_contradicted(self):
        assert not set(NO_DECLARED_ROW) - set(gpu_tasks())
        assert not set(NO_DECLARED_ROW) & set(declared())

    @pytest.mark.parametrize("name,kind,argument", [
        ("src.workers.circuit_capture_tasks.capture_circuit_activations", "circuit_capture", "run_id"),
        ("src.workers.circuit_capture_tasks.run_circuit_attribution", "circuit_attribution", "run_id"),
        ("src.workers.circuit_validation_tasks.validate_circuit_edges", "circuit_validation", "run_id"),
        ("src.workers.circuit_validation_tasks.run_circuit_faithfulness", "circuit_faithfulness", "circuit_id"),
        ("src.workers.circuit_record_tasks.run_circuit_record", "steering_record", "record_run_id"),
        ("src.workers.circuit_calibration_tasks.run_circuit_calibration", "circuit_calibration", "circuit_id"),
        ("src.workers.jlens_fit_tasks.fit_jlens_artifact", "jlens_task", None),
        ("src.workers.jlens_fit_tasks.revalidate_staged_artifact", "jlens_task", None),
        ("src.workers.jlens_band_tasks.compute_band_report", "jlens_task", None),
        ("src.workers.jlens_readout_tasks.compute_readout", "jlens_task", None),
        ("src.workers.jlens_acquire_tasks.acquire_jlens_artifact", "jlens_task", None),
        ("src.workers.jlens_intervention_tasks.run_intervention", "jlens_task", None),
        ("src.workers.jlens_probe_tasks.compute_probe", "jlens_task", None),
        ("src.workers.jlens_probe_tasks.compute_decomposition", "jlens_task", None),
        ("src.workers.jlens_probe_tasks.compute_template_lens", "jlens_task", None),
        # Not GPU tasks, and they start work on a terminal row the same way: a publish pushes to the Hub again.
        ("src.workers.jlens_acquire_tasks.publish_jlens_artifact", "jlens_task", None),
        ("src.workers.jlens_band_tasks.record_gate", "jlens_task", None),
        ("src.workers.jlens_annotation_tasks.sweep_annotations", "jlens_task", None),
        ("workers.model_tasks.extract_activations", "activation_extraction", "extraction_id"),
    ])
    def test_the_declaration_names_its_row(self, name, kind, argument):
        """By name: a list derived from the declarations would lose a task together with its declaration."""
        assert getattr(celery_app.tasks[name].run, "__refuses_terminal_row__", None) == (kind, argument)
        assert kind in SCOPES


def _call(name, monkeypatch, status):
    """The real task with `terminal_status_of` answering `status`. Returns (result, what it asked, reached the claim)."""
    asked = []

    def read(kind, target):
        asked.append((kind, target))
        return status

    monkeypatch.setattr("src.core.cancellation.terminal_status_of", read)

    class Reached(BaseException):
        pass

    def reached():
        raise Reached()

    monkeypatch.setattr("src.workers.gpu_job.per_card_workers", reached)
    task = celery_app.tasks[name]
    run = inspect.signature(task.run)
    kwargs = {p.name: "arg-" + p.name for p in run.parameters.values()
              if p.default is inspect._empty and p.name != "self"}
    argument = task.run.__refuses_terminal_row__[1]
    if argument and argument not in kwargs:
        kwargs[argument] = "arg-" + argument
    try:
        return task(**kwargs), asked, False
    except Reached:
        return None, asked, True


@pytest.mark.parametrize("name", [n for n in declared() if hasattr(celery_app.tasks[n].run, GPU_JOB_MARKER)])
class TestTheGuardOnEveryDeclaredGpuTask:
    @pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
    def test_an_ended_row_does_nothing(self, name, status, monkeypatch):
        result, asked, reached = _call(name, monkeypatch, status)
        assert not reached, "it reached the GPU claim"
        kind, argument = celery_app.tasks[name].run.__refuses_terminal_row__
        assert result["status"] == "skipped" and result["reason"] == f"already_{status}" and result["scope"] == kind
        assert len(asked) == 1 and asked[0][0] == kind
        if argument:
            assert asked[0][1] == "arg-" + argument

    def test_a_live_row_goes_on(self, name, monkeypatch):
        _result, asked, reached = _call(name, monkeypatch, None)
        assert reached and len(asked) == 1


class TestRealRows:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status,terminal", [("pending", False), ("estimated", False), ("running", False),
                                                 ("completed", True), ("failed", True), ("cancelled", True)])
    async def test_a_capture(self, async_session, status, terminal):
        row = CircuitCaptureRun(status=status, manifest={})
        async_session.add(row)
        await async_session.commit()
        assert terminal_status_of("circuit_capture", row.id) == (status if terminal else None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind,field", [("circuit_attribution", "attribution_status"),
                                            ("circuit_validation", "validation_status")])
    @pytest.mark.parametrize("status,terminal", [(None, False), ("pending", False), ("completed", True), ("cancelled", True)])
    async def test_a_discovery_runs_own_stage_field(self, async_session, kind, field, status, terminal):
        """The discovery row itself reads `completed` by then. A guard on `status` would refuse every pass."""
        row = CircuitDiscoveryRun(capture_run_id=str(uuid.uuid4()), status="completed", params={}, **{field: status})
        async_session.add(row)
        await async_session.commit()
        assert terminal_status_of(kind, row.id) == (status if terminal else None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind,field", [("circuit_faithfulness", "faithfulness_status"),
                                            ("circuit_calibration", "calibration_status")])
    @pytest.mark.parametrize("status,terminal", [("pending", False), ("completed", True), ("failed", True)])
    async def test_a_circuit_stage(self, async_session, kind, field, status, terminal):
        row = Circuit(id=f"crc_{uuid.uuid4().hex[:10]}", name="c", saes=[], members=[], edges=[], **{field: status})
        async_session.add(row)
        await async_session.commit()
        assert terminal_status_of(kind, row.id) == (status if terminal else None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status,terminal", [("pending", False), ("completed", True), ("cancelled", True)])
    async def test_a_recording(self, async_session, status, terminal):
        row = SteeringRecordRun(status=status, artifact_kind="circuit")
        async_session.add(row)
        await async_session.commit()
        assert terminal_status_of("steering_record", row.id) == (status if terminal else None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status,terminal", [("queued", False), ("running", False), ("completed", True), ("cancelled", True)])
    async def test_a_jlens_row_keyed_by_the_celery_id(self, async_session, status, terminal):
        task_id = str(uuid.uuid4())
        async_session.add(TaskQueue(id=f"tq_{uuid.uuid4().hex[:8]}", task_id=task_id, task_type="jlens_fit",
                                    entity_id="m", entity_type="model", status=status, retry_params={}, retry_count=0))
        await async_session.commit()
        assert terminal_status_of("jlens_task", task_id) == (status if terminal else None)
        assert terminal_status_of("jlens_task", None) is None


class TestAJlensTaskReadsItsOwnId:
    def test_the_guard_asks_about_the_celery_task_id(self, monkeypatch):
        name = "src.workers.jlens_probe_tasks.compute_probe"
        asked = []
        monkeypatch.setattr("src.core.cancellation.terminal_status_of",
                            lambda kind, target: asked.append((kind, target)) or "cancelled")
        task = celery_app.tasks[name]
        task.push_request(id="the-celery-id")
        try:
            result = task.run(model_id="m", prompt="p", tokens=[])
        finally:
            task.pop_request()
        assert asked == [("jlens_task", "the-celery-id")]
        assert result["reason"] == "already_cancelled"


class TestADeliberateRetryIsNotARedelivery:
    """The rule: at the moment of dispatch, the row must already read live."""

    @pytest.mark.asyncio
    async def test_a_capture_retry_of_a_failed_run(self, client, async_session, monkeypatch):
        run = CircuitCaptureRun(status="failed", manifest={}, gpu_request="auto")
        async_session.add(run)
        await async_session.commit()
        run_id = run.id
        seen = []

        def spy(task, gpu_request=None):
            def dispatch(*args, **kwargs):
                seen.append(terminal_status_of("circuit_capture", run_id))
                return MagicMock(id="task-1")
            return dispatch

        monkeypatch.setattr("src.api.v1.endpoints.circuit_discovery.gpu_delay", spy)
        monkeypatch.setattr("src.services.circuit_capture_service.CircuitCaptureService.assert_no_active_gpu_run",
                            staticmethod(lambda *a, **k: None))
        response = await client.post(f"/api/v1/circuit-capture/{run_id}/confirm", json={})
        assert response.status_code < 300, response.text
        assert seen == [None]

    @pytest.mark.asyncio
    async def test_an_attribution_rerun_after_a_completed_pass(self, client, async_session, monkeypatch):
        run = CircuitDiscoveryRun(capture_run_id=str(uuid.uuid4()), status="completed", params={},
                                  candidates=[{"a": 1}], attribution_status="completed")
        async_session.add(run)
        await async_session.commit()
        run_id = run.id
        seen = []

        def spy(task, gpu_request=None):
            def dispatch(*args, **kwargs):
                seen.append(terminal_status_of("circuit_attribution", run_id))
                return MagicMock(id="task-1")
            return dispatch

        monkeypatch.setattr("src.api.v1.endpoints.circuit_discovery.gpu_delay", spy)
        monkeypatch.setattr("src.services.circuit_capture_service.CircuitCaptureService.assert_no_active_gpu_run",
                            staticmethod(lambda *a, **k: None))
        response = await client.post(f"/api/v1/circuit-discovery/{run_id}/attribution", json={})
        assert response.status_code < 300, response.text
        assert seen == [None]


class TestSaeExtractionsOwnGuard:
    """It is keyed by the SAE and finds its job row itself, so it cannot declare. It had a completed branch and a failed
    branch, and a cancelled row fell through until the first progress check, after the models had loaded."""

    def test_a_cancelled_job_goes_no_further(self, monkeypatch):
        import contextlib
        from types import SimpleNamespace

        from src.models.external_sae import ExternalSAE
        from src.models.extraction_job import ExtractionJob

        asked = []

        def query(model):
            asked.append(model)
            found = MagicMock()
            job = SimpleNamespace(id="ej_1", status=ExtractionStatus.CANCELLED.value, statistics=None)
            found.filter.return_value.order_by.return_value.first.return_value = job
            found.filter.return_value.first.return_value = None
            return found

        @contextlib.contextmanager
        def session():
            yield MagicMock(query=query)

        monkeypatch.setattr("src.workers.base_task.get_sync_db", session)
        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
        monkeypatch.setattr("src.workers.websocket_emitter.emit_sae_extraction_progress", MagicMock())
        task = celery_app.tasks["src.workers.extraction_tasks.extract_features_from_sae"]
        result = task(sae_id="sae_1", config={})
        assert result == {"status": "cancelled", "scope": "sae_extraction", "target_id": "ej_1"}
        assert asked == [ExtractionJob], "it went on to the SAE"
        assert ExternalSAE not in asked
