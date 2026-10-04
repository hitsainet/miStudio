"""Work that reloads a probe's model must not mix precisions into the probe's evidence.

Review round 1 (2026-10-03), H1: evaluate, the GPU window re-cut and the offline score all
reload the run's model, which now loads at the checkpoint's precision (bfloat16 here), while every
existing probe was fitted and thresholded at float16. Nothing compared the two: re-evaluating
would write bfloat16 AUROCs into float16 evidence, and the re-cut would move bars silently.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from src.services import probe_definition_builder, probe_monitor_run
from src.services.probe_monitor_run import ProbePrecisionRefused, probe_precision, require_matching_precision


def _db(model_row):
    class _Q:
        def filter(self, *a, **k):
            return self

        def first(self):
            return model_row

    return SimpleNamespace(query=lambda model: _Q())


@pytest.fixture
def loads_at(monkeypatch, tmp_path):
    """A model row whose snapshot records `dtype`; returns a setter."""
    def make(dtype, quantization="FP16"):
        (tmp_path / "config.json").write_text('{"torch_dtype": "%s"}' % dtype)
        monkeypatch.setattr(probe_monitor_run, "resolve_weights_dir", lambda *a: str(tmp_path))
        import src.services.activation_service as act
        monkeypatch.setattr(act, "resolve_model_snapshot", lambda p: p)
        return SimpleNamespace(id="m", file_path=str(tmp_path), quantized_path=None, quantization=quantization)
    return make


def test_a_run_that_recorded_its_precision_and_matches_is_allowed(loads_at):
    run = SimpleNamespace(id="pmr_new", model_id="m", environment={"model_dtype": "bfloat16"})
    result = probe_precision(_db(loads_at("bfloat16")), run)
    assert result == {"recorded": "bfloat16", "loads_at": "bfloat16", "refusal": None}


def test_a_run_that_predates_recording_is_refused(loads_at):
    run = SimpleNamespace(id="pmr_old", model_id="m", environment={"seed": 1})
    result = probe_precision(_db(loads_at("bfloat16")), run)
    assert result["recorded"] is None and "predates precision recording" in result["refusal"]
    with pytest.raises(ProbePrecisionRefused):
        require_matching_precision(_db(loads_at("bfloat16")), run)


def test_a_run_whose_model_now_loads_differently_is_refused(loads_at):
    run = SimpleNamespace(id="pmr_x", model_id="m", environment={"model_dtype": "float16"})
    assert "now loads at bfloat16" in probe_precision(_db(loads_at("bfloat16")), run)["refusal"]


def test_a_run_whose_model_row_changed_quantization_is_refused_at_the_same_precision(loads_at):
    """Review round 1 (miLLM MED-2): Q4 and FP16 loads of one bfloat16 checkpoint share a precision
    and read different activations (~0.93 cosine per token)."""
    run = SimpleNamespace(id="pmr_q", model_id="m", environment={"model_dtype": "bfloat16", "quantization": "Q4"})
    refusal = probe_precision(_db(loads_at("bfloat16", quantization="FP16")), run)["refusal"]
    assert refusal and "same precision, different activations" in refusal


def _first_line_of(fn, name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)]
    return min(lines) if lines else None


@pytest.mark.parametrize("fn,gate", [
    (probe_monitor_run.evaluate_probe, "require_matching_precision"),
    (probe_monitor_run.recut_probe_windows_on_gpu, "probe_precision"),
    (probe_definition_builder.build, "probe_precision"),
])
def test_each_path_asks_before_it_loads_the_model(fn, gate):
    gate_line = _first_line_of(fn, gate)
    load_line = _first_line_of(fn, "_load_model_for_run") or _first_line_of(fn, "loader")
    assert gate_line is not None, f"{fn.__name__} never checks the precision"
    assert gate_line < load_line, f"{fn.__name__} loads the model before checking the precision"


def test_the_offline_score_reports_both_precisions():
    src = textwrap.dedent(inspect.getsource(probe_monitor_run.score_one))
    tree = ast.parse(src)
    keys = [k.value for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys
            if isinstance(k, ast.Constant)]
    assert {"precision", "fitted_at", "scored_at"} <= set(keys)


def test_the_three_gpu_endpoints_refuse_before_queueing():
    """A 409 at submit, not a 202 that fails in the worker."""
    from src.api.v1.endpoints import probe_monitors as ep

    for fn in (ep.evaluate_probe_endpoint, ep.build_definition, ep.recalibrate_probe_endpoint):
        gate = _first_line_of(fn, "refuse_precision_mismatch")
        queue = _first_line_of(fn, "gpu_delay")
        assert gate is not None and gate < queue, f"{fn.__name__} queues before checking precision"


@pytest.mark.parametrize("environment,has_definition,ends", [
    ({"seed": 1}, True, True),                      # legacy run, definition cached: a commit ends export
    ({"seed": 1}, False, False),                    # legacy run, nothing cached to lose
    ({"model_dtype": "bfloat16"}, True, False),     # recorded run: rebuildable
])
def test_the_recalibration_preview_says_when_a_commit_would_end_exportability(
    monkeypatch, environment, has_definition, ends
):
    """Review round 1, M3: a commit invalidates the cached definition, and a run that predates
    precision recording can never rebuild one. The PREVIEW must say so."""
    from src.services import probe_recalibration as rec

    probe = SimpleNamespace(id="pm_x", run_id="pmr_x", definition_path="/d.json" if has_definition else None)
    run = SimpleNamespace(id="pmr_x", model_id="m", environment=environment)

    class _Q:
        def __init__(self, row): self.row = row
        def filter(self, *a, **k): return self
        def one_or_none(self): return self.row
        def first(self): return self.row
        def all(self): return []

    def query(model):
        return _Q({"ProbeMonitor": probe, "ProbeMonitorRun": run}.get(model.__name__))

    monkeypatch.setattr(rec, "propose", lambda *a, **k: {})
    out = rec.recalibrate_probe(SimpleNamespace(query=query), "pm_x", target_fpr=0.01, commit=False)
    assert out["commit_ends_exportability"] is ends
    assert out["definition_rebuildable"] is (environment.get("model_dtype") is not None)


def test_the_evaluate_endpoint_returns_409_and_queues_nothing(monkeypatch):
    """Behaviour, not line order (round 3, MED-4): a refusal is a 409 and NO task is queued."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import HTTPException

    from src.api.v1.endpoints import probe_monitors as ep
    from src.services import probe_monitor_run as pmr

    probe = SimpleNamespace(id="pm_x", run_id="pmr_x")
    run = SimpleNamespace(id="pmr_x", model_id="m", environment={"seed": 1})

    class _Q:
        def __init__(self, row): self.row = row
        def filter(self, *a, **k): return self
        def first(self): return self.row

    class _Session:
        def query(self, model):
            return _Q(probe if model.__name__ == "ProbeMonitor" else run)
        def close(self): pass

    import src.core.database as database
    monkeypatch.setattr(database, "SyncSessionLocal", lambda: _Session())
    monkeypatch.setattr(pmr, "probe_precision", lambda db, r: {"refusal": "run pmr_x predates precision recording"})
    queued = MagicMock()
    monkeypatch.setattr(ep, "gpu_delay", lambda task, gpu: queued)
    db = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: probe))

    with pytest.raises(HTTPException) as caught:
        asyncio.run(ep.evaluate_probe_endpoint("pm_x", ["pmd_1"], db=db))
    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "precision_mismatch"
    queued.assert_not_called()
