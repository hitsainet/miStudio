"""Every endpoint that resolves a job's GPU says whether the job can run split.

Multi-GPU Phase 2, review round 2 (placement + loader), 2026-09-14.

`resolve_gpu_request(requested, *, can_split=True)` refused "all" with a 400 at
submit only when a caller passed `can_split=False` (3da80d83 + be889163). The
default was True, so it failed OPEN: a new endpoint for a job that runs on one
card, written without the flag, would accept "all", create its row, queue the
job behind whatever holds the GPU, and be refused only when a worker placed it.
That is the exact defect those commits fixed for the J-lens fit, cached-activation
training and the Neuronpedia logit lens.

`can_split` is now a REQUIRED keyword with no default. An endpoint that omits it
raises TypeError on its first request, and this module fails first: the table
below is exhaustive, so a new call site fails here until it is classified.

Why required, not `False` by default: False would also fail closed, but silently
the other way. A new endpoint for a job that DOES split would refuse "all" and
nobody would notice. Required forces the decision where the endpoint is written.

Each entry is the `can_split` expression the call passes, and each True names the
worker that places its job with `allow_shard=True`:
  circuit capture / confirm      workers/circuit_capture_tasks.py (capture)
  attribution                    workers/circuit_capture_tasks.py (attribution)
  validation / its reproduction  workers/circuit_validation_tasks.py
  faithfulness                   workers/circuit_validation_tasks.py
  calibration / its reproduction workers/circuit_calibration_tasks.py
  steering samples               workers/circuit_record_tasks.py
  J-lens revalidate, band report, readout, probe, acquire, intervention
                                 workers/jlens_{fit,band,readout,probe,acquire,intervention}_tasks.py
  local labeling judge           services/labeling_service.py
  model download / redownload    workers/model_tasks.py (download_and_load_model)
  activation extraction / retry  workers/model_tasks.py (extract_activations)
  SAE feature extraction (one, batch) services/extraction_service.py
  steering compare / sweep / combined services/steering_service.py (place_model)
  training evaluation re-run     workers/training_evaluation_tasks.py (evaluate_training)
The J-lens fit places without allow_shard. Training splits only when it loads a base
model. The Neuronpedia jobs need a GPU only for the logit lens.

MUTATION CONTROLS (review round 2, 2026-09-14; scratchpad p2-r2-place/mutate.py,
each applied alone, this module + test_all_is_refused_where_a_job_cannot_split.py
run, source restored and checked by sha256):
  G1  `can_split: bool = True` restored as the default  -> test_can_split_has_no_default
  G2  one call site drops its can_split (steering compare) -> test_every_call_site_is_classified
  G2  applied to saes.py's single-SAE extraction (steering's three calls share one spelling)
  G3  the fit's can_split=False becomes True            -> test_every_call_site_is_classified,
                                                           test_jlens_gpu_placement::TestAFitRefusesAll...
All 3 went red (G1 also reddens test_a_call_without_can_split_fails_loudly). Round 1's
controls on the lines this touched, re-run the same way:
  R-A1 the helper never refuses "all"            -> red (6 tests in test_all_is_refused...)
  R-A4 dashboard data always may split           -> red (the dashboard test + this module's table)
  R-A7 resolve_card looks "all" up as a UUID     -> red
Red on c8fccbcd before the fix: the first three tests here.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

#: (module under src/, enclosing function) -> the `can_split` expression it passes.
EXPECTED = {
    ("api/v1/endpoints/circuit_discovery.py", "create_capture"): "True",
    ("api/v1/endpoints/circuit_discovery.py", "confirm_capture"): "True",
    ("api/v1/endpoints/circuit_discovery.py", "start_attribution"): "True",
    ("api/v1/endpoints/circuit_validation.py", "start_validation"): "True",
    ("api/v1/endpoints/circuit_validation.py", "reproduce_manifest"): "True",
    ("api/v1/endpoints/circuits.py", "start_faithfulness"): "True",
    ("api/v1/endpoints/circuits.py", "start_calibration"): "True",
    ("api/v1/endpoints/circuits.py", "reproduce_calibration"): "True",
    ("api/v1/endpoints/circuits.py", "start_steering_samples"): "True",
    ("api/v1/endpoints/jlens.py", "fit"): "False",
    ("api/v1/endpoints/jlens.py", "revalidate_staged"): "True",
    ("api/v1/endpoints/jlens.py", "compute_band_report"): "True",
    ("api/v1/endpoints/jlens.py", "readout"): "True",
    ("api/v1/endpoints/jlens.py", "probe"): "True",
    ("api/v1/endpoints/jlens.py", "acquire_artifact"): "True",
    ("api/v1/endpoints/jlens.py", "run_intervention"): "True",
    ("api/v1/endpoints/labeling.py", "judge_config"): "True",
    ("api/v1/endpoints/models.py", "download_model"): "True",
    ("api/v1/endpoints/models.py", "redownload_model"): "True",
    ("api/v1/endpoints/models.py", "extract_model_activations"): "True",
    ("api/v1/endpoints/models.py", "retry_extraction"): "True",
    ("api/v1/endpoints/neuronpedia.py", "start_export"): "not request.config.include_logit_lens",
    ("api/v1/endpoints/neuronpedia.py", "compute_dashboard_data"): "not request.include_logit_lens",
    ("api/v1/endpoints/neuronpedia.py", "push_to_local_neuronpedia"): "not compute_dashboard_data",
    ("api/v1/endpoints/saes.py", "start_sae_extraction"): "True",
    ("api/v1/endpoints/saes.py", "start_batch_sae_extraction"): "True",
    ("api/v1/endpoints/steering.py", "submit_async_steering_comparison"): "True",
    ("api/v1/endpoints/steering.py", "submit_async_strength_sweep"): "True",
    ("api/v1/endpoints/steering.py", "submit_async_combined_steering"): "True",
    # Phase 3 review round 1: enter- and exit-mode resolve which CARDS' steering
    # workers to start or stop, not where a job places. "all" there means every
    # card's worker, so it must not be refused; the generations those workers run
    # place through services/steering_service.py (place_model), as above.
    ("api/v1/endpoints/steering.py", "_enter_card_steering_mode"): "True",
    ("api/v1/endpoints/steering.py", "exit_steering_mode"): "True",
    ("api/v1/endpoints/trainings.py", "create_training"): "extraction_ids_of(training) is None",
    # SAE training remediation item 6: re-running a completed training's evaluation
    # loads its base model; workers/training_evaluation_tasks.py places with
    # allow_shard=True and feeds each SAE on its own card.
    ("api/v1/endpoints/trainings.py", "evaluate_training"): "True",
    # 032. A probe run places on ONE card: `probe_monitor_run.py` calls `place_job`
    # without `allow_shard`, and its capture loop hooks a single model. `can_split=False`
    # is what turns an "all" request into a 400 at SUBMIT rather than a 202 for work
    # only the worker could refuse.
    ("api/v1/endpoints/probe_monitors.py", "submit_probe_run"): "False",
}


def _call_sites() -> dict:
    """Every call to resolve_gpu_request in src/, keyed like EXPECTED; None when it passes no can_split."""
    found: dict = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name != "resolve_gpu_request":
                    continue
                keyword = next((k for k in node.keywords if k.arg == "can_split"), None)
                key = (str(path.relative_to(SRC)), func.name)
                assert key not in found, f"{key} calls resolve_gpu_request twice; classify each"
                found[key] = None if keyword is None else ast.unparse(keyword.value)
    return found


def test_can_split_has_no_default():
    from src.api.v1.gpu_request import resolve_gpu_request

    parameter = inspect.signature(resolve_gpu_request).parameters["can_split"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty, (
        "can_split has a default: an endpoint that forgets it accepts or refuses 'all' silently"
    )


def test_a_call_without_can_split_fails_loudly():
    from src.api.v1.gpu_request import resolve_gpu_request

    with pytest.raises(TypeError):
        resolve_gpu_request("auto")


def test_every_call_site_is_classified():
    assert _call_sites() == EXPECTED


def test_the_walk_sees_the_call_sites():
    """Not vacuous: the J-lens fit and the trainings route are found by the walk itself."""
    sites = _call_sites()
    assert sites[("api/v1/endpoints/jlens.py", "fit")] == "False"
    assert len(sites) >= 30
