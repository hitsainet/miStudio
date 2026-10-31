"""An activation extraction whose MODEL LOAD runs out of memory is not told to halve its batch.

Review round 3 (2026-09-14), a pre-existing defect round 2 left open. `extract_activations`
wrapped every out-of-memory failure, the load's included, in the same
`ActivationExtractionError`, and `classify_extraction_error` matched "out of memory" in it
and suggested `batch_size // 2`. No batch had run: a model that does not fit is not helped
by a smaller one, and the suggestion sent the user to retry the same failure.

A load that runs out of memory now raises `ModelLoadOutOfMemory` (still an
`ActivationExtractionError`), classified `MODEL_LOAD_OOM` with no retry parameters. An
out-of-memory failure after the load is the batch's, as before.

MUTATION CONTROLS (review round 3, 2026-09-14; scratchpad p2-r3/mutate.py + mutations_r3.json,
each alone in a private copy of backend/, restored by sha256; run after the rebase onto the Phase 3
round 2 merge, which also changed model_tasks.py). All killed:
  N41 every out-of-memory failure raised as the plain error   -> test_a_load_that_runs_out_of_memory_is_a_model_load_failure
  N42 the load's return never recorded                        -> test_a_batch_that_runs_out_of_memory_after_the_load_...
  N43 classify_extraction_error's model-load branch removed   -> test_a_load_that_runs_out_of_memory_is_a_model_load_failure
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.services import activation_service
from src.services.activation_service import ActivationExtractionError, ActivationService, ModelLoadOutOfMemory
from src.services.gpu_placement import GpuCard, Placement
from src.workers.model_tasks import classify_extraction_error

RTX = Placement(card=GpuCard(index=1, uuid="GPU-247aa582-0d1b-e161-8156-983ed1fefc57",
                             name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
                device=torch.device("cuda", 1))


class _Model:
    def eval(self):
        return self


@pytest.fixture
def service(monkeypatch, tmp_path):
    monkeypatch.setattr(ActivationService, "_extraction_dir", lambda self, extraction_id: tmp_path)
    monkeypatch.setattr(ActivationService, "_release_gpu_memory", lambda self, devices: {})
    monkeypatch.setattr(ActivationService, "_cleanup_model", lambda self, model, devices: None)
    monkeypatch.setattr(ActivationService, "_log_gpu_memory", lambda self, stage, devices: None)
    monkeypatch.setattr(ActivationService, "_load_dataset", lambda self, path, max_samples: [])
    monkeypatch.setattr(activation_service, "_dataset_looks_packed", lambda dataset: False)
    return ActivationService.__new__(ActivationService)


def _extract(service):
    return service.extract_activations(
        model_id="m_1", model_path="/m", architecture="lfm2", quantization=SimpleNamespace(value="FP16"),
        dataset_path="/d", layer_indices=[11], hook_types=["residual"], max_samples=4, placement=RTX,
    )


def _out_of_memory(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def test_a_load_that_runs_out_of_memory_is_a_model_load_failure(service, monkeypatch):
    monkeypatch.setattr(ActivationService, "_load_model", _out_of_memory)

    with pytest.raises(ModelLoadOutOfMemory) as failure:
        _extract(service)

    assert classify_extraction_error(failure.value, batch_size=32) == ("MODEL_LOAD_OOM", {})


def test_a_batch_that_runs_out_of_memory_after_the_load_still_suggests_a_smaller_batch(service, monkeypatch):
    monkeypatch.setattr(ActivationService, "_load_model", lambda self, *a, **k: (_Model(), object()))
    monkeypatch.setattr(ActivationService, "_run_extraction", _out_of_memory)

    with pytest.raises(ActivationExtractionError) as failure:
        _extract(service)

    assert not isinstance(failure.value, ModelLoadOutOfMemory)
    assert classify_extraction_error(failure.value, batch_size=32) == ("OOM", {"batch_size": 16})
