"""
Unloading the model after an extraction must be measured, not announced.

Observed 2026-08-25 on the RTX 3090. Four consecutive extractions of a 12B
model logged "Model cleaned up from GPU 0 memory" every time. The readings
underneath told a different story:

    09:48:10  after_cleanup   Allocated: 5.38 GB   Reserved: 14.02 GB
    09:49:49  after_cleanup   Allocated: 9.32 GB   Reserved: 19.55 GB
    10:31:29  after_cleanup   Allocated: 0.01 GB   Reserved:  6.99 GB

The last one freed every tensor and still never handed the 6.99 GB pool back.
`nvidia-smi` showed the worker holding 7,474 MiB nine hours later, with nothing
alive in it -- unavailable to miLLM, to the next extraction, and to the VRAM
gauge in the UI.

The success line was logged unconditionally, so none of this could surface. A
claim that measures nothing cannot report its own failure. These tests pin the
measurement, and the config that lets `empty_cache()` actually shrink the pool.
"""

from unittest.mock import patch

import tempfile
from pathlib import Path as _Path

import pytest
import yaml


def tmp_dir():
    return _Path(tempfile.mkdtemp())

from pathlib import Path

from src.services.activation_service import ActivationService


def _service():
    return ActivationService.__new__(ActivationService)


class TestItReportsWhatTheCardDid:
    def test_a_pool_that_never_came_back_is_a_warning(self, caplog):
        svc = _service()
        before = {"allocated": 7.16, "reserved": 19.72}
        after = {"allocated": 0.01, "reserved": 6.99}   # the live incident

        with patch.object(ActivationService, "_gpu_memory", staticmethod(lambda gpu_id=0: after)):
            with caplog.at_level("INFO"):
                report = svc._report_cleanup(0, before)

        assert report["after"]["reserved"] == 6.99
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, (
            "6.99 GB reserved with nothing allocated was reported as success"
        )
        assert "did not give the memory back" in warnings[0].message

    def test_a_real_release_is_not_a_warning(self, caplog):
        svc = _service()
        before = {"allocated": 7.16, "reserved": 19.72}
        after = {"allocated": 0.0, "reserved": 0.3}     # context only

        with patch.object(ActivationService, "_gpu_memory", staticmethod(lambda gpu_id=0: after)):
            with caplog.at_level("INFO"):
                report = svc._report_cleanup(0, before)

        assert report["released"] == pytest.approx(19.42)
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_memory_still_allocated_is_a_warning(self, caplog):
        """The 5.38 GB and 9.32 GB cases, which read as success for months."""
        svc = _service()
        after = {"allocated": 9.32, "reserved": 19.55}

        with patch.object(ActivationService, "_gpu_memory", staticmethod(lambda gpu_id=0: after)):
            with caplog.at_level("INFO"):
                svc._report_cleanup(0, {"allocated": 16.47, "reserved": 19.83})

        assert [r for r in caplog.records if r.levelname == "WARNING"], (
            "9.32 GB still allocated was reported as a successful unload"
        )

    def test_cleanup_returns_the_measurement(self):
        """The caller must be able to act on it, so it cannot be None."""
        svc = _service()

        class _Model:
            def parameters(self): return []
            def buffers(self): return []
            def cpu(self): return self
            def named_children(self): return []

        with patch("src.services.activation_service.torch.cuda.is_available", return_value=False), \
             patch.object(ActivationService, "_log_gpu_memory", lambda *a, **k: None):
            report = svc._cleanup_model(_Model(), gpu_id=0)

        assert isinstance(report, dict)
        assert {"before", "after", "released"} <= set(report)


class TestTheAllocatorCanActuallyShrink:
    """Measurement alone changes nothing; the pool has to be returnable."""

    def test_the_gpu_worker_uses_expandable_segments(self):
        manifest = Path(__file__).resolve().parents[3] / "k8s" / "base" / "backend.yaml"
        if not manifest.exists():          # pragma: no cover
            pytest.skip("manifest not found")

        gpu_worker_conf = None
        for doc in yaml.safe_load_all(manifest.read_text()):
            if not doc or doc.get("kind") != "Deployment":
                continue
            for c in doc["spec"]["template"]["spec"]["containers"]:
                env = {e["name"]: e.get("value") for e in c.get("env", [])}
                if (
                    env.get("SERVICE_TYPE") == "celery-worker"
                    and env.get("CELERY_WORKER_NAME") == "gpu"
                ):
                    gpu_worker_conf = env.get("PYTORCH_CUDA_ALLOC_CONF")

        assert gpu_worker_conf is not None, (
            "the GPU worker sets no PYTORCH_CUDA_ALLOC_CONF, so the default "
            "fixed-size segments cannot be handed back once fragmented"
        )
        assert "expandable_segments:True" in gpu_worker_conf


class TestAFailedLoadStillGivesTheMemoryBack:
    """The path that actually bit: an OOM DURING the load.

    `model` is bound only when `_load_model` returns, so an OOM part-way
    through arrives in the finally with several GB resident and the name still
    None. That branch used to log "no model to clean up" and return, so the
    failure kept everything it had allocated -- and the next extraction died
    against a card reporting 21.75 MiB free of 23.56 GiB.
    """

    def _service(self):
        svc = ActivationService.__new__(ActivationService)
        return svc

    def test_the_finally_releases_when_the_load_raised(self):
        from src.services.activation_service import ActivationExtractionError

        svc = self._service()
        released = []

        def _boom(*a, **k):
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 4.00 GiB. GPU 0 has a "
                "total capacity of 23.56 GiB of which 21.75 MiB is free"
            )

        with patch.object(ActivationService, "_log_gpu_memory", lambda *a, **k: None), \
             patch.object(ActivationService, "_extraction_dir", lambda self, eid: tmp_dir()), \
             patch.object(ActivationService, "_load_model", _boom), \
             patch.object(
                 ActivationService,
                 "_release_gpu_memory",
                 lambda self, gpu_id=0: released.append(gpu_id) or {},
             ):
            with pytest.raises(ActivationExtractionError):
                svc.extract_activations(
                    model_id="m1", model_path="/m", architecture="gemma3",
                    quantization=_Q(), dataset_path="/d",
                    layer_indices=[10], hook_types=["residual"],
                    max_samples=16, batch_size=8,
                )

        assert released == [0], (
            "a load that OOM'd left the card untouched -- this is the branch "
            "that logged 'no model to clean up' and returned"
        )

    def test_an_oom_does_not_carry_the_failed_load_up_the_stack(self):
        """The traceback holds the frames that hold the tensors."""
        from src.services.activation_service import ActivationExtractionError

        svc = self._service()

        def _boom(*a, **k):
            raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")

        with patch.object(ActivationService, "_log_gpu_memory", lambda *a, **k: None), \
             patch.object(ActivationService, "_extraction_dir", lambda self, eid: tmp_dir()), \
             patch.object(ActivationService, "_load_model", _boom), \
             patch.object(ActivationService, "_release_gpu_memory", lambda *a, **k: {}):
            with pytest.raises(ActivationExtractionError) as exc:
                svc.extract_activations(
                    model_id="m1", model_path="/m", architecture="gemma3",
                    quantization=_Q(), dataset_path="/d",
                    layer_indices=[10], hook_types=["residual"],
                    max_samples=16, batch_size=8,
                )

        assert exc.value.__cause__ is None, (
            "the OOM chain is still attached, so the failed load's frames stay "
            "alive and hold their tensors"
        )
        assert "out of memory" in str(exc.value).lower(), (
            "the message no longer says OOM, so the retry logic cannot "
            "classify it and will not halve the batch size"
        )

    def test_the_retry_classifier_still_sees_an_oom(self):
        """Breaking the chain must not break the smaller-batch retry."""
        from src.services.activation_service import ActivationExtractionError
        from src.workers.model_tasks import classify_extraction_error

        wrapped = ActivationExtractionError(
            "Extraction failed: CUDA out of memory. Tried to allocate 4.00 GiB"
        )
        error_type, params = classify_extraction_error(wrapped, batch_size=128)[:2]

        assert error_type == "OOM"
        assert params["batch_size"] == 64

    def test_a_non_oom_failure_keeps_its_chain(self):
        """Dropping context is a cost paid only where memory is at stake."""
        from src.services.activation_service import ActivationExtractionError

        svc = self._service()
        original = ValueError("dataset is empty")

        def _boom(*a, **k):
            raise original

        with patch.object(ActivationService, "_log_gpu_memory", lambda *a, **k: None), \
             patch.object(ActivationService, "_extraction_dir", lambda self, eid: tmp_dir()), \
             patch.object(ActivationService, "_load_model", _boom), \
             patch.object(ActivationService, "_release_gpu_memory", lambda *a, **k: {}):
            with pytest.raises(ActivationExtractionError) as exc:
                svc.extract_activations(
                    model_id="m1", model_path="/m", architecture="gemma3",
                    quantization=_Q(), dataset_path="/d",
                    layer_indices=[10], hook_types=["residual"],
                    max_samples=16, batch_size=8,
                )

        assert exc.value.__cause__ is original


class _Q:
    """Stand-in for QuantizationFormat: only `.value` is read."""

    value = "q4"
