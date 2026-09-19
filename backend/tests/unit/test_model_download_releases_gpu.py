"""The download task must give the GPU back.

`download_and_load_model` loads a model for one reason: to read `architecture`,
`params_count` and `architecture_config` off it for the database. It uses
`device_map="auto"`, so that inspection happens on the GPU. Nothing downstream
wants the weights resident — training and extraction each load what they need.

It never released them. Downloading LFM2.5-2.6B (2,697,198,592 params at FP32)
left 10,696 MiB held on a 24 GB card, and it stays until the worker restarts,
which is days. The extraction path already calls `empty_cache()` with the
comment "in case previous task didn't complete cleanup" — this leak, worked
around at the far end, where it can only help extractions and never the serving
process that actually wants the card.

ASSERTED ON THIS BLOCK'S OWN LOG LINE, not on `empty_cache` having been called
by somebody. The first version of this test watched a torch stub for any
`empty_cache` call and passed against a build with the release disabled: an
emergency-cleanup handler elsewhere in the worker had called it, so the test was
measuring that instead. A shared observable proves nothing about which code path
reached it.
"""

import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch as real_torch

from src.services.gpu_placement import GpuCard, Placement

RELEASED = "Released the download-inspection model from GPU memory"


def _torch_stub(calls: list) -> types.ModuleType:
    import contextlib

    stub = types.ModuleType("torch")
    stub.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.append("empty_cache"),
        memory_reserved=lambda *a, **k: 0,
        # The release runs inside each placed card's device context.
        device=lambda device: contextlib.nullcontext(),
    )
    return stub


def _run(load_side_effect, caplog, tmp_path):
    """Drive the real task body with everything external stubbed out.

    THESE TESTS NEVER REACHED THE LOAD until 2026-09-13. They passed "fp32",
    which is not a QuantizationFormat value, and the model lookup returned None,
    so the task died on its first lines and both tests watched the `finally`
    release after an early failure — "a SUCCESSFUL load releases too" never
    loaded anything. The load-reached assertion below is what exposed it.
    """
    from src.workers import model_tasks

    calls: list = []
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = MagicMock()

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=db)
    ctx.__exit__ = MagicMock(return_value=False)

    # The job is PLACED before the load. Without this fake, placement fails on a
    # machine with no GPU, the load is never reached, and the release is
    # observed after a refusal instead of after the load these tests are about.
    placement = Placement(
        card=GpuCard(
            index=1, uuid="GPU-247aa582-0d1b-e161-8156-983ed1fefc57",
            name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000,
        ),
        device=real_torch.device("cuda", 1),
    )
    load = MagicMock(side_effect=load_side_effect)

    with patch.dict(sys.modules, {"torch": _torch_stub(calls)}), \
         patch("src.workers.base_task.DatabaseTask.get_db", return_value=ctx), \
         patch("src.services.gpu_placement.place_job", return_value=placement), \
         patch.object(model_tasks, "required_mb_for_load", MagicMock(return_value=None)), \
         patch.object(model_tasks, "load_model_from_hf", load), \
         patch.object(model_tasks, "send_progress_update", MagicMock()), \
         patch.object(model_tasks, "DownloadProgressMonitor", MagicMock()), \
         patch.object(model_tasks, "clear_cancel_request", MagicMock()), \
         patch.object(model_tasks, "cancel_checker", MagicMock()), \
         patch.object(model_tasks, "settings", types.SimpleNamespace(models_dir=tmp_path)), \
         caplog.at_level(logging.INFO, logger="src.workers.model_tasks"):
        try:
            model_tasks.download_and_load_model.run("m_1", "vendor/model", "FP32")
        except Exception:
            pass
    assert load.call_count == 1, "the load was never reached, so the release was not tested after it"
    return calls, caplog.text


class TestTheCardIsGivenBack:
    def test_a_FAILED_load_still_releases(self, caplog, tmp_path):
        """The path most likely to run on a card that is already full.

        A cleanup in `finally` is worth nothing if the failing path skips it,
        and a failed load is exactly when the card is most likely to be full.
        """
        calls, log = _run(RuntimeError("boom"), caplog, tmp_path)

        assert RELEASED in log, (
            "the load failed and this task's release never ran — a partially "
            "allocated model stays on the card until the worker restarts"
        )
        assert "empty_cache" in calls

    def test_a_SUCCESSFUL_load_releases_too(self, caplog, tmp_path):
        """The path that actually leaked the 10.7 GB."""
        meta = {
            "architecture": "lfm2",
            "params_count": 2_697_198_592,
            "architecture_config": {},
            "memory_required_bytes": 1,
            "quantization": "FP32",
        }
        calls, log = _run(
            lambda **kw: (MagicMock(), MagicMock(), MagicMock(), meta), caplog, tmp_path
        )

        assert RELEASED in log, (
            "the download succeeded and the weights were never handed back"
        )
        assert "empty_cache" in calls
