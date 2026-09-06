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

RELEASED = "Released the download-inspection model from GPU memory"


def _torch_stub(calls: list) -> types.ModuleType:
    stub = types.ModuleType("torch")
    stub.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.append("empty_cache"),
        memory_reserved=lambda *a, **k: 0,
    )
    return stub


def _run(load_side_effect, caplog):
    """Drive the real task body with everything external stubbed out."""
    from src.workers import model_tasks

    calls: list = []
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=db)
    ctx.__exit__ = MagicMock(return_value=False)

    with patch.dict(sys.modules, {"torch": _torch_stub(calls)}), \
         patch("src.workers.base_task.DatabaseTask.get_db", return_value=ctx), \
         patch.object(model_tasks, "load_model_from_hf", side_effect=load_side_effect), \
         patch.object(model_tasks, "send_progress_update", MagicMock()), \
         caplog.at_level(logging.INFO, logger="src.workers.model_tasks"):
        try:
            model_tasks.download_and_load_model.run("m_1", "vendor/model", "fp32")
        except Exception:
            pass
    return calls, caplog.text


class TestTheCardIsGivenBack:
    def test_a_FAILED_load_still_releases(self, caplog):
        """The path most likely to run on a card that is already full.

        A cleanup in `finally` is worth nothing if the failing path skips it,
        and a failed load is exactly when the card is most likely to be full.
        """
        calls, log = _run(RuntimeError("boom"), caplog)

        assert RELEASED in log, (
            "the load failed and this task's release never ran — a partially "
            "allocated model stays on the card until the worker restarts"
        )
        assert "empty_cache" in calls

    def test_a_SUCCESSFUL_load_releases_too(self, caplog):
        """The path that actually leaked the 10.7 GB."""
        meta = {
            "architecture": "lfm2",
            "params_count": 2_697_198_592,
            "architecture_config": {},
            "memory_required_bytes": 1,
            "quantization": "FP32",
        }
        calls, log = _run(
            lambda **kw: (MagicMock(), MagicMock(), MagicMock(), meta), caplog
        )

        assert RELEASED in log, (
            "the download succeeded and the weights were never handed back"
        )
        assert "empty_cache" in calls
