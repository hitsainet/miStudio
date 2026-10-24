"""A split planner broken by a library upgrade says so loudly; an unmappable checkpoint does not.

Multi-GPU Phase 2, review round 2 (placement + loader), 2026-09-14.

`ml/split_load.plan_split_load` never refuses a split it cannot map (it returns
None, and the load maps the split itself). That is right for a checkpoint that
will not build on the meta device. But it caught EVERY exception the same way and
logged the same WARNING, including the planner's own calls into transformers' and
accelerate's internals: `compute_module_sizes`, `infer_auto_device_map`,
`get_max_layer_size`, `get_max_memory`. Those are private APIs. The image ships
whatever version it resolves (memory: dev-venv-vs-image-version-drift), so an
upgrade that renames or re-signs one would switch the planner off for EVERY split
load. The fix round 1 made (the hold-back returned to the 3080 Ti) would then be
silently gone, with a warning indistinguishable from a single odd checkpoint.

Now:
* an import of an internal that fails, or an internal that raises once the model
  is built, is logged at ERROR, naming the transformers and accelerate versions and
  saying every split load is unmapped;
* a model that will not build on meta (remote code, a missing quantization
  package) stays a WARNING, as "not verifiable here";
* both still return None: a split is never refused for what the planner cannot compute.

MUTATION CONTROLS (review round 2, 2026-09-14; scratchpad p2-r2-place/mutate.py,
each applied alone, this module + test_split_load_maps_onto_the_gpus.py run, source
restored and checked by sha256):
  L1  the import check removed                     -> test_an_internal_that_cannot_be_imported_is_an_error
  L2  post-build failures logged as unverifiable   -> test_an_internal_that_raises_on_a_built_model_is_an_error
  L3  a map-inference failure logged as unverifiable -> test_a_map_inference_that_raises_is_an_error
  L4  the build failure logged as broken (ERROR)   -> test_a_model_that_will_not_build_on_meta_is_only_a_warning
All 4 went red. Red on c8fccbcd before the fix: the first three tests (one WARNING each,
indistinguishable from an odd checkpoint). Round 1's recorded controls on the split_load
lines this moved, re-run the same way:
  R-S1  reclaim = 0             -> red (6 tests there, and both "budgets hold" tests in
                                   test_every_split_load_is_mapped_first.py)
  R-S3  sized without the quantizer -> red
  R-S7  bitsandbytes' 90% fill ignored -> red
  R-S10 a spilled reclaim refused without trying transformers' own map -> red
"""

from __future__ import annotations

import logging

import pytest
import torch

from src.ml import split_load
from tests.unit.test_split_load_maps_onto_the_gpus import _config, _sizes_mib

LOGGER = "src.ml.split_load"


def _budget():
    sizes = _sizes_mib(_config())
    half = int(sizes[""] / 2) + 600
    return {0: f"{half}MiB", 1: f"{half}MiB"}


def _records(caplog, level):
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == level]


def test_an_internal_that_cannot_be_imported_is_an_error(monkeypatch, caplog):
    import transformers.integrations.accelerate as accelerate_integration

    monkeypatch.delattr(accelerate_integration, "compute_module_sizes")
    caplog.set_level(logging.INFO, logger=LOGGER)

    plan = split_load.plan_split_load(_config(), max_memory=_budget(), dtype=torch.float16)

    assert plan is None, "a planner that cannot run must never refuse a split"
    errors = _records(caplog, logging.ERROR)
    assert len(errors) == 1, [r.getMessage() for r in caplog.records]
    message = errors[0].getMessage()
    import accelerate
    import transformers

    assert transformers.__version__ in message and accelerate.__version__ in message, message
    assert "every split" in message.lower(), message


def test_an_internal_that_raises_on_a_built_model_is_an_error(monkeypatch, caplog):
    import transformers.integrations.accelerate as accelerate_integration

    def resigned(*args, **kwargs):
        raise TypeError("compute_module_sizes() got an unexpected keyword argument 'only_modules'")

    monkeypatch.setattr(accelerate_integration, "compute_module_sizes", resigned)
    caplog.set_level(logging.INFO, logger=LOGGER)

    assert split_load.plan_split_load(_config(), max_memory=_budget(), dtype=torch.float16) is None

    assert len(_records(caplog, logging.ERROR)) == 1, [r.getMessage() for r in caplog.records]


def test_a_map_inference_that_raises_is_an_error(monkeypatch, caplog):
    def broken(model, max_memory, quantizer):
        raise AttributeError("module 'transformers.integrations.accelerate' has no attribute 'get_max_memory'")

    monkeypatch.setattr(split_load, "_infer_map", broken)
    caplog.set_level(logging.INFO, logger=LOGGER)

    assert split_load.plan_split_load(_config(), max_memory=_budget(), dtype=torch.float16) is None

    assert len(_records(caplog, logging.ERROR)) == 1, [r.getMessage() for r in caplog.records]


def test_a_model_that_will_not_build_on_meta_is_only_a_warning(monkeypatch, caplog):
    def unbuildable(*args, **kwargs):
        raise RuntimeError("this architecture cannot be built on meta here")

    monkeypatch.setattr(split_load, "_skeleton", unbuildable)
    caplog.set_level(logging.INFO, logger=LOGGER)

    assert split_load.plan_split_load(_config(), max_memory=_budget(), dtype=torch.float16) is None

    assert _records(caplog, logging.ERROR) == []
    assert len(_records(caplog, logging.WARNING)) == 1


def test_a_split_the_planner_maps_logs_no_warning_or_error(caplog):
    """The healthy path stays quiet, so an ERROR above means something."""
    caplog.set_level(logging.INFO, logger=LOGGER)

    assert split_load.plan_split_load(_config(), max_memory=_budget(), dtype=torch.float16) is not None

    assert [r for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize("name", ["compute_module_sizes", "get_max_memory", "infer_auto_device_map"])
def test_the_internals_the_planner_uses_exist_in_this_transformers(name):
    """A plain CI tripwire for an upgrade: each private name the planner imports resolves."""
    import transformers.integrations.accelerate as accelerate_integration

    assert callable(getattr(accelerate_integration, name, None)), name
