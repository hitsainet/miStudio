"""A J-lens task is placed for its weights AND the memory its forward pass needs.

Multi-GPU Phase 2, review round 2 (2026-09-14, steering reviewer; review item C).
Every J-lens task (fit, revalidation, band report, readout, probe, intervention,
acquisition) placed itself with `required_mb=estimate_weights_mb(record)`: the
weights alone. Auto took the card whose free memory just held them, and the first
forward pass had nowhere to go. Circuits (`circuit_required_mb`) and steering
(`_placement_need_mb`) place with the loader preflight's 2 GiB of activation
headroom on top; `jlens_progress.place_on_card`, the one placement every J-lens
task goes through, now adds the same.

The behavioural case uses the real `place_job` over the node's two cards: a model
whose weights the 3090's free memory holds, but not with any room to run.

MUTATION CONTROLS (review round 2, 2026-09-14; scratchpad p2-r2-steer/mutate.py with
mutations_c.json, each alone, this module + test_jlens_gpu_placement.py +
test_jlens_split_gpu.py run, restored by sha256). Both killed, 10 tests red each:
  C1 the headroom line reverted (needed_mb = required_mb)   -> three cases here, every task case of
       test_jlens_gpu_placement::test_the_task_says_whether_it_may_split_and_how_big_its_model_is,
       test_jlens_split_gpu::test_a_fresh_readout_may_split_and_says_how_big_the_model_is
  C2 place_job handed the bare weights (required_mb=required_mb) -> the same ten
The unfixed code is C1: red on it before the fix, 10 tests.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.services import gpu_placement
from src.services.gpu_placement import GpuCard, Placement
from src.workers import jlens_progress

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000)
RTX = GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000)

#: Weights the 3090's 23,000 MB free holds with 500 MB to spare.
WEIGHTS_MB = 22_500.0
#: 2 GiB of headroom, by hand: the loader preflight's `_ACTIVATION_HEADROOM_GB`.
HEADROOM_MB = 2_048.0


def test_the_placement_is_asked_for_the_weights_plus_the_preflights_headroom():
    asked = []

    def place_job(requested, required_mb=None, allow_shard=False):
        asked.append(required_mb)
        return Placement(card=RTX, device=torch.device("cuda", 1))

    with patch.object(gpu_placement, "place_job", place_job), \
         patch.object(jlens_progress, "record_gpu", lambda *a, **k: True):
        jlens_progress.place_on_card("t-1", "auto", required_mb=WEIGHTS_MB, allow_shard=True)

    assert asked == [WEIGHTS_MB + HEADROOM_MB]


def test_a_task_that_cannot_be_sized_is_placed_without_a_size():
    asked = []

    def place_job(requested, required_mb=None, allow_shard=False):
        asked.append(required_mb)
        return Placement(card=RTX, device=torch.device("cuda", 1))

    with patch.object(gpu_placement, "place_job", place_job), \
         patch.object(jlens_progress, "record_gpu", lambda *a, **k: True):
        jlens_progress.place_on_card("t-1", "auto", required_mb=None, allow_shard=True)

    assert asked == [None]


@contextlib.contextmanager
def _two_cards():
    order = [TI_UUID[len("GPU-"):], RTX_UUID[len("GPU-"):]]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.is_available", return_value=True))
        stack.enter_context(patch("torch.cuda.device_count", return_value=2))
        stack.enter_context(patch("torch.cuda.get_device_properties",
                                  side_effect=lambda index: SimpleNamespace(uuid=order[index])))
        stack.enter_context(patch("torch.cuda.set_device"))
        stack.enter_context(patch.object(gpu_placement, "list_cards", lambda: [TI, RTX]))
        stack.enter_context(patch.object(jlens_progress, "record_gpu", lambda *a, **k: True))
        yield


def test_a_model_whose_weights_just_fit_one_card_is_not_placed_there_alone():
    with _two_cards():
        weights_only = gpu_placement.place_job("auto", required_mb=WEIGHTS_MB, allow_shard=True)
        placement = jlens_progress.place_on_card("t-1", "auto", required_mb=WEIGHTS_MB, allow_shard=True)

    assert not weights_only.is_shard and weights_only.uuid == RTX_UUID, (
        "precondition: sized at its weights alone, Auto puts the model on the 3090 by itself"
    )
    # 24,548 MB: no card holds it, so it splits across the 3090 and the 3080 Ti
    # (21,976 + 9,976 MB of split budget).
    assert placement.is_shard
    assert placement.uuids == [RTX_UUID, TI_UUID]


def test_a_task_that_may_not_split_is_refused_rather_than_run_out_of_memory():
    """The fit may not split. Sized at its weights it ran on the 3090 and ran out of
    memory in its first pass; with its headroom it is refused with the figures."""
    with _two_cards(), pytest.raises(gpu_placement.GpuPlacementError, match="No single GPU"):
        jlens_progress.place_on_card("t-fit", "auto", required_mb=WEIGHTS_MB)
