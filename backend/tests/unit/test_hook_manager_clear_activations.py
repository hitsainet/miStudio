"""HookManager.clear_activations releases every captured tensor, however many a hook stored.

PRE-EXISTING DEFECT (reported by WS-DATA, fixed in review R1-B, 2026-09-15). The method
deleted `activation_list[i]` for i in range(len(list)); each delete shifts the list, so any
layer holding two or more captures raised IndexError half-way through. Its callers
(activation extraction, extraction_service, circuit capture) clear after every batch, and a
module that fires twice in one forward, or two batches captured before a clear, reached it.

MUTATION CONTROL (R1-B C9): restoring the index-delete loop -> both tests fail with IndexError.
"""

import gc
import weakref

import torch

from src.ml.forward_hooks import HookManager


def _manager_with(activations):
    manager = HookManager.__new__(HookManager)  # no model needed to hold captures
    manager.activations = activations
    return manager


def test_a_layer_holding_several_captures_is_cleared():
    manager = _manager_with({
        "layer_1_residual": [torch.zeros(2), torch.zeros(2), torch.zeros(2)],
        "layer_2_residual": [torch.zeros(2), torch.zeros(2)],
    })
    manager.clear_activations()
    assert manager.activations == {}


def test_every_captured_tensor_is_released():
    tensors = [torch.zeros(4) for _ in range(5)]
    refs = [weakref.ref(t) for t in tensors]
    manager = _manager_with({"layer_0_mlp": tensors})
    del tensors
    manager.clear_activations()
    gc.collect()
    assert all(ref() is None for ref in refs), "a cleared capture is still referenced"
