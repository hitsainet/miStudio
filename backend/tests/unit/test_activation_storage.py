"""Captured activations reach disk exactly, or not at all (`src/ml/activation_storage.py`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ml.activation_storage import ActivationStorageOverflow, storage_dtype_for, to_storage


def test_bfloat16_to_float16_is_exact_in_range():
    """float16 carries more mantissa bits than bfloat16, so every in-range bf16 value survives."""
    torch.manual_seed(0)
    values = (torch.randn(64, 128) * 300).to(torch.bfloat16)   # Llama-3.1-8B max |x| was 326
    stored = to_storage(values)
    assert stored.dtype == np.float16
    assert np.array_equal(stored.astype(np.float32), values.to(torch.float32).numpy())


def test_a_value_beyond_float16s_range_is_refused_not_written_as_inf():
    values = torch.tensor([1.0, 7.0e4, -2.0], dtype=torch.bfloat16)
    with pytest.raises(ActivationStorageOverflow, match=r"1 activations value\(s\) exceed float16"):
        to_storage(values)


def test_a_value_the_model_already_made_non_finite_passes_through():
    """Refusing it here would turn a model defect into a storage error."""
    values = torch.tensor([1.0, float("inf"), float("nan")], dtype=torch.bfloat16)
    stored = to_storage(values)
    assert np.isinf(stored[1]) and np.isnan(stored[2])


def test_float32_stays_float32_and_float16_stays_float16():
    assert to_storage(torch.ones(3, dtype=torch.float32)).dtype == np.float32
    assert to_storage(torch.ones(3, dtype=torch.float16)).dtype == np.float16
    assert storage_dtype_for(torch.bfloat16) == "float16"
    assert storage_dtype_for(torch.float32) == "float32"


def test_an_explicit_storage_dtype_is_honoured():
    assert to_storage(torch.ones(3, dtype=torch.bfloat16), "float32").dtype == np.float32


def test_the_hook_manager_converts_bfloat16_instead_of_raising():
    """`stacked.numpy()` on a bfloat16 tensor raises TypeError; extraction called it directly."""
    from src.ml.forward_hooks import HookManager

    manager = HookManager.__new__(HookManager)
    manager.activations = {"layer_1": [torch.ones(2, 3, 4, dtype=torch.bfloat16) * 2.5]}
    out = manager.get_activations_as_numpy()
    assert out["layer_1"].dtype == np.float16
    assert np.all(out["layer_1"] == 2.5)
