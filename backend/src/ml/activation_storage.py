"""Captured activations to their on-disk dtype — exactly, or not at all.

⚠ WHY THIS EXISTS. Models now load at the checkpoint's own dtype (`ml/native_dtype.py`), which
for every model this estate runs is bfloat16. numpy has no bfloat16, so `tensor.numpy()` raises on
a bf16 activation — and the obvious repair, casting to float16, can OVERFLOW: float16 tops out at
65,504 while bfloat16 reaches ~3.4e38, and an overflowed value is written as `inf` with no error.
An `inf` in an SAE's training data or a probe's token cache is a silent corruption that surfaces,
if ever, as a NaN loss hours later.

THE RULE. A 16-bit load stores float16 — bf16 → float16 is EXACT for 6.1e-5 ≤ |x| ≤ 65,504,
because float16 carries more mantissa bits than bfloat16 — so disk cost and every reader stay as
they were. Below 6.1e-5 float16 is subnormal and rounds (absolute error under 6e-8, flushing to 0
below 2^-24): negligible beside residual-stream values, but not "exact", and said so here because
review round 1 caught the earlier claim. A
float32 load stores float32. A value that would become non-finite in the conversion is REFUSED,
naming how many and the largest magnitude, never written. (Llama-3.1-8B measured 2026-10-03:
max |x| 326 at resid_post L11/16/21, so this refusal is a guard, not an expected path.)

Values that were ALREADY non-finite before conversion are the model's own and are passed through
unchanged: refusing them here would move a model defect into a storage error.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

#: The largest finite float16.
FLOAT16_MAX = float(np.finfo(np.float16).max)

STORAGE_DTYPES = {"float16": np.float16, "float32": np.float32}


class ActivationStorageOverflow(ValueError):
    """A captured value would become non-finite at the storage dtype."""


def storage_dtype_for(tensor_dtype: torch.dtype) -> str:
    """The storage dtype a captured tensor's dtype implies: float32 stays float32, 16-bit → float16."""
    return "float32" if tensor_dtype == torch.float32 else "float16"


def to_storage(
    values: Union[torch.Tensor, np.ndarray],
    storage_dtype: Optional[str] = None,
    *,
    what: str = "activations",
) -> np.ndarray:
    """Convert to a numpy array at `storage_dtype`, refusing any value the conversion makes non-finite.

    `storage_dtype` defaults to the one the tensor's own dtype implies (`storage_dtype_for`).
    """
    if isinstance(values, torch.Tensor):
        if storage_dtype is None:
            storage_dtype = storage_dtype_for(values.dtype)
        source = values.detach().to("cpu")
        if source.dtype == torch.bfloat16 or source.dtype.is_floating_point is False:
            source = source.to(torch.float32)
        array = source.numpy()
    else:
        array = np.asarray(values)
        if storage_dtype is None:
            storage_dtype = "float32" if array.dtype == np.float32 else "float16"
    if storage_dtype not in STORAGE_DTYPES:
        raise ValueError(f"unknown storage dtype {storage_dtype!r}; known: {sorted(STORAGE_DTYPES)}")
    target = STORAGE_DTYPES[storage_dtype]
    if array.dtype == target:
        return array
    converted = array.astype(target)
    if target == np.float16 and array.size:
        became_non_finite = ~np.isfinite(converted) & np.isfinite(array)
        count = int(became_non_finite.sum())
        if count:
            largest = float(np.abs(array[became_non_finite]).max())
            raise ActivationStorageOverflow(
                f"{count} {what} value(s) exceed float16's range (largest |x| = {largest:.4g}, "
                f"limit {FLOAT16_MAX:.0f}) and would be written as inf. Store this model's "
                "activations at float32 instead; nothing was written."
            )
    return converted
