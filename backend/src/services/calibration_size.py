"""How many negatives a probe's bar was cut from, and whether that is enough (Feature 040 FR-19; 040-P4).

A target false-positive rate is a count: at 1% of 540 negatives the bar may let through 5 rows, and one
noisy row moves it by a fifth. Every calibration therefore states its size and the sentence a person
reads ("1% of 540 = 5 rows"), and is flagged `few_negatives` when it is below the floor.

The floor is the behaviour's own `calibration_floor` when its definition sets one, else the runtime
setting `calibration_floor_negatives` (default 2,000, spec D12); the source is recorded with every
judgement (040-P4). A probe calibrated before 040 S2 has no recorded count: it is derived from the length
of its stored negatives array and labelled `n_negatives_source: "array_length"`, or null when no array
exists (never guessed).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _array_length(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        import numpy as np

        return int(np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1).shape[0])
    except Exception:  # noqa: BLE001 - a missing array is "not recorded", stated below
        return None


def _pct(rate: float) -> str:
    text = f"{rate * 100:.4f}".rstrip("0").rstrip(".")
    return f"{text}%"


def calibration_size(probe: Any, floor_override: Optional[int], setting_floor: int,
                     target_fpr: Optional[float] = None) -> Dict[str, Any]:
    recorded = getattr(probe, "calibration_n_negatives", None)
    if recorded is not None:
        n, source = int(recorded), "recorded"
    else:
        n = _array_length(getattr(probe, "calibration_scores_path", None))
        source = "array_length" if n is not None else None
    floor, floor_source = (int(floor_override), "behaviour") if floor_override is not None else (int(setting_floor), "setting")
    rate = target_fpr if target_fpr is not None else getattr(probe, "target_fpr", None)
    allowed = int(rate * n) if (rate is not None and n is not None) else None
    sentence = (
        f"{_pct(rate)} of {n:,} = {allowed:,} row{'s' if allowed != 1 else ''}"
        if allowed is not None else None
    )
    return {
        "view_id": getattr(probe, "calibration_dataset_id", None),
        "threshold_source": getattr(probe, "threshold_source", None),
        "n_negatives": n,
        "n_negatives_source": source,
        "floor": floor,
        "floor_source": floor_source,
        # ⚠ `>=` IS ENOUGH: a calibration of exactly the floor is not few.
        "few_negatives": (n < floor) if n is not None else None,
        "target_fpr": rate,
        "allowed_false_positives": allowed,
        "sentence": sentence,
    }
