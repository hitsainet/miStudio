"""Two evaluations of one request write the same bytes (Feature 041 FR-15; T27, T29/G2; FTASKS 6.8, 8.6).

Seeded folds, seeded fits under ONE BLAS thread, float64 throughout, rounded and canonicalised by JCS
(`report_format.canonical_bytes`). The golden fold assignment is pinned to the scikit-learn version in the
file (an upgrade that reshuffles folds must be a decision, not a drift).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from src.services.probe_blend import engine
from src.services.probe_blend.report_format import canonical_bytes, round_floats
from tests.support import blend_fixtures as bf

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "blend" / "golden_fold_assignment.json"


def summary_bytes(seed: int) -> bytes:
    out = engine.run(bf.build(seed))
    out.pop("rows")
    return canonical_bytes(out)


def test_two_runs_byte_identical():
    first, second = engine.run(bf.build(21)), engine.run(bf.build(21))
    for a, b in zip(first["candidates"], second["candidates"]):
        if "weights" in a["params"]:
            assert np.array_equal(np.asarray(a["params"]["weights"]), np.asarray(b["params"]["weights"]))
    for view in first["rows"]:
        for index, values in first["rows"][view]["values"].items():
            assert np.array_equal(values, second["rows"][view]["values"][index])
    first.pop("rows"), second.pop("rows")
    assert canonical_bytes(first) == canonical_bytes(second)


def test_canonical_bytes_sort_keys_and_round():
    assert canonical_bytes({"b": 1.23456789, "a": True, "c": [1, 2.0000004]}) == b'{"a":true,"b":1.234568,"c":[1,2]}'
    assert round_floats({"x": np.float64(0.1234567)}) == {"x": 0.123457}


def test_golden_fold_assignment():
    import sklearn

    golden = json.loads(GOLDEN.read_text())
    assert sklearn.__version__ == golden["scikit_learn"], (
        f"scikit-learn moved from {golden['scikit_learn']} to {sklearn.__version__}: re-check the fold assignment "
        "deliberately and re-pin this file")
    out = engine.run(bf.build(golden["seed"]))
    assert out["folds"]["assignment_digest"] == golden["assignment_digest"]


def test_the_stack_fits_under_one_blas_thread(monkeypatch):
    import threadpoolctl

    seen = []
    real = threadpoolctl.threadpool_limits

    def spy(*args, **kwargs):
        seen.append(kwargs.get("limits", args[0] if args else None))
        return real(*args, **kwargs)

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", spy)
    engine.fit_stack(np.random.default_rng(0).normal(size=(50, 2)), np.arange(50) % 2, 0)
    assert seen == [1]


def test_concurrent_equals_sequential():
    """G2 fallback: two processes at once give the bytes one process gives in turn."""
    sequential = [summary_bytes(31), summary_bytes(32)]
    import multiprocessing

    # spawn: a fresh interpreter per worker, as a prefork child sees no parent thread state.
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        concurrent = list(pool.map(summary_bytes, [31, 32]))
    assert concurrent == sequential
