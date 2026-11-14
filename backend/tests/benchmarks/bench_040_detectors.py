"""Feature 040 benchmarks (FTDD §9, 040-T37; FTASKS 14.4, 15.7). NOT a test: not collected by `tests/unit`, and
no wall-clock assertion anywhere — targets are recorded in the controls review, and a cap the measurement
contradicts returns to the operator (D32).

Run on the deployed `celery-worker-cpu` container in a quiet window (no GPU job running):

    python -m tests.benchmarks.bench_040_detectors --rows 30000 --values 100 --filters 2

Measured here: the breakdown (`probe_breakdown.compute`, the path the route runs) cold and memoised, p50/p95
of 20 calls, over synthetic arrays of the stated size with a banded windowed probe. The success report and
the catalogue page need the database: they are measured through the deployed API with the commands printed
by `--print-api-commands` (the catalogue's bounded query count is a unit test; its time is measured here).
The calibration GPU time (5,000 vs 2,000 rows, 040-T34) is one operator-approved run, recorded by hand.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _arrays(directory: Path, rows: int, seed: int = 3):
    gen = np.random.default_rng(seed)
    paths = {}
    data = {
        "scores_path": gen.normal(size=rows).astype(np.float32),
        "positions_path": np.arange(rows, dtype=np.int64) * 2,
        "n_scored_path": gen.integers(1, 2000, size=rows).astype(np.int32),
        "labels_path": (gen.random(rows) < 0.4).astype(np.int8),
    }
    for column, values in data.items():
        path = directory / f"{column}.npy"
        np.save(path, values)
        paths[column] = str(path)
    return paths


def bench_breakdown(rows: int, values: int, filters: int) -> dict:
    from src.services import probe_breakdown, view_columns

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "probe_monitors"
        root.mkdir()
        paths = _arrays(root, rows)
        gen = np.random.default_rng(5)
        snapshots = {}
        columns = ["c0"] + [f"f{i}" for i in range(filters)]
        for i, column in enumerate(columns):
            codes = gen.integers(0, values if i == 0 else 3, size=rows * 2).astype(np.int32)
            dictionary = [f"v{j}" for j in range(values if i == 0 else 3)]
            snapshots[column] = view_columns._write_column(Path(tmp) / "views", column, codes, dictionary)
        bands = [{"min_tokens": 0, "max_tokens": 200, "threshold": 0.4}, {"min_tokens": 201, "max_tokens": None, "threshold": 0.9}]
        probe = SimpleNamespace(id="pm_bench", threshold=0.5, length_bands=bands, calibration_history=None,
                                window_decisions={"all": {"threshold": 0.7, "length_bands": bands}},
                                threshold_source="calibration_set", calibration_scores_path=None,
                                calibration_n_negatives=5000, target_fpr=0.01, calibration_dataset_id="pmd_c")
        evaluation = SimpleNamespace(id="pme_bench", dataset_id="pmd_v", **paths)
        view = SimpleNamespace(id="pmd_v", column_snapshots=snapshots, group_columns=columns)
        run = SimpleNamespace(config={"scope": "all"})
        settings = probe_breakdown.BreakdownSettings(min_subgroup_rows=30, page_default=100, page_max=500,
                                                    max_filters=8, calibration_floor=2000)
        request = probe_breakdown.BreakdownRequest(by_column="c0", filters=tuple((f"f{i}", "v1") for i in range(filters)))
        cold, warm = [], []
        for _ in range(20):
            probe_breakdown._memo.clear()
            t = time.perf_counter()
            probe_breakdown.compute(probe=probe, run=run, evaluation=evaluation, view=view, request=request,
                                    settings=settings, root=root)
            cold.append(time.perf_counter() - t)
            t = time.perf_counter()
            probe_breakdown.compute(probe=probe, run=run, evaluation=evaluation, view=view, request=request,
                                    settings=settings, root=root)
            warm.append(time.perf_counter() - t)

    def p(xs, q):
        return sorted(xs)[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]

    return {"rows": rows, "values": values, "filters": filters,
            "cold_p50_s": statistics.median(cold), "cold_p95_s": p(cold, 0.95),
            "memo_p50_s": statistics.median(warm), "memo_p95_s": p(warm, 0.95)}


API_COMMANDS = """\
# success report (one behaviour with 2 criteria, 3 test views, 5,000 calibration negatives):
time curl -s -X POST "$API/probe-monitors/behaviours/<behaviour-key>/success-reports" -H 'content-type: application/json' -d '{}'
# catalogue page (1,000 behaviours):
time curl -s "$API/probe-monitors/behaviours?limit=100"
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--values", type=int, default=100)
    parser.add_argument("--filters", type=int, default=2)
    parser.add_argument("--print-api-commands", action="store_true")
    args = parser.parse_args()
    if args.print_api_commands:
        print(API_COMMANDS)
        return
    print(json.dumps(bench_breakdown(args.rows, args.values, args.filters), indent=2))


if __name__ == "__main__":
    main()
