"""Synthetic blend inputs with planted structure (Feature 041 T29/G1; FTID §8).

Seeded `np.random.default_rng`; every view's rows live at NON-CONTIGUOUS split positions, and each member
keeps its own subset of them (a render that failed, a filter), so a join by array index gives different
figures from the join by position. Members: two informative (`strong`, `second`), one noise, and any extra.
Calibration views carry negatives only (as a real calibration view does).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.services.probe_blend import candidates as cand
from src.services.probe_blend import engine


def bar(threshold: Optional[float]) -> Dict[str, object]:
    """A member decision with only a global bar."""
    return {"probe_bar": threshold, "window_entry": None, "probe_bands": None, "probe_scope": "all", "window": None}


def view_rows(rng: np.random.Generator, n: int, positives: bool = True, gap: int = 3):
    positions = np.sort(rng.choice(n * gap, size=n, replace=False)).astype(np.int64)
    labels = (rng.random(n) < 0.5).astype(np.int8) if positives else np.zeros(n, dtype=np.int8)
    return positions, labels


def member_scores(rng: np.random.Generator, labels: np.ndarray, signal: float) -> np.ndarray:
    return (signal * labels + rng.normal(0.0, 1.0, labels.shape[0])).astype(np.float64)


def build(seed: int = 0, *, members: Sequence[str] = ("pm_a", "pm_b", "pm_c"), signals: Sequence[float] = (2.0, 1.2, 0.0),
          n_fit: int = 600, n_test: int = 400, n_cal: int = 800, keep: float = 0.98, groups: Optional[int] = 12,
          tests: Sequence[Dict[str, object]] = ({"id": "pmd_ood", "held_out": True}, {"id": "pmd_ind", "held_out": False}),
          candidates: Optional[List[str]] = None, folds: int = 5, bars: Optional[Sequence[Optional[float]]] = None,
          target_fpr: float = 0.05) -> engine.EngineInputs:
    rng = np.random.default_rng(seed)
    view_ids = ["pmd_fit", *[t["id"] for t in tests], "pmd_cal"]
    arrays: Dict[str, Dict[str, engine.MemberArrays]] = {}
    group_codes = None
    for view in view_ids:
        n = {"pmd_fit": n_fit, "pmd_cal": n_cal}.get(view, n_test)
        positions, labels = view_rows(rng, n, positives=(view != "pmd_cal"))
        if view == "pmd_fit" and groups:
            group_codes = rng.integers(0, groups, size=int(positions.max()) + 1).astype(np.int32)
        arrays[view] = {}
        for member, signal in zip(members, signals):
            scores = member_scores(rng, labels, signal)
            kept = np.sort(rng.choice(n, size=int(round(n * keep)), replace=False))
            order = rng.permutation(kept.size)                     # stored order is not position order
            idx = kept[order]
            arrays[view][member] = engine.MemberArrays(
                scores=scores[idx], positions=positions[idx], labels=labels[idx],
                n_scored=rng.integers(5, 200, size=idx.size).astype(np.int32))
    m = len(members)
    names = candidates or cand.vocabulary(m, False)
    return engine.EngineInputs(
        members=list(members), decisions=[bar(b) for b in (bars or [0.8] * m)], blend_fit="pmd_fit",
        tests=[dict(t) for t in tests], calibration="pmd_cal", arrays=arrays,
        candidates=[cand.parse_name(name, m, False) for name in names], folds=folds, seed=seed,
        target_fpr=target_fpr, min_rows_per_class=5, min_aligned_fraction=0.9, floor=100,
        group_codes=group_codes)
