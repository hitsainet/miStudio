"""Load, validate and page one evaluation's per-row arrays (Feature 035 XR-2; FTDD §5.4, D47; FTID TI8).

The evaluation stage writes four aligned `.npy` files per evaluation (`probe_monitor_run.
_evaluate_probes_on_sets`): scores (float32), positions (int64, each kept row's position in the
downloaded split), n_scored (int32) and labels (int8). This module is the only reader, and it is
pure over the paths a row records so it can be tested without the app.

⚠ A BROKEN ARTIFACT IS AN ERROR, NEVER AN EMPTY PAGE (FR-16). A missing file, a missing path
column, a path outside the probe artifact root, an unreadable array, or four arrays of different
lengths each raise `EvaluationArtifactError` naming the check that failed. Serving `rows: []` for
any of them would read as "this evaluation scored nothing", and a consumer reproducing an AUROC
from a partial page would report a number for data it never saw.

Arrays are memory-mapped (`mmap_mode="r"`) and only the requested slice is materialised, so a
50,000-row page of a million-row evaluation reads 50,000 rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: The four arrays and the evaluation-row attribute recording each path, in response order.
ARRAY_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("score", "scores_path"),
    ("position", "positions_path"),
    ("n_scored", "n_scored_path"),
    ("label", "labels_path"),
)

DEFAULT_LIMIT = 10_000
MAX_LIMIT = 50_000


class EvaluationArtifactError(Exception):
    """One evaluation's arrays cannot be served. `check` names the failed check."""

    def __init__(self, check: str, message: str):
        super().__init__(message)
        self.check = check
        self.message = message


@dataclass
class EvaluationArrays:
    arrays: Dict[str, np.ndarray]
    total: int


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_evaluation_arrays(row: Any, *, root: Path) -> EvaluationArrays:
    """Open the four arrays `row` records, under `root`, and check they align.

    `root` is the probe artifact root (`<data_dir>/probe_monitors`): every path the writer records
    lives there (`artifact_dir_for`), so anything else is refused rather than opened (FTID TI9).
    """
    evaluation_id = getattr(row, "id", "?")
    arrays: Dict[str, np.ndarray] = {}
    for field, column in ARRAY_COLUMNS:
        recorded: Optional[str] = getattr(row, column, None)
        if not recorded:
            raise EvaluationArtifactError(
                "path_not_recorded",
                f"evaluation {evaluation_id} records no {column}, so its {field} array cannot be served",
            )
        path = Path(recorded)
        if not _inside(path, root):
            raise EvaluationArtifactError(
                "path_outside_artifact_root",
                f"evaluation {evaluation_id}'s {column} ({recorded}) is not under {root}; refusing to read it",
            )
        if not path.is_file():
            raise EvaluationArtifactError(
                "file_missing",
                f"evaluation {evaluation_id}'s {field} array is missing on disk ({recorded})",
            )
        try:
            arrays[field] = np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - a corrupt array must not read as an empty one
            raise EvaluationArtifactError(
                "file_unreadable",
                f"evaluation {evaluation_id}'s {field} array at {recorded} could not be read ({exc})",
            ) from exc
    lengths = {field: int(a.shape[0]) for field, a in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise EvaluationArtifactError(
            "length_mismatch",
            f"evaluation {evaluation_id}'s arrays are not aligned: {lengths}",
        )
    return EvaluationArrays(arrays=arrays, total=lengths["score"])


def page(loaded: EvaluationArrays, offset: int, limit: int) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Rows `[offset, offset+limit)` in stored order, and the next offset (None at the end)."""
    end = min(offset + limit, loaded.total)
    if offset >= loaded.total:
        return [], None
    sliced = {field: loaded.arrays[field][offset:end].tolist() for field, _ in ARRAY_COLUMNS}
    rows = [
        {
            "position": int(sliced["position"][i]),
            "score": float(sliced["score"][i]),
            "label": int(sliced["label"][i]),
            "n_scored": int(sliced["n_scored"][i]),
        }
        for i in range(end - offset)
    ]
    return rows, (end if end < loaded.total else None)
