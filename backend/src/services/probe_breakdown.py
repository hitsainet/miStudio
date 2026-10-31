"""Recall at the deployed decision, by subgroup, from stored arrays (Feature 040 FR-15 – FR-18; FTID §3.4).

No GPU, no model: one evaluation's four 035 arrays (`probe_evaluation_scores.load_evaluation_arrays`,
memory-mapped) are joined to a declared column's snapshot (`view_columns`) BY SPLIT POSITION —
`codes[positions]`, never by array index — and every row is judged by `deployed_decision` in one
vectorised pass (`decide_many`), the rule miLLM serves (pinned by `decision-precedence-cases.json`).

* ``mode="recorded"`` (default): the decision in force — window bar, window bands, probe bands or the
  global bar per row, the window chosen from the run's scope (040-T25). `rule_counts` says how many rows
  each rule covered, and the decision revision is named (FR-18).
* ``mode="target_fpr"``: a bar cut with `calibrate` (the order statistic, never a quantile) over the
  probe's persisted CALIBRATION negatives — never over the test set's own negatives (FR-16). A probe
  with no calibration view is refused `no_calibration_view`.

Small groups are SHOWN and flagged `too_small` (below the runtime setting `breakdown_min_subgroup_rows`,
or the request's `min_rows`), never dropped (FR-17). Figures are float64, rounded to 6 decimals; the
body names every input (`inputs_digest`) and every setting it read, so the same request gives the same
bytes (FR-18). An in-process memo of 256 bodies is keyed by that digest (TI-18).
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import view_columns
from .detector_identity import identity, jcs_bytes, round6
from .detector_next import DetectorRefused, next_evaluate, next_update_columns
from .probe_monitor_metrics import DECISION_RULES, decide_many, wilson_interval

MEMO_SIZE = 256
_memo: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_memo_lock = threading.Lock()


@dataclass(frozen=True)
class BreakdownSettings:
    min_subgroup_rows: int
    page_default: int
    page_max: int
    max_filters: int
    calibration_floor: int

    @classmethod
    def from_settings(cls, settings: Any = None) -> "BreakdownSettings":
        if settings is None:
            from ..core.runtime_settings import effective

            settings = effective()
        return cls(
            min_subgroup_rows=settings.breakdown_min_subgroup_rows,
            page_default=settings.breakdown_page_default,
            page_max=settings.breakdown_page_max,
            max_filters=settings.breakdown_max_filters,
            calibration_floor=settings.calibration_floor_negatives,
        )

    def as_inputs(self) -> Dict[str, int]:
        """Every setting value read, under its runtime key name (TI-22)."""
        return {
            "breakdown_min_subgroup_rows": self.min_subgroup_rows,
            "breakdown_page_default": self.page_default,
            "breakdown_page_max": self.page_max,
            "breakdown_max_filters": self.max_filters,
            "calibration_floor_negatives": self.calibration_floor,
        }


@dataclass(frozen=True)
class BreakdownRequest:
    by_column: str
    filters: Tuple[Tuple[str, str], ...] = ()
    mode: str = "recorded"
    target_fpr: Optional[float] = None
    min_rows: Optional[int] = None
    limit: Optional[int] = None
    cursor: Optional[str] = None


def evaluation_window(run_scope: Optional[str]) -> Optional[str]:
    """040-T25: the contract window whose internal scope IS the run's scope, or None (`assistant`,
    `user`, `content` map to no window: probe bands, else the global bar)."""
    from .probe_monitor_render import CONTRACT_WINDOW_SCOPES

    matches = [window for window, scope in CONTRACT_WINDOW_SCOPES.items() if scope == run_scope]
    return matches[0] if len(matches) == 1 else None


def recorded_decisions(probe: Any, run_scope: Optional[str], n_scored: Any) -> Any:
    """The decision a deployment applies to each stored row of one evaluation of `probe` (FR-15).

    ⚠ THE ONE PLACE THE STORED PROBE IS TURNED INTO `decide_many`'s ARGUMENTS (review R1, 040 S1-S2). The
    breakdown, the success report and the recall selection each built them inline, and dropping the window
    entry or passing the run's internal scope survived every test in all three (a fixture scoped `all`
    with probe bands cannot tell): the window comes from the run's scope (040-T25), the entry is that
    window's own bar and bands, and the scope is the CONTRACT scope the definition exports."""
    inputs = decision_inputs(probe, run_scope)
    return decide_many(inputs["probe_bar"], inputs["window_entry"], inputs["probe_bands"], inputs["probe_scope"],
                       inputs["window"], n_scored)


def decision_inputs(probe: Any, run_scope: Optional[str]) -> Dict[str, Any]:
    """`decide_many`'s arguments for one stored probe, as plain data — the ONE place they are built.

    Feature 041 records them per member at the start of a blend evaluation (the decision in force THEN), and
    the blend engine passes them to `decide_many` unchanged; `recorded_decisions` above uses the same dict,
    so a blend member's verdict and this breakdown's can never be built two ways (040 review R1-3)."""
    window = evaluation_window(run_scope)
    entry = (getattr(probe, "window_decisions", None) or {}).get(window) if window else None
    return {"probe_bar": probe.threshold, "window_entry": entry, "probe_bands": probe.length_bands,
            "probe_scope": contract_scope_of(run_scope), "window": window}


def contract_scope_of(run_scope: Optional[str]) -> str:
    """The scope a definition exports for this run's internal scope; a scope the contract cannot express
    stays its internal name, which only ever equals itself.

    ⚠ THE BUILDER'S OWN MAPPING, CALLED (review R1, 040 S1-S2): this was a hand copy of
    `probe_definition_builder._contract_scope`, which would drift silently the day the exporter changes."""
    from .probe_definition_builder import ProbeExportRefused, _contract_scope

    scope = run_scope or "all"
    try:
        return _contract_scope(scope)
    except ProbeExportRefused:
        return scope


def parse_filters(raw: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    """TI-23: repeated `column=value` (the first `=` splits)."""
    out: List[Tuple[str, str]] = []
    for item in raw:
        if "=" not in item:
            raise DetectorRefused(422, "invalid_field", f"a filter is column=value, not {item!r}",
                                  {"field": "filter", "rule": "not_column_equals_value"})
        column, value = item.split("=", 1)
        out.append((column, value))
    return tuple(out)


def _value_key(value: Any) -> str:
    """How a filter's text is matched against a snapshot value: the value's own text form."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _encode_cursor(offset: int, digest: str) -> str:
    return base64.urlsafe_b64encode(jcs_bytes({"d": digest[:16], "o": offset})).decode().rstrip("=")


def _decode_cursor(cursor: str, digest: str) -> int:
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if data["d"] != digest[:16] or not isinstance(data["o"], int) or data["o"] < 0:
            raise ValueError("cursor")
        return data["o"]
    except Exception as exc:  # noqa: BLE001
        raise DetectorRefused(422, "invalid_field", "this cursor was not issued for this breakdown",
                              {"field": "cursor", "rule": "filter_changed"}) from exc


def calibration_block(probe: Any, floor_override: Optional[int], settings: BreakdownSettings,
                      target_fpr: Optional[float]) -> Optional[Dict[str, Any]]:
    from .calibration_size import calibration_size

    size = calibration_size(probe, floor_override, settings.calibration_floor, target_fpr)
    return size


def _groups(codes: np.ndarray, positives: np.ndarray, fires: np.ndarray, scores: np.ndarray,
            mask: np.ndarray, values: List[Any], min_rows: int) -> List[Dict[str, Any]]:
    n_values = len(values)
    keep = codes[mask]
    pos = positives[mask]
    fired = fires[mask]
    sc = np.asarray(scores, dtype=np.float64)[mask]
    n_pos = np.bincount(keep[pos], minlength=n_values)
    n_neg = np.bincount(keep[~pos], minlength=n_values)
    hit_pos = np.bincount(keep[pos & fired], minlength=n_values)
    hit_neg = np.bincount(keep[~pos & fired], minlength=n_values)
    order = np.argsort(keep, kind="stable")
    sorted_codes, sorted_scores = keep[order], sc[order]
    bounds = np.searchsorted(sorted_codes, np.arange(n_values + 1))
    out: List[Dict[str, Any]] = []
    for code in range(n_values):
        total = int(n_pos[code] + n_neg[code])
        if total == 0:
            continue
        p, n = int(n_pos[code]), int(n_neg[code])
        fp, fn_ = int(hit_pos[code]), int(hit_neg[code])
        group_scores = sorted_scores[bounds[code]: bounds[code + 1]]
        lo_r, hi_r = wilson_interval(fp, p)
        lo_f, hi_f = wilson_interval(fn_, n)
        out.append({
            "value": values[code],
            "positives": {"n": p, "flagged": fp, "recall": round6(fp / p) if p else None,
                          "ci": [round6(lo_r), round6(hi_r)] if p else None},
            "negatives": {"n": n, "flagged": fn_, "fpr": round6(fn_ / n) if n else None,
                          "ci": [round6(lo_f), round6(hi_f)] if n else None},
            "median_score": round6(float(np.median(group_scores))),
            "rows": total,
            "too_small": total < min_rows,
        })
    return out


def groups_of(codes: np.ndarray, positives: np.ndarray, fires: np.ndarray, scores: np.ndarray,
              mask: np.ndarray, values: List[Any], min_rows: int) -> List[Dict[str, Any]]:
    """Per-subgroup figures (recall, false-positive rate, Wilson intervals, median score, `too_small`) for
    rows already joined to a column BY POSITION. The breakdown below and Feature 041's blend reports both
    call this, so a blend candidate's subgroup figures are 040's figures (FR-12, FR-20)."""
    return _groups(codes, positives, fires, scores, mask, values, min_rows)


def compute(*, probe: Any, run: Any, evaluation: Any, view: Any, request: BreakdownRequest,
            settings: BreakdownSettings, root: Path, floor_override: Optional[int] = None) -> Dict[str, Any]:
    """The breakdown body. Pure over the rows and files it is given (the route loads them)."""
    from .probe_evaluation_scores import load_evaluation_arrays

    if not getattr(evaluation, "positions_path", None):
        raise DetectorRefused(
            409, "positions_not_recorded",
            f"evaluation {evaluation.id} was recorded before miStudio saved row positions, so its scores "
            f"cannot be joined to columns; re-evaluate the probe on this set",
            {"evaluation_id": evaluation.id},
            next=[next_evaluate(probe.id, evaluation.dataset_id)],
        )
    declared = view.column_snapshots or {}
    for column in [request.by_column, *[c for c, _ in request.filters]]:
        if column not in declared:
            raise DetectorRefused(
                422, "column_not_declared",
                f"column {column!r} is not declared on view {view.id}; declared: {sorted(declared)}",
                {"column": column, "declared": sorted(declared)},
                next=[next_update_columns(view.id, list(view.group_columns or []) + [column])],
            )
    if len(request.filters) > settings.max_filters:
        raise DetectorRefused(
            422, "over_cap", f"{len(request.filters)} filters; breakdown_max_filters allows {settings.max_filters}",
            {"setting": "breakdown_max_filters", "value": settings.max_filters, "field": "filter",
             "got": len(request.filters)})
    limit = request.limit if request.limit is not None else settings.page_default
    if limit > settings.page_max:
        raise DetectorRefused(
            422, "over_cap", f"a page of {limit} groups is more than breakdown_page_max ({settings.page_max})",
            {"setting": "breakdown_page_max", "value": settings.page_max, "field": "limit", "got": limit})
    min_rows = request.min_rows if request.min_rows is not None else settings.min_subgroup_rows

    run_scope = (run.config or {}).get("scope", "all") if run is not None else "all"
    window = evaluation_window(run_scope)
    from ..schemas.probe_identity import current_revision

    revision = current_revision(getattr(probe, "calibration_history", None))
    target = None
    cal_digest = None
    if request.mode == "target_fpr":
        if request.target_fpr is None or not (0.0 < request.target_fpr < 1.0):
            raise DetectorRefused(422, "invalid_field", "target_fpr mode needs target_fpr in (0, 1)",
                                  {"field": "target_fpr", "rule": "out_of_range"})
        if probe.threshold_source != "calibration_set" or not probe.calibration_scores_path:
            raise DetectorRefused(
                409, "no_calibration_view",
                f"probe {probe.id} has no calibration-set negatives on disk, so a target false-positive rate "
                f"cannot be cut for it here (a bar is never cut on the test set's own negatives)",
                {"probe_id": probe.id})
        target = request.target_fpr
        cal_digest = _file_sha256(probe.calibration_scores_path)
    elif request.mode != "recorded":
        raise DetectorRefused(422, "invalid_field", "mode is recorded or target_fpr",
                              {"field": "mode", "rule": "unknown_mode"})

    paths = [evaluation.scores_path, evaluation.positions_path, evaluation.n_scored_path, evaluation.labels_path]
    snapshots = {c: declared[c]["sha256"] for c in sorted({request.by_column, *[c for c, _ in request.filters]})}
    inputs = {
        "algorithm": 1,
        "probe_id": probe.id,
        "evaluation_id": evaluation.id,
        "arrays": [_file_sha256(p) for p in paths],
        "decision": {"mode": request.mode, "revision": revision, "threshold": probe.threshold,
                     "length_bands": probe.length_bands, "window": window,
                     "window_entry": (probe.window_decisions or {}).get(window) if window else None,
                     "target_fpr": target, "calibration_sha256": cal_digest},
        "snapshots": snapshots,
        "request": {"by_column": request.by_column, "filters": [list(f) for f in request.filters],
                    "min_rows": min_rows, "limit": limit},
        "floor_override": floor_override,
        "settings": settings.as_inputs(),
    }
    digest = identity(inputs)
    offset = _decode_cursor(request.cursor, digest) if request.cursor else 0
    memo_key = f"{digest}:{offset}"
    with _memo_lock:
        if memo_key in _memo:
            _memo.move_to_end(memo_key)
            return json.loads(json.dumps(_memo[memo_key]))

    loaded = load_evaluation_arrays(evaluation, root=root)
    scores = np.asarray(loaded.arrays["score"], dtype=np.float64)
    positions = np.asarray(loaded.arrays["position"], dtype=np.int64)
    n_scored = np.asarray(loaded.arrays["n_scored"])
    labels = np.asarray(loaded.arrays["label"])
    positives = labels == 1

    snap = view_columns.load(declared[request.by_column], request.by_column)
    if positions.size and int(positions.max()) >= int(snap.codes.shape[0]):
        raise view_columns.SnapshotBroken(
            "positions_outside_snapshot",
            f"evaluation {evaluation.id} has positions beyond column {request.by_column!r}'s snapshot")
    codes = np.asarray(snap.codes)[positions]          # ⚠ BY POSITION, never by array index
    mask = np.ones(scores.shape[0], dtype=bool)
    for column, value in request.filters:
        fsnap = view_columns.load(declared[column], column)
        wanted = [i for i, v in enumerate(fsnap.values) if _value_key(v) == value]
        fcodes = np.asarray(fsnap.codes)[positions]
        mask &= np.isin(fcodes, wanted) if wanted else np.zeros_like(mask)

    if target is None:
        decisions = recorded_decisions(probe, run_scope, n_scored)
        fires = decisions.fires(scores)
        rule_counts = decisions.rule_counts()
        threshold_used = None
    else:
        from .probe_monitor_trainer import calibrate
        from .probe_recalibration import load_calibration_array

        negatives = load_calibration_array(probe.calibration_scores_path, what="negatives")
        cut = calibrate(list(map(float, negatives)), target_fpr=target, source="calibration_set")
        threshold_used = cut.threshold
        fires = (scores >= threshold_used) if threshold_used is not None else np.zeros_like(positives)
        rule_counts = None

    groups = groups_of(codes, positives, fires, scores, mask, snap.values, min_rows)
    page = groups[offset: offset + limit]
    body = {
        "probe_id": probe.id,
        "evaluation_id": evaluation.id,
        "view_id": view.id,
        "decision": {"mode": request.mode, "revision": revision, "scope": run_scope, "window": window,
                     "rule_counts": rule_counts, "target_fpr": target,
                     "threshold": round6(threshold_used) if threshold_used is not None else None},
        "calibration": calibration_block(probe, floor_override, settings, target),
        "column": request.by_column,
        "filters": [{"column": c, "value": v} for c, v in request.filters],
        "min_rows": min_rows,
        "groups": page,
        "total": len(groups),
        "next_cursor": _encode_cursor(offset + limit, digest) if offset + limit < len(groups) else None,
        "inputs_digest": digest,
        "settings": settings.as_inputs(),
    }
    with _memo_lock:
        _memo[memo_key] = json.loads(json.dumps(body))
        while len(_memo) > MEMO_SIZE:
            _memo.popitem(last=False)
    return body
