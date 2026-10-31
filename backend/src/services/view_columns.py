"""Grouping columns declared on a probe dataset view, and their snapshots (Feature 040 FR-12, FR-14; FTDD §4.5).

A breakdown groups an evaluation's rows by a column of the split the view was registered over. The
column is DECLARED on the view and SNAPSHOTTED at declaration: each declared column is read once through
`probe_monitor_service.open_split` and written, indexed by SPLIT POSITION, under
``<data_dir>/probe_monitor_views/<view_id>/columns/<column digest>/`` as

* ``codes.npy`` — int32, one code per split row (`-1` never occurs: every value, null included, has a
  code); and
* ``values.json`` — the JSON values the codes index, in first-seen order.

So a breakdown is reproducible even if the download is later moved or re-pulled, it loads memory-mapped
and small, and it never holds row text: only declared grouping columns are snapshotted (040-T11).

⚠ ATOMIC PUBLISH (TI-7): a column is written into a ``.tmp-<uuid>`` directory and renamed into place
(atomic on one filesystem), and the row's `column_snapshots` is updated in the SAME transaction as the
history row; if the commit fails the published directory is removed. No half-written snapshot is ever
referenced.

⚠ ADDITIVE ONLY (FR-14): a later declaration may add columns; dropping or renaming one is refused
`column_not_additive`, because a report already computed over it names it. An archived view is refused
with the existing 409.

⚠ A PER-ROW ID IS NOT A GROUP: a column with more distinct values than the runtime setting
`group_column_max_values` is refused `too_many_values`, so a breakdown cannot become a copy of the view.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .detector_next import DetectorRefused, next_find_views

SNAPSHOT_DIR = "probe_monitor_views"


@dataclass(frozen=True)
class ColumnLimits:
    max_values: int
    max_columns: int

    @classmethod
    def from_settings(cls, settings: Any = None) -> "ColumnLimits":
        if settings is None:
            from ..core.runtime_settings import effective

            settings = effective()
        return cls(max_values=settings.group_column_max_values, max_columns=settings.view_max_group_columns)


def views_root() -> Path:
    from ..core.config import settings

    return Path(settings.data_dir) / SNAPSHOT_DIR


def view_dir(view_id: str) -> Path:
    """The view's snapshot directory; refuses (ValueError) an id that would escape `views_root()`."""
    from ..utils.file_utils import resolve_under

    return resolve_under(views_root(), view_id)


def column_key(column: str) -> str:
    """A filesystem-safe, collision-free name for a column (column names are arbitrary text)."""
    return hashlib.sha256(column.encode("utf-8")).hexdigest()[:24]


def _json_value(value: Any) -> Any:
    """A cell as a JSON scalar: str/int/float/bool/None as themselves, anything else as its text."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def encode(values: Sequence[Any], max_values: int, column: str) -> Tuple[np.ndarray, List[Any]]:
    """Dictionary-encode a column; refuses a column with more than `max_values` distinct values."""
    lookup: Dict[str, int] = {}
    dictionary: List[Any] = []
    codes = np.empty(len(values), dtype=np.int32)
    for index, raw in enumerate(values):
        value = _json_value(raw)
        token = json.dumps(value, sort_keys=True)
        code = lookup.get(token)
        if code is None:
            if len(dictionary) >= max_values:
                raise DetectorRefused(
                    422, "too_many_values",
                    f"column {column!r} has more than {max_values} distinct values (the runtime setting "
                    f"group_column_max_values); a grouping column groups rows, it is not a row id",
                    {"column": column, "setting": "group_column_max_values", "value": max_values},
                )
            code = len(dictionary)
            lookup[token] = code
            dictionary.append(value)
        codes[index] = code
    return codes, dictionary


def _digest(codes_bytes: bytes, values_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(codes_bytes)
    h.update(b"\0")
    h.update(values_bytes)
    return h.hexdigest()


def _write_column(target_root: Path, column: str, codes: np.ndarray, dictionary: List[Any]) -> Dict[str, Any]:
    """Write one column's snapshot atomically; returns its `column_snapshots` entry."""
    target_root.mkdir(parents=True, exist_ok=True)
    final = target_root / "columns" / column_key(column)
    tmp = target_root / f".tmp-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True)
    try:
        np.save(tmp / "codes.npy", codes, allow_pickle=False)
        values_bytes = json.dumps(dictionary, ensure_ascii=False).encode("utf-8")
        (tmp / "values.json").write_bytes(values_bytes)
        codes_bytes = (tmp / "codes.npy").read_bytes()
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():  # a re-declaration of the same column: replace whole, never merge
            shutil.rmtree(final)
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "path": str(final),
        "sha256": _digest(codes_bytes, values_bytes),
        "n_rows": int(codes.shape[0]),
        "n_values": len(dictionary),
        "dtype": "dictionary",
    }


@dataclass
class Snapshot:
    """One column's snapshot as loaded for a breakdown."""

    column: str
    codes: np.ndarray
    values: List[Any]
    sha256: str


class SnapshotBroken(Exception):
    """A snapshot that no longer matches its recorded sha256 or row count (an integrity failure)."""

    def __init__(self, check: str, message: str):
        super().__init__(message)
        self.check = check


def load(entry: Dict[str, Any], column: str) -> Snapshot:
    path = Path(entry["path"])
    codes_path, values_path = path / "codes.npy", path / "values.json"
    if not codes_path.is_file() or not values_path.is_file():
        raise SnapshotBroken("snapshot_missing", f"the snapshot of column {column!r} is missing on disk")
    codes_bytes, values_bytes = codes_path.read_bytes(), values_path.read_bytes()
    if _digest(codes_bytes, values_bytes) != entry["sha256"]:
        raise SnapshotBroken("snapshot_changed", f"the snapshot of column {column!r} no longer matches its sha256")
    codes = np.load(codes_path, mmap_mode="r", allow_pickle=False).reshape(-1)
    if int(codes.shape[0]) != int(entry["n_rows"]):
        raise SnapshotBroken("snapshot_rows", f"the snapshot of column {column!r} has the wrong row count")
    return Snapshot(column=column, codes=codes, values=json.loads(values_bytes), sha256=entry["sha256"])


def _read_split_columns(db_dataset: Any, view: Any, columns: Sequence[str]) -> Tuple[Dict[str, List[Any]], int]:
    from .probe_monitor_service import open_split, resolve_dataset_path

    data = open_split(resolve_dataset_path(db_dataset.raw_path), view.split)
    available = list(getattr(data, "column_names", []) or [])
    missing = [c for c in columns if c not in available]
    if missing:
        raise DetectorRefused(
            422, "column_absent",
            f"column(s) {missing} are not in this view's split; it has {available}",
            {"columns": missing, "available": available},
        )
    return {c: list(data[c]) for c in columns}, len(data)


async def _dataset_of(db: Any, view: Any) -> Any:
    from sqlalchemy import select

    from ..models.dataset import Dataset

    dataset = (await db.execute(select(Dataset).where(Dataset.id == view.dataset_id))).scalar_one_or_none()
    if dataset is None:
        raise DetectorRefused(409, "dataset_gone", f"the dataset behind view {view.id} is gone")
    return dataset


def _check_names(columns: Sequence[str], limits: ColumnLimits) -> List[str]:
    out: List[str] = []
    for index, column in enumerate(columns):
        if not isinstance(column, str) or not column.strip() or len(column) > 255:
            raise DetectorRefused(422, "invalid_field", "a column name is 1 to 255 characters",
                                  {"field": f"group_columns[{index}]", "rule": "invalid_column"})
        if column in out:
            raise DetectorRefused(422, "invalid_field", f"column {column!r} is declared twice",
                                  {"field": f"group_columns[{index}]", "rule": "duplicate_column"})
        out.append(column)
    if len(out) > limits.max_columns:
        raise DetectorRefused(
            422, "over_cap",
            f"{len(out)} grouping columns; the runtime setting view_max_group_columns allows {limits.max_columns}",
            {"setting": "view_max_group_columns", "value": limits.max_columns, "field": "group_columns", "got": len(out)},
        )
    return out


async def snapshot_columns(db: Any, view: Any, columns: Sequence[str], limits: ColumnLimits) -> Dict[str, Dict[str, Any]]:
    """Read and write the snapshots of `columns` for `view` (which must have an id). Returns the
    `column_snapshots` entries. The caller records them and the history in one transaction and calls
    `discard(entries)` if that transaction fails."""
    if not columns:
        return {}
    dataset = await _dataset_of(db, view)
    from starlette.concurrency import run_in_threadpool

    values, n_rows = await run_in_threadpool(_read_split_columns, dataset, view, list(columns))
    out: Dict[str, Dict[str, Any]] = {}
    written: List[Dict[str, Any]] = []
    try:
        for column in columns:
            codes, dictionary = encode(values[column], limits.max_values, column)
            entry = _write_column(view_dir(view.id), column, codes, dictionary)
            written.append(entry)
            out[column] = entry
    except BaseException:
        discard(written)
        raise
    return out


def discard(entries: Any) -> None:
    for entry in (entries.values() if isinstance(entries, dict) else entries):
        shutil.rmtree(entry["path"], ignore_errors=True)


def declare_on_registration(columns: Optional[Sequence[str]], limits: ColumnLimits) -> List[str]:
    return _check_names(list(columns or []), limits)


async def add(db: Any, view: Any, columns: Sequence[str], actor: str, limits: ColumnLimits) -> Dict[str, Any]:
    """FR-14: the view's grouping columns become `columns`, which must contain every column already
    declared (additive). Only new columns are read and snapshotted. Records the change in the
    behaviour's history when the view names a behaviour, else in `probe_monitor_dataset_column_changes`."""
    from ..models.detector_report import ProbeMonitorDatasetColumnChange
    from ..models.probe_behaviour import ProbeBehaviourHistory
    from .probe_dataset_lifecycle import refuse_archived_inputs

    await refuse_archived_inputs(db, [view.id])
    wanted = _check_names(list(columns), limits)
    before = list(view.group_columns or [])
    dropped = [c for c in before if c not in wanted]
    if dropped:
        raise DetectorRefused(
            422, "column_not_additive",
            f"column(s) {dropped} are already declared on this view and cannot be removed or renamed: "
            f"reports already computed over them name them. Send every declared column plus the new ones.",
            {"declared": before, "removed": dropped},
        )
    new = [c for c in wanted if c not in before]
    if not new:
        return {"view_id": view.id, "group_columns": before, "added": [], "column_snapshots": view.column_snapshots or {}}
    entries = await snapshot_columns(db, view, new, limits)
    snapshots = dict(view.column_snapshots or {})
    snapshots.update(entries)
    after = before + new
    view.group_columns = after
    view.column_snapshots = snapshots
    if view.behaviour:
        db.add(ProbeBehaviourHistory(
            key=view.behaviour, subject_kind="view", subject_id=view.id, field="group_columns",
            old_value=before, new_value=after, actor=actor,
        ))
    else:
        db.add(ProbeMonitorDatasetColumnChange(
            view_id=view.id, field="group_columns", old_value=before, new_value=after, actor=actor,
        ))
    try:
        await db.commit()
    except BaseException:
        await db.rollback()
        discard(entries)
        raise
    return {"view_id": view.id, "group_columns": after, "added": new, "column_snapshots": snapshots}


def purge_snapshots(view_id: str) -> Optional[str]:
    """Delete a purged view's snapshot directory; returns the path removed, or None when none existed."""
    path = view_dir(view_id)
    if not path.exists():
        return None
    shutil.rmtree(path, ignore_errors=True)
    return str(path)


def snapshot_files(view_id: str) -> Dict[str, str]:
    """`{"probe_monitor_views": <dir>}` when the view owns a snapshot directory, else `{}` (for the
    purge tombstone)."""
    path = view_dir(view_id)
    return {"probe_monitor_views": str(path)} if path.exists() else {}


# ── Feature 040 S3 (FR-12, FR-14; 040-T22, T23): the text column and producer rule columns ──────────

#: Where a dataset-version manifest lists the rule sets its build ran (016 §4.12; D31).
RULE_SET_LISTING_KEY = "midataworks.rule-sets/v1"


def listing_from_manifest(manifest: Any) -> Optional[Dict[str, Any]]:
    """The manifest's rule-set listing, checked against the vendored `RuleSetListing` shape, or None."""
    if not isinstance(manifest, dict):
        return None
    listing = (manifest.get("extensions") or {}).get(RULE_SET_LISTING_KEY)
    if listing is None:
        return None
    from jsonschema import Draft202012Validator

    from ..rule_signals.document import schema

    defs = schema()["$defs"]
    errors = list(Draft202012Validator({"$defs": defs, "$ref": "#/$defs/RuleSetListing"}).iter_errors(listing))
    if errors:
        raise DetectorRefused(422, "invalid_field", "the manifest's rule-set listing does not match its schema",
                              {"field": f"dataset_version_manifest.extensions.{RULE_SET_LISTING_KEY}",
                               "rule": errors[0].validator})
    return listing


def check_rule_columns(rule_columns: Optional[Dict[str, Any]], listing: Optional[Dict[str, Any]],
                       available: Sequence[str], text_column: Optional[str]) -> None:
    """040-T22: `rule_set_id`/`rule_set_digest` must be a set the manifest lists; the verdict columns and
    `text_column` must exist in the split. Nothing is auto-attached."""
    if text_column is not None and text_column not in available:
        raise DetectorRefused(422, "column_absent", f"text_column {text_column!r} is not in this view's split",
                              {"columns": [text_column], "available": list(available)})
    if rule_columns is None:
        return
    missing = [rule_columns[k] for k in ("rule_match", "rule_veto") if rule_columns.get(k) and rule_columns[k] not in available]
    if missing:
        raise DetectorRefused(422, "column_absent", f"rule column(s) {missing} are not in this view's split",
                              {"columns": missing, "available": list(available)})
    if listing is None:
        raise DetectorRefused(422, "invalid_field",
                              "producer rule columns are checked against the dataset-version manifest's rule-set "
                              "listing, and this view has none; register it with its manifest",
                              {"field": "rule_columns", "rule": "requires_rule_set_listing"})
    listed = {(f"{r['name']}@{r['version']}", r["sha256"]) for r in listing.get("rule_sets") or []}
    if (rule_columns["rule_set_id"], rule_columns["rule_set_digest"]) not in listed:
        raise DetectorRefused(422, "rule_set_not_listed",
                              f"rule set {rule_columns['rule_set_id']} ({rule_columns['rule_set_digest'][:12]}…) is "
                              f"not in the manifest's rule-set listing",
                              {"rule_set_id": rule_columns["rule_set_id"], "listed": sorted(i for i, _ in listed)})


async def split_columns(db: Any, view: Any) -> List[str]:
    from starlette.concurrency import run_in_threadpool

    from .probe_monitor_service import open_split, resolve_dataset_path

    dataset = await _dataset_of(db, view)
    data = await run_in_threadpool(open_split, resolve_dataset_path(dataset.raw_path), view.split)
    return list(getattr(data, "column_names", []) or [])


async def add_rule_fields(db: Any, view: Any, *, text_column: Optional[str], rule_columns: Optional[Dict[str, Any]],
                          actor: str, limits: ColumnLimits) -> Dict[str, Any]:
    """FR-14 for the S3 fields: `text_column` and `rule_columns` may be DECLARED once (additive); changing a
    declared one is refused `column_not_additive`. Producer verdict columns are snapshotted like grouping
    columns (040-T11); the text is never copied (it is read live, with the drift check)."""
    from ..models.detector_report import ProbeMonitorDatasetColumnChange
    from ..models.probe_behaviour import ProbeBehaviourHistory
    from .probe_dataset_lifecycle import refuse_archived_inputs

    await refuse_archived_inputs(db, [view.id])
    for field, new, old in (("text_column", text_column, view.text_column), ("rule_columns", rule_columns, view.rule_columns)):
        if new is not None and old is not None and new != old:
            raise DetectorRefused(422, "column_not_additive",
                                  f"{field} is already declared on this view and cannot change: results already "
                                  f"computed over it name it", {"field": field, "declared": old})
    text_new = text_column if text_column is not None and view.text_column is None else None
    rules_new = rule_columns if rule_columns is not None and view.rule_columns is None else None
    if text_new is None and rules_new is None:
        return {"view_id": view.id, "text_column": view.text_column, "rule_columns": view.rule_columns, "added": []}
    available = await split_columns(db, view)
    check_rule_columns(rules_new, view.rule_set_listing, available, text_new)
    entries: Dict[str, Any] = {}
    if rules_new is not None:
        columns = [c for c in (rules_new.get("rule_match"), rules_new.get("rule_veto")) if c and c not in (view.column_snapshots or {})]
        entries = await snapshot_columns(db, view, columns, limits)
    added: List[str] = []
    if text_new is not None:
        view.text_column = text_new
        added.append("text_column")
    if rules_new is not None:
        view.rule_columns = dict(rules_new)
        snapshots = dict(view.column_snapshots or {})
        snapshots.update(entries)
        view.column_snapshots = snapshots
        added.append("rule_columns")
    for field in added:
        new_value = getattr(view, field)
        if view.behaviour:
            db.add(ProbeBehaviourHistory(key=view.behaviour, subject_kind="view", subject_id=view.id, field=field,
                                         old_value=None, new_value=new_value, actor=actor))
        else:
            db.add(ProbeMonitorDatasetColumnChange(view_id=view.id, field=field, old_value=None, new_value=new_value,
                                                   actor=actor))
    try:
        await db.commit()
    except BaseException:
        await db.rollback()
        discard(entries)
        raise
    return {"view_id": view.id, "text_column": view.text_column, "rule_columns": view.rule_columns, "added": added}
