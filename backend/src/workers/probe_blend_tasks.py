"""The blend evaluation task (Feature 041 S2; FR-6, FR-15; FTID §3.9, TI6).

CPU only, on the `blend` queue consumed by `celery-worker-blend` (prefork, one BLAS thread per child, T24):
a blend never waits behind hour-long natural-language passes and never takes a GPU.

⚠ IDS IN, IDS AND COUNTS OUT. The argument is a report id; the inputs are the evaluation ids the report
FIXED AT START (`resolved`), read through `load_evaluation_arrays` (with its data-root containment check);
the output is files under `data/probe_monitors/blends/<id>/` and a status. No row text anywhere.

⚠ THE ROW IS COMMITTED BY THE API BEFORE DISPATCH, AND THIS TASK REFUSES WITHOUT ONE; a row already
terminal returns at once (an acks-late redelivery after a lost worker is not a restart — miStudio memory
`cancel-must-not-be-its-own-restart`).

⚠ THE CANCEL POLL LIVES HERE (TI6): `cancel_checker("blend_report", …)` is built in this module and handed
to the engine as `should_stop`, so `test_cancel_registry_completeness.py` (which scans this directory) sees
the scope polled.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..core.celery_app import celery_app
from ..core.clock import utc_now
from .base_task import DatabaseTask

logger = logging.getLogger(__name__)

#: Words only: a failure message names what happened, never a path, a value or an exception's text.
FAILURE_WORDS = {
    "inputs_removed": "an evaluation array or view snapshot this report recorded is gone; rerun the report",
    "members_misaligned": "the members' rows no longer align; rerun the report",
    "engine_error": "the blend engine could not complete this report; see the miStudio log",
}


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.probe_blend_tasks.run_blend_evaluation",
    max_retries=0,
)
def run_blend_evaluation(self, report_id: str) -> Dict[str, Any]:
    from ..core.cancellation import OperatorCancelled, cancel_checker, record_progress
    from ..models.probe_blend import TERMINAL_STATUSES, ProbeBlendReport

    with self.get_db() as db:
        row = db.query(ProbeBlendReport).filter(ProbeBlendReport.id == report_id).first()
        if row is None:
            raise ValueError(
                f"blend report {report_id} has no row. The API commits it before dispatch, so it was deleted — "
                "the task stops rather than doing work nothing can read"
            )
        if row.status in TERMINAL_STATUSES:
            return {"report_id": report_id, "status": row.status}
        if row.cancel_requested_at is not None:
            row.status, row.completed_at = "cancelled", utc_now()
            db.commit()
            return {"report_id": report_id, "status": "cancelled"}
        row.status, row.started_at, row.heartbeat_at = "running", utc_now(), utc_now()
        db.commit()
        # Polled at EVERY check (no time throttle): a report makes a few dozen checks in all, and a cancel
        # must land at the next fold or candidate, not two seconds later.
        checker = cancel_checker("blend_report", report_id, min_interval_s=0)

        def should_stop() -> None:
            checker.raise_if_cancelled("between folds or candidates")

        def beat() -> None:
            record_progress("blend_report", report_id, heartbeat_at=utc_now())

        try:
            outcome = execute(db, row, should_stop=should_stop, beat=beat)
            return outcome
        except OperatorCancelled:
            _finish(db, report_id, "cancelled", None)
            return {"report_id": report_id, "status": "cancelled"}
        except BaseException as exc:  # noqa: BLE001 - every failure ends the row, nothing re-raised
            from ..services.probe_blend import engine

            code = getattr(exc, "code", None) if isinstance(exc, engine.EngineRefused) else None
            if isinstance(exc, InputsRemoved):
                code = "inputs_removed"
            code = code if code in FAILURE_WORDS else "engine_error"
            logger.error("blend report %s failed: %s (%s)", report_id, code, type(exc).__name__)
            _finish(db, report_id, "failed", code)
            return {"report_id": report_id, "status": "failed", "failure_code": code}


class InputsRemoved(Exception):
    pass


def _finish(db: Any, report_id: str, status: str, code: Any) -> None:
    from ..models.probe_blend import ProbeBlendReport
    from ..services.probe_blend.service import remove_staging

    remove_staging(report_id)
    db.rollback()
    row = db.query(ProbeBlendReport).filter(ProbeBlendReport.id == report_id).populate_existing().first()
    if row is None or (row.status in ("done", "cancelled")):
        return
    row.status = status
    row.completed_at = row.completed_at or utc_now()
    if status == "failed":
        row.failure_code = code
        row.failure_message = FAILURE_WORDS.get(code, FAILURE_WORDS["engine_error"])
    db.commit()


def engine_inputs(db: Any, row: Any) -> Any:
    """The engine's inputs, from `resolved` alone (plus the files it names)."""
    from ..models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation
    from ..services import view_columns
    from ..services.probe_blend import candidates as cand
    from ..services.probe_blend import engine
    from ..services.probe_blend.service import load_member_arrays, probes_root
    from ..services.probe_evaluation_scores import EvaluationArtifactError

    resolved = row.resolved
    members = [m["probe_id"] for m in resolved["members"]]
    views = resolved["views"]
    view_ids = [views["blend_fit"], *[t["id"] for t in views["tests"]], views["calibration"]["id"]]
    evaluation_ids = {e for picked in resolved["evaluations"].values() for e in picked.values()}
    evaluations = {e.id: e for e in db.query(ProbeMonitorEvaluation).filter(ProbeMonitorEvaluation.id.in_(evaluation_ids))}
    root = probes_root()
    arrays: Dict[str, Dict[str, Any]] = {}
    try:
        for view in view_ids:
            arrays[view] = {}
            for member in members:
                evaluation = evaluations.get(resolved["evaluations"][member][view])
                if evaluation is None:
                    raise InputsRemoved(view)
                arrays[view][member] = load_member_arrays(evaluation, root)
    except EvaluationArtifactError as exc:
        raise InputsRemoved(exc.check) from exc
    rows = {v.id: v for v in db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id.in_(view_ids))}
    if set(view_ids) - set(rows):
        raise InputsRemoved("view")
    group_codes = None
    column = resolved.get("group_column")
    try:
        if column:
            snap = view_columns.load(rows[views["blend_fit"]].column_snapshots[column], column)
            group_codes = np.asarray(snap.codes)
        columns: Dict[str, Dict[str, Any]] = {}
        for test in views["tests"]:
            declared = rows[test["id"]].column_snapshots or {}
            columns[test["id"]] = {}
            for name in sorted(declared):
                snap = view_columns.load(declared[name], name)
                columns[test["id"]][name] = (np.asarray(snap.codes), list(snap.values))
    except (KeyError, view_columns.SnapshotBroken, FileNotFoundError) as exc:
        raise InputsRemoved("snapshot") from exc
    rules_resolved = resolved.get("rules")
    rules = None
    if rules_resolved:
        from ..services.detector_success_report import rule_rows_sync

        # 040's cached verdicts under the set, engine and reading the report FIXED AT START (by split position).
        rules = {}
        for view in view_ids:
            found = rule_rows_sync(db, rows[view], rules_resolved["scope"], rules_resolved["rule_set_digest"],
                                   rules_resolved["engine_digest"])
            if found is None:
                raise InputsRemoved("rule_verdicts")
            rules[view] = found
    has_rule_set = bool((resolved.get("version") or {}).get("rule_set_digest"))
    return engine.EngineInputs(
        members=members, decisions=[m["decision"] for m in resolved["members"]],
        blend_fit=views["blend_fit"], tests=list(views["tests"]), calibration=views["calibration"]["id"],
        arrays=arrays, candidates=[cand.parse_name(n, len(members), has_rule_set) for n in resolved["candidates"]],
        folds=int(resolved["folds"]), seed=int(resolved["seed"]), target_fpr=float(resolved["target_fpr"]),
        min_rows_per_class=int(resolved["min_rows_per_class"]),
        min_aligned_fraction=float(resolved["min_aligned_fraction"]), floor=int(resolved["floor"]),
        group_codes=group_codes, columns=columns, min_subgroup_rows=int(resolved["min_subgroup_rows"]),
        criteria=list(resolved.get("criteria") or []),
        has_rule_set=has_rule_set, rules=rules,
        rules_identity=({k: rules_resolved[k] for k in ("rule_set_digest", "engine_digest")} if rules_resolved else None),
        rule_unknown_share_max=float((rules_resolved or {}).get("unknown_share_max", 1.0)),
    )


def execute(db: Any, row: Any, *, should_stop: Any, beat: Any, runner: Any = None) -> Dict[str, Any]:
    """Run the engine and commit the report: staging directory, `os.replace` into place, then `done`.
    `runner` is the engine's `run` (injected by tests that block or fake the fit)."""
    from ..services.probe_blend import assemble, engine
    from ..services.probe_blend import candidates as cand
    from ..services.probe_blend.service import blends_root

    inputs = engine_inputs(db, row)
    result = (runner or engine.run)(inputs, should_stop=should_stop, beat=beat)
    report = assemble.report(row, result)
    from ..services.probe_blend.report_format import canonical_bytes

    body = canonical_bytes(report)
    final = blends_root() / row.id
    staging = blends_root() / f"{row.id}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "report.json").write_bytes(body)
    for view, rows in result["rows"].items():
        np.save(staging / f"positions__{view}.npy", np.asarray(rows["positions"], dtype=np.int64))
        np.save(staging / f"labels__{view}.npy", np.asarray(rows["labels"], dtype=np.int8))
        if rows.get("rule_match") is not None:           # the rule subsets' input, read back by 040's reports
            np.save(staging / f"rules__{view}.npy", np.asarray(rows["rule_match"], dtype=np.int8))
        wrote_nfired = False
        for index, values in rows["values"].items():
            entry = report["candidates"][index]
            if entry["produces"] == "verdict" and not cand.is_rules_name(entry["name"]):
                if not wrote_nfired:                     # PT-2 / TI8: one array serves every verdict kind
                    np.save(staging / f"nfired__{view}.npy", np.asarray(values, dtype=np.float64))
                    wrote_nfired = True
                continue
            np.save(staging / f"c{index:02d}__{view}.npy", np.asarray(values, dtype=np.float64))
    # ⚠ NOT RESURRECTED (Review R1): the janitor may have failed this row (`time_limit`, `worker_lost`) or the
    # operator cancelled it after the engine's last check, while this task kept running. The terminal state the
    # row reached stands; nothing is published and its admission slot is not reclaimed by a late `done`.
    db.refresh(row)
    if row.status != "running" or row.cancel_requested_at is not None:
        shutil.rmtree(staging, ignore_errors=True)
        if row.status == "running":
            row.status, row.completed_at = "cancelled", utc_now()
            db.commit()
        return {"report_id": row.id, "status": row.status}
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)
    row.report_path = str(final / "report.json")
    row.report_sha256 = hashlib.sha256(body).hexdigest()
    row.report_bytes = len(body)
    row.rows_dir = str(final)
    row.status = "done"
    row.completed_at = utc_now()
    db.commit()
    return {"report_id": row.id, "status": "done", "report_sha256": row.report_sha256}
