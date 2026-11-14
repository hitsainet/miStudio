"""Drain check before the P3 window (ENH-009 042 FR-042.38, BR-065; FTID §3.16; TPD-13). READ-ONLY.

Run in the backend pod the day before and at the window's open (S001 runbook 3.15, 3.23):

    kubectl -n mistudio exec -i deploy/mistudio-backend -c backend -- python3 -m scripts.rebrand.inflight

It reports everything that would be cut off mid-flight by the move, and exits 1 when anything is found, 0 when
the world is quiet:

* every mapped table with a status column (`status`, or a `*_status` column such as `nlp_status`), DISCOVERED
  from `Base.metadata`, never listed: rows whose value is not in `TERMINAL[(table, column)]`. A value nobody
  classified therefore counts as in flight (fail closed), and `test_inflight_check.py` requires every value a
  column can take to be classified, so a new status cannot silently read as finished (TI-12);
* `agent_approval_requests` rows still `pending` (a status column like the rest: `pending` is the only
  non-terminal value there);
* the length of every Celery queue the routes name, the default queue, every per-card GPU and steering
  queue present in Redis (kombu's priority sub-queues included) and the parked-GPU-job set.

It writes nothing: one `SELECT <col>::text, count(*) … GROUP BY 1` per column, Redis `SCAN`/`TYPE`/`LLEN`/`ZCARD`.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: Values that mean "nothing is running for this row" — shared by most job tables.
_DONE = frozenset({"completed", "failed", "cancelled", "error", "ready", "deleted", "denied", "expired",
                   "consumed", "approved", "done", "passed", "refused", "retired", "ok", "not_checked",
                   "no_sets", "estimated", "transition", "complete", "already_complete", "already_cancelled",
                   "succeeded", "skipped", "blocked", "orphaned", "none", "SUCCESS", "FAILURE"})
#: Values that mean "a row is mid-flight". Documentation for the classification test only: the script counts
#: every value outside `TERMINAL[key]`, so a per-table entry wins (a feature's `pending` label is terminal).
ACTIVE = frozenset({"pending", "queued", "running", "processing", "downloading", "loading", "extracting",
                    "saving", "computing", "packaging", "initializing", "paused", "labeling", "converting",
                    "preparing", "pushing", "quantizing", "estimating", "in_progress", "deleting", "retrying",
                    "starting", "uploading",
                    # database enums that store the member NAME (`activation_extractions`, `datasets`, `models`, …)
                    "QUEUED", "LOADING", "EXTRACTING", "SAVING", "PROCESSING", "DOWNLOADING", "QUANTIZING"})

#: `(table, column)` → the values that are finished. Every discovered column must have an entry
#: (`test_every_status_value_classified`); a value outside its entry is counted as in flight.
TERMINAL: dict[tuple[str, str], frozenset[str]] = {
    ("activation_extractions", "status"): _DONE | {"COMPLETED", "FAILED", "CANCELLED"},
    ("agent_approval_requests", "status"): _DONE,
    ("circuit_capture_runs", "status"): _DONE,
    ("circuit_discovery_runs", "status"): _DONE,
    ("circuit_discovery_runs", "attribution_status"): _DONE,
    ("circuit_discovery_runs", "validation_status"): _DONE,
    ("circuits", "faithfulness_status"): _DONE,
    ("circuits", "calibration_status"): _DONE,
    ("dataset_tokenizations", "status"): _DONE | {"READY", "ERROR"},
    ("datasets", "status"): _DONE | {"READY", "ERROR"},
    ("enhanced_labeling_jobs", "status"): _DONE,
    ("external_saes", "status"): _DONE,
    ("extraction_jobs", "status"): _DONE,
    ("extraction_jobs", "nlp_status"): _DONE,
    ("feature_grouping_runs", "status"): _DONE,
    # A feature's label state, not a job: `pending` is "never labelled" (the default of every feature), so only
    # `in_progress` (a labelling pass holding it) is in flight.
    ("features", "label_status"): _DONE | {"pending"},
    ("labeling_jobs", "status"): _DONE,
    ("labeling_resume_sweeps", "status"): _DONE,
    ("labeling_trial_runs", "status"): _DONE,
    ("models", "status"): _DONE | {"READY", "ERROR"},
    ("neuronpedia_export_jobs", "status"): _DONE,
    ("neuronpedia_pushes", "status"): _DONE,
    ("probe_blend_reports", "status"): _DONE,
    ("probe_monitor_evaluations", "status"): _DONE,
    ("probe_monitor_judge_runs", "status"): _DONE,
    ("probe_monitor_runs", "status"): _DONE,
    ("probe_rules_evaluations", "status"): _DONE,
    ("probe_spot_checks", "status"): _DONE,
    ("steering_record_runs", "status"): _DONE,
    ("task_queue", "status"): _DONE,
    ("trainings", "status"): _DONE,
}

#: The parked-GPU-job set and the queue patterns present only while a card has work (gpu_worker_queues.py).
QUEUE_PATTERNS = ("gpu.*", "steering.*")
DEFAULT_QUEUE = "celery"


def is_status_column(name: str) -> bool:
    return name == "status" or name.endswith("_status")


def status_columns(metadata: Any) -> list[tuple[str, str]]:
    """Every `(table, column)` of the mapped metadata whose name is a status column, sorted."""
    return sorted((t.name, c.name) for t in metadata.tables.values() for c in t.columns if is_status_column(c.name))


def read_counts(conn: Any, columns: Iterable[tuple[str, str]]) -> dict[tuple[str, str], dict[str, int]]:
    """`{(table, column): {value: rows}}` — NULL is not counted (a nullable phase column not yet started)."""
    from sqlalchemy import text

    out: dict[tuple[str, str], dict[str, int]] = {}
    for table, column in columns:
        rows = conn.execute(text(
            f'SELECT "{column}"::text AS v, count(*) AS n FROM "{table}" WHERE "{column}" IS NOT NULL GROUP BY 1'))
        out[(table, column)] = {r.v: int(r.n) for r in rows}
    return out


def queue_names() -> list[str]:
    """The queues the Celery routes name, plus the default one."""
    from src.core.celery_app import celery_app

    routes = celery_app.conf.task_routes or {}
    names = {DEFAULT_QUEUE, str(celery_app.conf.task_default_queue or DEFAULT_QUEUE)}
    for route in (routes.values() if isinstance(routes, Mapping) else []):
        if isinstance(route, Mapping) and route.get("queue"):
            names.add(str(route["queue"]))
    return sorted(names)


def read_queues(client: Any, names: Iterable[str]) -> dict[str, int]:
    """`{key: waiting messages}` for every named queue (and its kombu priority sub-queues), every key matching
    `QUEUE_PATTERNS`, and the parked-GPU-job set; empty keys are left out."""
    from src.services.gpu_worker_queues import WAITING_KEY

    keys: set[str] = set()
    for name in names:
        keys.add(name)
        keys.update(_text(k) for k in client.scan_iter(match=f"{name}\x06\x16*"))
    for pattern in QUEUE_PATTERNS:
        keys.update(_text(k) for k in client.scan_iter(match=pattern))
    keys.add(WAITING_KEY)
    out = {}
    for key in sorted(keys):
        kind = _text(client.type(key))
        n = client.llen(key) if kind == "list" else client.zcard(key) if kind == "zset" else 0
        if n:
            out[key] = int(n)
    return out


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def findings(counts: Mapping[tuple[str, str], Mapping[str, int]], queues: Mapping[str, int]) -> list[str]:
    out = []
    for (table, column), values in sorted(counts.items()):
        done = TERMINAL.get((table, column), frozenset())
        for value, n in sorted(values.items()):
            if n and value not in done:
                out.append(f"{table}.{column} = {value!r}: {n} row(s)")
    out += [f"queue {key}: {n} message(s)" for key, n in sorted(queues.items()) if n]
    return out


def main(argv: list[str] | None = None, *, connect: Callable[[], Any] | None = None,
         redis: Callable[[], Any] | None = None, out=sys.stdout) -> int:
    from src.core.database import Base
    import src.models  # noqa: F401 — registers every mapped class on Base.metadata

    if connect is None:
        from src.core.database import sync_engine
        connect = sync_engine.connect
    if redis is None:
        from src.services.gpu_worker_queues import redis_client
        redis = redis_client
    with connect() as conn:
        counts = read_counts(conn, status_columns(Base.metadata))
    queues = read_queues(redis(), queue_names())
    found = findings(counts, queues)
    for line in found:
        print(f"IN FLIGHT: {line}", file=out)
    print(f"inflight: {len(counts)} status columns, {len(queues)} non-empty queue(s); "
          f"{'NOT CLEAR: ' + str(len(found)) + ' finding(s)' if found else 'clear'}", file=out)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
