"""Builds the backend's `legacy_names` and `legacy_names_meta` (042 FR-042.9; FTDD §5.1; FTID §3.3; CF-18).

`in_use` = the API process's own registry (unflushed uses included) ∪ every stored use whose `last_seen` is
within `IN_USE_WINDOW` (BR-023: 24 hours in all four apps), deduplicated by (kind, old) keeping the newest.
`also_set` / `conflicts` = the API process's environment ∪ the latest report of each process that POSTs
(the MCP pod). `error` is `type(exc).__name__` on EVERY failed store read (CF-18: a failed read must never
read as "nothing in use"); the block then answers from memory and the route still answers 200.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from src.core import legacy_names
from src.schemas.legacy_names import LegacyNamesBlock, LegacyNamesMeta

IN_USE_WINDOW = timedelta(hours=24)

#: The latest `also_set` / `conflicts` each POSTing reporter sent ("now" facts, not stored).
_remote_lock = threading.Lock()
_remote_pairs: dict[str, tuple[list[dict[str, str]], list[dict[str, str]]]] = {}


def note_remote_pairs(reporter: str, also_set: list[dict[str, str]], conflicts: list[dict[str, str]]) -> None:
    with _remote_lock:
        _remote_pairs[reporter] = (list(also_set), list(conflicts))


def reset_remote_pairs() -> None:
    with _remote_lock:
        _remote_pairs.clear()


def _aware(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _pairs(*lists: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = {(p["old"], p["new"]) for lst in lists for p in lst}
    return [{"old": o, "new": n} for o, n in sorted(seen)]


def build(now: Optional[datetime] = None,
          read: Optional[Callable[[], tuple[list, dict, Optional[str]]]] = None,
          error: Optional[str] = None) -> dict[str, Any]:
    """`{"legacy_names": …, "legacy_names_meta": …}`, validated by the schemas. With `error` given the store
    is not read at all (the route's own bound expired) and the block answers from memory."""
    from src.services import legacy_names_store

    now = now or datetime.now(timezone.utc)
    if error is None:
        rows, reporters, error = (read or legacy_names_store.read)()
    else:
        rows, reporters = [], {}
    mine = legacy_names.snapshot(now)
    cutoff = now - IN_USE_WINDOW
    newest: dict[tuple[str, str], dict[str, Any]] = {}
    for use in mine["in_use"]:
        newest[(use["kind"], use["old"])] = {**use, "last_seen": _aware(use["last_seen"])}
    for row in rows:
        seen = _aware(row["last_seen"])
        if seen < cutoff:
            continue
        key = (row["kind"], row["old"])
        if key not in newest or seen > newest[key]["last_seen"]:
            newest[key] = {"kind": row["kind"], "old": row["old"], "new": row["new"], "last_seen": seen}
    in_use = [newest[k] for k in sorted(newest)]
    with _remote_lock:
        remote = list(_remote_pairs.values())
    block = LegacyNamesBlock(
        in_use=in_use,
        also_set=_pairs(mine["also_set"], *(r[0] for r in remote)),
        conflicts=_pairs(mine["conflicts"], *(r[1] for r in remote)),
        retired_but_set=mine["retired_but_set"],
    )
    own = legacy_names.meta()
    first_reports = [_aware(first) for first, _last in reporters.values()]
    meta = LegacyNamesMeta(
        observing_since=min(first_reports) if first_reports else _aware(own["observing_since"]),
        in_use_window_hours=int(IN_USE_WINDOW.total_seconds() // 3600),
        watching=own["watching"],
        unobserved_hosts=own["unobserved_hosts"],
        reporters={name: _aware(last) for name, (_first, last) in sorted(reporters.items())},
        error=error,
    )
    return {"legacy_names": block.model_dump(mode="json"), "legacy_names_meta": meta.model_dump(mode="json")}
