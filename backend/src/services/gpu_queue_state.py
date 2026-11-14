"""What is sitting on the GPU queues, read from the broker itself (ENH-006 D116, G13 and the wedge watch).

An evaluation, a re-cut or a score has no row of its own, so while it waits the only record of it is its message.
`list_active_jobs` reads tables, and one such evaluation sat in `gpu.auto` unseen behind a stuck worker. And a worker
that finishes a task and never acks it takes nothing more (prefetch 1: kombu stops polling until the ack), while looking
exactly like an idle worker: it logs nothing and holds no card. Only the broker shows either.

So this reads kombu's own structures, read-only:
  * each GPU queue, a Redis list of messages;
  * `unacked`, a hash of delivery tag to `[message, exchange, routing_key]`, with `unacked_index`, scored by delivery time;
  * the parked hand-offs, `gpu_worker_queues.WAITING_KEY`.

NAMES COME FROM THE MESSAGES, never from a list of tasks, so a new GPU task is visible the day it ships. Arguments are
never returned: a score's arguments are the text being scored, and these answers leave the box.

Nothing here restarts, acks, requeues or signals anything. `assess` says what it sees.
"""
from __future__ import annotations

import base64
import inspect
import json
import logging
import time
from datetime import datetime
from typing import Any, Optional

from .gpu_worker_queues import (
    CARD_QUEUE_PATTERN,
    GPU_QUEUE_PATTERN,
    STEERING_QUEUE_PATTERN,
    WAITING_KEY,
    _text,
)

logger = logging.getLogger(__name__)

UNACKED_KEY = "unacked"
UNACKED_INDEX_KEY = "unacked_index"

#: A finished task's ack follows within seconds (the liveness stop is bounded at about 7 s). Past this it is not coming.
FINISHED_GRACE_S = 120.0

#: How long "finished and unacked, messages waiting, no card held" must hold before it is called wedged rather than
#: suspect: a healthy worker between two jobs can show all three for a moment.
WEDGE_CONFIRM_S = 300.0

#: Set by the beat check when the full condition first holds, deleted when it stops. Epoch seconds.
FIRST_SEEN_KEY = "mistudio:gpu:wedge-first-seen"
FIRST_SEEN_TTL_S = 7 * 24 * 3600

READY_STATES = {"SUCCESS", "FAILURE", "REVOKED"}


def _client() -> Any:
    """Short timeouts: this is read on an operator's request, and must answer "cannot read" rather than hang."""
    import redis

    from ..core.config import settings

    return redis.Redis.from_url(str(settings.celery_broker_url), socket_connect_timeout=1.0, socket_timeout=2.0)


def _is_gpu_queue(queue: Optional[str]) -> bool:
    return bool(queue) and (queue.startswith("gpu.") or queue.startswith("steering.gpu-"))


def _card_of(queue: Optional[str]) -> Optional[str]:
    """The card a queue names: `gpu.<uuid>` or `steering.<uuid>`; None for `gpu.auto`."""
    if not queue:
        return None
    name = queue.split("\x06", 1)[0]
    rest = name.split(".", 1)[1] if "." in name else ""
    return rest if rest.startswith("gpu-") else None


def _facts(message: Any) -> Optional[dict[str, Any]]:
    """Task id and name from a kombu message dict. None for one that is not a Celery task message."""
    try:
        headers = message["headers"]
        return {"task_id": str(headers["id"]), "task": headers.get("task")}
    except (KeyError, TypeError):
        return None


def orphan_reason(name: Optional[str], task_id: Optional[str], args: list, kwargs: dict) -> Optional[str]:
    """Why a WAITING message will do nothing when a worker takes it, or None (ENH-006 G41).

    Found the way the task's own guard finds its row — the `refuses_terminal_row` declaration, read from the live
    registry: `row_missing` when the row is gone and the scope says a missing row stops the task
    (`missing_row="cancelled"`, e.g. a deleted probe run); `row_<status>` when the row has ended. A task with no
    declaration, a live row, or a row that cannot be read is None: never orphaned on a guess."""
    from ..core.cancellation import get_scope, row_state_of
    from ..core.celery_app import celery_app

    task = celery_app.tasks.get(name) if name else None
    declared = getattr(getattr(task, "run", None), "__refuses_terminal_row__", None)
    if declared is None:
        return None
    kind, argument = declared
    target = task_id if argument is None else (
        inspect.signature(task.run).bind_partial(*args, **kwargs).arguments.get(argument))
    if target is None:
        return None
    scope = get_scope(kind)
    found = row_state_of(kind, target)
    if found == "missing":
        return "row_missing" if scope.missing_row == "cancelled" else None
    return f"row_{found}" if found in scope.terminal_values else None


def _orphan_of(entry: dict[str, Any], args: list, kwargs: dict) -> Optional[str]:
    try:
        return orphan_reason(entry.get("task"), entry.get("task_id"), args, kwargs)
    except Exception:  # noqa: BLE001 - an unreadable row is not an orphan
        logger.warning("Could not read the row of queued task %s", entry.get("task_id"), exc_info=True)
        return None


def read_queued(client: Any, *, with_rows: bool = False) -> list[dict[str, Any]]:
    """Every message waiting on a GPU queue, and every parked hand-off. Raises if the broker cannot be read.

    `with_rows` adds `orphan_reason` to each (`orphan_reason`); the arguments it reads are never returned."""
    out: list[dict[str, Any]] = []
    seen_keys = set()
    for pattern in (GPU_QUEUE_PATTERN, CARD_QUEUE_PATTERN, STEERING_QUEUE_PATTERN):
        for raw_key in client.scan_iter(match=pattern):
            key = _text(raw_key)
            if key in seen_keys or _text(client.type(key)) != "list":
                continue
            seen_keys.add(key)
            for raw in client.lrange(key, 0, -1):
                message = None
                try:
                    message = json.loads(_text(raw))
                    facts = _facts(message)
                except ValueError:
                    facts = None
                if facts is None:
                    logger.warning("A message on %s is not a task message; counted, not named", key)
                    facts = {"task_id": None, "task": None}
                    message = None
                item = {**facts, "queue": key.split("\x06", 1)[0], "state": "queued"}
                if with_rows:
                    try:
                        args, kwargs = _call_arguments(message) if message is not None else ([], {})
                    except (ValueError, KeyError, TypeError, IndexError):
                        args, kwargs = [], {}
                    item["orphan_reason"] = _orphan_of(item, args, kwargs) if message is not None else None
                out.append(item)
    for raw, due_at in client.zrange(WAITING_KEY, 0, -1, withscores=True):
        try:
            entry = json.loads(_text(raw))
            item = {"task_id": str(entry["options"]["task_id"]), "task": entry.get("task"),
                    "queue": entry["options"].get("queue"), "state": "parked", "due_at": float(due_at)}
        except (ValueError, KeyError, TypeError):
            logger.warning("A parked GPU job could not be read")
            continue
        if with_rows:
            item["orphan_reason"] = _orphan_of(item, list(entry.get("args") or []), dict(entry.get("kwargs") or {}))
        out.append(item)
    return out


def read_unacked(client: Any) -> list[dict[str, Any]]:
    """Every message a worker holds and has not acked: its facts, routing key, delivery time and body."""
    delivered_at = {_text(tag): float(score) for tag, score in client.zrange(UNACKED_INDEX_KEY, 0, -1, withscores=True)}
    out = []
    for raw_tag, raw in client.hgetall(UNACKED_KEY).items():
        tag = _text(raw_tag)
        try:
            message, _exchange, routing_key = json.loads(_text(raw))
        except (ValueError, TypeError):
            continue
        facts = _facts(message)
        if facts is None:
            continue
        out.append({**facts, "delivery_tag": tag, "queue": _text(routing_key) if routing_key else None,
                    "delivered_at": delivered_at.get(tag), "_message": message})
    return out


def read_gpu_messages(client: Any = None, *, with_rows: bool = False) -> list[dict[str, Any]]:
    """G13: everything on the GPU queues, waiting or held by a worker. Raises when the broker cannot be read, so a
    caller reports it rather than reading it as "nothing queued". `with_rows` (G41) adds each WAITING message's
    `orphan_reason`; a delivered one is judged by `finished_evidence`, since a row cancelled under a running task is
    a task on its way to a checkpoint, not an orphan."""
    client = client if client is not None else _client()
    held = [
        {"task_id": m["task_id"], "task": m["task"], "queue": m["queue"], "state": "delivered"}
        for m in read_unacked(client) if _is_gpu_queue(m["queue"])
    ]
    if with_rows:
        for item in held:
            item["orphan_reason"] = None
    return read_queued(client, with_rows=with_rows) + held


def _call_arguments(message: dict[str, Any]) -> tuple[list, dict]:
    """A protocol-2 body: base64 of JSON `[args, kwargs, embed]`."""
    body = json.loads(base64.b64decode(message["body"]))
    return list(body[0]), dict(body[1])


def _row_status(message: dict[str, Any], task_id: str, name: Optional[str]) -> Optional[str]:
    """The terminal status of the task's own row, found the way the task's guard finds it: read from the live registry,
    never from a list here."""
    from ..core.cancellation import terminal_status_of
    from ..core.celery_app import celery_app

    task = celery_app.tasks.get(name) if name else None
    declared = getattr(getattr(task, "run", None), "__refuses_terminal_row__", None)
    if declared is None:
        return None
    kind, argument = declared
    if argument is None:
        return terminal_status_of(kind, task_id)
    args, kwargs = _call_arguments(message)
    bound = inspect.signature(task.run).bind_partial(*args, **kwargs).arguments
    return terminal_status_of(kind, bound.get(argument))


def _epoch(iso: Optional[str]) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(iso)).timestamp() if iso else None
    except ValueError:
        return None


def finished_evidence(held: dict[str, Any], client: Any, *, now: float) -> Optional[dict[str, Any]]:
    """Why this held message's task has ended, or None if it has not, or nothing says so.

    A HEALTHY LONG TASK CAN SHOW NONE OF THESE: no stamp and no result exist until it returns, and its row is live. A row
    cancelled under a running task is not "ended": the worker is still alive and on its way to its next checkpoint, which
    its liveness stamp says, so a terminal row counts only once the stamp has gone.
    """
    from ..workers import task_completion, task_liveness

    task_id, name = held["task_id"], held["task"]
    done = task_completion.completed_at(task_id, client=client)
    if done is not None:
        return {"evidence": "completed", "finished_at": _epoch(done)}
    raw = client.get(f"celery-task-meta-{task_id}")
    if raw is not None:
        try:
            meta = json.loads(raw)
        except ValueError:
            meta = {}
        if meta.get("status") in READY_STATES:
            return {"evidence": f"result {meta['status']}", "finished_at": _epoch(meta.get("date_done"))}
    try:
        status = _row_status(held["_message"], task_id, name)
    except Exception:  # noqa: BLE001 - the row is one witness of three
        logger.warning("Could not read the row of held task %s", task_id, exc_info=True)
        status = None
    if status is not None and task_liveness.has_live_stamp(task_id, client=client) is False:
        return {"evidence": f"row {status}", "finished_at": held.get("delivered_at")}
    return None


def _cards_held() -> int:
    from ..core.database import get_sync_db
    from .gpu_leases import host_reserves

    with get_sync_db() as db:
        return len(host_reserves(db))


def assess(*, client: Any = None, now: Optional[float] = None, cards_held: Optional[int] = None) -> dict[str, Any]:
    """`{state, stuck, waiting, cards_held, ...}`. `state`:

    * `ok`: no held message's task has ended;
    * `suspect`: one has, and either nothing is waiting, a card is held, or it has not lasted long enough to be sure. Also
      what a killed worker's message reads until the visibility timeout restores it;
    * `wedged`: one has, messages are waiting, no card is held, and that has lasted `WEDGE_CONFIRM_S`. `waiting` leaves
      out orphans (a message whose row is gone or has ended, which the worker will refuse); `orphaned` counts them;
    * `unknown`: the broker could not be read, which is not "ok".
    """
    now = time.time() if now is None else now
    try:
        client = client if client is not None else _client()
        held = read_unacked(client)
        stuck = []
        for message in held:
            evidence = finished_evidence(message, client, now=now)
            if evidence is None:
                continue
            finished_at = evidence["finished_at"]
            age = now - finished_at if finished_at is not None else None
            if age is None or age < FINISHED_GRACE_S:
                continue
            stuck.append({"task_id": message["task_id"], "task": message["task"], "queue": message["queue"],
                          "evidence": evidence["evidence"], "finished_seconds_ago": round(age)})
        if not stuck:
            return {"state": "ok", "stuck": [], "waiting": None, "cards_held": None}
        # An orphan (its row gone or ended, ENH-006 G41) is not work waiting: the worker will refuse it, so a stuck worker
        # with only orphans behind it blocks nothing yet. A row that cannot be read is never an orphan, so it counts.
        queued = [m for m in read_queued(client, with_rows=True) if m["state"] == "queued"]
        waiting = sum(1 for m in queued if not m.get("orphan_reason"))
        orphaned = len(queued) - waiting
        first_seen = client.get(FIRST_SEEN_KEY)
    except Exception as exc:  # noqa: BLE001 - never "ok" on an unreadable broker
        logger.warning("The worker watch could not read the broker: %s", exc)
        return {"state": "unknown", "stuck": [], "reason": f"{type(exc).__name__}: {exc}"}
    try:
        cards = _cards_held() if cards_held is None else cards_held
    except Exception as exc:  # noqa: BLE001
        return {"state": "suspect", "stuck": stuck, "waiting": waiting, "cards_held": None, "orphaned": orphaned,
                "reason": f"could not read the GPU leases: {type(exc).__name__}: {exc}"}
    condition = waiting > 0 and cards == 0
    confirmed = condition and first_seen is not None and now - float(first_seen) >= WEDGE_CONFIRM_S
    names = ", ".join(f"{s['task']} {s['task_id']} ({s['evidence']}, {s['finished_seconds_ago']} s ago, from {s['queue']})"
                      for s in stuck)
    answer = {"state": "wedged" if confirmed else "suspect", "stuck": stuck, "waiting": waiting, "cards_held": cards,
              "condition_holds": condition, "orphaned": orphaned}
    if confirmed:
        answer["message"] = (
            f"A GPU worker looks wedged: {names} ended and was never acked, so with prefetch 1 its worker takes nothing "
            f"more, while {waiting} message(s) wait and no card is held. Nothing restarts it automatically. Capture a stack "
            "first (`kill -USR2 <worker pid>` writes every thread to stderr), then restart the worker. A redelivered copy of "
            "a task that has ended does nothing."
        )
    else:
        answer["message"] = (
            f"{names} ended and was never acked. Either its worker is stuck or it was killed and the message is waiting for "
            "the visibility timeout. Not called wedged: "
            + ("the condition has not lasted long enough." if condition else
               ("a card is held." if cards else
                "nothing is waiting" + (f" ({orphaned} orphaned message(s), which the worker will refuse, are not "
                                        "counted)." if orphaned else ".")))
        )
    return answer
