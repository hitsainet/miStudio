"""The Redis side of per-card GPU workers: parked jobs and stranded card queues (Phase 3).

PARKED JOBS. A GPU job whose card is busy is handed back to wait
(``services/gpu_job_claim.JobHandoff`` with a delay). Celery's own ``countdown``
is the wrong tool on the Redis transport: the ETA message is delivered at once
and held UNACKED in some worker's memory until due, so a worker killed while
holding it strands the job for the full 12-hour visibility timeout. Instead the
re-dispatch is parked here, in a sorted set scored by its due time, and the GPU
supervisor republishes it when due. Nothing waits in a worker; a pod restart
loses nothing, because the set lives in Redis.

STRANDED CARD QUEUES. A pod restarted with a card removed spawns no worker for
it, so ``gpu.<uuid>`` for that card has no consumer. The supervisor moves those
messages to ``gpu.auto``; the worker that takes one resolves the request, finds
no such card, and the job fails with the placement's own message ("No GPU
'GPU-…' on this node. Available: …") — never a silent move to another card, and
never a job that waits for ever.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable, Optional

from .gpu_claim import AUTO_QUEUE, queue_for

logger = logging.getLogger(__name__)

#: The sorted set of parked GPU job re-dispatches, scored by due time (epoch s).
WAITING_KEY = "mistudio:gpu:waiting"

#: A parked entry whose republish failed is tried again this much later.
REPUBLISH_RETRY_S = 15.0

#: Card queues as ``gpu_claim.queue_for`` spells them.
CARD_QUEUE_PATTERN = "gpu.gpu-*"
STEERING_QUEUE_PATTERN = "steering.gpu-*"

#: kombu's Redis transport stores a priority's messages under the queue name plus
#: this separator and the priority; priority 0 uses the bare name.
_KOMBU_PRIORITY_SEP = "\x06\x16"


def redis_client():
    """A client on the Celery broker."""
    import redis

    from ..core.config import settings

    return redis.Redis.from_url(str(settings.celery_broker_url))


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


# ── jobs waiting for a GPU, for the janitors ────────────────────────────────

#: A job WAITING for a GPU — handed off, or waiting in place — is marked here,
#: keyed by its Celery task id. Every hand-off and every in-place poll writes it
#: again; a successful lease clears it.
WAITING_MARK_PREFIX = "mistudio:gpu:waiting-task:"

#: Long enough to cover one hand-off cycle (the park delay, the supervisor's
#: republish tick, a worker picking the job up and re-running its pre-placement
#: work) many times over; short enough that a dead job's mark is gone well before
#: any janitor's threshold.
WAITING_MARK_TTL_S = 300

#: Every queue a GPU job waits on: ``gpu.auto`` and each card's ``gpu.<uuid>``,
#: kombu's priority sub-queues included.
GPU_QUEUE_PATTERN = "gpu.*"


def mark_waiting(task_id: Optional[str], *, client: Any = None, ttl_s: float = WAITING_MARK_TTL_S) -> None:
    """Record that this job is waiting for a GPU. Never raises: it is narration for the janitors."""
    if not task_id:
        return
    try:
        (client if client is not None else redis_client()).set(
            WAITING_MARK_PREFIX + str(task_id), "1", ex=int(ttl_s)
        )
    except Exception:  # noqa: BLE001 - the job's placement is the point
        logger.warning("Could not mark GPU job %s as waiting; a janitor judges it by its row alone",
                       task_id, exc_info=True)


def clear_waiting(task_id: Optional[str], *, client: Any = None) -> None:
    """The job has its cards: it is no longer waiting. Never raises."""
    if not task_id:
        return
    try:
        (client if client is not None else redis_client()).delete(WAITING_MARK_PREFIX + str(task_id))
    except Exception:  # noqa: BLE001 - the mark expires anyway
        logger.warning("Could not clear the waiting mark of GPU job %s", task_id, exc_info=True)


def _parked_task_ids(client: Any) -> set:
    ids = set()
    for raw in client.zrange(WAITING_KEY, 0, -1):
        try:
            ids.add(str(json.loads(_text(raw))["options"]["task_id"]))
        except (ValueError, KeyError, TypeError):
            continue
    return ids


def _queued_task_ids(client: Any) -> set:
    """Task ids of the messages on every GPU queue (kombu stores ``headers.id``)."""
    ids = set()
    for raw_key in client.scan_iter(match=GPU_QUEUE_PATTERN):
        key = _text(raw_key)
        if _text(client.type(key)) != "list":
            continue
        for raw in client.lrange(key, 0, -1):
            try:
                ids.add(str(json.loads(_text(raw))["headers"]["id"]))
            except (ValueError, KeyError, TypeError):
                continue
    return ids


def is_waiting_for_gpu(task_id: Optional[str], *, client: Any = None) -> bool:
    """Whether a job is only WAITING for a GPU, so a janitor must not reap it as dead.

    Waiting means a fresh waiting mark, a parked re-dispatch, or a message on a
    GPU queue. A job whose worker died has none of these, so it is still reaped.
    When Redis cannot be read the job is spared, as every janitor treats a broker
    hiccup: a false reap of a live job is the worse error.
    """
    if not task_id:
        return False
    task_id = str(task_id)
    try:
        client = client if client is not None else redis_client()
        if client.exists(WAITING_MARK_PREFIX + task_id):
            return True
        return task_id in _parked_task_ids(client) or task_id in _queued_task_ids(client)
    except Exception:  # noqa: BLE001 - never condemn on an unreadable broker
        logger.warning("Could not tell whether GPU job %s is waiting; sparing it", task_id, exc_info=True)
        return True


def park_job(
    task_name: str,
    args: list,
    kwargs: dict,
    options: dict,
    *,
    due_at: float,
    client: Any = None,
    key: str = WAITING_KEY,
) -> None:
    """Park one re-dispatch until ``due_at`` (epoch seconds)."""
    entry = json.dumps(
        {"task": task_name, "args": list(args), "kwargs": dict(kwargs), "options": dict(options)},
        sort_keys=True,
    )
    (client if client is not None else redis_client()).zadd(key, {entry: due_at})


def release_due_jobs(
    client: Any,
    publish: Callable[[dict], None],
    *,
    now: float,
    key: str = WAITING_KEY,
    limit: int = 100,
) -> int:
    """Republish every parked entry due by ``now``. Returns how many were published.

    AT MOST ONCE. The entry is removed before it is published, and only by the
    caller whose ZREM removed it, so two releasers can never both publish it — a
    duplicate GPU job would run twice, on two cards. A publish that raises puts
    the entry back, due a little later.
    """
    released = 0
    for raw in client.zrangebyscore(key, "-inf", now, start=0, num=limit):
        if not client.zrem(key, raw):
            continue
        entry = json.loads(_text(raw))
        try:
            publish(entry)
        except Exception:  # noqa: BLE001 - keep the job; try again shortly
            logger.exception("Could not republish parked GPU job %s; retrying later", entry.get("task"))
            client.zadd(key, {raw: now + REPUBLISH_RETRY_S})
            continue
        released += 1
    return released


def publish_with_celery(entry: dict) -> None:
    """Republish a parked entry through Celery, by task name, with its options."""
    from ..core.celery_app import celery_app

    celery_app.send_task(entry["task"], args=entry["args"], kwargs=entry["kwargs"], **entry["options"])


def requeue_stranded_messages(
    client: Any,
    live_queues: Iterable[str],
    *,
    pattern: str,
    target: str,
) -> dict:
    """Move every message on a queue matching ``pattern`` that is not in ``live_queues`` to ``target``.

    RPOPLPUSH takes the OLDEST message (kombu publishes with LPUSH and consumes
    with BRPOP) and pushes it at the newest end of ``target``, so a moved batch
    keeps its order. Returns ``{queue: moved}``.

    NOTHING MOVES WHEN NOTHING IS LIVE (review round 1). An empty inventory cannot
    tell "this node has no GPU" from "NVML could not be read" (``list_cards``
    returns [] for both), and on the GPU node the second is the real case. Every
    card queue would look stranded, and every job waiting for a card that is
    still there would be moved to where it fails — irreversibly. A card that is
    really gone is swept as soon as any card is seen.
    """
    live_queues = list(live_queues)
    if not live_queues:
        logger.warning(
            "No GPU is visible, so no %s queue is treated as stranded; their jobs stay queued", pattern,
        )
        return {}
    live = set(live_queues) | {target}
    moved: dict = {}
    for raw_key in client.scan_iter(match=pattern):
        key = _text(raw_key)
        base = key.split(_KOMBU_PRIORITY_SEP, 1)[0]
        if base in live or _text(client.type(key)) != "list":
            continue
        count = 0
        while client.rpoplpush(key, target) is not None:
            count += 1
        if count:
            moved[base] = moved.get(base, 0) + count
    for queue, count in moved.items():
        logger.warning(
            "%d job(s) were queued for %s, whose GPU is not on this node; moved to %s, where "
            "each fails with the placement's message", count, queue, target,
        )
    return moved


def requeue_stranded_card_messages(client: Any, live_uuids: Iterable[str]) -> dict:
    """``gpu.<uuid>`` queues for cards with no worker -> ``gpu.auto``."""
    return requeue_stranded_messages(
        client, [queue_for(uuid) for uuid in live_uuids], pattern=CARD_QUEUE_PATTERN, target=AUTO_QUEUE,
    )


def maintenance_tick(
    live_uuids: Iterable[str],
    *,
    client_factory: Callable[[], Any] = redis_client,
    publish: Callable[[dict], None] = publish_with_celery,
    clock: Optional[Callable[[], float]] = None,
    wall_clock: Optional[Callable[[], float]] = None,
    release_every_s: float = 5.0,
    sweep_every_s: float = 60.0,
) -> Callable[[], None]:
    """The GPU supervisor's periodic work: republish due parked jobs, rescue stranded queues.

    Time-throttled; the first call does both (so a restart rescues at once).
    """
    import time

    clock = clock or time.monotonic
    wall_clock = wall_clock or time.time
    live = list(live_uuids)
    state = {"client": None, "released_at": None, "swept_at": None}

    def tick() -> None:
        if state["client"] is None:
            state["client"] = client_factory()
        client = state["client"]
        now = clock()
        if state["released_at"] is None or now - state["released_at"] >= release_every_s:
            state["released_at"] = now
            release_due_jobs(client, publish, now=wall_clock())
        if state["swept_at"] is None or now - state["swept_at"] >= sweep_every_s:
            state["swept_at"] = now
            requeue_stranded_card_messages(client, live)

    return tick
