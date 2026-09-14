"""The API side of one steering worker per GPU (multi-GPU Phase 3).

Steering runs in a dedicated solo worker the API process spawns on demand and
the worker's own post-task hook exits (see ``workers/steering_tasks.py``). With
one such worker for the node, a steering request for one card waited behind a
generation on the other. In per-card mode (``settings.gpu_worker_mode``):

* a request is sent to the steering queue of ONE card — the card it names, or for
  Auto the idle card with the most free memory now (a generation is seconds, so
  "now" is effectively when it starts); the task still leases its cards in place
  when it places, so it never runs on a card another job holds;
* that card's worker is spawned with ``MISTUDIO_WORKER_GPU_UUID`` set, its own
  queue, hostname, PID file and log (``services/steering_worker_card.py``);
* workers are found by PID file and killed only by the PIDs this process
  recorded when it spawned them — NEVER by matching a command line.

Single mode keeps ``api/v1/endpoints/steering.py``'s one worker unchanged; every
entry here returns control to it when there is no card.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
from typing import Any, Callable, Iterable, Optional

from ...core.config import settings
from ...services import steering_worker_card as paths
from ...services.gpu_dispatch import is_named_card, per_card_workers
from ...services.gpu_placement import AUTO, GpuCard, GpuRequest, is_all, list_cards, normalise_uuid
from ...workers.gpu_supervisor import WORKER_GPU_ENV

logger = logging.getLogger(__name__)

#: PIDs of per-card steering workers THIS process spawned, by card.
SPAWNED_PIDS_BY_CARD: dict = {}

#: The start time (clock ticks since boot) of every steering worker this process
#: spawned, by PID — the single worker's and each card's. A PID is IDENTITY only
#: while its process lives: once the worker exits, the kernel may give the number to
#: any process, and a kill by the recorded number alone would SIGKILL that one.
SPAWN_START_TICKS: dict = {}

#: How long a freshly spawned worker has to write its PID file.
SPAWN_WAIT_S = 10


def _live_leases() -> dict:
    from ...core.database import get_sync_db
    from ...services import gpu_leases

    try:
        with get_sync_db() as db:
            return gpu_leases.live_leases(db)
    except Exception:  # noqa: BLE001 - a lease read must not stop a steering request
        logger.warning("Could not read GPU leases to choose a steering card", exc_info=True)
        return {}


def choose_steering_card(gpu_request: GpuRequest, cards: Iterable[GpuCard], leases: dict) -> Optional[str]:
    """The card whose steering worker takes a request. Pure.

    A named card; otherwise the idle card with the most free memory (any card if
    none is idle — the worker waits for a lease). None with no cards.
    """
    cards = list(cards)
    if not cards:
        return None
    if is_named_card(gpu_request):
        wanted = normalise_uuid(gpu_request)
        return next((card.uuid for card in cards if normalise_uuid(card.uuid) == wanted), str(gpu_request))
    busy = {normalise_uuid(uuid) for uuid in leases}
    pool = [card for card in cards if normalise_uuid(card.uuid) not in busy] or cards
    return max(pool, key=lambda card: (card.free_mb, -card.index)).uuid


async def steering_card_for(gpu_request: GpuRequest) -> Optional[str]:
    """The card for a steering request in per-card mode; None in single mode."""
    if not per_card_workers():
        return None
    cards = await asyncio.to_thread(list_cards)
    leases = await asyncio.to_thread(_live_leases)
    return choose_steering_card(gpu_request, cards, leases)


def steering_queue_option(card: Optional[str]) -> dict:
    """``apply_async`` options for a card's worker; nothing for the single worker."""
    return {} if card is None else {"queue": paths.queue(card)}


def spawn_card_worker(card: str, popen: Callable = subprocess.Popen) -> int:
    """Start one card's steering worker and RECORD ITS PID. Returns the PID."""
    backend_dir = settings.backend_dir
    venv_celery = backend_dir / "venv" / "bin" / "celery"
    celery_bin = str(venv_celery) if venv_celery.exists() else "celery"
    env = os.environ.copy()
    env[WORKER_GPU_ENV] = card
    log_file = open(paths.log_file(card), "a")
    try:
        spawned = popen(
            [
                celery_bin, "-A", "src.core.celery_app", "worker",
                "-Q", paths.consumed_queues(card), "-c", "1", "--pool=solo", "--loglevel=info",
                f"--hostname={paths.hostname(card)}", "--max-tasks-per-child=1",
                f"--pidfile={paths.pid_file(card)}",
            ],
            cwd=str(backend_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        SPAWNED_PIDS_BY_CARD.setdefault(card, set()).add(spawned.pid)
        record_spawned(spawned.pid)
        return spawned.pid
    finally:
        log_file.close()


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a running process — a ZOMBIE is not.

    The API spawns steering workers and never waits for them, so a worker that
    exited (it exits after every generation) or was killed stays a zombie until
    Python's subprocess module reaps it at the NEXT spawn. ``os.kill(pid, 0)``
    succeeds on a zombie, so a stale PID file read as a live worker: the mode
    endpoint said steering was on, and exit-mode reported "may still be running".
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state != "Z"


# ── which process a PID is (Phase 3 review round 2) ────────────────────────
#
# Exit-mode, the ensure before each request and the orphan sweep SIGKILL steering
# workers by PID: the PIDs recorded at spawn and the PID in a worker's PID file.
# Both outlive the worker. A worker that is SIGKILLed leaves its PID file behind
# (Celery removes it only on a clean exit), and the recorded set is never pruned,
# so after the number is reused either path would kill an unrelated process —
# the API process itself, or anything else in the container. A PID is acted on
# only when /proc says it is still the process that was spawned or that wrote
# the file.


def process_start_ticks(pid: int) -> Optional[int]:
    """When ``pid`` started, in clock ticks since boot (/proc/<pid>/stat field 22), or None."""
    try:
        with open(f"/proc/{int(pid)}/stat") as stat:
            fields = stat.read().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _boot_time() -> Optional[float]:
    try:
        with open("/proc/stat") as stat:
            for line in stat:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


#: Slack between a process's start (/proc/stat's boot time is whole seconds) and
#: the modification time of a file it wrote.
PID_FILE_CLOCK_SLACK_S = 2.0


def process_started_at(pid: int) -> Optional[float]:
    """When ``pid`` started, as epoch seconds (whole-second precision), or None."""
    ticks, boot = process_start_ticks(pid), _boot_time()
    if ticks is None or boot is None:
        return None
    return boot + ticks / os.sysconf("SC_CLK_TCK")


def record_spawned(pid: int) -> None:
    """Remember which process a freshly spawned worker's PID names. Called right after Popen."""
    ticks = process_start_ticks(pid)
    if ticks is not None:
        SPAWN_START_TICKS[pid] = ticks


def is_spawned_process(pid: int) -> bool:
    """Whether ``pid`` is still the very process this API process spawned under that number.

    False when nothing was recorded for it, or when a process with another start time
    now has the number — never a guess in favour of killing.
    """
    recorded = SPAWN_START_TICKS.get(pid)
    return recorded is not None and process_start_ticks(pid) == recorded


def pid_file_names_process(pid: int, pid_file) -> bool:
    """Whether the process now numbered ``pid`` is the one that wrote ``pid_file``.

    A worker writes its PID file after it starts, so its process started no later
    than the file's modification time. A process that got the number after that
    worker died started later than the file — and is not the worker.
    """
    started = process_started_at(pid)
    try:
        written = os.stat(pid_file).st_mtime
    except OSError:
        return False
    return started is not None and started <= written + PID_FILE_CLOCK_SLACK_S


def worker_pid_from_file(pid_file) -> Optional[int]:
    """The PID in ``pid_file`` when it names the live process that wrote it, else None."""
    try:
        pid = int(open(pid_file).read().strip())
    except (OSError, ValueError):
        return None
    if not pid_alive(pid):
        return None
    if not pid_file_names_process(pid, pid_file):
        logger.warning(
            "%s names PID %s, but that process started after the file was written; it is not the "
            "steering worker, and it is left alone", pid_file, pid,
        )
        return None
    return pid


def kill_spawned(pid: int) -> bool:
    """SIGKILL a PID recorded at spawn — only while it is still that process. Returns whether signalled."""
    same = is_spawned_process(pid)
    SPAWN_START_TICKS.pop(pid, None)
    if not same:
        return False
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        logger.warning("Not permitted to kill steering worker PID %s; leaving it", pid)
        return False


def card_worker_pid(card: str) -> Optional[int]:
    """The live PID in a card's PID file, or None. No command-line matching; see
    :func:`worker_pid_from_file` for why a live PID is not enough."""
    return worker_pid_from_file(paths.pid_file(card))


def card_worker_status(cards: Iterable[GpuCard]) -> list:
    """Every card's steering worker, running or not, for the mode endpoint.

    ``busy`` only when the card's busy marker names the live worker (a killed
    worker's leftover marker does not make its successor busy).
    """
    from ...workers.steering_worker_state import read_busy_marker

    out = []
    for card in cards:
        pid = card_worker_pid(card.uuid)
        marker = read_busy_marker(card.uuid) if pid else None
        busy = marker is not None and marker.get("pid") in (pid, -1)
        out.append({
            "card": card.uuid, "index": card.index, "name": card.name, "worker_pid": pid,
            "busy": busy, "task_id": marker.get("task_id") if busy else None,
        })
    return out


def mode_cards(gpu_request: Optional[GpuRequest], cards: Iterable[GpuCard], leases: dict, *, default_all: bool) -> list:
    """The cards an enter- or exit-mode request acts on. Pure.

    ``"all"`` is every card; a named card is that card; Auto is the card a steering
    request would go to now (:func:`choose_steering_card`). No request is every
    card when ``default_all`` (exit-mode), otherwise Auto (enter-mode).
    """
    cards = list(cards)
    if not cards:
        return []
    if gpu_request is None:
        if default_all:
            return [card.uuid for card in cards]
        gpu_request = AUTO
    if is_all(gpu_request):
        return [card.uuid for card in cards]
    chosen = choose_steering_card(gpu_request, cards, leases)
    return [chosen] if chosen else []


async def enter_card_workers(card_uuids: Iterable[str], *, ensure: Callable = None) -> list:
    """Start the steering worker of each card that has none; a running one is left as it is."""
    ensure = ensure or ensure_card_worker
    results = []
    for card in card_uuids:
        pid = await asyncio.to_thread(card_worker_pid, card)
        if pid:
            results.append({"card": card, "ok": True, "worker_pid": pid, "already_active": True})
            continue
        ok, pid = await ensure(card)
        results.append({"card": card, "ok": ok, "worker_pid": pid, "already_active": False})
    return results


def kill_card_worker(card: str) -> int:
    """SIGKILL one card's steering worker: the PIDs recorded for it at spawn, then its PID file.

    Never another card's worker, never a command-line pattern. Clears the card's
    PID file and busy marker. Returns how many processes were signalled.
    """
    signalled = set()
    for pid in sorted(SPAWNED_PIDS_BY_CARD.pop(card, set())):
        if kill_spawned(pid):
            signalled.add(pid)
    from_file = card_worker_pid(card)
    if from_file and from_file not in signalled:
        try:
            os.kill(from_file, signal.SIGKILL)
            signalled.add(from_file)
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warning("Not permitted to kill steering worker PID %s; leaving it", from_file)
    _clear_card_files(card)
    return len(signalled)


def kill_card_workers(card: Optional[str] = None) -> int:
    """SIGKILL the steering workers this process spawned — for one card, or every card.

    Only PIDs recorded at spawn. Returns how many were signalled.
    """
    killed = 0
    for key in ([card] if card is not None else list(SPAWNED_PIDS_BY_CARD)):
        for pid in sorted(SPAWNED_PIDS_BY_CARD.pop(key, set())):
            if kill_spawned(pid):
                killed += 1
    return killed


def kill_all_card_workers(cards: Iterable[GpuCard]) -> int:
    """Every card's steering worker: the PIDs this process recorded, then each card's PID file.

    The PID file covers a worker spawned before this API process restarted, whose
    PID the in-memory record lost. Both are PIDs the spawn tracked — never a
    command-line pattern. Returns how many were signalled.
    """
    killed = 0
    for card in cards:
        killed += kill_card_worker(card.uuid)
    # Recorded for a card that has since left the inventory.
    return killed + kill_card_workers()


def _clear_card_files(card: str) -> None:
    for path in (paths.pid_file(card), paths.busy_marker(card)):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


async def ensure_card_worker(card: str) -> tuple:
    """A FRESH steering worker for one card, as the single worker's ensure does for the node.

    An idle worker is replaced; a worker mid-generation (its busy marker names its
    PID) is left alone and the task queues behind it.
    """
    from ...workers.steering_worker_state import read_busy_marker

    pid = await asyncio.to_thread(card_worker_pid, card)
    if pid:
        busy = await asyncio.to_thread(read_busy_marker, card)
        if busy is not None and busy.get("pid") in (pid, -1):
            logger.info("Steering worker PID %s for %s is mid-task; queueing behind it", pid, card)
            return True, pid
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warning("Not permitted to kill steering worker PID %s", pid)
        kill_card_workers(card)
        await asyncio.sleep(1)
    _clear_card_files(card)
    try:
        spawn_card_worker(card)
    except Exception:  # noqa: BLE001 - reported as a failed start
        logger.exception("Failed to spawn the steering worker for %s", card)
        return False, None
    for _ in range(SPAWN_WAIT_S):
        await asyncio.sleep(1)
        pid = await asyncio.to_thread(card_worker_pid, card)
        if pid:
            logger.info("Started steering worker PID %s for %s", pid, card)
            return True, pid
    logger.error("The steering worker for %s did not start within %d s", card, SPAWN_WAIT_S)
    return False, None


async def ensure_steering_worker(card: Optional[str], single_worker_ensure: Callable) -> tuple:
    """The per-card ensure, or — with no card — the single worker's own."""
    if card is None:
        return await single_worker_ensure()
    return await ensure_card_worker(card)


def running_card_workers(cards: Iterable[GpuCard]) -> list:
    """(card uuid, pid) for every card whose steering worker is alive."""
    out = []
    for card in cards:
        pid = card_worker_pid(card.uuid)
        if pid:
            out.append((card.uuid, pid))
    return out


async def _queue_depth(name: str) -> int:
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(str(settings.redis_url))
        try:
            return int(await client.llen(name))
        finally:
            await client.aclose()
    except Exception:  # noqa: BLE001 - no spawn rather than a failed beat cycle
        logger.exception("Could not read the depth of %s", name)
        return 0


def _rescue_stranded_queues(live_uuids: list) -> dict:
    from ...services.gpu_worker_queues import (
        STEERING_QUEUE_PATTERN,
        redis_client,
        requeue_stranded_messages,
    )

    return requeue_stranded_messages(
        redis_client(), [paths.queue(uuid) for uuid in live_uuids],
        pattern=STEERING_QUEUE_PATTERN, target=paths.LEGACY_QUEUE,
    )


async def reconcile_card_workers(
    *,
    depth: Callable = _queue_depth,
    ensure: Callable = ensure_card_worker,
    rescue: Callable = _rescue_stranded_queues,
    inventory: Callable = list_cards,
) -> dict:
    """Spawn a card's steering worker when its queue has work and nothing consumes it.

    Also moves messages off the steering queue of a card that has left the node to
    the legacy ``steering`` queue, which every card's worker drains; the task then
    refuses the missing card with the placement's message.
    """
    cards = await asyncio.to_thread(inventory)
    moved = await asyncio.to_thread(rescue, [card.uuid for card in cards])
    spawned = []
    for card in cards:
        if await depth(paths.queue(card.uuid)) > 0 and card_worker_pid(card.uuid) is None:
            ok, pid = await ensure(card.uuid)
            spawned.append({"card": card.uuid, "ok": ok, "worker_pid": pid})
    legacy = await depth(paths.LEGACY_QUEUE)
    if legacy > 0 and not running_card_workers(cards) and not spawned:
        card = choose_steering_card(AUTO, cards, await asyncio.to_thread(_live_leases))
        if card is not None:
            ok, pid = await ensure(card)
            spawned.append({"card": card, "ok": ok, "worker_pid": pid})
    return {
        "status": "ok" if all(s["ok"] for s in spawned) else "error",
        "action": "spawned" if spawned else "none",
        "spawned": spawned,
        "moved": moved,
        "legacy_queue_depth": legacy,
    }
