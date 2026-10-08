"""One solo Celery worker per GPU, started from the live inventory (multi-GPU Phase 3, D7).

Every worker here is `--pool=solo -c 1`: one job at a time, and cooperative
cancellation depends on that (see core/cancellation.py). With one GPU worker for
the whole node, the second card sat idle whenever a job ran. This supervisor
starts one worker PER CARD:

* each consumes its card's own queue (`gpu.<uuid>`, for jobs that name the card)
  and the shared `gpu.auto` queue (Auto and "all" jobs, decision 5: a job's card
  is chosen when a worker is free to run it);
* each is told its card in `MISTUDIO_WORKER_GPU_UUID`, but still SEES every card
  — a split needs them all, and indices are matched by UUID anyway;
* adding a card is a restart, not an edit (D7).

A WORKER THAT EXITS IS RESTARTED ALONE (review round 1). This used to stop every
worker and exit, so an out-of-memory crash on the 3080 Ti also killed a training
on the 3090 — and a training acks its message when it starts, so it was never
redelivered. Now only the exited worker is restarted, after a backoff that doubles
with each exit in :attr:`RestartPolicy.window_s`. A worker that exits more than
:attr:`RestartPolicy.max_restarts_in_window` times in that window is CRASH-LOOPING:
it is logged as an error and not restarted again, and the others keep running.
While a card has no worker (backing off, or crash-looping) the supervisor holds a
GPU lease on it (:class:`CardAvailability`), so no other worker sends an Auto job
to a queue nothing consumes. When EVERY worker is crash-looping the supervisor
exits non-zero, so the container restarts and Kubernetes shows the crash loop.

STOPPING (SIGTERM/SIGINT) IS A WARM DRAIN, NEVER A COLD ONE. Every worker gets
SIGTERM once; an idle worker exits at once, a busy one finishes its task; the
supervisor waits for all of them up to :data:`DRAIN_TIMEOUT_S` and only then kills
what is left. It never sends SIGQUIT: a cold shutdown on the solo pool calls
`cancel_active_requests`, which raises NotImplementedError INSIDE the running task
("solo.TaskPool does not implement kill_job"), so the task fails and its
acks_late message is acked — the job is lost instead of redelivered (measured
against a real worker and Redis, 2026-09-14). The entrypoint must also exec this
process with no watching parent: `su` SIGKILLs its child two seconds after
SIGTERM (util-linux 2.37.2), which cut every drain to two seconds.

AS PID 1 it adopts orphaned processes, so it reaps their zombies
(:func:`reap_orphans`) — but never one of its own workers, whose exit status
`Popen.poll` must read.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

#: The env var a card's worker reads to know which card it runs jobs on.
WORKER_GPU_ENV = "MISTUDIO_WORKER_GPU_UUID"

#: How long a stop waits for busy workers before killing them. Below the pod's
#: `terminationGracePeriodSeconds` (30 s in k8s/base/backend.yaml), so the
#: supervisor, not the kubelet, decides what is killed and logs it.
DRAIN_TIMEOUT_S = 25.0

#: How long, in all, a stop waits for killed workers to be reaped.
KILL_WAIT_S = 3.0

#: Exit code when every worker is crash-looping.
EXIT_ALL_CRASH_LOOPING = 3

RUNNING = "running"
RESTARTING = "restarting"
CRASH_LOOPING = "crash_looping"
STOPPED = "stopped"


@dataclass(frozen=True)
class WorkerSpec:
    """How to start one worker: its argv, the environment it adds, and its card."""

    name: str
    argv: tuple[str, ...]
    env: dict
    #: The card UUID this worker runs jobs on; None for the no-GPU worker.
    card: Optional[str] = None


def worker_specs(
    cards: Sequence,
    *,
    python: str = sys.executable,
    max_tasks_per_child: int = 100,
    loglevel: str = "INFO",
) -> list[WorkerSpec]:
    """One spec per card, or a single `gpu.auto` worker when there are no cards.

    Pure: the inventory comes in, argv comes out, so every rule is testable
    without a GPU or a broker.
    """
    from ..services.gpu_claim import AUTO_QUEUE, queue_for

    def spec(name: str, queues: Iterable[str], hostname: str, env: dict, card: Optional[str]) -> WorkerSpec:
        return WorkerSpec(
            name=name,
            argv=(
                python, "-m", "celery", "-A", "src.core.celery_app", "worker",
                "-Q", ",".join(queues),
                "-c", "1", "--pool=solo",
                f"--hostname={hostname}@%h",
                f"--loglevel={loglevel}",
                f"--max-tasks-per-child={max_tasks_per_child}",
            ),
            env=env,
            card=card,
        )

    if not cards:
        # A machine with no GPU (development): one worker takes Auto jobs, which
        # place on the CPU as they always have. Named-card jobs are refused at submit.
        return [spec("gpu-none", [AUTO_QUEUE], "gpu-none", {}, None)]

    specs = []
    for card in cards:
        # The index makes the hostname readable in logs; the UUID prefix keeps it
        # unique and recognisable across a renumbering.
        short = card.uuid.removeprefix("GPU-")[:8]
        name = f"gpu-{card.index}-{short}"
        specs.append(spec(name, [queue_for(card.uuid), AUTO_QUEUE], name, {WORKER_GPU_ENV: card.uuid}, card.uuid))
    return specs


@dataclass(frozen=True)
class RestartPolicy:
    """When an exited worker is started again, and when it is given up on."""

    #: Delay before the first restart; doubled for each further exit in the window.
    base_backoff_s: float = 5.0
    max_backoff_s: float = 120.0
    #: Exits older than this no longer count.
    window_s: float = 600.0
    #: Restarts allowed in the window; the next exit is a crash loop.
    max_restarts_in_window: int = 5

    def backoff_for(self, exits_in_window: int) -> float:
        return min(self.base_backoff_s * 2 ** max(exits_in_window - 1, 0), self.max_backoff_s)


@dataclass
class _Worker:
    spec: WorkerSpec
    process: Any = None
    state: str = RUNNING
    exits: deque = field(default_factory=deque)
    restarts: int = 0
    last_exit_code: Optional[int] = None
    restart_at: Optional[float] = None
    signalled: bool = False

    def status(self) -> dict:
        return {
            "name": self.spec.name,
            "card": self.spec.card,
            "state": self.state,
            "pid": getattr(self.process, "pid", None) if self.state == RUNNING else None,
            "restarts": self.restarts,
            "exits_in_window": len(self.exits),
            "last_exit_code": self.last_exit_code,
        }


def reap_orphans(known_pids: Iterable[int]) -> list:
    """Reap zombie children this process adopted, leaving its own workers alone.

    As a container's PID 1, the supervisor inherits every process orphaned in the
    container (a worker killed while running `nvidia-smi`, say), and nothing
    else will ever wait for them. ``WNOWAIT`` peeks first: a waitable child that
    is one of ``known_pids`` is left for ``Popen.poll``, which would otherwise read
    ECHILD and report exit code 0 for a worker that crashed.

    Returns the PIDs reaped.
    """
    if not hasattr(os, "waitid"):
        return []
    known = set(known_pids)
    reaped = []
    while True:
        try:
            info = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return reaped
        if info is None or info.si_pid in known:
            return reaped
        try:
            os.waitpid(info.si_pid, os.WNOHANG)
        except ChildProcessError:
            return reaped
        reaped.append(info.si_pid)


def supervise(
    specs: Sequence[WorkerSpec],
    *,
    popen: Callable = subprocess.Popen,
    poll_interval: float = 1.0,
    install_signal_handlers: bool = True,
    on_tick: Optional[Callable[[], None]] = None,
    on_status: Optional[Callable[[list], None]] = None,
    policy: RestartPolicy = RestartPolicy(),
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    drain_timeout_s: float = DRAIN_TIMEOUT_S,
    reap: Optional[Callable[[Iterable[int]], list]] = None,
) -> int:
    """Run every worker, restarting each alone when it exits, until stopped.

    ``on_tick`` runs on every poll (it throttles itself): the parked-job release
    and stranded-queue rescue of ``services/gpu_worker_queues.maintenance_tick``.
    ``on_status`` gets every worker's :meth:`_Worker.status` on every poll
    (:class:`CardAvailability` leases the cards of workers that are not running).
    A callback that raises is logged and retried on the next poll; it never
    stops the workers.

    Returns 0 after a stop signal, or :data:`EXIT_ALL_CRASH_LOOPING` when every
    worker is crash-looping.
    """
    workers = [_Worker(spec) for spec in specs]
    stopping: dict = {"signal": None, "at": None}

    def start(worker: _Worker) -> None:
        env = {**os.environ, **worker.spec.env}
        logger.info("Starting GPU worker %s: %s", worker.spec.name, " ".join(worker.spec.argv))
        worker.process = popen(list(worker.spec.argv), env=env)
        worker.state = RUNNING
        worker.signalled = False
        worker.restart_at = None

    def terminate_once(worker: _Worker) -> None:
        if worker.state == RUNNING and not worker.signalled and worker.process.poll() is None:
            worker.process.send_signal(signal.SIGTERM)
            worker.signalled = True

    def forward(signum, _frame):
        if stopping["signal"] is None:
            stopping["signal"] = signum
            stopping["at"] = clock()
        for worker in workers:
            terminate_once(worker)

    def exited(worker: _Worker, code: int, now: float) -> None:
        worker.last_exit_code = code
        worker.process = None
        while worker.exits and now - worker.exits[0] > policy.window_s:
            worker.exits.popleft()
        worker.exits.append(now)
        if len(worker.exits) > policy.max_restarts_in_window:
            worker.state = CRASH_LOOPING
            logger.error(
                "GPU worker %s exited with %s, its %d exit in %.0f s: it is CRASH-LOOPING and will not be "
                "restarted. Its card is marked unavailable to other jobs; the other workers keep running. "
                "Fix the cause and restart the pod.",
                worker.spec.name, code, len(worker.exits), policy.window_s,
            )
            return
        delay = policy.backoff_for(len(worker.exits))
        worker.state = RESTARTING
        worker.restart_at = now + delay
        logger.error(
            "GPU worker %s exited with %s; restarting it alone in %.0f s (exit %d of %d allowed in %.0f s). "
            "The other workers keep running.",
            worker.spec.name, code, delay, len(worker.exits), policy.max_restarts_in_window + 1, policy.window_s,
        )

    def callbacks() -> None:
        if reap is not None:
            try:
                reap([w.process.pid for w in workers if w.process is not None])
            except Exception:  # noqa: BLE001 - reaping must not stop the workers
                logger.exception("Could not reap orphaned processes")
        if on_status is not None:
            try:
                on_status([w.status() for w in workers])
            except Exception:  # noqa: BLE001 - status must not stop the workers
                logger.exception("GPU supervisor status update failed; retrying on the next poll")
        if on_tick is not None:
            try:
                on_tick()
            except Exception:  # noqa: BLE001 - maintenance must not stop the workers
                logger.exception("GPU supervisor maintenance failed; retrying on the next poll")

    for worker in workers:
        start(worker)

    if install_signal_handlers:
        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)

    while stopping["signal"] is None:
        now = clock()
        for worker in workers:
            if worker.state == RUNNING:
                code = worker.process.poll()
                if code is not None:
                    exited(worker, code, now)
            elif worker.state == RESTARTING and now >= worker.restart_at and stopping["signal"] is None:
                worker.restarts += 1
                start(worker)
        if workers and all(worker.state == CRASH_LOOPING for worker in workers):
            logger.critical("Every GPU worker is crash-looping; exiting so the container restarts")
            callbacks()
            return EXIT_ALL_CRASH_LOOPING
        callbacks()
        sleep(poll_interval)

    return _drain(workers, terminate_once, clock=clock, sleep=sleep, poll_interval=poll_interval,
                  drain_timeout_s=drain_timeout_s, stopped_at=stopping["at"])


def _drain(workers, terminate_once, *, clock, sleep, poll_interval, drain_timeout_s, stopped_at=None) -> int:
    """Warm-stop every running worker; kill only what outlives the drain.

    THE DRAIN IS TIMED FROM THE STOP SIGNAL (review round 2), which is when every
    worker got its SIGTERM and when the kubelet's grace period began. Timed from the
    moment the supervisor's loop noticed the stop, a maintenance tick blocked on
    Redis or the database pushed every kill past `terminationGracePeriodSeconds`.
    And every worker still running is KILLED before any is waited for: waiting
    10 s on each in turn (a process stuck in the driver can outlive SIGKILL for a
    while) delayed the next worker's kill past the grace as well.
    """
    for worker in workers:
        if worker.state != RUNNING:
            worker.state = STOPPED
        else:
            # A worker started while the signal handler ran was not signalled.
            terminate_once(worker)
    deadline = (clock() if stopped_at is None else stopped_at) + drain_timeout_s
    while True:
        running = [w for w in workers if w.state == RUNNING]
        for worker in running:
            code = worker.process.poll()
            if code is not None:
                worker.last_exit_code = code
                worker.state = STOPPED
        running = [w for w in workers if w.state == RUNNING]
        if not running or clock() >= deadline:
            break
        sleep(min(poll_interval, 0.5))
    for worker in running:
        logger.error(
            "GPU worker %s was still running its task %.0f s after the stop signal; killing it",
            worker.spec.name, drain_timeout_s,
        )
        worker.process.kill()
    reap_by = clock() + KILL_WAIT_S
    for worker in running:
        try:
            worker.process.wait(timeout=max(reap_by - clock(), 0.0))
        except subprocess.TimeoutExpired:
            logger.error("GPU worker %s did not die after SIGKILL", worker.spec.name)
        worker.state = STOPPED
    return 0


class CardAvailability:
    """Holds a GPU lease on each card whose worker is not running, so no job is sent there.

    With no worker, a card's `gpu.<uuid>` queue has no consumer, and its card has
    the MOST free memory on the node — exactly what an Auto job's claim prefers
    (``gpu_claim.decide_claim`` sends a job to the idle card with the most free
    memory). Without a lease every Auto job that fits there would be handed to a
    queue nothing reads, for the length of a backoff or, crash-looping, for good.
    A lease is what every claim already obeys.

    The holder is deterministic per card (:func:`holder_for`), so a restarted
    supervisor clears whatever a previous one left (:meth:`reset`) before its
    workers start. Idempotent and retried: a database error leaves the next
    update to try again, and a lease the supervisor stops renewing expires at its
    TTL.
    """

    HOLDER_PREFIX = "gpu-worker-down"

    def __init__(
        self,
        session: Callable[[], Any],
        *,
        clock: Callable[[], float] = time.monotonic,
        renew_every_s: float = 60.0,
        retry_every_s: float = 10.0,
    ) -> None:
        self._session = session
        self._clock = clock
        self._renew_every_s = renew_every_s
        self._retry_every_s = retry_every_s
        self._held: dict = {}      # card -> when last taken or renewed
        self._tried: dict = {}     # card -> when last attempted without success

    @classmethod
    def holder_for(cls, card: str) -> str:
        from ..services.gpu_placement import normalise_uuid

        return f"{cls.HOLDER_PREFIX}:{normalise_uuid(card)}"

    def reset(self, cards: Iterable[str]) -> None:
        """Release every card's down-marker a previous supervisor left. Never raises."""
        from ..services import gpu_leases

        for card in cards:
            try:
                with self._session() as db:
                    gpu_leases.release(db, self.holder_for(card))
            except Exception:  # noqa: BLE001 - it expires at its TTL anyway
                logger.exception("Could not clear the down-marker lease of %s", card)
        self._held.clear()

    def update(self, statuses: list) -> None:
        from ..services import gpu_leases

        now = self._clock()
        for status in statuses:
            card = status.get("card")
            if not card:
                continue
            if status["state"] == RUNNING:
                if card in self._held:
                    with self._session() as db:
                        gpu_leases.release(db, self.holder_for(card))
                    del self._held[card]
                    logger.info("GPU worker for %s is running again; its card is available", card)
                self._tried.pop(card, None)
                continue
            last = self._held.get(card)
            if last is not None and now - last < self._renew_every_s:
                continue
            if last is None and now - self._tried.get(card, float("-inf")) < self._retry_every_s:
                continue
            with self._session() as db:
                taken = gpu_leases.acquire(db, [card], self.holder_for(card))
            if taken:
                if last is None:
                    logger.warning(
                        "GPU worker for %s is %s; its card is leased as unavailable until it runs again",
                        card, status["state"],
                    )
                self._held[card] = now
                self._tried.pop(card, None)
            else:
                # Its crashed job's lease is still live: the card is unavailable anyway.
                self._held.pop(card, None)
                self._tried[card] = now


def main() -> int:  # pragma: no cover - process entry point, exercised on the node
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from ..core.database import get_sync_db
    from ..services.gpu_placement import list_cards
    from ..services.gpu_worker_queues import maintenance_tick

    cards = list_cards()
    specs = worker_specs(
        cards,
        # The names docker-entrypoint.sh already uses for the single worker.
        max_tasks_per_child=int(os.environ.get("CELERY_MAX_TASKS", "100")),
        loglevel=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )
    logger.info("GPU supervisor: %d card(s), %d worker(s)", len(cards), len(specs))
    # BEFORE the workers start: jobs queued for a card this restart no longer
    # has are moved where they will be refused with a message, and jobs parked
    # while the pod was down are republished.
    tick = maintenance_tick([card.uuid for card in cards])
    try:
        tick()
    except Exception:  # noqa: BLE001 - the loop retries it
        logger.exception("GPU supervisor start-up maintenance failed; retrying while the workers run")
    availability = CardAvailability(lambda: get_sync_db())
    availability.reset(card.uuid for card in cards)
    return supervise(
        specs,
        on_tick=tick,
        on_status=availability.update,
        reap=reap_orphans if os.getpid() == 1 else None,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
