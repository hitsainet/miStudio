"""Live per-card GPU state, keyed by UUID, with who holds each card (MCP gap G9).

`/system/gpu-list` answers "which cards exist" (static facts). Before a GPU run an agent needs
"which cards are in use, how much, and by what" — which took `ssh` + `nvidia-smi` on the node.
This reads the same NVML the backend already uses, for EVERY card it can see (one or many; the
count is whatever NVML reports, never assumed), and joins three ownership facts:

* the card's miStudio GPU LEASE (`gpu_leases`) — the job that holds it, taken the moment a job's
  card is chosen, so it names the owner even before the job has allocated anything;
* the miStudio jobs that RECORDED this card (`active_jobs`);
* a compute process whose PID is a live miStudio steering worker's.

⚠ A PROCESS IS NOT A JOB. NVML reports PIDs in the host's (or the pod's) PID namespace; miStudio's
Celery worker, miLLM and miForge run in OTHER pods, so their PIDs mean nothing here and are
reported unattributed — `owner: null`, never a guess. Memory on a card with no lease and no job
belongs to something outside miStudio (miLLM's resident model, miForge, another process).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

#: Job statuses that have not started work on a card yet. Everything else that recorded the card
#: — including an IDLE steering worker, which keeps its model in VRAM — is using it.
WAITING_STATUSES = frozenset({"queued", "pending"})


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_cards(nvml: Any = None) -> tuple[list[dict[str, Any]], Optional[str]]:
    """`(cards, reason)` — every card NVML sees, with memory, utilisation and processes.

    `reason` is set (and `cards` empty) when NVML cannot be read: a CPU-only host is a normal
    state, reported in words rather than as an error. `nvml` is injectable for tests.
    """
    if nvml is None:
        try:
            import pynvml as nvml  # type: ignore[no-redef]
        except ImportError:
            return [], "NVML (pynvml) is not installed in this process, so no GPU can be read"
    try:
        nvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - no driver is a normal state
        return [], f"NVML could not start ({exc}); no GPU is visible to miStudio"
    try:
        cards = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = {"gpu_percent": int(util.gpu), "memory_percent": int(util.memory)}
            except Exception:  # noqa: BLE001 - some cards do not report it
                utilization = {"gpu_percent": None, "memory_percent": None}
            processes = []
            try:
                for proc in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    used = getattr(proc, "usedGpuMemory", None)
                    processes.append({
                        "pid": int(proc.pid),
                        "used_mib": None if used is None else round(used / MIB, 1),
                    })
                processes_reason = None
            except Exception as exc:  # noqa: BLE001
                processes, processes_reason = None, f"NVML could not list processes: {exc}"
            card = {
                "uuid": _text(nvml.nvmlDeviceGetUUID(handle)),
                "index": index,
                "name": _text(nvml.nvmlDeviceGetName(handle)),
                "memory": {
                    "total_mib": round(memory.total / MIB, 1),
                    "used_mib": round(memory.used / MIB, 1),
                    "free_mib": round(memory.free / MIB, 1),
                },
                "utilization": utilization,
                "processes": processes,
            }
            if processes_reason:
                card["processes_reason"] = processes_reason
            cards.append(card)
        return cards, None
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - best effort
            pass


async def live_leases(db: Any) -> dict[str, dict[str, Any]]:
    """`{normalised uuid: lease}` for every unexpired miStudio GPU lease."""
    from sqlalchemy import select

    from ..models.gpu_lease import GpuLease
    from .gpu_placement import normalise_uuid

    now = datetime.now(timezone.utc)
    rows = (await db.execute(select(GpuLease).where(GpuLease.expires_at > now))).scalars().all()
    out = {}
    for row in rows:
        holder = row.holder or ""
        out[normalise_uuid(row.gpu_uuid)] = {
            "holder": holder,
            "kind": holder.split(":", 1)[0] if ":" in holder else None,
            "task_id": row.task_id,
            "acquired_at": row.acquired_at.isoformat() if row.acquired_at else None,
            "heartbeat_at": row.heartbeat_at.isoformat() if row.heartbeat_at else None,
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        }
    return out


def assemble(
    cards: list[dict[str, Any]],
    leases: dict[str, dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join leases, jobs and steering-worker PIDs onto each card, and say whether it is in use."""
    from .active_jobs import STEERING_WORKER_KIND, jobs_on_card
    from .gpu_placement import normalise_uuid

    worker_pids = {
        job["pid"]: job for job in jobs if job.get("kind") == STEERING_WORKER_KIND and job.get("pid")
    }
    out = []
    for card in cards:
        key = normalise_uuid(card["uuid"])
        on_card = jobs_on_card(jobs, card["uuid"])
        processes = card.get("processes")
        if processes is not None:
            processes = [
                {**proc, "owner": (
                    {"kind": STEERING_WORKER_KIND, "id": worker_pids[proc["pid"]]["id"]}
                    if proc["pid"] in worker_pids else None
                )}
                for proc in processes
            ]
        lease = leases.get(key)
        in_use_by = []
        if lease:
            in_use_by.append(f"miStudio lease held by {lease['holder']}")
        in_use_by += [f"miStudio {job['kind']} {job['id']} ({job['status']})"
                      for job in on_card if job["status"] not in WAITING_STATUSES]
        if processes:
            unattributed = [p for p in processes if p.get("owner") is None]
            if unattributed:
                in_use_by.append(
                    f"{len(unattributed)} compute process(es) not attributable to a miStudio "
                    "job (another pod or app — see miLLM / miForge in gpu_preflight)"
                )
        out.append({
            **card,
            "processes": processes,
            "lease": lease,
            "jobs": on_card,
            "in_use": bool(in_use_by),
            "in_use_by": in_use_by,
        })
    return out


PID_NOTE = (
    "Ownership comes from miStudio's GPU lease on the card and from the jobs that recorded it. "
    "A compute process is attributed only when its PID is a miStudio steering worker's; miStudio's "
    "Celery worker, miLLM and miForge run in other pods, so their PIDs are reported with "
    "owner null rather than guessed."
)
