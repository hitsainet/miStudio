"""
GPU Memory Watchdog Task

This Celery Beat task runs periodically to detect and handle stuck GPU processes.
It monitors GPU memory usage and can optionally kill processes that appear stuck.

Prevention strategy for zombie processes holding GPU memory:
1. Detect processes using GPU memory for too long
2. Log warnings for investigation
3. Optionally kill stuck processes (configurable)
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from celery import shared_task

logger = logging.getLogger(__name__)


@dataclass
class GPUProcessInfo:
    """Information about a process using GPU memory."""
    pid: int
    name: str
    gpu_id: int
    memory_mib: int
    first_seen: datetime
    duration_seconds: float


# Track when we first saw each process using GPU
_process_first_seen: Dict[int, datetime] = {}

# Threshold for warning about long-running processes (5 minutes)
LONG_RUNNING_THRESHOLD_SECONDS = 300

# Threshold for critical warning (30 minutes)
CRITICAL_THRESHOLD_SECONDS = 1800


def get_gpu_processes() -> List[GPUProcessInfo]:
    """
    Get list of processes currently using GPU memory.

    Returns:
        List of GPUProcessInfo objects
    """
    processes = []

    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            logger.warning(f"[GPU Watchdog] nvidia-smi failed: {result.stderr}")
            return []

        # Get GPU index mapping
        uuid_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_uuid,index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        uuid_to_idx = {}
        if uuid_result.returncode == 0:
            for line in uuid_result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        uuid_to_idx[parts[0].strip()] = int(parts[1].strip())

        now = datetime.utcnow()

        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                try:
                    pid = int(parts[0])
                    name = parts[1]
                    gpu_uuid = parts[2]
                    memory_mib = int(parts[3])

                    gpu_id = uuid_to_idx.get(gpu_uuid, 0)

                    # Track first seen time
                    if pid not in _process_first_seen:
                        _process_first_seen[pid] = now

                    first_seen = _process_first_seen[pid]
                    duration = (now - first_seen).total_seconds()

                    processes.append(GPUProcessInfo(
                        pid=pid,
                        name=name,
                        gpu_id=gpu_id,
                        memory_mib=memory_mib,
                        first_seen=first_seen,
                        duration_seconds=duration,
                    ))
                except (ValueError, IndexError) as e:
                    logger.warning(f"[GPU Watchdog] Failed to parse line: {line}, error: {e}")

    except Exception as e:
        logger.error(f"[GPU Watchdog] Error getting GPU processes: {e}")

    return processes


def is_process_zombie(pid: int) -> bool:
    """Check if a process is a zombie."""
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("State:"):
                    return "Z" in line  # Z = zombie
    except:
        pass
    return False


def get_process_cpu_percent(pid: int) -> Optional[float]:
    """Get CPU usage percentage for a process."""
    try:
        import subprocess
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None


@shared_task(
    name="gpu_watchdog",
    bind=True,
    max_retries=0,
    soft_time_limit=30,
    time_limit=60,
)
def gpu_watchdog_task(self):
    """
    Monitor GPU memory usage and detect stuck processes.

    This task runs every minute via Celery Beat and:
    1. Gets list of processes using GPU memory
    2. Tracks how long each process has been using GPU
    3. Logs warnings for long-running processes
    4. Logs critical alerts for very long-running processes
    5. Detects zombie processes holding GPU memory
    """
    try:
        processes = get_gpu_processes()

        if not processes:
            # Clean up tracking for processes that are no longer running
            current_pids = set()
            pids_to_remove = [
                pid for pid in _process_first_seen.keys()
                if pid not in current_pids
            ]
            for pid in pids_to_remove:
                del _process_first_seen[pid]
            return {"status": "ok", "gpu_processes": 0}

        current_pids = {p.pid for p in processes}

        # Clean up tracking for processes no longer using GPU
        pids_to_remove = [
            pid for pid in _process_first_seen.keys()
            if pid not in current_pids
        ]
        for pid in pids_to_remove:
            del _process_first_seen[pid]

        warnings = []
        critical = []
        zombies = []

        for proc in processes:
            # Check for zombies
            if is_process_zombie(proc.pid):
                zombies.append(proc)
                logger.critical(
                    f"[GPU Watchdog] ZOMBIE PROCESS detected! "
                    f"PID={proc.pid}, GPU={proc.gpu_id}, Memory={proc.memory_mib}MiB, "
                    f"Duration={proc.duration_seconds:.0f}s"
                )
                continue

            # Check for long-running processes
            if proc.duration_seconds >= CRITICAL_THRESHOLD_SECONDS:
                cpu_percent = get_process_cpu_percent(proc.pid)
                critical.append(proc)
                logger.critical(
                    f"[GPU Watchdog] CRITICAL: Process running >{CRITICAL_THRESHOLD_SECONDS}s! "
                    f"PID={proc.pid}, Name={proc.name}, GPU={proc.gpu_id}, "
                    f"Memory={proc.memory_mib}MiB, Duration={proc.duration_seconds:.0f}s, "
                    f"CPU={cpu_percent}%"
                )
            elif proc.duration_seconds >= LONG_RUNNING_THRESHOLD_SECONDS:
                cpu_percent = get_process_cpu_percent(proc.pid)
                warnings.append(proc)
                logger.warning(
                    f"[GPU Watchdog] Long-running GPU process: "
                    f"PID={proc.pid}, Name={proc.name}, GPU={proc.gpu_id}, "
                    f"Memory={proc.memory_mib}MiB, Duration={proc.duration_seconds:.0f}s, "
                    f"CPU={cpu_percent}%"
                )

        return {
            "status": "ok",
            "gpu_processes": len(processes),
            "warnings": len(warnings),
            "critical": len(critical),
            "zombies": len(zombies),
            "total_memory_mib": sum(p.memory_mib for p in processes),
        }

    except Exception as e:
        logger.error(f"[GPU Watchdog] Task error: {e}")
        return {"status": "error", "error": str(e)}
