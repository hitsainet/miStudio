"""The container runs the GPU supervisor, and the manifest runs that container (Phase 3).

``docker-entrypoint.sh`` is EXECUTED here — with fake ``su``/``setpriv`` that record
what they were asked to run — rather than scraped: a text scrape matches the
comment describing the command and passes for the wrong reason. The manifest is
read from ``k8s/base/backend.yaml`` itself.

THE DRAIN (review round 1). ``su`` stays as a parent, forwards SIGTERM and SIGKILLs
its child two seconds later — measured in the backend image (util-linux 2.37.2:
"Session terminated, killing shell... ...killed", su exit 143 two seconds after
TERM, child gone). Every worker's drain was two seconds, so an idle worker could
not close its broker connection and an acks_late job stayed unacked for the
12-hour visibility timeout. The worker service types now exec through
``setpriv``; :class:`TestTheWorkerReceivesTheStopItself` runs the real entrypoint
and requires the service to BE the entrypoint's process (same PID, no parent in
between) and to finish a three-second drain after SIGTERM. In the image the
environment under setpriv matched su's except MAIL and su's reset PATH.

MUTATION CONTROLS (2026-09-14; each alone, restored byte-identically, sha256 checked,
`git diff` clean) — all red:
  S1 the entrypoint's celery-gpu-workers case execs a plain celery worker -> runs the supervisor
  S2 GPU_WORKER_MODE removed from the general worker                      -> every process in per-card mode
  S3 NVIDIA_VISIBLE_DEVICES removed from celery-gpu-workers               -> the container sees every card
Review round 1:
  E1 celery-gpu-workers launched with `su -s /bin/bash mistudio -c "python -m …"` again
        -> test_the_service_is_the_entrypoints_process_and_drains[celery-gpu-workers]
  E2 the celery-worker case launched through su again
        -> test_the_service_is_the_entrypoints_process_and_drains[celery-worker]
  E3 exec_as_mistudio drops `--reuid=mistudio` (the worker would run as root)
        -> test_workers_run_as_mistudio_with_its_home
  E4 terminationGracePeriodSeconds lowered to 20 (below the supervisor's drain)
        -> test_the_pod_grace_outlasts_the_supervisors_drain
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from src.services.gpu_claim import AUTO_QUEUE, queue_for
from src.services.gpu_dispatch import gpu_queue_for
from src.services.gpu_placement import GpuCard
from src.workers import gpu_supervisor

BACKEND = Path(__file__).resolve().parents[2]
ENTRYPOINT = BACKEND / "docker-entrypoint.sh"
MANIFEST = BACKEND.parent / "k8s" / "base" / "backend.yaml"
TI = GpuCard(0, "GPU-f47ba814-49a2-603f-3595-275284140251", "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_900)
RTX = GpuCard(1, "GPU-247aa582-0d1b-e161-8156-983ed1fefc57", "NVIDIA GeForce RTX 3090", 24_576, 21_500)


def _fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def _environment(service_type: str, tmp_path: Path, bin_dir: Path) -> dict:
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SERVICE_TYPE": service_type,
        "DATA_DIR": str(tmp_path / "data"),
        "HOME": str(tmp_path),
    }


def _start(service_type: str, tmp_path: Path) -> str:
    """Run the real entrypoint with `su` and `setpriv` replaced; return the command it would exec."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorded = tmp_path / "launch-args"
    _fake(bin_dir, "su", f'echo "su $*" > "{recorded}"\n')
    _fake(bin_dir, "setpriv", f'echo "setpriv $*" > "{recorded}"\n')
    result = subprocess.run(["bash", str(ENTRYPOINT)], env=_environment(service_type, tmp_path, bin_dir),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return recorded.read_text()


def _containers():
    assert MANIFEST.exists(), f"manifest not found at {MANIFEST}"
    for doc in yaml.safe_load_all(MANIFEST.read_text()):
        if doc and doc.get("kind") == "Deployment":
            for container in doc["spec"]["template"]["spec"]["containers"]:
                yield container["name"], {e["name"]: e.get("value") for e in container.get("env", [])}


class TestTheEntrypoint:
    def test_the_gpu_workers_service_runs_the_supervisor(self, tmp_path):
        command = _start("celery-gpu-workers", tmp_path)
        assert "python -m src.workers.gpu_supervisor" in command

    def test_the_general_worker_is_still_one_celery_worker(self, tmp_path):
        """Negative control: the new service type did not replace the old one."""
        command = _start("celery-worker", tmp_path)
        assert "celery -A src.core.celery_app worker" in command
        assert "gpu_supervisor" not in command

    @pytest.mark.parametrize("service_type", ["celery-worker", "celery-gpu-workers"])
    def test_workers_run_as_mistudio_with_its_home(self, service_type, tmp_path):
        command = _start(service_type, tmp_path)
        assert command.startswith("setpriv --reuid=mistudio --regid=mistudio --init-groups env HOME=")
        assert " USER=mistudio LOGNAME=mistudio " in command


class TestTheWorkerReceivesTheStopItself:
    @pytest.mark.parametrize("service_type, program", [("celery-worker", "celery"), ("celery-gpu-workers", "python")])
    def test_the_service_is_the_entrypoints_process_and_drains(self, service_type, program, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        out = tmp_path / "service.out"
        # setpriv drops its own options and execs the rest, as the real one does.
        _fake(bin_dir, "setpriv", 'while [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"\n')
        _fake(bin_dir, "su", f'echo "su $*" >> "{out}"\n')
        _fake(bin_dir, program, (
            f'trap \'echo term >> "{out}"; sleep 3; echo drained >> "{out}"; exit 0\' TERM\n'
            f'echo "started $$" >> "{out}"\n'
            "while true; do sleep 0.1; done\n"
        ))
        entrypoint = subprocess.Popen(["bash", str(ENTRYPOINT)], env=_environment(service_type, tmp_path, bin_dir),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 30
            while "started" not in (out.read_text() if out.exists() else ""):
                if entrypoint.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(f"the service never started: {out.read_text() if out.exists() else ''!r} "
                                f"{entrypoint.communicate()[0]}")
                time.sleep(0.05)
            started_pid = int(out.read_text().split("started ", 1)[1].split()[0])
            assert started_pid == entrypoint.pid, "a parent process sits between the container's signal and the worker"

            stopped = time.monotonic()
            entrypoint.send_signal(signal.SIGTERM)
            code = entrypoint.wait(timeout=30)
        finally:
            if entrypoint.poll() is None:
                entrypoint.kill()
        assert out.read_text().splitlines()[1:] == ["term", "drained"]
        assert code == 0 and time.monotonic() - stopped >= 3, "the worker was killed while it drained"


class TestTheManifest:
    def test_exactly_one_container_runs_the_gpu_workers_and_sees_every_card(self):
        gpu = [env for name, env in _containers() if env.get("SERVICE_TYPE") == "celery-gpu-workers"]
        assert len(gpu) == 1
        env = gpu[0]
        assert env.get("NVIDIA_VISIBLE_DEVICES") == "all"
        assert env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
        assert env.get("GPU_WORKER_MODE") == "per_card"
        assert "CELERY_QUEUES" not in env, "its queues come from the inventory, not a fixed list"

    def test_every_process_that_dispatches_or_runs_gpu_jobs_is_in_per_card_mode(self):
        """A worker left in single mode would run a GPU job it took from a legacy queue with no lease."""
        modes = {name: env.get("GPU_WORKER_MODE") for name, env in _containers()
                 if env.get("SERVICE_TYPE") in {"api", "celery-worker", "celery-gpu-workers"}}
        assert modes and set(modes.values()) == {"per_card"}, modes

    def test_the_queues_dispatch_uses_are_the_queues_the_supervisor_consumes(self):
        """The routing decision and the consumers agree, card by card — from the two functions themselves."""
        specs = gpu_supervisor.worker_specs([TI, RTX], python="py")
        consumed = [set(spec.argv[list(spec.argv).index("-Q") + 1].split(",")) for spec in specs]
        for card, queues in zip([TI, RTX], consumed):
            assert gpu_queue_for(card.uuid) in queues
            assert gpu_queue_for("auto") in queues
        assert all(queues <= {queue_for(TI.uuid), queue_for(RTX.uuid), AUTO_QUEUE} for queues in consumed), (
            "a per-card GPU worker consumes a non-GPU queue, so cleanup could wait behind GPU work")

    def test_the_pod_grace_outlasts_the_supervisors_drain(self):
        """Otherwise the kubelet, not the supervisor, kills a draining worker — and nothing logs which."""
        (deployment,) = [doc for doc in yaml.safe_load_all(MANIFEST.read_text())
                         if doc and doc.get("kind") == "Deployment" and doc["metadata"]["name"] == "mistudio-backend"]
        grace = deployment["spec"]["template"]["spec"].get("terminationGracePeriodSeconds", 30)
        assert grace >= gpu_supervisor.DRAIN_TIMEOUT_S + 3
