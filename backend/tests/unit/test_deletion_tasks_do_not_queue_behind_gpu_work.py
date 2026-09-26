"""Deleting things must not wait behind GPU work.

2026-09-12, from production. `DELETE /features/extractions/{id}` queued
`delete_extraction` on the `extraction` queue, which only the single
`--pool=solo -c 1` GPU worker consumes. Four activation extractions were queued
there (~3 hours). The endpoint returned 202, nothing visibly happened, the
operator retried, and seven delete tasks for four extractions sat behind the GPU
backlog while the CPU worker (`low_priority`) was idle.

Deletion is database and filesystem work. It belongs with the other cleanup
tasks on `low_priority`.

Asserted through Celery's own router rather than by reading the config dict:
`task_routes` mixes exact names and globs (`src.workers.extraction_tasks.*` →
`extraction`), and a routing test that reads the dict cannot tell which one wins.
"""

import pytest

from src.core.celery_app import celery_app


def _queue_for(task_name: str) -> str:
    route = celery_app.amqp.router.route({}, task_name)
    queue = route.get("queue")
    return getattr(queue, "name", queue)


@pytest.mark.parametrize(
    "task_name",
    [
        "delete_extraction",
        "workers.model_tasks.delete_model_files",
        "src.workers.dataset_tasks.delete_dataset_files",
    ],
)
def test_deletion_tasks_route_to_the_cpu_worker(task_name):
    assert _queue_for(task_name) == "low_priority", (
        f"{task_name} routes to {_queue_for(task_name)!r}; the only worker on the "
        f"GPU queues is solo, so a deletion would wait behind every extraction"
    )


def test_gpu_extraction_work_still_routes_to_the_gpu_worker():
    """NEGATIVE CONTROL: the router really distinguishes the two queues."""
    assert _queue_for("src.workers.extraction_tasks.extract_features_from_sae") == "extraction"


def test_the_cpu_worker_actually_consumes_low_priority():
    """A route to a queue nothing consumes is worse than a slow one."""
    from pathlib import Path

    import yaml

    manifest = Path(__file__).resolve().parents[3] / "k8s" / "base" / "backend.yaml"
    docs = [d for d in yaml.safe_load_all(manifest.read_text()) if d]
    queues = {}
    for doc in docs:
        for container in (doc.get("spec", {}).get("template", {}).get("spec", {}).get("containers") or []):
            for env in container.get("env") or []:
                if env.get("name") == "CELERY_QUEUES":
                    queues[container["name"]] = env.get("value", "").split(",")
    assert "low_priority" in queues.get("celery-worker-cpu", []), queues
    assert "low_priority" not in queues.get("celery-worker", []), (
        "the GPU worker consumes low_priority too, so this routing would not isolate anything"
    )


@pytest.mark.parametrize(
    "task_name",
    [
        "delete_extraction",
        "workers.model_tasks.delete_model_files",
        "src.workers.dataset_tasks.delete_dataset_files",
    ],
)
def test_deletion_is_never_sent_to_a_per_card_gpu_queue(task_name, monkeypatch):
    """Multi-GPU Phase 3: in per-card mode the dispatcher routes only GPU-queue
    tasks to gpu.<uuid>/gpu.auto, and no per-card GPU worker consumes low_priority,
    so a deletion cannot wait behind a GPU job on either path."""
    from src.core.config import settings
    from src.services.gpu_dispatch import runs_on_gpu_queues
    from src.services.gpu_placement import GpuCard
    from src.workers.gpu_supervisor import worker_specs

    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    assert runs_on_gpu_queues(celery_app.tasks[task_name]) is False
    card = GpuCard(0, "GPU-f47ba814-49a2-603f-3595-275284140251", "RTX 3080 Ti", 12_288, 11_900)
    for spec in worker_specs([card], python="py"):
        assert _queue_for(task_name) not in spec.argv[list(spec.argv).index("-Q") + 1].split(",")
