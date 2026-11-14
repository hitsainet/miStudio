"""The blend worker is deployed (Feature 041 T24, TI15; FTASKS 8.3): k8s and compose both run a container
consuming ONLY `blend`, prefork, concurrency 2, one BLAS thread per child, no GPU environment."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]


def _k8s_container():
    for doc in yaml.safe_load_all((REPO / "k8s" / "base" / "backend.yaml").read_text()):
        if not doc or doc.get("kind") != "Deployment":
            continue
        for container in doc["spec"]["template"]["spec"]["containers"]:
            if container["name"] == "celery-worker-blend":
                return container
    return None


def test_k8s_runs_the_blend_worker():
    container = _k8s_container()
    assert container is not None, "no celery-worker-blend container in k8s/base/backend.yaml"
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["CELERY_QUEUES"] == "blend" and env["CELERY_WORKER_NAME"] == "blend"
    assert (env["CELERY_POOL"], env["CELERY_CONCURRENCY"]) == ("prefork", "2")
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert not any(k.startswith(("NVIDIA", "CUDA")) for k in env), "the blend worker sees no card"
    assert env["GPU_WORKER_MODE"] == "per_card"           # never the legacy single mode (manifest guard)
    assert {m["mountPath"] for m in container["volumeMounts"]} == {"/data"}


def test_compose_runs_the_blend_worker():
    service = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]["celery-worker-blend"]
    env = service["environment"]
    assert env["CELERY_QUEUES"] == "blend" and env["CELERY_POOL"] == "prefork" and str(env["CELERY_CONCURRENCY"]) == "2"
    assert str(env["OMP_NUM_THREADS"]) == "1"


def test_the_task_routes_to_blend():
    from src.core.celery_app import celery_app

    assert celery_app.conf.task_routes["src.workers.probe_blend_tasks.run_blend_evaluation"] == {"queue": "blend"}


def test_a_worker_registers_the_041_tasks():
    """A worker registers only the modules in `celery_app`'s include list (the API process imports more), so
    each 041 task is looked up in a fresh interpreter's registry — dropping `probe_blend_tasks` or
    `cleanup_stuck_blend_reports` from the list turns this red (S2 control S2-39 survived without it)."""
    from tests.unit.test_training_evaluation_task_registered_in_worker import _worker_registry

    registry = _worker_registry()
    assert {"src.workers.probe_blend_tasks.run_blend_evaluation", "cleanup_stuck_blend_reports",
            "src.workers.probe_monitor_tasks.score_probe_family",
            "src.workers.probe_monitor_tasks.publish_probe_family"} <= registry
