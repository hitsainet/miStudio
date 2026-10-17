"""The docker-compose deployment keeps the same queue split as k8s (MIS-E2E-147 item 4).

`test_worker_queue_coverage.py` pins the split in `k8s/base/backend.yaml`. The compose
file — the documented development startup (`docker compose up -d`) — kept ONE worker on
every queue, so a multi-hour NLP pass or a janitor blocked every extraction and training
behind it there, exactly the 2026-07-26 incident the k8s split fixed. FAILS CLOSED: a
missing file or an empty worker list is a failure, never a skip.

"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
COMPOSE = REPO / "docker-compose.yml"
DYNAMIC_QUEUES = {"steering"}


def _workers(path: Path):
    assert path.exists(), f"{path} is gone; point this guard at its replacement"
    doc = yaml.safe_load(path.read_text())
    workers = {}
    for name, svc in (doc.get("services") or {}).items():
        env = svc.get("environment") or {}
        if isinstance(env, list):
            env = dict(item.split("=", 1) for item in env if "=" in item)
        if env.get("SERVICE_TYPE") != "celery-worker":
            continue
        reservations = (((svc.get("deploy") or {}).get("resources") or {}).get("reservations") or {})
        workers[name] = {
            "queues": [q for q in str(env.get("CELERY_QUEUES") or "").split(",") if q],
            "name": env.get("CELERY_WORKER_NAME"),
            "gpu": bool(reservations.get("devices")),
        }
    return workers


@pytest.fixture(scope="module")
def workers():
    found = _workers(COMPOSE)
    assert found, "no celery-worker service found in docker-compose.yml"
    return found


def test_every_routed_queue_has_a_consumer(workers):
    from tests.unit.test_worker_queue_coverage import _routed_queues

    consumed = set().union(*(set(w["queues"]) for w in workers.values()))
    orphaned = _routed_queues() - consumed - DYNAMIC_QUEUES
    assert not orphaned, f"compose: tasks routed to {sorted(orphaned)} have no consumer"


def test_every_worker_declares_its_queues(workers):
    for name, w in workers.items():
        assert w["queues"], f"{name} inherits the entrypoint's all-queues default"


def test_low_priority_is_not_on_the_gpu_worker(workers):
    gpu = [n for n, w in workers.items() if "extraction" in w["queues"]]
    assert gpu, "no compose worker consumes the extraction queue"
    for name in gpu:
        assert "low_priority" not in workers[name]["queues"], (
            f"compose {name} consumes extraction AND low_priority on one solo slot"
        )


def test_low_priority_has_its_own_consumer_without_a_gpu(workers):
    cpu = [n for n, w in workers.items() if "low_priority" in w["queues"]]
    assert cpu, "nothing in compose consumes low_priority (janitors, NLP, prune)"
    for name in cpu:
        assert not workers[name]["gpu"], f"{name} reserves a GPU it never uses"


def test_worker_names_are_unique(workers):
    names = [w["name"] for w in workers.values()]
    assert all(names), "a compose worker has no CELERY_WORKER_NAME"
    assert len(set(names)) == len(names), f"duplicate compose worker names: {names}"
