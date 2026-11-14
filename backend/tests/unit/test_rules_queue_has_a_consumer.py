"""The `rules` queue has a consumer in every manifest (Feature 040 S3; FR-37, 040-T31; FTASKS 21.7).

Reads the live Celery routes and both manifests as YAML (never a regex over source): every queue a registered
task routes to has a container that consumes it, in `k8s/base/backend.yaml` (what ArgoCD deploys) AND in
`docker-compose.yml`; and `rules` has exactly one consumer, its own `celery-worker-rules` (ENH-006 D52, replacing
FTDD §6.6's `celery-worker-cpu`, whose one slot the janitors need), with the entrypoint's default (single-worker)
list naming it too. Controls: drop `rules` from the rules container, or re-add it to celery-worker-cpu → red.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]


def _routed_queues() -> set:
    from src.core.celery_app import celery_app

    celery_app.loader.import_default_modules()
    return {route["queue"] for route in (celery_app.conf.task_routes or {}).values() if isinstance(route, dict)
            and route.get("queue")}


def _k8s_queues() -> dict:
    out = {}
    for doc in yaml.safe_load_all((REPO / "k8s" / "base" / "backend.yaml").read_text()):
        for container in (((doc or {}).get("spec") or {}).get("template") or {}).get("spec", {}).get("containers", []):
            env = {e["name"]: e.get("value") for e in container.get("env", []) if "name" in e}
            if env.get("CELERY_QUEUES"):
                out[container["name"]] = set(env["CELERY_QUEUES"].split(","))
    return out


def _compose_queues() -> dict:
    services = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]
    out = {}
    for name, service in services.items():
        env = service.get("environment") or {}
        if isinstance(env, list):
            env = dict(item.split("=", 1) for item in env if "=" in item)
        if env.get("CELERY_QUEUES"):
            out[name] = set(str(env["CELERY_QUEUES"]).split(","))
    return out


def test_the_rules_task_is_routed_to_rules():
    assert "rules" in _routed_queues()


def test_every_queue_the_rules_feature_routes_to_has_a_consumer():
    """The full routed-queue coverage of the k8s manifest is `test_worker_queue_coverage.py`'s; this pins the
    queues this feature routes to in BOTH manifests."""
    k8s = set().union(*_k8s_queues().values())
    compose = set().union(*_compose_queues().values())
    for queue in ("rules", "low_priority"):  # the task, and the reaper
        assert queue in k8s and queue in compose, queue


def test_rules_has_exactly_one_consumer_and_it_is_its_own_container():
    """ENH-006 D52 (2026-10-24; Review R1-F4): `rules` runs in celery-worker-rules ONLY. On celery-worker-cpu (one
    solo slot) a rules evaluation held every low_priority janitor for its whole run. Controls: drop `rules` from
    the new container, or re-add it to celery-worker-cpu → red."""
    for queues in (_k8s_queues(), _compose_queues()):
        consumers = sorted(name for name, q in queues.items() if "rules" in q)
        assert consumers == ["celery-worker-rules"], consumers
        assert queues["celery-worker-rules"] == {"rules"}
        assert "rules" not in queues["celery-worker-cpu"] and "low_priority" in queues["celery-worker-cpu"]


def _k8s_container(name: str) -> dict:
    for doc in yaml.safe_load_all((REPO / "k8s" / "base" / "backend.yaml").read_text()):
        for container in (((doc or {}).get("spec") or {}).get("template") or {}).get("spec", {}).get("containers", []):
            if container["name"] == name:
                return container
    raise AssertionError(f"no {name} container in k8s/base/backend.yaml")


def test_the_rules_worker_is_cpu_only_and_bounded():
    container = _k8s_container("celery-worker-rules")
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert (env["CELERY_WORKER_NAME"], env["CELERY_POOL"], env["CELERY_CONCURRENCY"]) == ("rules", "solo", "1")
    assert env["GPU_WORKER_MODE"] == "per_card"           # never the legacy single mode (manifest guard)
    assert not any(k.startswith(("NVIDIA", "CUDA")) for k in env), "the rules worker sees no card"
    resources = container.get("resources") or {}
    assert "nvidia.com/gpu" not in str(resources)
    assert resources.get("requests", {}).get("memory") and resources.get("limits", {}).get("memory")
    assert {m["mountPath"] for m in container["volumeMounts"]} == {"/data"}


def test_the_entrypoint_default_list_names_rules():
    text = (REPO / "backend" / "docker-entrypoint.sh").read_text()
    default = re.search(r"-Q \$\{CELERY_QUEUES:-([^}]+)\}", text)
    assert default is not None, "the entrypoint's -Q default moved; this check would assert nothing"
    assert "rules" in default.group(1).split(",")
