"""Every scheduled task must be registered, discoverable, and correctly routed.

TWO THINGS PINNED HERE.

1. REACHABILITY of the NLP janitor. A cleanup task is exactly the kind of
   capability that can be fully implemented, unit-tested and documented while
   never actually running — nothing fails when it doesn't fire, it just stays
   quietly dark, which is the failure mode it exists to prevent. Per the repo
   rule, removing its wiring must go red.

2. ROUTING of every beat task. task_routes globs match the TASK NAME, not the
   module path. These tasks register SHORT names ("cleanup_stuck_extractions"),
   so `"src.workers.cleanup_stuck_extractions.*"` never matched any of them and
   they all resolved to the DEFAULT queue — `datasets`, which the GPU worker
   consumes. The beat entries pass an explicit options.queue, which is the only
   reason the scheduled runs landed on low_priority; any direct .delay() put a
   CPU janitor on the GPU worker.

   Found 2026-07-26 while adding cleanup_stuck_nlp, by RESOLVING the route
   rather than reading the config back. Reading it back is what hid it: the
   glob is present and looks right.

MUTATION CONTROLS:
  * delete the "cleanup-stuck-nlp" beat entry       -> schedule test fails
  * drop "src.workers.cleanup_stuck_nlp" from
    autodiscover_tasks                              -> discovery test fails
  * remove an exact-name route                      -> routing test fails
  * point the beat entry at a nonexistent task name -> registration test fails
"""

import pytest

from src.core.celery_app import celery_app

BEAT_ENTRY = "cleanup-stuck-nlp"
TASK_NAME = "cleanup_stuck_nlp"

# The queue the GPU worker consumes when nothing routes a task elsewhere.
DEFAULT_QUEUE = "datasets"


def _queue_name(route: dict) -> str:
    q = route.get("queue")
    return getattr(q, "name", str(q))


class TestTheNlpJanitorIsReachable:
    def test_it_is_registered_with_celery(self):
        """Registry membership, not importability."""
        assert TASK_NAME in celery_app.tasks, (
            f"{TASK_NAME} is not in the live task registry — beat would fire a "
            "name no worker can execute"
        )

    def test_it_is_scheduled(self):
        entry = celery_app.conf.beat_schedule.get(BEAT_ENTRY)
        assert entry is not None, (
            "the NLP janitor has no beat entry, so it never runs and an "
            "abandoned pass claims 'processing' forever — the exact bug it fixes"
        )
        assert entry["task"] == TASK_NAME
        assert entry["schedule"] > 0

    def test_its_module_is_autodiscovered(self):
        """Without this the worker may not import the module at all."""
        import inspect

        from src.core import celery_app as mod

        src = inspect.getsource(mod)
        assert '"src.workers.cleanup_stuck_nlp"' in src, (
            "cleanup_stuck_nlp is not in autodiscover_tasks"
        )


class TestEveryScheduledTaskIsRegistered:
    def test_no_beat_entry_names_a_task_that_does_not_exist(self):
        missing = sorted(
            {
                entry["task"]
                for entry in celery_app.conf.beat_schedule.values()
                if entry["task"] not in celery_app.tasks
            }
        )
        assert not missing, (
            f"beat is scheduled to fire tasks nothing registers: {missing}. "
            "These fail silently at dispatch time."
        )


class TestRoutingDoesNotDependOnBeatOptions:
    """A task must reach the right queue however it is called."""

    def test_no_scheduled_task_falls_back_to_the_default_queue(self):
        offenders = {}
        for entry in celery_app.conf.beat_schedule.values():
            name = entry["task"]
            queue = _queue_name(celery_app.amqp.router.route({}, name))
            if queue == DEFAULT_QUEUE:
                offenders[name] = queue

        assert not offenders, (
            "these tasks route to the default queue when called directly, so a "
            f".delay() puts CPU work on the GPU worker: {sorted(offenders)}. "
            "task_routes globs match the TASK NAME — a short name needs an "
            "exact-name entry."
        )

    def test_the_nlp_janitor_routes_to_low_priority_without_options(self):
        queue = _queue_name(celery_app.amqp.router.route({}, TASK_NAME))
        assert queue == "low_priority", (
            f"{TASK_NAME} routes to {queue!r} when called directly"
        )

    def test_beat_options_and_bare_routing_agree(self):
        """If they disagree, the scheduled run and a manual run behave
        differently — the kind of split that hides for months."""
        disagreements = []
        for entry in celery_app.conf.beat_schedule.values():
            name = entry["task"]
            bare = _queue_name(celery_app.amqp.router.route({}, name))
            with_opts = _queue_name(
                celery_app.amqp.router.route(entry.get("options", {}), name)
            )
            if bare != with_opts:
                disagreements.append((name, bare, with_opts))

        assert not disagreements, (
            f"scheduled and direct invocation use different queues: {disagreements}"
        )

    def test_the_cpu_worker_actually_consumes_that_queue(self):
        """Correct routing to a queue nothing consumes is still a black hole."""
        from pathlib import Path

        import yaml

        manifest = Path(__file__).resolve().parents[3] / "k8s" / "base" / "backend.yaml"
        if not manifest.exists():          # pragma: no cover
            pytest.skip("manifest not found")

        consumed = set()
        for doc in yaml.safe_load_all(manifest.read_text()):
            if not doc or doc.get("kind") != "Deployment":
                continue
            for c in doc["spec"]["template"]["spec"]["containers"]:
                env = {e["name"]: e.get("value") for e in c.get("env", [])}
                if env.get("SERVICE_TYPE") == "celery-worker":
                    consumed |= set((env.get("CELERY_QUEUES") or "").split(","))

        queue = _queue_name(celery_app.amqp.router.route({}, TASK_NAME))
        assert queue in consumed, (
            f"{TASK_NAME} routes to {queue!r}, which no worker container "
            f"subscribes to (consumed: {sorted(consumed)})"
        )
