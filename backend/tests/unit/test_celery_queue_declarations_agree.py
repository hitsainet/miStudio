"""OSD-10 — a task's declared queue must be the queue it is actually routed to.

`@celery_app.task(queue=...)` and `task_routes` can disagree, and **task_routes
wins**. `download_and_load_model` declared `processing` while celery resolved it
to `high_priority`, so the decorator described a queue the task never used. One
task in sixty-four disagreed; this keeps it that way.

Two facts worth keeping beside each other: `task_routes` globs match the TASK
NAME (a short name silently lands in the default queue), and the single
`celery-worker` consumes `high_priority, datasets, processing, training,
extraction, sae` with `--pool=solo`, so queue choice alone does not buy
concurrency — it only decides which worker can pick the task up.
"""
import importlib
import pkgutil

import pytest

from src.core.celery_app import celery_app


def _all_tasks():
    import src.workers as workers_package
    for module in pkgutil.iter_modules(workers_package.__path__):
        try:
            importlib.import_module(f"src.workers.{module.name}")
        except Exception:                       # noqa: BLE001 - optional deps
            continue
    return {
        name: task for name, task in celery_app.tasks.items()
        if not name.startswith("celery.")
    }


def _resolved_queue(name, task):
    options = getattr(task, "options", {}) or {}
    route = celery_app.amqp.router.route({}, name, (), {}, options)
    queue = route.get("queue")
    return getattr(queue, "name", queue)


def _declared_queue(task):
    return (getattr(task, "options", {}) or {}).get("queue") or getattr(task, "queue", None)


def test_the_scan_sees_the_real_registry():
    tasks = _all_tasks()
    assert len(tasks) > 50, f"only {len(tasks)} tasks registered — the scan broke"


def test_no_task_declares_a_queue_it_is_not_routed_to():
    disagreements = []
    for name, task in _all_tasks().items():
        declared = _declared_queue(task)
        if declared is None:
            continue                            # relies on task_routes alone
        resolved = _resolved_queue(name, task)
        if declared != resolved:
            disagreements.append(f"{name}: decorator={declared!r} routed={resolved!r}")
    assert not disagreements, (
        "task_routes wins, so a disagreeing decorator documents a queue the task "
        "never uses:\n  " + "\n  ".join(disagreements)
    )


@pytest.mark.parametrize("name,expected", [
    ("workers.model_tasks.download_and_load_model", "high_priority"),
    ("workers.model_tasks.extract_activations", "extraction"),
])
def test_the_queue_a_named_task_resolves_to(name, expected):
    tasks = _all_tasks()
    assert name in tasks, f"{name} is not registered"
    assert _resolved_queue(name, tasks[name]) == expected
