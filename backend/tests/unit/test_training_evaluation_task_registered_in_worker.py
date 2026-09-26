"""A Celery WORKER knows the evaluation task — not just the API process that imports its module.

SAE TRAINING REMEDIATION, ITEM 6. ``POST /trainings/{id}/evaluate`` dispatches
``src.workers.training_evaluation_tasks.evaluate_training``. The endpoint imports
the task module, so in the test process (and the API process) the task is always
registered. A worker process does not import the endpoint: it registers exactly
the modules in ``celery_app``'s ``include`` list. Drop the module from that list
and every in-process test stays green while the worker answers each queued
evaluation with "Received unregistered task" and the training's evaluation sits at
``pending`` for ever.

Found as a surviving mutation (E5: the include line deleted, 55 tests green). This
reproduces the worker's registration in a fresh interpreter that imports nothing
but the Celery app, then asserts the task is in the LIVE registry.

MUTATION CONTROLS (2026-09-15; applied alone, this file run, source restored and
checked by sha256):
  E5r "src.workers.training_evaluation_tasks" removed from include -> RED, test_a_worker_registers_the_evaluation_task
      (the same mutation as E5, which left the in-process endpoint suite green)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
TASK = "src.workers.training_evaluation_tasks.evaluate_training"


def _worker_registry() -> set:
    code = (
        "import json; from src.core.celery_app import celery_app; "
        "celery_app.loader.import_default_modules(); "
        "print(json.dumps(sorted(celery_app.tasks)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True,
        env=os.environ.copy(), timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_a_worker_registers_the_evaluation_task():
    registry = _worker_registry()
    assert TASK in registry, (
        "a worker started from celery_app does not register the evaluation task; every "
        "POST /trainings/{id}/evaluate would be dropped as an unregistered task"
    )


def test_the_probe_sees_a_worker_registry_not_an_empty_one():
    """NEGATIVE CONTROL on the probe: the training task a worker certainly runs is there too."""
    registry = _worker_registry()
    assert "train_sae" in registry
