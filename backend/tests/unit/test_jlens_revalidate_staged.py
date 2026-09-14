"""Re-validating a staged artifact must be REACHABLE, not just implemented.

When validation refuses, `_validate_and_commit` keeps the staging directory and
logs:

    "The staged fit is kept at <dir> so it can be re-validated without
     refitting."

Until 2026-09-05 nothing could do that. There was no route, no worker task and
no MCP tool that validated a staged artifact and committed it — the sequence
existed only inside `_fit_and_publish`, reachable only by running a whole fit.
`/artifacts/{slug}/validate` sees only COMMITTED artifacts and cannot run
SEMANTIC (it has no loaded model); `/publish` is a HuggingFace upload and
refuses a staged artifact explicitly. So the operator-facing message described a
capability that did not exist, and the real cost of a refusal was the fit again:
53 minutes on gemma-4-12B, whose artifact turned out to be correct.

These tests hold the code to that sentence.
"""

import dis

import pytest
from fastapi.routing import APIRoute


def _underlying(task):
    """The real function behind a Celery Task and its `owns_its_failure` wrap.

    Order matters: the Task object has no `__wrapped__`, so unwrapping first
    silently returns the Task and every bytecode assertion below inspects
    `Task.__call__` instead of the body — passing or failing for reasons that
    have nothing to do with the code under test.
    """
    fn = getattr(task, "run", task)
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


# ── the route is in the live registry ──────────────────────────────────────

def _routes():
    from src.api.v1.endpoints.jlens import router

    return {
        (r.path, m)
        for r in router.routes
        if isinstance(r, APIRoute)
        for m in r.methods
    }


def test_the_revalidate_route_is_registered():
    """Presence in the LIVE ROUTER, never "the module imports".

    A test that imports the handler directly passes against a route no client
    can call — the shape that shipped 16 unreachable MCP tools here.
    """
    assert ("/jlens/artifacts/revalidate", "POST") in _routes(), (
        "there is no re-validate route; the refusal message still promises a "
        "capability nothing provides"
    )


def test_the_route_reaches_the_revalidate_task_specifically():
    """...and reaches the RIGHT task.

    A route that exists and dispatches a fit would satisfy the test above while
    costing the full GPU run this feature exists to avoid.
    """
    from src.api.v1.endpoints import jlens

    loads = [
        i.argval
        for i in dis.get_instructions(jlens.revalidate_staged)
        if i.opname.startswith("LOAD")
    ]
    assert "revalidate_staged_artifact" in loads, (
        "the route does not dispatch revalidate_staged_artifact"
    )
    assert "fit_jlens_artifact" not in loads, (
        "the re-validate route dispatches a FIT — it would pay for the whole "
        "run the staged artifact exists to make unnecessary"
    )


# ── the task actually validates and commits ────────────────────────────────

def test_the_task_is_registered_with_celery():
    from src.core.celery_app import celery_app
    import src.workers.jlens_fit_tasks  # noqa: F401 - registers the task

    assert (
        "src.workers.jlens_fit_tasks.revalidate_staged_artifact"
        in celery_app.tasks
    ), "the task is defined but not registered; no worker will ever run it"


def test_the_task_routes_to_the_gpu_queue():
    """SEMANTIC runs a real forward pass, so this must not land on the CPU queue.

    Routing globs in this project match the TASK NAME, and a name outside the
    module's glob silently uses the default queue — the trap recorded in
    CLAUDE.md that once sent GPU work to a CPU worker.
    """
    from src.core.celery_app import celery_app

    routes = celery_app.conf.task_routes or {}
    name = "src.workers.jlens_fit_tasks.revalidate_staged_artifact"
    fit = "src.workers.jlens_fit_tasks.fit_jlens_artifact"

    def _queue_for(task_name):
        import fnmatch

        for pattern, spec in routes.items():
            if fnmatch.fnmatch(task_name, pattern):
                return spec.get("queue")
        return None

    assert _queue_for(name) is not None, (
        f"{name} matches no route pattern and will use the default queue"
    )
    assert _queue_for(name) == _queue_for(fit), (
        "re-validation does not share the fit's queue, so it can run "
        "concurrently with a fit on a single GPU"
    )


def test_the_task_calls_the_shared_validate_and_commit():
    """It must reuse the fit's sequence, not grow a second copy.

    Two implementations of "what publishable means" is how two paths come to
    disagree — the reason `_local_pass` was removed from this module.
    """
    from src.workers.jlens_fit_tasks import revalidate_staged_artifact

    fn = _underlying(revalidate_staged_artifact)
    loads = {
        i.argval
        for i in dis.get_instructions(fn)
        if i.opname.startswith("LOAD")
    }
    assert "_validate_and_commit" in loads, (
        "the re-validate task does not call the shared helper; it has its own "
        "copy of the publish rule, or it does not publish at all"
    )


def test_the_task_does_not_refit():
    """The whole point: no fitter, no prompts, no accumulation."""
    from src.workers.jlens_fit_tasks import revalidate_staged_artifact

    fn = _underlying(revalidate_staged_artifact)
    loads = {
        i.argval
        for i in dis.get_instructions(fn)
        if i.opname.startswith("LOAD")
    }
    for forbidden in ("JacobianFitter", "_fit_and_publish"):
        assert forbidden not in loads, (
            f"the re-validate task reaches {forbidden}; it is refitting, which "
            f"is the cost this task exists to avoid"
        )


def test_the_layers_come_from_the_artifact_not_the_caller():
    """`validate` compares what it finds against `expected_layers`.

    A caller-supplied list could turn a complete artifact into a
    missing-layers failure, or pass a partial one as whole. The request schema
    must therefore offer no layers field at all.
    """
    from src.api.v1.endpoints.jlens import RevalidateRequest

    assert "layers" not in RevalidateRequest.model_fields, (
        "the caller can name the layers to validate against, so a wrong list "
        "changes the verdict on an unchanged artifact"
    )


def test_the_progress_row_uses_its_own_task_type():
    """Active Operations must not label a re-validation as a fit.

    A user watching a 'jlens_fit' row would reasonably expect the GPU to be
    busy for the length of a fit.
    """
    from src.workers import jlens_progress

    assert jlens_progress.REVALIDATE == "jlens_revalidate"
    assert jlens_progress.REVALIDATE != jlens_progress.FIT


def test_the_shared_helper_is_what_the_fit_uses_too():
    """Negative control for the extraction.

    Every test above asserts the re-validate path reaches
    `_validate_and_commit`. That is worth nothing if the FIT quietly kept its
    own copy — the two would drift and the re-validation would be validating
    against a different rule than the fit it is recovering.
    """
    from src.workers.jlens_fit_tasks import _fit_and_publish

    loads = {
        i.argval
        for i in dis.get_instructions(_fit_and_publish)
        if i.opname.startswith("LOAD")
    }
    assert "_validate_and_commit" in loads, (
        "the fit no longer uses the shared helper, so the two paths can now "
        "disagree about what publishable means"
    )
