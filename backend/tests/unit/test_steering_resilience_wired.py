"""MIS-E2E-062 — the steering resilience layer must be REACHABLE.

508 lines implementing a circuit breaker, a concurrency limiter and process
isolation, and not one state-mutating function had a caller. `_failure_count`
starts at 0 and is incremented only in `record_failure`; `_state` starts CLOSED
and leaves it only inside the same dead function. So:

    GET /steering/status  ->  "healthy" if circuit_breaker.state == "closed"

could only ever return "healthy", no matter how many steering tasks had failed,
and `POST /steering/reset` reset state that was never non-default. The
endpoint's own docstring says "use this endpoint to monitor steering health and
diagnose issues".

This is the same shape as the 16 unregistered `millm_circuit_*` MCP tools that
`test_reachability.py` was written to prevent — that harness guards the MCP
surface only, and nothing guarded the service layer.

`CLAUDE.md`'s rule is "a capability is not shipped until a test FAILS when its
wiring is removed." It could not be applied here before, because nothing was
wired. It is applied here now: delete `_guard_steering_dispatch()` from a
dispatch site, or `_record_steering_outcome` from the result endpoint, and this
file goes red.
"""

import ast
import inspect

import pytest

from src.api.v1.endpoints import steering as steering_ep
from src.services import steering_resilience as res


# ── The breaker is wired at both ends ──────────────────────────────────────

def _function_names_calling(module, callee: str) -> set[str]:
    """Functions in `module` whose body contains a call to `callee`."""
    tree = ast.parse(inspect.getsource(module))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name == callee:
                    found.add(node.name)
    return found


def test_every_async_dispatch_endpoint_consults_the_breaker():
    """All three, not one representative.

    "Fixed one representative, never generalized" is this audit's most repeated
    anti-pattern — five independent instances.
    """
    gated = _function_names_calling(steering_ep, "_guard_steering_dispatch")
    dispatchers = {
        n for n in _function_names_calling(steering_ep, "apply_async")
        if "steering" in n or "async" in n
    }
    # Fail closed if the scan stops finding things: a discovery-based test that
    # discovers nothing asserts nothing, which is how source-derived guards in
    # this repo have failed OPEN twice.
    assert len(dispatchers) == 3, (
        f"expected the three async steering dispatch endpoints, found "
        f"{sorted(dispatchers)} — the scan broke and this test would pass "
        f"vacuously"
    )
    missing = dispatchers - gated
    assert not missing, (
        f"these dispatch endpoints do not consult the circuit breaker: {missing}. "
        f"An ungated dispatch means the breaker can open and requests still flow."
    )


def test_the_result_endpoint_feeds_outcomes_back():
    """Without this the breaker never leaves CLOSED and /status is a constant."""
    recorders = _function_names_calling(steering_ep, "_record_steering_outcome")
    assert "get_steering_task_result" in recorders, (
        "GET /async/result/{task_id} is where the API first learns a task's "
        "outcome; if it does not record, nothing ever does"
    )


def test_the_guard_helpers_call_the_breaker_and_not_a_stub():
    src = inspect.getsource(steering_ep._guard_steering_dispatch)
    assert "can_execute" in src
    rec = inspect.getsource(steering_ep._record_steering_outcome)
    assert "record_success" in rec and "record_failure" in rec


# ── The breaker actually transitions ───────────────────────────────────────

@pytest.mark.asyncio
async def test_the_breaker_opens_after_repeated_failures():
    """The behaviour /status is supposed to be reporting."""
    cb = res.CircuitBreaker(res.CircuitBreakerConfig(failure_threshold=3))

    allowed, _ = await cb.can_execute()
    assert allowed

    for _ in range(3):
        await cb.record_failure(Exception("gpu died"))

    allowed, reason = await cb.can_execute()
    assert not allowed, "three failures at threshold 3 must open the breaker"
    assert reason


@pytest.mark.asyncio
async def test_an_outcome_is_recorded_once_per_task_however_often_it_is_polled():
    """The client polls until terminal; the breaker must count the TASK.

    Without de-duplication three polls of one failed task open a
    threshold-of-three breaker on its own.
    """
    steering_ep._recorded_task_outcomes.clear()
    cb = res.get_circuit_breaker()
    await cb.reset()

    for _ in range(5):
        await steering_ep._record_steering_outcome("task-1", succeeded=False, error="boom")

    stats = await cb.get_stats()
    assert stats.failure_count == 1, (
        f"five polls of one failed task recorded {stats.failure_count} failures"
    )
    await cb.reset()
    steering_ep._recorded_task_outcomes.clear()


@pytest.mark.asyncio
async def test_status_reports_degraded_once_the_breaker_is_open():
    """The whole point: /status must be able to say something other than healthy."""
    cb = res.get_circuit_breaker()
    await cb.reset()

    status = await res.get_resilience_status()
    assert status["circuit_breaker"]["state"] == "closed"

    for _ in range(10):
        await cb.record_failure(Exception("gpu died"))

    status = await res.get_resilience_status()
    assert status["circuit_breaker"]["state"] != "closed", (
        "the endpoint computes 'healthy' from exactly this field; if it cannot "
        "change, /steering/status is a constant"
    )
    assert status["circuit_breaker"]["failure_count"] >= 10
    await cb.reset()


# ── The unfit components are gone, not left dead ───────────────────────────

@pytest.mark.parametrize("dead", ["ConcurrencyLimiter", "ProcessIsolationManager",
                                  "get_concurrency_limiter", "get_process_isolation"])
def test_the_architecturally_unfit_components_are_deleted(dead):
    """A semaphore cannot bound a fire-and-forget `apply_async`.

    Leaving them in place would have been a second always-healthy constant, so
    they are removed rather than wired. Deleting is a fix; leaving dead code
    that looks like a control is the finding.
    """
    assert not hasattr(res, dead), (
        f"{dead} is unreachable by construction under Celery dispatch — there "
        f"is no in-process operation to bound"
    )


@pytest.mark.asyncio
async def test_status_does_not_report_components_it_no_longer_has():
    status = await res.get_resilience_status()
    assert "concurrency" not in status
    assert "process_isolation" not in status
    assert status["circuit_breaker"]["scope"] == "api-process", (
        "the breaker is in-process; the endpoint must say so rather than imply "
        "it observes the whole system"
    )
