"""MIS-E2E-003 / -099 — process kill and process spawn over HTTP.

Both are privilege operations regardless of who can reach the port, which is
why neither is covered by the accepted network-boundary posture (MIS-E2E-002).
That posture concedes "anyone who can reach the host can read the API"; it does
not concede terminating processes.

  * `POST /steering/reset` and `/exit-mode` shelled out to
    `pkill -9 -f steering@` — a PATTERN kill that SIGKILLs any process on the
    host whose command line contains "steering@": another user's shell, an
    unrelated container sharing the PID namespace, someone's `grep steering@`.
  * `POST /system/restart` took no arguments, required nothing, and returned
    200. Unauthenticated, unrated and idempotent, so a caller repeating it kept
    the backend permanently unavailable — the restart policy that makes the
    feature work is what makes the loop self-sustaining.
"""

import ast
import inspect

import pytest


# ── MIS-E2E-003 · no pattern kills ─────────────────────────────────────────

def test_no_endpoint_shells_out_to_pkill():
    """Parsed as a CALL, not grepped — the replacement comments name `pkill`
    in order to explain why it was removed."""
    from src.api.v1.endpoints import steering

    tree = ast.parse(inspect.getsource(steering))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for arg in node.args:
            if isinstance(arg, ast.List):
                first = arg.elts[0] if arg.elts else None
                if isinstance(first, ast.Constant) and first.value == "pkill":
                    offenders.append(node.lineno)
    assert not offenders, (
        f"`pkill` is invoked at lines {offenders}; a cmdline-pattern kill "
        f"reachable over HTTP can SIGKILL unrelated processes on the host"
    )


def test_the_orphan_sweep_kills_only_pids_this_process_spawned():
    from src.api.v1.endpoints import steering

    src = inspect.getsource(steering._kill_orphan_steering_workers)
    assert "_SPAWNED_WORKER_PIDS" in src
    assert "os.kill" in src


def test_every_worker_spawn_records_its_pid():
    """A sweep over a set nothing populates would kill nothing — the fix would
    look right and do nothing."""
    from src.api.v1.endpoints import steering

    tree = ast.parse(inspect.getsource(steering))
    popens = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "Popen"
    ]
    assert popens, "no worker spawn found — the scan broke"

    registrations = inspect.getsource(steering).count("_SPAWNED_WORKER_PIDS.add")
    assert registrations >= len(popens), (
        f"{len(popens)} spawn sites but only {registrations} record their pid; "
        f"an unrecorded worker can never be reaped"
    )


@pytest.mark.asyncio
async def test_the_sweep_tolerates_an_already_dead_pid():
    """The normal case: the worker exited on its own."""
    from src.api.v1.endpoints import steering

    steering._SPAWNED_WORKER_PIDS.clear()
    steering._SPAWNED_WORKER_PIDS.add(2_147_480_000)   # not a live pid
    killed = await steering._kill_orphan_steering_workers()
    assert killed == 0
    assert not steering._SPAWNED_WORKER_PIDS, "the set must be drained either way"


# ── MIS-E2E-099 · restart is gated ─────────────────────────────────────────

def test_restart_requires_the_internal_token():
    from src.api.v1.endpoints import system

    import textwrap

    src = inspect.getsource(system.restart_backend)
    assert "x_internal_token" in src, "restart takes no credential at all"

    # Parse the CALL. The docstring names `hmac.compare_digest` in order to say
    # why `==` is wrong, so a substring check passes against a version using
    # `==`. (Sixth time this trap has appeared in this remediation.)
    tree = ast.parse(textwrap.dedent(src))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "compare_digest" in calls, (
        f"the token is not compared with hmac.compare_digest (calls found: "
        f"{sorted(calls)}); the internal endpoints in main.py use it and so "
        f"must this"
    )
    assert "403" in src


@pytest.mark.asyncio
async def test_restart_refuses_without_a_token():
    """Exercise it, not just read it."""
    from fastapi import BackgroundTasks, HTTPException

    from src.api.v1.endpoints.system import restart_backend

    with pytest.raises(HTTPException) as exc:
        await restart_backend(BackgroundTasks(), x_internal_token=None)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_restart_refuses_a_wrong_token():
    from fastapi import BackgroundTasks, HTTPException

    from src.api.v1.endpoints.system import restart_backend

    with pytest.raises(HTTPException) as exc:
        await restart_backend(BackgroundTasks(), x_internal_token="not-the-secret")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_restart_accepts_the_real_token():
    """Negative control for the direction: the feature must still work, or the
    operator loses the only way to clear orphaned GPU memory."""
    from fastapi import BackgroundTasks

    from src.api.v1.endpoints.system import restart_backend
    from src.core.config import settings

    tasks = BackgroundTasks()
    result = await restart_backend(tasks, x_internal_token=settings.internal_api_secret)
    assert result["status"] == "restarting"
    # And it scheduled the exit rather than performing it inline.
    assert tasks.tasks, "the restart was authorised but nothing was scheduled"


class TestCalibrationJudgeEndpointIsValidated:
    """MIS-E2E-075: free-form input handed to a client running inside the pod.

    `judge_endpoint` arrives per-request and is passed to an `OpenAI` client
    constructed in the backend. Unvalidated, that is a server-side request
    forgery primitive: the caller picks the host the pod connects to, and the
    pod reaches things the caller cannot — cloud instance metadata above all.

    The check reuses `validate_llm_endpoint_url`, the same one the labeling path
    applies to the same kind of field, so the two cannot drift apart.
    """

    def _body(self, **kwargs):
        from src.api.v1.endpoints.circuits import CalibrationBody

        return CalibrationBody(**kwargs)

    def test_a_non_http_scheme_is_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._body(judge_endpoint="file:///etc/passwd")

    def test_cloud_metadata_is_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._body(judge_endpoint="http://169.254.169.254/latest/meta-data/")

    def test_an_internal_llm_server_is_still_allowed(self):
        """The normal case. A guard that blocks it would just be turned off."""
        body = self._body(judge_endpoint="http://127.0.0.1:8001/v1")
        assert body.judge_endpoint == "http://127.0.0.1:8001/v1"

    def test_omitting_it_is_still_allowed(self):
        assert self._body().judge_endpoint is None

    def test_it_uses_the_shared_validator_not_a_second_copy(self):
        import inspect

        from src.api.v1.endpoints import circuits

        source = inspect.getsource(circuits.CalibrationBody)
        assert "validate_llm_endpoint_url" in source, (
            "a second hand-rolled URL check here will drift from the labeling "
            "path's; MIS-E2E-072 and -092 are both that shape"
        )
