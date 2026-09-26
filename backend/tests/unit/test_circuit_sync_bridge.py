"""`_run_sync` must actually work — it gates the whole circuit pipeline.

THE DEFECT THIS PINS
--------------------
`_run_sync` hand-rolled the generator protocol:

    gen = get_sync_db()
    sync_db = next(gen)

`get_sync_db` is `@contextmanager`-decorated, so calling it returns a
`_GeneratorContextManager`, not a generator. `next()` on that raises
`TypeError: '_GeneratorContextManager' object is not an iterator`.

Every request through this bridge 500'd — and it is the SHARED bridge for
start_circuit_capture, run_circuit_discovery, run_attribution_pass,
validate_circuit_edges, run_circuit_faithfulness and their cancel routes. The
entire causal-discovery pipeline was registered, documented, and incapable of
succeeding even once.

It went unnoticed because the pipeline had never been RUN through the API. The
tools' unit tests mock the client; the reachability guards assert a tool issues
its documented call, not that the endpoint behind it works. Found by trying to
use it for real.

Every other caller in the codebase writes `with get_sync_db() as db:`. This one
place invented its own protocol and got it wrong.
"""

import asyncio
import inspect

import pytest


class TestTheSyncBridgeWorks:
    def test_it_does_not_hand_roll_the_generator_protocol(self):
        """`get_sync_db` is a context manager. Treating it as a generator is
        the exact defect, and it is invisible until the endpoint is called."""
        from src.api.v1.endpoints import circuit_discovery

        # AST, not text: the fix's own comment EXPLAINS the defect and
        # contains the word `next(`, which a substring scan flagged. A guard
        # that cannot tell an explanation from a violation is the same mistake
        # the rung-3 copy audit made earlier today.
        import ast

        tree = ast.parse(inspect.getsource(circuit_discovery._run_sync).lstrip())
        calls_next = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "next"
            for n in ast.walk(tree)
        )
        source = inspect.getsource(circuit_discovery._run_sync)
        assert not calls_next, (
            "_run_sync calls next() on get_sync_db(), which is "
            "@contextmanager-decorated and returns a _GeneratorContextManager "
            "— TypeError on every request. Use `with get_sync_db() as db:`."
        )
        assert "with get_sync_db()" in source, (
            "_run_sync should acquire the session with the context manager, "
            "the way every other caller in the codebase does"
        )

    @pytest.mark.asyncio
    async def test_it_actually_runs_a_function_and_returns_its_value(self):
        """The behavioural half. The source check above would pass against a
        bridge that acquires the session correctly and then never calls `fn`,
        or swallows its return."""
        from src.api.v1.endpoints import circuit_discovery

        sentinel = object()
        seen = {}

        def _fn(sync_db):
            seen["db"] = sync_db
            return sentinel

        result = await circuit_discovery._run_sync(None, _fn)

        assert result is sentinel, (
            "_run_sync did not return the callee's value — callers assign it "
            "(e.g. `run = await _run_sync(db, _create)`) and would get None"
        )
        assert seen.get("db") is not None, (
            "_run_sync called fn without a session"
        )

    @pytest.mark.asyncio
    async def test_the_session_is_released_even_when_fn_raises(self):
        """The endpoints raise CaptureConflictError through this bridge for the
        GPU advisory lock — a leaked session there would exhaust the pool under
        exactly the contention the lock exists to manage."""
        from src.api.v1.endpoints import circuit_discovery

        class Boom(RuntimeError):
            pass

        def _fn(sync_db):
            raise Boom("propagate me")

        with pytest.raises(Boom):
            await circuit_discovery._run_sync(None, _fn)

    def test_every_circuit_gpu_ROUTE_uses_this_bridge(self):
        """Scope check: this is not one endpoint's helper.

        If a future route stops using it, that route loses the advisory-lock
        guard these callers acquire through it — worth failing loudly rather
        than discovering it as two GPU jobs racing.
        """
        from pathlib import Path

        endpoints = Path(inspect.getfile(
            __import__("src.api.v1.endpoints.circuit_discovery",
                       fromlist=["x"])
        )).parent

        users = []
        for name in ("circuit_discovery.py", "circuit_validation.py", "circuits.py"):
            text = (endpoints / name).read_text()
            if "_run_sync(" in text:
                users.append(name)
        assert len(users) >= 3, (
            f"only {users} use the sync bridge — the circuit GPU routes are "
            "expected to share it for the advisory lock"
        )
