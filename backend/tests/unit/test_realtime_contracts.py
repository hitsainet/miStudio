"""Task 9 — realtime paths whose failure mode is going quietly dark.

MIS-E2E-067 — a failed extraction showed no reason.
    The `extraction:failed` emit sent the key `"error"` where every other
    emitter and every frontend consumer uses `error_message`. The store
    spread-merges the payload, so `error_message: undefined` OVERWROTE the real
    message the database already had, and nothing triggered a refetch. Worse on
    the OOM path, whose whole value is its diagnostics — and that is the path a
    user hit twice on 2026-08-23.

    Both emits also sent `status: "extracting"` — on the COMPLETED and FAILED
    events — so a finished job was written back as still running.

MIS-E2E-139 — the system monitor could die silently and permanently.
    A crash during startup left `_running = True` with no task, so `start()`
    refused every retry with "already running", logged at WARNING. Every
    `system/*` channel went silent and the Monitor page simply froze: the
    frontend fallback keys on CONNECTION, not data freshness, so a live socket
    delivering nothing looks healthy.

MIS-E2E-141 — `emit_system_metrics` emitted `"metrics"`, which nothing listens
    for, and returned True.
"""

import ast
import asyncio
import inspect

import pytest


# ── MIS-E2E-067: the emit payloads ─────────────────────────────────────────

def _emit_payloads(event_name: str) -> list[dict]:
    """Every `emit_progress(event=<event_name>, data={...})` literal, parsed."""
    from src.services import extraction_service

    tree = ast.parse(inspect.getsource(extraction_service))
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "emit_progress"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        ev = kw.get("event")
        if isinstance(ev, ast.Constant) and ev.value == event_name:
            data = kw.get("data")
            if isinstance(data, ast.Dict):
                found.append(
                    {
                        k.value: v
                        for k, v in zip(data.keys, data.values)
                        if isinstance(k, ast.Constant)
                    }
                )
    return found


def test_the_failure_emit_uses_the_key_the_frontend_reads():
    payloads = _emit_payloads("extraction:failed")
    assert payloads, "no extraction:failed emit found — the scan broke"
    for p in payloads:
        assert "error_message" in p, (
            "the failure emit does not carry `error_message`; the store "
            "spread-merges it, so the real reason is overwritten with undefined"
        )
        assert "error" not in p, (
            "`error` is not the key any consumer reads — it silently blanks the "
            "message instead of delivering it"
        )


@pytest.mark.parametrize(
    "event, expected_status",
    [("extraction:failed", "FAILED"), ("extraction:completed", "COMPLETED")],
)
def test_the_emitted_status_matches_the_event(event, expected_status):
    """A terminal event carrying `status: extracting` un-finishes the job."""
    payloads = _emit_payloads(event)
    assert payloads, f"no {event} emit found"
    for p in payloads:
        status = p.get("status")
        assert status is not None, f"{event} carries no status"
        rendered = ast.unparse(status)
        assert expected_status in rendered, (
            f"{event} emits status {rendered}, which contradicts the event"
        )


def test_the_progress_emit_still_says_extracting():
    """Negative control for the direction of the fix.

    A blanket rename would have broken the in-progress event, where
    `status: extracting` is correct.
    """
    payloads = _emit_payloads("extraction:progress")
    assert payloads
    assert any("EXTRACTING" in ast.unparse(p["status"]) for p in payloads if "status" in p)


# ── MIS-E2E-139: the monitor cannot die silently ───────────────────────────

def test_a_failed_start_does_not_wedge_the_monitor(monkeypatch):
    """`_running` must not survive a failed start, or nothing can restart it."""
    import httpx

    from src.services.background_monitor import BackgroundMonitor

    monitor = BackgroundMonitor(interval_seconds=1)

    def _boom(*a, **k):
        raise RuntimeError("no event loop / client construction failed")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)

    with pytest.raises(RuntimeError):
        asyncio.run(monitor.start())

    assert monitor._running is False, (
        "a crashed start left _running=True, so every retry is refused with "
        "'already running' and every system/* channel stays silent forever"
    )
    assert monitor._task is None

    # THE ORDER IS THE FIX, not the reset in the except block.
    #
    # Mutation control C95 first survived because deleting `_running = False`
    # from the handler changed nothing — `_running` is now set AFTER the setup
    # succeeds, so on the failure path it was never True to begin with. The
    # reset is defence in depth; the ordering is what makes recovery possible.
    # Pinned directly, so moving the assignment back above the try goes red.
    # `textwrap.dedent`, not `cleandoc` — cleandoc leaves the FIRST line
    # unindented and dedents the rest, which is right for docstrings and
    # produces invalid Python for a method body.
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(BackgroundMonitor.start)))
    fn = tree.body[0]
    try_index = next(
        i for i, node in enumerate(fn.body) if isinstance(node, ast.Try)
    )
    running_assignments = [
        i
        for i, node in enumerate(fn.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_running" for t in node.targets
        )
    ]
    assert running_assignments, "no `self._running = ...` in start()"
    assert all(i > try_index for i in running_assignments), (
        "`self._running` is assigned BEFORE the setup try-block; a crash then "
        "leaves it True with no task and start() refuses every retry"
    )


def test_the_monitor_can_start_after_a_failed_start(monkeypatch):
    """The consequence, not just the flag: recovery must actually work."""
    import httpx

    from src.services.background_monitor import BackgroundMonitor

    monitor = BackgroundMonitor(interval_seconds=1)
    calls = {"n": 0}
    real = httpx.AsyncClient

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return real(*a, **k)

    monkeypatch.setattr(httpx, "AsyncClient", _flaky)

    async def scenario():
        with pytest.raises(RuntimeError):
            await monitor.start()
        await monitor.start()          # must be allowed to proceed
        assert monitor._running is True
        await monitor.stop()

    asyncio.run(scenario())


def test_stop_closes_the_client_even_if_the_loop_died():
    """A loop that died of something other than cancellation must not take the
    HTTP client down with it.

    Mutation control C96 first SURVIVED: the test parsed `stop()` for an
    `except Exception` handler and found the one wrapping `aclose()`, so
    deleting the handler on the TASK AWAIT — the one that matters — changed
    nothing. Shape, not behaviour. Rewritten to actually run it.
    """
    from src.services.background_monitor import BackgroundMonitor

    monitor = BackgroundMonitor(interval_seconds=1)

    class _DeadTask:
        """Awaits to an exception, as a loop that crashed would."""

        def cancel(self):
            return True

        def __await__(self):
            async def _raise():
                raise RuntimeError("the monitor loop died")

            return _raise().__await__()

    closed = {"n": 0}

    class _Client:
        async def aclose(self):
            closed["n"] += 1

    async def scenario():
        monitor._task = _DeadTask()
        monitor._http_client = _Client()
        monitor._running = True
        await monitor.stop()      # must not raise

    asyncio.run(scenario())

    assert closed["n"] == 1, (
        "the HTTP client was not closed — the dead loop's exception escaped "
        "stop() and took the FastAPI lifespan shutdown with it"
    )
    assert monitor._task is None
    assert monitor._http_client is None
    assert monitor._running is False


# ── MIS-E2E-141: the event name nothing listens for ────────────────────────

def test_every_system_metrics_emit_uses_the_same_event_name():
    """One emitter said `"metrics"` while every sibling said `system:metrics`.

    It had no callers, so it delivered to nobody and returned True — the
    combination that makes a broken emit invisible to its own caller.
    """
    from src.workers import websocket_emitter

    tree = ast.parse(inspect.getsource(websocket_emitter))
    names = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "emit_progress"):
            continue
        # positional form: emit_progress(channel, "<event>", data)
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            ev = node.args[1].value
            if isinstance(ev, str) and "metric" in ev:
                names.add(ev)
    assert names == {"system:metrics"}, (
        f"system metrics are emitted under {sorted(names)}; the frontend "
        f"listens for 'system:metrics' only"
    )


# ── MIS-E2E-136 · a sync emit must not POST to its own event loop ──────────

def test_emit_takes_the_in_process_path_when_a_loop_is_running(monkeypatch):
    """The self-deadlock, closed at the emitter rather than the call site.

    Called from an `async def` handler, the sync HTTP POST to
    `/api/internal/ws/emit` blocks the single event loop waiting for a response
    only that loop can produce: ReadTimeout after 5.01s, whole API frozen, event
    dropped anyway. `datasets.py` calls it twice in sequence.

    `training_service.py` wrapped ONE call in `asyncio.to_thread`; 13 others
    were left. So the guard goes in the emitter, where it covers all of them and
    every future one.
    """
    from src.core import websocket as ws_mod
    from src.workers import websocket_emitter as we

    emitted: list[tuple] = []

    async def _fake_emit(channel, event, data, namespace="/"):
        emitted.append((channel, event, data))

    monkeypatch.setattr(ws_mod.ws_manager, "emit_event", _fake_emit)

    def _no_http(*a, **k):
        raise AssertionError(
            "the emitter POSTed to the API's own endpoint from inside the loop "
            "— this is the deadlock"
        )

    monkeypatch.setattr(we, "_get_http_client", _no_http)

    async def scenario():
        assert we.emit_progress("trainings/t1/progress", "training:progress", {"p": 1}) is True
        # Fire-and-forget: let the scheduled task run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert emitted == [("trainings/t1/progress", "training:progress", {"p": 1})]


def test_emit_still_uses_http_when_there_is_no_loop(monkeypatch):
    """Negative control for the direction.

    A Celery worker has no running loop and no in-process manager to reach, so
    the loopback is the correct path there. A fix that always emitted
    in-process would silently drop every worker event.
    """
    from src.workers import websocket_emitter as we

    posted = {"n": 0}

    class _Resp:
        status_code = 200

    class _Client:
        def post(self, *a, **k):
            posted["n"] += 1
            return _Resp()

    monkeypatch.setattr(we, "_get_http_client", lambda: _Client())

    assert we.emit_progress("trainings/t1/progress", "training:progress", {}) is True
    assert posted["n"] == 1


class TestRetryClassificationIsPinned:
    """MIS-E2E-137 / -142, mutation M26.

    The emitter caught only `httpx.TimeoutException`, so a `ConnectError`
    (connection refused during a pod restart) or a `RemoteProtocolError`
    (half-closed keep-alive) fell through to `except Exception` and was
    abandoned on the first attempt. The events configured with retries are the
    TERMINAL ones — `steering:completed`, `neuronpedia:push_completed`,
    `enhanced_labeling:completed` — where a dropped emit leaves the UI showing
    a finished job as still running, forever.

    The fix widened the handler to `TransportError`. M26 re-narrowed it and
    survived: nothing asserted WHICH failures are retryable, so the decision
    was undefended.
    """

    def test_transport_error_is_the_retry_boundary(self):
        import inspect

        from src.workers import websocket_emitter

        # Whitespace removed: the tokenizer joins with spaces, so
        # `httpx.TransportError` comes back as `httpx . TransportError` and a
        # plain substring check misses it — which is how this test first failed
        # against code that was already correct.
        source = "".join(_code_only_ws(inspect.getsource(websocket_emitter)).split())
        assert "httpx.TransportError" in source, (
            "the retry handler no longer catches TransportError; a connection "
            "refused mid-restart is abandoned on the first attempt"
        )
        assert "excepthttpx.TimeoutException" not in source, (
            "the handler is narrowed back to TimeoutException, which excludes "
            "ConnectError and RemoteProtocolError"
        )

    def test_the_classification_matches_the_installed_httpx(self):
        """The premise, checked against the library rather than assumed."""
        import httpx

        assert issubclass(httpx.ConnectError, httpx.TransportError)
        assert issubclass(httpx.RemoteProtocolError, httpx.TransportError)
        assert issubclass(httpx.ReadTimeout, httpx.TransportError)
        # ...and the exclusion is deliberate: the server answered, so retrying
        # just repeats a rejected request.
        assert not issubclass(httpx.HTTPStatusError, httpx.TransportError)

    def test_the_narrow_class_really_would_miss_them(self):
        """Without this the assertion above could be vacuous."""
        import httpx

        assert not issubclass(httpx.ConnectError, httpx.TimeoutException)
        assert not issubclass(httpx.RemoteProtocolError, httpx.TimeoutException)


def _code_only_ws(source: str) -> str:
    """Comments and docstrings stripped — the module explains M26 in prose."""
    import io
    import tokenize

    out, prev = [], tokenize.INDENT
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and prev in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT,
            ):
                prev = tok.type
                continue
            out.append(tok.string)
            if tok.type not in (tokenize.NL, tokenize.NEWLINE):
                prev = tok.type
    except tokenize.TokenError:  # pragma: no cover
        return source
    return " ".join(out)
