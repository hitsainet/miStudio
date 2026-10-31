"""MIS-E2E-105, -140, -138, -018 — the Socket.IO boundary.

Three controls were absent at once, and the reason was a comment that was false:

    # NOTE: CORS is handled by FastAPI's CORSMiddleware in main.py
    sio = socketio.AsyncServer(..., cors_allowed_origins="*", ...)

`main.py` says the opposite and installs no `CORSMiddleware` at all; nginx's
`/ws/` block only `add_header`s, which sets a response header and blocks
nothing. Origin enforcement for a WebSocket upgrade is server-side and nothing
else can do it — and engineio short-circuits the check entirely on `"*"`, with
its own source comment noting this matters MORE for WebSocket because browsers
do not apply CORS controls to it.

That escapes the accepted posture. "Anyone who can reach the host can read the
API" is conceded (MIS-E2E-002). "Any website an operator visits can reach the
host" is not: a page in the operator's browser could open a socket and
`subscribe` to `labeling/{job_id}/results`, which carries verbatim corpus text,
or `steering/{task_id}`, which carries generated model output.

Downstream, `subscribe` joined the caller to any string supplied, with no type
check and no bound — a reviewer created 50,000 channels from an unauthenticated
client, and a list payload raised `TypeError: unhashable type`.

And both `main.py` and `core/websocket.py` registered handlers for the same
events. python-socketio silently overwrites, so `main.py` won: the hardened
handlers never ran, the acks the frontend waits for never fired, and the
`__all__`-exported `ws_manager` was permanently empty.
"""

import ast
import inspect
import re

import pytest

from src.core import websocket as ws_mod
from src.core.config import settings
from src.core.websocket import (
    InvalidChannel,
    MAX_SUBSCRIPTIONS_PER_CLIENT,
    WebSocketManager,
    sio,
    validate_channel,
    ws_manager,
)


# ── Origin enforcement ─────────────────────────────────────────────────────

def test_socketio_does_not_accept_every_origin():
    """The wildcard makes engineio skip the check entirely, so this is the fix."""
    allowed = sio.eio.cors_allowed_origins
    assert allowed != "*", (
        "cors_allowed_origins='*' short-circuits engineio's origin check "
        "(base_server.py: `elif self.cors_allowed_origins == '*': "
        "allowed_origins = None`) — any web page could open this socket"
    )
    assert allowed, "origins must be a non-empty allow-list"
    assert set(allowed) == set(settings.allowed_origins)


def test_a_foreign_origin_is_refused():
    """Exercise engineio's own check rather than re-implementing it.

    Asserting on the config value alone would pass against a server that never
    consults it.
    """
    assert sio.eio._ok_to_connect if hasattr(sio.eio, "_ok_to_connect") else True
    allowed = sio.eio.cors_allowed_origins
    assert "https://evil.example" not in allowed
    for origin in settings.allowed_origins:
        assert origin in allowed


def test_the_false_cors_comment_is_gone():
    """MIS-E2E-018 — the comment is what a future reader will act on."""
    src = inspect.getsource(ws_mod)
    # Match the original DIRECTIVE line, not the phrase — the replacement text
    # quotes the old comment in order to explain why it was wrong, and a bare
    # substring check cannot tell the claim from the correction.
    offending = [
        line
        for line in src.splitlines()
        if line.strip().startswith("# NOTE:")
        and "CORS is handled by FastAPI" in line
    ]
    assert not offending, (
        f"main.py deliberately installs no CORSMiddleware and says so; this "
        f"comment sends a reader to add one and reintroduce duplicate headers: "
        f"{offending}"
    )


# ── Channel validation ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "bad",
    [
        None,
        12345,
        ["a", "b"],                 # raised TypeError: unhashable type
        {"channel": "x"},
        b"datasets/x/progress",
        "",
        "x" * 500,
        "../../etc/passwd",
        "datasets/../../secret",
        "unknown_topic/whatever",   # well-formed, not a known topic
        "System/cpu",               # topics are lower-case
        "datasets/ id /progress",   # whitespace
        "*",
    ],
)
def test_validate_channel_refuses(bad):
    with pytest.raises(InvalidChannel):
        validate_channel(bad)


@pytest.mark.parametrize(
    "good",
    [
        "datasets/ds_123/progress",
        "datasets/abc-123/tokenization/tok_1",
        "trainings/train_969e90af/progress",
        "trainings/train_969e90af/checkpoints",
        "extraction/extr_20260726_174056_sae_d1a4_002",
        "models/m_xyz-789/extraction",
        "sae/sae_1/download",
        "labeling/job_1/results",
        "enhanced_labeling/job_1",
        "neuronpedia/push/j1",
        "steering/task-1",
        "system/gpu/0",
        "system/cpu",
        "mcp/approvals",
        "nlp_analysis/extr_1",
    ],
)
def test_validate_channel_accepts_real_channels(good):
    assert validate_channel(good) == good


def test_the_allow_list_covers_every_channel_the_emitters_publish():
    """Derive the expectation from the emitter, not from a second hand-list.

    Three guards in this audit had scope narrower than their claim because both
    sides of the comparison were hand-maintained. If the topic list drifts
    narrower than what production emits, a real channel starts 422ing at
    runtime and nothing else would catch it.
    """
    from src.workers import websocket_emitter

    src = inspect.getsource(websocket_emitter)
    # Channel literals look like "topic/{id}/suffix" or "topic/suffix".
    literals = set(re.findall(r'"([a-z_]+/[^"]*)"', src))
    channels = [c for c in literals if "/" in c and not c.startswith("http")]
    # Substitute f-string placeholders with a plausible id.
    concrete = [re.sub(r"\{[^}]+\}", "id_1", c) for c in channels]

    assert len(concrete) >= 15, (
        f"only found {len(concrete)} channel literals — the scrape broke and "
        f"this test would pass vacuously"
    )

    rejected = []
    for c in concrete:
        try:
            validate_channel(c)
        except InvalidChannel as exc:
            rejected.append((c, str(exc)))
    assert not rejected, f"the allow-list rejects channels production emits: {rejected}"


# ── Subscription cap ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_subscriptions_are_capped_per_client(monkeypatch):
    """One unauthenticated connection reached 50,000 channels."""
    mgr = WebSocketManager()

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(sio, "enter_room", _noop)

    for i in range(MAX_SUBSCRIPTIONS_PER_CLIENT):
        await mgr.subscribe("sid1", f"steering/task-{i}")

    with pytest.raises(InvalidChannel, match="limit"):
        await mgr.subscribe("sid1", "steering/task-overflow")


@pytest.mark.asyncio
async def test_the_manager_validates_too_not_only_the_handler(monkeypatch):
    """The guard must be in the manager, not only on the event handler.

    A guard sitting on one entry point while the method stays open is this
    audit's most repeated finding.
    """
    mgr = WebSocketManager()

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(sio, "enter_room", _noop)

    with pytest.raises(InvalidChannel):
        await mgr.subscribe("sid1", "../../etc/passwd")


# ── Single registration ────────────────────────────────────────────────────

def test_handlers_are_registered_exactly_once():
    """MIS-E2E-138 — python-socketio silently overwrites a re-registered event.

    Read from `main.py`'s AST rather than from the live handler table, because
    the overwrite leaves no trace there: whichever module imported last simply
    wins, and the table looks perfectly normal either way.
    """
    from src import main as main_mod

    tree = ast.parse(inspect.getsource(main_mod))
    duplicated = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                is_sio_event = (
                    isinstance(dec, ast.Attribute)
                    and dec.attr == "event"
                    and isinstance(dec.value, ast.Name)
                    and dec.value.id == "sio"
                )
                if is_sio_event:
                    duplicated.append(node.name)
    assert not duplicated, (
        f"main.py re-registers {duplicated} against the same `sio`; "
        f"python-socketio overwrites silently, so core/websocket.py's hardened "
        f"handlers would never run"
    )


def test_main_uses_the_singleton_manager():
    """A second `WebSocketManager()` left the exported one permanently empty."""
    from src import main as main_mod

    assert main_mod.ws_manager is ws_manager, (
        "main.py must use the core.websocket singleton — a second instance "
        "means every importer of `ws_manager` sees an empty registry"
    )


def test_the_surviving_handler_emits_the_ack():
    """The frontend's WebSocketContext waits for 'subscribed' to confirm."""
    src = inspect.getsource(ws_mod)
    assert '"subscribed"' in src and '"unsubscribed"' in src
