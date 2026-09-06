"""
WebSocket manager for real-time communication using Socket.IO.

This module provides a WebSocket manager for broadcasting progress updates,
training metrics, and other real-time events to connected clients.
"""

import re
from typing import Any, Dict, Optional, Set

import socketio

from .config import settings

# ORIGIN ENFORCEMENT IS THIS SERVER'S JOB AND NOTHING ELSE'S (MIS-E2E-105).
#
# The removed comment said "CORS is handled by FastAPI's CORSMiddleware in
# main.py". `main.py` says the opposite — it deliberately installs no
# `CORSMiddleware` — and nginx's `/ws/` block only `add_header`s, which sets a
# response header and blocks nothing. So the wildcard was the ONLY policy in
# force, and engineio short-circuits the check entirely on "*"
# (`base_server.py`: `elif self.cors_allowed_origins == '*': allowed_origins =
# None`), with its own comment noting this matters MORE for WebSocket, because
# browsers do not apply CORS controls to it.
#
# That escaped the accepted no-app-auth posture. "Anyone who can reach the host
# can read the API" is conceded; "any website an operator visits can reach the
# host" is not. A page in the operator's browser could open a socket and
# `subscribe` to `labeling/{job_id}/results`, which carries verbatim corpus
# text, or `steering/{task_id}`, which carries generated model output.
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.allowed_origins,
    logger=settings.is_development,
    engineio_logger=settings.is_development,
    ping_interval=settings.websocket_ping_interval,
    ping_timeout=settings.websocket_ping_timeout,
    max_http_buffer_size=1_000_000,  # 1MB cap — prevents unbounded binary attachment DoS
)

# ASGI app for mounting in FastAPI
# NOTE: When mounted at "/ws" in FastAPI, Starlette's Mount strips the prefix.
# Using socketio_path="" (empty string) allows the ASGIApp to handle all paths
# under /ws/*, which is necessary because the frontend connects with path
# "/ws/socket.io". With empty string, any path under the mount point works.
socket_app = socketio.ASGIApp(
    sio,
    socketio_path="",
)


# The channel vocabulary, declared in ONE place.
#
# MIS-E2E-140: `subscribe` took the client's string unchecked — a list raised
# `TypeError: unhashable type`, a non-dict payload raised `AttributeError`, and
# a reviewer created 50,000 channels from an unauthenticated client.
#
# A first-segment allow-list rather than a full pattern per channel: it is the
# part that decides WHAT KIND of data the subscriber receives, so it is the part
# worth constraining, and it does not have to be re-edited every time an id
# format changes. `tests/unit/test_websocket_boundary.py` asserts that every
# channel the emitters actually publish is accepted here — so this list cannot
# drift narrower than production without a red test.
_CHANNEL_TOPICS: frozenset[str] = frozenset({
    "datasets",
    "trainings",
    "extraction",
    "extractions",
    "models",
    "sae",
    "labeling",
    "enhanced_labeling",
    "nlp_analysis",
    "neuronpedia",
    "steering",
    "system",
    "mcp",
})

# Segments are conservative on purpose: no "..", no "*", no whitespace, no
# empty segment. Ids in this product are slugs, uuids and step numbers.
_CHANNEL_RE = re.compile(r"^[a-z][a-z0-9_]*(?:/[A-Za-z0-9_.:-]+)*$")

MAX_CHANNEL_LENGTH = 200
MAX_SUBSCRIPTIONS_PER_CLIENT = 100


class InvalidChannel(ValueError):
    """A subscribe request named something that is not a channel."""


def validate_channel(channel: Any) -> str:
    """Return `channel` if it is a well-formed, known-topic channel name.

    Raises `InvalidChannel` otherwise. Type-checking is part of the job: the
    handler used to hand whatever arrived straight to a `set.add`.
    """
    if not isinstance(channel, str):
        raise InvalidChannel(f"channel must be a string, got {type(channel).__name__}")
    if not channel or len(channel) > MAX_CHANNEL_LENGTH:
        raise InvalidChannel("channel is empty or too long")
    if ".." in channel:
        raise InvalidChannel("channel may not contain '..'")
    if not _CHANNEL_RE.match(channel):
        raise InvalidChannel(f"channel {channel!r} is not a well-formed channel name")
    topic = channel.split("/", 1)[0]
    if topic not in _CHANNEL_TOPICS:
        raise InvalidChannel(f"unknown channel topic {topic!r}")
    return channel


class WebSocketManager:
    """
    Manager for WebSocket connections and event broadcasting.

    Handles channel subscriptions, event emission, and connection lifecycle.
    """

    def __init__(self):
        """Initialize WebSocket manager with empty channel subscriptions."""
        # Track which clients are subscribed to which channels
        # Format: {channel_name: {sid1, sid2, ...}}
        self.subscriptions: Dict[str, Set[str]] = {}

    async def connect(self, sid: str, environ: dict) -> None:
        """
        Handle client connection.

        Args:
            sid: Session ID of connected client
            environ: ASGI environ dict with connection info

        Notes:
            Called automatically by Socket.IO on client connect
        """
        if settings.is_development:
            print(f"WebSocket client connected: {sid}")

    async def disconnect(self, sid: str) -> None:
        """
        Handle client disconnection.

        Args:
            sid: Session ID of disconnected client

        Notes:
            - Automatically removes client from all subscriptions
            - Called automatically by Socket.IO on client disconnect
        """
        # Remove from all channels
        for channel in list(self.subscriptions.keys()):
            if sid in self.subscriptions[channel]:
                self.subscriptions[channel].remove(sid)

            # Clean up empty channels
            if not self.subscriptions[channel]:
                del self.subscriptions[channel]

        if settings.is_development:
            print(f"WebSocket client disconnected: {sid}")

    async def subscribe(self, sid: str, channel: str) -> None:
        """
        Subscribe client to a channel.

        Args:
            sid: Session ID of client
            channel: Channel name to subscribe to

        Usage:
            Channel naming conventions:
            - 'datasets/{id}/progress' - Dataset download/processing progress
            - 'trainings/{id}/progress' - Training job progress
            - 'extractions/{id}/progress' - Feature extraction progress
            - 'system' - System-wide notifications

        Notes:
            Clients automatically join Socket.IO room for the channel
        """
        # Validated here, in the manager, not only in the event handler —
        # `emit_event` and any future caller reach this method too, and a guard
        # that only sits on one entry point is this audit's most repeated
        # finding. `validate_channel` is idempotent, so double-checking costs
        # nothing.
        channel = validate_channel(channel)

        # Cap per client (MIS-E2E-140). Without it one unauthenticated
        # connection grew the registry to 50,000 channels.
        held = sum(1 for subs in self.subscriptions.values() if sid in subs)
        if channel not in self.subscriptions.get(channel, set()) and held >= MAX_SUBSCRIPTIONS_PER_CLIENT:
            raise InvalidChannel(
                f"subscription limit reached ({MAX_SUBSCRIPTIONS_PER_CLIENT})"
            )

        # Create channel if doesn't exist
        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()

        # Add client to channel
        self.subscriptions[channel].add(sid)

        # Join Socket.IO room for efficient broadcasting
        await sio.enter_room(sid, channel)

        if settings.is_development:
            print(f"Client {sid} subscribed to channel: {channel}")

    async def unsubscribe(self, sid: str, channel: str) -> None:
        """
        Unsubscribe client from a channel.

        Args:
            sid: Session ID of client
            channel: Channel name to unsubscribe from
        """
        if channel in self.subscriptions and sid in self.subscriptions[channel]:
            self.subscriptions[channel].remove(sid)

            # Leave Socket.IO room
            await sio.leave_room(sid, channel)

            # Clean up empty channels
            if not self.subscriptions[channel]:
                del self.subscriptions[channel]

            if settings.is_development:
                print(f"Client {sid} unsubscribed from channel: {channel}")

    async def emit_event(
        self,
        channel: str,
        event: str,
        data: Dict[str, Any],
        namespace: str = "/",
    ) -> None:
        """
        Emit event to all subscribers of a channel.

        Args:
            channel: Channel name to emit to
            event: Event name (e.g., 'progress', 'completed', 'error')
            data: Event data payload
            namespace: Socket.IO namespace (default: '/')

        Usage:
            ```python
            await ws_manager.emit_event(
                channel='datasets/ds_123/progress',
                event='progress',
                data={
                    'progress': 45.5,
                    'status': 'downloading',
                    'message': 'Downloading dataset...'
                }
            )
            ```

        Notes:
            - Uses Socket.IO rooms for efficient broadcasting
            - Only subscribed clients receive events
            - Non-blocking operation
            - Events are always emitted; Socket.IO handles delivery to room members
        """
        # Always emit to Socket.IO room - the room system handles delivery
        # If no one is subscribed to the room, the event is simply not delivered
        #
        # Use the actual event name for Socket.IO emission - this allows
        # the frontend to listen for specific event types (e.g., 'system:metrics')
        # across multiple channels. The room mechanism ensures only subscribers
        # receive the event.
        await sio.emit(
            event,  # Use actual event name (e.g., 'system:metrics', 'progress', etc.)
            data,
            room=channel,
            namespace=namespace,
        )

        if settings.is_development:
            subscriber_count = len(self.subscriptions.get(channel, set()))
            print(f"Emitted '{event}' to channel '{channel}' ({subscriber_count} subscribers): {data}")

    async def broadcast(
        self,
        event: str,
        data: Dict[str, Any],
        namespace: str = "/",
        exclude_sid: Optional[str] = None,
    ) -> None:
        """
        Broadcast event to all connected clients.

        Args:
            event: Event name
            data: Event data payload
            namespace: Socket.IO namespace (default: '/')
            exclude_sid: Optional session ID to exclude from broadcast

        Usage:
            ```python
            await ws_manager.broadcast(
                event='system_notification',
                data={
                    'type': 'warning',
                    'message': 'System maintenance in 5 minutes'
                }
            )
            ```

        Notes:
            - Sends to ALL connected clients regardless of subscriptions
            - Use for system-wide notifications
            - Use emit_event() for channel-specific events
        """
        await sio.emit(
            event,
            data,
            namespace=namespace,
            skip_sid=exclude_sid,
        )

        if settings.is_development:
            print(f"Broadcasted '{event}' to all clients: {data}")

    async def get_subscriptions(self, sid: str) -> list[str]:
        """
        Get all channels a client is subscribed to.

        Args:
            sid: Session ID of client

        Returns:
            list[str]: List of channel names
        """
        return [
            channel
            for channel, sids in self.subscriptions.items()
            if sid in sids
        ]

    async def get_subscribers(self, channel: str) -> list[str]:
        """
        Get all clients subscribed to a channel.

        Args:
            channel: Channel name

        Returns:
            list[str]: List of session IDs
        """
        return list(self.subscriptions.get(channel, set()))

    async def channel_exists(self, channel: str) -> bool:
        """
        Check if a channel has any subscribers.

        Args:
            channel: Channel name

        Returns:
            bool: True if channel has subscribers
        """
        return channel in self.subscriptions and bool(self.subscriptions[channel])


# Global WebSocket manager instance
ws_manager = WebSocketManager()


# Socket.IO event handlers
@sio.event
async def connect(sid: str, environ: dict):
    """Handle client connection."""
    await ws_manager.connect(sid, environ)


@sio.event
async def disconnect(sid: str):
    """Handle client disconnection."""
    await ws_manager.disconnect(sid)


@sio.event
async def subscribe(sid: str, data: dict):
    """
    Handle client subscription request.

    Expected data format:
        {
            "channel": "datasets/ds_123/progress"
        }

    THIS IS THE ONLY REGISTRATION OF THIS EVENT (MIS-E2E-138). `main.py` used to
    register its own `subscribe`/`unsubscribe`/`connect`/`disconnect` against
    the same `sio`, and python-socketio silently overwrites on re-registration —
    so `main.py` won, these never ran, the `subscribed`/`unsubscribed`
    acknowledgements the frontend waits for never fired, and the `__all__`-
    exported `ws_manager` singleton was never populated. Do not re-register
    these anywhere else; `tests/unit/test_websocket_boundary.py` fails if you do.
    """
    if not isinstance(data, dict):
        await sio.emit(
            "subscribe_error", {"error": "payload must be an object"}, room=sid
        )
        return
    try:
        channel = validate_channel(data.get("channel"))
    except InvalidChannel as exc:
        # Tell the client, and do not disconnect: a malformed payload used to
        # raise out of the handler.
        await sio.emit("subscribe_error", {"error": str(exc)}, room=sid)
        return
    try:
        await ws_manager.subscribe(sid, channel)
    except InvalidChannel as exc:
        await sio.emit("subscribe_error", {"error": str(exc)}, room=sid)
        return
    await sio.emit("subscribed", {"channel": channel}, room=sid)


@sio.event
async def unsubscribe(sid: str, data: dict):
    """
    Handle client unsubscription request.

    Expected data format:
        {
            "channel": "datasets/ds_123/progress"
        }
    """
    if not isinstance(data, dict):
        await sio.emit(
            "unsubscribe_error", {"error": "payload must be an object"}, room=sid
        )
        return
    channel = data.get("channel")
    # Unsubscribe is not a privilege escalation — leaving a room you are not in
    # is a no-op — but it must not crash on a malformed payload either.
    if not isinstance(channel, str) or not channel:
        await sio.emit(
            "unsubscribe_error", {"error": "channel must be a non-empty string"},
            room=sid,
        )
        return
    await ws_manager.unsubscribe(sid, channel)
    await sio.emit("unsubscribed", {"channel": channel}, room=sid)


@sio.event
async def ping(sid: str):
    """Handle ping request from client."""
    await sio.emit("pong", room=sid)


# Export commonly used objects
__all__ = [
    "sio",
    "socket_app",
    "ws_manager",
]
