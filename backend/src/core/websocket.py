"""
WebSocket manager for real-time communication using Socket.IO.

This module provides a WebSocket manager for broadcasting progress updates,
training metrics, and other real-time events to connected clients.
"""

from typing import Any, Dict, Optional, Set

import socketio

from .config import settings

# Create Socket.IO server with async support
# NOTE: CORS is handled by FastAPI's CORSMiddleware in main.py
# Setting cors_allowed_origins="*" here prevents duplicate CORS headers
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",  # Let FastAPI handle CORS validation
    logger=settings.is_development,
    engineio_logger=settings.is_development,
    ping_interval=settings.websocket_ping_interval,
    ping_timeout=settings.websocket_ping_timeout,
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
    """
    channel = data.get("channel")
    if channel:
        await ws_manager.subscribe(sid, channel)
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
    channel = data.get("channel")
    if channel:
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
