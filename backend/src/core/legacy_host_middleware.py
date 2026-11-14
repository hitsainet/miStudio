"""Records a request that arrived on an OLD host name (042 FR-042.9; FTID §3.5; TPD-4; TI-9).

A pure ASGI middleware (no Starlette base class, so it costs one dict lookup per request): it reads the
`host` header of `http` and `websocket` scopes, asks `legacy_names.host_is_old()`, and records
`("host", old, new)`. It never blocks, never raises and never changes the request. In the MCP app it sits
OUTSIDE the bearer-token check, so a refused request on an old host still counts: a request that arrived
is evidence of use. Only the `Host` header is read (not `X-Forwarded-Host`, which any client can set).
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from src.core import legacy_names

logger = logging.getLogger(__name__)

Scope = dict[str, Any]
ASGIApp = Callable[[Scope, Callable[..., Awaitable[Any]], Callable[..., Awaitable[Any]]], Awaitable[None]]


class LegacyHostMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive, send) -> None:
        if scope.get("type") in ("http", "websocket"):
            try:
                for name, value in scope.get("headers") or ():
                    if name == b"host":
                        host = value.decode("latin-1")
                        new = legacy_names.host_is_old(host)
                        if new is not None:
                            old = host.strip().lower().rsplit(":", 1)[0].rstrip(".")
                            legacy_names.record("host", old, new)
                        break
            except Exception as exc:  # noqa: BLE001 — evidence never breaks a request
                logger.debug("legacy host middleware: %s", type(exc).__name__)
        await self.app(scope, receive, send)
