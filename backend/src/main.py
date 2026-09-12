"""
MechInterp Studio (miStudio) - FastAPI Application

Main application entry point for the backend API.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import hmac

from typing import Optional

from fastapi import FastAPI, Header, HTTPException

from .api.v1.router import api_router
from .core.config import settings
from .core.database import get_db
from .core.body_limit import BodySizeLimitMiddleware
from .core.websocket import socket_app, sio, ws_manager
from .db.schema_validator import validate_schema_on_startup
from .ml.transformers_compat import patch_transformers_compatibility
from .services.background_monitor import get_background_monitor

logger = logging.getLogger(__name__)

# Apply transformers compatibility patches for newer models (Phi-4, etc.)
patch_transformers_compatibility()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Starts background tasks on startup and stops them on shutdown.
    """
    # Startup
    logger.info("Starting miStudio backend...")

    # Validate database schema
    logger.info("Validating database schema...")
    try:
        async for db in get_db():
            is_valid = await validate_schema_on_startup(db)
            if not is_valid:
                logger.warning(
                    "Schema validation failed - some features may not work correctly. "
                    "Run 'alembic upgrade head' to apply missing migrations."
                )
            break
    except Exception as e:
        logger.error(f"Schema validation encountered an error: {e}")
        # Continue startup even if validation fails to avoid blocking deployment

    # Start background system monitor (runs independently of Celery)
    background_monitor = get_background_monitor()
    await background_monitor.start()

    yield

    # Shutdown
    logger.info("Shutting down miStudio backend...")

    # Stop background monitor
    await background_monitor.stop()


app = FastAPI(
    title="MechInterp Studio API",
    description="Edge-deployed mechanistic interpretability platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Initialize WebSocket manager
# MIS-E2E-138: use the SINGLETON from core.websocket, not a second instance.
# This module used to construct its own `WebSocketManager()`, so the
# `__all__`-exported one that other modules import was permanently empty —
# including the direct-emit path.

# MIS-E2E-036: a body-size ceiling for every route.
# It sits in middleware rather than in each handler because FastAPI parses the
# body before a handler's first statement runs — a check inside the handler
# rejects the response, never the cost — and because a `Content-Length`-only
# check is skipped entirely by a chunked request.
app.add_middleware(BodySizeLimitMiddleware)

# Mount Socket.IO app
app.mount("/ws", socket_app)

# NOTE: CORS is handled by nginx reverse proxy
# Do not add CORSMiddleware here as it will create duplicate headers

# Include API router
app.include_router(api_router, prefix="/api")


# Socket.IO event handlers live in `core/websocket.py` and are registered by
# importing `sio` above. They used to be duplicated here — python-socketio
# silently overwrites a re-registered event, so THESE won and the hardened ones
# never ran: no `subscribed`/`unsubscribed` acks, no channel validation, and a
# second manager instance that left the exported singleton empty (MIS-E2E-138).
# Do not re-add them; `tests/unit/test_websocket_boundary.py` fails if you do.


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "miStudio Backend API",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "MechInterp Studio API",
        "docs": "/api/docs"
    }


@app.post("/api/internal/ws/emit")
async def emit_websocket_event(
    request: dict,
    x_internal_token: Optional[str] = Header(None, alias="X-Internal-Token"),
):
    """Internal endpoint for Celery workers to emit WebSocket events.

    Protected by a shared secret derived from SECRET_KEY. Nginx also
    blocks this path from external traffic via `deny all`.
    Missing or incorrect token always returns 403 — never 422 — so the
    existence of the header requirement is not disclosed to callers.
    """
    if x_internal_token is None or not hmac.compare_digest(x_internal_token, settings.internal_api_secret):
        raise HTTPException(status_code=403, detail="Forbidden")

    channel = request.get("channel")
    event = request.get("event")
    data = request.get("data")

    # data may legitimately be an empty dict; only reject if absent
    if not channel or not event or data is None:
        raise HTTPException(status_code=400, detail="Missing required fields")

    await ws_manager.emit_event(channel=channel, event=event, data=data)

    return {"status": "ok"}


@app.post("/api/internal/steering/reconcile-worker")
async def reconcile_steering_worker(
    x_internal_token: Optional[str] = Header(None, alias="X-Internal-Token"),
):
    """Respawn the steering worker when tasks are stranded without one.

    Called by the steering_worker_reconcile beat task. The dedicated
    steering worker exits after each generation task (solo pool ignores
    max-tasks-per-child), and any death mode — self-exit, crash, kill —
    can leave queued/restored messages with no consumer. This closes that
    gap: steering queue non-empty + no live worker => spawn one.
    Same auth/exposure model as /api/internal/ws/emit.
    """
    if x_internal_token is None or not hmac.compare_digest(x_internal_token, settings.internal_api_secret):
        raise HTTPException(status_code=403, detail="Forbidden")

    from .api.v1.endpoints.steering import (
        _ensure_steering_worker_running,
        _is_steering_worker_running,
        _steering_queue_depth,
    )

    depth = await _steering_queue_depth()
    if depth <= 0:
        return {"status": "ok", "action": "none", "queue_depth": depth}

    is_active, pid = await asyncio.to_thread(_is_steering_worker_running)
    if is_active:
        return {"status": "ok", "action": "none", "queue_depth": depth, "worker_pid": pid}

    ok, new_pid = await _ensure_steering_worker_running()
    return {
        "status": "ok" if ok else "error",
        "action": "spawned" if ok else "spawn_failed",
        "queue_depth": depth,
        "worker_pid": new_pid,
    }
