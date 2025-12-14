"""
MechInterp Studio (miStudio) - FastAPI Application

Main application entry point for the backend API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.v1.router import api_router
from .core.config import settings
from .core.websocket import socket_app, sio, WebSocketManager
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
ws_manager = WebSocketManager()

# Mount Socket.IO app
app.mount("/ws", socket_app)

# NOTE: CORS is handled by nginx reverse proxy
# Do not add CORSMiddleware here as it will create duplicate headers

# Include API router
app.include_router(api_router, prefix="/api")


# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    await ws_manager.connect(sid, environ)


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    await ws_manager.disconnect(sid)


@sio.event
async def subscribe(sid, data):
    """Handle channel subscription."""
    channel = data.get("channel")
    if channel:
        await ws_manager.subscribe(sid, channel)


@sio.event
async def unsubscribe(sid, data):
    """Handle channel unsubscription."""
    channel = data.get("channel")
    if channel:
        await ws_manager.unsubscribe(sid, channel)


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
async def emit_websocket_event(request: dict):
    """
    Internal endpoint for Celery workers to emit WebSocket events.

    This endpoint should only be called from within the backend system.
    """
    channel = request.get("channel")
    event = request.get("event")
    data = request.get("data")

    if not all([channel, event, data]):
        return {"error": "Missing required fields"}, 400

    await ws_manager.emit_event(channel=channel, event=event, data=data)

    return {"status": "ok"}
