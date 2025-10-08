"""
MechInterp Studio (miStudio) - FastAPI Application

Main application entry point for the backend API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1.router import api_router
from .core.config import settings
from .core.websocket import socket_app, sio, WebSocketManager

app = FastAPI(
    title="MechInterp Studio API",
    description="Edge-deployed mechanistic interpretability platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Initialize WebSocket manager
ws_manager = WebSocketManager()

# Mount Socket.IO app
app.mount("/ws", socket_app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
