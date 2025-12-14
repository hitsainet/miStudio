"""
oLLM Server - FastAPI Application

Provides OpenAI-compatible REST API endpoints for chat completions,
model management, and health checks.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 11434

Or as a module:
    python -m services.ollm_server.main
"""

import logging
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from .config import settings
from .inference import inference_engine, ModelTooLargeError, InferenceTimeoutError
from .models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelListResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info(f"Starting oLLM Server v0.1.0")
    logger.info(f"Host: {settings.host}:{settings.port}")
    logger.info(f"Default model: {settings.default_model}")
    logger.info(f"GPU available: {inference_engine.gpu_available}")

    # Optionally preload default model
    if settings.default_model:
        try:
            logger.info(f"Preloading default model: {settings.default_model}")
            await inference_engine.load_model(settings.default_model)
        except Exception as e:
            logger.warning(f"Failed to preload default model: {e}")

    yield

    # Shutdown
    logger.info("Shutting down oLLM Server")
    await inference_engine.unload_model()


# Create FastAPI app
app = FastAPI(
    title="oLLM Server",
    description="OpenAI-compatible API wrapper for oLLM inference",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Error Handling
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions in OpenAI format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse.create(
            message=str(exc.detail),
            type="invalid_request_error" if exc.status_code == 400 else "server_error",
        ).model_dump(),
    )


@app.exception_handler(ModelTooLargeError)
async def model_too_large_handler(request: Request, exc: ModelTooLargeError):
    """Handle model too large errors with helpful suggestions"""
    logger.error(f"Model too large: {exc}")
    return JSONResponse(
        status_code=400,
        content=ErrorResponse.create(
            message=str(exc),
            type="model_too_large",
            code="model_too_large",
        ).model_dump(),
    )


@app.exception_handler(InferenceTimeoutError)
async def inference_timeout_handler(request: Request, exc: InferenceTimeoutError):
    """Handle inference timeout errors"""
    logger.error(f"Inference timeout: {exc}")
    return JSONResponse(
        status_code=504,
        content=ErrorResponse.create(
            message=str(exc),
            type="timeout_error",
            code="inference_timeout",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse.create(
            message="Internal server error",
            type="server_error",
        ).model_dump(),
    )


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=inference_engine.model_loaded,
        current_model=inference_engine.current_model,
        gpu_available=inference_engine.gpu_available,
        memory_usage=inference_engine.get_memory_usage(),
    )


# ============================================================================
# Model Management Endpoints (OpenAI-compatible)
# ============================================================================

@app.get("/v1/models", response_model=ModelListResponse)
@app.get("/api/tags", response_model=ModelListResponse)  # Ollama compatibility
async def list_models():
    """List available models from cache directory"""
    import os
    models = []
    seen = set()

    # Add currently loaded model first
    if inference_engine.model_loaded:
        models.append(ModelInfo(
            id=inference_engine.current_model,
            owned_by="ollm",
        ))
        seen.add(inference_engine.current_model)

    # Scan cache directory for downloaded models
    cache_dir = settings.model_cache_dir
    if os.path.exists(cache_dir):
        # HuggingFace cache structure: models--org--model_name
        for item in os.listdir(cache_dir):
            if item.startswith("models--"):
                # Convert models--google--gemma-3-4b-it to google/gemma-3-4b-it
                parts = item.replace("models--", "").split("--")
                if len(parts) >= 2:
                    model_id = "/".join(parts)
                    if model_id not in seen:
                        models.append(ModelInfo(
                            id=model_id,
                            owned_by="huggingface",
                        ))
                        seen.add(model_id)

    # Add default model if not already listed
    if settings.default_model and settings.default_model not in seen:
        models.append(ModelInfo(
            id=settings.default_model,
            owned_by="ollm",
        ))

    return ModelListResponse(data=models)


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    """Get information about a specific model"""
    return ModelInfo(
        id=model_id,
        owned_by="ollm",
    )


@app.post("/api/pull")  # Ollama compatibility - pull/load a model
async def pull_model(request: Request):
    """Pull/load a model (Ollama compatibility endpoint)"""
    body = await request.json()
    model_name = body.get("name") or body.get("model")

    if not model_name:
        raise HTTPException(status_code=400, detail="Model name required")

    try:
        await inference_engine.load_model(model_name)
        return {"status": "success", "model": model_name}
    except ModelTooLargeError as e:
        # Re-raise to be handled by specific exception handler
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Chat Completions Endpoint (OpenAI-compatible)
# ============================================================================

@app.post("/v1/chat/completions")
@app.post("/api/chat")  # Ollama compatibility
async def chat_completions(request: ChatCompletionRequest):
    """
    Create a chat completion.

    This is the main endpoint for generating responses from the LLM.
    Supports both streaming and non-streaming responses.
    """
    logger.info(f"Chat completion request: model={request.model}, stream={request.stream}")

    try:
        if request.stream:
            # Streaming response
            async def generate_stream():
                try:
                    async for chunk in inference_engine.generate_stream(request):
                        data = chunk.model_dump_json()
                        yield f"data: {data}\n\n"
                    yield "data: [DONE]\n\n"
                except ModelTooLargeError as e:
                    logger.error(f"Model too large for streaming: {e}")
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "model_too_large",
                            "code": "model_too_large",
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )
        else:
            # Non-streaming response
            response = await inference_engine.generate(request)
            return response

    except ModelTooLargeError as e:
        # Re-raise to be handled by specific exception handler
        raise
    except InferenceTimeoutError as e:
        # Re-raise to be handled by specific exception handler
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Legacy/Compatibility Endpoints
# ============================================================================

@app.post("/api/generate")  # Ollama compatibility - text generation
async def generate(request: Request):
    """Generate text (Ollama compatibility endpoint)"""
    body = await request.json()

    # Convert Ollama format to OpenAI format
    model = body.get("model", settings.default_model)
    prompt = body.get("prompt", "")
    system = body.get("system", "")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    chat_request = ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(**m) for m in messages],
        temperature=body.get("options", {}).get("temperature", settings.default_temperature),
        max_tokens=body.get("options", {}).get("num_predict", settings.default_max_tokens),
        stream=body.get("stream", False),
    )

    return await chat_completions(chat_request)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the server"""
    uvicorn.run(
        "services.ollm_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
