"""
Steering API endpoints.

This module defines REST API endpoints for model steering operations including:
- Generating steered and unsteered text comparisons
- Running strength sweeps to test different steering intensities
- Managing steering experiments
"""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....core.config import settings
from ....models.external_sae import SAEStatus
from ....schemas.steering import (
    SteeringComparisonRequest,
    SteeringComparisonResponse,
    SteeringStrengthSweepRequest,
    StrengthSweepResponse,
)
from ....services.sae_manager_service import SAEManagerService
from ....services.steering_service import get_steering_service
from ....services.model_service import ModelService

logger = logging.getLogger(__name__)

# Steering configuration
STEERING_TIMEOUT_SECONDS = getattr(settings, 'steering_timeout_seconds', 30)
RATE_LIMIT_REQUESTS = 5  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds


class RateLimiter:
    """Simple in-memory rate limiter per client IP."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make a request."""
        now = time.time()
        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id]
            if now - t < self.window_seconds
        ]
        # Check limit
        if len(self._requests[client_id]) >= self.max_requests:
            return False
        # Record request
        self._requests[client_id].append(now)
        return True

    def time_until_allowed(self, client_id: str) -> float:
        """Get seconds until client can make another request."""
        if not self._requests[client_id]:
            return 0
        oldest = min(self._requests[client_id])
        return max(0, self.window_seconds - (time.time() - oldest))


# Global rate limiter for steering endpoints
_rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)


def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Use X-Forwarded-For if behind proxy, otherwise use client host
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


router = APIRouter(prefix="/steering", tags=["Steering"])


@router.post("/compare", response_model=SteeringComparisonResponse)
async def generate_steering_comparison(
    request: SteeringComparisonRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a steering comparison with steered and unsteered outputs.

    Takes a prompt, SAE, and selected features with steering strengths.
    Returns both unsteered baseline and steered outputs for comparison.

    Steering strength interpretation:
    - -100: Full suppression (0x activation)
    - 0: No change (1x activation)
    - +100: Double activation (2x)
    - +200: Triple activation (3x)
    - +300: Quadruple activation (4x)

    Rate limited to 5 requests per minute per client.
    Times out after 30 seconds.
    """
    # Rate limiting
    client_id = get_client_id(http_request)
    if not _rate_limiter.is_allowed(client_id):
        retry_after = int(_rate_limiter.time_until_allowed(client_id)) + 1
        raise HTTPException(
            429,
            f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    # Get SAE from database
    sae = await SAEManagerService.get_sae(db, request.sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {request.sae_id}")

    if sae.status != SAEStatus.READY.value:
        raise HTTPException(400, f"SAE is not ready: {sae.status}")

    if not sae.local_path:
        raise HTTPException(400, "SAE has no local path")

    sae_path = Path(sae.local_path)
    if not sae_path.exists():
        raise HTTPException(400, f"SAE path does not exist: {sae.local_path}")

    # Determine model to use
    model_id = request.model_id
    if not model_id:
        # Try to use SAE's linked model
        if sae.model_id:
            model_id = sae.model_id
        elif sae.model_name:
            model_id = sae.model_name
        else:
            raise HTTPException(
                400,
                "No model specified and SAE has no linked model. "
                "Please provide a model_id in the request."
            )

    # Look up model from database to get actual file_path
    model_path = None
    model = await ModelService.get_model(db, model_id)
    if model and model.file_path:
        model_path = model.file_path
        # Use model name or repo_id as the identifier for HF loading
        model_id = model.repo_id or model.name

    # Get steering service
    steering_service = get_steering_service()

    try:
        response = await asyncio.wait_for(
            steering_service.generate_comparison(
                request=request,
                sae_path=sae_path,
                model_id=model_id,
                model_path=model_path,
                # Pass SAE metadata from database for miStudio format SAEs
                sae_layer=sae.layer,
                sae_d_model=sae.d_model,
                sae_n_features=sae.n_features,
                sae_architecture=sae.architecture,
            ),
            timeout=STEERING_TIMEOUT_SECONDS,
        )
        return response

    except asyncio.TimeoutError:
        logger.warning(f"Steering comparison timed out after {STEERING_TIMEOUT_SECONDS}s")
        raise HTTPException(
            408,
            f"Generation timed out after {STEERING_TIMEOUT_SECONDS} seconds. "
            "Try reducing max_new_tokens or using fewer features.",
        )

    except Exception as e:
        logger.exception(f"Error generating steering comparison: {e}")
        raise HTTPException(500, f"Steering generation failed: {str(e)}")


@router.post("/sweep", response_model=StrengthSweepResponse)
async def generate_strength_sweep(
    request: SteeringStrengthSweepRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a strength sweep testing multiple steering intensities.

    Useful for finding the optimal steering strength for a feature.
    Tests a single feature at multiple strength values and returns
    outputs for comparison.

    Rate limited to 5 requests per minute per client.
    Times out after 30 seconds.
    """
    # Rate limiting
    client_id = get_client_id(http_request)
    if not _rate_limiter.is_allowed(client_id):
        retry_after = int(_rate_limiter.time_until_allowed(client_id)) + 1
        raise HTTPException(
            429,
            f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )
    # Get SAE from database
    sae = await SAEManagerService.get_sae(db, request.sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {request.sae_id}")

    if sae.status != SAEStatus.READY.value:
        raise HTTPException(400, f"SAE is not ready: {sae.status}")

    if not sae.local_path:
        raise HTTPException(400, "SAE has no local path")

    sae_path = Path(sae.local_path)
    if not sae_path.exists():
        raise HTTPException(400, f"SAE path does not exist: {sae.local_path}")

    # Determine model to use
    model_id = request.model_id
    if not model_id:
        if sae.model_id:
            model_id = sae.model_id
        elif sae.model_name:
            model_id = sae.model_name
        else:
            raise HTTPException(
                400,
                "No model specified and SAE has no linked model."
            )

    # Look up model from database to get actual file_path
    model_path = None
    model = await ModelService.get_model(db, model_id)
    if model and model.file_path:
        model_path = model.file_path
        # Use model name or repo_id as the identifier for HF loading
        model_id = model.repo_id or model.name

    # Get steering service
    steering_service = get_steering_service()

    try:
        response = await asyncio.wait_for(
            steering_service.generate_strength_sweep(
                request=request,
                sae_path=sae_path,
                model_id=model_id,
                model_path=model_path,
                # Pass SAE metadata from database for miStudio format SAEs
                sae_layer=sae.layer,
                sae_d_model=sae.d_model,
                sae_n_features=sae.n_features,
                sae_architecture=sae.architecture,
            ),
            timeout=STEERING_TIMEOUT_SECONDS,
        )
        return response

    except asyncio.TimeoutError:
        logger.warning(f"Strength sweep timed out after {STEERING_TIMEOUT_SECONDS}s")
        raise HTTPException(
            408,
            f"Generation timed out after {STEERING_TIMEOUT_SECONDS} seconds. "
            "Try reducing the number of strength values or max_new_tokens.",
        )

    except Exception as e:
        logger.exception(f"Error generating strength sweep: {e}")
        raise HTTPException(500, f"Strength sweep failed: {str(e)}")


@router.post("/cache/clear")
async def clear_steering_cache():
    """
    Clear all cached models and SAEs from the steering service and free GPU memory.

    This aggressively clears ALL GPU memory from this process, not just steering models.
    Use this to free GPU memory when switching between models or if memory is low.

    Returns:
        Dict with cache clearing results including:
        - models_unloaded: Number of models unloaded
        - saes_unloaded: Number of SAEs unloaded
        - other_services_cleared: Number of stray models moved to CPU
        - vram_before_gb: VRAM usage before clearing (GB)
        - vram_after_gb: VRAM usage after clearing (GB)
        - vram_freed_gb: Amount of VRAM freed (GB)
        - was_already_clear: True if nothing needed to be cleared
        - message: Human-readable status message
    """
    steering_service = get_steering_service()
    result = steering_service.clear_cache()

    # Generate appropriate message
    total_unloaded = result["models_unloaded"] + result["saes_unloaded"]
    vram_freed = result["vram_freed_gb"]
    vram_after = result["vram_after_gb"]

    if vram_freed >= 0.1:
        result["message"] = f"Freed {vram_freed:.1f} GB VRAM (now {vram_after:.1f} GB in use)"
    elif vram_after < 1.0:
        result["message"] = f"GPU memory clear ({vram_after:.1f} GB baseline)"
    elif result.get("needs_restart"):
        result["message"] = f"Orphaned GPU memory detected ({vram_after:.1f} GB). Restart backend to fully clear."
    else:
        result["message"] = f"Cache cleared. {vram_after:.1f} GB still in use."

    return result


@router.delete("/cache/sae/{sae_id}")
async def unload_sae(sae_id: str):
    """
    Unload a specific SAE from the cache.
    """
    steering_service = get_steering_service()
    if steering_service.unload_sae(sae_id):
        return {"message": f"SAE {sae_id} unloaded"}
    raise HTTPException(404, f"SAE {sae_id} not in cache")


@router.delete("/cache/model/{model_id:path}")
async def unload_model(model_id: str):
    """
    Unload a specific model from the cache.
    """
    steering_service = get_steering_service()
    if steering_service.unload_model(model_id):
        return {"message": f"Model {model_id} unloaded"}
    raise HTTPException(404, f"Model {model_id} not in cache")
