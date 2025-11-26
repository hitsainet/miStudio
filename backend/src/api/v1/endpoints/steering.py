"""
Steering API endpoints.

This module defines REST API endpoints for model steering operations including:
- Generating steered and unsteered text comparisons
- Running strength sweeps to test different steering intensities
- Managing steering experiments
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/steering", tags=["Steering"])


@router.post("/compare", response_model=SteeringComparisonResponse)
async def generate_steering_comparison(
    request: SteeringComparisonRequest,
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
    """
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

    # Check if it's a local model
    model_path = None
    local_model_path = settings.data_dir / "models" / model_id
    if local_model_path.exists():
        model_path = str(local_model_path)

    # Get steering service
    steering_service = get_steering_service()

    try:
        response = await steering_service.generate_comparison(
            request=request,
            sae_path=sae_path,
            model_id=model_id,
            model_path=model_path,
        )
        return response

    except Exception as e:
        logger.exception(f"Error generating steering comparison: {e}")
        raise HTTPException(500, f"Steering generation failed: {str(e)}")


@router.post("/sweep", response_model=StrengthSweepResponse)
async def generate_strength_sweep(
    request: SteeringStrengthSweepRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a strength sweep testing multiple steering intensities.

    Useful for finding the optimal steering strength for a feature.
    Tests a single feature at multiple strength values and returns
    outputs for comparison.
    """
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

    # Check if it's a local model
    model_path = None
    local_model_path = settings.data_dir / "models" / model_id
    if local_model_path.exists():
        model_path = str(local_model_path)

    # Get steering service
    steering_service = get_steering_service()

    try:
        response = await steering_service.generate_strength_sweep(
            request=request,
            sae_path=sae_path,
            model_id=model_id,
            model_path=model_path,
        )
        return response

    except Exception as e:
        logger.exception(f"Error generating strength sweep: {e}")
        raise HTTPException(500, f"Strength sweep failed: {str(e)}")


@router.post("/cache/clear")
async def clear_steering_cache():
    """
    Clear all cached models and SAEs from the steering service.

    Use this to free GPU memory when switching between models.
    """
    steering_service = get_steering_service()
    steering_service.clear_cache()
    return {"message": "Cache cleared successfully"}


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
