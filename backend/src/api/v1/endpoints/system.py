"""
System Monitoring API Endpoints

This module provides REST API endpoints for system and GPU monitoring.

Endpoints:
- GET /system/gpu-list: List available GPUs
- GET /system/gpu-metrics: Get GPU metrics for specified GPU
- GET /system/gpu-info: Get static GPU information
- GET /system/gpu-processes: Get processes using specified GPU
- GET /system/metrics: Get system resource metrics

Author: miStudio Team
Created: 2025-10-16
"""

import logging
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from src.services.gpu_monitor_service import get_gpu_monitor_service
from src.services.system_monitor_service import get_system_monitor_service
from src.services.resource_config import ResourceConfig
from src.models.training import Training
from src.core.deps import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


# Response models
class GPUListResponse(BaseModel):
    """Response model for GPU list"""
    gpu_count: int = Field(..., description="Number of available GPUs")
    gpus: List[Dict[str, Any]] = Field(..., description="List of GPU information")


class GPUMetricsResponse(BaseModel):
    """Response model for GPU metrics"""
    gpu_id: int = Field(..., description="GPU index")
    metrics: Dict[str, Any] = Field(..., description="GPU metrics")


class GPUInfoResponse(BaseModel):
    """Response model for GPU information"""
    gpu_id: int = Field(..., description="GPU index")
    info: Dict[str, Any] = Field(..., description="Static GPU information")


class GPUProcessesResponse(BaseModel):
    """Response model for GPU processes"""
    gpu_id: int = Field(..., description="GPU index")
    process_count: int = Field(..., description="Number of active processes")
    processes: List[Dict[str, Any]] = Field(..., description="List of processes")


class SystemMetricsResponse(BaseModel):
    """Response model for system metrics"""
    metrics: Dict[str, Any] = Field(..., description="System resource metrics")


class ResourceEstimateResponse(BaseModel):
    """Response model for resource estimation"""
    system_resources: Dict[str, Any] = Field(..., description="Current system resource availability")
    recommended_settings: Dict[str, int] = Field(..., description="Recommended resource settings")
    current_settings: Dict[str, int] = Field(..., description="User-specified or default settings")
    resource_estimates: Dict[str, Any] = Field(..., description="Estimated resource usage and warnings")


# Endpoints
@router.get("/gpu-list", response_model=GPUListResponse)
async def get_gpu_list():
    """
    Get list of available GPUs.

    Returns:
        GPUListResponse containing GPU count and information for each GPU

    Raises:
        HTTPException 503: If GPU monitoring is not available
    """
    try:
        gpu_service = get_gpu_monitor_service()

        if not gpu_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="GPU monitoring is not available. Ensure nvidia-smi is installed and GPUs are accessible."
            )

        gpu_count = gpu_service.get_device_count()
        gpu_info_list = gpu_service.get_all_gpu_info()

        return GPUListResponse(
            gpu_count=gpu_count,
            gpus=[info.to_dict() for info in gpu_info_list]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get GPU list: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU list: {str(e)}")


@router.get("/gpu-metrics", response_model=GPUMetricsResponse)
async def get_gpu_metrics(
    gpu_id: int = Query(0, ge=0, description="GPU index to query")
):
    """
    Get current metrics for specified GPU.

    Args:
        gpu_id: GPU index (default: 0)

    Returns:
        GPUMetricsResponse containing current GPU metrics

    Raises:
        HTTPException 400: If invalid GPU ID
        HTTPException 503: If GPU monitoring not available
    """
    try:
        gpu_service = get_gpu_monitor_service()

        if not gpu_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="GPU monitoring is not available."
            )

        # Validate GPU ID
        if gpu_id >= gpu_service.get_device_count():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid GPU ID: {gpu_id}. Available GPUs: 0-{gpu_service.get_device_count() - 1}"
            )

        metrics = gpu_service.get_gpu_metrics(gpu_id)

        return GPUMetricsResponse(
            gpu_id=gpu_id,
            metrics=metrics.to_dict()
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get GPU metrics for GPU {gpu_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU metrics: {str(e)}")


@router.get("/gpu-metrics/all")
async def get_all_gpu_metrics():
    """
    Get current metrics for all available GPUs.

    Returns:
        Dictionary mapping GPU IDs to their metrics

    Raises:
        HTTPException 503: If GPU monitoring not available
    """
    try:
        gpu_service = get_gpu_monitor_service()

        if not gpu_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="GPU monitoring is not available."
            )

        all_metrics = gpu_service.get_all_gpu_metrics()

        return {
            "gpu_count": len(all_metrics),
            "gpus": [metrics.to_dict() for metrics in all_metrics]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get all GPU metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU metrics: {str(e)}")


@router.get("/gpu-info", response_model=GPUInfoResponse)
async def get_gpu_info(
    gpu_id: int = Query(0, ge=0, description="GPU index to query")
):
    """
    Get static information about specified GPU.

    Args:
        gpu_id: GPU index (default: 0)

    Returns:
        GPUInfoResponse containing static GPU information

    Raises:
        HTTPException 400: If invalid GPU ID
        HTTPException 503: If GPU monitoring not available
    """
    try:
        gpu_service = get_gpu_monitor_service()

        if not gpu_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="GPU monitoring is not available."
            )

        # Validate GPU ID
        if gpu_id >= gpu_service.get_device_count():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid GPU ID: {gpu_id}. Available GPUs: 0-{gpu_service.get_device_count() - 1}"
            )

        info = gpu_service.get_gpu_info(gpu_id)

        return GPUInfoResponse(
            gpu_id=gpu_id,
            info=info.to_dict()
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get GPU info for GPU {gpu_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU info: {str(e)}")


@router.get("/gpu-processes", response_model=GPUProcessesResponse)
async def get_gpu_processes(
    gpu_id: int = Query(0, ge=0, description="GPU index to query")
):
    """
    Get list of processes using specified GPU.

    Args:
        gpu_id: GPU index (default: 0)

    Returns:
        GPUProcessesResponse containing active GPU processes

    Raises:
        HTTPException 400: If invalid GPU ID
        HTTPException 503: If GPU monitoring not available
    """
    try:
        gpu_service = get_gpu_monitor_service()

        if not gpu_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="GPU monitoring is not available."
            )

        # Validate GPU ID
        if gpu_id >= gpu_service.get_device_count():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid GPU ID: {gpu_id}. Available GPUs: 0-{gpu_service.get_device_count() - 1}"
            )

        processes = gpu_service.get_gpu_processes(gpu_id)

        return GPUProcessesResponse(
            gpu_id=gpu_id,
            process_count=len(processes),
            processes=[proc.to_dict() for proc in processes]
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get GPU processes for GPU {gpu_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU processes: {str(e)}")


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """
    Get current system resource metrics (CPU, RAM, swap, disk I/O, network I/O).

    Returns:
        SystemMetricsResponse containing system metrics

    Raises:
        HTTPException 500: If failed to collect metrics
    """
    try:
        system_service = get_system_monitor_service()
        metrics = system_service.get_system_metrics()

        return SystemMetricsResponse(
            metrics=metrics.to_dict()
        )

    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


@router.get("/disk-usage")
async def get_disk_usage(
    mount_points: Optional[str] = Query(None, description="Comma-separated list of mount points (e.g., '/,/data/')")
):
    """
    Get disk usage for specified mount points.

    Args:
        mount_points: Comma-separated list of mount points (default: "/,/data/")

    Returns:
        Dictionary containing disk usage information

    Raises:
        HTTPException 500: If failed to collect disk usage
    """
    try:
        system_service = get_system_monitor_service()

        # Parse mount points
        mount_point_list = None
        if mount_points:
            mount_point_list = [mp.strip() for mp in mount_points.split(",")]

        disk_usage = system_service.get_disk_usage(mount_point_list)

        return {
            "mount_points": [usage.to_dict() for usage in disk_usage]
        }

    except Exception as e:
        logger.error(f"Failed to get disk usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get disk usage: {str(e)}")


@router.get("/network-rates")
async def get_network_rates():
    """
    Get current network I/O rates (bytes per second).

    Returns:
        Dictionary containing network send/receive rates

    Note:
        Rates are calculated based on difference from last call.
        First call may return 0 or inaccurate values.
    """
    try:
        system_service = get_system_monitor_service()
        rates = system_service.get_network_rates()

        return {
            "network_rates": rates.to_dict()
        }

    except Exception as e:
        logger.error(f"Failed to get network rates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get network rates: {str(e)}")


@router.get("/disk-rates")
async def get_disk_rates():
    """
    Get current disk I/O rates (bytes per second).

    Returns:
        Dictionary containing disk read/write rates

    Note:
        Rates are calculated based on difference from last call.
        First call may return 0 or inaccurate values.
    """
    try:
        system_service = get_system_monitor_service()
        rates = system_service.get_disk_rates()

        return {
            "disk_rates": rates.to_dict()
        }

    except Exception as e:
        logger.error(f"Failed to get disk rates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get disk rates: {str(e)}")


@router.get("/all")
async def get_all_monitoring_data(
    gpu_id: int = Query(0, ge=0, description="GPU index to query for GPU-specific metrics")
):
    """
    Get all monitoring data in a single call (GPU metrics, system metrics, disk usage, I/O rates).

    This endpoint is optimized for dashboard views that need all metrics at once.

    Args:
        gpu_id: GPU index for GPU-specific metrics (default: 0)

    Returns:
        Dictionary containing all monitoring data

    Raises:
        HTTPException 503: If GPU monitoring not available
    """
    try:
        gpu_service = get_gpu_monitor_service()
        system_service = get_system_monitor_service()

        # Check GPU availability
        gpu_available = gpu_service.is_available()

        response = {
            "gpu_available": gpu_available,
            "system": system_service.get_system_metrics().to_dict(),
            "disk_usage": [usage.to_dict() for usage in system_service.get_disk_usage()],
            "network_rates": system_service.get_network_rates().to_dict(),
            "disk_rates": system_service.get_disk_rates().to_dict(),
        }

        if gpu_available:
            # Add GPU data
            if gpu_id >= gpu_service.get_device_count():
                gpu_id = 0  # Fallback to GPU 0 if invalid

            response["gpu"] = {
                "gpu_count": gpu_service.get_device_count(),
                "selected_gpu_id": gpu_id,
                "metrics": gpu_service.get_gpu_metrics(gpu_id).to_dict(),
                "info": gpu_service.get_gpu_info(gpu_id).to_dict(),
                "processes": [proc.to_dict() for proc in gpu_service.get_gpu_processes(gpu_id)],
            }

        return response

    except Exception as e:
        logger.error(f"Failed to get all monitoring data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring data: {str(e)}")


@router.get("/resource-estimate", response_model=ResourceEstimateResponse)
async def get_resource_estimate(
    training_id: str = Query(..., description="Training ID to extract features from"),
    evaluation_samples: int = Query(10000, ge=1000, le=100000, description="Number of samples to evaluate"),
    top_k_examples: int = Query(100, ge=10, le=1000, description="Number of top examples per feature"),
    batch_size: Optional[int] = Query(None, ge=8, le=256, description="Batch size (optional, will use recommended if not provided)"),
    num_workers: Optional[int] = Query(None, ge=1, le=32, description="Number of CPU workers (optional, will use recommended if not provided)"),
    db_commit_batch: Optional[int] = Query(None, ge=500, le=5000, description="Database commit batch size (optional, will use recommended if not provided)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Estimate resource usage for a feature extraction job.

    This endpoint calculates estimated RAM, GPU memory, and duration for an extraction
    job based on the provided configuration. It also validates against available system
    resources and provides recommendations.

    Args:
        training_id: ID of the training to extract features from
        evaluation_samples: Number of dataset samples to evaluate
        top_k_examples: Number of top-activating examples to store per feature
        batch_size: Optional batch size (defaults to recommended)
        num_workers: Optional number of CPU workers (defaults to recommended)
        db_commit_batch: Optional database commit batch size (defaults to recommended)

    Returns:
        ResourceEstimateResponse with resource estimates and recommendations

    Raises:
        HTTPException 404: If training not found
        HTTPException 500: If estimation fails
    """
    try:
        # Fetch training to get hyperparameters
        result = await db.execute(
            select(Training).where(Training.id == training_id)
        )
        training = result.scalar_one_or_none()

        if not training:
            raise HTTPException(
                status_code=404,
                detail=f"Training {training_id} not found"
            )

        # Get training hyperparameters
        latent_dim = training.hyperparameters.get("latent_dim", 8192)
        hidden_dim = training.hyperparameters.get("hidden_dim", 768)
        max_length = 512  # Default sequence length

        # Get system resources
        system_resources = ResourceConfig.get_system_resources()

        # Calculate recommended settings
        recommended_settings = ResourceConfig.calculate_extraction_config(
            num_features=latent_dim,
            top_k_examples=top_k_examples,
            sequence_length=max_length,
            hidden_dim=hidden_dim
        )

        # Use provided settings or fall back to recommended
        current_batch_size = batch_size if batch_size is not None else recommended_settings["batch_size"]
        current_num_workers = num_workers if num_workers is not None else recommended_settings["num_workers"]
        current_db_commit_batch = db_commit_batch if db_commit_batch is not None else recommended_settings["db_commit_batch"]

        current_settings = {
            "batch_size": current_batch_size,
            "num_workers": current_num_workers,
            "db_commit_batch": current_db_commit_batch
        }

        # Estimate resource usage
        resource_estimates = ResourceConfig.estimate_resource_usage(
            num_features=latent_dim,
            top_k_examples=top_k_examples,
            batch_size=current_batch_size,
            num_workers=current_num_workers,
            evaluation_samples=evaluation_samples,
            sequence_length=max_length,
            hidden_dim=hidden_dim
        )

        return ResourceEstimateResponse(
            system_resources=system_resources,
            recommended_settings=recommended_settings,
            current_settings=current_settings,
            resource_estimates=resource_estimates
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to estimate resources for training {training_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to estimate resources: {str(e)}"
        )
