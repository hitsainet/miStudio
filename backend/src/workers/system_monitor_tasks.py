"""
System Monitoring Background Tasks

This module provides Celery tasks for periodic system resource monitoring and
WebSocket emission. These tasks replace the old HTTP polling-based approach
with real-time WebSocket push for better efficiency and consistency.

Tasks:
    - monitor_system_metrics: Collect and emit all system metrics via WebSocket

Author: miStudio Team
Created: 2025-10-22
"""

import logging
from datetime import datetime
from typing import Dict, Any

from celery import shared_task

from ..services.system_monitor_service import get_system_monitor_service
from ..services.gpu_monitor_service import get_gpu_monitor_service
from .websocket_emitter import (
    emit_gpu_metrics,
    emit_cpu_metrics,
    emit_memory_metrics,
    emit_disk_metrics,
    emit_network_metrics,
)

logger = logging.getLogger(__name__)


@shared_task(name="workers.monitor_system_metrics", bind=True)
def monitor_system_metrics(self):
    """
    Collect system metrics and emit via WebSocket.

    This task runs periodically (configured in Celery Beat) to collect:
    - GPU metrics (per-GPU if available)
    - CPU utilization
    - RAM and Swap usage
    - Disk I/O statistics
    - Network I/O statistics

    All metrics are emitted to WebSocket channels for real-time frontend updates.

    Returns:
        Dict with status and metrics emitted

    Raises:
        Exception: If critical error occurs during metric collection
    """
    try:
        timestamp = datetime.utcnow().isoformat() + "Z"
        metrics_emitted = []

        # Get services
        system_service = get_system_monitor_service()
        gpu_service = get_gpu_monitor_service()

        # ================================================================
        # GPU Metrics (if available)
        # ================================================================
        if gpu_service.is_available():
            try:
                gpu_count = gpu_service.get_device_count()

                for gpu_id in range(gpu_count):
                    try:
                        gpu_metrics = gpu_service.get_gpu_metrics(gpu_id)

                        # Prepare metrics payload
                        metrics_data = gpu_metrics.to_dict()
                        metrics_data["timestamp"] = timestamp

                        # Emit to WebSocket channel: system/gpu/{gpu_id}
                        success = emit_gpu_metrics(gpu_id=gpu_id, metrics=metrics_data)

                        if success:
                            metrics_emitted.append(f"gpu/{gpu_id}")
                        else:
                            logger.warning(f"Failed to emit GPU {gpu_id} metrics via WebSocket")

                    except Exception as e:
                        logger.error(f"Failed to collect GPU {gpu_id} metrics: {e}")
                        continue

            except Exception as e:
                logger.error(f"Failed to collect GPU metrics: {e}")
        else:
            logger.debug("GPU monitoring not available, skipping GPU metrics")

        # ================================================================
        # System Metrics (CPU, RAM, Swap, Disk I/O, Network I/O)
        # ================================================================
        try:
            system_metrics = system_service.get_system_metrics()
            system_dict = system_metrics.to_dict()

            # CPU Metrics
            cpu_data = {
                **system_dict["cpu"],
                "timestamp": timestamp,
            }
            if emit_cpu_metrics(metrics=cpu_data):
                metrics_emitted.append("cpu")

            # Memory Metrics (RAM + Swap)
            memory_data = {
                "ram": system_dict["ram"],
                "swap": system_dict["swap"],
                "timestamp": timestamp,
            }
            if emit_memory_metrics(metrics=memory_data):
                metrics_emitted.append("memory")

            # Disk I/O Metrics
            disk_data = {
                **system_dict["disk_io"],
                "timestamp": timestamp,
            }
            if emit_disk_metrics(metrics=disk_data):
                metrics_emitted.append("disk")

            # Network I/O Metrics
            network_data = {
                **system_dict["network_io"],
                "timestamp": timestamp,
            }
            if emit_network_metrics(metrics=network_data):
                metrics_emitted.append("network")

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise

        # ================================================================
        # Task Result
        # ================================================================
        result = {
            "status": "success",
            "metrics_emitted": metrics_emitted,
            "timestamp": timestamp,
            "metrics_count": len(metrics_emitted),
        }

        logger.debug(f"System metrics emitted successfully: {metrics_emitted}")
        return result

    except Exception as e:
        logger.error(f"Failed to monitor system metrics: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


# Additional monitoring tasks can be added here in the future
# For example:
# - monitor_gpu_processes: Track processes using GPUs
# - monitor_disk_usage: Track disk space usage alerts
# - monitor_training_resource_correlation: Correlate training jobs with GPU usage
