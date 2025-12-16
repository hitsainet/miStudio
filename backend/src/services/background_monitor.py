"""
Background System Monitor Service

Asyncio-based system monitoring that runs independently of Celery.
This avoids the blocking issue where long-running Celery tasks prevent
monitoring tasks from executing.

The monitor runs as a background asyncio task started with the FastAPI app.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


class BackgroundMonitor:
    """
    Asyncio-based background monitor for system metrics.

    Runs independently of Celery to ensure monitoring is never blocked
    by long-running tasks like training.
    """

    def __init__(self, interval_seconds: float = 2.0):
        """
        Initialize the background monitor.

        Args:
            interval_seconds: How often to collect and emit metrics (default 2s)
        """
        self.interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Start the background monitoring task."""
        if self._running:
            logger.warning("Background monitor already running")
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=5.0)
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Background monitor started (interval: {self.interval_seconds}s)")

    async def stop(self):
        """Stop the background monitoring task."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("Background monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop - collects and emits metrics periodically."""
        # Import here to avoid circular imports
        from .system_monitor_service import get_system_monitor_service
        from .gpu_monitor_service import get_gpu_monitor_service

        system_service = get_system_monitor_service()
        gpu_service = get_gpu_monitor_service()

        logger.info("Background monitor loop starting...")

        while self._running:
            try:
                await self._collect_and_emit_metrics(system_service, gpu_service)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)

            # Wait for next interval
            await asyncio.sleep(self.interval_seconds)

    async def _collect_and_emit_metrics(self, system_service, gpu_service):
        """Collect all system metrics and emit via WebSocket."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        metrics_emitted = []

        # ================================================================
        # GPU Metrics (if available)
        # ================================================================
        if gpu_service.is_available():
            try:
                gpu_count = gpu_service.get_device_count()

                for gpu_id in range(gpu_count):
                    try:
                        gpu_metrics = gpu_service.get_gpu_metrics(gpu_id)
                        metrics_data = gpu_metrics.to_dict()
                        metrics_data["timestamp"] = timestamp

                        if await self._emit_to_channel(
                            channel=f"system/gpu/{gpu_id}",
                            event="system:metrics",
                            data=metrics_data
                        ):
                            metrics_emitted.append(f"gpu/{gpu_id}")

                    except Exception as e:
                        logger.error(f"Failed to collect GPU {gpu_id} metrics: {e}")

            except Exception as e:
                logger.error(f"Failed to collect GPU metrics: {e}")

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
            if await self._emit_to_channel("system/cpu", "system:metrics", cpu_data):
                metrics_emitted.append("cpu")

            # Memory Metrics (RAM + Swap)
            memory_data = {
                "ram": system_dict["ram"],
                "swap": system_dict["swap"],
                "timestamp": timestamp,
            }
            if await self._emit_to_channel("system/memory", "system:metrics", memory_data):
                metrics_emitted.append("memory")

            # Disk I/O Metrics
            disk_data = {
                **system_dict["disk_io"],
                "timestamp": timestamp,
            }
            if await self._emit_to_channel("system/disk", "system:metrics", disk_data):
                metrics_emitted.append("disk")

            # Network I/O Metrics
            network_data = {
                **system_dict["network_io"],
                "timestamp": timestamp,
            }
            if await self._emit_to_channel("system/network", "system:metrics", network_data):
                metrics_emitted.append("network")

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

        if metrics_emitted:
            logger.debug(f"Emitted metrics: {metrics_emitted}")

    async def _emit_to_channel(self, channel: str, event: str, data: dict) -> bool:
        """
        Emit data to a WebSocket channel via the internal API.

        This calls the same internal endpoint that Celery workers use,
        ensuring consistent behavior.
        """
        if not self._http_client:
            return False

        try:
            response = await self._http_client.post(
                settings.websocket_emit_url,
                json={
                    "channel": channel,
                    "event": event,
                    "data": data,
                }
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to emit to {channel}: {e}")
            return False


# Singleton instance
_background_monitor: Optional[BackgroundMonitor] = None


def get_background_monitor() -> BackgroundMonitor:
    """Get or create the singleton background monitor instance."""
    global _background_monitor

    if _background_monitor is None:
        interval = getattr(settings, 'system_monitor_interval_seconds', 2.0)
        _background_monitor = BackgroundMonitor(interval_seconds=interval)

    return _background_monitor
