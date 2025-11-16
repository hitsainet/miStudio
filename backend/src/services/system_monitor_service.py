"""
System Monitoring Service

This service provides real-time system resource metrics collection using psutil.
It monitors CPU, memory, swap, network I/O, and disk I/O.

Author: miStudio Team
Created: 2025-10-16
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import psutil


logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Data class for system resource metrics"""
    cpu_percent: float  # Overall CPU utilization percentage
    cpu_count: int  # Number of CPU cores
    ram_used: int  # Used RAM in bytes
    ram_total: int  # Total RAM in bytes
    ram_available: int  # Available RAM in bytes
    swap_used: int  # Used swap in bytes
    swap_total: int  # Total swap in bytes
    disk_read_bytes: int  # Total disk bytes read since boot
    disk_write_bytes: int  # Total disk bytes written since boot
    network_sent_bytes: int  # Total network bytes sent since boot
    network_recv_bytes: int  # Total network bytes received since boot

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "cpu": {
                "percent": self.cpu_percent,
                "count": self.cpu_count,
            },
            "ram": {
                "used": self.ram_used,
                "total": self.ram_total,
                "available": self.ram_available,
                "used_gb": round(self.ram_used / (1024**3), 2),
                "total_gb": round(self.ram_total / (1024**3), 2),
                "used_percent": round((self.ram_used / self.ram_total) * 100, 1),
            },
            "swap": {
                "used": self.swap_used,
                "total": self.swap_total,
                "used_gb": round(self.swap_used / (1024**3), 2),
                "total_gb": round(self.swap_total / (1024**3), 2),
                "used_percent": round((self.swap_used / self.swap_total) * 100, 1) if self.swap_total > 0 else 0,
            },
            "disk_io": {
                "read_bytes": self.disk_read_bytes,
                "write_bytes": self.disk_write_bytes,
                "read_mb": round(self.disk_read_bytes / (1024**2), 2),
                "write_mb": round(self.disk_write_bytes / (1024**2), 2),
            },
            "network_io": {
                "sent_bytes": self.network_sent_bytes,
                "recv_bytes": self.network_recv_bytes,
                "sent_mb": round(self.network_sent_bytes / (1024**2), 2),
                "recv_mb": round(self.network_recv_bytes / (1024**2), 2),
            },
        }


@dataclass
class DiskUsage:
    """Data class for disk usage information"""
    mount_point: str
    total: int  # Total disk space in bytes
    used: int  # Used disk space in bytes
    free: int  # Free disk space in bytes
    percent: float  # Usage percentage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "mount_point": self.mount_point,
            "total": self.total,
            "used": self.used,
            "free": self.free,
            "total_gb": round(self.total / (1024**3), 2),
            "used_gb": round(self.used / (1024**3), 2),
            "free_gb": round(self.free / (1024**3), 2),
            "percent": self.percent,
        }


@dataclass
class NetworkRates:
    """Data class for network I/O rates"""
    sent_rate: float  # Bytes per second sent
    recv_rate: float  # Bytes per second received

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "sent_rate": self.sent_rate,
            "recv_rate": self.recv_rate,
            "sent_mbps": round(self.sent_rate * 8 / (1024**2), 2),  # Convert to Mbps
            "recv_mbps": round(self.recv_rate * 8 / (1024**2), 2),  # Convert to Mbps
        }


@dataclass
class DiskRates:
    """Data class for disk I/O rates"""
    read_rate: float  # Bytes per second read
    write_rate: float  # Bytes per second written

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "read_rate": self.read_rate,
            "write_rate": self.write_rate,
            "read_mbps": round(self.read_rate / (1024**2), 2),  # Convert to MB/s
            "write_mbps": round(self.write_rate / (1024**2), 2),  # Convert to MB/s
        }


class SystemMonitorService:
    """
    Service for monitoring system resources using psutil.

    This service provides methods to:
    - Get current CPU, RAM, swap usage
    - Get disk usage for mount points
    - Get network I/O statistics and rates
    - Get disk I/O statistics and rates
    """

    def __init__(self):
        """Initialize the system monitoring service"""
        # Store previous values for rate calculation
        self._last_network_io: Optional[Dict[str, int]] = None
        self._last_network_time: Optional[float] = None
        self._last_disk_io: Optional[Dict[str, int]] = None
        self._last_disk_time: Optional[float] = None

        # Initialize with current values
        try:
            net_io = psutil.net_io_counters()
            self._last_network_io = {
                "sent": net_io.bytes_sent,
                "recv": net_io.bytes_recv,
            }
            self._last_network_time = time.time()

            disk_io = psutil.disk_io_counters()
            self._last_disk_io = {
                "read": disk_io.read_bytes,
                "write": disk_io.write_bytes,
            }
            self._last_disk_time = time.time()

            logger.info("System monitoring initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize system monitoring: {e}")

    def get_system_metrics(self) -> SystemMetrics:
        """
        Get current system resource metrics.

        Returns:
            SystemMetrics object containing current metrics
        """
        try:
            # Get CPU usage per-core and sum them (100% = 1 full core)
            # On 16-core system: max is 1600% (all cores at 100%)
            per_core_percents = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_percent = sum(per_core_percents)  # Sum all core percentages
            cpu_count = psutil.cpu_count()

            # Get RAM usage
            ram = psutil.virtual_memory()
            ram_used = ram.used
            ram_total = ram.total
            ram_available = ram.available

            # Get swap usage
            swap = psutil.swap_memory()
            swap_used = swap.used
            swap_total = swap.total

            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_bytes = disk_io.read_bytes
            disk_write_bytes = disk_io.write_bytes

            # Get network I/O
            net_io = psutil.net_io_counters()
            network_sent_bytes = net_io.bytes_sent
            network_recv_bytes = net_io.bytes_recv

            return SystemMetrics(
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                ram_used=ram_used,
                ram_total=ram_total,
                ram_available=ram_available,
                swap_used=swap_used,
                swap_total=swap_total,
                disk_read_bytes=disk_read_bytes,
                disk_write_bytes=disk_write_bytes,
                network_sent_bytes=network_sent_bytes,
                network_recv_bytes=network_recv_bytes,
            )

        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            raise

    def get_disk_usage(self, mount_points: Optional[List[str]] = None) -> List[DiskUsage]:
        """
        Get disk usage for specified mount points.

        Args:
            mount_points: List of mount points to check (default: ["/", "/data/"])

        Returns:
            List of DiskUsage objects
        """
        if mount_points is None:
            mount_points = ["/", "/data/"]

        disk_usage_list = []

        for mount_point in mount_points:
            try:
                # Check if mount point exists
                if not psutil.disk_usage(mount_point):
                    logger.warning(f"Mount point {mount_point} not found")
                    continue

                usage = psutil.disk_usage(mount_point)
                disk_usage_list.append(DiskUsage(
                    mount_point=mount_point,
                    total=usage.total,
                    used=usage.used,
                    free=usage.free,
                    percent=usage.percent,
                ))

            except FileNotFoundError:
                logger.warning(f"Mount point {mount_point} not found")
                continue
            except PermissionError:
                logger.warning(f"Permission denied for mount point {mount_point}")
                continue
            except Exception as e:
                logger.error(f"Failed to get disk usage for {mount_point}: {e}")
                continue

        return disk_usage_list

    def get_network_rates(self) -> NetworkRates:
        """
        Get current network I/O rates (bytes per second).

        Returns:
            NetworkRates object with send/receive rates

        Note:
            Rates are calculated based on difference from last call.
            First call may return 0 or inaccurate values.
        """
        try:
            current_time = time.time()
            net_io = psutil.net_io_counters()

            current_sent = net_io.bytes_sent
            current_recv = net_io.bytes_recv

            if self._last_network_io is None or self._last_network_time is None:
                # First call, initialize and return 0
                self._last_network_io = {
                    "sent": current_sent,
                    "recv": current_recv,
                }
                self._last_network_time = current_time
                return NetworkRates(sent_rate=0.0, recv_rate=0.0)

            # Calculate time delta
            time_delta = current_time - self._last_network_time

            if time_delta == 0:
                return NetworkRates(sent_rate=0.0, recv_rate=0.0)

            # Calculate rates
            sent_rate = (current_sent - self._last_network_io["sent"]) / time_delta
            recv_rate = (current_recv - self._last_network_io["recv"]) / time_delta

            # Update last values
            self._last_network_io = {
                "sent": current_sent,
                "recv": current_recv,
            }
            self._last_network_time = current_time

            return NetworkRates(sent_rate=sent_rate, recv_rate=recv_rate)

        except Exception as e:
            logger.error(f"Failed to get network rates: {e}")
            return NetworkRates(sent_rate=0.0, recv_rate=0.0)

    def get_disk_rates(self) -> DiskRates:
        """
        Get current disk I/O rates (bytes per second).

        Returns:
            DiskRates object with read/write rates

        Note:
            Rates are calculated based on difference from last call.
            First call may return 0 or inaccurate values.
        """
        try:
            current_time = time.time()
            disk_io = psutil.disk_io_counters()

            current_read = disk_io.read_bytes
            current_write = disk_io.write_bytes

            if self._last_disk_io is None or self._last_disk_time is None:
                # First call, initialize and return 0
                self._last_disk_io = {
                    "read": current_read,
                    "write": current_write,
                }
                self._last_disk_time = current_time
                return DiskRates(read_rate=0.0, write_rate=0.0)

            # Calculate time delta
            time_delta = current_time - self._last_disk_time

            if time_delta == 0:
                return DiskRates(read_rate=0.0, write_rate=0.0)

            # Calculate rates
            read_rate = (current_read - self._last_disk_io["read"]) / time_delta
            write_rate = (current_write - self._last_disk_io["write"]) / time_delta

            # Update last values
            self._last_disk_io = {
                "read": current_read,
                "write": current_write,
            }
            self._last_disk_time = current_time

            return DiskRates(read_rate=read_rate, write_rate=write_rate)

        except Exception as e:
            logger.error(f"Failed to get disk rates: {e}")
            return DiskRates(read_rate=0.0, write_rate=0.0)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all system metrics in a single call.

        Returns:
            Dictionary containing system metrics, disk usage, and I/O rates
        """
        try:
            system_metrics = self.get_system_metrics()
            disk_usage = self.get_disk_usage()
            network_rates = self.get_network_rates()
            disk_rates = self.get_disk_rates()

            return {
                "system": system_metrics.to_dict(),
                "disk_usage": [d.to_dict() for d in disk_usage],
                "network_rates": network_rates.to_dict(),
                "disk_rates": disk_rates.to_dict(),
            }

        except Exception as e:
            logger.error(f"Failed to get all metrics: {e}")
            raise


# Global instance
_system_monitor_service: Optional[SystemMonitorService] = None


def get_system_monitor_service() -> SystemMonitorService:
    """
    Get the global system monitoring service instance.

    Returns:
        SystemMonitorService instance
    """
    global _system_monitor_service

    if _system_monitor_service is None:
        _system_monitor_service = SystemMonitorService()

    return _system_monitor_service
