"""
GPU Monitoring Service

This service provides real-time GPU metrics collection using pynvml (NVIDIA Management Library).
It supports single and multi-GPU systems and handles various GPU models.

Author: miStudio Team
Created: 2025-10-16
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available. GPU monitoring will be disabled.")


logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """Data class for GPU metrics"""
    gpu_id: int
    utilization_gpu: float  # GPU utilization percentage (0-100)
    utilization_memory: float  # Memory controller utilization (0-100)
    memory_used: int  # Used memory in bytes
    memory_total: int  # Total memory in bytes
    memory_free: int  # Free memory in bytes
    temperature: int  # Temperature in Celsius
    power_usage: float  # Power usage in Watts
    power_limit: float  # Power limit in Watts
    fan_speed: int  # Fan speed percentage (0-100)
    gpu_clock: int  # GPU clock speed in MHz
    memory_clock: int  # Memory clock speed in MHz

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "gpu_id": self.gpu_id,
            "utilization": {
                "gpu": self.utilization_gpu,
                "memory": self.utilization_memory,
            },
            "memory": {
                "used": self.memory_used,
                "total": self.memory_total,
                "free": self.memory_free,
                "used_gb": round(self.memory_used / (1024**3), 2),
                "total_gb": round(self.memory_total / (1024**3), 2),
                "used_percent": round((self.memory_used / self.memory_total) * 100, 1),
            },
            "temperature": self.temperature,
            "power": {
                "usage": self.power_usage,
                "limit": self.power_limit,
                "usage_percent": round((self.power_usage / self.power_limit) * 100, 1) if self.power_limit > 0 else 0,
            },
            "fan_speed": self.fan_speed,
            "clocks": {
                "gpu": self.gpu_clock,
                "memory": self.memory_clock,
            },
        }


@dataclass
class GPUInfo:
    """Data class for static GPU information"""
    gpu_id: int
    name: str
    uuid: str
    pci_bus_id: str
    driver_version: str
    cuda_version: str
    compute_capability_major: int
    compute_capability_minor: int
    total_memory: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "compute_capability": f"{self.compute_capability_major}.{self.compute_capability_minor}",
            "total_memory_gb": round(self.total_memory / (1024**3), 2),
        }


@dataclass
class GPUProcess:
    """Data class for GPU process information"""
    pid: int
    process_name: str
    gpu_memory_used: int  # Memory used by process in bytes
    gpu_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "pid": self.pid,
            "process_name": self.process_name,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_memory_used_mb": round(self.gpu_memory_used / (1024**2), 2),
            "gpu_id": self.gpu_id,
        }


class GPUMonitorService:
    """
    Service for monitoring NVIDIA GPU metrics using pynvml.

    This service provides methods to:
    - List available GPUs
    - Get current GPU metrics (utilization, memory, temperature, etc.)
    - Get static GPU information
    - Get GPU process information
    """

    def __init__(self):
        """Initialize the GPU monitoring service"""
        self._initialized = False
        self._device_count = 0

        if not PYNVML_AVAILABLE:
            logger.error("pynvml is not available. GPU monitoring is disabled.")
            return

        try:
            pynvml.nvmlInit()
            self._initialized = True
            self._device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"GPU monitoring initialized. Found {self._device_count} GPU(s).")
        except Exception as e:
            logger.error(f"Failed to initialize pynvml: {e}")
            self._initialized = False

    def is_available(self) -> bool:
        """Check if GPU monitoring is available"""
        return self._initialized and self._device_count > 0

    def get_device_count(self) -> int:
        """Get the number of available GPUs"""
        return self._device_count if self._initialized else 0

    def _get_handle(self, gpu_id: int):
        """Get NVML device handle by GPU ID"""
        if not self._initialized:
            raise RuntimeError("GPU monitoring not initialized")

        if gpu_id < 0 or gpu_id >= self._device_count:
            raise ValueError(f"Invalid GPU ID: {gpu_id}. Available GPUs: 0-{self._device_count - 1}")

        try:
            return pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except Exception as e:
            logger.error(f"Failed to get handle for GPU {gpu_id}: {e}")
            raise

    def get_gpu_metrics(self, gpu_id: int = 0) -> GPUMetrics:
        """
        Get current GPU metrics for specified GPU.

        Args:
            gpu_id: GPU index (default: 0)

        Returns:
            GPUMetrics object containing current metrics

        Raises:
            RuntimeError: If GPU monitoring not available
            ValueError: If invalid GPU ID
        """
        handle = self._get_handle(gpu_id)

        try:
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization_gpu = float(util.gpu)
            utilization_memory = float(util.memory)

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = mem_info.used
            memory_total = mem_info.total
            memory_free = mem_info.free

            # Get temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Get power usage
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            except pynvml.NVMLError:
                power_usage = 0.0
                logger.warning(f"Power usage not available for GPU {gpu_id}")

            # Get power limit
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # Convert mW to W
            except pynvml.NVMLError:
                power_limit = 0.0
                logger.warning(f"Power limit not available for GPU {gpu_id}")

            # Get fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except pynvml.NVMLError:
                fan_speed = 0
                logger.warning(f"Fan speed not available for GPU {gpu_id}")

            # Get clock speeds
            try:
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            except pynvml.NVMLError:
                gpu_clock = 0
                logger.warning(f"GPU clock not available for GPU {gpu_id}")

            try:
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except pynvml.NVMLError:
                memory_clock = 0
                logger.warning(f"Memory clock not available for GPU {gpu_id}")

            return GPUMetrics(
                gpu_id=gpu_id,
                utilization_gpu=utilization_gpu,
                utilization_memory=utilization_memory,
                memory_used=memory_used,
                memory_total=memory_total,
                memory_free=memory_free,
                temperature=temperature,
                power_usage=power_usage,
                power_limit=power_limit,
                fan_speed=fan_speed,
                gpu_clock=gpu_clock,
                memory_clock=memory_clock,
            )

        except Exception as e:
            logger.error(f"Failed to get metrics for GPU {gpu_id}: {e}")
            raise

    def get_all_gpu_metrics(self) -> List[GPUMetrics]:
        """
        Get current metrics for all GPUs.

        Returns:
            List of GPUMetrics objects for all GPUs
        """
        if not self._initialized:
            return []

        metrics = []
        for gpu_id in range(self._device_count):
            try:
                metrics.append(self.get_gpu_metrics(gpu_id))
            except Exception as e:
                logger.error(f"Failed to get metrics for GPU {gpu_id}: {e}")
                continue

        return metrics

    def get_gpu_info(self, gpu_id: int = 0) -> GPUInfo:
        """
        Get static information about specified GPU.

        Args:
            gpu_id: GPU index (default: 0)

        Returns:
            GPUInfo object containing static GPU information
        """
        handle = self._get_handle(gpu_id)

        try:
            # Get device name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            # Get UUID
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()

            # Get PCI bus ID
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus_id = pci_info.busId
            if isinstance(pci_bus_id, bytes):
                pci_bus_id = pci_bus_id.decode()

            # Get driver version
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver_version, bytes):
                    driver_version = driver_version.decode()
            except Exception:
                driver_version = "N/A"

            # Get CUDA version
            try:
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            except Exception:
                cuda_version = "N/A"

            # Get compute capability
            try:
                cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
            except Exception:
                cc_major = 0
                cc_minor = 0

            # Get total memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = mem_info.total

            return GPUInfo(
                gpu_id=gpu_id,
                name=name,
                uuid=uuid,
                pci_bus_id=pci_bus_id,
                driver_version=driver_version,
                cuda_version=cuda_version,
                compute_capability_major=cc_major,
                compute_capability_minor=cc_minor,
                total_memory=total_memory,
            )

        except Exception as e:
            logger.error(f"Failed to get info for GPU {gpu_id}: {e}")
            raise

    def get_all_gpu_info(self) -> List[GPUInfo]:
        """
        Get static information for all GPUs.

        Returns:
            List of GPUInfo objects for all GPUs
        """
        if not self._initialized:
            return []

        info_list = []
        for gpu_id in range(self._device_count):
            try:
                info_list.append(self.get_gpu_info(gpu_id))
            except Exception as e:
                logger.error(f"Failed to get info for GPU {gpu_id}: {e}")
                continue

        return info_list

    def get_gpu_processes(self, gpu_id: int = 0) -> List[GPUProcess]:
        """
        Get list of processes using specified GPU.

        Args:
            gpu_id: GPU index (default: 0)

        Returns:
            List of GPUProcess objects
        """
        handle = self._get_handle(gpu_id)

        try:
            processes = []

            # Get compute processes
            try:
                compute_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in compute_processes:
                    try:
                        # Get process name
                        import psutil
                        process = psutil.Process(proc.pid)
                        process_name = process.name()
                    except Exception:
                        process_name = "Unknown"

                    processes.append(GPUProcess(
                        pid=proc.pid,
                        process_name=process_name,
                        gpu_memory_used=proc.usedGpuMemory,
                        gpu_id=gpu_id,
                    ))
            except pynvml.NVMLError as e:
                logger.warning(f"Failed to get compute processes for GPU {gpu_id}: {e}")

            # Get graphics processes (may not be available on all GPUs)
            try:
                graphics_processes = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                for proc in graphics_processes:
                    try:
                        # Get process name
                        import psutil
                        process = psutil.Process(proc.pid)
                        process_name = process.name()
                    except Exception:
                        process_name = "Unknown"

                    # Check if process already in list (avoid duplicates)
                    if not any(p.pid == proc.pid for p in processes):
                        processes.append(GPUProcess(
                            pid=proc.pid,
                            process_name=process_name,
                            gpu_memory_used=proc.usedGpuMemory,
                            gpu_id=gpu_id,
                        ))
            except (pynvml.NVMLError, AttributeError) as e:
                logger.debug(f"Graphics processes not available for GPU {gpu_id}: {e}")

            # Sort by GPU memory usage (descending)
            processes.sort(key=lambda p: p.gpu_memory_used, reverse=True)

            return processes

        except Exception as e:
            logger.error(f"Failed to get processes for GPU {gpu_id}: {e}")
            raise

    def get_all_gpu_processes(self) -> Dict[int, List[GPUProcess]]:
        """
        Get processes for all GPUs.

        Returns:
            Dictionary mapping GPU ID to list of GPUProcess objects
        """
        if not self._initialized:
            return {}

        all_processes = {}
        for gpu_id in range(self._device_count):
            try:
                all_processes[gpu_id] = self.get_gpu_processes(gpu_id)
            except Exception as e:
                logger.error(f"Failed to get processes for GPU {gpu_id}: {e}")
                all_processes[gpu_id] = []

        return all_processes

    def shutdown(self):
        """Shutdown the GPU monitoring service"""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
                self._initialized = False
                logger.info("GPU monitoring shut down successfully.")
            except Exception as e:
                logger.error(f"Failed to shutdown pynvml: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.shutdown()


# Global instance
_gpu_monitor_service: Optional[GPUMonitorService] = None


def get_gpu_monitor_service() -> GPUMonitorService:
    """
    Get the global GPU monitoring service instance.

    Returns:
        GPUMonitorService instance
    """
    global _gpu_monitor_service

    if _gpu_monitor_service is None:
        _gpu_monitor_service = GPUMonitorService()

    return _gpu_monitor_service
