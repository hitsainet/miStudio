# Technical Design Document: System Monitoring

**Document ID:** 008_FTDD|System_Monitoring
**Version:** 1.1
**Last Updated:** 2025-12-16
**Status:** Implemented
**Related PRD:** [008_FPRD|System_Monitoring](../prds/008_FPRD|System_Monitoring.md)

---

## 1. System Architecture

### 1.1 Monitoring Flow
```
┌─────────────────────────────────────────────────────────────────┐
│                   System Monitoring Architecture                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Celery Beat                              │ │
│  │              (every 2 seconds)                              │ │
│  └────────────────────────┬───────────────────────────────────┘ │
│                           │                                     │
│                           ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │            collect_system_metrics_task                      │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │ │
│  │  │   pynvml   │  │   psutil   │  │    calculations    │   │ │
│  │  │ (GPU data) │  │(CPU/Mem/IO)│  │   (rates, etc.)    │   │ │
│  │  └─────┬──────┘  └─────┬──────┘  └─────────┬──────────┘   │ │
│  └────────┼───────────────┼───────────────────┼───────────────┘ │
│           │               │                   │                 │
│           └───────────────┼───────────────────┘                 │
│                           ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  WebSocket Emitter                          │ │
│  │  system/gpu/0  system/cpu  system/memory  system/disk      │ │
│  └────────────────────────┬───────────────────────────────────┘ │
│                           │                                     │
│                           ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Frontend Store                           │ │
│  │                 systemMonitorStore                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 WebSocket Channel Structure
```
                    WebSocket Channels
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ system/gpu/0  │ │ system/gpu/1  │ │ system/cpu    │
│ (per GPU)     │ │ (per GPU)     │ │ (all cores)   │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ system/memory │ │ system/disk   │ │ system/network│
│ (RAM + Swap)  │ │ (I/O rates)   │ │ (I/O rates)   │
└───────────────┘ └───────────────┘ └───────────────┘
```

---

## 2. Data Collection

### 2.1 GPU Metrics (pynvml)
```python
import pynvml

class GPUMetricsCollector:
    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()

    def collect(self, gpu_index: int) -> GPUMetrics:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        memory_util = util.memory

        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used = mem_info.used
        memory_total = mem_info.total

        # Temperature
        temperature = pynvml.nvmlDeviceGetTemperature(
            handle,
            pynvml.NVML_TEMPERATURE_GPU
        )

        # Power
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W

        return GPUMetrics(
            gpu_index=gpu_index,
            utilization=gpu_util,
            memory_used=memory_used,
            memory_total=memory_total,
            temperature=temperature,
            power_draw=power_draw
        )
```

### 2.2 System Metrics (psutil)
```python
import psutil

class SystemMetricsCollector:
    def __init__(self):
        self.prev_disk_io = psutil.disk_io_counters()
        self.prev_net_io = psutil.net_io_counters()
        self.prev_time = time.time()

    def collect_cpu(self) -> CPUMetrics:
        per_cpu = psutil.cpu_percent(percpu=True)
        overall = sum(per_cpu) / len(per_cpu)
        return CPUMetrics(
            overall_utilization=overall,
            per_core_utilization=per_cpu
        )

    def collect_memory(self) -> MemoryMetrics:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return MemoryMetrics(
            ram_used=mem.used,
            ram_total=mem.total,
            swap_used=swap.used,
            swap_total=swap.total
        )

    def collect_disk_io(self) -> DiskIOMetrics:
        current = psutil.disk_io_counters()
        current_time = time.time()
        elapsed = current_time - self.prev_time

        read_rate = (current.read_bytes - self.prev_disk_io.read_bytes) / elapsed
        write_rate = (current.write_bytes - self.prev_disk_io.write_bytes) / elapsed

        self.prev_disk_io = current
        self.prev_time = current_time

        return DiskIOMetrics(
            read_rate_mbps=read_rate / (1024 * 1024),
            write_rate_mbps=write_rate / (1024 * 1024)
        )

    def collect_network_io(self) -> NetworkIOMetrics:
        current = psutil.net_io_counters()
        current_time = time.time()
        elapsed = current_time - self.prev_time

        upload_rate = (current.bytes_sent - self.prev_net_io.bytes_sent) / elapsed
        download_rate = (current.bytes_recv - self.prev_net_io.bytes_recv) / elapsed

        self.prev_net_io = current

        return NetworkIOMetrics(
            upload_rate_mbps=upload_rate / (1024 * 1024),
            download_rate_mbps=download_rate / (1024 * 1024)
        )
```

---

## 3. Celery Beat Configuration

### 3.1 Task Registration
```python
# backend/src/core/celery_app.py

celery_app.conf.beat_schedule = {
    'collect-system-metrics': {
        'task': 'src.workers.system_monitor_tasks.collect_system_metrics_task',
        'schedule': settings.system_monitor_interval_seconds,  # Default: 2
    },
}

celery_app.conf.task_routes = {
    'src.workers.system_monitor_tasks.*': {'queue': 'monitoring'},
}
```

### 3.2 Monitoring Task
```python
# backend/src/workers/system_monitor_tasks.py

@celery_app.task
def collect_system_metrics_task():
    """Collect and emit system metrics."""
    gpu_collector = GPUMetricsCollector()
    system_collector = SystemMetricsCollector()

    # Collect GPU metrics for each GPU
    for gpu_idx in range(gpu_collector.device_count):
        gpu_metrics = gpu_collector.collect(gpu_idx)
        emit_gpu_metrics(gpu_idx, gpu_metrics)

    # Collect system metrics
    cpu_metrics = system_collector.collect_cpu()
    emit_cpu_metrics(cpu_metrics)

    memory_metrics = system_collector.collect_memory()
    emit_memory_metrics(memory_metrics)

    disk_metrics = system_collector.collect_disk_io()
    emit_disk_metrics(disk_metrics)

    network_metrics = system_collector.collect_network_io()
    emit_network_metrics(network_metrics)
```

---

## 4. WebSocket Emission

### 4.1 Emitter Functions
```python
# backend/src/workers/websocket_emitter.py

async def emit_gpu_metrics(gpu_index: int, metrics: GPUMetrics):
    """Emit GPU metrics to WebSocket channel."""
    await emit_to_channel(
        channel=f"system/gpu/{gpu_index}",
        event="metrics",
        data={
            "gpu_index": gpu_index,
            "utilization": metrics.utilization,
            "memory_used": metrics.memory_used,
            "memory_total": metrics.memory_total,
            "temperature": metrics.temperature,
            "power_draw": metrics.power_draw,
            "timestamp": datetime.now().isoformat()
        }
    )

async def emit_cpu_metrics(metrics: CPUMetrics):
    await emit_to_channel(
        channel="system/cpu",
        event="metrics",
        data={
            "overall_utilization": metrics.overall_utilization,
            "per_core_utilization": metrics.per_core_utilization,
            "timestamp": datetime.now().isoformat()
        }
    )

async def emit_memory_metrics(metrics: MemoryMetrics):
    await emit_to_channel(
        channel="system/memory",
        event="metrics",
        data={
            "ram_used": metrics.ram_used,
            "ram_total": metrics.ram_total,
            "swap_used": metrics.swap_used,
            "swap_total": metrics.swap_total,
            "timestamp": datetime.now().isoformat()
        }
    )
```

---

## 5. Frontend Architecture

### 5.1 systemMonitorStore
```typescript
interface SystemMonitorState {
  gpuMetrics: Record<number, GPUMetrics[]>;  // GPU index -> history
  cpuMetrics: CPUMetrics[];
  memoryMetrics: MemoryMetrics[];
  diskMetrics: DiskIOMetrics[];
  networkMetrics: NetworkIOMetrics[];
  isWebSocketConnected: boolean;
  historyDuration: number;  // 1 hour in ms

  // Actions
  addGPUMetrics: (gpuIndex: number, metrics: GPUMetrics) => void;
  addCPUMetrics: (metrics: CPUMetrics) => void;
  addMemoryMetrics: (metrics: MemoryMetrics) => void;
  setWebSocketConnected: (connected: boolean) => void;
  pruneOldMetrics: () => void;
}

// Implementation with history pruning
const useSystemMonitorStore = create<SystemMonitorState>((set, get) => ({
  gpuMetrics: {},
  cpuMetrics: [],
  memoryMetrics: [],
  diskMetrics: [],
  networkMetrics: [],
  isWebSocketConnected: false,
  historyDuration: 60 * 60 * 1000,  // 1 hour

  addGPUMetrics: (gpuIndex, metrics) => {
    set((state) => {
      const history = state.gpuMetrics[gpuIndex] || [];
      const cutoff = Date.now() - state.historyDuration;
      const pruned = history.filter(m => new Date(m.timestamp).getTime() > cutoff);
      return {
        gpuMetrics: {
          ...state.gpuMetrics,
          [gpuIndex]: [...pruned, metrics]
        }
      };
    });
  },
  // ... other methods
}));
```

### 5.2 WebSocket Hook
```typescript
// frontend/src/hooks/useSystemMonitorWebSocket.ts

export function useSystemMonitorWebSocket() {
  const store = useSystemMonitorStore();

  useEffect(() => {
    // Subscribe to all system channels
    const channels = [
      'system/gpu/0',
      'system/cpu',
      'system/memory',
      'system/disk',
      'system/network'
    ];

    channels.forEach(channel => {
      socket.emit('subscribe', channel);
    });

    // Handle GPU metrics
    socket.on('system/gpu/0/metrics', (data) => {
      store.addGPUMetrics(0, data);
    });

    // Handle CPU metrics
    socket.on('system/cpu/metrics', (data) => {
      store.addCPUMetrics(data);
    });

    // Handle memory metrics
    socket.on('system/memory/metrics', (data) => {
      store.addMemoryMetrics(data);
    });

    // Connection status
    socket.on('connect', () => store.setWebSocketConnected(true));
    socket.on('disconnect', () => store.setWebSocketConnected(false));

    return () => {
      channels.forEach(channel => {
        socket.emit('unsubscribe', channel);
      });
    };
  }, []);
}
```

### 5.3 Fallback Polling
```typescript
// If WebSocket disconnects, fallback to HTTP polling
useEffect(() => {
  if (!isWebSocketConnected) {
    const interval = setInterval(async () => {
      const metrics = await api.get('/system/metrics');
      // Update store with polled metrics
    }, 2000);

    return () => clearInterval(interval);
  }
}, [isWebSocketConnected]);
```

---

## 6. API Endpoints

### 6.1 HTTP Fallback Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/system/metrics` | Current metrics snapshot |
| GET | `/system/history` | Historical metrics (1 hour) |
| GET | `/system/gpu` | GPU-specific info |

### 6.2 Response Schema
```python
class SystemMetricsResponse(BaseModel):
    gpus: List[GPUMetrics]
    cpu: CPUMetrics
    memory: MemoryMetrics
    disk: DiskIOMetrics
    network: NetworkIOMetrics
    timestamp: datetime
```

---

## 7. Visualization Components

### 7.1 UtilizationChart (GPU + Temperature)
```typescript
interface UtilizationChartProps {
  data: GPUMetrics[];
  timeRange: number;  // 1 hour
}

// Dual Y-axis chart:
// - Left axis: Utilization (0-100%)
// - Right axis: Temperature (0-100°C)
// - Line 1: GPU Utilization
// - Line 2: Temperature

<ResponsiveContainer>
  <LineChart data={data}>
    <XAxis dataKey="timestamp" />
    <YAxis yAxisId="left" domain={[0, 100]} />
    <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
    <Line yAxisId="left" dataKey="utilization" stroke="#10b981" />
    <Line yAxisId="right" dataKey="temperature" stroke="#f59e0b" />
    <Tooltip />
  </LineChart>
</ResponsiveContainer>
```

### 7.2 Dashboard Layout
```
┌─────────────────────────────────────────────────────────────┐
│ System Monitor                                              │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐ ┌─────────────────────────────┐ │
│ │ GPU Utilization & Temp  │ │ Memory Usage                │ │
│ │ [Dual axis line chart]  │ │ [RAM + Swap bars]           │ │
│ └─────────────────────────┘ └─────────────────────────────┘ │
│ ┌─────────────────────────┐ ┌─────────────────────────────┐ │
│ │ CPU Utilization         │ │ I/O Rates                   │ │
│ │ [Per-core bars]         │ │ [Disk + Network]            │ │
│ └─────────────────────────┘ └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Configuration

```python
# backend/src/core/config.py

class Settings(BaseSettings):
    system_monitor_interval_seconds: int = 2
    system_monitor_history_hours: int = 1
    system_monitor_enabled: bool = True
```

---

## 9. Stability Improvements (Added Dec 2025)

### 9.1 Error Handling in Collectors
```python
class GPUMetricsCollector:
    def collect(self, gpu_index: int) -> Optional[GPUMetrics]:
        """Collect with graceful error handling."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            # ... collection logic
            return GPUMetrics(...)
        except pynvml.NVMLError as e:
            logger.warning(f"GPU {gpu_index} metrics unavailable: {e}")
            return None  # Skip this GPU
        except Exception as e:
            logger.error(f"Unexpected error collecting GPU metrics: {e}")
            return None
```

### 9.2 Multi-GPU Memory Reporting
Fixed issue where VRAM reporting was incorrect for systems with multiple GPUs:
```python
def collect_all_gpus(self) -> List[GPUMetrics]:
    """Collect metrics for all available GPUs."""
    results = []
    for gpu_idx in range(self.device_count):
        metrics = self.collect(gpu_idx)
        if metrics:
            results.append(metrics)
    return results
```

### 9.3 Rate Calculation Stability
Improved I/O rate calculation to handle edge cases:
```python
def collect_disk_io(self) -> DiskIOMetrics:
    current = psutil.disk_io_counters()
    current_time = time.time()
    elapsed = current_time - self.prev_time

    # Guard against division by zero or negative elapsed time
    if elapsed <= 0:
        return self._last_disk_metrics  # Return cached metrics

    # Guard against counter wrap-around
    read_delta = max(0, current.read_bytes - self.prev_disk_io.read_bytes)
    write_delta = max(0, current.write_bytes - self.prev_disk_io.write_bytes)

    read_rate = read_delta / elapsed
    write_rate = write_delta / elapsed
    # ...
```

### 9.4 Connection Recovery
WebSocket reconnection now properly re-subscribes to all channels:
```typescript
socket.on('connect', () => {
  store.setWebSocketConnected(true);
  // Re-subscribe to all system channels after reconnection
  resubscribeToSystemChannels();
});
```

---

*Related: [PRD](../prds/008_FPRD|System_Monitoring.md) | [TID](../tids/008_FTID|System_Monitoring.md) | [FTASKS](../tasks/008_FTASKS|System_Monitoring.md)*
