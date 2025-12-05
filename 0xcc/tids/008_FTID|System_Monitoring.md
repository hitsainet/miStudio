# Technical Implementation Document: System Monitoring

**Document ID:** 008_FTID|System_Monitoring
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related TDD:** [008_FTDD|System_Monitoring](../tdds/008_FTDD|System_Monitoring.md)

---

## 1. Implementation Order

### Phase 1: Backend Metrics Collection
1. System monitor service (psutil + pynvml)
2. Metrics data structures
3. Celery Beat task for periodic collection
4. WebSocket emission functions

### Phase 2: WebSocket Infrastructure
1. Define channel conventions
2. Implement emit functions
3. Configure Celery Beat scheduler

### Phase 3: Frontend
1. System monitor store
2. WebSocket subscription hook
3. HTTP polling fallback
4. Chart components (Recharts)

### Phase 4: Integration
1. Connect to panel layout
2. Add to navigation
3. Historical data API

---

## 2. File-by-File Implementation

### 2.1 Backend - Metrics Collection

#### `backend/src/services/system_monitor_service.py`
```python
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except:
    HAS_NVML = False

@dataclass
class GPUMetrics:
    index: int
    name: str
    utilization: float
    memory_used: int
    memory_total: int
    temperature: float
    power_draw: float

@dataclass
class CPUMetrics:
    overall: float
    per_core: List[float]

@dataclass
class MemoryMetrics:
    ram_used: int
    ram_total: int
    swap_used: int
    swap_total: int

@dataclass
class IOMetrics:
    disk_read_rate: float
    disk_write_rate: float
    net_upload_rate: float
    net_download_rate: float

class SystemMonitorService:
    """Collect system metrics using psutil and pynvml."""

    def __init__(self):
        self._last_disk_io = None
        self._last_net_io = None
        self._last_time = None

    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get metrics for all GPUs."""
        if not HAS_NVML:
            return []

        metrics = []
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
            except pynvml.NVMLError:
                power = 0

            metrics.append(GPUMetrics(
                index=i,
                name=name,
                utilization=util.gpu,
                memory_used=mem.used,
                memory_total=mem.total,
                temperature=temp,
                power_draw=power
            ))

        return metrics

    def get_cpu_metrics(self) -> CPUMetrics:
        """Get CPU utilization metrics."""
        per_core = psutil.cpu_percent(percpu=True)
        overall = sum(per_core) / len(per_core) if per_core else 0

        return CPUMetrics(
            overall=overall,
            per_core=per_core
        )

    def get_memory_metrics(self) -> MemoryMetrics:
        """Get memory usage metrics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return MemoryMetrics(
            ram_used=mem.used,
            ram_total=mem.total,
            swap_used=swap.used,
            swap_total=swap.total
        )

    def get_io_metrics(self) -> IOMetrics:
        """Get I/O rate metrics."""
        import time
        current_time = time.time()

        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()

        # Calculate rates
        if self._last_time and self._last_disk_io:
            time_delta = current_time - self._last_time

            disk_read_rate = (disk_io.read_bytes - self._last_disk_io.read_bytes) / time_delta
            disk_write_rate = (disk_io.write_bytes - self._last_disk_io.write_bytes) / time_delta
            net_upload_rate = (net_io.bytes_sent - self._last_net_io.bytes_sent) / time_delta
            net_download_rate = (net_io.bytes_recv - self._last_net_io.bytes_recv) / time_delta
        else:
            disk_read_rate = disk_write_rate = 0
            net_upload_rate = net_download_rate = 0

        # Update last values
        self._last_disk_io = disk_io
        self._last_net_io = net_io
        self._last_time = current_time

        return IOMetrics(
            disk_read_rate=disk_read_rate,
            disk_write_rate=disk_write_rate,
            net_upload_rate=net_upload_rate,
            net_download_rate=net_download_rate
        )

    def get_all_metrics(self) -> Dict:
        """Get all system metrics."""
        return {
            "gpu": [vars(g) for g in self.get_gpu_metrics()],
            "cpu": vars(self.get_cpu_metrics()),
            "memory": vars(self.get_memory_metrics()),
            "io": vars(self.get_io_metrics())
        }
```

**Key Implementation Notes:**
- Handle missing pynvml gracefully (no GPU systems)
- Calculate I/O rates as deltas between collections
- Use dataclasses for type safety

#### `backend/src/workers/system_monitor_tasks.py`
```python
from celery import shared_task
from src.services.system_monitor_service import SystemMonitorService
from src.workers.websocket_emitter import (
    emit_gpu_metrics,
    emit_cpu_metrics,
    emit_memory_metrics,
    emit_disk_metrics,
    emit_network_metrics
)

# Singleton service instance for rate calculations
_monitor_service = None

def get_monitor_service() -> SystemMonitorService:
    global _monitor_service
    if _monitor_service is None:
        _monitor_service = SystemMonitorService()
    return _monitor_service

@shared_task(queue='monitoring')
def collect_system_metrics():
    """Collect and emit system metrics via WebSocket."""
    service = get_monitor_service()

    # GPU metrics (per GPU)
    gpu_metrics = service.get_gpu_metrics()
    for gpu in gpu_metrics:
        emit_gpu_metrics(
            gpu_id=gpu.index,
            utilization=gpu.utilization,
            memory_used=gpu.memory_used,
            memory_total=gpu.memory_total,
            temperature=gpu.temperature,
            power_draw=gpu.power_draw
        )

    # CPU metrics
    cpu = service.get_cpu_metrics()
    emit_cpu_metrics(
        utilization=cpu.overall,
        per_core=cpu.per_core
    )

    # Memory metrics
    mem = service.get_memory_metrics()
    emit_memory_metrics(
        ram_used=mem.ram_used,
        ram_total=mem.ram_total,
        swap_used=mem.swap_used,
        swap_total=mem.swap_total
    )

    # I/O metrics
    io = service.get_io_metrics()
    emit_disk_metrics(
        read_rate=io.disk_read_rate,
        write_rate=io.disk_write_rate
    )
    emit_network_metrics(
        upload_rate=io.net_upload_rate,
        download_rate=io.net_download_rate
    )
```

#### `backend/src/workers/websocket_emitter.py`
```python
import requests
from src.core.config import settings

def emit_to_channel(channel: str, event: str, data: dict):
    """Emit data to WebSocket channel via internal endpoint."""
    try:
        requests.post(
            f"{settings.internal_api_url}/api/internal/ws/emit",
            json={
                "channel": channel,
                "event": event,
                "data": data
            },
            timeout=1
        )
    except Exception as e:
        # Log but don't fail
        print(f"WebSocket emit failed: {e}")

# GPU Metrics
def emit_gpu_metrics(
    gpu_id: int,
    utilization: float,
    memory_used: int,
    memory_total: int,
    temperature: float,
    power_draw: float
):
    """Emit GPU metrics to system/gpu/{id} channel."""
    emit_to_channel(
        channel=f"system/gpu/{gpu_id}",
        event="metrics",
        data={
            "gpu_id": gpu_id,
            "utilization": utilization,
            "memory_used": memory_used,
            "memory_total": memory_total,
            "temperature": temperature,
            "power_draw": power_draw,
            "timestamp": get_timestamp()
        }
    )

# CPU Metrics
def emit_cpu_metrics(utilization: float, per_core: list):
    """Emit CPU metrics to system/cpu channel."""
    emit_to_channel(
        channel="system/cpu",
        event="metrics",
        data={
            "utilization": utilization,
            "per_core": per_core,
            "timestamp": get_timestamp()
        }
    )

# Memory Metrics
def emit_memory_metrics(ram_used: int, ram_total: int, swap_used: int, swap_total: int):
    """Emit memory metrics to system/memory channel."""
    emit_to_channel(
        channel="system/memory",
        event="metrics",
        data={
            "ram_used": ram_used,
            "ram_total": ram_total,
            "swap_used": swap_used,
            "swap_total": swap_total,
            "timestamp": get_timestamp()
        }
    )

# Disk I/O Metrics
def emit_disk_metrics(read_rate: float, write_rate: float):
    """Emit disk I/O metrics to system/disk channel."""
    emit_to_channel(
        channel="system/disk",
        event="metrics",
        data={
            "read_rate": read_rate,
            "write_rate": write_rate,
            "timestamp": get_timestamp()
        }
    )

# Network Metrics
def emit_network_metrics(upload_rate: float, download_rate: float):
    """Emit network metrics to system/network channel."""
    emit_to_channel(
        channel="system/network",
        event="metrics",
        data={
            "upload_rate": upload_rate,
            "download_rate": download_rate,
            "timestamp": get_timestamp()
        }
    )

def get_timestamp():
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"
```

#### `backend/src/core/celery_app.py` (Excerpt)
```python
from celery import Celery
from celery.schedules import crontab
from src.core.config import settings

celery_app = Celery(
    "mistudio",
    broker=settings.redis_url,
    backend=settings.redis_url
)

# Celery Beat schedule for system monitoring
celery_app.conf.beat_schedule = {
    'collect-system-metrics': {
        'task': 'src.workers.system_monitor_tasks.collect_system_metrics',
        'schedule': settings.system_monitor_interval_seconds,  # Default: 2.0
    },
}

celery_app.conf.task_routes = {
    'src.workers.system_monitor_tasks.*': {'queue': 'monitoring'},
    'src.workers.training_tasks.*': {'queue': 'sae'},
    'src.workers.export_tasks.*': {'queue': 'export'},
}

celery_app.autodiscover_tasks([
    'src.workers.training_tasks',
    'src.workers.dataset_tasks',
    'src.workers.model_tasks',
    'src.workers.system_monitor_tasks',
    'src.workers.neuronpedia_tasks',
])
```

### 2.2 Frontend - Store and Hook

#### `frontend/src/stores/systemMonitorStore.ts`
```typescript
import { create } from 'zustand';
import { systemApi } from '../api/system';

interface GPUMetrics {
  gpu_id: number;
  utilization: number;
  memory_used: number;
  memory_total: number;
  temperature: number;
  power_draw: number;
  timestamp: string;
}

interface CPUMetrics {
  utilization: number;
  per_core: number[];
  timestamp: string;
}

interface MemoryMetrics {
  ram_used: number;
  ram_total: number;
  swap_used: number;
  swap_total: number;
  timestamp: string;
}

interface IOMetrics {
  read_rate?: number;
  write_rate?: number;
  upload_rate?: number;
  download_rate?: number;
  timestamp: string;
}

interface SystemMonitorState {
  // Current metrics
  gpuMetrics: Record<number, GPUMetrics>;
  cpuMetrics: CPUMetrics | null;
  memoryMetrics: MemoryMetrics | null;
  diskMetrics: IOMetrics | null;
  networkMetrics: IOMetrics | null;

  // Historical data (1 hour)
  gpuHistory: Record<number, GPUMetrics[]>;
  cpuHistory: CPUMetrics[];
  memoryHistory: MemoryMetrics[];

  // Connection state
  isWebSocketConnected: boolean;
  pollingIntervalId: number | null;

  // Actions
  setGpuMetrics: (metrics: GPUMetrics) => void;
  setCpuMetrics: (metrics: CPUMetrics) => void;
  setMemoryMetrics: (metrics: MemoryMetrics) => void;
  setDiskMetrics: (metrics: IOMetrics) => void;
  setNetworkMetrics: (metrics: IOMetrics) => void;
  setIsWebSocketConnected: (connected: boolean) => void;
  startPollingFallback: () => void;
  stopPollingFallback: () => void;
  fetchCurrentMetrics: () => Promise<void>;
}

const MAX_HISTORY_POINTS = 1800;  // 1 hour at 2-second intervals

export const useSystemMonitorStore = create<SystemMonitorState>((set, get) => ({
  gpuMetrics: {},
  cpuMetrics: null,
  memoryMetrics: null,
  diskMetrics: null,
  networkMetrics: null,
  gpuHistory: {},
  cpuHistory: [],
  memoryHistory: [],
  isWebSocketConnected: false,
  pollingIntervalId: null,

  setGpuMetrics: (metrics) => {
    set(state => {
      const history = state.gpuHistory[metrics.gpu_id] || [];
      const newHistory = [...history, metrics].slice(-MAX_HISTORY_POINTS);

      return {
        gpuMetrics: { ...state.gpuMetrics, [metrics.gpu_id]: metrics },
        gpuHistory: { ...state.gpuHistory, [metrics.gpu_id]: newHistory }
      };
    });
  },

  setCpuMetrics: (metrics) => {
    set(state => ({
      cpuMetrics: metrics,
      cpuHistory: [...state.cpuHistory, metrics].slice(-MAX_HISTORY_POINTS)
    }));
  },

  setMemoryMetrics: (metrics) => {
    set(state => ({
      memoryMetrics: metrics,
      memoryHistory: [...state.memoryHistory, metrics].slice(-MAX_HISTORY_POINTS)
    }));
  },

  setDiskMetrics: (metrics) => set({ diskMetrics: metrics }),
  setNetworkMetrics: (metrics) => set({ networkMetrics: metrics }),

  setIsWebSocketConnected: (connected) => {
    set({ isWebSocketConnected: connected });

    // Auto-manage polling fallback
    if (connected) {
      get().stopPollingFallback();
    } else {
      get().startPollingFallback();
    }
  },

  startPollingFallback: () => {
    const { pollingIntervalId } = get();
    if (pollingIntervalId) return;  // Already polling

    const id = window.setInterval(() => {
      get().fetchCurrentMetrics();
    }, 2000);

    set({ pollingIntervalId: id });
  },

  stopPollingFallback: () => {
    const { pollingIntervalId } = get();
    if (pollingIntervalId) {
      window.clearInterval(pollingIntervalId);
      set({ pollingIntervalId: null });
    }
  },

  fetchCurrentMetrics: async () => {
    try {
      const data = await systemApi.getCurrentMetrics();

      // Update all metrics
      data.gpu?.forEach((gpu: GPUMetrics) => get().setGpuMetrics(gpu));
      if (data.cpu) get().setCpuMetrics(data.cpu);
      if (data.memory) get().setMemoryMetrics(data.memory);
      if (data.disk) get().setDiskMetrics(data.disk);
      if (data.network) get().setNetworkMetrics(data.network);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  }
}));
```

#### `frontend/src/hooks/useSystemMonitorWebSocket.ts`
```typescript
import { useEffect } from 'react';
import { socket } from '../api/websocket';
import { useSystemMonitorStore } from '../stores/systemMonitorStore';

export function useSystemMonitorWebSocket() {
  const {
    setGpuMetrics,
    setCpuMetrics,
    setMemoryMetrics,
    setDiskMetrics,
    setNetworkMetrics,
    setIsWebSocketConnected
  } = useSystemMonitorStore();

  useEffect(() => {
    // Subscribe to all system channels
    const channels = [
      'system/gpu/0',  // Add more GPUs as needed
      'system/cpu',
      'system/memory',
      'system/disk',
      'system/network'
    ];

    channels.forEach(channel => {
      socket.emit('join', channel);
    });

    // Handle metrics events
    const handleMetrics = (data: any) => {
      if (data.gpu_id !== undefined) {
        setGpuMetrics(data);
      } else if (data.per_core !== undefined) {
        setCpuMetrics(data);
      } else if (data.ram_used !== undefined) {
        setMemoryMetrics(data);
      } else if (data.read_rate !== undefined) {
        setDiskMetrics(data);
      } else if (data.upload_rate !== undefined) {
        setNetworkMetrics(data);
      }
    };

    // Connection state handlers
    const handleConnect = () => setIsWebSocketConnected(true);
    const handleDisconnect = () => setIsWebSocketConnected(false);

    socket.on('metrics', handleMetrics);
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);

    // Set initial connection state
    setIsWebSocketConnected(socket.connected);

    return () => {
      channels.forEach(channel => {
        socket.emit('leave', channel);
      });
      socket.off('metrics', handleMetrics);
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
    };
  }, []);
}
```

### 2.3 Frontend - Chart Components

#### `frontend/src/components/SystemMonitor/UtilizationChart.tsx`
```typescript
import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';

interface UtilizationChartProps {
  gpuId?: number;
}

export function UtilizationChart({ gpuId = 0 }: UtilizationChartProps) {
  const gpuHistory = useSystemMonitorStore(s => s.gpuHistory[gpuId] || []);

  // Format data for chart
  const data = gpuHistory.map(m => ({
    time: new Date(m.timestamp).toLocaleTimeString(),
    utilization: m.utilization,
    temperature: m.temperature
  }));

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h3 className="text-sm font-medium text-slate-400 mb-2">
        GPU {gpuId} - Utilization & Temperature
      </h3>

      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <XAxis
            dataKey="time"
            stroke="#64748b"
            tick={{ fontSize: 10 }}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="left"
            stroke="#64748b"
            domain={[0, 100]}
            tick={{ fontSize: 10 }}
            label={{ value: '%', position: 'insideLeft', offset: -5 }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#64748b"
            domain={[0, 100]}
            tick={{ fontSize: 10 }}
            label={{ value: '°C', position: 'insideRight', offset: 5 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: 'none',
              borderRadius: '8px'
            }}
            formatter={(value: number, name: string) => [
              `${value.toFixed(1)}${name === 'utilization' ? '%' : '°C'}`,
              name === 'utilization' ? 'Utilization' : 'Temperature'
            ]}
          />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="utilization"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
            name="Utilization"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="temperature"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            name="Temperature"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

#### `frontend/src/components/SystemMonitor/SystemMonitor.tsx`
```typescript
import React, { useEffect } from 'react';
import { useSystemMonitorWebSocket } from '../../hooks/useSystemMonitorWebSocket';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
import { UtilizationChart } from './UtilizationChart';
import { MemoryChart } from './MemoryChart';
import { IOChart } from './IOChart';
import { formatBytes } from '../../utils/formatters';

export function SystemMonitor() {
  // Initialize WebSocket subscription
  useSystemMonitorWebSocket();

  const {
    gpuMetrics,
    cpuMetrics,
    memoryMetrics,
    diskMetrics,
    networkMetrics,
    isWebSocketConnected,
    fetchCurrentMetrics
  } = useSystemMonitorStore();

  // Fetch initial data on mount
  useEffect(() => {
    fetchCurrentMetrics();
  }, []);

  return (
    <div className="space-y-4">
      {/* Connection indicator */}
      <div className="flex items-center gap-2 text-sm">
        <div className={`w-2 h-2 rounded-full ${
          isWebSocketConnected ? 'bg-green-500' : 'bg-yellow-500'
        }`} />
        <span className="text-slate-400">
          {isWebSocketConnected ? 'Live' : 'Polling'}
        </span>
      </div>

      {/* GPU Section */}
      {Object.keys(gpuMetrics).map(gpuId => (
        <div key={gpuId}>
          <UtilizationChart gpuId={parseInt(gpuId)} />

          {/* Current GPU stats */}
          <div className="grid grid-cols-4 gap-2 mt-2 text-sm">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Utilization</div>
              <div className="text-lg font-medium">
                {gpuMetrics[parseInt(gpuId)]?.utilization?.toFixed(0)}%
              </div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Memory</div>
              <div className="text-lg font-medium">
                {formatBytes(gpuMetrics[parseInt(gpuId)]?.memory_used || 0)}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Temperature</div>
              <div className="text-lg font-medium">
                {gpuMetrics[parseInt(gpuId)]?.temperature?.toFixed(0)}°C
              </div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Power</div>
              <div className="text-lg font-medium">
                {gpuMetrics[parseInt(gpuId)]?.power_draw?.toFixed(0)}W
              </div>
            </div>
          </div>
        </div>
      ))}

      {/* Memory and CPU grid */}
      <div className="grid grid-cols-2 gap-4">
        <MemoryChart />

        {/* CPU per-core */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-slate-400 mb-2">
            CPU Utilization
          </h3>
          <div className="space-y-1">
            {cpuMetrics?.per_core?.map((util, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs w-12">Core {i}</span>
                <div className="flex-1 h-2 bg-slate-700 rounded overflow-hidden">
                  <div
                    className="h-full bg-emerald-500"
                    style={{ width: `${util}%` }}
                  />
                </div>
                <span className="text-xs w-10 text-right">{util.toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* I/O Section */}
      <IOChart
        diskMetrics={diskMetrics}
        networkMetrics={networkMetrics}
      />
    </div>
  );
}
```

---

## 3. Common Patterns

### 3.1 Automatic Polling Fallback
```typescript
// In store: automatically switch between WebSocket and polling
setIsWebSocketConnected: (connected) => {
  set({ isWebSocketConnected: connected });

  if (connected) {
    get().stopPollingFallback();  // WebSocket active, stop polling
  } else {
    get().startPollingFallback();  // WebSocket disconnected, start polling
  }
}
```

### 3.2 History Trimming
```typescript
// Keep only last N data points (e.g., 1 hour of 2-second intervals)
const MAX_HISTORY = 1800;

setMetrics: (metrics) => {
  set(state => ({
    history: [...state.history, metrics].slice(-MAX_HISTORY)
  }));
}
```

---

## 4. Testing Strategy

### 4.1 Backend Tests
```python
# backend/tests/test_system_monitor_service.py
def test_get_gpu_metrics():
    service = SystemMonitorService()
    metrics = service.get_gpu_metrics()

    # May be empty if no GPU
    for gpu in metrics:
        assert 0 <= gpu.utilization <= 100
        assert gpu.memory_used <= gpu.memory_total

def test_get_cpu_metrics():
    service = SystemMonitorService()
    metrics = service.get_cpu_metrics()

    assert 0 <= metrics.overall <= 100
    assert len(metrics.per_core) > 0

def test_io_rate_calculation():
    service = SystemMonitorService()

    # First call initializes
    first = service.get_io_metrics()
    assert first.disk_read_rate == 0  # No previous data

    import time
    time.sleep(0.1)

    # Second call calculates rate
    second = service.get_io_metrics()
    assert second.disk_read_rate >= 0  # Now has rate
```

### 4.2 Frontend Tests
```typescript
// frontend/src/stores/systemMonitorStore.test.ts
import { useSystemMonitorStore } from './systemMonitorStore';

test('setGpuMetrics updates current and history', () => {
  const store = useSystemMonitorStore.getState();

  store.setGpuMetrics({
    gpu_id: 0,
    utilization: 50,
    memory_used: 1000000,
    memory_total: 8000000,
    temperature: 70,
    power_draw: 150,
    timestamp: new Date().toISOString()
  });

  const state = useSystemMonitorStore.getState();
  expect(state.gpuMetrics[0].utilization).toBe(50);
  expect(state.gpuHistory[0].length).toBe(1);
});
```

---

## 5. Common Pitfalls

### Pitfall 1: pynvml Not Initialized
```python
# WRONG - Crashes if no GPU
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# RIGHT - Handle gracefully
try:
    pynvml.nvmlInit()
    HAS_NVML = True
except:
    HAS_NVML = False

def get_gpu_metrics():
    if not HAS_NVML:
        return []
    # ... proceed
```

### Pitfall 2: Memory Leak in History
```typescript
// WRONG - Unbounded history growth
setMetrics: (m) => set(s => ({ history: [...s.history, m] }))

// RIGHT - Bounded history
setMetrics: (m) => set(s => ({
  history: [...s.history, m].slice(-MAX_HISTORY)
}))
```

### Pitfall 3: Race Condition in Polling
```typescript
// WRONG - Multiple intervals can stack
startPolling: () => {
  setInterval(fetch, 2000);  // Called multiple times!
}

// RIGHT - Check before starting
startPolling: () => {
  if (get().pollingIntervalId) return;
  const id = setInterval(fetch, 2000);
  set({ pollingIntervalId: id });
}
```

---

## 6. Performance Tips

1. **Batch WebSocket Emissions**
   ```python
   # Instead of 5 separate emissions, batch into one
   emit_to_channel("system/all", "metrics", {
       "gpu": gpu_metrics,
       "cpu": cpu_metrics,
       "memory": memory_metrics,
       "disk": disk_metrics,
       "network": network_metrics
   })
   ```

2. **Debounce Chart Updates**
   ```typescript
   // Don't re-render chart on every data point
   const debouncedData = useMemo(() => {
     // Sample every Nth point for display
     return history.filter((_, i) => i % 10 === 0);
   }, [history.length]);
   ```

3. **Use Canvas for High-Frequency Updates**
   ```typescript
   // For 60fps monitoring, use Canvas instead of SVG
   import { ResponsiveContainer } from 'recharts';
   // ... or use a Canvas-based library like chart.js
   ```

---

*Related: [PRD](../prds/008_FPRD|System_Monitoring.md) | [TDD](../tdds/008_FTDD|System_Monitoring.md) | [FTASKS](../tasks/008_FTASKS|System_Monitoring.md)*
