# Technical Implementation Document: System Monitor

**Feature Number:** 006
**Feature Name:** System Monitor (GPU Monitoring Dashboard)
**Status:** ✅ Implemented
**Created:** 2025-10-18
**Last Updated:** 2025-10-18

---

## 1. Document Overview

### 1.1 Purpose
This Technical Implementation Document (TID) provides practical guidance, code patterns, and best practices for implementing and maintaining the System Monitor feature. It serves as a developer's handbook with concrete examples and implementation tips.

### 1.2 Scope
- Step-by-step implementation guidance
- Code snippets and patterns
- Common pitfalls and solutions
- Testing approaches
- Debugging tips

### 1.3 Related Documents
- **PRD**: `006_FPRD|System_Monitor.md` - What to build
- **TDD**: `006_FTDD|System_Monitor.md` - How it's architected
- **Tasks**: `003_FTASKS|System_Monitor.md` - Implementation tasks

---

## 2. Quick Start Guide

### 2.1 Development Environment Setup

**Backend Setup:**
```bash
cd /home/x-sean/app/miStudio/backend
source venv/bin/activate

# Install dependencies
pip install pynvml==11.5.0 psutil==5.9.5

# Test GPU access
python -c "import pynvml; pynvml.nvmlInit(); print('GPU access OK')"

# Start backend
uvicorn src.main:app --reload --port 8000
```

**Frontend Setup:**
```bash
cd /home/x-sean/app/miStudio/frontend

# Install dependencies (if needed)
npm install

# Start dev server
npm run dev
```

**Test the Feature:**
1. Navigate to http://localhost:3000
2. Click "System Monitor" tab
3. Should see real-time GPU metrics updating

### 2.2 File Locations

**Backend Files:**
- API Router: `backend/src/api/v1/endpoints/system.py`
- Service: `backend/src/services/system_monitor_service.py`
- Schemas: `backend/src/schemas/system.py`
- Utils: `backend/src/utils/resource_estimation.py`

**Frontend Files:**
- Main Component: `frontend/src/components/SystemMonitor/SystemMonitor.tsx`
- Compact Status: `frontend/src/components/SystemMonitor/CompactGPUStatus.tsx`
- Store: `frontend/src/stores/systemMonitorStore.ts`
- Hook: `frontend/src/hooks/useHistoricalData.ts`
- API Client: `frontend/src/api/system.ts`
- Types: `frontend/src/types/system.ts`
- Utils: `frontend/src/utils/metricHelpers.ts`

---

## 3. Implementation Patterns

### 3.1 Backend: Data Collection Service

**Pattern: Safe GPU Metric Collection**

```python
# backend/src/services/system_monitor_service.py
import pynvml
import psutil
from typing import Optional

class SystemMonitorService:
    def __init__(self):
        self.gpu_available = False
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
        except Exception as e:
            print(f"GPU not available: {e}")

    def get_gpu_metrics(self, gpu_id: int = 0) -> Optional[dict]:
        """
        Collect all GPU metrics safely with error handling.
        Returns None if GPU not available.
        """
        if not self.gpu_available:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU
            )

            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

            # Fan
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except:
                fan_speed = 0  # Not all GPUs support fan query

            # Clocks
            gpu_clock = pynvml.nvmlDeviceGetClockInfo(
                handle,
                pynvml.NVML_CLOCK_GRAPHICS
            )
            mem_clock = pynvml.nvmlDeviceGetClockInfo(
                handle,
                pynvml.NVML_CLOCK_MEM
            )
            gpu_max_clock = pynvml.nvmlDeviceGetMaxClockInfo(
                handle,
                pynvml.NVML_CLOCK_GRAPHICS
            )
            mem_max_clock = pynvml.nvmlDeviceGetMaxClockInfo(
                handle,
                pynvml.NVML_CLOCK_MEM
            )

            # Encoder/Decoder (may not be supported on all GPUs)
            try:
                encoder_util, _ = pynvml.nvmlDeviceGetEncoderUtilization(handle)
                decoder_util, _ = pynvml.nvmlDeviceGetDecoderUtilization(handle)
            except:
                encoder_util = 0
                decoder_util = 0

            # GPU Name
            gpu_name = pynvml.nvmlDeviceGetName(handle)

            return {
                "gpu_id": gpu_id,
                "gpu_name": gpu_name,
                "utilization": {
                    "gpu": float(util.gpu),
                    "memory": float(util.memory),
                    "compute": float(util.gpu),  # Same as GPU for most cases
                },
                "memory": {
                    "used_gb": mem_info.used / (1024 ** 3),
                    "total_gb": mem_info.total / (1024 ** 3),
                    "used_percent": (mem_info.used / mem_info.total) * 100,
                },
                "temperature": float(temp),
                "power": {
                    "current_watts": power,
                    "max_watts": power_limit,
                    "usage_percent": (power / power_limit) * 100,
                },
                "fan_speed": float(fan_speed),
                "clocks": {
                    "gpu_mhz": gpu_clock,
                    "gpu_max_mhz": gpu_max_clock,
                    "memory_mhz": mem_clock,
                    "memory_max_mhz": mem_max_clock,
                },
                "encoder_utilization": float(encoder_util),
                "decoder_utilization": float(decoder_util),
            }

        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
            return None

    def get_system_metrics(self) -> dict:
        """Collect system metrics using psutil."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # Memory
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Network I/O
        net_io = psutil.net_io_counters()

        # Disk I/O
        disk_io = psutil.disk_io_counters()

        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "core_count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else 0,
            },
            "memory": {
                "used_gb": mem.used / (1024 ** 3),
                "total_gb": mem.total / (1024 ** 3),
                "used_percent": mem.percent,
                "available_gb": mem.available / (1024 ** 3),
            },
            "swap": {
                "used_gb": swap.used / (1024 ** 3),
                "total_gb": swap.total / (1024 ** 3),
                "used_percent": swap.percent,
            },
            "network": {
                "upload_mbps": net_io.bytes_sent / (1024 ** 2),
                "download_mbps": net_io.bytes_recv / (1024 ** 2),
            },
            "disk_io": {
                "read_mbps": disk_io.read_bytes / (1024 ** 2),
                "write_mbps": disk_io.write_bytes / (1024 ** 2),
            },
        }

    def get_gpu_processes(self, gpu_id: int = 0) -> list:
        """Get list of processes using the GPU."""
        if not self.gpu_available:
            return []

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

            result = []
            for proc in processes:
                try:
                    p = psutil.Process(proc.pid)
                    result.append({
                        "pid": proc.pid,
                        "process_name": p.name(),
                        "gpu_memory_mb": proc.usedGpuMemory / (1024 ** 2),
                        "cpu_percent": p.cpu_percent(interval=0.1),
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return sorted(result, key=lambda x: x["gpu_memory_mb"], reverse=True)

        except Exception as e:
            print(f"Error getting GPU processes: {e}")
            return []

    def get_storage_info(self) -> list:
        """Get disk usage for mounted filesystems."""
        partitions = psutil.disk_partitions()
        result = []

        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                result.append({
                    "mount_point": partition.mountpoint,
                    "used_gb": usage.used / (1024 ** 3),
                    "total_gb": usage.total / (1024 ** 3),
                    "used_percent": usage.percent,
                })
            except PermissionError:
                continue

        return result
```

**Key Points:**
- Always check `gpu_available` before GPU operations
- Wrap GPU calls in try/except (not all features supported on all GPUs)
- Convert units consistently (bytes to GB, mW to W)
- Use `interval=0.1` for psutil CPU to get instant reading

### 3.2 Backend: API Router

**Pattern: Combined Endpoint for Efficiency**

```python
# backend/src/api/v1/endpoints/system.py
from fastapi import APIRouter, HTTPException, Query
from src.services.system_monitor_service import SystemMonitorService
from src.schemas.system import MonitoringDataResponse

router = APIRouter(prefix="/system", tags=["system"])
service = SystemMonitorService()

@router.get("/monitoring/all", response_model=MonitoringDataResponse)
async def get_all_monitoring_data(gpu_id: int = Query(0, ge=0, lt=8)):
    """
    Get all monitoring data in a single call for efficiency.
    Combines GPU metrics, system metrics, processes, and storage.
    """
    try:
        # Get GPU list first
        gpu_list = []
        if service.gpu_available:
            gpu_count = pynvml.nvmlDeviceGetCount()
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_list.append({
                    "id": i,
                    "name": name,
                    "memory_total_gb": mem_info.total / (1024 ** 3),
                    # ... other GPU info
                })

        # Validate GPU ID
        if gpu_id >= len(gpu_list) and service.gpu_available:
            raise HTTPException(status_code=400, detail=f"Invalid GPU ID: {gpu_id}")

        # Collect all data
        gpu_metrics = service.get_gpu_metrics(gpu_id)
        system_metrics = service.get_system_metrics()
        gpu_processes = service.get_gpu_processes(gpu_id)
        storage = service.get_storage_info()

        return {
            "gpu_available": service.gpu_available,
            "gpu_list": gpu_list,
            "gpu_metrics": gpu_metrics,
            "system_metrics": system_metrics,
            "gpu_processes": gpu_processes,
            "storage": storage,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect monitoring data: {str(e)}"
        )
```

**Key Points:**
- Single endpoint reduces HTTP overhead
- Validate GPU ID against available GPUs
- Return all data in one response
- Proper error handling with HTTP status codes

### 3.3 Frontend: Zustand Store

**Pattern: Zustand Store with Persist Middleware**

```typescript
// frontend/src/stores/systemMonitorStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import * as systemApi from '../api/system';
import type { GPUMetrics, SystemMetrics, /* ... */ } from '../types/system';

interface SystemMonitorStore {
  // State
  gpuAvailable: boolean;
  gpuMetrics: GPUMetrics | null;
  systemMetrics: SystemMetrics | null;
  // ... more state

  // Actions
  fetchAllMetrics: () => Promise<void>;
  startPolling: (interval: number) => void;
  stopPolling: () => void;
  // ... more actions
}

export const useSystemMonitorStore = create<SystemMonitorStore>()(
  persist(
    (set, get) => ({
      // Initial State
      gpuAvailable: false,
      gpuList: [],
      selectedGPU: 0,
      gpuMetrics: null,
      systemMetrics: null,
      gpuProcesses: [],
      storageInfo: [],
      viewMode: 'single',
      updateInterval: 1000,
      isPolling: false,
      loading: false,
      error: null,
      errorType: null,
      isConnected: true,
      consecutiveErrors: 0,
      lastSuccessfulFetch: null,
      pollingIntervalId: null,

      // Actions
      fetchAllMetrics: async () => {
        try {
          set({ loading: true });

          const { selectedGPU } = get();
          const response = await systemApi.getAllMonitoringData(selectedGPU);

          set({
            gpuAvailable: response.gpu_available,
            gpuList: response.gpu_list,
            gpuMetrics: response.gpu_metrics,
            systemMetrics: response.system_metrics,
            gpuProcesses: response.gpu_processes,
            storageInfo: response.storage,
            error: null,
            errorType: null,
            isConnected: true,
            consecutiveErrors: 0,
            lastSuccessfulFetch: new Date(),
            loading: false,
          });
        } catch (error: any) {
          const newErrorCount = get().consecutiveErrors + 1;

          // Classify error
          let errorType: ErrorType = 'general';
          if (!error.response) {
            errorType = 'connection';
          } else if (error.response.status === 503) {
            errorType = 'gpu';
          } else if (error.response.status >= 500) {
            errorType = 'api';
          }

          set({
            error: error.response?.data?.detail || error.message,
            errorType,
            isConnected: false,
            consecutiveErrors: newErrorCount,
            loading: false,
          });

          // Stop after 5 failures
          if (newErrorCount >= 5) {
            get().stopPolling();
          }
        }
      },

      startPolling: (interval: number) => {
        const { stopPolling } = get();
        stopPolling(); // Clear existing

        const intervalId = setInterval(() => {
          get().fetchAllMetrics();
        }, interval);

        set({
          isPolling: true,
          updateInterval: interval,
          pollingIntervalId: intervalId,
        });

        // Immediate first fetch
        get().fetchAllMetrics();
      },

      stopPolling: () => {
        const { pollingIntervalId } = get();
        if (pollingIntervalId) {
          clearInterval(pollingIntervalId);
        }
        set({
          isPolling: false,
          pollingIntervalId: null,
        });
      },

      setSelectedGPU: (gpuId: number) => {
        set({ selectedGPU: gpuId });
        get().fetchAllMetrics(); // Immediate update
      },

      setViewMode: (mode: ViewMode) => {
        set({ viewMode: mode });
      },

      setUpdateInterval: (interval: UpdateInterval) => {
        set({ updateInterval: interval });
      },

      retryConnection: () => {
        set({
          consecutiveErrors: 0,
          error: null,
          errorType: null,
        });
        get().startPolling(get().updateInterval);
      },
    }),
    {
      name: 'system-monitor-storage',
      partialize: (state) => ({
        viewMode: state.viewMode,
        selectedGPU: state.selectedGPU,
        updateInterval: state.updateInterval,
      }),
    }
  )
);
```

**Key Points:**
- Use `persist` middleware to save settings to localStorage
- Only persist UI preferences, not metrics
- Always clean up intervals in `stopPolling()`
- Implement consecutive error tracking

### 3.4 Frontend: Safe Metric Display

**Pattern: MetricValue Component**

```typescript
// frontend/src/components/SystemMonitor/MetricValue.tsx
import { isValidMetric } from '@/utils/metricHelpers';

interface MetricValueProps {
  value: number | null | undefined;
  format: 'percent' | 'memory' | 'temperature' | 'power' | 'number';
  decimals?: number;
  unit?: string;
}

export function MetricValue({
  value,
  format,
  decimals = 1,
  unit,
}: MetricValueProps) {
  if (!isValidMetric(value)) {
    return <span className="text-slate-500">N/A</span>;
  }

  const formatted = formatValue(value, format, decimals);
  const displayUnit = unit || getDefaultUnit(format);

  return (
    <span className="font-mono">
      {formatted}
      {displayUnit && (
        <span className="text-slate-400 ml-1">{displayUnit}</span>
      )}
    </span>
  );
}

function formatValue(
  value: number,
  format: string,
  decimals: number
): string {
  switch (format) {
    case 'percent':
      return value.toFixed(decimals);
    case 'memory':
      return value.toFixed(decimals);
    case 'temperature':
      return Math.round(value).toString();
    case 'power':
      return value.toFixed(decimals);
    case 'number':
      return value.toFixed(decimals);
    default:
      return value.toString();
  }
}

function getDefaultUnit(format: string): string {
  switch (format) {
    case 'percent':
      return '%';
    case 'memory':
      return 'GB';
    case 'temperature':
      return '°C';
    case 'power':
      return 'W';
    default:
      return '';
  }
}
```

**Utility Helper:**
```typescript
// frontend/src/utils/metricHelpers.ts
export function isValidMetric(value: any): value is number {
  return (
    typeof value === 'number' &&
    !isNaN(value) &&
    isFinite(value)
  );
}

export function safeGet<T>(
  obj: any,
  path: string,
  defaultValue: T
): T {
  const keys = path.split('.');
  let result = obj;

  for (const key of keys) {
    if (result == null) return defaultValue;
    result = result[key];
  }

  return isValidMetric(result) ? result : defaultValue;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function getTemperatureColor(temp: number | null | undefined): string {
  if (!isValidMetric(temp)) return 'text-slate-400';
  if (temp < 70) return 'text-emerald-400';
  if (temp < 85) return 'text-yellow-400';
  return 'text-red-400';
}

export function getUtilizationColor(util: number | null | undefined): string {
  if (!isValidMetric(util)) return 'bg-slate-700';
  if (util < 70) return 'bg-gradient-to-r from-emerald-500 to-emerald-400';
  if (util < 85) return 'bg-gradient-to-r from-yellow-500 to-yellow-400';
  return 'bg-gradient-to-r from-red-500 to-red-400';
}
```

**Key Points:**
- Always validate metrics before displaying
- Provide N/A fallback for invalid values
- Use helper functions for consistent formatting
- Avoid crashes from null/undefined/NaN

### 3.5 Frontend: Historical Data Management

**Pattern: useHistoricalData Hook**

```typescript
// frontend/src/hooks/useHistoricalData.ts
import { useState, useEffect, useMemo } from 'react';
import { useSystemMonitorStore } from '@/stores/systemMonitorStore';
import type { MetricDataPoint } from '@/types/system';

export function useHistoricalData(timeRange: '1h' | '6h' | '24h') {
  const gpuMetrics = useSystemMonitorStore(state => state.gpuMetrics);
  const systemMetrics = useSystemMonitorStore(state => state.systemMetrics);

  const [dataPoints, setDataPoints] = useState<MetricDataPoint[]>([]);

  // Accumulate data points
  useEffect(() => {
    if (!gpuMetrics || !systemMetrics) return;

    const newPoint: MetricDataPoint = {
      timestamp: new Date(),
      gpu_utilization: gpuMetrics.utilization.gpu,
      cpu_utilization: systemMetrics.cpu.usage_percent,
      gpu_memory_used_gb: gpuMetrics.memory.used_gb,
      ram_used_gb: systemMetrics.memory.used_gb,
      temperature: gpuMetrics.temperature,
    };

    setDataPoints(prev => [...prev, newPoint]);
  }, [gpuMetrics, systemMetrics]);

  // Prune old data every 60 seconds
  useEffect(() => {
    const pruneInterval = setInterval(() => {
      const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24h
      setDataPoints(prev =>
        prev.filter(point => point.timestamp > cutoff)
      );
    }, 60000);

    return () => clearInterval(pruneInterval);
  }, []);

  // Aggregate data based on time range
  const aggregatedData = useMemo(() => {
    const now = Date.now();
    let cutoffTime: number;
    let aggregationMs: number;

    switch (timeRange) {
      case '1h':
        cutoffTime = now - 60 * 60 * 1000;
        aggregationMs = 1000; // No aggregation
        break;
      case '6h':
        cutoffTime = now - 6 * 60 * 60 * 1000;
        aggregationMs = 5000; // 5 seconds
        break;
      case '24h':
        cutoffTime = now - 24 * 60 * 60 * 1000;
        aggregationMs = 15000; // 15 seconds
        break;
    }

    const filtered = dataPoints.filter(
      p => p.timestamp.getTime() > cutoffTime
    );

    if (aggregationMs > 1000) {
      return aggregatePoints(filtered, aggregationMs);
    }

    return filtered;
  }, [dataPoints, timeRange]);

  return aggregatedData;
}

function aggregatePoints(
  points: MetricDataPoint[],
  intervalMs: number
): MetricDataPoint[] {
  const buckets = new Map<number, MetricDataPoint[]>();

  // Group into buckets
  points.forEach(point => {
    const bucketKey = Math.floor(point.timestamp.getTime() / intervalMs);
    if (!buckets.has(bucketKey)) {
      buckets.set(bucketKey, []);
    }
    buckets.get(bucketKey)!.push(point);
  });

  // Average each bucket
  const result: MetricDataPoint[] = [];
  buckets.forEach((bucket, key) => {
    result.push({
      timestamp: new Date(key * intervalMs),
      gpu_utilization: avg(bucket.map(p => p.gpu_utilization)),
      cpu_utilization: avg(bucket.map(p => p.cpu_utilization)),
      gpu_memory_used_gb: avg(bucket.map(p => p.gpu_memory_used_gb)),
      ram_used_gb: avg(bucket.map(p => p.ram_used_gb)),
      temperature: avg(bucket.map(p => p.temperature)),
    });
  });

  return result.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
}

function avg(numbers: number[]): number {
  return numbers.reduce((a, b) => a + b, 0) / numbers.length;
}
```

**Key Points:**
- Accumulate data as metrics arrive
- Prune old data automatically (prevent memory leaks)
- Use `useMemo` for expensive aggregation
- Different granularity for different time ranges

---

## 4. Common Pitfalls & Solutions

### 4.1 Pitfall: Memory Leaks from Polling

**Problem:**
```typescript
// BAD: No cleanup
useEffect(() => {
  setInterval(() => {
    fetchMetrics();
  }, 1000);
}, []);
```

**Solution:**
```typescript
// GOOD: Cleanup interval
useEffect(() => {
  const intervalId = setInterval(() => {
    fetchMetrics();
  }, 1000);

  return () => clearInterval(intervalId);
}, []);
```

### 4.2 Pitfall: GPU Not Available Crashes

**Problem:**
```python
# BAD: Assumes GPU always available
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
```

**Solution:**
```python
# GOOD: Check availability first
if not self.gpu_available:
    return None

try:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
except Exception as e:
    print(f"GPU error: {e}")
    return None
```

### 4.3 Pitfall: NaN/Null Values Crash UI

**Problem:**
```tsx
// BAD: Direct display
<span>{gpuMetrics.temperature}°C</span>
```

**Solution:**
```tsx
// GOOD: Safe display
<MetricValue value={gpuMetrics?.temperature} format="temperature" />
```

### 4.4 Pitfall: Excessive Re-renders

**Problem:**
```typescript
// BAD: Creates new function every render
<button onClick={() => setInterval(1000)}>
```

**Solution:**
```typescript
// GOOD: Memoized callback
const handleClick = useCallback(() => {
  setInterval(1000);
}, [setInterval]);

<button onClick={handleClick}>
```

### 4.5 Pitfall: Chart Performance Degradation

**Problem:**
```tsx
// BAD: Animations with large datasets
<Line type="monotone" dataKey="value" isAnimationActive={true} />
```

**Solution:**
```tsx
// GOOD: Disable animations
<Line
  type="monotone"
  dataKey="value"
  isAnimationActive={false}
  dot={false}
/>
```

---

## 5. Testing Guide

### 5.1 Backend Unit Tests

```python
# tests/unit/test_system_monitor_service.py
import pytest
from unittest.mock import patch, MagicMock
from src.services.system_monitor_service import SystemMonitorService

@pytest.fixture
def service():
    return SystemMonitorService()

@patch('pynvml.nvmlInit')
def test_gpu_not_available(mock_init, service):
    mock_init.side_effect = Exception("NVML not found")
    service.__init__()
    assert service.gpu_available == False

@patch('pynvml.nvmlDeviceGetUtilizationRates')
def test_get_gpu_metrics_success(mock_util, service):
    mock_util.return_value = MagicMock(gpu=75, memory=60)
    metrics = service.get_gpu_metrics(0)
    assert metrics["utilization"]["gpu"] == 75

def test_get_system_metrics(service):
    metrics = service.get_system_metrics()
    assert "cpu" in metrics
    assert "memory" in metrics
    assert metrics["cpu"]["usage_percent"] >= 0
```

### 5.2 Frontend Component Tests

```typescript
// tests/unit/MetricValue.test.tsx
import { render, screen } from '@testing-library/react';
import { MetricValue } from '@/components/SystemMonitor/MetricValue';

describe('MetricValue', () => {
  it('displays formatted percent', () => {
    render(<MetricValue value={75.5} format="percent" />);
    expect(screen.getByText('75.5')).toBeInTheDocument();
    expect(screen.getByText('%')).toBeInTheDocument();
  });

  it('displays N/A for null', () => {
    render(<MetricValue value={null} format="percent" />);
    expect(screen.getByText('N/A')).toBeInTheDocument();
  });
});
```

### 5.3 Manual Testing Checklist

**Basic Functionality:**
- [ ] System Monitor tab loads without errors
- [ ] GPU metrics update every interval
- [ ] System metrics update synchronously
- [ ] Compact status visible in navigation
- [ ] No console errors

**Multi-GPU Testing:**
- [ ] GPU selector shows all GPUs
- [ ] Switching GPUs updates metrics
- [ ] Comparison view shows all GPUs
- [ ] View mode toggle works

**Historical Data:**
- [ ] Charts populate after 10 seconds
- [ ] Time range selector changes view
- [ ] Data persists during navigation

**Error Handling:**
- [ ] No GPU: Shows appropriate message
- [ ] Backend down: Shows error banner with retry
- [ ] After 5 errors: Stops polling automatically

**Performance:**
- [ ] No lag scrolling with 24h of data
- [ ] Memory stable over 1 hour session
- [ ] Switching tabs doesn't break polling

---

## 6. Debugging Tips

### 6.1 Backend Debugging

**Check GPU Access:**
```bash
python -c "import pynvml; pynvml.nvmlInit(); print('GPU OK')"
```

**Test Single Endpoint:**
```bash
curl http://localhost:8000/api/v1/system/monitoring/all?gpu_id=0
```

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 6.2 Frontend Debugging

**Check Store State:**
```typescript
// In console
window.store = useSystemMonitorStore.getState();
console.log(window.store.gpuMetrics);
```

**Monitor Polling:**
```typescript
useEffect(() => {
  console.log('Polling started:', isPolling, 'Interval:', updateInterval);
}, [isPolling, updateInterval]);
```

**Check API Calls:**
```bash
# In browser Network tab
# Filter: /api/v1/system/monitoring/all
# Should see regular requests at configured interval
```

### 6.3 Common Error Messages

**Error**: `GPU not available`
- **Cause**: NVIDIA drivers not installed or GPU not detected
- **Solution**: Check `nvidia-smi` in terminal

**Error**: `Failed to collect GPU metrics`
- **Cause**: pynvml error (GPU busy, driver issue)
- **Solution**: Restart backend, check GPU processes

**Error**: `Connection refused`
- **Cause**: Backend not running
- **Solution**: Start backend with `uvicorn src.main:app --reload`

**Error**: `Memory leak detected`
- **Cause**: Historical data not pruning
- **Solution**: Check pruning interval in `useHistoricalData`

---

## 7. Performance Optimization Tips

### 7.1 Reduce Polling Frequency

```typescript
// For non-critical monitoring
startPolling(2000); // 2 seconds instead of 1 second
```

### 7.2 Lazy Load Charts

```typescript
const Charts = lazy(() => import('./HistoricalCharts'));

<Suspense fallback={<LoadingSkeleton />}>
  <Charts data={historicalData} />
</Suspense>
```

### 7.3 Debounce UI Updates

```typescript
const debouncedMetrics = useMemo(
  () => debounce(gpuMetrics, 100),
  [gpuMetrics]
);
```

### 7.4 Server-Side Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_gpu_info_cached():
    # Static GPU info doesn't change
    return get_gpu_info()
```

---

## 8. Deployment Checklist

### 8.1 Pre-Deployment

- [ ] All tests passing
- [ ] No console errors in production build
- [ ] Environment variables configured
- [ ] NVIDIA drivers installed on target system
- [ ] Backend can access GPU via pynvml

### 8.2 Configuration

**Backend `.env`:**
```bash
ENABLE_GPU_MONITORING=true
GPU_POLL_INTERVAL=1
LOG_LEVEL=INFO
```

**Frontend `.env`:**
```bash
VITE_API_URL=http://mistudio.mcslab.io/api
```

### 8.3 Post-Deployment

- [ ] Test all endpoints returning data
- [ ] Verify polling works correctly
- [ ] Check error handling (disconnect backend temporarily)
- [ ] Monitor memory usage over 24 hours
- [ ] Verify compact status works across all tabs

---

## 9. Future Enhancements

### 9.1 WebSocket Support

Replace polling with WebSocket for true real-time updates:

```python
# backend
from fastapi import WebSocket

@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    while True:
        metrics = get_all_metrics()
        await websocket.send_json(metrics)
        await asyncio.sleep(1)
```

```typescript
// frontend
useEffect(() => {
  const ws = new WebSocket('ws://localhost:8000/api/v1/system/ws/metrics');

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateMetrics(data);
  };

  return () => ws.close();
}, []);
```

### 9.2 Export Metrics

```typescript
function exportMetrics() {
  const csv = convertToCSV(historicalData);
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'gpu-metrics.csv';
  a.click();
}
```

### 9.3 Alert Thresholds

```typescript
interface AlertThreshold {
  metric: string;
  threshold: number;
  enabled: boolean;
}

const [alerts, setAlerts] = useState<AlertThreshold[]>([
  { metric: 'temperature', threshold: 85, enabled: true },
  { metric: 'memory_percent', threshold: 95, enabled: true },
]);

useEffect(() => {
  alerts.forEach(alert => {
    if (alert.enabled && gpuMetrics[alert.metric] > alert.threshold) {
      new Notification(`GPU ${alert.metric} exceeded ${alert.threshold}`);
    }
  });
}, [gpuMetrics, alerts]);
```

---

## 10. Quick Reference

### 10.1 Common Commands

```bash
# Start backend
cd backend && uvicorn src.main:app --reload

# Start frontend
cd frontend && npm run dev

# Test GPU
python -c "import pynvml; pynvml.nvmlInit(); print('OK')"

# Check API
curl http://localhost:8000/api/v1/system/monitoring/all

# Run tests
cd backend && pytest
cd frontend && npm test
```

### 10.2 Key Files

**Backend:**
- `backend/src/api/v1/endpoints/system.py` - API router
- `backend/src/services/system_monitor_service.py` - Data collection

**Frontend:**
- `frontend/src/stores/systemMonitorStore.ts` - State management
- `frontend/src/components/SystemMonitor/SystemMonitor.tsx` - Main component
- `frontend/src/hooks/useHistoricalData.ts` - Data aggregation

### 10.3 Important Constants

```typescript
// Update intervals
const INTERVALS = [500, 1000, 2000, 5000]; // ms

// Time ranges
const TIME_RANGES = ['1h', '6h', '24h'];

// Data retention
const MAX_DATA_AGE = 24 * 60 * 60 * 1000; // 24 hours

// Polling limits
const MAX_CONSECUTIVE_ERRORS = 5;
const PRUNE_INTERVAL = 60000; // 60 seconds
```

---

## Document Metadata

**Version**: 1.0
**Status**: Complete - Implementation Guide
**Last Updated**: 2025-10-18
**Maintainer**: Engineering Team

---

**END OF TECHNICAL IMPLEMENTATION DOCUMENT**
