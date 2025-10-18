# Technical Design Document: System Monitor

**Feature Number:** 006
**Feature Name:** System Monitor (GPU Monitoring Dashboard)
**Status:** ✅ Implemented
**Created:** 2025-10-18
**Last Updated:** 2025-10-18

---

## 1. Document Overview

### 1.1 Purpose
This Technical Design Document (TDD) provides detailed technical specifications for the System Monitor feature in miStudio. It describes the architecture, data models, API contracts, component design, and implementation patterns used to build the real-time GPU and system monitoring dashboard.

### 1.2 Scope
This document covers:
- Backend API design and data collection services
- Frontend component architecture and state management
- Data flow and communication patterns
- Database schemas (if applicable)
- Performance optimization strategies
- Error handling approaches

### 1.3 Audience
- Backend engineers implementing API endpoints
- Frontend engineers building UI components
- DevOps engineers deploying the system
- QA engineers writing test plans

### 1.4 Related Documents
- **PRD**: `006_FPRD|System_Monitor.md`
- **TID**: `006_FTID|System_Monitor.md` (implementation guide)
- **Tasks**: `003_FTASKS|System_Monitor.md`

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │         SystemMonitor Component Tree              │  │
│  │  ┌──────────────────────────────────────────┐    │  │
│  │  │   CompactGPUStatus (in App.tsx nav)      │    │  │
│  │  └──────────────────────────────────────────┘    │  │
│  │  ┌──────────────────────────────────────────┐    │  │
│  │  │   SystemMonitor.tsx (Full Dashboard)     │    │  │
│  │  │   ├── Historical Charts                  │    │  │
│  │  │   ├── Comparison View / Single View      │    │  │
│  │  │   └── Metric Panels                      │    │  │
│  │  └──────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────┘  │
│                       ↓ ↑                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │      Zustand Store (systemMonitorStore.ts)       │  │
│  │  - GPU metrics state                             │  │
│  │  - System metrics state                          │  │
│  │  - UI state (viewMode, interval, etc.)           │  │
│  │  - Polling logic                                 │  │
│  └──────────────────────────────────────────────────┘  │
│                       ↓ ↑                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Polling Mechanism (setInterval)           │  │
│  │  - Configurable interval (0.5s - 5s)             │  │
│  │  - Calls fetchAllMetrics() action                │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓ ↑
                    HTTP REST API
                         ↓ ↑
┌─────────────────────────────────────────────────────────┐
│                  Backend (FastAPI)                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │      API Router (system.py)                      │  │
│  │  GET /api/v1/system/monitoring/all               │  │
│  │  GET /api/v1/system/gpu-metrics                  │  │
│  │  GET /api/v1/system/gpu-list                     │  │
│  │  GET /api/v1/system/metrics                      │  │
│  └──────────────────────────────────────────────────┘  │
│                       ↓                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │   System Monitor Service (system_monitor.py)     │  │
│  │  - get_gpu_metrics()                             │  │
│  │  - get_system_metrics()                          │  │
│  │  - get_gpu_list()                                │  │
│  │  - get_storage_info()                            │  │
│  └──────────────────────────────────────────────────┘  │
│                       ↓                                  │
│  ┌────────────────┬─────────────────────────────────┐  │
│  │   pynvml       │      psutil                     │  │
│  │  (GPU data)    │  (System metrics)               │  │
│  └────────────────┴─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                 Hardware Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   NVIDIA GPU │  │   CPU/RAM    │  │  Disk/Network│  │
│  │   (CUDA)     │  │   (psutil)   │  │   (psutil)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Real-Time Monitoring Flow:**

1. **User Opens System Monitor Tab**
   - `SystemMonitor.tsx` mounts
   - Calls `startPolling(interval)` in store
   - Store initializes `setInterval()` with configured interval

2. **Polling Cycle (every 0.5s - 5s)**
   - Interval fires → calls `fetchAllMetrics()` action
   - Action makes HTTP GET to `/api/v1/system/monitoring/all`
   - Backend queries pynvml and psutil
   - Backend returns combined JSON response
   - Store updates state with new metrics
   - React re-renders affected components

3. **Historical Data Accumulation**
   - `useHistoricalData` hook subscribes to metric updates
   - On each update, appends data point to arrays
   - Every 60 seconds, prunes data older than 24 hours
   - Charts consume aggregated data via `useMemo`

4. **Compact Status (Always Active)**
   - `CompactGPUStatus` component in `App.tsx`
   - Starts own polling at 2s interval on mount
   - Shares same store state
   - Updates independently from full dashboard

**Error Handling Flow:**

1. API call fails → catch block executes
2. Increment `consecutiveErrors` counter
3. Set `errorType` based on error response
4. Display `ErrorBanner` component
5. If `consecutiveErrors >= 5`, stop polling
6. User clicks "Retry" → reset counter, restart polling

### 2.3 Component Hierarchy

```
App.tsx
└── Navigation Bar
    └── CompactGPUStatus.tsx ← Always visible, independent polling

SystemMonitor.tsx (Tab content)
├── Header
│   ├── Title + Live Indicator (inline)
│   ├── ViewModeToggle.tsx
│   └── Settings Button → Opens SettingsModal.tsx
├── ErrorBanner.tsx (conditional)
├── LoadingSkeleton.tsx (conditional, first load only)
├── GPUSelector.tsx (if multiple GPUs)
├── Historical Trends Section
│   ├── Section Header
│   ├── TimeRangeSelector.tsx
│   └── Charts Grid (responsive)
│       ├── UtilizationChart.tsx (GPU + CPU)
│       ├── MemoryUsageChart.tsx (VRAM + RAM)
│       └── TemperatureChart.tsx (GPU temp with thresholds)
├── Comparison View (if viewMode === 'compare')
│   └── Grid of GPUCard.tsx components (one per GPU)
│       └── Each card shows: Utilization, Memory, Temp, Power, Fan
└── Single View (if viewMode === 'single')
    ├── GPU Metrics Grid (2x2 cards)
    │   ├── UtilizationPanel.tsx
    │   ├── TemperaturePanel.tsx
    │   ├── PowerUsagePanel.tsx
    │   └── FanSpeedPanel.tsx
    ├── Hardware Metrics Panel (clock speeds, PCIe, encoder/decoder)
    ├── System Resources Panel (CPU, RAM, swap, network, disk I/O)
    ├── Storage Panel (disk usage for mount points)
    ├── GPU Processes Panel (table of active processes)
    └── System Information Panel (static GPU info)

Shared Components:
├── MetricValue.tsx (safe metric rendering with N/A fallback)
├── MetricWarning.tsx (critical threshold warnings)
└── SettingsModal.tsx (update interval configuration)
```

---

## 3. Data Models & Interfaces

### 3.1 TypeScript Interfaces (Frontend)

**File**: `frontend/src/types/system.ts`

```typescript
// GPU Metrics
export interface GPUMetrics {
  gpu_id: number;
  gpu_name: string;
  utilization: {
    gpu: number;        // 0-100%
    memory: number;     // 0-100%
    compute: number;    // 0-100%
  };
  memory: {
    used_gb: number;
    total_gb: number;
    used_percent: number;
  };
  temperature: number;  // Celsius
  power: {
    current_watts: number;
    max_watts: number;
    usage_percent: number;
  };
  fan_speed: number;    // 0-100%
  clocks: {
    gpu_mhz: number;
    gpu_max_mhz: number;
    memory_mhz: number;
    memory_max_mhz: number;
  };
  pcie: {
    bandwidth_gbps: number;
    generation: number;
    width: number;
  };
  encoder_utilization: number; // 0-100%
  decoder_utilization: number; // 0-100%
}

// GPU Information (Static)
export interface GPUInfo {
  id: number;
  name: string;
  memory_total_gb: number;
  cuda_version: string;
  driver_version: string;
  compute_capability: string;
}

// System Metrics
export interface SystemMetrics {
  cpu: {
    usage_percent: number;
    core_count: number;
    frequency_mhz: number;
  };
  memory: {
    used_gb: number;
    total_gb: number;
    used_percent: number;
    available_gb: number;
  };
  swap: {
    used_gb: number;
    total_gb: number;
    used_percent: number;
  };
  network: {
    upload_mbps: number;
    download_mbps: number;
  };
  disk_io: {
    read_mbps: number;
    write_mbps: number;
  };
}

// GPU Process
export interface GPUProcess {
  pid: number;
  process_name: string;
  gpu_memory_mb: number;
  cpu_percent: number;
}

// Storage Info
export interface StorageInfo {
  mount_point: string;
  used_gb: number;
  total_gb: number;
  used_percent: number;
}

// Combined API Response
export interface MonitoringDataResponse {
  gpu_available: boolean;
  gpu_list: GPUInfo[];
  gpu_metrics: GPUMetrics | null;
  system_metrics: SystemMetrics;
  gpu_processes: GPUProcess[];
  storage: StorageInfo[];
}

// Historical Data Point
export interface MetricDataPoint {
  timestamp: Date;
  gpu_utilization: number;
  cpu_utilization: number;
  gpu_memory_used_gb: number;
  ram_used_gb: number;
  temperature: number;
}

// View Mode
export type ViewMode = 'single' | 'compare';

// Update Interval Options
export type UpdateInterval = 500 | 1000 | 2000 | 5000;

// Error Types
export type ErrorType = 'connection' | 'gpu' | 'api' | 'general';
```

### 3.2 Backend Response Schemas

**File**: `backend/src/schemas/system.py` (example)

```python
from pydantic import BaseModel
from typing import Optional, List

class GPUUtilization(BaseModel):
    gpu: float
    memory: float
    compute: float

class GPUMemory(BaseModel):
    used_gb: float
    total_gb: float
    used_percent: float

class GPUPower(BaseModel):
    current_watts: float
    max_watts: float
    usage_percent: float

class GPUClocks(BaseModel):
    gpu_mhz: int
    gpu_max_mhz: int
    memory_mhz: int
    memory_max_mhz: int

class GPUPCIe(BaseModel):
    bandwidth_gbps: float
    generation: int
    width: int

class GPUMetrics(BaseModel):
    gpu_id: int
    gpu_name: str
    utilization: GPUUtilization
    memory: GPUMemory
    temperature: float
    power: GPUPower
    fan_speed: float
    clocks: GPUClocks
    pcie: GPUPCIe
    encoder_utilization: float
    decoder_utilization: float

class GPUInfo(BaseModel):
    id: int
    name: str
    memory_total_gb: float
    cuda_version: str
    driver_version: str
    compute_capability: str

class CPUMetrics(BaseModel):
    usage_percent: float
    core_count: int
    frequency_mhz: float

class MemoryMetrics(BaseModel):
    used_gb: float
    total_gb: float
    used_percent: float
    available_gb: float

class NetworkMetrics(BaseModel):
    upload_mbps: float
    download_mbps: float

class DiskIOMetrics(BaseModel):
    read_mbps: float
    write_mbps: float

class SystemMetrics(BaseModel):
    cpu: CPUMetrics
    memory: MemoryMetrics
    swap: MemoryMetrics
    network: NetworkMetrics
    disk_io: DiskIOMetrics

class GPUProcess(BaseModel):
    pid: int
    process_name: str
    gpu_memory_mb: float
    cpu_percent: float

class StorageInfo(BaseModel):
    mount_point: str
    used_gb: float
    total_gb: float
    used_percent: float

class MonitoringDataResponse(BaseModel):
    gpu_available: bool
    gpu_list: List[GPUInfo]
    gpu_metrics: Optional[GPUMetrics]
    system_metrics: SystemMetrics
    gpu_processes: List[GPUProcess]
    storage: List[StorageInfo]
```

---

## 4. API Design

### 4.1 Primary Endpoint (Implemented)

**GET /api/v1/system/monitoring/all**

Returns all monitoring data in single response for efficiency.

**Request:**
```http
GET /api/v1/system/monitoring/all?gpu_id=0 HTTP/1.1
```

**Query Parameters:**
- `gpu_id` (optional, int): GPU device ID to query. Defaults to 0 if not specified.

**Response (200 OK):**
```json
{
  "gpu_available": true,
  "gpu_list": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 3090",
      "memory_total_gb": 24.0,
      "cuda_version": "12.1",
      "driver_version": "535.183.01",
      "compute_capability": "8.6"
    }
  ],
  "gpu_metrics": {
    "gpu_id": 0,
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "utilization": {
      "gpu": 78.5,
      "memory": 65.2,
      "compute": 80.1
    },
    "memory": {
      "used_gb": 15.6,
      "total_gb": 24.0,
      "used_percent": 65.0
    },
    "temperature": 72.0,
    "power": {
      "current_watts": 280.5,
      "max_watts": 350.0,
      "usage_percent": 80.1
    },
    "fan_speed": 65.0,
    "clocks": {
      "gpu_mhz": 1800,
      "gpu_max_mhz": 1950,
      "memory_mhz": 9751,
      "memory_max_mhz": 9751
    },
    "pcie": {
      "bandwidth_gbps": 15.75,
      "generation": 4,
      "width": 16
    },
    "encoder_utilization": 0.0,
    "decoder_utilization": 0.0
  },
  "system_metrics": {
    "cpu": {
      "usage_percent": 35.2,
      "core_count": 16,
      "frequency_mhz": 3600
    },
    "memory": {
      "used_gb": 45.8,
      "total_gb": 64.0,
      "used_percent": 71.6,
      "available_gb": 18.2
    },
    "swap": {
      "used_gb": 0.5,
      "total_gb": 8.0,
      "used_percent": 6.25
    },
    "network": {
      "upload_mbps": 2.5,
      "download_mbps": 8.3
    },
    "disk_io": {
      "read_mbps": 120.5,
      "write_mbps": 45.2
    }
  },
  "gpu_processes": [
    {
      "pid": 12345,
      "process_name": "python",
      "gpu_memory_mb": 8192,
      "cpu_percent": 25.5
    },
    {
      "pid": 12346,
      "process_name": "celery",
      "gpu_memory_mb": 4096,
      "cpu_percent": 15.2
    }
  ],
  "storage": [
    {
      "mount_point": "/",
      "used_gb": 450.2,
      "total_gb": 1000.0,
      "used_percent": 45.0
    },
    {
      "mount_point": "/data",
      "used_gb": 2500.5,
      "total_gb": 5000.0,
      "used_percent": 50.0
    }
  ]
}
```

**Error Responses:**

**503 Service Unavailable** (No GPU detected):
```json
{
  "detail": "GPU monitoring not available on this system"
}
```

**500 Internal Server Error** (pynvml failure):
```json
{
  "detail": "Failed to collect GPU metrics: [error message]"
}
```

### 4.2 Legacy Endpoints (Backward Compatibility)

These endpoints still exist for backward compatibility but are not used by the frontend:

**GET /api/v1/system/gpu-list**
- Returns: `List[GPUInfo]`

**GET /api/v1/system/gpu-metrics?gpu_id={id}**
- Returns: `GPUMetrics`

**GET /api/v1/system/metrics**
- Returns: `SystemMetrics`

**GET /api/v1/system/gpu-info/{gpu_id}**
- Returns: `GPUInfo`

---

## 5. State Management

### 5.1 Zustand Store Design

**File**: `frontend/src/stores/systemMonitorStore.ts`

**Store Structure:**

```typescript
interface SystemMonitorStore {
  // ===== GPU State =====
  gpuAvailable: boolean;
  gpuList: GPUInfo[];
  selectedGPU: number;
  gpuMetrics: GPUMetrics | null;

  // ===== System State =====
  systemMetrics: SystemMetrics | null;
  gpuProcesses: GPUProcess[];
  storageInfo: StorageInfo[];

  // ===== UI State =====
  viewMode: ViewMode;
  updateInterval: UpdateInterval;

  // ===== Polling State =====
  isPolling: boolean;
  loading: boolean;

  // ===== Error State =====
  error: string | null;
  errorType: ErrorType | null;
  isConnected: boolean;
  consecutiveErrors: number;
  lastSuccessfulFetch: Date | null;

  // ===== Actions =====
  fetchAllMetrics: () => Promise<void>;
  fetchGPUList: () => Promise<void>;
  setSelectedGPU: (gpuId: number) => void;
  setViewMode: (mode: ViewMode) => void;
  setUpdateInterval: (interval: UpdateInterval) => void;
  startPolling: (interval: UpdateInterval) => void;
  stopPolling: () => void;
  retryConnection: () => void;
}
```

**Persistence:**

Uses `persist` middleware to save to localStorage:
- `viewMode`
- `selectedGPU`
- `updateInterval`

**Key Actions Implementation:**

```typescript
// Primary data fetching action
fetchAllMetrics: async () => {
  try {
    set({ loading: true });

    // Validate selected GPU
    const { selectedGPU, gpuList } = get();
    if (gpuList.length > 0 && !gpuList.find(g => g.id === selectedGPU)) {
      console.warn(`Invalid GPU ${selectedGPU}, falling back to GPU 0`);
      set({ selectedGPU: 0 });
    }

    // Single API call for all data
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
  } catch (error) {
    const errorMessage = error.response?.data?.detail || error.message;
    const newErrorCount = get().consecutiveErrors + 1;

    // Classify error type
    let errorType: ErrorType = 'general';
    if (error.code === 'ECONNREFUSED' || !error.response) {
      errorType = 'connection';
    } else if (error.response?.status === 503) {
      errorType = 'gpu';
    } else if (error.response?.status >= 500) {
      errorType = 'api';
    }

    set({
      error: errorMessage,
      errorType,
      isConnected: false,
      consecutiveErrors: newErrorCount,
      loading: false,
    });

    // Stop polling after 5 consecutive errors
    if (newErrorCount >= 5) {
      get().stopPolling();
    }
  }
},

// Polling management
startPolling: (interval: UpdateInterval) => {
  const { stopPolling } = get();
  stopPolling(); // Clear any existing interval

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

// Retry after error
retryConnection: () => {
  set({
    consecutiveErrors: 0,
    error: null,
    errorType: null,
  });
  get().startPolling(get().updateInterval);
},
```

### 5.2 Historical Data Hook

**File**: `frontend/src/hooks/useHistoricalData.ts`

```typescript
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
      const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24h ago
      setDataPoints(prev =>
        prev.filter(point => point.timestamp > cutoff)
      );
    }, 60000); // Every 60 seconds

    return () => clearInterval(pruneInterval);
  }, []);

  // Aggregate data based on time range
  const aggregatedData = useMemo(() => {
    const now = Date.now();
    let cutoffTime: number;
    let aggregationInterval: number;

    switch (timeRange) {
      case '1h':
        cutoffTime = now - 60 * 60 * 1000;
        aggregationInterval = 1000; // 1 second (no aggregation)
        break;
      case '6h':
        cutoffTime = now - 6 * 60 * 60 * 1000;
        aggregationInterval = 5000; // 5 seconds
        break;
      case '24h':
        cutoffTime = now - 24 * 60 * 60 * 1000;
        aggregationInterval = 15000; // 15 seconds
        break;
    }

    // Filter to time range
    const filtered = dataPoints.filter(
      point => point.timestamp.getTime() > cutoffTime
    );

    // Aggregate if needed
    if (aggregationInterval > 1000) {
      return aggregateDataPoints(filtered, aggregationInterval);
    }

    return filtered;
  }, [dataPoints, timeRange]);

  return aggregatedData;
}

// Helper: Average data points within intervals
function aggregateDataPoints(
  points: MetricDataPoint[],
  intervalMs: number
): MetricDataPoint[] {
  if (points.length === 0) return [];

  const buckets = new Map<number, MetricDataPoint[]>();

  points.forEach(point => {
    const bucketKey = Math.floor(point.timestamp.getTime() / intervalMs);
    if (!buckets.has(bucketKey)) {
      buckets.set(bucketKey, []);
    }
    buckets.get(bucketKey)!.push(point);
  });

  const aggregated: MetricDataPoint[] = [];
  buckets.forEach((bucketPoints, bucketKey) => {
    const avg = {
      timestamp: new Date(bucketKey * intervalMs),
      gpu_utilization: average(bucketPoints.map(p => p.gpu_utilization)),
      cpu_utilization: average(bucketPoints.map(p => p.cpu_utilization)),
      gpu_memory_used_gb: average(bucketPoints.map(p => p.gpu_memory_used_gb)),
      ram_used_gb: average(bucketPoints.map(p => p.ram_used_gb)),
      temperature: average(bucketPoints.map(p => p.temperature)),
    };
    aggregated.push(avg);
  });

  return aggregated.sort((a, b) =>
    a.timestamp.getTime() - b.timestamp.getTime()
  );
}
```

---

## 6. Component Design Patterns

### 6.1 Metric Display Pattern

**Component**: `MetricValue.tsx`

Safe display of metrics with automatic N/A handling:

```typescript
interface MetricValueProps {
  value: number | null | undefined;
  format: 'percent' | 'memory' | 'temperature' | 'power' | 'number';
  decimals?: number;
  unit?: string;
}

export function MetricValue({ value, format, decimals = 1, unit }: MetricValueProps) {
  if (!isValidMetric(value)) {
    return <span className="text-slate-500">N/A</span>;
  }

  const formatted = formatMetricValue(value, format, decimals);
  const displayUnit = unit || getDefaultUnit(format);

  return (
    <span className="font-mono">
      {formatted}
      {displayUnit && <span className="text-slate-400 ml-1">{displayUnit}</span>}
    </span>
  );
}
```

**Usage:**
```tsx
<MetricValue value={gpuMetrics?.utilization.gpu} format="percent" decimals={1} />
<MetricValue value={gpuMetrics?.memory.used_gb} format="memory" unit="GB" />
<MetricValue value={gpuMetrics?.temperature} format="temperature" unit="°C" />
```

### 6.2 Progress Bar Pattern

Common pattern across metric panels:

```tsx
<div className="space-y-2">
  <div className="flex justify-between items-center">
    <span className="text-sm text-slate-400">GPU Utilization</span>
    <MetricValue value={utilization} format="percent" />
  </div>
  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
    <div
      className={`h-full transition-all duration-500 ${getUtilizationColor(utilization)}`}
      style={{ width: `${clamp(utilization, 0, 100)}%` }}
    />
  </div>
  {utilization > 95 && (
    <MetricWarning
      message="High GPU utilization"
      severity="info"
    />
  )}
</div>
```

Color helper:
```typescript
function getUtilizationColor(value: number | null | undefined): string {
  if (!isValidMetric(value)) return 'bg-slate-700';
  if (value < 70) return 'bg-gradient-to-r from-emerald-500 to-emerald-400';
  if (value < 85) return 'bg-gradient-to-r from-yellow-500 to-yellow-400';
  return 'bg-gradient-to-r from-red-500 to-red-400';
}
```

### 6.3 Chart Configuration Pattern

Consistent Recharts setup across all charts:

```tsx
<ResponsiveContainer width="100%" height={300}>
  <LineChart data={chartData}>
    {/* Grid */}
    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />

    {/* X-Axis (Time) */}
    <XAxis
      dataKey="timestamp"
      stroke="#94a3b8"
      tick={{ fill: '#94a3b8', fontSize: 12 }}
      tickFormatter={(timestamp) => format(new Date(timestamp), 'HH:mm')}
    />

    {/* Y-Axis */}
    <YAxis
      stroke="#94a3b8"
      tick={{ fill: '#94a3b8', fontSize: 12 }}
      domain={[0, 100]}
    />

    {/* Tooltip */}
    <Tooltip
      contentStyle={{
        backgroundColor: '#1e293b',
        border: '1px solid #334155',
        borderRadius: '8px',
      }}
      labelFormatter={(timestamp) =>
        format(new Date(timestamp), 'MMM dd, HH:mm:ss')
      }
    />

    {/* Legend */}
    <Legend
      wrapperStyle={{ fontSize: 12 }}
      iconType="line"
    />

    {/* Data Lines */}
    <Line
      type="monotone"
      dataKey="gpu_utilization"
      name="GPU"
      stroke="#a78bfa"
      strokeWidth={2}
      dot={false}
      isAnimationActive={false}
    />
    <Line
      type="monotone"
      dataKey="cpu_utilization"
      name="CPU"
      stroke="#3b82f6"
      strokeWidth={2}
      dot={false}
      isAnimationActive={false}
    />
  </LineChart>
</ResponsiveContainer>
```

**Key Settings:**
- `isAnimationActive={false}` - Disables animations for performance
- `dot={false}` - No dots on data points for cleaner look
- `type="monotone"` - Smooth line interpolation
- Color scheme matches miStudio slate theme

### 6.4 Error Boundary Pattern

**Component**: `ErrorBanner.tsx`

```tsx
interface ErrorBannerProps {
  type: ErrorType;
  message: string;
  isRetrying: boolean;
  onRetry: () => void;
}

export function ErrorBanner({ type, message, isRetrying, onRetry }: ErrorBannerProps) {
  const { icon, title, color } = getErrorConfig(type);

  return (
    <div className={`p-4 rounded-lg border ${color.bg} ${color.border}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {icon}
          <div>
            <div className={`font-medium ${color.text}`}>{title}</div>
            <div className="text-sm text-slate-400 mt-1">{message}</div>
          </div>
        </div>
        <button
          onClick={onRetry}
          disabled={isRetrying}
          className={`px-4 py-2 rounded-lg transition-colors ${
            isRetrying
              ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
              : `${color.buttonBg} ${color.buttonHover} text-white`
          }`}
        >
          {isRetrying ? 'Retrying...' : 'Retry'}
        </button>
      </div>
    </div>
  );
}
```

---

## 7. Performance Optimization

### 7.1 React Optimization

**Memoization Strategy:**

```typescript
// Chart components - prevent re-render unless data changes
export const UtilizationChart = React.memo(({ data }) => {
  // ...
}, (prevProps, nextProps) => {
  return prevProps.data === nextProps.data;
});

// Expensive calculations - cache results
const aggregatedData = useMemo(() => {
  return aggregateDataPoints(dataPoints, aggregationInterval);
}, [dataPoints, aggregationInterval]);

// Event handlers - prevent function recreation
const handleIntervalChange = useCallback((interval: UpdateInterval) => {
  setUpdateInterval(interval);
  startPolling(interval);
}, [setUpdateInterval, startPolling]);
```

**Key Optimizations:**
- All chart components wrapped in `React.memo()`
- Data aggregation uses `useMemo()` to cache expensive calculations
- Event handlers use `useCallback()` to prevent re-creation
- Chart animations disabled: `isAnimationActive={false}`

### 7.2 Data Management

**Memory Leak Prevention:**

```typescript
// Automatic data pruning every 60 seconds
useEffect(() => {
  const pruneInterval = setInterval(() => {
    const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000);
    setDataPoints(prev => prev.filter(point => point.timestamp > cutoff));
  }, 60000);

  return () => clearInterval(pruneInterval);
}, []);

// Cleanup polling on unmount
useEffect(() => {
  return () => {
    stopPolling();
  };
}, [stopPolling]);
```

**Data Aggregation:**
- 1h view: No aggregation (1-second data points)
- 6h view: 5-second aggregation (averaging)
- 24h view: 15-second aggregation (averaging)

### 7.3 API Optimization

**Single Endpoint Strategy:**
- Frontend makes ONE API call per poll cycle
- Backend collects all metrics in parallel
- Response combines GPU + system + process data
- Reduces HTTP overhead significantly

**Configurable Polling:**
- User can select: 0.5s, 1s, 2s, 5s intervals
- Lower frequency = better performance
- Compact status uses 2s independent of main dashboard

---

## 8. Error Handling Architecture

### 8.1 Error Classification

```typescript
type ErrorType = 'connection' | 'gpu' | 'api' | 'general';

function classifyError(error: any): ErrorType {
  if (error.code === 'ECONNREFUSED' || !error.response) {
    return 'connection'; // Backend not reachable
  }
  if (error.response?.status === 503) {
    return 'gpu'; // GPU not available
  }
  if (error.response?.status >= 500) {
    return 'api'; // Backend error
  }
  return 'general'; // Other errors
}
```

### 8.2 Retry Logic

```typescript
// In store's fetchAllMetrics()
catch (error) {
  const newErrorCount = get().consecutiveErrors + 1;

  set({
    error: errorMessage,
    errorType: classifyError(error),
    consecutiveErrors: newErrorCount,
  });

  // Auto-stop after 5 failures to prevent spam
  if (newErrorCount >= 5) {
    get().stopPolling();
  }
}

// Manual retry resets counter
retryConnection: () => {
  set({ consecutiveErrors: 0, error: null });
  get().startPolling(get().updateInterval);
}
```

### 8.3 Graceful Degradation

**No GPU Scenario:**
```tsx
{!gpuAvailable ? (
  <div className="text-center py-12">
    <AlertCircle className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
    <h3 className="text-lg font-medium text-slate-300 mb-2">
      No GPU Detected
    </h3>
    <p className="text-sm text-slate-400">
      System monitoring is available for CPU and memory metrics.
    </p>
  </div>
) : (
  // Full GPU monitoring UI
)}
```

**Missing Metrics:**
```tsx
<MetricValue value={gpuMetrics?.encoder_utilization} format="percent" />
// Automatically displays "N/A" if value is null/undefined/NaN
```

---

## 9. Testing Strategy

### 9.1 Unit Testing

**Backend Services:**
```python
# tests/unit/test_system_monitor_service.py
import pytest
from unittest.mock import patch, MagicMock

@patch('pynvml.nvmlDeviceGetHandleByIndex')
@patch('pynvml.nvmlDeviceGetUtilizationRates')
def test_get_gpu_metrics_success(mock_util, mock_handle):
    mock_handle.return_value = MagicMock()
    mock_util.return_value = MagicMock(gpu=75, memory=60)

    service = SystemMonitorService()
    metrics = service.get_gpu_metrics(gpu_id=0)

    assert metrics.utilization.gpu == 75
    assert metrics.utilization.memory == 60

@patch('pynvml.nvmlInit')
def test_get_gpu_metrics_no_gpu(mock_init):
    mock_init.side_effect = Exception("NVML not available")

    service = SystemMonitorService()
    with pytest.raises(GPUNotAvailableError):
        service.get_gpu_metrics(gpu_id=0)
```

**Frontend Components:**
```typescript
// tests/unit/MetricValue.test.tsx
import { render, screen } from '@testing-library/react';
import { MetricValue } from '@/components/SystemMonitor/MetricValue';

describe('MetricValue', () => {
  it('displays formatted percentage', () => {
    render(<MetricValue value={75.5} format="percent" />);
    expect(screen.getByText('75.5')).toBeInTheDocument();
    expect(screen.getByText('%')).toBeInTheDocument();
  });

  it('displays N/A for null value', () => {
    render(<MetricValue value={null} format="percent" />);
    expect(screen.getByText('N/A')).toBeInTheDocument();
  });

  it('displays N/A for NaN value', () => {
    render(<MetricValue value={NaN} format="percent" />);
    expect(screen.getByText('N/A')).toBeInTheDocument();
  });
});
```

### 9.2 Integration Testing

**API Integration:**
```typescript
// tests/integration/system-api.test.ts
import { systemApi } from '@/api/system';

describe('System API Integration', () => {
  it('fetches all monitoring data successfully', async () => {
    const data = await systemApi.getAllMonitoringData(0);

    expect(data.gpu_available).toBeDefined();
    expect(data.system_metrics).toBeDefined();
    expect(data.system_metrics.cpu.usage_percent).toBeGreaterThanOrEqual(0);
    expect(data.system_metrics.cpu.usage_percent).toBeLessThanOrEqual(100);
  });

  it('handles GPU not available', async () => {
    // Test on system without GPU
    const data = await systemApi.getAllMonitoringData(0);
    expect(data.gpu_available).toBe(false);
    expect(data.gpu_metrics).toBeNull();
  });
});
```

### 9.3 Performance Testing

**Memory Leak Test:**
```typescript
// Run dashboard for 24 hours and monitor memory
describe('Memory Leak Test', () => {
  it('does not accumulate excessive data', async () => {
    const initialMemory = performance.memory.usedJSHeapSize;

    // Simulate 24 hours of data accumulation
    // (with pruning every 60 seconds)
    for (let i = 0; i < 24 * 60; i++) {
      // Add 1 minute of data points
      for (let j = 0; j < 60; j++) {
        addDataPoint(mockMetricData());
      }
      pruneOldData(); // Should keep only 24h max
    }

    const finalMemory = performance.memory.usedJSHeapSize;
    const memoryGrowth = finalMemory - initialMemory;

    // Memory growth should stabilize (< 50MB increase)
    expect(memoryGrowth).toBeLessThan(50 * 1024 * 1024);
  });
});
```

---

## 10. Deployment Considerations

### 10.1 Backend Requirements

**Python Dependencies:**
```txt
# requirements.txt
pynvml==11.5.0          # NVIDIA GPU monitoring
psutil==5.9.5           # System metrics
fastapi==0.104.1        # API framework
pydantic==2.5.0         # Data validation
```

**System Requirements:**
- NVIDIA GPU with CUDA drivers installed
- nvidia-smi accessible
- Python 3.10+

**Environment Variables:**
```bash
# backend/.env
ENABLE_GPU_MONITORING=true  # Set to false to disable GPU features
GPU_POLL_INTERVAL=1         # Backend collection interval (seconds)
```

### 10.2 Frontend Build

**Environment:**
```bash
# frontend/.env
VITE_API_URL=http://localhost:8000
```

**Build Optimization:**
```json
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'recharts': ['recharts'],  // Separate chunk for charts
          'vendor': ['react', 'react-dom', 'zustand'],
        },
      },
    },
  },
});
```

### 10.3 Nginx Configuration

```nginx
# Proxy API requests
location /api/v1/system/ {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

    # No caching for real-time metrics
    proxy_cache_bypass 1;
    proxy_no_cache 1;
}
```

---

## 11. Security Considerations

### 11.1 API Security

**Rate Limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.get("/monitoring/all")
@limiter.limit("10/second")  # Prevent DoS
async def get_all_monitoring_data(gpu_id: int = 0):
    # ...
```

**Input Validation:**
```python
@router.get("/gpu-metrics")
async def get_gpu_metrics(gpu_id: int = Query(0, ge=0, lt=8)):
    # Validates gpu_id is 0-7
    if gpu_id >= len(available_gpus):
        raise HTTPException(status_code=400, detail="Invalid GPU ID")
```

### 11.2 Data Privacy

**No PII Collection:**
- Process names sanitized (show only executable name, not full path)
- No network packet inspection (only bandwidth metrics)
- No logging of sensitive system information

**Local-Only Data:**
- All metrics stay client-side (in-memory)
- No transmission to external services
- Historical data cleared on browser close

---

## 12. Future Enhancements

### 12.1 Planned Improvements

**Backend:**
- WebSocket support for true real-time streaming (instead of polling)
- Metric caching to reduce pynvml overhead
- Historical data persistence (optional PostgreSQL storage)
- Multi-node monitoring (remote GPU servers)

**Frontend:**
- Export metrics to CSV/JSON
- Configurable alert thresholds with browser notifications
- Customizable dashboard layouts (drag-and-drop panels)
- Light/dark theme toggle
- Process management (kill/pause from UI)

**Performance:**
- Server-side data aggregation
- Incremental updates (only changed metrics)
- Service worker for background monitoring

---

## 13. Appendices

### 13.1 File Directory Structure

```
backend/
├── src/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── system.py              # API router
│   ├── services/
│   │   └── system_monitor_service.py      # Data collection
│   ├── schemas/
│   │   └── system.py                      # Pydantic models
│   └── utils/
│       └── resource_estimation.py         # GPU calculations
└── tests/
    └── unit/
        └── test_system_monitor_service.py

frontend/
├── src/
│   ├── components/
│   │   └── SystemMonitor/
│   │       ├── SystemMonitor.tsx          # Root component
│   │       ├── CompactGPUStatus.tsx       # Nav bar status
│   │       ├── ErrorBanner.tsx
│   │       ├── LoadingSkeleton.tsx
│   │       ├── GPUSelector.tsx
│   │       ├── ViewModeToggle.tsx
│   │       ├── TimeRangeSelector.tsx
│   │       ├── SettingsModal.tsx
│   │       ├── GPUCard.tsx
│   │       ├── MetricValue.tsx
│   │       ├── MetricWarning.tsx
│   │       ├── UtilizationPanel.tsx
│   │       ├── TemperaturePanel.tsx
│   │       ├── PowerUsagePanel.tsx
│   │       ├── FanSpeedPanel.tsx
│   │       ├── HardwareMetricsPanel.tsx
│   │       ├── SystemResourcesPanel.tsx
│   │       ├── StoragePanel.tsx
│   │       ├── GPUProcessesPanel.tsx
│   │       ├── SystemInformationPanel.tsx
│   │       ├── UtilizationChart.tsx
│   │       ├── MemoryUsageChart.tsx
│   │       └── TemperatureChart.tsx
│   ├── hooks/
│   │   └── useHistoricalData.ts
│   ├── stores/
│   │   └── systemMonitorStore.ts
│   ├── api/
│   │   └── system.ts                      # API client
│   ├── types/
│   │   └── system.ts                      # TypeScript interfaces
│   └── utils/
│       └── metricHelpers.ts               # Safe access utilities
└── tests/
    ├── unit/
    │   ├── MetricValue.test.tsx
    │   └── systemMonitorStore.test.ts
    └── integration/
        └── system-api.test.ts
```

### 13.2 Glossary

- **pynvml**: Python bindings for NVIDIA Management Library
- **psutil**: Cross-platform library for process and system monitoring
- **Zustand**: Lightweight state management for React
- **Recharts**: React charting library built on D3
- **Polling**: Periodic HTTP requests to fetch updated data
- **Aggregation**: Averaging data points over time intervals

### 13.3 References

- NVIDIA Management Library (NVML) Documentation
- psutil Documentation: https://psutil.readthedocs.io/
- Recharts Documentation: https://recharts.org/
- Zustand Documentation: https://github.com/pmndrs/zustand
- React Performance Optimization: https://react.dev/learn/render-and-commit

---

## Document Metadata

**Version**: 1.0
**Status**: Complete - Reflects Implemented System
**Last Reviewed**: 2025-10-18
**Reviewers**: Engineering Team

**Change Log**:
- 2025-10-18: Initial TDD created from implemented system

---

**END OF TECHNICAL DESIGN DOCUMENT**
