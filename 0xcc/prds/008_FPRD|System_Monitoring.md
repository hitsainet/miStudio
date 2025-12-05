# Feature PRD: System Monitoring

**Document ID:** 008_FPRD|System_Monitoring
**Version:** 1.0 (MVP Complete)
**Last Updated:** 2025-12-05
**Status:** Implemented
**Priority:** P1 (Important Feature)

---

## 1. Overview

### 1.1 Purpose
Provide real-time system resource monitoring during long-running operations like training and extraction.

### 1.2 User Problem
Researchers need visibility into system resources because:
- Training can consume significant GPU memory
- Memory leaks can cause job failures
- Resource bottlenecks affect training speed
- Temperature monitoring prevents thermal throttling

### 1.3 Solution
A comprehensive system monitoring dashboard with real-time GPU, CPU, memory, disk, and network metrics streamed via WebSocket.

---

## 2. Functional Requirements

### 2.1 GPU Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | GPU utilization percentage | Implemented |
| FR-1.2 | GPU memory used/total | Implemented |
| FR-1.3 | GPU temperature | Implemented |
| FR-1.4 | GPU power draw | Implemented |
| FR-1.5 | Per-GPU metrics (multi-GPU) | Implemented |

### 2.2 CPU Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | Overall CPU utilization | Implemented |
| FR-2.2 | Per-core utilization | Implemented |
| FR-2.3 | CPU frequency | Planned |

### 2.3 Memory Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | RAM used/total | Implemented |
| FR-3.2 | Swap used/total | Implemented |
| FR-3.3 | Memory pressure indicator | Planned |

### 2.4 I/O Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | Disk read/write rates (MB/s) | Implemented |
| FR-4.2 | Network upload/download rates (MB/s) | Implemented |

### 2.5 Visualization
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-5.1 | Real-time updating charts | Implemented |
| FR-5.2 | 1-hour rolling history | Implemented |
| FR-5.3 | Combined utilization + temperature chart | Implemented |
| FR-5.4 | Grid layout for multiple metrics | Implemented |

---

## 3. Monitoring Architecture

### 3.1 Data Collection
```
Celery Beat (every 2s)
       │
       ▼
┌──────────────────┐
│ collect_system_  │
│ metrics task     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│  GPU Metrics     │     │  System Metrics  │
│  (pynvml)        │     │  (psutil)        │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
         ┌──────────────────┐
         │  WebSocket Emit  │
         │  (per channel)   │
         └────────┬─────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
system/gpu/0  system/cpu  system/memory
```

### 3.2 Fallback Pattern
- Primary: WebSocket streaming
- Fallback: HTTP polling (on disconnect)
- Auto-switch on reconnection

---

## 4. User Interface

### 4.1 System Monitor Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│ System Monitor                                              │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐ ┌─────────────────────────────┐ │
│ │ GPU Utilization & Temp  │ │ Memory Usage                │ │
│ │ ┌─────────────────────┐ │ │ ┌─────────────────────────┐ │ │
│ │ │     ▄▄▄▄████████    │ │ │ │ RAM: ████████░░ 80%     │ │ │
│ │ │   ▄█████████████    │ │ │ │ 12.8 GB / 16 GB         │ │ │
│ │ │ ▄████████████████   │ │ │ │                         │ │ │
│ │ │ Util: 75% | Temp: 72°│ │ │ │ Swap: ███░░░░░░ 30%     │ │ │
│ │ └─────────────────────┘ │ │ │ 2.4 GB / 8 GB           │ │ │
│ └─────────────────────────┘ │ └─────────────────────────┘ │ │
│                             └─────────────────────────────┘ │
│ ┌─────────────────────────┐ ┌─────────────────────────────┐ │
│ │ CPU Utilization         │ │ I/O Rates                   │ │
│ │ ┌─────────────────────┐ │ │ ┌─────────────────────────┐ │ │
│ │ │ Core 0: ████████░░  │ │ │ │ Disk R: 150 MB/s        │ │ │
│ │ │ Core 1: █████░░░░░  │ │ │ │ Disk W: 45 MB/s         │ │ │
│ │ │ Core 2: ██████████  │ │ │ │ Net ↓: 2.3 MB/s         │ │ │
│ │ │ Core 3: ███░░░░░░░  │ │ │ │ Net ↑: 0.5 MB/s         │ │ │
│ │ └─────────────────────┘ │ │ └─────────────────────────┘ │ │
│ └─────────────────────────┘ └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 GPU Memory Widget (Sidebar)
```
┌──────────────────┐
│ GPU Memory       │
│ ██████████░░░░░░ │
│ 6.2 / 8.0 GB     │
└──────────────────┘
```

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/system/metrics` | GET | Current system metrics |
| `/api/v1/system/history` | GET | Historical metrics (1 hour) |
| `/api/v1/system/gpu` | GET | GPU-specific metrics |

---

## 6. WebSocket Channels

| Channel | Events | Payload | Interval |
|---------|--------|---------|----------|
| `system/gpu/{id}` | `metrics` | `{utilization, memory_used, memory_total, temperature, power}` | 2s |
| `system/cpu` | `metrics` | `{utilization, per_core: [...]}` | 2s |
| `system/memory` | `metrics` | `{ram_used, ram_total, swap_used, swap_total}` | 2s |
| `system/disk` | `metrics` | `{read_rate, write_rate}` | 2s |
| `system/network` | `metrics` | `{upload_rate, download_rate}` | 2s |

---

## 7. Key Files

### Backend
- `backend/src/services/system_monitor_service.py` - Metrics collection
- `backend/src/workers/system_monitor_tasks.py` - Celery Beat task
- `backend/src/workers/websocket_emitter.py` - WebSocket emission
- `backend/src/api/v1/endpoints/system.py` - API routes

### Frontend
- `frontend/src/components/SystemMonitor/SystemMonitor.tsx` - Main dashboard
- `frontend/src/components/SystemMonitor/UtilizationChart.tsx` - GPU chart
- `frontend/src/hooks/useSystemMonitorWebSocket.ts` - WebSocket hook
- `frontend/src/stores/systemMonitorStore.ts` - Zustand store

---

## 8. Metrics Collection

### 8.1 GPU Metrics (pynvml)
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)

utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
temperature = pynvml.nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
```

### 8.2 System Metrics (psutil)
```python
import psutil

cpu_percent = psutil.cpu_percent(percpu=True)
memory = psutil.virtual_memory()
swap = psutil.swap_memory()
disk_io = psutil.disk_io_counters()
net_io = psutil.net_io_counters()
```

---

## 9. Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `system_monitor_interval_seconds` | Metrics collection interval | 2 |
| `system_monitor_history_hours` | History retention | 1 |

---

## 10. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| SAE Training | Monitors during training |
| Feature Discovery | Monitors during extraction |

---

## 11. Testing Checklist

- [x] GPU metrics collection
- [x] CPU metrics per core
- [x] Memory metrics
- [x] Disk I/O rates
- [x] Network I/O rates
- [x] WebSocket streaming
- [x] HTTP polling fallback
- [x] Chart visualization
- [x] 1-hour history

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/008_FTDD|System_Monitoring.md) | [TID](../tids/008_FTID|System_Monitoring.md)*
