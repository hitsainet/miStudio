# Feature Tasks: System Monitoring

**Document ID:** 008_FTASKS|System_Monitoring
**Version:** 1.1
**Last Updated:** 2025-12-16
**Status:** Implemented
**Related PRD:** [008_FPRD|System_Monitoring](../prds/008_FPRD|System_Monitoring.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Metrics Collection | 4 tasks | ✅ Complete |
| Phase 2: WebSocket Infrastructure | 4 tasks | ✅ Complete |
| Phase 3: Celery Beat | 3 tasks | ✅ Complete |
| Phase 4: API Endpoints | 2 tasks | ✅ Complete |
| Phase 5: Frontend Store | 3 tasks | ✅ Complete |
| Phase 6: UI Components | 5 tasks | ✅ Complete |

**Total: 21 tasks**

---

## Phase 1: Metrics Collection

### Task 1.1: Create System Monitor Service
- [x] Initialize pynvml
- [x] Handle missing GPU gracefully
- [x] Store last values for rate calculation

**Files:**
- `backend/src/services/system_monitor_service.py`

### Task 1.2: Implement GPU Metrics
- [x] Get GPU count
- [x] Get utilization per GPU
- [x] Get memory usage
- [x] Get temperature
- [x] Get power draw

### Task 1.3: Implement CPU/Memory Metrics
- [x] Get overall CPU utilization
- [x] Get per-core utilization
- [x] Get RAM usage
- [x] Get swap usage

### Task 1.4: Implement I/O Metrics
- [x] Get disk read/write rates
- [x] Get network up/down rates
- [x] Calculate deltas between samples

---

## Phase 2: WebSocket Infrastructure

### Task 2.1: Define Channel Conventions
- [x] system/gpu/{id} for per-GPU
- [x] system/cpu for CPU
- [x] system/memory for RAM/Swap
- [x] system/disk for disk I/O
- [x] system/network for network I/O

### Task 2.2: Create Emission Functions
- [x] emit_gpu_metrics()
- [x] emit_cpu_metrics()
- [x] emit_memory_metrics()
- [x] emit_disk_metrics()
- [x] emit_network_metrics()

**Files:**
- `backend/src/workers/websocket_emitter.py`

### Task 2.3: Add Timestamp
- [x] Include ISO timestamp
- [x] Use UTC timezone

### Task 2.4: Internal Emit Endpoint
- [x] POST /api/internal/ws/emit
- [x] Accept channel, event, data
- [x] Route to Socket.IO room

---

## Phase 3: Celery Beat

### Task 3.1: Create Monitor Task
- [x] Define collect_system_metrics task
- [x] Use singleton service instance
- [x] Emit all metric types

**Files:**
- `backend/src/workers/system_monitor_tasks.py`

### Task 3.2: Configure Beat Schedule
- [x] Add to beat_schedule
- [x] Set interval (2 seconds default)
- [x] Configure task queue

**Files:**
- `backend/src/core/celery_app.py`

### Task 3.3: Add Configuration
- [x] Add system_monitor_interval_seconds
- [x] Default to 2 seconds
- [x] Document in settings

**Files:**
- `backend/src/core/config.py`

---

## Phase 4: API Endpoints

### Task 4.1: Current Metrics Endpoint
- [x] GET /system/metrics
- [x] Return all current metrics
- [x] Fallback for WebSocket disconnect

**Files:**
- `backend/src/api/v1/endpoints/system.py`

### Task 4.2: GPU List Endpoint
- [x] GET /system/gpus
- [x] Return GPU info with availability
- [x] Include current job info

---

## Phase 5: Frontend Store

### Task 5.1: Create System Monitor Store
- [x] Current metrics state
- [x] Historical data arrays
- [x] WebSocket connection state

**Files:**
- `frontend/src/stores/systemMonitorStore.ts`

### Task 5.2: Implement History Management
- [x] Max history points (1 hour)
- [x] Trim old data
- [x] Per-GPU history

### Task 5.3: Implement Fallback Logic
- [x] Detect WebSocket disconnect
- [x] Start polling fallback
- [x] Stop polling on reconnect

---

## Phase 6: UI Components

### Task 6.1: Create WebSocket Hook
- [x] Subscribe to all system channels
- [x] Handle metrics events
- [x] Track connection state
- [x] Clean up on unmount

**Files:**
- `frontend/src/hooks/useSystemMonitorWebSocket.ts`

### Task 6.2: Create UtilizationChart
- [x] Dual Y-axis (utilization, temperature)
- [x] Line chart with Recharts
- [x] Responsive container
- [x] Tooltip formatting

**Files:**
- `frontend/src/components/SystemMonitor/UtilizationChart.tsx`

### Task 6.3: Create MemoryChart
- [x] RAM and Swap bars
- [x] Percentage display
- [x] Absolute values

**Files:**
- `frontend/src/components/SystemMonitor/MemoryChart.tsx`

### Task 6.4: Create IOChart
- [x] Disk read/write rates
- [x] Network up/down rates
- [x] Format as MB/s

**Files:**
- `frontend/src/components/SystemMonitor/IOChart.tsx`

### Task 6.5: Create SystemMonitor Component
- [x] Initialize WebSocket hook
- [x] Grid layout
- [x] Connection indicator
- [x] Initial data fetch

**Files:**
- `frontend/src/components/SystemMonitor/SystemMonitor.tsx`

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `backend/src/services/system_monitor_service.py` | Metrics collection |
| `backend/src/workers/system_monitor_tasks.py` | Celery Beat task |
| `backend/src/workers/websocket_emitter.py` | WebSocket emission |
| `backend/src/core/celery_app.py` | Beat schedule |
| `backend/src/api/v1/endpoints/system.py` | API routes |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/stores/systemMonitorStore.ts` | Zustand store |
| `frontend/src/hooks/useSystemMonitorWebSocket.ts` | WebSocket hook |
| `frontend/src/components/SystemMonitor/SystemMonitor.tsx` | Main component |
| `frontend/src/components/SystemMonitor/UtilizationChart.tsx` | GPU chart |
| `frontend/src/components/SystemMonitor/MemoryChart.tsx` | Memory chart |
| `frontend/src/components/SystemMonitor/IOChart.tsx` | I/O chart |

---

*Related: [PRD](../prds/008_FPRD|System_Monitoring.md) | [TDD](../tdds/008_FTDD|System_Monitoring.md) | [TID](../tids/008_FTID|System_Monitoring.md)*
