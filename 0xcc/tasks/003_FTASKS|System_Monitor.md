# Feature Tasks: System Monitor

**Feature Number:** 003
**Feature Name:** System Monitor
**Status:** 🔄 In Progress
**Created:** 2025-10-16
**Last Updated:** 2025-10-16

## Overview

Implementation of real-time GPU and system resource monitoring dashboard for miStudio. This feature provides comprehensive monitoring of GPU utilization, memory usage, temperature, power consumption, and system resources during model training, inference, and dataset operations.

**Reference Materials:**
- PRD: `/home/x-sean/app/miStudio/0xcc/project-specs/reference-implementation/SystemMonitor/GPU_Monitoring_Dashboard_PRD.md`
- Reference Implementation: `/home/x-sean/app/miStudio/0xcc/project-specs/reference-implementation/SystemMonitor/SystemMonitorTab.jsx`
- UI Screenshots: GPU_Top.jpg, GPU_Mid.jpg, GPU_Bot.jpg

## Implementation Milestones

### Milestone 1: MVP Release (Week 1-2) - P0

#### Backend: Core API Endpoints

- [ ] **M1.1: Install Python Dependencies**
  - Install pynvml for NVIDIA GPU monitoring
  - Install psutil for system resource monitoring
  - Add to requirements.txt
  - Test CUDA availability and nvidia-smi access

- [ ] **M1.2: Create GPU Monitoring Service**
  - File: `backend/src/services/system_monitor_service.py`
  - Implement `GPUMonitorService` class
  - Method: `get_gpu_metrics()` - Current GPU utilization, memory, temp, power, fan
  - Method: `get_gpu_list()` - Available GPU devices and their IDs
  - Method: `get_gpu_info(gpu_id)` - Static GPU information (name, driver, CUDA version)
  - Error handling for when nvidia-smi/pynvml unavailable

- [ ] **M1.3: Create System Monitoring Service**
  - File: `backend/src/services/system_monitor_service.py`
  - Implement `SystemMonitorService` class
  - Method: `get_system_metrics()` - CPU usage, RAM, swap, disk I/O
  - Method: `get_storage_metrics()` - Mount point usage for /data/
  - Method: `get_network_metrics()` - Network I/O statistics

- [ ] **M1.4: Create System Monitor API Router**
  - File: `backend/src/api/routes/system.py`
  - Endpoint: `GET /api/v1/system/gpu-metrics` - Current GPU metrics
  - Endpoint: `GET /api/v1/system/gpu-list` - Available GPUs
  - Endpoint: `GET /api/v1/system/gpu-info/{gpu_id}` - Static GPU info
  - Endpoint: `GET /api/v1/system/metrics` - System resource metrics
  - Response schemas with proper typing
  - Register router in main FastAPI app

- [ ] **M1.5: Create Database Models (Optional)**
  - File: `backend/src/db/models/system_metrics.py`
  - Table: `system_metrics_history` (optional for M1, required for M2)
  - Columns: timestamp, gpu_id, utilization, memory_used, temperature, etc.
  - Consider if historical data needed for M1 or defer to M2

- [ ] **M1.6: Write Backend Unit Tests**
  - File: `backend/tests/unit/test_system_monitor_service.py`
  - Mock pynvml and psutil for testing
  - Test all service methods
  - Test error handling when GPU unavailable
  - Aim for >80% coverage

#### Frontend: Basic System Monitor UI

- [ ] **M1.7: Create System Monitor Types**
  - File: `frontend/src/types/system.ts`
  - Interface: `GPUMetrics` (utilization, memory, temp, power, fan)
  - Interface: `SystemMetrics` (CPU, RAM, swap, disk)
  - Interface: `GPUInfo` (name, driver, CUDA version)
  - Type: `ViewMode = 'single' | 'compare'`

- [ ] **M1.8: Create System Monitor API Client**
  - File: `frontend/src/api/system.ts`
  - Function: `getGPUMetrics()` - Fetch current GPU metrics
  - Function: `getGPUList()` - Fetch available GPUs
  - Function: `getGPUInfo(gpuId)` - Fetch GPU information
  - Function: `getSystemMetrics()` - Fetch system metrics
  - Error handling with axios

- [ ] **M1.9: Create System Monitor Zustand Store**
  - File: `frontend/src/stores/systemMonitorStore.ts`
  - State: current GPU metrics, system metrics, GPU list, selected GPU
  - Actions: `fetchGPUMetrics()`, `fetchSystemMetrics()`, `selectGPU(id)`
  - Auto-refresh every 1 second when store is active
  - Clean up intervals on unmount

- [ ] **M1.10: Create GPU Metrics Panel Component**
  - File: `frontend/src/components/SystemMonitor/GPUMetricsPanel.tsx`
  - Display: GPU utilization percentage with progress bar
  - Display: Memory usage (used/total GB) with progress bar
  - Display: Temperature (current/max °C) with color coding
  - Display: Power usage (current/max W) with progress bar
  - Display: Fan speed percentage
  - Tailwind slate theme styling per reference

- [ ] **M1.11: Create System Metrics Panel Component**
  - File: `frontend/src/components/SystemMonitor/SystemMetricsPanel.tsx`
  - Display: CPU usage percentage
  - Display: RAM usage (used/total GB)
  - Display: Swap usage (used/total GB)
  - Display: Disk I/O (read/write MB/s)
  - Compact grid layout

- [ ] **M1.12: Create System Information Panel Component**
  - File: `frontend/src/components/SystemMonitor/SystemInfoPanel.tsx`
  - Display: GPU device name (e.g., "NVIDIA Jetson Orin Nano")
  - Display: Driver version
  - Display: CUDA version
  - Display: Total GPU memory
  - Display: Compute capability
  - Display: Memory clock speed
  - Static information, no polling needed

- [ ] **M1.13: Create System Monitor Page**
  - File: `frontend/src/pages/SystemMonitor.tsx`
  - Integrate GPUMetricsPanel, SystemMetricsPanel, SystemInfoPanel
  - useEffect to start/stop polling on mount/unmount
  - Loading state while fetching initial data
  - Error state if GPU unavailable
  - Responsive grid layout

- [ ] **M1.14: Add System Monitor Route**
  - Update: `frontend/src/App.tsx` or router config
  - Add route: `/system-monitor`
  - Add navigation item to main menu/sidebar
  - Use activity icon from lucide-react

- [ ] **M1.15: Write Frontend Unit Tests**
  - File: `frontend/src/stores/systemMonitorStore.test.ts`
  - File: `frontend/src/api/system.test.ts`
  - Mock axios for API tests
  - Test store actions and state updates
  - Test error handling

- [ ] **M1.16: Manual Integration Testing**
  - Test System Monitor page loads without errors
  - Verify GPU metrics update every second
  - Verify system metrics update every second
  - Check display accuracy against nvidia-smi output
  - Test error handling when backend unavailable
  - Test on Jetson Orin Nano hardware

### Milestone 2: Enhanced Monitoring (Week 3-4) - P1

#### Backend: Historical Data & Advanced Metrics

- [ ] **M2.1: Create Historical Metrics Storage**
  - Create Alembic migration for `system_metrics_history` table
  - Columns: id, timestamp, gpu_id, utilization, memory_used, memory_total, temperature, power_usage
  - Index on timestamp and gpu_id for fast queries
  - Run migration: `alembic upgrade head`

- [ ] **M2.2: Create Background Metrics Collector**
  - File: `backend/src/workers/system_monitor_tasks.py`
  - Celery task: `collect_system_metrics()` - Runs every 1 second
  - Store metrics in database with timestamp
  - Implement data pruning: keep 24h at 1s resolution, 7d at 1m resolution
  - Register task in Celery beat schedule

- [ ] **M2.3: Add Historical Data Endpoints**
  - Endpoint: `GET /api/v1/system/gpu-metrics/history` - Query params: gpu_id, start_time, end_time, resolution
  - Endpoint: `GET /api/v1/system/gpu-processes` - Active processes using GPU
  - Return time-series data formatted for Recharts
  - Implement time-range aggregation (1h, 6h, 24h)

- [ ] **M2.4: Implement GPU Process Monitoring**
  - Method in GPUMonitorService: `get_gpu_processes(gpu_id)`
  - Return: PID, process name, GPU memory usage, GPU utilization per process
  - Use pynvml to query running processes

- [ ] **M2.5: Add Hardware Metrics Collection**
  - Method: `get_hardware_metrics(gpu_id)` - GPU clock, memory clock, PCIe bandwidth
  - Method: `get_encoder_decoder_stats(gpu_id)` - Encoder/decoder utilization if available
  - Add to periodic collection task

#### Frontend: Charts & Advanced Visualizations

- [ ] **M2.6: Install Chart Dependencies**
  - Install recharts: `npm install recharts`
  - Verify D3.js available if needed

- [ ] **M2.7: Create Time-Series Chart Component**
  - File: `frontend/src/components/SystemMonitor/MetricsChart.tsx`
  - Generic chart component accepting time-series data
  - Support multiple series (utilization, memory, temp)
  - X-axis: time, Y-axis: percentage/value
  - Responsive sizing
  - Hover tooltips with exact values

- [ ] **M2.8: Create Historical Metrics Panel**
  - File: `frontend/src/components/SystemMonitor/HistoricalMetricsPanel.tsx`
  - Display: GPU utilization chart (last 1h/6h/24h)
  - Display: GPU memory chart
  - Display: Temperature chart
  - Time range selector: 1h, 6h, 24h buttons
  - Auto-refresh every 5 seconds

- [ ] **M2.9: Create Hardware Metrics Panel**
  - File: `frontend/src/components/SystemMonitor/HardwareMetricsPanel.tsx`
  - Display: GPU clock speed (MHz)
  - Display: Memory clock speed (MHz)
  - Display: PCIe bandwidth (read/write)
  - Display: Encoder/decoder utilization
  - Progress bars and numeric values

- [ ] **M2.10: Create GPU Processes Panel**
  - File: `frontend/src/components/SystemMonitor/GPUProcessesPanel.tsx`
  - Table: PID, Process Name, GPU Memory, GPU Utilization
  - Show "X active" process count
  - Sort by GPU memory usage (descending)
  - Refresh every 2 seconds

- [ ] **M2.11: Update Store for Historical Data**
  - Add state: historical metrics, time range
  - Action: `fetchHistoricalMetrics(gpuId, timeRange)`
  - Action: `setTimeRange('1h' | '6h' | '24h')`
  - Action: `fetchGPUProcesses(gpuId)`

- [ ] **M2.12: Update System Monitor Page**
  - Add HistoricalMetricsPanel below current metrics
  - Add HardwareMetricsPanel to right column
  - Add GPUProcessesPanel below GPU metrics
  - Update layout to accommodate new panels

- [ ] **M2.13: Add Storage Metrics Display**
  - File: `frontend/src/components/SystemMonitor/StoragePanel.tsx`
  - Display: / mount point usage
  - Display: /data/ mount point usage (if separate)
  - Progress bars with used/total GB and percentage

- [ ] **M2.14: Test Historical Data Collection**
  - Verify Celery beat task runs every 1 second
  - Check database for metric accumulation
  - Test data pruning after 24 hours
  - Verify chart updates with real historical data

### Milestone 3: Multi-GPU Support (Week 5-6) - P1

#### Backend: Multi-GPU APIs

- [ ] **M3.1: Extend GPU Monitoring for Multiple GPUs**
  - Update `get_gpu_metrics()` to accept optional gpu_id (default: all)
  - Return metrics for all GPUs when gpu_id=null
  - Update historical collection to store all GPU metrics

- [ ] **M3.2: Add Multi-GPU Comparison Endpoint**
  - Endpoint: `GET /api/v1/system/gpu-metrics/compare` - Returns side-by-side metrics for all GPUs
  - Format optimized for comparison view
  - Include relative performance indicators

#### Frontend: Multi-GPU UI

- [ ] **M3.3: Add View Mode Selector**
  - File: `frontend/src/components/SystemMonitor/ViewModeSelector.tsx`
  - Toggle: "Single GPU" vs "Compare GPUs"
  - Store selected mode in systemMonitorStore
  - Disable "Compare" if only 1 GPU available

- [ ] **M3.4: Create Multi-GPU Comparison View**
  - File: `frontend/src/components/SystemMonitor/MultiGPUComparisonPanel.tsx`
  - Side-by-side display of all GPU metrics
  - Same metrics as single view but in grid layout
  - Highlight highest/lowest values for easy comparison

- [ ] **M3.5: Create GPU Selector Dropdown**
  - File: `frontend/src/components/SystemMonitor/GPUSelector.tsx`
  - Dropdown: List all available GPUs
  - Display: "GPU 0: NVIDIA Jetson Orin Nano"
  - OnSelect: Update store and refetch metrics for selected GPU
  - Only visible in Single GPU mode

- [ ] **M3.6: Update System Monitor Page for Multi-GPU**
  - Add ViewModeSelector at top
  - Conditionally render Single vs Compare view based on mode
  - Add GPUSelector in single mode
  - Ensure historical charts work for selected GPU

- [ ] **M3.7: Test Multi-GPU Support**
  - Test on system with multiple GPUs (if available)
  - Test graceful degradation to single GPU mode
  - Verify comparison view shows all GPUs accurately
  - Test GPU selection switching

### Milestone 4: Polish & Optimization (Week 7-8) - P2

#### Performance & Real-time Updates

- [ ] **M4.1: Implement WebSocket for Real-time Metrics**
  - File: `backend/src/api/websockets/system_metrics.py`
  - WebSocket endpoint: `/ws/system-metrics`
  - Push GPU + system metrics every 1 second
  - Support room-based subscriptions per GPU

- [ ] **M4.2: Update Frontend to Use WebSocket**
  - File: `frontend/src/hooks/useSystemMetricsWebSocket.ts`
  - Custom hook to connect to WebSocket
  - Auto-reconnect on disconnect
  - Update store with streamed metrics
  - Fall back to polling if WebSocket fails

- [ ] **M4.3: Optimize Chart Rendering**
  - Implement virtual scrolling for long time series
  - Debounce chart updates to reduce redraws
  - Use memo/useMemo for expensive calculations
  - Test with 24h of 1-second resolution data

- [ ] **M4.4: Implement Data Pruning Strategy**
  - Background task: Aggregate old data to reduce storage
  - Keep: 1h at 1s, 6h at 10s, 24h at 1m, 7d at 5m resolution
  - Delete data older than 7 days
  - Run pruning task daily at midnight

#### Error Handling & Edge Cases

- [ ] **M4.5: Add Comprehensive Error Handling**
  - Backend: Handle pynvml initialization failures gracefully
  - Backend: Return 503 when GPU monitoring unavailable
  - Frontend: Show friendly error message if no GPU detected
  - Frontend: Display "GPU unavailable" state instead of crashing
  - Add retry logic for transient failures

- [ ] **M4.6: Add Loading States**
  - Skeleton loaders for all panels while data fetching
  - Smooth transitions between loading and loaded states
  - Prevent layout shift during load

- [ ] **M4.7: Handle Edge Cases**
  - Test with 0 GPUs (CPU-only system)
  - Test with GPU that doesn't support certain metrics
  - Test with very old GPU driver versions
  - Test with GPU in use by other processes

#### Documentation & Testing

- [ ] **M4.8: Write API Documentation**
  - Document all /api/v1/system/* endpoints in OpenAPI schema
  - Add examples for each endpoint
  - Document WebSocket protocol
  - Update API reference docs

- [ ] **M4.9: Write User Documentation**
  - File: `docs/features/system-monitor.md`
  - Document how to access System Monitor
  - Explain each metric and what it means
  - Add troubleshooting section
  - Include screenshots from UI

- [ ] **M4.10: Integration Tests**
  - File: `backend/tests/integration/test_system_monitor_workflow.py`
  - Test full workflow: fetch GPU list → select GPU → fetch metrics → fetch history
  - Test WebSocket connection and metric streaming
  - Test multi-GPU scenarios

- [ ] **M4.11: E2E Tests (Optional)**
  - Playwright test: Navigate to System Monitor page
  - Verify metrics displayed correctly
  - Test time range selection
  - Test GPU selection (if multi-GPU)
  - Verify charts render

- [ ] **M4.12: Performance Testing**
  - Load test: Monitor backend under continuous metric requests
  - Verify frontend performance with 24h data
  - Check memory usage doesn't grow unbounded
  - Optimize query performance if needed

#### Final Polish

- [ ] **M4.13: UI/UX Refinements**
  - Match exact styling from reference screenshots
  - Ensure all colors match slate theme
  - Add smooth animations for metric updates
  - Polish typography and spacing
  - Add tooltips for technical metrics

- [ ] **M4.14: Accessibility**
  - Add ARIA labels to all interactive elements
  - Ensure keyboard navigation works
  - Test screen reader compatibility
  - Add focus indicators

- [ ] **M4.15: Final Testing & Bug Fixes**
  - Manual testing on Jetson Orin Nano
  - Fix any discovered bugs
  - Verify all acceptance criteria met
  - User acceptance testing

## Acceptance Criteria

### Milestone 1 (MVP)
- [ ] System Monitor page accessible from main navigation
- [ ] Real-time GPU metrics displayed: utilization, memory, temperature, power, fan speed
- [ ] Real-time system metrics displayed: CPU, RAM, swap, disk I/O
- [ ] Static GPU information displayed: device name, driver, CUDA version
- [ ] Metrics update every 1 second
- [ ] All backend endpoints functional and tested (>80% coverage)
- [ ] All frontend components render without errors
- [ ] Works on Jetson Orin Nano hardware

### Milestone 2 (Enhanced)
- [ ] Historical GPU metrics stored in database for 24 hours
- [ ] Time-series charts display GPU utilization, memory, and temperature
- [ ] Time range selector works: 1h, 6h, 24h
- [ ] Hardware metrics panel shows clock speeds, PCIe bandwidth
- [ ] GPU processes panel shows active processes with memory usage
- [ ] Storage metrics displayed for relevant mount points
- [ ] Charts update automatically with new data
- [ ] Data pruning prevents unbounded database growth

### Milestone 3 (Multi-GPU)
- [ ] View mode selector toggles between Single and Compare views
- [ ] GPU selector dropdown lists all available GPUs (single mode)
- [ ] Compare view shows side-by-side metrics for all GPUs
- [ ] Historical charts work for selected GPU in single mode
- [ ] Graceful degradation when only 1 GPU available
- [ ] All metrics accurate for each GPU

### Milestone 4 (Polish)
- [ ] WebSocket real-time updates functional (with polling fallback)
- [ ] Chart rendering optimized for large datasets
- [ ] Error states handled gracefully with user-friendly messages
- [ ] Loading states prevent layout shift
- [ ] API documentation complete
- [ ] User documentation written with troubleshooting guide
- [ ] Integration tests cover main workflows (>70% coverage)
- [ ] UI matches reference screenshots exactly
- [ ] Performance meets requirements: <100ms metric fetch, <1s chart render
- [ ] No memory leaks detected during 24h run

## Technical Decisions

### Backend Technology
- **pynvml**: Python bindings for NVIDIA Management Library (preferred for GPU monitoring)
- **psutil**: Cross-platform system resource monitoring
- **Celery Beat**: Periodic task scheduling for metric collection
- **PostgreSQL**: Time-series data storage with indexing on timestamp
- **WebSocket (Socket.IO)**: Real-time metric streaming

### Frontend Technology
- **Recharts**: Time-series chart library (matches reference implementation)
- **Zustand**: State management for metrics and UI state
- **Socket.IO Client**: WebSocket connection with auto-reconnect
- **Lucide React**: Icons (activity, cpu, memory-stick, thermometer)

### Data Retention Strategy
- **1 hour**: 1-second resolution (3,600 records)
- **6 hours**: 10-second resolution (2,160 records)
- **24 hours**: 1-minute resolution (1,440 records)
- **7 days**: 5-minute resolution (2,016 records)
- **Total**: ~9,200 records per GPU (reasonable storage)

### Polling vs WebSocket
- **Default**: Polling every 1 second for simplicity
- **M4 Enhancement**: WebSocket for reduced latency and server load
- **Fallback**: Automatic fallback to polling if WebSocket fails

## Integration Points

### Existing miStudio Components
- **Main Navigation**: Add "System Monitor" link to sidebar/menu
- **Dashboard**: Optional widget showing current GPU utilization
- **Training Jobs**: Link to System Monitor during active training
- **Model Downloads**: Show GPU metrics while downloading large models

### Database
- New table: `system_metrics_history`
- Migrations managed via Alembic
- No changes to existing tables

### API
- New router: `/api/v1/system/*`
- No changes to existing API routes

## Dependencies

### Python Packages (New)
```
pynvml>=11.5.0
psutil>=5.9.0
```

### JavaScript Packages (New)
```
recharts@^2.10.0
```

### External Requirements
- NVIDIA GPU with CUDA support
- nvidia-smi available on system PATH
- GPU driver version 450.80.02 or higher

## Known Limitations

1. **NVIDIA GPUs Only**: Currently only supports NVIDIA GPUs via pynvml. AMD/Intel GPUs not supported.
2. **Linux Only**: System monitoring optimized for Linux. Windows/macOS may have limited functionality.
3. **Jetson-Specific**: Some metrics may not be available on datacenter GPUs or vice versa.
4. **Single User**: No multi-user session support; all users see same metrics.
5. **Historical Data**: Limited to 7 days of history; older data is aggregated and pruned.

## Testing Strategy

### Unit Tests
- Mock pynvml.nvmlDeviceGetHandleByIndex and all nvml* functions
- Mock psutil.cpu_percent, virtual_memory, disk_io_counters
- Test service methods return correct data structures
- Test error handling when GPU unavailable

### Integration Tests
- Test full API workflow with real backend
- Test database storage and retrieval of historical data
- Test WebSocket connection and streaming
- Test data pruning task

### Manual Testing
- Run on Jetson Orin Nano hardware
- Compare nvidia-smi output with displayed metrics
- Verify accuracy of memory, temperature, power readings
- Test during high GPU load (model training)
- Test during idle periods

## Relevant Files

### Backend Files (To Be Created)
- `backend/src/services/gpu_monitor_service.py` - GPU monitoring via pynvml
- `backend/src/services/system_monitor_service.py` - System resource monitoring via psutil
- `backend/src/api/routes/system.py` - System Monitor API endpoints
- `backend/src/api/websockets/system_metrics.py` - WebSocket for real-time metrics
- `backend/src/workers/system_monitor_tasks.py` - Celery tasks for periodic collection
- `backend/src/db/models/system_metrics.py` - Database model for metrics history
- `backend/tests/unit/test_gpu_monitor_service.py` - Unit tests
- `backend/tests/unit/test_system_monitor_service.py` - Unit tests
- `backend/tests/integration/test_system_monitor_workflow.py` - Integration tests

### Frontend Files (To Be Created)
- `frontend/src/types/system.ts` - TypeScript types for metrics
- `frontend/src/api/system.ts` - API client functions
- `frontend/src/stores/systemMonitorStore.ts` - Zustand store
- `frontend/src/hooks/useSystemMetricsWebSocket.ts` - WebSocket hook
- `frontend/src/pages/SystemMonitor.tsx` - Main page component
- `frontend/src/components/SystemMonitor/GPUMetricsPanel.tsx` - GPU metrics display
- `frontend/src/components/SystemMonitor/SystemMetricsPanel.tsx` - System metrics display
- `frontend/src/components/SystemMonitor/SystemInfoPanel.tsx` - Static GPU info
- `frontend/src/components/SystemMonitor/HistoricalMetricsPanel.tsx` - Time-series charts
- `frontend/src/components/SystemMonitor/HardwareMetricsPanel.tsx` - Hardware metrics
- `frontend/src/components/SystemMonitor/GPUProcessesPanel.tsx` - Process list
- `frontend/src/components/SystemMonitor/MetricsChart.tsx` - Reusable chart component
- `frontend/src/components/SystemMonitor/ViewModeSelector.tsx` - Single/Compare toggle
- `frontend/src/components/SystemMonitor/GPUSelector.tsx` - GPU dropdown
- `frontend/src/components/SystemMonitor/MultiGPUComparisonPanel.tsx` - Multi-GPU view
- `frontend/src/components/SystemMonitor/StoragePanel.tsx` - Storage metrics
- `frontend/src/stores/systemMonitorStore.test.ts` - Store tests
- `frontend/src/api/system.test.ts` - API client tests

### Database Migration Files (To Be Created)
- `backend/alembic/versions/XXX_add_system_metrics_history.py` - Migration for metrics table

### Documentation Files (To Be Created)
- `docs/features/system-monitor.md` - User-facing documentation
- `docs/api/system-monitor-api.md` - API reference

### Existing Files (To Be Modified)
- `backend/requirements.txt` - Add pynvml, psutil
- `backend/src/main.py` - Register system router
- `frontend/package.json` - Add recharts
- `frontend/src/App.tsx` - Add System Monitor route
- Navigation component - Add System Monitor link

## Notes

- **Reference Implementation**: The SystemMonitorTab.jsx reference uses simulated data with setInterval. Our implementation will use real GPU metrics via pynvml and real-time updates via WebSocket/polling.
- **Jetson Optimization**: On Jetson Orin Nano, some metrics (like encoder/decoder) may not be available. Handle gracefully.
- **Performance**: Aim for <100ms API response time for current metrics, <1s for historical queries (up to 24h).
- **Mobile Responsive**: System Monitor is primarily a desktop feature. Mobile support is P2 priority.
- **Dark Theme**: Strictly follow slate color palette from Mock UI: bg-slate-950, bg-slate-900, text-slate-100, etc.

## Current Status

**Overall Progress:** 0/67 tasks completed (0%)

**Current Milestone:** M1 - MVP Release (Backend + Frontend Core)
**Current Phase:** Planning and task breakdown complete
**Blocked By:** None
**Next Action:** Begin M1.1 - Install Python dependencies (pynvml, psutil)

---

**Task List Version:** 1.0
**Last Updated:** 2025-10-16
