# Supplemental Task List: Progress Architecture Improvements

**Created:** 2025-10-22
**Based On:** Multi-Agent Architecture Review (Session: review_progress_monitoring_architecture_2025-10-22)
**Scope:** Architectural consistency, testing expansion, scalability improvements
**Excluded:** Security (authentication, rate limiting) per user request

---

## 🔴 HIGH PRIORITY TASKS (Complete within 2 weeks)

### Task HP-1: Migrate System Monitoring to WebSocket Pattern

**Goal:** Achieve architectural consistency by migrating system monitoring from polling to WebSocket emission (matching job progress pattern)

**Impact:**
- Consistent architecture across all monitoring
- 50% reduction in HTTP requests (1 req/sec → WebSocket push)
- Real-time system metrics updates
- Easier maintenance (one pattern to understand)

**Estimated Effort:** 8-12 hours

#### Sub-Tasks:

- [ ] **HP-1.1: Add System Metrics WebSocket Emission Utilities**
  - File: `backend/src/workers/websocket_emitter.py`
  - Add function: `emit_system_metrics(channel, metrics_data)`
  - Add function: `emit_gpu_metrics(gpu_id, metrics_data)`
  - Add function: `emit_cpu_metrics(metrics_data)`
  - Add function: `emit_memory_metrics(metrics_data)`
  - Follow existing pattern from `emit_training_progress()`
  - Include error handling for WebSocket failures
  - **Acceptance:** Functions callable, emit to correct channels, handle errors gracefully

- [ ] **HP-1.2: Create System Monitor Background Task**
  - File: `backend/src/workers/system_monitor_tasks.py` (NEW)
  - Create Celery Beat task: `monitor_system_metrics()`
  - Schedule: Every 2 seconds (configurable)
  - Collect: GPU metrics, CPU, RAM, Swap, Disk I/O, Network I/O
  - Emit via WebSocket: Each metric type to appropriate channel
  - Reuse: `SystemMonitorService` for data collection
  - **Acceptance:** Task runs every 2 seconds, emits all metrics via WebSocket

- [ ] **HP-1.3: Update Celery Beat Schedule Configuration**
  - File: `backend/src/core/celery_app.py`
  - Add beat schedule: `monitor-system-metrics` → 2 second interval
  - Make interval configurable via settings: `SYSTEM_MONITOR_INTERVAL_SECONDS`
  - **Acceptance:** Task automatically scheduled, interval configurable

- [ ] **HP-1.4: Define WebSocket Channel Names for System Metrics**
  - Document channel naming convention:
    - `system/gpu/{gpu_id}` → Per-GPU metrics
    - `system/cpu` → CPU utilization
    - `system/memory` → RAM + Swap
    - `system/disk` → Disk I/O
    - `system/network` → Network I/O
  - Update: `backend/src/workers/websocket_emitter.py` docstrings
  - **Acceptance:** Channel names documented, consistent with job progress pattern

- [ ] **HP-1.5: Create Frontend WebSocket Hook for System Monitoring**
  - File: `frontend/src/hooks/useSystemMonitorWebSocket.ts` (NEW)
  - Pattern: Similar to `useTrainingWebSocket.ts`
  - Subscribe to: `system/gpu/{id}`, `system/cpu`, `system/memory`, `system/disk`, `system/network`
  - On message: Call store update methods
  - Lifecycle: Subscribe on mount, unsubscribe on unmount
  - **Acceptance:** Hook subscribes to all system channels, updates store on events

- [ ] **HP-1.6: Update SystemMonitorStore to Use WebSocket**
  - File: `frontend/src/stores/systemMonitorStore.ts`
  - Add action: `updateGPUMetricsFromWebSocket(gpu_id, metrics)`
  - Add action: `updateCPUMetricsFromWebSocket(metrics)`
  - Add action: `updateMemoryMetricsFromWebSocket(metrics)`
  - Keep polling methods as fallback (WebSocket disconnected state)
  - Add state: `isWebSocketConnected: boolean`
  - Logic: Use WebSocket when connected, fall back to polling when disconnected
  - **Acceptance:** Store prefers WebSocket, falls back to polling seamlessly

- [ ] **HP-1.7: Update SystemMonitor Component to Use WebSocket Hook**
  - File: `frontend/src/components/SystemMonitor/SystemMonitor.tsx`
  - Import and use: `useSystemMonitorWebSocket()`
  - Remove: Direct polling calls (keep polling as fallback only)
  - Add: Connection state indicator (green dot = WebSocket, yellow = polling)
  - **Acceptance:** Component uses WebSocket by default, shows connection state

- [ ] **HP-1.8: Add WebSocket Fallback Logic**
  - File: `frontend/src/stores/systemMonitorStore.ts`
  - On WebSocket disconnect: Automatically start polling (2 second interval)
  - On WebSocket reconnect: Stop polling, resume WebSocket updates
  - Track: `consecutiveWebSocketFailures` counter
  - **Acceptance:** Seamless fallback to polling, automatic recovery on reconnect

- [ ] **HP-1.9: Test System Monitoring WebSocket Flow**
  - Manual test: Start app, verify SystemMonitor receives WebSocket updates
  - Manual test: Disconnect WebSocket, verify polling fallback
  - Manual test: Reconnect WebSocket, verify polling stops
  - Check: Network tab shows no polling requests when WebSocket connected
  - Check: Backend logs show Celery beat task emitting metrics
  - **Acceptance:** All tests pass, no polling when WebSocket active

- [ ] **HP-1.10: Update Documentation**
  - File: `backend/src/workers/system_monitor_tasks.py` - Add docstrings
  - File: `frontend/src/hooks/useSystemMonitorWebSocket.ts` - Add usage comments
  - Update: CLAUDE.md with system monitoring architecture change
  - **Acceptance:** Documentation complete and accurate

---

### Task HP-2: Expand Test Coverage - Phase 1

**Goal:** Increase unit test coverage from 40% to 60%, focus on critical progress tracking logic

**Impact:**
- Higher reliability
- Faster debugging
- Confidence in deployments
- Catch regressions early

**Estimated Effort:** 16-20 hours

#### Sub-Tasks:

##### Backend Unit Tests

- [ ] **HP-2.1: Unit Tests for Training Progress Calculation**
  - File: `backend/tests/workers/test_training_progress.py` (NEW)
  - Test: `test_calculate_training_progress_start()` → progress = 0
  - Test: `test_calculate_training_progress_midpoint()` → progress = 50
  - Test: `test_calculate_training_progress_complete()` → progress = 100
  - Test: `test_calculate_training_progress_with_checkpoints()` → correct percentage
  - Test: `test_training_progress_multi_layer()` → aggregated metrics correct
  - Mock: Database, WebSocket
  - **Acceptance:** 100% coverage of progress calculation logic, all edge cases tested

- [ ] **HP-2.2: Unit Tests for Extraction Progress Calculation**
  - File: `backend/tests/workers/test_extraction_progress.py` (NEW)
  - Test: `test_extraction_progress_loading_phase()` → 0-10%
  - Test: `test_extraction_progress_extracting_phase()` → 10-90% based on samples
  - Test: `test_extraction_progress_saving_phase()` → 90-100%
  - Test: `test_extraction_progress_samples_processed()` → accurate percentage
  - Test: `test_extraction_progress_callback()` → callback invoked correctly
  - Mock: Model loading, dataset loading, WebSocket
  - **Acceptance:** 100% coverage of 3-phase progress formula

- [ ] **HP-2.3: Unit Tests for Model Download Progress Monitor**
  - File: `backend/tests/workers/test_model_download_progress.py` (NEW)
  - Test: `test_progress_monitor_thread_start()` → thread starts successfully
  - Test: `test_progress_monitor_calculates_percentage()` → accurate calculation
  - Test: `test_progress_monitor_stops_on_completion()` → thread terminates
  - Test: `test_progress_monitor_handles_missing_files()` → graceful error handling
  - Test: `test_progress_monitor_emits_websocket_events()` → events emitted at 1% intervals
  - Mock: File system, WebSocket
  - **Acceptance:** Thread safety verified, progress calculation accurate

- [ ] **HP-2.4: Unit Tests for Dataset Progress Steps**
  - File: `backend/tests/workers/test_dataset_progress.py` (NEW)
  - Test: `test_dataset_download_progress_steps()` → 0%, 10%, 70%, 90%, 100%
  - Test: `test_dataset_tokenization_progress_steps()` → 0%, 10%, 20%, 40%, 80%, 95%, 100%
  - Test: `test_dataset_progress_error_handling()` → status set to ERROR
  - Test: `test_dataset_progress_websocket_emission()` → events emitted at each step
  - Mock: HuggingFace API, tokenizer, WebSocket
  - **Acceptance:** All progress steps verified, error handling tested

- [ ] **HP-2.5: Integration Tests for WebSocket Emission Flows**
  - File: `backend/tests/workers/test_websocket_emission_integration.py` (NEW)
  - Test: `test_emit_training_progress_flow()` → database updated, WebSocket emitted
  - Test: `test_emit_extraction_progress_flow()` → database updated, WebSocket emitted
  - Test: `test_emit_model_progress_flow()` → database updated, WebSocket emitted
  - Test: `test_emit_dataset_progress_flow()` → database updated, WebSocket emitted
  - Use: Real WebSocket manager (test mode), real database (test DB)
  - **Acceptance:** End-to-end emission flow verified for all job types

- [ ] **HP-2.6: Unit Tests for Error Classification Logic**
  - File: `backend/tests/workers/test_error_classification.py` (NEW)
  - Test: `test_classify_oom_error()` → error_type = "OOM"
  - Test: `test_classify_validation_error()` → error_type = "VALIDATION"
  - Test: `test_classify_timeout_error()` → error_type = "TIMEOUT"
  - Test: `test_classify_unknown_error()` → error_type = "UNKNOWN"
  - Test: `test_suggest_retry_params_for_oom()` → batch_size reduced
  - **Acceptance:** All error types correctly classified, retry suggestions accurate

##### Frontend Unit Tests

- [ ] **HP-2.7: Unit Tests for WebSocket Hook Subscription Logic**
  - File: `frontend/src/hooks/useTrainingWebSocket.test.ts` (NEW)
  - Test: `test_hook_subscribes_on_mount()` → subscription called
  - Test: `test_hook_unsubscribes_on_unmount()` → unsubscription called
  - Test: `test_hook_updates_store_on_message()` → store action invoked
  - Test: `test_hook_resubscribes_on_training_id_change()` → old unsubscribed, new subscribed
  - Mock: WebSocketClient, trainingsStore
  - **Acceptance:** Hook lifecycle verified, store updates tested

- [ ] **HP-2.8: Expand trainingsStore Unit Tests**
  - File: `frontend/src/stores/trainingsStore.test.ts` (EXPAND)
  - Add test: `test_update_training_status_from_websocket()` → state updated correctly
  - Add test: `test_update_multiple_trainings_simultaneously()` → no race conditions
  - Add test: `test_training_completion_sets_progress_100()` → completion logic
  - Add test: `test_training_failure_preserves_last_progress()` → failure handling
  - Current: 50 tests, Target: 65 tests
  - **Acceptance:** WebSocket update paths fully tested

- [ ] **HP-2.9: Unit Tests for modelsStore Extraction Progress Updates**
  - File: `frontend/src/stores/modelsStore.test.ts` (NEW)
  - Test: `test_update_extraction_progress()` → extraction state updated
  - Test: `test_update_extraction_failure()` → error state set correctly
  - Test: `test_clear_extraction_progress()` → state cleared
  - Test: `test_check_active_extraction_with_null_data()` → handles null response
  - Mock: Fetch API, WebSocket
  - **Acceptance:** Extraction progress update logic fully tested

- [ ] **HP-2.10: Run Coverage Report and Verify Target**
  - Backend: Run `pytest --cov=src --cov-report=html`
  - Frontend: Run `npm run test:coverage`
  - Verify: Backend unit coverage ≥ 60%
  - Verify: Frontend unit coverage ≥ 60%
  - Identify: Remaining gaps for Phase 2
  - **Acceptance:** 60% unit test coverage achieved

---

## 🟡 MEDIUM PRIORITY TASKS (Complete within 1 month)

### Task MP-1: Implement WebSocket Clustering

**Goal:** Enable horizontal scaling with multiple backend instances sharing WebSocket state

**Impact:**
- Production-ready horizontal scaling
- Load balancing across multiple instances
- High availability (instance failures don't disconnect all clients)

**Estimated Effort:** 12-16 hours

#### Sub-Tasks:

- [ ] **MP-1.1: Install Socket.IO Redis Adapter**
  - File: `backend/requirements.txt`
  - Add dependency: `python-socketio[asyncio_redis]>=5.10.0`
  - Run: `pip install -r requirements.txt` in venv
  - **Acceptance:** Dependency installed successfully

- [ ] **MP-1.2: Configure Redis Connection for WebSocket**
  - File: `backend/src/core/config.py`
  - Add setting: `REDIS_WS_URL = os.getenv("REDIS_WS_URL", "redis://localhost:6379/1")`
  - Note: Use DB 1 for WebSocket (DB 0 for Celery)
  - **Acceptance:** Configuration added, environment variable supported

- [ ] **MP-1.3: Update WebSocket Manager to Use Redis Adapter**
  - File: `backend/src/core/websocket.py`
  - Import: `socketio.AsyncRedisManager`
  - Create: `redis_manager = AsyncRedisManager(settings.REDIS_WS_URL)`
  - Update: `sio = socketio.AsyncServer(client_manager=redis_manager, ...)`
  - Add: Connection error handling for Redis
  - **Acceptance:** WebSocket manager uses Redis for pub/sub

- [ ] **MP-1.4: Update Docker Compose for Multi-Instance Testing**
  - File: `docker-compose.dev.yml`
  - Add service: `backend-2` (duplicate of backend, port 8001)
  - Add service: `backend-3` (duplicate of backend, port 8002)
  - Update nginx: Load balance across 8000, 8001, 8002
  - Update environment: All backends use same `REDIS_WS_URL`
  - **Acceptance:** Can run 3 backend instances simultaneously

- [ ] **MP-1.5: Update Nginx Configuration for WebSocket Load Balancing**
  - File: `nginx/nginx.conf`
  - Add upstream: `backend_servers` with 3 instances
  - Configure: `proxy_http_version 1.1;` for WebSocket
  - Configure: `proxy_set_header Upgrade $http_upgrade;`
  - Configure: `proxy_set_header Connection "upgrade";`
  - Add: Sticky sessions based on client IP (optional)
  - **Acceptance:** Nginx properly load balances WebSocket connections

- [ ] **MP-1.6: Test Multi-Instance WebSocket Communication**
  - Manual test: Start 3 backend instances
  - Manual test: Connect client to instance 1, subscribe to channel
  - Manual test: Emit event from instance 2 to same channel
  - Verify: Client receives event (via Redis pub/sub)
  - Check: Redis monitor shows pub/sub messages
  - **Acceptance:** WebSocket messages delivered across instances

- [ ] **MP-1.7: Test Failover Behavior**
  - Manual test: Connect client to instance 1
  - Manual test: Kill instance 1 process
  - Verify: Client reconnects to instance 2 or 3 automatically
  - Verify: Client resubscribes to channels
  - Verify: Client continues receiving updates
  - **Acceptance:** Graceful failover with minimal disruption

- [ ] **MP-1.8: Document Clustering Setup**
  - File: `docs/DEPLOYMENT.md` or similar
  - Document: Redis requirement for multi-instance deployment
  - Document: Environment variable configuration
  - Document: Nginx load balancing setup
  - Document: Testing procedure for clustering
  - **Acceptance:** Documentation complete, deployment guide clear

---

### Task MP-2: Add TrainingMetric Archival Strategy

**Goal:** Prevent unbounded table growth, maintain query performance at scale

**Impact:**
- Query performance maintained as data grows
- Reduced storage costs
- Historical data preserved but archived
- 1000+ training runs won't slow down dashboard

**Estimated Effort:** 10-14 hours

#### Sub-Tasks:

- [ ] **MP-2.1: Design Archival Strategy**
  - Decision: Partition by `training_id` vs date-based partitioning
  - Decision: Archive to separate table vs delete old data
  - Recommendation: Date-based partitioning (monthly), archive after 30 days
  - Document: Strategy in task notes
  - **Acceptance:** Strategy decided and documented

- [ ] **MP-2.2: Create TrainingMetric Archive Table**
  - File: `backend/alembic/versions/xxx_create_training_metric_archive.py`
  - Create table: `training_metrics_archive` (same schema as `training_metrics`)
  - Add indexes: `(training_id)`, `(timestamp)`, `(training_id, step)`
  - **Acceptance:** Migration creates archive table with indexes

- [ ] **MP-2.3: Implement Archival Service**
  - File: `backend/src/services/training_metrics_archival_service.py` (NEW)
  - Function: `archive_old_metrics(days_old=30)` → Copy to archive, delete from main
  - Function: `get_training_metrics(training_id, include_archived=False)` → Query both tables
  - Function: `get_metric_count_by_age()` → Statistics for monitoring
  - Use: Raw SQL for efficiency (COPY ... TO, DELETE WHERE)
  - **Acceptance:** Service can archive and retrieve metrics correctly

- [ ] **MP-2.4: Create Celery Beat Task for Archival**
  - File: `backend/src/workers/maintenance_tasks.py` (NEW or EXPAND)
  - Task: `archive_training_metrics()` → Run daily at 2 AM
  - Call: `training_metrics_archival_service.archive_old_metrics(days_old=30)`
  - Log: Metrics archived count, errors
  - Add to: Celery beat schedule in `celery_app.py`
  - **Acceptance:** Task runs daily, archives old metrics automatically

- [ ] **MP-2.5: Update TrainingMetricsService to Query Both Tables**
  - File: `backend/src/services/training_metrics_service.py`
  - Update: `get_metrics_by_training_id(training_id, include_archived=True)`
  - Logic: Query main table, if `include_archived`, also query archive table, merge results
  - Order: By step ASC
  - **Acceptance:** Service transparently queries both tables

- [ ] **MP-2.6: Add Configuration for Archival Threshold**
  - File: `backend/src/core/config.py`
  - Add setting: `TRAINING_METRIC_ARCHIVE_DAYS = int(os.getenv("TRAINING_METRIC_ARCHIVE_DAYS", "30"))`
  - Use: In archival service and Celery task
  - **Acceptance:** Archival threshold configurable via environment variable

- [ ] **MP-2.7: Test Archival Flow**
  - Create: 100 test training metrics (50 recent, 50 old)
  - Run: `archive_training_metrics()` manually
  - Verify: Old metrics moved to archive table, recent metrics remain
  - Query: `get_training_metrics(training_id, include_archived=True)` → all metrics returned
  - Query: `get_training_metrics(training_id, include_archived=False)` → only recent metrics
  - **Acceptance:** Archival works correctly, queries work with both tables

- [ ] **MP-2.8: Add Monitoring for Archive Process**
  - Log: Metrics archived count, time taken, errors
  - Add: Prometheus metrics (optional): `training_metrics_archived_total`, `archival_duration_seconds`
  - **Acceptance:** Archival process observable, errors logged

- [ ] **MP-2.9: Document Archival Strategy**
  - File: `docs/DATABASE_MAINTENANCE.md` or similar
  - Document: Archival process, schedule, configuration
  - Document: How to query archived metrics
  - Document: How to restore archived metrics if needed
  - **Acceptance:** Documentation complete

---

### Task MP-3: Create Unified Operations Dashboard

**Goal:** Single UI showing all active operations + resource usage, correlation visible

**Impact:**
- Better user experience ("what's running and using resources")
- Resource-job correlation visible
- Single source of truth for system state

**Estimated Effort:** 16-20 hours

#### Sub-Tasks:

- [ ] **MP-3.1: Design Operations Dashboard UI/UX**
  - Reference: Mock UI for design patterns
  - Layout: Top section = active operations, bottom = resource usage
  - Components: OperationCard (job info + resource usage), ResourceVisualization
  - Sketch: Wireframe or description of layout
  - **Acceptance:** Design documented, ready for implementation

- [ ] **MP-3.2: Create OperationsStore (Zustand)**
  - File: `frontend/src/stores/operationsStore.ts` (NEW)
  - State: `activeOperations: Operation[]` (training, extraction, downloads)
  - State: `systemResources: SystemResources` (GPU, CPU, RAM)
  - Action: `fetchAllActiveOperations()` → Query all stores for active jobs
  - Action: `updateResourceUsage(resources)` → Update from SystemMonitorStore
  - Computed: `operationsWithResources` → Correlate jobs with GPU/CPU usage (heuristic)
  - **Acceptance:** Store manages unified state, correlates jobs with resources

- [ ] **MP-3.3: Implement Resource Correlation Heuristic**
  - File: `frontend/src/stores/operationsStore.ts`
  - Logic: Training job → likely using GPU (if training_layers includes GPU-based training)
  - Logic: Extraction job → likely using GPU if status = "extracting"
  - Logic: Model download → likely using disk I/O
  - Logic: Dataset tokenization → likely using CPU
  - Return: `operation.estimatedResourceUsage = { gpu: number[], cpu: boolean }`
  - Note: This is best-effort heuristic, not guaranteed accurate without process tracking
  - **Acceptance:** Operations tagged with likely resource usage

- [ ] **MP-3.4: Create OperationCard Component**
  - File: `frontend/src/components/OperationsDashboard/OperationCard.tsx` (NEW)
  - Display: Job type, name, status, progress bar
  - Display: Estimated resource usage (GPU #, CPU, Disk I/O)
  - Display: Duration, start time
  - Actions: Cancel, retry (if applicable)
  - Style: Consistent with existing cards (TrainingCard, etc.)
  - **Acceptance:** Card shows all relevant info, actions work

- [ ] **MP-3.5: Create ResourceCorrelationVisualization Component**
  - File: `frontend/src/components/OperationsDashboard/ResourceCorrelationVisualization.tsx` (NEW)
  - Visualization: Per-GPU bar showing utilization + which job is likely using it
  - Example: "GPU 0: 85% - Training 'sae_v1' (estimated)"
  - Visualization: CPU bar with operations using CPU
  - Use: Recharts or custom bars
  - **Acceptance:** Visual correlation between jobs and resources clear

- [ ] **MP-3.6: Create OperationsDashboard Component**
  - File: `frontend/src/components/OperationsDashboard/OperationsDashboard.tsx` (NEW)
  - Layout: Grid of OperationCards (top section)
  - Layout: ResourceCorrelationVisualization (bottom section)
  - Data: From `operationsStore.operationsWithResources`
  - Auto-refresh: Subscribe to WebSocket updates
  - **Acceptance:** Dashboard shows all active operations + resources

- [ ] **MP-3.7: Add Operations Dashboard to Main Navigation**
  - File: `frontend/src/App.tsx` or navigation component
  - Add route: `/operations` → OperationsDashboard
  - Add nav item: "Operations" (icon: Activity or List)
  - Position: Between "System Monitor" and "Settings"
  - **Acceptance:** Dashboard accessible from main navigation

- [ ] **MP-3.8: Implement Auto-Refresh Logic**
  - File: `frontend/src/components/OperationsDashboard/OperationsDashboard.tsx`
  - Subscribe: Training, extraction, model, dataset WebSocket updates
  - Subscribe: System monitoring WebSocket updates
  - On update: Call `operationsStore.fetchAllActiveOperations()`
  - **Acceptance:** Dashboard updates in real-time as operations progress

- [ ] **MP-3.9: Add Empty State**
  - Show: "No active operations" message when no jobs running
  - Show: Suggestion to start training, download model, etc.
  - **Acceptance:** Friendly empty state displayed

- [ ] **MP-3.10: Test Operations Dashboard**
  - Manual test: Start training job, verify appears in dashboard
  - Manual test: Check resource correlation shows GPU usage
  - Manual test: Start extraction, verify appears and correlates with GPU
  - Manual test: Complete job, verify removed from dashboard
  - Manual test: Multiple concurrent jobs, verify all shown
  - **Acceptance:** Dashboard accurately reflects system state

---

### Task MP-4: Expand Test Coverage - Phase 2

**Goal:** Increase test coverage to 70% unit, 40% integration, 30% e2e

**Impact:**
- Production reliability confidence
- Comprehensive regression testing
- Complex scenarios tested

**Estimated Effort:** 20-24 hours

#### Sub-Tasks:

##### Integration Tests

- [ ] **MP-4.1: E2E Test for Complete Training Flow with Progress Monitoring**
  - File: `backend/tests/e2e/test_training_flow_e2e.py` (NEW)
  - Flow: Create training → Start training → Monitor progress via WebSocket → Complete → Verify metrics saved
  - Use: Real database (test DB), real Celery worker (test mode), real WebSocket
  - Verify: Progress updates received, metrics logged at intervals, completion triggered
  - **Acceptance:** Full training flow tested end-to-end

- [ ] **MP-4.2: Integration Test for WebSocket Reconnection**
  - File: `frontend/src/hooks/useTrainingWebSocket.reconnection.test.ts` (NEW)
  - Test: Simulate WebSocket disconnect → Verify reconnection attempt
  - Test: Verify resubscription to channels after reconnect
  - Test: Verify state synchronization after reconnect (fetch latest progress)
  - Mock: WebSocket server, simulate disconnect/reconnect
  - **Acceptance:** Reconnection logic fully tested

- [ ] **MP-4.3: Integration Test for Concurrent Training Operations**
  - File: `backend/tests/workers/test_concurrent_training_operations.py` (NEW)
  - Test: Start 3 training jobs simultaneously
  - Test: Verify all progress updates isolated (no cross-contamination)
  - Test: Verify WebSocket events emitted for correct training IDs
  - Test: Verify database updates don't conflict (transaction isolation)
  - **Acceptance:** Concurrent operations work correctly, no race conditions

- [ ] **MP-4.4: Integration Test for Concurrent Extraction Operations**
  - File: `backend/tests/workers/test_concurrent_extraction_operations.py` (NEW)
  - Test: Start 2 extraction jobs on different models simultaneously
  - Test: Verify progress updates isolated
  - Test: Verify both complete successfully
  - Test: Verify retry logic doesn't interfere between jobs
  - **Acceptance:** Concurrent extractions work correctly

- [ ] **MP-4.5: Performance Test for TrainingMetric Queries**
  - File: `backend/tests/performance/test_training_metric_query_performance.py` (NEW)
  - Setup: Create 1000 training runs with 100 metrics each (100k total rows)
  - Test: Query single training metrics → Verify < 100ms response time
  - Test: Query aggregated metrics across 100 trainings → Verify < 500ms
  - Test: Archive old metrics, re-run queries → Verify performance maintained
  - **Acceptance:** Query performance acceptable at scale

##### E2E Tests (Playwright)

- [ ] **MP-4.6: E2E Test for Training Progress Visualization**
  - File: `frontend/tests/e2e/training-progress.spec.ts` (NEW)
  - Flow: Navigate to Training panel → Start training → Watch progress bar update
  - Verify: Progress bar animates from 0% to 100%
  - Verify: Current loss, sparsity displayed and update
  - Verify: Training completes, status changes to "Completed"
  - **Acceptance:** Full training visualization tested in browser

- [ ] **MP-4.7: E2E Test for System Monitor Real-Time Updates**
  - File: `frontend/tests/e2e/system-monitor.spec.ts` (NEW)
  - Flow: Navigate to System Monitor → Verify GPU metrics update
  - Verify: Utilization chart updates (check for data points increasing)
  - Verify: Memory chart updates
  - Verify: Connection state indicator shows connected (green dot)
  - **Acceptance:** System monitor e2e tested

- [ ] **MP-4.8: E2E Test for Operations Dashboard**
  - File: `frontend/tests/e2e/operations-dashboard.spec.ts` (NEW)
  - Flow: Start training → Navigate to Operations Dashboard
  - Verify: Training job appears in active operations
  - Verify: Resource correlation shows GPU usage
  - Flow: Complete training → Verify removed from dashboard
  - **Acceptance:** Operations dashboard e2e tested

- [ ] **MP-4.9: Run Full Test Suite and Verify Coverage Targets**
  - Backend: Run `pytest --cov=src --cov-report=html`
  - Frontend: Run `npm run test:coverage`
  - Verify: Backend unit coverage ≥ 70%
  - Verify: Backend integration coverage ≥ 40%
  - Verify: Frontend unit coverage ≥ 70%
  - Verify: E2E tests cover critical paths ≥ 30%
  - **Acceptance:** All coverage targets met

---

## 🟢 LOW PRIORITY TASKS (Complete within 2 months)

### Task LP-1: Progress History Visualization

**Goal:** Store and visualize historical progress patterns for training optimization

**Impact:**
- Debugging slow training runs
- Identifying performance patterns
- Training optimization insights

**Estimated Effort:** 12-16 hours

#### Sub-Tasks:

- [ ] **LP-1.1: Design Progress History Data Model**
  - Decision: Store progress snapshots (every 5% or 1 minute intervals)
  - Schema: `progress_history` table with `(entity_type, entity_id, timestamp, progress, metrics)`
  - **Acceptance:** Schema designed and documented

- [ ] **LP-1.2: Create ProgressHistory Database Model**
  - File: `backend/src/models/progress_history.py` (NEW)
  - Table: `progress_history`
  - Columns: `id, entity_type, entity_id, timestamp, progress, metrics (JSON), created_at`
  - Indexes: `(entity_type, entity_id, timestamp)`
  - **Acceptance:** Migration creates table

- [ ] **LP-1.3: Implement Progress Snapshot Service**
  - File: `backend/src/services/progress_history_service.py` (NEW)
  - Function: `store_progress_snapshot(entity_type, entity_id, progress, metrics)`
  - Function: `get_progress_history(entity_type, entity_id, start_time, end_time)`
  - Function: `compare_progress_patterns(entity_ids)` → Return comparative data
  - **Acceptance:** Service can store and retrieve progress history

- [ ] **LP-1.4: Update Workers to Store Progress Snapshots**
  - Files: `training_tasks.py`, `model_tasks.py`, `dataset_tasks.py`
  - Add: Call `store_progress_snapshot()` at progress update intervals
  - Throttle: Only store every 5% progress change or 1 minute (whichever is less frequent)
  - **Acceptance:** Progress snapshots automatically stored during operations

- [ ] **LP-1.5: Create ProgressHistoryChart Component**
  - File: `frontend/src/components/training/ProgressHistoryChart.tsx` (NEW)
  - Visualization: Line chart showing progress over time
  - X-axis: Time elapsed (minutes)
  - Y-axis: Progress (0-100%)
  - Support: Multiple training runs overlaid for comparison
  - Use: Recharts LineChart
  - **Acceptance:** Chart displays progress history clearly

- [ ] **LP-1.6: Add Progress History to Training Detail View**
  - File: `frontend/src/components/training/TrainingDetailModal.tsx` or similar
  - Tab: "Progress History"
  - Show: ProgressHistoryChart for this training
  - Show: Option to compare with other trainings
  - **Acceptance:** Users can view progress history for completed trainings

- [ ] **LP-1.7: Implement Progress Comparison Feature**
  - File: `frontend/src/components/training/ProgressComparisonView.tsx` (NEW)
  - UI: Multi-select trainings to compare
  - Visualization: Overlaid progress curves
  - Insights: "Training A was 30% faster than Training B"
  - **Acceptance:** Users can compare progress patterns across trainings

- [ ] **LP-1.8: Add Data Retention Policy for Progress History**
  - Configuration: Keep progress history for 90 days
  - Celery task: Archive or delete old progress history monthly
  - **Acceptance:** Automatic cleanup prevents unbounded growth

---

### Task LP-2: Performance Optimization

**Goal:** Reduce bandwidth usage, improve responsiveness, optimize resource usage

**Impact:**
- Lower bandwidth costs
- Faster user experience
- Improved scalability

**Estimated Effort:** 8-10 hours

#### Sub-Tasks:

- [ ] **LP-2.1: Enable WebSocket Compression**
  - File: `backend/src/core/websocket.py`
  - Configure: Socket.IO compression (permessage-deflate)
  - Setting: `sio = socketio.AsyncServer(compression_threshold=1024, ...)`
  - Test: Verify compression active (check network tab, message sizes)
  - **Acceptance:** WebSocket messages compressed, bandwidth reduced

- [ ] **LP-2.2: Make System Monitor Intervals Configurable**
  - File: `frontend/src/stores/systemMonitorStore.ts`
  - Add: `setUpdateInterval(ms)` action
  - UI: Dropdown in SystemMonitor to select interval (1s, 2s, 5s, 10s)
  - Default: 2 seconds (reduced from 1 second)
  - Persist: User preference in localStorage
  - **Acceptance:** Users can adjust update frequency, default reduced

- [ ] **LP-2.3: Add Covering Indexes for Common TrainingMetric Queries**
  - File: `backend/alembic/versions/xxx_add_training_metric_indexes.py`
  - Add index: `(training_id, step)` covering index for single training queries
  - Add index: `(training_id, timestamp)` covering index for time-range queries
  - Analyze: Query planner before/after, verify index usage
  - **Acceptance:** Query performance improved, indexes used

- [ ] **LP-2.4: Optimize JSON Serialization in WebSocket Payloads**
  - File: `backend/src/workers/websocket_emitter.py`
  - Review: Payload structures, remove redundant fields
  - Optimize: Use compact field names where possible (without losing clarity)
  - Example: `samples_processed` → `samples` (if clear from context)
  - Test: Verify smaller payload sizes, same functionality
  - **Acceptance:** WebSocket payloads optimized without breaking changes

- [ ] **LP-2.5: Implement WebSocket Message Batching (Optional)**
  - File: `backend/src/core/websocket.py`
  - Logic: If multiple progress updates happen within 100ms, batch into single message
  - Use case: High-frequency updates during training (every step)
  - Trade-off: Slightly delayed updates vs reduced message count
  - Configuration: `WEBSOCKET_BATCH_INTERVAL_MS` (0 = disabled)
  - **Acceptance:** Batching reduces message count without noticeable latency

- [ ] **LP-2.6: Add Database Connection Pooling Optimization**
  - File: `backend/src/core/database.py`
  - Review: Current pool size, overflow settings
  - Optimize: `pool_size=20, max_overflow=40` (adjust based on concurrency needs)
  - Add: `pool_pre_ping=True` to detect stale connections
  - Monitor: Connection pool usage, adjust as needed
  - **Acceptance:** Connection pooling optimized for production load

- [ ] **LP-2.7: Test Performance Improvements**
  - Measure: WebSocket bandwidth usage before/after compression
  - Measure: System monitor API call frequency before/after interval change
  - Measure: TrainingMetric query time before/after indexes
  - Document: Performance improvements in numbers
  - **Acceptance:** Measurable improvements, no regressions

---

### Task LP-3: Documentation

**Goal:** Comprehensive documentation for architecture, patterns, and development

**Impact:**
- Easier maintenance
- Faster onboarding for new developers
- Clear reference for patterns

**Estimated Effort:** 8-12 hours

#### Sub-Tasks:

- [ ] **LP-3.1: Create Progress Tracking Architecture Diagram**
  - File: `docs/architecture/progress-tracking-flow.md` (NEW)
  - Diagram: Flow from Celery Worker → Database → WebSocket → Frontend
  - Diagram: Channel naming conventions visual
  - Diagram: System monitoring flow (after migration to WebSocket)
  - Format: Mermaid diagrams or similar
  - **Acceptance:** Clear visual representation of architecture

- [ ] **LP-3.2: Write Developer Guide for Adding New Job Types**
  - File: `docs/development/adding-job-types.md` (NEW)
  - Guide: Step-by-step to add new job with progress tracking
  - Example: "Adding a new 'Model Fine-Tuning' job"
  - Sections:
    1. Create database model with progress fields
    2. Create Celery task inheriting from BaseTask
    3. Add WebSocket emission calls
    4. Create frontend store actions
    5. Add WebSocket subscription hook
    6. Create UI component
  - Code snippets: For each step
  - **Acceptance:** Developer can follow guide to add new job type

- [ ] **LP-3.3: Document WebSocket Channel Naming Conventions**
  - File: `docs/architecture/websocket-channels.md` (NEW)
  - Convention: `{entity_type}/{entity_id}/{event_type}`
  - Examples:
    - `trainings/{training_id}/progress`
    - `models/{model_id}/extraction`
    - `datasets/{dataset_id}/progress`
    - `system/gpu/{gpu_id}`
  - Reserved prefixes: `system/`, `admin/`
  - **Acceptance:** Channel naming documented, examples clear

- [ ] **LP-3.4: Write Testing Strategy Document**
  - File: `docs/development/testing-strategy.md` (NEW)
  - Coverage targets: Unit 70%, Integration 50%, E2E 30%
  - Test organization: Where to put different test types
  - Mocking strategy: When to mock, when to use real instances
  - Running tests: Commands for different test suites
  - CI/CD: How tests run in pipeline
  - **Acceptance:** Testing approach documented

- [ ] **LP-3.5: Document WebSocket Clustering Setup**
  - File: `docs/deployment/websocket-clustering.md` (NEW)
  - Prerequisites: Redis instance for pub/sub
  - Configuration: Environment variables, Redis connection
  - Setup: Docker Compose multi-instance example
  - Nginx: Load balancing configuration
  - Testing: How to verify clustering works
  - Troubleshooting: Common issues and solutions
  - **Acceptance:** Production deployment guide complete

- [ ] **LP-3.6: Update CLAUDE.md with Architecture Changes**
  - File: `CLAUDE.md`
  - Add: System monitoring now uses WebSocket (update from polling)
  - Add: WebSocket clustering available for production
  - Add: Progress tracking architecture diagram reference
  - Update: Test coverage targets and current status
  - **Acceptance:** Project memory updated with latest architecture

- [ ] **LP-3.7: Create API Documentation for WebSocket Events**
  - File: `docs/api/websocket-events.md` (NEW)
  - Document: All WebSocket event types
  - For each event:
    - Channel name
    - Event data structure (JSON schema)
    - When emitted
    - Example payload
  - Examples:
    - `training:progress` event structure
    - `checkpoint:created` event structure
    - `system/gpu/{id}` metrics structure
  - **Acceptance:** Complete WebSocket API reference

---

## 📊 Summary Statistics

### Effort Estimates:
- **High Priority (2 weeks):** 24-32 hours total
  - HP-1: System Monitoring WebSocket Migration: 8-12 hours
  - HP-2: Test Coverage Phase 1: 16-20 hours

- **Medium Priority (1 month):** 58-74 hours total
  - MP-1: WebSocket Clustering: 12-16 hours
  - MP-2: TrainingMetric Archival: 10-14 hours
  - MP-3: Operations Dashboard: 16-20 hours
  - MP-4: Test Coverage Phase 2: 20-24 hours

- **Low Priority (2 months):** 28-38 hours total
  - LP-1: Progress History Visualization: 12-16 hours
  - LP-2: Performance Optimization: 8-10 hours
  - LP-3: Documentation: 8-12 hours

**Total Estimated Effort:** 110-144 hours (14-18 developer days)

### Task Breakdown:
- **High Priority:** 2 major tasks, 20 sub-tasks
- **Medium Priority:** 4 major tasks, 37 sub-tasks
- **Low Priority:** 3 major tasks, 22 sub-tasks
- **Total:** 9 major tasks, 79 sub-tasks

### Coverage Targets:
- **Current:** 40% unit, 20% integration, 10% e2e
- **After Phase 1:** 60% unit, 25% integration, 15% e2e
- **After Phase 2:** 70% unit, 40% integration, 30% e2e

---

## 🎯 Recommended Execution Order

### Week 1-2: High Priority
1. Start HP-1 (System Monitoring WebSocket) - Architectural consistency
2. Start HP-2 (Test Coverage Phase 1) - In parallel

### Week 3-4: Medium Priority - Core Infrastructure
1. Start MP-1 (WebSocket Clustering) - Scalability foundation
2. Start MP-2 (TrainingMetric Archival) - Prevent future issues

### Week 5-6: Medium Priority - User Experience
1. Start MP-3 (Operations Dashboard) - User-facing improvement
2. Start MP-4 (Test Coverage Phase 2) - Continue quality improvements

### Week 7-8: Low Priority - Polish
1. Start LP-1 (Progress History) - Advanced features
2. Start LP-2 (Performance Optimization) - Efficiency gains
3. Start LP-3 (Documentation) - Knowledge capture

---

## 📝 Notes

- All tasks assume current architecture remains stable (no major refactoring)
- Security and rate limiting explicitly excluded per user request
- Test coverage targets are aggressive but achievable with dedicated effort
- WebSocket clustering (MP-1) should be completed before production deployment
- TrainingMetric archival (MP-2) can be deferred if training volume is low (<100 trainings)
- Operations Dashboard (MP-3) is nice-to-have but provides significant UX improvement
- Progress History (LP-1) is optional, mainly for advanced users/optimization
- Documentation (LP-3) should not be skipped - prevents future knowledge loss

---

**Task List Created:** 2025-10-22
**Review Session:** review_progress_monitoring_architecture_2025-10-22
**Excluding:** Security (authentication, authorization, rate limiting)
