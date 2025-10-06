# Task List: SAE Training

**Feature:** Sparse Autoencoder Training Configuration and Execution
**PRD Reference:** 003_FPRD|SAE_Training.md
**TDD Reference:** 003_FTDD|SAE_Training.md
**TID Reference:** 003_FTID|SAE_Training.md
**Mock UI Reference:** Lines 1628-2156 (TrainingPanel, TrainingCard, Checkpoints, Live Metrics)
**Status:** Ready for Implementation
**Created:** 2025-10-06

---

## Relevant Files

### Backend
- `backend/src/models/training.py` - SQLAlchemy model for trainings table
- `backend/src/models/training_metric.py` - SQLAlchemy model for training_metrics table
- `backend/src/models/checkpoint.py` - SQLAlchemy model for checkpoints table
- `backend/src/schemas/training.py` - Pydantic schemas for training configuration and validation
- `backend/src/services/training_service.py` - Business logic for training lifecycle (create, pause, resume, stop)
- `backend/src/services/checkpoint_service.py` - Checkpoint save/load/delete, retention policy
- `backend/src/workers/training_tasks.py` - Celery task: train_sae_task (main training loop)
- `backend/src/ml/sae_model.py` - PyTorch SAE model (SparseAutoencoder, SkipAutoencoder, Transcoder)
- `backend/src/api/routes/trainings.py` - FastAPI routes for training endpoints
- `backend/alembic/versions/003_create_training_tables.py` - Database migration for trainings tables

### Frontend
- `frontend/src/types/training.types.ts` - TypeScript interfaces for Training, TrainingMetric, Checkpoint
- `frontend/src/stores/trainingsStore.ts` - Zustand store for training state management
- `frontend/src/components/panels/TrainingPanel.tsx` - Configuration UI (Mock UI lines 1628-1842)
- `frontend/src/components/trainings/TrainingCard.tsx` - Progress tracking and controls (Mock UI lines 1845-2156)
- `frontend/src/components/trainings/CheckpointManagement.tsx` - Checkpoint UI (Mock UI lines 1941-2027)
- `frontend/src/components/trainings/LiveMetrics.tsx` - Loss/sparsity charts and logs (Mock UI lines 2031-2086)

### Tests
- `backend/tests/test_training_service.py` - Unit tests for TrainingService
- `backend/tests/test_sae_model.py` - Unit tests for SAE forward pass and loss calculation
- `backend/tests/test_training_tasks.py` - Integration tests for train_sae_task
- `frontend/tests/TrainingPanel.test.tsx` - Unit tests for TrainingPanel component
- `frontend/tests/TrainingCard.test.tsx` - Unit tests for TrainingCard component
- `backend/tests/e2e/test_training_flow.py` - E2E test: create training → pause → resume → complete

---

## Tasks

- [ ] 1.0 Phase 1: Backend Infrastructure - Database Schema and Models
  - [ ] 1.1 Create migration `003_create_training_tables.py` with trainings, training_metrics, checkpoints tables (TDD lines 264-407)
  - [ ] 1.2 Add indexes: idx_trainings_status, idx_trainings_model_id, idx_trainings_dataset_id, idx_trainings_active, idx_training_metrics_training_step
  - [ ] 1.3 Create SQLAlchemy model `Training` in `backend/src/models/training.py` with all fields from TDD lines 264-327
  - [ ] 1.4 Create SQLAlchemy model `TrainingMetric` in `backend/src/models/training_metric.py` (TDD lines 345-371)
  - [ ] 1.5 Create SQLAlchemy model `Checkpoint` in `backend/src/models/checkpoint.py` (TDD lines 380-405)
  - [ ] 1.6 Add `hyperparameters` JSONB field with structure from TDD lines 329-341
  - [ ] 1.7 Add `progress` computed column: `current_step::FLOAT / total_steps * 100` (TDD line 287)
  - [ ] 1.8 Add foreign keys: `model_id REFERENCES models(id)`, `dataset_id REFERENCES datasets(id)` with ON DELETE RESTRICT
  - [ ] 1.9 Run migration: `alembic upgrade head` and verify tables created
  - [ ] 1.10 Write unit tests for model relationships and constraints

- [ ] 2.0 Phase 2: PyTorch SAE Model Implementation
  - [ ] 2.1 Create `backend/src/ml/sae_model.py` with `SparseAutoencoder` class (TID lines 29-119)
  - [ ] 2.2 Implement `__init__`: encoder Linear(hidden_dim, latent_dim), decoder Linear(latent_dim, hidden_dim), l1_alpha parameter
  - [ ] 2.3 Implement `forward` method: encoder → ReLU → decoder, return (reconstructed, latent, l0_sparsity) (TID lines 70-89)
  - [ ] 2.4 Implement `compute_loss` method: reconstruction_loss (MSE) + sparsity_loss (L1 penalty) (TID lines 91-118)
  - [ ] 2.5 Add weight initialization: `nn.init.xavier_uniform_` for encoder/decoder weights (TID line 66)
  - [ ] 2.6 Add optional `tie_weights` parameter for tied encoder/decoder (TID lines 59-63)
  - [ ] 2.7 Implement `SkipAutoencoder` variant with residual connection (TDD lines 1514-1526)
  - [ ] 2.8 Implement `Transcoder` variant for cross-layer mapping (TDD lines 1529-1540)
  - [ ] 2.9 Add `calculate_dead_neurons` utility function (TID lines 247-251)
  - [ ] 2.10 Write unit tests for SAE forward pass, loss calculation, and architecture variants
  - [ ] 2.11 Test memory footprint: verify 8x expansion fits in 6GB GPU VRAM (TDD lines 1575-1587)
  - [ ] 2.12 Test gradient flow: ensure no NaN/Inf gradients during backprop

- [ ] 3.0 Phase 3: Backend Pydantic Schemas and Validation
  - [ ] 3.1 Create `backend/src/schemas/training.py` with `TrainingConfigRequest` schema (TDD lines 412-434)
  - [ ] 3.2 Add `HyperparametersSchema` with field validation: learningRate (0.000001-0.1), batchSize (64-2048), l1Coefficient (0.00001-0.01), etc. (TDD lines 435-444)
  - [ ] 3.3 Add `@validator` for `dataset_id`: check `dataset.status == 'ready'` in database (TDD lines 420-426)
  - [ ] 3.4 Add `@validator` for `model_id`: check `model.status == 'ready'` in database (TDD lines 428-433)
  - [ ] 3.5 Add `encoder_type` field: Literal["sparse", "skip", "transcoder"] (TDD line 417)
  - [ ] 3.6 Create `TrainingResponse` schema with all training fields for API responses
  - [ ] 3.7 Create `TrainingMetricResponse` schema for metrics API (TDD lines 649-674)
  - [ ] 3.8 Create `CheckpointResponse` schema for checkpoint API (TDD lines 686-698)
  - [ ] 3.9 Write unit tests for validation logic (invalid hyperparameters, non-ready datasets/models)
  - [ ] 3.10 Test edge cases: batch_size too large (>2048), learning_rate negative, expansion_factor < 4

- [ ] 4.0 Phase 4: Backend Services - TrainingService
  - [ ] 4.1 Create `backend/src/services/training_service.py` with `TrainingService` class
  - [ ] 4.2 Implement `create_training(config)`: validate config, create DB record (status='queued'), enqueue Celery task (TDD lines 846-851)
  - [ ] 4.3 Implement `pause_training(training_id)`: validate status='training', set Redis pause signal, return training (TDD lines 852-854)
  - [ ] 4.4 Implement `resume_training(training_id)`: validate status='paused', re-enqueue Celery task with resume=True (TDD lines 855-858)
  - [ ] 4.5 Implement `stop_training(training_id)`: validate status in ['training', 'paused'], set Redis stop signal (TDD lines 859-862)
  - [ ] 4.6 Implement `get_training(training_id)`: fetch training with model and dataset joined
  - [ ] 4.7 Implement `list_trainings(status_filter, model_id, dataset_id, limit, offset)`: support filtering and pagination (TDD lines 546-580)
  - [ ] 4.8 Implement `save_metrics(training_id, step, metrics)`: insert into training_metrics table (TDD lines 345-371)
  - [ ] 4.9 Implement `update_training_status(training_id, status, error_msg)`: update status and denormalized metrics fields
  - [ ] 4.10 Write unit tests for all service methods with mocked database

- [ ] 5.0 Phase 5: Backend Services - CheckpointService
  - [ ] 5.1 Create `backend/src/services/checkpoint_service.py` with `CheckpointService` class (TID lines 441-531)
  - [ ] 5.2 Implement `__init__(training_id)`: set checkpoint_dir to `/data/checkpoints/{training_id}`, create directory (TID lines 451-454)
  - [ ] 5.3 Implement `save_checkpoint(training_id, step, sae_state, optimizer_state, is_best)`: save to safetensors format (TID lines 456-477)
  - [ ] 5.4 Use safetensors naming: `step_{step}.safetensors` or `best.safetensors` if is_best=True (TID line 464)
  - [ ] 5.5 Implement `load_checkpoint(checkpoint_path)`: load from safetensors, return state_dict (TID lines 479-492)
  - [ ] 5.6 Implement `delete_checkpoint(checkpoint_id)`: delete file from filesystem and DB record
  - [ ] 5.7 Implement `enforce_retention_policy(training_id, keep_first, keep_last, keep_every_n)`: keep first, last, every Nth, best (TID lines 494-530)
  - [ ] 5.8 Add checkpoint integrity validation: verify file size matches DB record
  - [ ] 5.9 Write unit tests for save/load/delete with temporary filesystem
  - [ ] 5.10 Test retention policy: verify correct checkpoints kept/deleted

- [ ] 6.0 Phase 6: Celery Training Task - Core Training Loop
  - [ ] 6.1 Create `backend/src/workers/training_tasks.py` with `@celery_app.task(bind=True)` decorator (TID line 136)
  - [ ] 6.2 Implement `train_sae_task(self, training_id, config)` with async wrapper (TID lines 136-245)
  - [ ] 6.3 Load training configuration: fetch Training record, extract hyperparameters (TDD lines 904-906)
  - [ ] 6.4 Update status to 'initializing', emit WebSocket 'training:status_changed' event (TDD lines 908-910)
  - [ ] 6.5 Load model via ModelRegistry: `model, tokenizer = load_model(training.model_id)` (TID line 157)
  - [ ] 6.6 Load dataset: `dataset = load_dataset(training.dataset_id)` (TID line 158)
  - [ ] 6.7 Create SAE architecture based on encoder_type: sparse/skip/transcoder (TID lines 159-163)
  - [ ] 6.8 Initialize optimizer: Adam/AdamW/SGD based on config (TID line 166)
  - [ ] 6.9 Initialize LR scheduler: constant/linear/cosine/exponential (TID lines 167-169)
  - [ ] 6.10 Initialize mixed precision scaler: `torch.cuda.amp.GradScaler()` (TID line 172)
  - [ ] 6.11 Create ActivationExtractor with forward hooks (TID lines 175-177)
  - [ ] 6.12 Update status to 'training', emit WebSocket event (TDD lines 937-939)

- [ ] 7.0 Phase 7: Celery Training Task - Training Loop Execution
  - [ ] 7.1 Implement training loop: `for step in range(config['total_steps'])` (TID line 180)
  - [ ] 7.2 Check pause signal: `if redis.get(f'training:{training_id}:pause_signal')` → save checkpoint, update status='paused', return (TDD lines 946-950)
  - [ ] 7.3 Check stop signal: `if redis.get(f'training:{training_id}:stop_signal')` → save final checkpoint, update status='stopped', return (TDD lines 952-956)
  - [ ] 7.4 Get batch from dataset: `batch = get_next_batch(dataset, config['batch_size'])` (TID line 182)
  - [ ] 7.5 Extract activations: `activations = extractor.extract_from_batch(input_ids)` (TID lines 186-187)
  - [ ] 7.6 Forward pass with mixed precision: `with autocast(): reconstructed, latent, l0_sparsity = sae(x)` (TID lines 191-192)
  - [ ] 7.7 Compute loss: `loss_dict = sae.compute_loss(x, reconstructed, latent)` (TID line 193)
  - [ ] 7.8 Backward pass: `scaler.scale(loss).backward()`, gradient clipping, `scaler.step(optimizer)` (TID lines 197-200)
  - [ ] 7.9 Scheduler step: `scheduler.step()` (TID line 200)
  - [ ] 7.10 Calculate metrics every 10 steps: loss, l0_sparsity, dead_neurons, learning_rate (TID lines 203-210)
  - [ ] 7.11 Save metrics to database: `await service.save_metrics(training_id, metrics)` (TID line 213)
  - [ ] 7.12 Emit WebSocket progress event with metrics (TID lines 216-217)
  - [ ] 7.13 Auto-save checkpoint if interval reached (TID lines 220-222)
  - [ ] 7.14 Check manual checkpoint save signal from Redis (TDD lines 1019-1022)
  - [ ] 7.15 Clear GPU cache: `torch.cuda.empty_cache()` (TID line 239, TDD line 1025)

- [ ] 8.0 Phase 8: Celery Training Task - Error Handling and Completion
  - [ ] 8.1 Wrap training loop in try-except for OOM errors (TID lines 236-243)
  - [ ] 8.2 On OOM: reduce batch_size by 50%, clear GPU cache, log warning, continue (TDD lines 1028-1033)
  - [ ] 8.3 Implement gradient accumulation if batch_size falls below threshold (TDD lines 1194-1196)
  - [ ] 8.4 Save final checkpoint: `save_checkpoint(sae, optimizer, scheduler, config['total_steps'], final_path)` (TID lines 225-226)
  - [ ] 8.5 Update training status to 'completed', set completed_at timestamp (TID lines 229-231)
  - [ ] 8.6 Emit WebSocket 'training:completed' event (TDD line 1040)
  - [ ] 8.7 Cleanup activation extractor: `extractor.cleanup()` (TID line 234)
  - [ ] 8.8 Handle generic exceptions: log error, update status='error', set error_message (TID lines 242-243)
  - [ ] 8.9 Implement checkpoint resume logic: load checkpoint if resume=True flag set (TDD lines 928-935)
  - [ ] 8.10 Write integration tests for training loop: mock model/dataset, run 100 steps, verify metrics saved

- [ ] 9.0 Phase 9: FastAPI Routes - Training Endpoints
  - [ ] 9.1 Create `backend/src/api/routes/trainings.py` with APIRouter
  - [ ] 9.2 Implement `POST /api/trainings`: validate config, call `training_service.create_training()`, return 201 Created (TDD lines 502-543)
  - [ ] 9.3 Implement `GET /api/trainings`: list trainings with filtering (status, model_id, dataset_id), pagination (TDD lines 545-580)
  - [ ] 9.4 Implement `GET /api/trainings/:id`: fetch single training with model and dataset joined
  - [ ] 9.5 Implement `POST /api/trainings/:id/pause`: call `training_service.pause_training()`, return 200 OK (TDD lines 582-600)
  - [ ] 9.6 Implement `POST /api/trainings/:id/resume`: call `training_service.resume_training()`, return 200 OK (TDD lines 602-620)
  - [ ] 9.7 Implement `POST /api/trainings/:id/stop`: call `training_service.stop_training()`, return 200 OK (TDD lines 621-638)
  - [ ] 9.8 Implement `GET /api/trainings/:id/metrics`: fetch training_metrics with filtering (start_step, end_step, interval) (TDD lines 639-674)
  - [ ] 9.9 Implement `POST /api/trainings/:id/checkpoints`: set Redis checkpoint save signal, return 201 Created (TDD lines 676-704)
  - [ ] 9.10 Add error handling: return 400 for validation errors, 404 for not found, 409 for status conflicts (TDD lines 760-783)
  - [ ] 9.11 Add JWT authentication to all endpoints using FastAPI dependency
  - [ ] 9.12 Write API integration tests for all endpoints with test client

- [ ] 10.0 Phase 10: WebSocket Integration - Real-time Progress Updates
  - [ ] 10.1 Create WebSocket event emitter utility in `backend/src/core/websocket.py`
  - [ ] 10.2 Implement `emit_event(channel, data)` function for publishing to WebSocket rooms
  - [ ] 10.3 Add 'training:created' event emission in `training_service.create_training()` (TDD lines 717-723)
  - [ ] 10.4 Add 'training:status_changed' event emission in `update_training_status()` (TDD lines 726-733)
  - [ ] 10.5 Add 'training:progress' event emission in training loop (every 10 steps) (TDD lines 736-748, TID lines 216-217)
  - [ ] 10.6 Add 'checkpoint:created' event emission in `checkpoint_service.save_checkpoint()` (TDD lines 750-758)
  - [ ] 10.7 Include all metrics in progress event: current_step, progress%, loss, sparsity, dead_neurons, gpu_utilization (TDD lines 738-746)
  - [ ] 10.8 Test WebSocket events: verify correct channel and payload structure
  - [ ] 10.9 Add WebSocket reconnection logic on frontend for connection drops
  - [ ] 10.10 Add polling fallback if WebSocket unavailable: poll `/api/trainings/:id` every 5 seconds (TDD lines 1144-1147)

- [ ] 11.0 Phase 11: Frontend Types and Store
  - [ ] 11.1 Create `frontend/src/types/training.types.ts` with `Training` interface (all fields from TDD lines 264-327)
  - [ ] 11.2 Add `TrainingMetric` interface with fields: step, loss, sparsity, reconstruction_error, dead_neurons, learning_rate (TDD lines 345-365)
  - [ ] 11.3 Add `Checkpoint` interface with fields: id, training_id, step, loss, storage_path, file_size_bytes, created_at (TDD lines 380-405)
  - [ ] 11.4 Add `HyperparametersConfig` type with all hyperparameter fields from TDD lines 329-341
  - [ ] 11.5 Create `frontend/src/stores/trainingsStore.ts` with Zustand store (TDD lines 1050-1067)
  - [ ] 11.6 Add `trainings: Training[]` state array
  - [ ] 11.7 Add `selectedConfig: TrainingConfig` state for form
  - [ ] 11.8 Implement `fetchTrainings()` action: GET /api/trainings
  - [ ] 11.9 Implement `createTraining(config)` action: POST /api/trainings, add to trainings array
  - [ ] 11.10 Implement `pauseTraining(trainingId)` action: POST /api/trainings/:id/pause
  - [ ] 11.11 Implement `resumeTraining(trainingId)` action: POST /api/trainings/:id/resume
  - [ ] 11.12 Implement `stopTraining(trainingId)` action: POST /api/trainings/:id/stop
  - [ ] 11.13 Implement `updateTrainingStatus(trainingId, updates)` action for WebSocket updates
  - [ ] 11.14 Write unit tests for store actions with mocked API calls

- [ ] 12.0 Phase 12: Frontend UI - TrainingPanel Component
  - [ ] 12.1 Create `frontend/src/components/panels/TrainingPanel.tsx` matching Mock UI lines 1628-1842 (TID lines 280-357)
  - [ ] 12.2 Add state: `config` object with model_id, dataset_id, encoder_type, hyperparameters (TID lines 295-306)
  - [ ] 12.3 Add state: `showAdvanced` boolean for collapsible hyperparameters section (TID line 308)
  - [ ] 12.4 Implement configuration form: 3-column grid with Model, Dataset, Encoder Type dropdowns (Mock UI lines 1653-1665, TID lines 317-327)
  - [ ] 12.5 Filter models: only show models with status='ready' (TID line 292)
  - [ ] 12.6 Filter datasets: only show datasets with status='ready' (TID line 293)
  - [ ] 12.7 Implement encoder type dropdown: sparse/skip/transcoder options (Mock UI lines 1659-1663)
  - [ ] 12.8 Add "Advanced Configuration" collapsible button (Mock UI lines 1667-1670, TID lines 330-333)
  - [ ] 12.9 Implement advanced hyperparameters section: 2-column grid with all inputs (Mock UI lines 1673-1814, TID lines 336-339)
  - [ ] 12.10 Add hyperparameter inputs: layer (dropdown 0-23), latent_dim (8192/16384/32768), l1_alpha (0.0001-0.01), learning_rate (0.0001-0.001), batch_size (64/128/256/512), total_steps (1000-100000), optimizer (Adam/AdamW/SGD), lr_schedule (constant/linear/cosine/exponential)
  - [ ] 12.11 Add ghost gradient penalty toggle switch (Mock UI lines 1797-1814, TID example)
  - [ ] 12.12 Implement "Start Training" button: disabled if !model_id || !dataset_id, calls `createTraining()` (Mock UI lines 1816-1824, TID lines 342-346)
  - [ ] 12.13 Add "Training Jobs" section below form listing all trainings with TrainingCard components (Mock UI lines 1827-1841, TID lines 350-353)
  - [ ] 12.14 Use exact Tailwind classes from Mock UI: bg-slate-900/50, border-slate-800, focus:border-emerald-500, disabled:bg-slate-700
  - [ ] 12.15 Write unit tests for TrainingPanel: form validation, button states, model/dataset filtering

- [ ] 13.0 Phase 13: Frontend UI - TrainingCard Component (Header and Progress)
  - [ ] 13.1 Create `frontend/src/components/trainings/TrainingCard.tsx` matching Mock UI lines 1845-2156 (TID lines 362-435)
  - [ ] 13.2 Add state: `showMetrics` boolean, `showCheckpoints` boolean (Mock UI lines 1848-1849, TID line 373)
  - [ ] 13.3 Implement header section: model + dataset name, encoder type, start time, status badge (Mock UI lines 1862-1876, TID lines 387-397)
  - [ ] 13.4 Add status icons: Activity (training, animate-pulse), CheckCircle (completed), Pause (paused), Loader (initializing, animate-spin) (Mock UI lines 1868-1871)
  - [ ] 13.5 Implement progress section: only show if status in ['training', 'completed', 'paused'] (Mock UI line 1878)
  - [ ] 13.6 Add progress bar: h-2, bg-slate-800, gradient from-emerald-500 to-emerald-400 (Mock UI lines 1880-1889, TID lines 402-411)
  - [ ] 13.7 Implement metrics grid: 4 columns (Loss, L0 Sparsity, Dead Neurons, GPU Util) (Mock UI lines 1891-1916, TID lines 414-420)
  - [ ] 13.8 Use color coding: Loss (emerald-400), L0 Sparsity (blue-400), Dead Neurons (red-400), GPU Util (purple-400) (Mock UI lines 1894-1914)
  - [ ] 13.9 Calculate metrics from training state: loss, l0_sparsity, dead_neurons (Mock UI lines 1853-1856)
  - [ ] 13.10 Show "—" placeholder if training.progress <= 10 (not enough data yet) (Mock UI line 1895)
  - [ ] 13.11 Add "Show/Hide Live Metrics" button: only show if status='training' (Mock UI lines 1918-1927)
  - [ ] 13.12 Add "Checkpoints" button with count badge (Mock UI lines 1928-1937)
  - [ ] 13.13 Write unit tests for TrainingCard: status badge rendering, metrics calculation, button visibility

- [ ] 14.0 Phase 14: Frontend UI - CheckpointManagement Component
  - [ ] 14.1 Create CheckpointManagement section in TrainingCard (Mock UI lines 1941-2028)
  - [ ] 14.2 Add conditional rendering: only show if `showCheckpoints === true` (Mock UI line 1941)
  - [ ] 14.3 Implement "Save Now" button: calls `saveCheckpoint(trainingId)` API (Mock UI lines 1945-1954)
  - [ ] 14.4 Implement checkpoint list: scrollable max-h-48, map over trainingCheckpoints array (Mock UI lines 1957-1996)
  - [ ] 14.5 Display checkpoint item: step number, loss (4 decimals), timestamp (Mock UI lines 1960-1966)
  - [ ] 14.6 Add "Load" button: calls `loadCheckpoint(trainingId, checkpointId)` API (Mock UI lines 1968-1977)
  - [ ] 14.7 Add "Delete" button: red-400 color, calls `deleteCheckpoint(trainingId, checkpointId)` API (Mock UI lines 1978-1987)
  - [ ] 14.8 Show "No checkpoints saved yet" placeholder if empty (Mock UI lines 1992-1996)
  - [ ] 14.9 Implement auto-save configuration section: toggle switch + interval input (Mock UI lines 1998-2027)
  - [ ] 14.10 Add auto-save toggle: w-12 h-6 rounded-full, bg-emerald-600 (on) or bg-slate-600 (off) (Mock UI lines 2001-2012)
  - [ ] 14.11 Add auto-save interval input: only show if autoSave=true, min=100 max=10000 step=100 (Mock UI lines 2014-2026)
  - [ ] 14.12 Use border-t border-slate-700 pt-3 spacing between sections (Mock UI line 1998)
  - [ ] 14.13 Write unit tests for checkpoint management: save/load/delete actions, auto-save toggle

- [ ] 15.0 Phase 15: Frontend UI - LiveMetrics Component
  - [ ] 15.1 Create LiveMetrics section in TrainingCard (Mock UI lines 2031-2086)
  - [ ] 15.2 Add conditional rendering: only show if `showMetrics === true && status === 'training'` (Mock UI line 2031)
  - [ ] 15.3 Implement Loss Curve chart: bar chart with 20 bars, bg-emerald-500 (Mock UI lines 2033-2047)
  - [ ] 15.4 Use h-24 height, flex items-end gap-1 for bar chart container (Mock UI line 2035)
  - [ ] 15.5 Calculate bar heights: decreasing trend for loss (100 - i * 3 - random) (Mock UI line 2037)
  - [ ] 15.6 Implement L0 Sparsity chart: bar chart with 20 bars, bg-blue-500 (Mock UI lines 2049-2063)
  - [ ] 15.7 Calculate bar heights: increasing trend for sparsity (30 + i * 2 + random) (Mock UI line 2053)
  - [ ] 15.8 Implement Training Logs section: bg-slate-950, font-mono text-xs, h-32 overflow-y-auto (Mock UI lines 2065-2084)
  - [ ] 15.9 Display log entries: timestamp (slate-500), step number, loss, sparsity, dead_neurons, GPU util (Mock UI lines 2071-2082)
  - [ ] 15.10 Add "Live" badge in logs header: text-emerald-400 text-xs (Mock UI line 2068)
  - [ ] 15.11 Use WebSocket subscription to update charts in real-time every 10 steps
  - [ ] 15.12 Keep last 20 metrics points for charts: `metrics.slice(-20)` (TID line 379)
  - [ ] 15.13 Write unit tests for live metrics: chart rendering, log entries, WebSocket integration

- [ ] 16.0 Phase 16: Frontend UI - Control Buttons
  - [ ] 16.1 Implement control buttons section: border-t border-slate-700 pt-4 (Mock UI lines 2090-2153)
  - [ ] 16.2 Add conditional rendering: hide if status='completed' (Mock UI line 2090)
  - [ ] 16.3 If status='training': show Pause and Stop buttons (Mock UI lines 2092-2114)
  - [ ] 16.4 Pause button: bg-yellow-600 hover:bg-yellow-700, pause icon, calls `pauseTraining(trainingId)` (Mock UI lines 2094-2103)
  - [ ] 16.5 Stop button: bg-red-600 hover:bg-red-700, stop icon, calls `stopTraining(trainingId)` (Mock UI lines 2104-2113)
  - [ ] 16.6 If status='paused': show Resume and Stop buttons (Mock UI lines 2117-2137)
  - [ ] 16.7 Resume button: bg-emerald-600 hover:bg-emerald-700, Play icon, calls `resumeTraining(trainingId)` (Mock UI lines 2119-2126)
  - [ ] 16.8 If status='stopped': show Retry button (Mock UI lines 2140-2151)
  - [ ] 16.9 Retry button: bg-blue-600 hover:bg-blue-700, retry icon, calls `retryTraining(trainingId)` (Mock UI lines 2141-2150)
  - [ ] 16.10 Use flex-1 for buttons to split width evenly, gap-2 between buttons (Mock UI lines 2091, 2094, 2119)
  - [ ] 16.11 Add transition-colors for smooth hover effects (Mock UI line 2098)
  - [ ] 16.12 Write unit tests for control buttons: correct buttons shown per status, API calls triggered

- [ ] 17.0 Phase 17: WebSocket Frontend Integration
  - [ ] 17.1 Create `frontend/src/hooks/useWebSocket.ts` custom hook for WebSocket connection
  - [ ] 17.2 Implement `subscribe(channel)` and `unsubscribe(channel)` functions
  - [ ] 17.3 Add WebSocket connection to trainingsStore: connect on mount, disconnect on unmount
  - [ ] 17.4 Subscribe to 'training:created' events: add new training to trainings array (TDD lines 717-723)
  - [ ] 17.5 Subscribe to 'training:status_changed' events: update training status (TDD lines 726-733)
  - [ ] 17.6 Subscribe to 'training:progress' events: update current_step, progress, metrics (TDD lines 736-748)
  - [ ] 17.7 Subscribe to 'checkpoint:created' events: add checkpoint to checkpoints array (TDD lines 750-758)
  - [ ] 17.8 Update TrainingCard to use real-time metrics from WebSocket instead of mock data (TID lines 376-382)
  - [ ] 17.9 Implement automatic reconnection on WebSocket disconnect with exponential backoff
  - [ ] 17.10 Add connection status indicator in UI (connected/disconnected badge)
  - [ ] 17.11 Test WebSocket subscription: verify events received and state updated correctly

- [ ] 18.0 Phase 18: Memory Optimization and OOM Handling
  - [ ] 18.1 Add dynamic batch size reduction in training loop on OOM error (TDD lines 1028-1033)
  - [ ] 18.2 Implement gradient accumulation when batch_size < 64 to maintain effective batch size (TDD line 1194)
  - [ ] 18.3 Add GPU cache clearing: `torch.cuda.empty_cache()` after every training step (TID line 239)
  - [ ] 18.4 Implement memory monitoring: log GPU memory usage every 100 steps
  - [ ] 18.5 Add memory budget validation before training start: estimate model + SAE + activations + gradients + optimizer state (TDD lines 1575-1587)
  - [ ] 18.6 Show OOM error message in UI with actionable suggestions: "Reduce batch size or expansion factor" (TDD lines 768-776)
  - [ ] 18.7 Add retry count tracking: increment on OOM, stop after 3 retries (training table has retry_count field)
  - [ ] 18.8 Test OOM handling: simulate OOM error, verify batch_size reduced, training continues
  - [ ] 18.9 Add ghost gradient penalty to reduce dead neurons (TDD line 339, Mock UI lines 1797-1814)
  - [ ] 18.10 Test ghost gradient penalty: verify dead neuron count decreases with penalty enabled

- [ ] 19.0 Phase 19: Testing - Unit Tests
  - [ ] 19.1 Write unit tests for `SparseAutoencoder` forward pass: verify output shapes, no NaN values (TDD lines 1242-1252)
  - [ ] 19.2 Write unit tests for SAE loss calculation: verify reconstruction loss + L1 penalty (TDD lines 1242-1252)
  - [ ] 19.3 Write unit tests for `HyperparametersSchema` validation: valid config, invalid ranges (TDD lines 1222-1239)
  - [ ] 19.4 Write unit tests for `TrainingService.create_training()`: verify DB record created, Celery task enqueued
  - [ ] 19.5 Write unit tests for `TrainingService.pause_training()`: verify Redis signal set, status updated
  - [ ] 19.6 Write unit tests for `CheckpointService.save_checkpoint()`: verify safetensors file saved, DB record created
  - [ ] 19.7 Write unit tests for `CheckpointService.enforce_retention_policy()`: verify correct checkpoints kept/deleted
  - [ ] 19.8 Write unit tests for frontend `trainingsStore`: verify API calls, state updates
  - [ ] 19.9 Write unit tests for `TrainingPanel` component: form validation, button disabled states
  - [ ] 19.10 Write unit tests for `TrainingCard` component: status badge rendering, metrics display
  - [ ] 19.11 Achieve >70% unit test coverage for backend services and frontend components

- [ ] 20.0 Phase 20: Testing - Integration and E2E Tests
  - [ ] 20.1 Write integration test: POST /api/trainings → verify training created, Celery task enqueued (TDD lines 1257-1277)
  - [ ] 20.2 Write integration test: training loop execution → verify 100 steps run, metrics saved, WebSocket events emitted
  - [ ] 20.3 Write integration test: pause training → verify checkpoint saved, status='paused', loop exits
  - [ ] 20.4 Write integration test: resume training → verify checkpoint loaded, training continues from current_step
  - [ ] 20.5 Write integration test: stop training → verify final checkpoint saved, status='stopped'
  - [ ] 20.6 Write E2E test: full training flow (create → train 50 steps → pause → resume → complete) (TDD lines 1257-1277)
  - [ ] 20.7 Write E2E test: checkpoint management (save manual checkpoint → load → verify state restored)
  - [ ] 20.8 Write E2E test: OOM handling (trigger OOM → verify batch_size reduced → training continues)
  - [ ] 20.9 Write performance test: training throughput benchmark on Jetson Orin Nano (target: >10 steps/sec) (TDD lines 1284-1292)
  - [ ] 20.10 Write performance test: checkpoint save time (target: <5 seconds for SAE <100M params)
  - [ ] 20.11 All integration and E2E tests passing before merging to main

---

## Notes

- **PRIMARY REFERENCE:** Mock UI lines 1628-2156 (TrainingPanel, TrainingCard, Checkpoints, Live Metrics) - production UI MUST match exactly
- **Architecture:** FastAPI + Celery for non-blocking training execution, PyTorch 2.0+ with mixed precision (AMP), WebSocket for real-time updates
- **Memory Optimization:** Critical for Jetson Orin Nano (6GB GPU VRAM) - use gradient accumulation, dynamic batch sizing, cache clearing
- **Checkpoint Format:** safetensors (safer and faster than pickle), retention policy (keep first, last, every 1000 steps, best loss)
- **Training Loop:** Extract activations → SAE forward → compute loss (reconstruction + L1 penalty) → backward → optimizer step → scheduler step
- **Metrics:** Saved every 10 steps to training_metrics table, emitted via WebSocket for real-time UI updates
- **Error Handling:** OOM errors gracefully handled with batch size reduction and retry logic, all errors logged with actionable messages
- **Testing:** Unit tests (>70% coverage), integration tests (training loop, pause/resume), E2E tests (full flow), performance tests (throughput, checkpoint save time)
- **Hyperparameters:** Learning rate (0.0001-0.001), batch size (64-512), L1 coefficient (0.0001-0.001), expansion factor (4-64), optimizer (Adam/AdamW/SGD), LR schedule (constant/linear/cosine/exponential), ghost gradient penalty (boolean)
- **Status Flow:** queued → initializing → training → (paused ↔ training)* → stopped/completed/failed
- **WebSocket Events:** training:created, training:status_changed, training:progress (every 10 steps), checkpoint:created
- **Control Actions:** Pause (save checkpoint, exit loop), Resume (load checkpoint, continue), Stop (save final checkpoint, exit permanently)
- **UI Polish:** Exact Tailwind classes from Mock UI (slate-900/50, emerald-600, blue-400, red-400, purple-400), smooth transitions, loading states, disabled states
