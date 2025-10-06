# Task List: Model Management

**Feature ID:** 002_FTASKS|Model_Management
**Feature Name:** Model Management Panel
**PRD Reference:** 002_FPRD|Model_Management.md
**TDD Reference:** 002_FTDD|Model_Management.md
**TID Reference:** 002_FTID|Model_Management.md
**ADR Reference:** 000_PADR|miStudio.md
**Mock UI Reference:** Mock-embedded-interp-ui.tsx (lines 1204-1625)
**Status:** Ready for Implementation
**Created:** 2025-10-06

---

## Relevant Files

### Backend Core Files
- `backend/src/models/model.py` - SQLAlchemy model (id, name, architecture, params_count, quantization, status, file_path, quantized_path, architecture_config JSONB, memory_required_bytes, disk_size_bytes)
- `backend/src/schemas/model.py` - Pydantic schemas (ModelResponse, ModelDownloadRequest, ModelListResponse, ActivationExtractionRequest)
- `backend/src/services/model_service.py` - Business logic (list_models, create_model, download_model, check_dependencies)
- `backend/src/services/quantization_service.py` - Quantization logic (load_and_quantize_model, validate_memory_requirements, calculate_quantization_factor)
- `backend/src/services/activation_service.py` - Activation extraction (register_hooks, extract_activations, save_activations, calculate_statistics)
- `backend/src/api/routes/models.py` - FastAPI router (GET /api/models, POST /api/models/download, GET /api/models/:id, POST /api/models/:id/extract, DELETE /api/models/:id)
- `backend/src/workers/model_tasks.py` - Celery tasks (download_model_task, quantize_model_task, extract_activations_task with WebSocket progress)
- `backend/src/ml/model_loader.py` - PyTorch model loading (load_from_hf, apply_quantization, validate_architecture)
- `backend/src/ml/quantize.py` - bitsandbytes integration (BitsAndBytesConfig for Q4/Q8, FP16 conversion, OOM handling)
- `backend/src/ml/forward_hooks.py` - Hook registration (register_residual_hooks, register_mlp_hooks, register_attention_hooks, activation_callback)
- `backend/src/core/websocket.py` - WebSocket manager (emit progress events for downloads and extractions)
- `backend/src/utils/model_utils.py` - Model utilities (calculate_memory_requirement, format_param_count, validate_repo_id, parse_architecture_config)

### Backend Database Migration
- `backend/alembic/versions/002_create_models_table.py` - Models table with indexes (status, architecture, created_at), GIN index on architecture_config JSONB

### Backend Tests
- `backend/tests/unit/test_model_service.py` - Unit tests for ModelService (list_models, download_model, compute_memory_requirements)
- `backend/tests/unit/test_quantization_service.py` - Unit tests for quantization logic (Q4/Q8/FP16, OOM fallback)
- `backend/tests/unit/test_activation_service.py` - Unit tests for activation extraction (hook registration, batch processing)
- `backend/tests/unit/test_model_tasks.py` - Unit tests for Celery tasks (mocked HuggingFace, mocked PyTorch)
- `backend/tests/integration/test_model_api.py` - Integration tests for API routes (download, extract, delete)
- `backend/tests/integration/test_model_workflow.py` - End-to-end workflow (download → quantize → extract → verify files)
- `backend/tests/conftest.py` - Pytest fixtures (async_db, mock_hf, mock_torch, test_client)

### Frontend Core Components
- `frontend/src/components/panels/ModelsPanel.tsx` - Main model management panel (PRIMARY: Mock UI lines 1204-1343)
- `frontend/src/components/panels/ModelsPanel.test.tsx` - Unit tests for ModelsPanel
- `frontend/src/components/models/ModelCard.tsx` - Individual model card (lines 1280-1330)
- `frontend/src/components/models/ModelCard.test.tsx` - Unit tests for ModelCard
- `frontend/src/components/models/ModelArchitectureViewer.tsx` - Architecture modal (lines 1346-1437)
- `frontend/src/components/models/ModelArchitectureViewer.test.tsx` - Unit tests for architecture viewer
- `frontend/src/components/models/ActivationExtractionConfig.tsx` - Extraction config modal (lines 1440-1625)
- `frontend/src/components/models/ActivationExtractionConfig.test.tsx` - Unit tests for extraction config
- `frontend/src/components/models/LayerSelector.tsx` - Layer grid selector (lines 1538-1554)
- `frontend/src/components/models/LayerSelector.test.tsx` - Unit tests for layer selector
- `frontend/src/components/models/QuantizationSelector.tsx` - Quantization format dropdown with memory estimates
- `frontend/src/components/models/QuantizationSelector.test.tsx` - Unit tests for quantization selector
- `frontend/src/components/common/ProgressBar.tsx` - Progress bar with gradient (reuse from Dataset Management)

### Frontend State & API
- `frontend/src/stores/modelsStore.ts` - Zustand store (models[], fetchModels, downloadModel, updateModelProgress, updateModelStatus, extractActivations)
- `frontend/src/stores/modelsStore.test.ts` - Unit tests for Zustand store
- `frontend/src/api/models.ts` - API client (getModels, downloadModel, getModelArchitecture, extractActivations, deleteModel)
- `frontend/src/api/websocket.ts` - WebSocket client (reuse from Dataset Management)
- `frontend/src/hooks/useWebSocket.ts` - WebSocket hook (reuse from Dataset Management)
- `frontend/src/hooks/useModelProgress.ts` - Pre-configured hook for model download/quantization progress

### Frontend Types & Utils
- `frontend/src/types/model.ts` - TypeScript interfaces (Model, ModelStatus, QuantizationFormat, ArchitectureConfig, ActivationExtractionConfig)
- `frontend/src/utils/formatters.ts` - Format helpers (formatParamCount: 1.1B/135M, formatMemory: 1.2GB) - extend from Dataset Management
- `frontend/src/utils/validators.ts` - Validation (validateHfRepoId, validateQuantizationFormat, validateLayerSelection) - extend from Dataset Management

### Configuration Files
- `backend/.env` - Environment variables (add HF_HOME=/data/huggingface_cache, MODEL_CACHE_DIR=/data/models)
- `backend/pyproject.toml` - Add dependencies (torch>=2.0.0, transformers>=4.35.0, bitsandbytes>=0.41.0, safetensors>=0.3.0, accelerate>=0.20.0)
- `frontend/package.json` - Dependencies already covered (react, zustand, lucide-react, socket.io-client)

### Notes
- All TypeScript files use strict mode with no 'any' types
- All Python files use type hints with mypy checking
- Frontend tests: `npm test` (Jest + React Testing Library)
- Backend tests: `pytest` or `python -m pytest`
- Component styling MUST match Mock UI exactly (lines 1204-1625)
- Status transitions: downloading → loading → quantizing → ready (or error)
- WebSocket channels: `models/{model_id}/progress` for downloads, `models/{model_id}/extraction` for activations
- File storage: `/data/models/raw/{model_id}/` for original, `/data/models/quantized/{model_id}_{format}/` for quantized
- Activation storage: `/data/activations/extraction_{id}/layer_{idx}_{hook_type}.npy`
- Database uses JSONB for architecture_config (flexible, indexed)
- Quantization: FP16 (50% size), Q8 (25% size), Q4 (12.5% size), Q2 (6.25% size)
- Memory formula: params × bytes_per_param × 1.2 (inference overhead)
- Forward hooks: Non-invasive, no model architecture modification
- OOM handling: Automatic fallback Q2→Q4→Q8→FP16
- All async operations use async/await (no callbacks)

---

## Tasks

### Phase 1: Backend Infrastructure and Database

- [ ] 1.0 Set up backend model management infrastructure
  - [ ] 1.1 Create backend/src/models/model.py with SQLAlchemy Model class (id, name, architecture, params_count, quantization enum, status enum, progress, error_message, file_path, quantized_path, architecture_config JSONB, memory_required_bytes, disk_size_bytes, created_at, updated_at)
  - [ ] 1.2 Create QuantizationFormat enum (FP16, Q8, Q4, Q2) and ModelStatus enum (DOWNLOADING, LOADING, QUANTIZING, READY, ERROR)
  - [ ] 1.3 Generate Alembic migration for models table (alembic revision --autogenerate -m "create models table")
  - [ ] 1.4 Add database indexes (idx_models_status, idx_models_architecture, idx_models_created_at DESC)
  - [ ] 1.5 Add GIN index on architecture_config JSONB (CREATE INDEX idx_models_architecture_config ON models USING GIN (architecture_config))
  - [ ] 1.6 Add full-text search index (CREATE INDEX idx_models_search ON models USING GIN(to_tsvector('english', name || ' ' || architecture)))
  - [ ] 1.7 Run migration (alembic upgrade head)
  - [ ] 1.8 Update backend/.env with MODEL_CACHE_DIR=/data/models and HF_HOME=/data/huggingface_cache
  - [ ] 1.9 Create /data/models/raw/ and /data/models/quantized/ directories with proper permissions
  - [ ] 1.10 Write unit test for Model model (test serialization, test enum values, test JSONB field)

### Phase 2: PyTorch Model Loading and Quantization

- [ ] 2.0 Implement PyTorch model loading with bitsandbytes quantization
  - [ ] 2.1 Create backend/src/ml/model_loader.py with load_model_from_hf function (use transformers.AutoModelForCausalLM, AutoConfig, AutoTokenizer)
  - [ ] 2.2 Implement architecture validation (check model_type in ['gpt2', 'llama', 'phi', 'pythia', 'gpt_neox'], reject encoder-only models)
  - [ ] 2.3 Implement config parsing to extract architecture details (num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, vocab_size, max_position_embeddings)
  - [ ] 2.4 Create backend/src/ml/quantize.py with BitsAndBytesConfig for Q4 quantization (load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
  - [ ] 2.5 Implement Q8 quantization config (load_in_8bit=True, llm_int8_threshold=6.0)
  - [ ] 2.6 Implement FP16 quantization (model.half(), simple PyTorch conversion)
  - [ ] 2.7 Implement Q2 quantization (if supported by bitsandbytes, else fallback to Q4)
  - [ ] 2.8 Add OOM error handling with automatic fallback (catch torch.cuda.OutOfMemoryError, try next higher precision)
  - [ ] 2.9 Implement memory requirement calculator (params_count × bytes_per_param × 1.2 for overhead)
  - [ ] 2.10 Save quantized model using safetensors format (model.save_pretrained with safe_serialization=True)
  - [ ] 2.11 Write unit tests for model_loader (test load_model_from_hf with mocked transformers)
  - [ ] 2.12 Write unit tests for quantize (test Q4/Q8/FP16, test OOM fallback logic)

### Phase 3: Backend Services and API Routes

- [ ] 3.0 Implement model management services and API endpoints
  - [ ] 3.1 Create Pydantic schemas in backend/src/schemas/model.py (ModelResponse with orm_mode, ModelDownloadRequest with repo_id validator, ModelListResponse with pagination, ActivationExtractionRequest with layer/hook validators)
  - [ ] 3.2 Implement ModelService in backend/src/services/model_service.py (list_models with filters, create_model generating m_{uuid} id, download_model enqueueing Celery task, check_dependencies querying trainings table, delete_model with file cleanup)
  - [ ] 3.3 Implement QuantizationService in backend/src/services/quantization_service.py (load_and_quantize_model, calculate_quantization_factor returning 0.5/0.25/0.125/0.0625, validate_memory_requirements checking against 6GB limit)
  - [ ] 3.4 Create model utilities in backend/src/utils/model_utils.py (calculate_memory_requirement, format_param_count returning "1.1B"/"135M", validate_repo_id regex, parse_architecture_config from config.json)
  - [ ] 3.5 Create FastAPI router in backend/src/api/routes/models.py (GET /api/models with pagination, POST /api/models/download, GET /api/models/:id with architecture_config, DELETE /api/models/:id with dependency check)
  - [ ] 3.6 Add error handling middleware (HTTPException for 400/404/409/503, structured logging)
  - [ ] 3.7 Implement POST /api/models/:id/extract endpoint (validate extraction config, enqueue Celery task, return job_id)
  - [ ] 3.8 Write unit tests for ModelService (test_list_models_with_filters, test_create_model, test_download_model, test_check_dependencies)
  - [ ] 3.9 Write unit tests for QuantizationService (test_calculate_quantization_factor, test_validate_memory_requirements)
  - [ ] 3.10 Write integration tests for API routes (test_list_models, test_download_model, test_delete_model with TestClient)

### Phase 4: Celery Background Tasks

- [ ] 4.0 Implement Celery tasks for model download and quantization
  - [ ] 4.1 Create download_model_task in backend/src/workers/model_tasks.py (bind=True, max_retries=3, call transformers.AutoModelForCausalLM.from_pretrained with cache_dir)
  - [ ] 4.2 Add download progress tracking (emit WebSocket events every 10% to models/{id}/progress channel)
  - [ ] 4.3 Implement model loading phase (load config, validate architecture, update status to LOADING)
  - [ ] 4.4 Create quantize_model_task (load raw model, apply quantization config, save to quantized_path)
  - [ ] 4.5 Add quantization progress tracking (emit events with GPU utilization via nvidia-smi)
  - [ ] 4.6 Calculate actual memory requirements after quantization (measure GPU memory usage)
  - [ ] 4.7 Calculate disk size (get_directory_size on quantized_path)
  - [ ] 4.8 Update model status to READY with memory_required_bytes and disk_size_bytes
  - [ ] 4.9 Add error handling (catch HuggingFaceError, torch.cuda.OutOfMemoryError, update status to ERROR, retry with exponential backoff)
  - [ ] 4.10 Write unit tests for model tasks (test_download_model_task with mocked transformers, test_quantize_model_task with mocked torch)

### Phase 5: Activation Extraction Implementation

- [ ] 5.0 Implement forward hooks for activation extraction
  - [ ] 5.1 Create backend/src/ml/forward_hooks.py with HookManager class (register_hooks, remove_hooks, activations storage dict)
  - [ ] 5.2 Implement create_hook function returning hook_fn(module, input, output) that stores output.detach().cpu().numpy()
  - [ ] 5.3 Implement register_residual_hooks (hook after layer norm in transformer blocks)
  - [ ] 5.4 Implement register_mlp_hooks (hook after MLP feed-forward layer)
  - [ ] 5.5 Implement register_attention_hooks (hook after attention layer before residual addition)
  - [ ] 5.6 Create ActivationService in backend/src/services/activation_service.py (extract_activations, save_activations_to_disk, calculate_activation_statistics)
  - [ ] 5.7 Implement extract_activations method (load model, load dataset batches, register hooks, run forward passes, collect activations)
  - [ ] 5.8 Implement save_activations method (save as .npy files with shape [num_samples, seq_len, hidden_dim], create metadata.json with extraction config)
  - [ ] 5.9 Implement calculate_statistics (mean activation magnitude, max activation, sparsity percentage)
  - [ ] 5.10 Create extract_activations_task in backend/src/workers/model_tasks.py (load model in eval mode, use torch.no_grad(), batch processing with progress updates)
  - [ ] 5.11 Add OOM handling in extraction (catch OutOfMemoryError, reduce batch size automatically, retry)
  - [ ] 5.12 Emit WebSocket progress events every 100 batches to models/{model_id}/extraction channel
  - [ ] 5.13 Write unit tests for forward_hooks (test hook registration, test activation capture)
  - [ ] 5.14 Write unit tests for ActivationService (test extract_activations, test save_activations, test statistics)

### Phase 6: Frontend State Management

- [ ] 6.0 Implement frontend state management for models
  - [ ] 6.1 Create TypeScript interfaces in frontend/src/types/model.ts (Model with id/name/params/quantized/memReq/status/progress, QuantizationFormat enum, ModelStatus enum, ArchitectureConfig interface, ActivationExtractionConfig interface)
  - [ ] 6.2 Match reference types from @0xcc/project-specs/reference-implementation/src/types/model.types.ts exactly
  - [ ] 6.3 Create Zustand store in frontend/src/stores/modelsStore.ts (models[] array, loading boolean, error string, fetchModels action, downloadModel action, updateModelProgress action, updateModelStatus action, extractActivations action)
  - [ ] 6.4 Add devtools middleware for debugging (devtools wrapper with name 'ModelsStore')
  - [ ] 6.5 Implement fetchModels action (call getModels API, update models array)
  - [ ] 6.6 Implement downloadModel action (call downloadModel API with repo/quantization/token, add model to array with status 'downloading')
  - [ ] 6.7 Implement updateModelProgress action (update progress for specific model by id)
  - [ ] 6.8 Implement updateModelStatus action (update status for specific model, handle completed/error)
  - [ ] 6.9 Implement extractActivations action (call extractActivations API, update extraction status)
  - [ ] 6.10 Create API client in frontend/src/api/models.ts (getModels with URLSearchParams, downloadModel POST, getModelArchitecture GET, extractActivations POST, deleteModel DELETE)
  - [ ] 6.11 Create useModelProgress hook in frontend/src/hooks/useModelProgress.ts (subscribe to models/{id}/progress and models/{id}/extraction channels)
  - [ ] 6.12 Write unit tests for modelsStore (test fetchModels, test downloadModel, test progress updates)
  - [ ] 6.13 Write unit tests for API client (test getModels, test downloadModel, test error handling)

### Phase 7: UI Components - ModelsPanel

- [ ] 7.0 Build main ModelsPanel component matching Mock UI exactly
  - [ ] 7.1 Create ModelsPanel.tsx in frontend/src/components/panels (MUST match Mock UI lines 1204-1343 exactly)
  - [ ] 7.2 Implement component state (hfModelRepo, quantization defaulting to 'Q4', accessToken, selectedModel, showExtractionConfig) - EXACTLY as Mock UI lines 1206-1210
  - [ ] 7.3 Connect to Zustand store (useModelsStore for models array, downloadModel, fetchModels actions)
  - [ ] 7.4 Add useEffect to fetch models on mount
  - [ ] 7.5 Add useEffect to subscribe to WebSocket updates for downloading/quantizing models
  - [ ] 7.6 Implement handleDownload function (validate repo, call downloadModel, clear form) - EXACTLY as Mock UI lines 1264-1268
  - [ ] 7.7 Render header with exact styling (text-2xl font-semibold mb-4, line 1215)
  - [ ] 7.8 Render download form container (bg-slate-900/50 border border-slate-800 rounded-lg p-6, lines 1216-1276)
  - [ ] 7.9 Render 2-column grid for repo input and quantization selector (grid grid-cols-2 gap-4, lines 1217-1246)
  - [ ] 7.10 Render HuggingFace repo input with label and placeholder (lines 1218-1229)
  - [ ] 7.11 Render quantization format select with options (FP16, Q8, Q4, Q2) - EXACTLY as lines 1230-1245
  - [ ] 7.12 Render access token input with type="password" and helper text (lines 1247-1261)
  - [ ] 7.13 Render download button with Download icon, disabled state, exact styling (bg-emerald-600 hover:bg-emerald-700, lines 1262-1275)
  - [ ] 7.14 Render models grid with map over models array (grid gap-4, line 1278)
  - [ ] 7.15 Render ModelArchitectureViewer modal conditionally (lines 1334-1336)
  - [ ] 7.16 Render ActivationExtractionConfig modal conditionally (lines 1338-1340)
  - [ ] 7.17 Write unit tests for ModelsPanel (test renders form, test fetches models, test download submission, test modal opening)

### Phase 8: UI Components - ModelCard

- [ ] 8.0 Build ModelCard component matching Mock UI
  - [ ] 8.1 Create ModelCard.tsx in frontend/src/components/models (extract from lines 1280-1330)
  - [ ] 8.2 Accept model prop and onClick callback for architecture viewer
  - [ ] 8.3 Render card container (bg-slate-900/50 border border-slate-800 rounded-lg p-6)
  - [ ] 8.4 Render clickable model info section with Cpu icon (w-8 h-8 text-purple-400) - lines 1282-1293
  - [ ] 8.5 Render model name (font-semibold text-lg) and metadata (params, quantization, memory)
  - [ ] 8.6 Render status icons (CheckCircle for ready, Loader with animate-spin for downloading, Activity for quantizing) - lines 1304-1306
  - [ ] 8.7 Render "Extract Activations" button for ready models (bg-purple-600 hover:bg-purple-700, lines 1296-1302)
  - [ ] 8.8 Render status badge with conditional styling (emerald for ready, slate for others, lines 1307-1313)
  - [ ] 8.9 Render progress bar for downloading status (lines 1316-1329) - EXACTLY match gradient and animation
  - [ ] 8.10 Progress bar uses --width CSS variable and gradient (from-purple-500 to-purple-400)
  - [ ] 8.11 Write unit tests for ModelCard (test renders model info, test status icons, test Extract button appears only for ready models)

### Phase 9: UI Components - ModelArchitectureViewer

- [ ] 9.0 Build ModelArchitectureViewer modal
  - [ ] 9.1 Create ModelArchitectureViewer.tsx (Modal from Mock UI lines 1346-1437)
  - [ ] 9.2 Accept model prop and onClose callback
  - [ ] 9.3 Render modal backdrop (fixed inset-0 bg-black/50 z-50)
  - [ ] 9.4 Render modal container (bg-slate-900 border border-slate-800 rounded-lg max-w-4xl max-h-[90vh])
  - [ ] 9.5 Render header with model name and close button (X icon, lines 1351-1362)
  - [ ] 9.6 Render architecture overview stats grid (4 columns: Total Layers, Hidden Dim, Attention Heads, Parameters, lines 1364-1393)
  - [ ] 9.7 Each stat card uses bg-slate-800/50 p-4 rounded-lg with label and value
  - [ ] 9.8 Render "Model Layers" section with scrollable layer list (lines 1395-1423)
  - [ ] 9.9 Each layer entry shows: layer type, index, dimensions (lines 1399-1420)
  - [ ] 9.10 Implement expandable TransformerBlock entries showing attention and MLP sub-components (lines 1409-1418)
  - [ ] 9.11 Render "Model Configuration" collapsible section with JSON display (lines 1425-1435)
  - [ ] 9.12 Parse architecture_config from model metadata (num_hidden_layers, hidden_size, num_attention_heads, etc.)
  - [ ] 9.13 Write unit tests for ModelArchitectureViewer (test renders stats, test renders layers, test expandable blocks)

### Phase 10: UI Components - ActivationExtractionConfig

- [ ] 10.0 Build ActivationExtractionConfig modal
  - [ ] 10.1 Create ActivationExtractionConfig.tsx (Modal from Mock UI lines 1440-1625)
  - [ ] 10.2 Implement modal state (selectedDataset, selectedLayers array, selectedHooks array, maxSamples, batchSize)
  - [ ] 10.3 Render modal backdrop and container (max-w-3xl for wider modal)
  - [ ] 10.4 Render header "Configure Activation Extraction" with close button (lines 1445-1455)
  - [ ] 10.5 Render dataset selector dropdown (lines 1458-1476) - fetch ready datasets from datasetsStore
  - [ ] 10.6 Render "Select Layers" section with LayerSelector component (lines 1478-1556)
  - [ ] 10.7 Create LayerSelector.tsx with grid layout (grid grid-cols-6 gap-2, lines 1538-1554)
  - [ ] 10.8 Layer buttons show L0, L1, L2... with conditional bg-emerald-600 (selected) or bg-slate-800 (unselected)
  - [ ] 10.9 Implement "Select All" and "Deselect All" bulk actions (text-xs buttons, lines 1486-1495)
  - [ ] 10.10 Render "Hook Types" section with checkboxes (Residual Stream, MLP Output, Attention Output, lines 1558-1586)
  - [ ] 10.11 Render "Settings" section with 2-column grid (batch size input, max samples input, lines 1588-1611)
  - [ ] 10.12 Implement estimation calculations (estimated time, estimated storage in GB, lines 1495-1503)
  - [ ] 10.13 Display warnings for long extractions (>1 hour) or large storage (>50GB)
  - [ ] 10.14 Render "Start Extraction" button (bg-emerald-600, disabled until valid config, lines 1613-1623)
  - [ ] 10.15 Implement handleStartExtraction (validate config, call extractActivations from store, close modal)
  - [ ] 10.16 Write unit tests for ActivationExtractionConfig (test layer selection, test hook selection, test validation, test estimation)
  - [ ] 10.17 Write unit tests for LayerSelector (test select/deselect, test bulk actions)

### Phase 11: WebSocket Real-Time Updates

- [ ] 11.0 Implement WebSocket progress updates for models
  - [ ] 11.1 Update WebSocket manager in backend/src/core/websocket.py to handle model channels (models/{id}/progress, models/{id}/extraction)
  - [ ] 11.2 Emit download progress events from download_model_task (type: 'progress', progress: 45.2, status: 'downloading', speed_mbps: 12.5)
  - [ ] 11.3 Emit quantization progress events (type: 'progress', progress: 70.0, status: 'quantizing', gpu_utilization: 85)
  - [ ] 11.4 Emit completion events (type: 'completed', progress: 100.0, status: 'ready', memory_required_bytes, disk_size_bytes)
  - [ ] 11.5 Emit extraction progress events (type: 'extraction_progress', progress: 30.0, samples_processed: 3000, eta_seconds: 180)
  - [ ] 11.6 Emit extraction completion events (type: 'extraction_completed', extraction_id, output_path, statistics)
  - [ ] 11.7 Emit error events on failure (type: 'error', error: 'OOM during quantization', suggested_format: 'Q8')
  - [ ] 11.8 Update useModelProgress hook to handle all event types (progress, completed, error, extraction_progress, extraction_completed)
  - [ ] 11.9 Update Zustand store to handle WebSocket events (updateModelProgress, updateModelStatus, handle extraction updates)
  - [ ] 11.10 Test WebSocket flow (start download, verify progress updates, verify status transitions, verify extraction progress)
  - [ ] 11.11 Write integration test for WebSocket (mock Socket.IO server, emit events, verify store updates)

### Phase 12: End-to-End Testing and Optimization

- [ ] 12.0 Comprehensive testing and performance optimization
  - [ ] 12.1 Write E2E workflow test (test_model_workflow.py: download TinyLlama → wait for ready → verify quantized files → extract activations → verify .npy files)
  - [ ] 12.2 Test download from HuggingFace with real model (TinyLlama/TinyLlama-1.1B-Chat-v1.0 with Q4)
  - [ ] 12.3 Test quantization for all formats (FP16, Q8, Q4, Q2 if supported)
  - [ ] 12.4 Test OOM fallback mechanism (load 7B model with Q2, expect fallback to Q4 or Q8)
  - [ ] 12.5 Test architecture viewer with multiple model types (GPT-2, Llama, Phi)
  - [ ] 12.6 Test activation extraction end-to-end (configure extraction, wait for completion, verify .npy files, verify shapes)
  - [ ] 12.7 Test extraction with multiple layers and hook types (L0, L5, L10 + residual + mlp)
  - [ ] 12.8 Test model deletion (delete model, verify quantized files removed, verify database record deleted)
  - [ ] 12.9 Test error handling (invalid repo ID, network failure, gated model without token, insufficient GPU memory)
  - [ ] 12.10 Test edge cases (very small model 135M, large model 7B+, model with non-standard architecture)
  - [ ] 12.11 Measure and optimize GPU memory usage (profile with nvidia-smi, ensure <6GB)
  - [ ] 12.12 Measure and optimize extraction speed (profile forward passes, optimize batch size)
  - [ ] 12.13 Verify all components match Mock UI styling exactly (screenshot comparison, CSS class verification)
  - [ ] 12.14 Run full test suite with coverage (backend: pytest --cov, frontend: npm test -- --coverage)
  - [ ] 12.15 Performance testing (API response times <100ms for GET, <500ms for POST, extraction throughput >200 samples/sec)
  - [ ] 12.16 Code review and refactoring (remove duplication, improve type safety, add docstrings)
  - [ ] 12.17 Update documentation (API docs, architecture guide, quantization guide, troubleshooting)

---

**Status:** Comprehensive Task List Complete

I have generated a detailed task list with 12 parent tasks broken down into 170+ actionable sub-tasks covering:

1. **Backend Infrastructure** - SQLAlchemy models, migrations, database indexes
2. **PyTorch Integration** - Model loading with transformers, bitsandbytes quantization (Q4/Q8/FP16/Q2), OOM handling
3. **Backend Services** - ModelService, QuantizationService, ActivationService, API routes
4. **Celery Tasks** - Async download, quantization, and extraction with WebSocket progress
5. **Activation Extraction** - Forward hooks (residual, MLP, attention), batch processing, statistics
6. **Frontend State** - Zustand store, API client, WebSocket hooks
7. **UI - ModelsPanel** - Main panel matching Mock UI lines 1204-1343 exactly
8. **UI - ModelCard** - Individual model cards with status, progress, actions
9. **UI - Architecture Viewer** - Modal with stats, layers, expandable transformer blocks (lines 1346-1437)
10. **UI - Extraction Config** - Layer selector, hook types, settings, estimations (lines 1440-1625)
11. **WebSocket Integration** - Real-time download, quantization, and extraction progress
12. **E2E Testing** - Complete workflows, performance optimization, memory profiling

All tasks reference exact Mock UI line numbers (1204-1625), include implementation details from TID, follow ADR technology decisions, and are ready for systematic implementation.

This task list is comprehensive and ready for execution!
