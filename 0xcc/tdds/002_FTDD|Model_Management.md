# Technical Design Document: Model Management

**Document ID:** 002_FTDD|Model_Management
**Feature:** Language Model Loading, Configuration, and Activation Extraction
**PRD Reference:** 002_FPRD|Model_Management.md
**ADR Reference:** 000_PADR|miStudio.md
**Status:** Draft
**Created:** 2025-10-06
**Last Updated:** 2025-10-06
**Owner:** miStudio Development Team

---

## 1. Executive Summary

This Technical Design Document defines the architecture and implementation approach for the Model Management feature of miStudio. The feature enables users to load, configure, and extract activations from various language models optimized for edge deployment on NVIDIA Jetson Orin Nano.

**Business Objective:** Provide researchers with efficient, memory-optimized access to language model activations for mechanistic interpretability research.

**Technical Approach:**
- FastAPI backend with Celery for async model loading and processing
- PyTorch 2.0+ with bitsandbytes quantization for memory efficiency
- PostgreSQL for model metadata and status tracking
- WebSocket for real-time loading progress updates
- React frontend with model configuration UI matching Mock-embedded-interp-ui.tsx design

**Key Design Decisions:**
1. Use INT4/INT8 quantization to fit larger models in 6GB GPU memory
2. Implement activation caching to avoid redundant forward passes
3. Support multiple models loaded simultaneously (up to 2 on edge device)
4. Extract layer-wise activations via forward hooks without model architecture modifications

**Success Metrics:**
- Model loading completes within 60 seconds for 7B parameter models
- Activation extraction processes 1000 samples within 5 minutes
- GPU memory usage stays under 6GB threshold
- Support models up to 13B parameters with INT4 quantization

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  ModelsPanel │  │ ModelDetails │  │ ActivationExtraction │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘ │
│         │                  │                      │              │
│         └──────────────────┴──────────────────────┘              │
│                            │                                     │
│                  REST API + WebSocket                            │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                   Backend (FastAPI)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              API Routes (/api/models)                     │   │
│  │  POST /models (create) │ GET /models (list)               │   │
│  │  GET /models/:id       │ POST /models/:id/load            │   │
│  │  POST /models/:id/extract-activations                     │   │
│  └─────────────┬────────────────────────────────────────────┘   │
│                │                                                 │
│  ┌─────────────┴────────────────────────────────────────────┐   │
│  │         Services Layer                                    │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐   │   │
│  │  │ ModelRegistry   │  │ ActivationExtractor          │   │   │
│  │  │ - Load models   │  │ - Register hooks             │   │   │
│  │  │ - Apply quant   │  │ - Extract activations        │   │   │
│  │  │ - Cache models  │  │ - Cache results              │   │   │
│  │  └────────┬────────┘  └──────────┬───────────────────┘   │   │
│  └───────────┼────────────────────────┼─────────────────────┘   │
│              │                        │                          │
│  ┌───────────┴────────────────────────┴─────────────────────┐   │
│  │         Celery Workers (Background Tasks)                 │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐   │   │
│  │  │ load_model_task │  │ extract_activations_task     │   │   │
│  │  └─────────────────┘  └──────────────────────────────┘   │   │
│  └───────────┬────────────────────────────────────────────┘   │
└──────────────┼───────────────────────────────────────────────┘
               │
┌──────────────┼───────────────────────────────────────────────┐
│         Data Layer                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  PostgreSQL     │  │  Redis       │  │  Filesystem     │  │
│  │  - models       │  │  - Celery    │  │  - Model cache  │  │
│  │  - activations  │  │  - Progress  │  │  - Activations  │  │
│  └─────────────────┘  └──────────────┘  └─────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Component Relationships

**Frontend Components:**
- **ModelsPanel:** Main model list, add model, filter UI (lines 813-1260 in Mock UI)
- **ModelCard:** Individual model card with status, actions (lines 1263-1435)
- **ModelDetailsModal:** Model configuration, activation extraction (lines 1441-1623)

**Backend Services:**
- **ModelRegistry:** Singleton service managing loaded models in GPU memory
- **ActivationExtractor:** Service for extracting and caching layer-wise activations
- **ModelLoader:** Celery task for async model loading with progress tracking

**Data Flow:**
1. User adds model → API creates DB record with status='pending'
2. User clicks "Load Model" → API triggers Celery task
3. Celery task loads model with quantization → updates status to 'ready'
4. User requests activation extraction → ActivationExtractor registers hooks
5. Forward pass extracts activations → saves to cache directory
6. Frontend displays extraction progress via WebSocket updates

### Integration Points

**With Dataset Management (001_FPRD):**
- Activation extraction requires tokenized dataset samples
- Uses dataset metadata for batch size configuration

**With SAE Training (003_FPRD):**
- Provides activation extraction capability for training
- Shares quantization configuration for memory consistency

**With Feature Discovery (004_FPRD):**
- Provides model for SAE inference and feature analysis
- Shares activation extraction utilities

**With Model Steering (005_FPRD):**
- Provides models with forward hook capabilities for interventions
- Enables real-time activation manipulation during generation

---

## 3. Technical Stack

### Core Technologies

**Backend:**
- **FastAPI 0.104+:** Async API framework for model endpoints
- **Celery 5.3+:** Distributed task queue for model loading
- **Redis 7.0+:** Message broker for Celery, progress tracking
- **PyTorch 2.0+:** ML framework with CUDA support
- **Transformers 4.35+:** Hugging Face library for model loading
- **bitsandbytes 0.41+:** INT4/INT8 quantization for memory efficiency
- **safetensors 0.4+:** Fast, safe model serialization format

**Frontend:**
- **React 18+:** UI framework
- **Zustand 4.4+:** State management
- **WebSocket:** Real-time progress updates
- **Lucide React:** Icon library

**Database:**
- **PostgreSQL 14+:** Model metadata, activation cache metadata
- **File System:** Model cache (`/data/models/`), activation cache (`/data/activations/`)

### Technology Justifications

**PyTorch over TensorFlow:**
- Better Hugging Face Transformers integration
- Superior bitsandbytes quantization support
- More flexible forward hook API for activation extraction

**bitsandbytes Quantization:**
- Reduces memory by 50-75% (INT8) to 87.5% (INT4)
- Enables 13B models on 6GB GPU (INT4) vs 7B only (FP16)
- Minimal accuracy degradation (<2% perplexity increase)

**Celery for Model Loading:**
- Non-blocking API responses (loading takes 30-60 seconds)
- Progress tracking via task status updates
- Automatic retry on transient errors (OOM, network issues)

**safetensors over pickle:**
- Faster load times (2-3x speedup)
- Memory-efficient loading (no redundant copies)
- Security: prevents arbitrary code execution

### Dependencies and Versions

```python
# requirements.txt additions for Model Management
torch==2.1.0+cu118
transformers==4.35.0
bitsandbytes==0.41.1
safetensors==0.4.0
accelerate==0.24.1  # For device_map="auto"
sentencepiece==0.1.99  # For LLaMA tokenizers
protobuf==3.20.3  # For tokenizer loading
```

**Version Requirements:**
- PyTorch 2.0+ required for bitsandbytes compatibility
- CUDA 11.8+ required for Jetson Orin Nano
- Transformers 4.35+ required for Mistral/LLaMA support

**Dependency Conflicts:**
- bitsandbytes requires specific PyTorch version (CUDA-enabled build)
- Accelerate required for `device_map="auto"` feature

---

## 4. Data Design

### Database Schema Expansion

#### models Table (from 003_SPEC|Postgres_Usecase_Details_and_Guidance.md)

```sql
CREATE TABLE models (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'm_gpt2_medium_abc123'

    -- Model identification
    name VARCHAR(500) NOT NULL,
    model_type VARCHAR(100) NOT NULL,  -- 'gpt2', 'llama', 'mistral'
    architecture VARCHAR(100),  -- 'CausalLM', 'EncoderDecoder'
    source VARCHAR(50) NOT NULL,  -- 'huggingface', 'local', 'custom'

    -- Model configuration
    config JSONB NOT NULL,  -- Complete model config (hidden_size, num_layers, etc.)
    quantization VARCHAR(50),  -- 'none', 'int8', 'int4', 'fp16'

    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
        -- Values: 'pending', 'loading', 'ready', 'failed', 'unloaded'
    progress FLOAT DEFAULT 0,  -- Loading progress 0-100

    -- Resource tracking
    memory_usage_bytes BIGINT,  -- GPU memory consumed
    load_time_seconds FLOAT,

    -- Model metadata
    parameter_count BIGINT,  -- Total parameters (7B, 13B, etc.)
    hidden_size INTEGER,  -- Extracted from config
    num_layers INTEGER,  -- Extracted from config
    vocab_size INTEGER,  -- Extracted from config

    -- Cache paths
    cache_dir VARCHAR(1000),  -- Local cache directory
    model_path VARCHAR(1000),  -- Path to model files

    -- Error tracking
    error_message TEXT,
    error_code VARCHAR(100),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    loaded_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT models_status_check CHECK (status IN
        ('pending', 'loading', 'ready', 'failed', 'unloaded'))
);

CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_model_type ON models(model_type);
CREATE INDEX idx_models_source ON models(source);
CREATE INDEX idx_models_created_at ON models(created_at DESC);
CREATE INDEX idx_models_last_used_at ON models(last_used_at DESC);

-- Composite index for filtering ready models
CREATE INDEX idx_models_ready ON models(status, model_type) WHERE status = 'ready';
```

**Config JSONB Structure:**
```json
{
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "vocab_size": 50257,
  "max_position_embeddings": 1024,
  "layer_norm_epsilon": 1e-5,
  "model_type": "gpt2",
  "architectures": ["GPT2LMHeadModel"]
}
```

#### activation_cache Table

```sql
CREATE TABLE activation_cache (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'act_cache_abc123'
    model_id VARCHAR(255) NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id VARCHAR(255) REFERENCES datasets(id) ON DELETE SET NULL,

    -- Extraction configuration
    layer_indices INTEGER[],  -- Which layers extracted [0, 6, 11]
    sample_count INTEGER NOT NULL,

    -- Cache location
    storage_path VARCHAR(1000) NOT NULL,  -- /data/activations/{id}/
    file_size_bytes BIGINT,

    -- Metadata
    extraction_time_seconds FLOAT,
    activation_shape INTEGER[],  -- [num_samples, seq_len, hidden_size]

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP WITH TIME ZONE,

    -- Composite unique constraint (one cache per model+dataset+layers)
    CONSTRAINT activation_cache_unique UNIQUE (model_id, dataset_id, layer_indices)
);

CREATE INDEX idx_activation_cache_model ON activation_cache(model_id);
CREATE INDEX idx_activation_cache_dataset ON activation_cache(dataset_id);
CREATE INDEX idx_activation_cache_created_at ON activation_cache(created_at DESC);
```

**Storage Estimate:**
- Model metadata: ~2KB per model
- Activation cache metadata: ~1KB per cache entry
- Activation cache files: 1000 samples × 1024 seq_len × 768 hidden_dim × 4 bytes = 3GB per cache

### Data Validation Strategy

**Model Configuration Validation:**
- Validate `model_type` against supported types: gpt2, llama, mistral
- Ensure `quantization` is one of: none, int8, int4, fp16
- Verify `config` JSONB contains required fields: hidden_size, num_layers
- Check `parameter_count` is within edge device capacity (<13B for INT4)

**Activation Cache Validation:**
- Validate layer_indices are within model's num_layers range
- Ensure storage_path exists and has write permissions
- Verify file_size_bytes matches actual file size on disk
- Check activation_shape matches expected dimensions

**Data Consistency:**
- Use PostgreSQL CHECK constraints for status values
- Implement database-level foreign key constraints (CASCADE deletes)
- Use UNIQUE constraints to prevent duplicate caches
- Implement optimistic locking (updated_at timestamp checks) for concurrent modifications

### Migration Strategy

**Initial Migration (Create Tables):**
```sql
-- Migration 002: Create models and activation_cache tables
BEGIN;

CREATE TABLE models (...);
CREATE TABLE activation_cache (...);

-- Create indexes
CREATE INDEX idx_models_status ON models(status);
-- ... additional indexes

COMMIT;
```

**Data Preservation:**
- No existing data (new feature)
- Future migrations will use Alembic (FastAPI standard)
- Implement rollback strategy for each migration

---

## 5. API Design

### RESTful API Conventions

**Base Path:** `/api/models`

**HTTP Method Semantics:**
- GET: Retrieve model(s) - idempotent, cacheable
- POST: Create new model or trigger actions (load, extract)
- PATCH: Update model configuration
- DELETE: Remove model and cleanup cache

**Status Codes:**
- 200 OK: Successful GET/PATCH
- 201 Created: Successful POST (create)
- 204 No Content: Successful DELETE
- 400 Bad Request: Validation error
- 404 Not Found: Model not found
- 409 Conflict: Status conflict (e.g., already loading)
- 500 Internal Server Error: Unexpected error

**Request/Response Format:**
- Content-Type: application/json
- UTF-8 encoding
- ISO 8601 timestamps (YYYY-MM-DDTHH:MM:SSZ)
- Snake_case field names (following Python conventions)

### API Endpoint Specifications

#### POST /api/models
**Purpose:** Create new model record and optionally trigger loading

**Request Body:**
```json
{
  "name": "GPT-2 Medium",
  "model_type": "gpt2",
  "source": "huggingface",
  "model_identifier": "gpt2-medium",
  "quantization": "int8",
  "auto_load": true
}
```

**Response:** 201 Created
```json
{
  "id": "m_gpt2_medium_abc123",
  "name": "GPT-2 Medium",
  "model_type": "gpt2",
  "source": "huggingface",
  "quantization": "int8",
  "status": "pending",
  "progress": 0,
  "config": {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "vocab_size": 50257
  },
  "parameter_count": 355000000,
  "created_at": "2025-10-06T10:00:00Z"
}
```

**Validation:**
- `model_type` must be supported (gpt2, llama, mistral)
- `quantization` must be valid (none, int8, int4, fp16)
- `model_identifier` must be valid Hugging Face model ID or local path

#### POST /api/models/:id/load
**Purpose:** Load model into GPU memory with quantization

**Request Body:**
```json
{
  "quantization": "int8"
}
```

**Response:** 200 OK
```json
{
  "id": "m_gpt2_medium_abc123",
  "status": "loading",
  "task_id": "celery_task_xyz789",
  "message": "Model loading initiated"
}
```

**Status Transition:** pending → loading → ready (or failed)

**WebSocket Events:** Emits `model:loading` with progress updates

#### POST /api/models/:id/extract-activations
**Purpose:** Extract layer-wise activations for dataset samples

**Request Body:**
```json
{
  "dataset_id": "ds_pile_abc123",
  "layer_indices": [0, 6, 11],
  "sample_count": 1000,
  "batch_size": 32
}
```

**Response:** 202 Accepted
```json
{
  "extraction_id": "ext_act_abc123",
  "status": "queued",
  "estimated_time_seconds": 300,
  "message": "Activation extraction queued"
}
```

**Query Endpoint:** GET /api/models/:id/activations/:extraction_id
```json
{
  "extraction_id": "ext_act_abc123",
  "status": "completed",
  "progress": 100,
  "cache_id": "act_cache_xyz789",
  "storage_path": "/data/activations/act_cache_xyz789/",
  "file_size_bytes": 3145728000,
  "extraction_time_seconds": 285.6
}
```

#### GET /api/models
**Purpose:** List all models with filtering

**Query Parameters:**
- `status` (optional): Filter by status (e.g., "ready")
- `model_type` (optional): Filter by model type
- `limit` (default 50): Results per page
- `offset` (default 0): Pagination offset

**Response:** 200 OK
```json
{
  "models": [
    {
      "id": "m_gpt2_medium_abc123",
      "name": "GPT-2 Medium",
      "model_type": "gpt2",
      "status": "ready",
      "quantization": "int8",
      "parameter_count": 355000000,
      "memory_usage_bytes": 400000000,
      "loaded_at": "2025-10-06T10:05:00Z"
    }
  ],
  "total": 5,
  "limit": 50,
  "offset": 0
}
```

### Error Handling Strategy

**Error Response Format:**
```json
{
  "error": {
    "code": "MODEL_LOAD_FAILED",
    "message": "Failed to load model: insufficient GPU memory",
    "details": {
      "required_memory_gb": 8.5,
      "available_memory_gb": 6.0
    },
    "timestamp": "2025-10-06T10:15:00Z"
  }
}
```

**Error Code Categories:**
- `VALIDATION_ERROR`: Invalid input (400)
- `MODEL_NOT_FOUND`: Model ID doesn't exist (404)
- `MODEL_LOAD_FAILED`: Loading error (500)
- `INSUFFICIENT_MEMORY`: OOM error (500)
- `EXTRACTION_FAILED`: Activation extraction error (500)

**Retry Strategy:**
- Transient errors (OOM, network): Auto-retry 3 times with exponential backoff
- Permanent errors (invalid model): Fail immediately, no retry

### Security Considerations

**Authentication:**
- All endpoints require valid JWT token (from auth middleware)
- Token extracted from `Authorization: Bearer <token>` header

**Authorization:**
- Model CRUD operations: Require "model:write" permission
- Model viewing: Require "model:read" permission

**Input Sanitization:**
- Validate model_identifier against regex: `^[\w\-\/]+$` (prevent path traversal)
- Sanitize file paths before filesystem access
- Validate layer_indices are integers within valid range

**Rate Limiting:**
- Model loading: Max 10 requests per hour per user
- Activation extraction: Max 5 simultaneous extractions per system

### Performance Optimization

**Caching:**
- Model metadata: Cache in Redis for 5 minutes (reduce DB queries)
- Activation cache: Check filesystem cache before extraction

**Pagination:**
- Default limit: 50 models per page
- Max limit: 500 (prevent large response sizes)

**Database Query Optimization:**
- Use indexes on status, model_type for filtering
- Select only required fields (avoid SELECT *)
- Use connection pooling (max 20 connections)

---

## 6. Component Architecture

### Frontend Component Hierarchy

```
ModelsPanel
├── AddModelModal (dialog for adding new models)
│   ├── ModelSourceSelector (HuggingFace, Local, Custom)
│   ├── ModelTypeDropdown (GPT-2, LLaMA, Mistral)
│   ├── QuantizationSelector (None, INT8, INT4, FP16)
│   └── SubmitButton
├── ModelsList (grid of ModelCard components)
│   └── ModelCard (individual model with actions)
│       ├── ModelHeader (name, type, status badge)
│       ├── ModelStats (parameters, memory, layers)
│       ├── ProgressBar (loading progress)
│       └── ModelActions (Load, Configure, Delete)
└── ModelDetailsModal (detailed view for configuration)
    ├── ConfigurationPanel (quantization, cache settings)
    ├── ActivationExtractionPanel (layer selection, dataset)
    ├── ModelInfoPanel (metadata, config JSON)
    └── ActionButtons (Extract Activations, Unload)
```

**Component Responsibilities:**

**ModelsPanel (Lines 813-1260):**
- Fetch and display model list
- Handle add model dialog
- Filter models by status
- Subscribe to WebSocket updates for loading progress

**ModelCard (Lines 1263-1435):**
- Display individual model summary
- Show loading progress bar
- Handle action buttons (Load, Configure, Delete)
- Status indicator (color-coded badge)

**ModelDetailsModal (Lines 1441-1623):**
- Display full model configuration
- Configure activation extraction parameters
- Trigger extraction jobs
- Show extraction progress

### Backend Service Architecture

```
┌─────────────────────────────────────────────────────┐
│              ModelRegistry (Singleton)               │
│  - loaded_models: Dict[str, LoadedModel]            │
│  - load_model(model_id, quantization)               │
│  - unload_model(model_id)                           │
│  - get_model(model_id) -> LoadedModel               │
│  - check_memory_available(required_gb) -> bool      │
└─────────────────────────────────────────────────────┘
                          │
                          │ uses
                          ↓
┌─────────────────────────────────────────────────────┐
│          ActivationExtractor (Service)               │
│  - extract_activations(model, dataset, layers)      │
│  - register_hooks(model, layer_indices)             │
│  - save_activations_to_cache(activations, cache_id) │
│  - load_activations_from_cache(cache_id)            │
└─────────────────────────────────────────────────────┘
                          │
                          │ called by
                          ↓
┌─────────────────────────────────────────────────────┐
│           Celery Tasks (Background Workers)          │
│  - load_model_task(model_id, config)                │
│  - extract_activations_task(model_id, extraction_config) │
│  - cleanup_model_cache_task(model_id)               │
└─────────────────────────────────────────────────────┘
```

**ModelRegistry Service:**
- **Pattern:** Singleton (one instance per worker process)
- **Responsibility:** Manage loaded models in GPU memory
- **State Management:** In-memory dictionary of model_id → LoadedModel
- **Thread Safety:** Use threading.Lock for concurrent access
- **Memory Management:** Track total GPU memory usage, enforce limits

**ActivationExtractor Service:**
- **Pattern:** Stateless service (can be instantiated per request)
- **Responsibility:** Extract activations using forward hooks
- **Hook Management:** Register/unregister hooks on model layers
- **Caching:** Save/load activations from filesystem
- **Optimization:** Batch processing to maximize GPU utilization

**Celery Tasks:**
- **Pattern:** Async tasks with progress tracking
- **Responsibility:** Long-running operations (loading, extraction)
- **Progress Updates:** Emit WebSocket events every 5% progress
- **Error Handling:** Catch exceptions, update model status, log errors

### State Management (Frontend)

**Zustand Store Structure:**
```typescript
interface ModelStore {
  models: Model[];
  selectedModel: Model | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchModels: () => Promise<void>;
  addModel: (config: AddModelConfig) => Promise<Model>;
  loadModel: (modelId: string) => Promise<void>;
  selectModel: (model: Model) => void;
  updateModelStatus: (modelId: string, status: ModelStatus) => void;

  // WebSocket subscription
  subscribeToModelUpdates: () => void;
  unsubscribeFromModelUpdates: () => void;
}
```

**State Update Flow:**
1. User action (e.g., click "Load Model") → dispatch action
2. Action calls API endpoint → returns task_id
3. WebSocket receives progress updates → calls `updateModelStatus`
4. Component re-renders with updated status/progress
5. Completion triggers final state update (status='ready')

**State Persistence:**
- selectedModel: Session storage (persist across page refresh)
- models: No persistence (refetch on mount)

### Separation of Concerns

**Presentation Layer (React Components):**
- Only UI rendering and user interaction
- No business logic or API calls
- Use hooks for state access

**State Management Layer (Zustand):**
- API calls and data fetching
- State updates and synchronization
- WebSocket event handling

**API Layer (FastAPI):**
- Request validation and routing
- Business logic orchestration
- Database operations

**Service Layer (Python Services):**
- Model loading and management
- Activation extraction logic
- Caching and optimization

---

## 7. State Management

### Application State Organization

**Global State (Zustand Store):**
- `models[]`: Array of all model records
- `loadingModels`: Set of model IDs currently loading
- `extractionJobs`: Map of extraction_id → ExtractionStatus

**Component State (React useState):**
- Modal open/closed state
- Form input values
- Temporary UI state (hover, focus)

**Server State (React Query alternative: custom hooks):**
- Model list cache (5 minute TTL)
- Model details cache (refetch on focus)

### State Flow Patterns

**Model Loading Flow:**
```
1. User clicks "Load Model" button
   → Component calls modelStore.loadModel(modelId)

2. Store dispatches API call POST /api/models/:id/load
   → API returns task_id, updates DB status='loading'

3. WebSocket receives 'model:loading' event
   → Store calls updateModelStatus(modelId, {status: 'loading', progress: 10})

4. Progress updates continue (10%, 30%, 60%, 100%)
   → Each update triggers store update → component re-render

5. Loading completes, WebSocket receives 'model:ready' event
   → Store updates status='ready', removes from loadingModels set

6. Component shows success state (green checkmark, "Ready" badge)
```

**Activation Extraction Flow:**
```
1. User configures extraction (layers, dataset, samples)
   → Component calls modelStore.extractActivations(modelId, config)

2. Store dispatches API call POST /api/models/:id/extract-activations
   → API returns extraction_id, creates DB record

3. Store adds extraction to extractionJobs map
   → { extraction_id: { status: 'queued', progress: 0 } }

4. Celery worker starts extraction, emits progress events
   → WebSocket receives 'extraction:progress' events

5. Store updates extractionJobs[extraction_id].progress
   → Component progress bar updates

6. Extraction completes, WebSocket receives 'extraction:completed'
   → Store updates status='completed', cache_id populated

7. Component shows completion message, enables "View Activations" button
```

### Side Effects Handling

**WebSocket Connection:**
- Initialize on app mount: `modelStore.subscribeToModelUpdates()`
- Cleanup on unmount: `modelStore.unsubscribeFromModelUpdates()`
- Auto-reconnect on disconnect (exponential backoff)

**Polling Fallback:**
- If WebSocket fails, fallback to polling every 5 seconds
- Poll only for models with status='loading' or extractions in progress
- Stop polling when status changes to terminal state

**Optimistic Updates:**
- When user clicks "Load Model", immediately update status='loading' in UI
- If API call fails, rollback to previous status
- Show error toast on failure

### Caching Strategy

**API Response Caching:**
- Model list: Cache for 5 minutes (stale-while-revalidate)
- Model details: Cache for 2 minutes
- Invalidate cache on mutations (add, load, delete)

**Activation Cache:**
- Check `activation_cache` table before extraction
- If cache exists and recent (<24 hours), reuse
- Display "Using cached activations" message to user

**Cache Invalidation:**
- Delete model → invalidate all related caches (activations, metadata)
- Update model config → invalidate model details cache
- Manual refresh → force refetch from API

---

## 8. Security Considerations

### Authentication & Authorization

**JWT-Based Authentication:**
- All API endpoints require valid JWT token
- Token extracted from `Authorization: Bearer <token>` header
- Token validated by FastAPI dependency middleware

**Permission Levels:**
- `model:read`: View models and configurations
- `model:write`: Add, load, configure, delete models
- `model:admin`: Manage quantization, cache cleanup

**Authorization Middleware:**
```python
def require_model_permission(permission: str):
    def dependency(token: str = Depends(get_token)):
        user = decode_jwt(token)
        if permission not in user.permissions:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return dependency

@router.post("/models/{model_id}/load")
async def load_model(
    model_id: str,
    user: User = Depends(require_model_permission("model:write"))
):
    # ... load model logic
```

### Data Validation & Sanitization

**Input Validation (Pydantic Schemas):**
```python
class AddModelRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    model_type: Literal["gpt2", "llama", "mistral"]
    source: Literal["huggingface", "local", "custom"]
    model_identifier: str = Field(..., regex="^[\\w\\-\\/]+$")
    quantization: Literal["none", "int8", "int4", "fp16"] = "none"
    auto_load: bool = False

    @validator("model_identifier")
    def validate_model_path(cls, v, values):
        if values["source"] == "local":
            # Prevent path traversal attacks
            if ".." in v or v.startswith("/"):
                raise ValueError("Invalid local model path")
        return v
```

**SQL Injection Prevention:**
- Use SQLAlchemy ORM (parameterized queries)
- Never concatenate user input into SQL strings

**Path Traversal Prevention:**
- Validate file paths against whitelist directories
- Use `os.path.abspath()` and check prefix matches allowed directories

### Security Best Practices

**Secret Management:**
- Store Hugging Face API tokens in environment variables
- Use secrets manager for production (AWS Secrets Manager, Vault)
- Never log sensitive tokens or credentials

**API Rate Limiting:**
- Limit model loading to 10 requests/hour per user
- Limit activation extraction to 5 concurrent jobs per system
- Return 429 Too Many Requests on rate limit exceeded

**Error Messages:**
- Don't expose internal paths or stack traces to frontend
- Log detailed errors server-side, return generic messages to client
- Example: "Failed to load model" instead of "FileNotFoundError: /internal/path/model.bin"

**CORS Configuration:**
- Restrict origins to frontend domain only
- Allow credentials for JWT cookies
- Limit allowed methods to required set (GET, POST, PATCH, DELETE)

### Privacy & Compliance

**Data Retention:**
- Model cache files deleted on model deletion
- Activation cache files deleted after 30 days of inactivity
- User activity logs retained for 90 days

**GDPR Considerations:**
- No personal data stored in model metadata
- Activation caches contain only model activations (no user data)
- User has right to delete their models and caches

---

## 9. Performance & Scalability

### Performance Optimization Principles

**Model Loading Optimization:**
1. **Quantization:** Use INT8 (50% memory reduction) or INT4 (87.5% reduction)
2. **Lazy Loading:** Load model layers incrementally (not all at once)
3. **Device Mapping:** Use `device_map="auto"` for optimal layer placement
4. **safetensors:** Use safetensors format for 2-3x faster loading

**Activation Extraction Optimization:**
1. **Batch Processing:** Process 32 samples at once (maximize GPU utilization)
2. **Forward Hook Efficiency:** Register hooks only on target layers
3. **Memory Pinning:** Use `pin_memory=True` for faster CPU-GPU transfer
4. **Caching:** Save extracted activations to disk, reuse on subsequent requests

**API Response Optimization:**
1. **Database Indexes:** Index on status, model_type, created_at
2. **Pagination:** Limit results to 50 per page (configurable)
3. **Field Selection:** Allow clients to request only needed fields
4. **Compression:** Gzip compress responses >1KB

### Caching Strategy

**Model Cache (Filesystem):**
- Location: `/data/models/{model_id}/`
- Format: safetensors files
- Retention: Persist until model deleted
- Size Limit: 50GB total cache (LRU eviction)

**Activation Cache (Filesystem):**
- Location: `/data/activations/{cache_id}/`
- Format: PyTorch tensor files (.pt)
- Retention: 30 days of inactivity
- Size Limit: 100GB total cache (LRU eviction)

**Redis Cache (Metadata):**
- Model list: Cache for 5 minutes
- Model details: Cache for 2 minutes
- Extraction status: Real-time (no cache)

**Cache Invalidation:**
- Model updated/deleted → invalidate model metadata cache
- Activation extraction completed → invalidate extraction status cache

### Database Query Optimization

**Index Strategy:**
```sql
-- Primary indexes (created in schema)
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_model_type ON models(model_type);

-- Composite index for common query pattern
CREATE INDEX idx_models_ready_type ON models(status, model_type)
    WHERE status = 'ready';

-- Partial index for active models
CREATE INDEX idx_models_active ON models(id, name, status)
    WHERE status IN ('loading', 'ready');
```

**Query Patterns:**
```python
# Efficient: Uses idx_models_ready_type
models = db.query(Model).filter(
    Model.status == "ready",
    Model.model_type == "gpt2"
).all()

# Inefficient: Full table scan
models = db.query(Model).filter(
    Model.config["hidden_size"].astext.cast(Integer) == 768
).all()
```

**Connection Pooling:**
- Min connections: 5
- Max connections: 20
- Connection timeout: 30 seconds
- Connection recycling: 3600 seconds (1 hour)

### Scalability Considerations

**Horizontal Scaling:**
- **API Servers:** Stateless FastAPI servers (can scale to N instances)
- **Celery Workers:** Multiple workers for parallel model loading (1 per GPU)
- **Database:** PostgreSQL with read replicas for scaling reads

**Vertical Scaling:**
- **GPU Memory:** Jetson Orin Nano limited to 6GB (use quantization for larger models)
- **CPU Cores:** 8 cores available (parallelize tokenization, preprocessing)
- **Storage:** NVMe SSD for fast model/activation cache access

**Resource Limits:**
- Max 2 simultaneous model loads (memory constrained)
- Max 5 simultaneous activation extractions (GPU constrained)
- Max 3 models loaded in GPU memory (memory constrained)

**Auto-Scaling Strategy:**
- Use queue length as scaling trigger (Celery queue size)
- If queue >10 tasks, add additional Celery worker (if GPU available)
- If GPU memory >90%, trigger cache eviction (unload LRU model)

---

## 10. Testing Strategy

### Testing Approach

**Unit Tests (70% coverage target):**
- Test individual functions and methods in isolation
- Mock external dependencies (DB, API, GPU)
- Focus on business logic, validation, error handling

**Integration Tests (20% coverage):**
- Test API endpoints end-to-end
- Use test database (PostgreSQL test instance)
- Mock slow operations (model loading, extraction)

**Performance Tests (10% coverage):**
- Benchmark model loading time (target: <60 seconds)
- Benchmark activation extraction throughput (target: 1000 samples in 5 minutes)
- Measure API response time (target: <200ms for list, <1s for details)

### Test Organization

**Directory Structure:**
```
tests/
├── unit/
│   ├── test_model_registry.py
│   ├── test_activation_extractor.py
│   └── test_validation.py
├── integration/
│   ├── test_api_models.py
│   ├── test_celery_tasks.py
│   └── test_websocket.py
├── performance/
│   ├── test_model_loading_speed.py
│   └── test_activation_extraction_speed.py
└── fixtures/
    ├── models.py (test model configurations)
    └── activations.py (sample activations)
```

### Testing Patterns

**Unit Test Example (Model Validation):**
```python
import pytest
from app.schemas import AddModelRequest
from pydantic import ValidationError

def test_add_model_request_valid():
    request = AddModelRequest(
        name="GPT-2 Medium",
        model_type="gpt2",
        source="huggingface",
        model_identifier="gpt2-medium",
        quantization="int8"
    )
    assert request.name == "GPT-2 Medium"
    assert request.quantization == "int8"

def test_add_model_request_invalid_type():
    with pytest.raises(ValidationError):
        AddModelRequest(
            name="Invalid Model",
            model_type="invalid_type",  # Should fail
            source="huggingface",
            model_identifier="gpt2"
        )

def test_add_model_request_path_traversal():
    with pytest.raises(ValidationError):
        AddModelRequest(
            name="Malicious Model",
            model_type="gpt2",
            source="local",
            model_identifier="../../../etc/passwd"  # Should fail
        )
```

**Integration Test Example (API Endpoint):**
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_model_success(auth_headers):
    response = client.post("/api/models", json={
        "name": "Test GPT-2",
        "model_type": "gpt2",
        "source": "huggingface",
        "model_identifier": "gpt2",
        "quantization": "int8"
    }, headers=auth_headers)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test GPT-2"
    assert data["status"] == "pending"
    assert "id" in data

def test_load_model_unauthorized():
    response = client.post("/api/models/m_test_123/load")
    assert response.status_code == 401
```

### Mock and Fixture Strategy

**Mocking GPU Operations:**
```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_gpu():
    with patch("torch.cuda.is_available", return_value=True), \\
         patch("torch.cuda.get_device_properties") as mock_props:
        mock_props.return_value.total_memory = 6 * 1024**3  # 6GB
        yield mock_props

def test_model_loading_with_mock_gpu(mock_gpu):
    # Test model loading without actual GPU
    registry = ModelRegistry()
    model_id = registry.load_model("gpt2", quantization="int8")
    assert model_id in registry.loaded_models
```

**Test Fixtures:**
```python
@pytest.fixture
def sample_model_config():
    return {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "vocab_size": 50257
    }

@pytest.fixture
def sample_model_record(db_session, sample_model_config):
    model = Model(
        id="m_test_gpt2_123",
        name="Test GPT-2",
        model_type="gpt2",
        config=sample_model_config,
        status="pending"
    )
    db_session.add(model)
    db_session.commit()
    return model
```

---

## 11. Deployment & DevOps

### Deployment Pipeline

**Deployment Stages:**
1. **Build Stage:** Docker image build with PyTorch + CUDA dependencies
2. **Test Stage:** Run unit and integration tests in CI
3. **Staging Deploy:** Deploy to staging environment (Jetson Orin Nano dev board)
4. **Manual Approval:** QA validation and smoke tests
5. **Production Deploy:** Deploy to production Jetson Orin Nano

**CI/CD Tool:** GitHub Actions

**Pipeline Configuration (.github/workflows/deploy.yml):**
```yaml
name: Deploy Model Management

on:
  push:
    branches: [main]
    paths:
      - 'backend/app/models/**'
      - 'backend/app/services/model_registry.py'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/unit tests/integration --cov

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t mistudio-backend:latest .
      - name: Push to registry
        run: docker push mistudio-backend:latest

  deploy-staging:
    needs: build
    runs-on: self-hosted  # Jetson Orin Nano runner
    steps:
      - name: Pull latest image
        run: docker pull mistudio-backend:latest
      - name: Deploy to staging
        run: docker-compose -f docker-compose.staging.yml up -d
```

### Environment Configuration

**Environment Variables:**
```bash
# .env.production
DATABASE_URL=postgresql://user:pass@localhost:5432/mistudio
REDIS_URL=redis://localhost:6379/0
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
GPU_MEMORY_LIMIT_GB=6
MODEL_CACHE_DIR=/data/models
ACTIVATION_CACHE_DIR=/data/activations
LOG_LEVEL=INFO
CORS_ORIGINS=https://mistudio.example.com
JWT_SECRET_KEY=xxxxxxxxxxxxx
```

**Configuration Management:**
- Development: `.env.development` (local settings)
- Staging: `.env.staging` (Jetson Orin Nano test board)
- Production: `.env.production` (Jetson Orin Nano production)

### Monitoring & Logging

**Logging Strategy:**
- **Application Logs:** JSON structured logs to stdout
- **Access Logs:** Uvicorn access logs (request/response)
- **Error Logs:** Sentry integration for error tracking
- **Performance Logs:** GPU memory, model loading time, extraction duration

**Log Levels:**
- DEBUG: Development only (detailed traces)
- INFO: Normal operations (model loaded, extraction completed)
- WARNING: Non-critical issues (cache miss, high memory usage)
- ERROR: Errors requiring attention (model load failed, OOM)

**Monitoring Metrics:**
- GPU memory usage (alert if >90%)
- Model loading success rate (target: >95%)
- API response time (p50, p95, p99)
- Celery queue length (alert if >20)

**Monitoring Tools:**
- **Prometheus:** Metrics collection (GPU, API latency)
- **Grafana:** Dashboard visualization
- **Alertmanager:** Alert routing (Slack, email)

**Alert Thresholds:**
- GPU memory >90%: Warning alert
- Model loading failure rate >5%: Error alert
- Celery queue length >20: Warning alert
- API p99 latency >5s: Warning alert

### Rollback Strategy

**Deployment Rollback:**
1. Detect failure (monitoring alerts, health check failures)
2. Stop new deployment (docker-compose down)
3. Revert to previous Docker image version (docker pull mistudio-backend:v1.2.3)
4. Restart services (docker-compose up -d)
5. Verify health checks pass

**Database Migration Rollback:**
```sql
-- Alembic rollback to previous version
alembic downgrade -1

-- Manual rollback if needed
BEGIN;
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS activation_cache;
COMMIT;
```

**Data Backup:**
- Daily PostgreSQL backups (pg_dump)
- Model cache snapshots (incremental backups)
- Retention: 7 days

---

## 12. Risk Assessment

### Technical Risks

**Risk 1: GPU Memory Exhaustion (High Likelihood, Critical Impact)**
- **Description:** Model loading or activation extraction exceeds 6GB GPU memory limit
- **Mitigation:**
  - Enforce memory limits in ModelRegistry (reject load if insufficient memory)
  - Use aggressive quantization (INT4 for large models)
  - Implement model unloading (LRU eviction)
  - Monitor GPU memory and trigger cleanup at 90% threshold
- **Contingency:** Fallback to CPU inference (slow but functional)

**Risk 2: Model Loading Failures (Medium Likelihood, High Impact)**
- **Description:** Network errors, corrupted downloads, unsupported models
- **Mitigation:**
  - Implement retry logic (3 attempts with exponential backoff)
  - Validate model files after download (checksum verification)
  - Maintain whitelist of tested/supported models
  - Cache successful downloads to avoid re-downloading
- **Contingency:** Manual download and local model loading

**Risk 3: Activation Cache Storage Exhaustion (Medium Likelihood, Medium Impact)**
- **Description:** Activation caches fill 100GB storage limit
- **Mitigation:**
  - Implement LRU cache eviction (delete oldest unused caches)
  - Set 30-day inactivity threshold for auto-deletion
  - Monitor storage usage, alert at 80% full
  - Provide manual cache cleanup UI
- **Contingency:** Recompute activations on demand (slower but functional)

**Risk 4: Slow Model Loading on Edge Device (High Likelihood, Medium Impact)**
- **Description:** Loading 13B model takes >2 minutes, poor UX
- **Mitigation:**
  - Use safetensors format (2-3x faster than pickle)
  - Implement incremental loading progress updates (better UX)
  - Pre-load frequently used models on system startup
  - Cache models on disk after first load
- **Contingency:** Display estimated time, allow background loading

### Dependencies & Blockers

**Dependency 1: Hugging Face API Availability**
- **Risk:** Hugging Face Hub downtime blocks model downloads
- **Mitigation:** Cache downloaded models, implement local fallback
- **Blocker Severity:** Medium (affects new model additions only)

**Dependency 2: bitsandbytes Compatibility**
- **Risk:** bitsandbytes version incompatible with PyTorch or CUDA version
- **Mitigation:** Pin versions in requirements.txt, test compatibility matrix
- **Blocker Severity:** High (affects all quantized models)

**Dependency 3: CUDA Driver Version**
- **Risk:** Jetson Orin Nano CUDA driver outdated, incompatible with PyTorch
- **Mitigation:** Document required CUDA version (11.8+), test on target hardware
- **Blocker Severity:** Critical (affects all GPU operations)

### Complexity Assessment

**Implementation Complexity: 7/10**
- High complexity: GPU memory management, quantization, forward hooks
- Medium complexity: API endpoints, Celery tasks, database schema
- Low complexity: Frontend UI (follows existing patterns)

**Testing Complexity: 8/10**
- Difficult to test GPU operations without hardware
- Mocking PyTorch and CUDA operations is complex
- Performance testing requires real Jetson Orin Nano

**Maintenance Complexity: 6/10**
- Need to keep up with Hugging Face Transformers updates
- bitsandbytes versioning can be brittle
- Model compatibility testing for new architectures

### Alternative Approaches Considered

**Alternative 1: Use TensorFlow instead of PyTorch**
- **Pros:** Potentially better edge optimization (TensorFlow Lite)
- **Cons:** Hugging Face Transformers primarily PyTorch-based, poor quantization support
- **Decision:** Rejected due to ecosystem mismatch

**Alternative 2: Use ONNX Runtime for inference**
- **Pros:** Faster inference, better optimization
- **Cons:** Limited quantization support, poor forward hook support for activation extraction
- **Decision:** Rejected due to activation extraction requirements

**Alternative 3: Load models on-demand (no persistent loading)**
- **Pros:** Simpler implementation, no memory management
- **Cons:** Extremely slow (60s load time per request), poor UX
- **Decision:** Rejected due to performance concerns

---

## 13. Development Phases

### Phase 1: Core Model Loading (Weeks 1-2)

**Objectives:**
- Implement model database schema and API endpoints
- Implement ModelRegistry service with quantization support
- Implement model loading Celery task with progress tracking
- Frontend: ModelsPanel, ModelCard components

**Deliverables:**
- POST /api/models, POST /api/models/:id/load endpoints
- ModelRegistry service with memory management
- Celery task: load_model_task
- Frontend components with loading UI

**Acceptance Criteria:**
- Can add GPT-2 Medium model via API
- Can load model with INT8 quantization
- Loading progress updates via WebSocket
- Model status transitions: pending → loading → ready

**Estimated Effort:** 60-80 hours (2 developers × 1.5-2 weeks)

---

### Phase 2: Activation Extraction (Weeks 3-4)

**Objectives:**
- Implement activation_cache database schema
- Implement ActivationExtractor service with forward hooks
- Implement activation extraction Celery task
- Frontend: ModelDetailsModal with extraction configuration

**Deliverables:**
- POST /api/models/:id/extract-activations endpoint
- ActivationExtractor service with layer selection
- Celery task: extract_activations_task
- Frontend extraction UI with progress tracking

**Acceptance Criteria:**
- Can extract activations from loaded model
- Can select specific layers for extraction
- Activations saved to cache directory
- Cache reused on subsequent requests

**Estimated Effort:** 50-70 hours (2 developers × 1.5-2 weeks)

---

### Phase 3: Optimization & Polish (Week 5)

**Objectives:**
- Implement cache eviction (LRU, storage limits)
- Optimize model loading speed (safetensors, device_map)
- Add error handling and retry logic
- Frontend polish (loading states, error messages)

**Deliverables:**
- Cache cleanup job (Celery periodic task)
- Optimized loading with progress updates
- Comprehensive error handling
- Polished UI matching Mock design

**Acceptance Criteria:**
- Model loading <60 seconds for 7B models
- GPU memory stays <6GB during operations
- Graceful error handling (OOM, network errors)
- UI matches Mock-embedded-interp-ui.tsx

**Estimated Effort:** 30-40 hours (2 developers × 1 week)

---

### Phase 4: Testing & Documentation (Week 6)

**Objectives:**
- Write unit tests (70% coverage target)
- Write integration tests (API endpoints)
- Performance testing on Jetson Orin Nano
- Write deployment and operation documentation

**Deliverables:**
- Unit test suite (pytest)
- Integration test suite (API tests)
- Performance benchmarks
- Deployment guide and runbook

**Acceptance Criteria:**
- Unit test coverage >70%
- All integration tests passing
- Performance meets targets (load <60s, extract <5min)
- Documentation complete and reviewed

**Estimated Effort:** 30-40 hours (2 developers × 1 week)

---

### Total Estimated Timeline: 6 weeks (2 developers)

**Dependencies Between Phases:**
- Phase 2 depends on Phase 1 (needs ModelRegistry)
- Phase 3 depends on Phase 2 (needs ActivationExtractor)
- Phase 4 depends on all previous phases (needs complete feature)

**Critical Path:**
- Model loading (Phase 1) is critical path (blocks all other work)
- Activation extraction (Phase 2) blocks SAE Training feature

**Milestone Definitions:**
- **M1 (Week 2):** Core model loading functional, can load GPT-2
- **M2 (Week 4):** Activation extraction functional, can extract from loaded models
- **M3 (Week 5):** Feature complete, optimized for edge deployment
- **M4 (Week 6):** Feature tested, documented, ready for production

---

## 14. Appendix

### A. Model Type Support Matrix

| Model Type | Supported | Quantization | Max Parameters (INT4) | Notes |
|------------|-----------|--------------|----------------------|-------|
| GPT-2 | ✅ Yes | INT8, INT4 | 1.5B | Full support |
| GPT-Neo | ✅ Yes | INT8, INT4 | 2.7B | Full support |
| LLaMA | ✅ Yes | INT8, INT4 | 13B | Requires custom tokenizer |
| LLaMA-2 | ✅ Yes | INT8, INT4 | 13B | Full support |
| Mistral | ✅ Yes | INT8, INT4 | 7B | Full support |
| Falcon | ⚠️ Experimental | INT8 | 7B | Limited testing |
| BLOOM | ⚠️ Experimental | INT8 | 7B | Limited testing |
| GPT-J | ⚠️ Experimental | INT8, INT4 | 6B | Limited testing |

### B. Quantization Performance Comparison

| Quantization | Memory Reduction | Perplexity Increase | Load Time | Inference Speed |
|--------------|------------------|---------------------|-----------|-----------------|
| None (FP16) | 0% (baseline) | 0% (baseline) | 1.0x | 1.0x |
| INT8 | 50% | <1% | 0.8x | 1.2x |
| INT4 | 87.5% | <5% | 0.6x | 1.5x |

**Recommendations:**
- Use INT8 for 7B models (good balance of quality and memory)
- Use INT4 for 13B models (only way to fit in 6GB GPU)
- Use FP16 only for small models (<1B parameters)

### C. GPU Memory Budget Breakdown

**Example: GPT-2 Medium (355M parameters) with INT8 quantization**
```
Model Weights: 355M params × 1 byte (INT8) = 355 MB
Activations (batch=32): 32 × 1024 seq × 1024 hidden × 4 bytes = 134 MB
Gradients: 0 MB (inference only, no gradients)
PyTorch Overhead: ~200 MB (CUDA context, etc.)
Total: ~690 MB
```

**Example: LLaMA-7B with INT4 quantization**
```
Model Weights: 7B params × 0.5 bytes (INT4) = 3.5 GB
Activations (batch=16): 16 × 2048 seq × 4096 hidden × 4 bytes = 537 MB
Gradients: 0 MB
PyTorch Overhead: ~500 MB
Total: ~4.5 GB (fits in 6GB limit)
```

### D. Activation Extraction Performance

**Benchmark: GPT-2 Medium, 1000 samples, all layers (12 layers)**
- Extraction time: ~180 seconds (3 minutes)
- Throughput: ~5.5 samples/second
- Cache size: 1000 × 1024 × 1024 × 4 bytes × 12 layers = ~50 GB
- GPU memory usage: ~1.2 GB (model + activations buffer)

**Optimization Notes:**
- Batch size 32 provides best throughput (GPU utilization ~85%)
- Extracting fewer layers proportionally reduces time and storage
- Common use case: Extract 3-5 key layers (reduces storage by 60-75%)

### E. WebSocket Event Specifications

**Event: model:loading**
```json
{
  "model_id": "m_gpt2_medium_abc123",
  "status": "loading",
  "progress": 45.2,
  "message": "Loading model layers...",
  "timestamp": "2025-10-06T10:15:23Z"
}
```

**Event: model:ready**
```json
{
  "model_id": "m_gpt2_medium_abc123",
  "status": "ready",
  "memory_usage_bytes": 690000000,
  "load_time_seconds": 52.3,
  "timestamp": "2025-10-06T10:16:15Z"
}
```

**Event: extraction:progress**
```json
{
  "extraction_id": "ext_act_abc123",
  "model_id": "m_gpt2_medium_abc123",
  "status": "extracting",
  "progress": 67.8,
  "samples_processed": 678,
  "total_samples": 1000,
  "timestamp": "2025-10-06T10:20:45Z"
}
```

---

**Document End**
**Total Sections:** 14
**Estimated Implementation Time:** 6 weeks (2 developers)
**Review Status:** Pending stakeholder review
