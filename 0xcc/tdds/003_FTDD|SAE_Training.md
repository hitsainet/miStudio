# Technical Design Document: SAE Training

**Document ID:** 003_FTDD|SAE_Training
**Feature:** Sparse Autoencoder (SAE) Training Configuration and Execution
**PRD Reference:** 003_FPRD|SAE_Training.md
**ADR Reference:** 000_PADR|miStudio.md
**Status:** Draft
**Created:** 2025-10-06
**Last Updated:** 2025-10-06
**Owner:** miStudio Development Team

---

## 1. Executive Summary

This Technical Design Document defines the architecture and implementation approach for the SAE Training feature of miStudio. The feature enables users to configure, launch, and monitor Sparse Autoencoder (SAE) training jobs that extract interpretable features from language model activations, forming the core of the mechanistic interpretability workflow.

**Business Objective:** Transform raw model activations into structured, interpretable feature representations through efficient SAE training optimized for edge deployment on NVIDIA Jetson Orin Nano.

**Technical Approach:**
- FastAPI backend with Celery for distributed training job execution
- PyTorch 2.0+ training loop with automatic mixed precision (AMP)
- SAE model architectures: Sparse, Skip, and Transcoder variants
- Real-time progress tracking via WebSocket with metrics streaming
- Checkpoint management with safetensors format for pause/resume
- Memory optimization: gradient accumulation, dynamic batch sizing, cache management
- React frontend matching Mock-embedded-interp-ui.tsx TrainingPanel and TrainingCard components

**Key Design Decisions:**
1. Use Celery workers for long-running training jobs (non-blocking API)
2. Implement gradient accumulation to handle OOM errors gracefully
3. Store metrics in separate `training_metrics` table for efficient time-series queries
4. Use safetensors format for checkpoint serialization (safer than pickle)
5. Emit WebSocket events every 10 steps for real-time UI updates
6. Implement checkpoint retention policy (keep first, last, every 1000 steps, best loss)

**Success Metrics:**
- Training throughput: >10 steps/second on Jetson Orin Nano (batch=256, expansion=8x)
- GPU memory usage: <6GB during training (includes model + SAE + activations)
- Checkpoint save time: <5 seconds for SAE models with <100M parameters
- Training stability: >95% completion rate without OOM or NaN errors
- Dead neuron rate: <5% across trained SAEs (target: <5% dead neurons)

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │TrainingPanel │  │ TrainingCard │  │ CheckpointManagement │ │
│  │(Config UI)   │  │(Progress)    │  │(Save/Load/Delete)    │ │
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
│  │              API Routes (/api/trainings)                  │   │
│  │  POST /trainings (create+start) │ GET /trainings (list)   │   │
│  │  GET /trainings/:id             │ POST /trainings/:id/pause│   │
│  │  POST /trainings/:id/resume     │ POST /trainings/:id/stop│   │
│  │  POST /trainings/:id/checkpoints (save)                   │   │
│  │  GET /trainings/:id/metrics (time-series)                 │   │
│  └─────────────┬────────────────────────────────────────────┘   │
│                │                                                 │
│  ┌─────────────┴────────────────────────────────────────────┐   │
│  │         Services Layer                                    │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐   │   │
│  │  │ TrainingService │  │ CheckpointService            │   │   │
│  │  │ - Validate cfg  │  │ - Save/load checkpoints      │   │   │
│  │  │ - Create jobs   │  │ - Retention policy           │   │   │
│  │  │ - Control flow  │  │ - Filesystem management      │   │   │
│  │  └────────┬────────┘  └──────────┬───────────────────┘   │   │
│  └───────────┼────────────────────────┼─────────────────────┘   │
│              │                        │                          │
│  ┌───────────┴────────────────────────┴─────────────────────┐   │
│  │         Celery Workers (Background Training)              │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │ train_sae_task(training_id, config)                 │ │   │
│  │  │   1. Load model + dataset                           │ │   │
│  │  │   2. Create SAE architecture (encoder/decoder)      │ │   │
│  │  │   3. Initialize optimizer + scheduler               │ │   │
│  │  │   4. Training loop (extract → encode → decode)      │ │   │
│  │  │   5. Calculate metrics (L0 sparsity, dead neurons)  │ │   │
│  │  │   6. Emit WebSocket progress events                 │ │   │
│  │  │   7. Save checkpoints (auto + manual)               │ │   │
│  │  │   8. Handle pause/resume/stop signals               │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └───────────┬────────────────────────────────────────────┘   │
└──────────────┼───────────────────────────────────────────────┘
               │
┌──────────────┼───────────────────────────────────────────────┐
│         Data Layer                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  PostgreSQL     │  │  Redis       │  │  Filesystem     │  │
│  │  - trainings    │  │  - Celery    │  │  - Checkpoints  │  │
│  │  - metrics      │  │  - Signals   │  │  - Logs         │  │
│  │  - checkpoints  │  │  - Progress  │  │  - Config       │  │
│  └─────────────────┘  └──────────────┘  └─────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Training Flow Diagram

```
User clicks "Start Training"
    ↓
Frontend: POST /api/trainings (config)
    ↓
FastAPI: Validate config, create training record (status='queued')
    ↓
FastAPI: Enqueue Celery task → train_sae_task.delay(training_id)
    ↓
Celery Worker: Picks up task, update status='initializing'
    ↓
Worker: Load model from ModelRegistry (via Model Management)
    ↓
Worker: Load dataset from filesystem (via Dataset Management)
    ↓
Worker: Create SAE architecture (encoder/decoder based on encoder_type)
    ↓
Worker: Initialize optimizer (Adam/AdamW/SGD) + LR scheduler
    ↓
Worker: Update status='training', emit WebSocket 'training:started'
    ↓
Worker: Training loop START
    ├─ FOR step in range(0, total_steps):
    │   ├─ Check pause/stop signals (Redis flags)
    │   ├─ Get batch from dataset (batch_size samples)
    │   ├─ Extract model activations (forward hooks)
    │   ├─ SAE forward pass: activations → encoder → decoder
    │   ├─ Calculate loss: reconstruction_loss + L1_penalty + ghost_penalty
    │   ├─ Backward pass: loss.backward()
    │   ├─ Gradient clipping (max_norm=1.0)
    │   ├─ Optimizer step + scheduler step
    │   ├─ Every 10 steps:
    │   │   ├─ Calculate metrics (L0 sparsity, dead neurons)
    │   │   ├─ Save metrics to training_metrics table
    │   │   ├─ Update denormalized fields in trainings table
    │   │   ├─ Emit WebSocket 'training:progress' event
    │   ├─ If auto-save enabled and step % interval == 0:
    │   │   └─ Save checkpoint (model, optimizer, scheduler state)
    │   ├─ Handle OOM error:
    │   │   ├─ Reduce batch_size by 50%
    │   │   ├─ Implement gradient accumulation
    │   │   ├─ Clear GPU cache
    │   │   └─ Retry step
    │   └─ Clear GPU cache
    └─ Training loop END
    ↓
Worker: Save final checkpoint (marked as final=True)
    ↓
Worker: Update status='completed', emit WebSocket 'training:completed'
    ↓
Frontend: Receives completion event, displays results
```

### Component Relationships

**Frontend Components (React):**
- **TrainingPanel (Lines 1628-1842):** Configuration UI, dataset/model selection, hyperparameter inputs, "Start Training" button
- **TrainingCard (Lines 1845-2156):** Progress tracking, metrics display, checkpoint management, control buttons (pause/resume/stop)
- **CheckpointManagement (Lines 1941-2027):** Checkpoint list, save/load/delete actions, auto-save configuration
- **LiveMetrics (Lines 2031-2086):** Loss curve, L0 sparsity curve, training logs

**Backend Services:**
- **TrainingService:** Handles training lifecycle (create, validate, control flow)
- **CheckpointService:** Manages checkpoint save/load, retention policy, filesystem operations
- **ModelRegistry (from 002_FTDD):** Provides loaded models for activation extraction
- **ActivationExtractor (from 002_FTDD):** Extracts activations during training

**Celery Tasks:**
- **train_sae_task:** Main training loop execution (long-running background job)
- **cleanup_checkpoints_task:** Periodic job to enforce retention policy

### Integration Points

**With Dataset Management (001_FPRD):**
- Requires tokenized datasets for training data
- Uses dataset.storage_path to load samples during training
- Validates dataset.status='ready' before training start

**With Model Management (002_FPRD):**
- Requires models loaded in GPU memory for activation extraction
- Uses ModelRegistry.get_model() to access loaded models
- Shares quantization configuration for memory consistency

**With Feature Discovery (004_FPRD - Downstream):**
- Provides trained SAE checkpoints for feature extraction
- Shares SAE architecture configuration for inference

**With Model Steering (005_FPRD - Downstream):**
- Provides trained SAE models for feature-based interventions
- Enables real-time activation manipulation during generation

---

## 3. Technical Stack

### Core Technologies

**Backend:**
- **FastAPI 0.104+:** Async API framework for training endpoints
- **Celery 5.3+:** Distributed task queue for training execution
- **Redis 7.0+:** Message broker for Celery, signals for pause/stop
- **PyTorch 2.0+:** ML framework with CUDA support for training
- **bitsandbytes 0.41+:** Memory-efficient optimizers (8-bit Adam)
- **safetensors 0.4+:** Safe checkpoint serialization (no pickle)

**Frontend:**
- **React 18+:** UI framework
- **Zustand 4.4+:** State management for training state
- **WebSocket (socket.io-client):** Real-time progress updates
- **Lucide React:** Icons for training UI

**Database:**
- **PostgreSQL 14+:** Training metadata, metrics, checkpoints
- **File System:** Checkpoint files (`/data/trainings/{training_id}/checkpoints/`), training logs

### Technology Justifications

| Technology | Justification | Alternative Considered | Why Rejected |
|------------|--------------|------------------------|--------------|
| **Celery for Training** | Non-blocking API (training takes minutes), automatic retry on errors, progress tracking | Threading/asyncio | Training blocks for minutes, needs separate process |
| **PyTorch AMP** | 40% memory reduction via mixed precision, 1.5x faster training | FP16 manual conversion | AMP handles overflow/underflow automatically |
| **safetensors Format** | 2-3x faster load, no arbitrary code execution (security) | PyTorch .pt files | pickle is unsafe, slower |
| **WebSocket for Progress** | Real-time updates (sub-second latency), bidirectional | Polling | Polling wastes resources, slower updates |
| **Gradient Accumulation** | Enables training with small batch sizes on OOM | Reduce batch size only | Maintains effective batch size for stability |
| **Separate metrics Table** | Efficient time-series queries, 10,000+ records per training | JSONB in trainings table | JSONB slow for range queries |

### Dependencies and Versions

```python
# requirements.txt additions for SAE Training
torch==2.1.0+cu118
celery==5.3.4
redis==5.0.1
safetensors==0.4.0
torch-ema==0.3  # For exponential moving average of weights (optional)
```

**Version Requirements:**
- PyTorch 2.0+ required for automatic mixed precision (torch.cuda.amp)
- Celery 5.3+ required for asyncio support
- Redis 7.0+ required for persistent task results

**Dependency Conflicts:**
- None identified (all dependencies compatible)

---

## 4. Data Design

### Database Schema Expansion

#### trainings Table (from 003_SPEC|Postgres_Usecase_Details_and_Guidance.md lines 223-346)

```sql
CREATE TABLE trainings (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'tr_abc123_1696596000'

    -- Foreign keys
    model_id VARCHAR(255) NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    dataset_id VARCHAR(255) NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,

    -- Configuration
    encoder_type VARCHAR(50) NOT NULL,  -- 'sparse', 'skip', 'transcoder'
    hyperparameters JSONB NOT NULL,  -- Complete hyperparameter set

    -- State
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
        -- Values: 'queued', 'initializing', 'training', 'paused',
        --         'stopped', 'completed', 'failed', 'error'

    -- Progress tracking
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER NOT NULL,
    progress FLOAT GENERATED ALWAYS AS (
        CASE WHEN total_steps > 0
        THEN (current_step::FLOAT / total_steps * 100)
        ELSE 0 END
    ) STORED,

    -- Latest metrics (denormalized for quick access)
    latest_loss FLOAT,
    latest_sparsity FLOAT,
    latest_dead_neurons INTEGER,

    -- Error handling
    error_message TEXT,
    error_code VARCHAR(100),
    retry_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    paused_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Resource allocation
    gpu_id INTEGER,
    worker_id VARCHAR(100),

    CONSTRAINT trainings_status_check CHECK (status IN
        ('queued', 'initializing', 'training', 'paused', 'stopped',
         'completed', 'failed', 'error'))
);

CREATE INDEX idx_trainings_status ON trainings(status);
CREATE INDEX idx_trainings_model_id ON trainings(model_id);
CREATE INDEX idx_trainings_dataset_id ON trainings(dataset_id);
CREATE INDEX idx_trainings_created_at ON trainings(created_at DESC);
CREATE INDEX idx_trainings_completed_at ON trainings(completed_at DESC NULLS LAST);

-- Composite index for filtering active trainings
CREATE INDEX idx_trainings_active ON trainings(status, updated_at)
    WHERE status IN ('queued', 'initializing', 'training', 'paused');
```

**Hyperparameters JSONB Structure:**
```json
{
  "learningRate": 0.001,
  "batchSize": 256,
  "l1Coefficient": 0.0001,
  "expansionFactor": 8,
  "trainingSteps": 10000,
  "optimizer": "AdamW",
  "lrSchedule": "cosine",
  "ghostGradPenalty": true
}
```

#### training_metrics Table

```sql
CREATE TABLE training_metrics (
    id BIGSERIAL PRIMARY KEY,
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Metrics
    step INTEGER NOT NULL,
    loss FLOAT NOT NULL,
    sparsity FLOAT,  -- L0 sparsity (average active features)
    reconstruction_error FLOAT,
    dead_neurons INTEGER,
    explained_variance FLOAT,

    -- Learning rate (for debugging)
    learning_rate FLOAT,

    -- Timestamp
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite unique constraint (one metric per step per training)
    CONSTRAINT training_metrics_unique_step UNIQUE (training_id, step)
);

-- Critical index for time-series queries
CREATE INDEX idx_training_metrics_training_step ON training_metrics(training_id, step);
CREATE INDEX idx_training_metrics_timestamp ON training_metrics(training_id, timestamp);
```

**Storage Estimate:** ~50 bytes per metric × 10,000 steps = 500KB per training

**Query Performance:**
- List metrics for training: `SELECT * FROM training_metrics WHERE training_id = ? ORDER BY step` → Uses idx_training_metrics_training_step (fast)
- Get latest 20 metrics: `SELECT * FROM training_metrics WHERE training_id = ? ORDER BY step DESC LIMIT 20` → <1ms query time

#### checkpoints Table

```sql
CREATE TABLE checkpoints (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'cp_abc123'
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Checkpoint details
    step INTEGER NOT NULL,
    loss FLOAT,
    sparsity FLOAT,

    -- File storage
    storage_path VARCHAR(1000) NOT NULL,  -- /data/trainings/{training_id}/checkpoints/{id}.pt
    file_size_bytes BIGINT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite unique constraint
    CONSTRAINT checkpoints_unique_step UNIQUE (training_id, step)
);

CREATE INDEX idx_checkpoints_training_id ON checkpoints(training_id);
CREATE INDEX idx_checkpoints_step ON checkpoints(training_id, step DESC);
CREATE INDEX idx_checkpoints_created_at ON checkpoints(created_at DESC);
```

**Storage Estimate:** ~500 bytes per checkpoint × 10 checkpoints = 5KB per training (metadata only)

### Data Validation Strategy

**Training Configuration Validation (Pydantic):**
```python
class TrainingConfigRequest(BaseModel):
    dataset_id: str = Field(..., regex="^ds_[a-zA-Z0-9_]+$")
    model_id: str = Field(..., regex="^m_[a-zA-Z0-9_]+$")
    encoder_type: Literal["sparse", "skip", "transcoder"]
    hyperparameters: HyperparametersSchema

    @validator("dataset_id")
    def validate_dataset_ready(cls, v):
        # Check dataset.status='ready' in database
        dataset = db.query(Dataset).filter(Dataset.id == v).first()
        if not dataset or dataset.status != "ready":
            raise ValueError(f"Dataset {v} not ready for training")
        return v

    @validator("model_id")
    def validate_model_ready(cls, v):
        # Check model.status='ready' in database
        model = db.query(Model).filter(Model.id == v).first()
        if not model or model.status != "ready":
            raise ValueError(f"Model {v} not ready for training")
        return v

class HyperparametersSchema(BaseModel):
    learningRate: float = Field(..., ge=0.000001, le=0.1)
    batchSize: int = Field(..., ge=64, le=2048)
    l1Coefficient: float = Field(..., ge=0.00001, le=0.01)
    expansionFactor: int = Field(..., ge=4, le=64)
    trainingSteps: int = Field(..., ge=1000, le=1000000)
    optimizer: Literal["Adam", "AdamW", "SGD"]
    lrSchedule: Literal["constant", "linear", "cosine", "exponential"]
    ghostGradPenalty: bool = True
```

**Checkpoint Validation:**
- Validate checkpoint file exists before load
- Verify file_size_bytes matches actual file size
- Check training_id matches checkpoint metadata
- Validate hyperparameters match current training configuration

### Migration Strategy

**Migration 003: Create SAE Training Tables**
```sql
BEGIN;

CREATE TABLE trainings (...);
CREATE TABLE training_metrics (...);
CREATE TABLE checkpoints (...);

-- Create indexes
CREATE INDEX idx_trainings_status ON trainings(status);
-- ... additional indexes

COMMIT;
```

**Rollback Strategy:**
```sql
BEGIN;
DROP TABLE IF EXISTS checkpoints CASCADE;
DROP TABLE IF EXISTS training_metrics CASCADE;
DROP TABLE IF EXISTS trainings CASCADE;
COMMIT;
```

---

## 5. API Design

### RESTful API Conventions

**Base Path:** `/api/trainings`

**HTTP Method Semantics:**
- GET: Retrieve training(s) - idempotent, cacheable
- POST: Create training, trigger actions (pause/resume/stop)
- DELETE: Cancel and cleanup training

**Status Codes:**
- 200 OK: Successful GET/POST (actions)
- 201 Created: Successful POST (create)
- 202 Accepted: Action queued (async operation)
- 400 Bad Request: Validation error
- 404 Not Found: Training not found
- 409 Conflict: Status conflict (e.g., already paused)
- 500 Internal Server Error: Unexpected error

### API Endpoint Specifications

#### POST /api/trainings
**Purpose:** Create and start new training job

**Request Body:**
```json
{
  "dataset_id": "ds_pile_123abc",
  "model_id": "m_gpt2_medium_456def",
  "encoder_type": "sparse",
  "hyperparameters": {
    "learningRate": 0.001,
    "batchSize": 256,
    "l1Coefficient": 0.0001,
    "expansionFactor": 8,
    "trainingSteps": 10000,
    "optimizer": "AdamW",
    "lrSchedule": "cosine",
    "ghostGradPenalty": true
  }
}
```

**Response:** 201 Created
```json
{
  "id": "tr_abc123_1696596000",
  "status": "queued",
  "created_at": "2025-10-06T10:30:00Z",
  "dataset": { "id": "ds_pile_123abc", "name": "The Pile" },
  "model": { "id": "m_gpt2_medium_456def", "name": "GPT-2 Medium" },
  "encoder_type": "sparse",
  "hyperparameters": { ... },
  "progress": 0,
  "current_step": 0,
  "total_steps": 10000
}
```

**Implementation Notes:**
- Enqueue Celery task immediately after DB record creation
- Return quickly (<100ms) to avoid blocking
- WebSocket updates handle progress tracking

#### GET /api/trainings
**Purpose:** List all training jobs with filtering

**Query Parameters:**
- `status` (optional): Comma-separated statuses (e.g., "training,paused,completed")
- `model_id` (optional): Filter by model
- `dataset_id` (optional): Filter by dataset
- `limit` (default: 50): Max results
- `offset` (default: 0): Pagination offset

**Response:** 200 OK
```json
{
  "trainings": [
    {
      "id": "tr_abc123",
      "dataset_id": "ds_pile_123",
      "model_id": "m_gpt2_456",
      "encoder_type": "sparse",
      "status": "training",
      "progress": 45.2,
      "current_step": 4520,
      "total_steps": 10000,
      "latest_loss": 0.0234,
      "latest_sparsity": 52.3,
      "latest_dead_neurons": 45,
      "created_at": "2025-10-06T10:00:00Z",
      "started_at": "2025-10-06T10:05:00Z",
      "updated_at": "2025-10-06T10:45:00Z"
    }
  ],
  "total": 15,
  "limit": 50,
  "offset": 0
}
```

#### POST /api/trainings/:id/pause
**Purpose:** Pause active training

**Request Body:** (empty)

**Response:** 200 OK
```json
{
  "status": "paused",
  "paused_at": "2025-10-06T10:50:00Z",
  "checkpoint_id": "cp_pause_abc123"
}
```

**Implementation:**
- Set Redis flag: `training:{training_id}:pause_signal = 1`
- Worker checks flag every training step
- Worker saves checkpoint before pausing
- Worker updates status='paused' and exits loop

#### POST /api/trainings/:id/resume
**Purpose:** Resume paused training

**Request Body:** (empty)

**Response:** 200 OK
```json
{
  "status": "training",
  "resumed_at": "2025-10-06T11:00:00Z"
}
```

**Implementation:**
- Load most recent checkpoint
- Re-enqueue Celery task with `resume=True` flag
- Worker loads checkpoint and continues from current_step

#### POST /api/trainings/:id/stop
**Purpose:** Permanently stop training

**Request Body:** (empty)

**Response:** 200 OK
```json
{
  "status": "stopped",
  "stopped_at": "2025-10-06T11:05:00Z",
  "final_checkpoint_id": "cp_final_abc123"
}
```

**Implementation:**
- Set Redis flag: `training:{training_id}:stop_signal = 1`
- Worker saves final checkpoint
- Worker updates status='stopped' and exits

#### GET /api/trainings/:id/metrics
**Purpose:** Retrieve time-series training metrics

**Query Parameters:**
- `start_step` (optional): Start of step range
- `end_step` (optional): End of step range
- `interval` (optional): Downsample to every Nth step
- `limit` (default: 1000): Max records

**Response:** 200 OK
```json
{
  "training_id": "tr_abc123",
  "metrics": [
    {
      "step": 100,
      "loss": 0.0456,
      "sparsity": 48.3,
      "reconstruction_error": 0.0412,
      "dead_neurons": 78,
      "learning_rate": 0.001,
      "timestamp": "2025-10-06T10:10:00Z"
    },
    {
      "step": 200,
      "loss": 0.0398,
      "sparsity": 51.2,
      "reconstruction_error": 0.0354,
      "dead_neurons": 62,
      "learning_rate": 0.00095,
      "timestamp": "2025-10-06T10:15:00Z"
    }
  ],
  "total": 4520
}
```

#### POST /api/trainings/:id/checkpoints
**Purpose:** Manually save checkpoint

**Request Body:**
```json
{
  "note": "Before hyperparameter change"
}
```

**Response:** 201 Created
```json
{
  "id": "cp_manual_abc123",
  "training_id": "tr_abc123",
  "step": 4520,
  "loss": 0.0234,
  "sparsity": 52.3,
  "storage_path": "/data/trainings/tr_abc123/checkpoints/cp_manual_abc123.pt",
  "file_size_bytes": 45670234,
  "created_at": "2025-10-06T10:50:00Z"
}
```

**Implementation:**
- Set Redis flag: `training:{training_id}:save_checkpoint = 1`
- Worker detects flag, saves checkpoint
- Worker clears flag after save

### WebSocket Event Protocol

**Connection:** `socket.io` connection to `/ws`

**Event Subscription:**
```javascript
const socket = io('/ws');
socket.emit('subscribe', { channel: `training:${trainingId}` });
```

**Events:**

**training:created**
```json
{
  "training_id": "tr_abc123",
  "status": "queued",
  "timestamp": "2025-10-06T10:30:00Z"
}
```

**training:status_changed**
```json
{
  "training_id": "tr_abc123",
  "old_status": "initializing",
  "new_status": "training",
  "timestamp": "2025-10-06T10:35:00Z"
}
```

**training:progress** (every 10 steps)
```json
{
  "training_id": "tr_abc123",
  "current_step": 4520,
  "progress": 45.2,
  "latest_loss": 0.0234,
  "latest_sparsity": 52.3,
  "latest_dead_neurons": 45,
  "gpu_utilization": 78.5,
  "timestamp": "2025-10-06T10:45:00Z"
}
```

**checkpoint:created**
```json
{
  "training_id": "tr_abc123",
  "checkpoint_id": "cp_manual_abc123",
  "step": 4520,
  "timestamp": "2025-10-06T10:50:00Z"
}
```

### Error Handling Strategy

**Error Response Format:**
```json
{
  "error": {
    "code": "TRAINING_FAILED",
    "message": "Training failed: insufficient GPU memory",
    "details": {
      "required_memory_gb": 8.5,
      "available_memory_gb": 6.0,
      "suggestion": "Reduce batch size or expansion factor"
    },
    "timestamp": "2025-10-06T10:15:00Z"
  }
}
```

**Error Code Categories:**
- `VALIDATION_ERROR`: Invalid configuration (400)
- `TRAINING_NOT_FOUND`: Training ID doesn't exist (404)
- `TRAINING_FAILED`: Training error (500)
- `INSUFFICIENT_MEMORY`: OOM error (500)
- `STATUS_CONFLICT`: Invalid status transition (409)

---

## 6. Component Architecture

### Frontend Component Hierarchy

```
TrainingPanel (Lines 1628-1842)
├── Configuration Form
│   ├── Dataset Dropdown (readyDatasets filter)
│   ├── Model Dropdown (ready models filter)
│   ├── Encoder Type Dropdown (sparse/skip/transcoder)
│   └── Advanced Hyperparameters (collapsible)
│       ├── Learning Rate Input
│       ├── Batch Size Dropdown
│       ├── L1 Coefficient Input
│       ├── Expansion Factor Dropdown
│       ├── Training Steps Input
│       ├── Optimizer Dropdown
│       ├── LR Schedule Dropdown
│       └── Ghost Gradient Penalty Toggle
├── Start Training Button (disabled if !model || !dataset)
└── Training Jobs List
    └── TrainingCard[] (one per training)

TrainingCard (Lines 1845-2156)
├── Header
│   ├── Model + Dataset Name
│   ├── Encoder Type + Start Time
│   └── Status Badge (with icon)
├── Progress Section (if training/paused/completed)
│   ├── Progress Bar (0-100%)
│   └── Metrics Grid (4 columns)
│       ├── Loss (emerald)
│       ├── L0 Sparsity (blue)
│       ├── Dead Neurons (red)
│       └── GPU Utilization (purple)
├── Action Buttons
│   ├── Show/Hide Live Metrics Button
│   └── Checkpoints Button (with count)
├── Live Metrics Panel (if showMetrics)
│   ├── Loss Curve (bar chart, last 20 steps)
│   ├── L0 Sparsity Curve (bar chart, last 20 steps)
│   └── Training Logs (monospace, last 10 entries)
├── Checkpoint Management Panel (if showCheckpoints)
│   ├── Save Now Button
│   ├── Checkpoint List (scrollable)
│   │   └── Checkpoint Item (step, loss, timestamp, load/delete)
│   └── Auto-save Configuration
│       ├── Auto-save Toggle
│       └── Interval Input (100-10000 steps)
└── Control Buttons (border-top)
    ├── If status='training': Pause, Stop
    ├── If status='paused': Resume, Stop
    └── If status='stopped': Retry
```

### Backend Service Architecture

```
TrainingService
├── create_training(config: TrainingConfigRequest) -> Training
│   ├── Validate config (dataset ready, model ready, hyperparameters)
│   ├── Create training record in DB (status='queued')
│   ├── Enqueue Celery task: train_sae_task.delay(training.id)
│   └── Return training record
├── pause_training(training_id: str) -> Training
│   ├── Validate status='training'
│   ├── Set Redis pause signal
│   └── Update status='paused'
├── resume_training(training_id: str) -> Training
│   ├── Validate status='paused'
│   ├── Re-enqueue Celery task with resume=True
│   └── Update status='training'
└── stop_training(training_id: str) -> Training
    ├── Validate status in ['training', 'paused']
    ├── Set Redis stop signal
    └── Update status='stopped'

CheckpointService
├── save_checkpoint(training_id, step, state_dict) -> Checkpoint
│   ├── Create checkpoint ID: cp_{uuid}
│   ├── Save to filesystem: /data/trainings/{training_id}/checkpoints/{id}.pt
│   ├── Create checkpoint record in DB
│   └── Emit WebSocket 'checkpoint:created' event
├── load_checkpoint(checkpoint_id: str) -> StateDict
│   ├── Load checkpoint file from filesystem
│   ├── Validate checkpoint integrity
│   └── Return state_dict (model, optimizer, scheduler, step)
├── delete_checkpoint(checkpoint_id: str) -> None
│   ├── Delete checkpoint file from filesystem
│   └── Delete checkpoint record from DB
└── enforce_retention_policy(training_id: str) -> None
    ├── Get all checkpoints for training
    ├── Keep: first, last, every 1000 steps, best loss
    └── Delete others

SAEModel (PyTorch nn.Module)
├── __init__(input_dim, hidden_dim, encoder_type)
│   ├── encoder: Linear(input_dim, hidden_dim)
│   ├── decoder: Linear(hidden_dim, input_dim)
│   └── Optional: skip_connection (if encoder_type='skip')
├── forward(activations: Tensor) -> (encoded, reconstructed)
│   ├── encoded = encoder(activations)
│   ├── Apply ReLU activation
│   ├── reconstructed = decoder(encoded)
│   └── If skip: reconstructed += activations
└── calculate_loss(activations, encoded, reconstructed, l1_coef)
    ├── recon_loss = F.mse_loss(reconstructed, activations)
    ├── l1_penalty = l1_coef * encoded.abs().sum(dim=-1).mean()
    └── total_loss = recon_loss + l1_penalty
```

### Celery Task: train_sae_task

**Pseudocode:**
```python
@celery_app.task(bind=True)
def train_sae_task(self, training_id: str, resume: bool = False):
    # 1. Load training configuration
    training = db.query(Training).filter(Training.id == training_id).first()
    config = training.hyperparameters

    # 2. Update status to 'initializing'
    update_training_status(training_id, 'initializing')
    emit_websocket('training:status_changed', {...})

    # 3. Load model and dataset
    model = model_registry.get_model(training.model_id)
    dataset = load_dataset(training.dataset_id)

    # 4. Create SAE architecture
    sae = SAEModel(
        input_dim=model.hidden_size,
        hidden_dim=model.hidden_size * config.expansionFactor,
        encoder_type=training.encoder_type
    ).cuda()

    # 5. Initialize optimizer and scheduler
    optimizer = create_optimizer(config.optimizer, sae.parameters(), lr=config.learningRate)
    scheduler = create_scheduler(config.lrSchedule, optimizer, config.trainingSteps)

    # 6. Load checkpoint if resuming
    if resume:
        checkpoint = load_latest_checkpoint(training_id)
        sae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
    else:
        start_step = 0

    # 7. Update status to 'training'
    update_training_status(training_id, 'training')
    emit_websocket('training:status_changed', {...})

    # 8. Training loop
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    for step in range(start_step, config.trainingSteps):
        # Check for pause/stop signals
        if redis.get(f'training:{training_id}:pause_signal'):
            save_checkpoint(training_id, step, sae, optimizer, scheduler)
            update_training_status(training_id, 'paused')
            redis.delete(f'training:{training_id}:pause_signal')
            return

        if redis.get(f'training:{training_id}:stop_signal'):
            save_checkpoint(training_id, step, sae, optimizer, scheduler, final=True)
            update_training_status(training_id, 'stopped')
            redis.delete(f'training:{training_id}:stop_signal')
            return

        try:
            # Get batch
            batch = dataset.get_batch(config.batchSize)
            input_ids = tokenize(batch)

            # Extract activations
            with torch.no_grad():
                activations = extract_activations(model, input_ids)

            # Forward pass (mixed precision)
            with torch.cuda.amp.autocast():
                encoded, reconstructed = sae(activations)
                loss = sae.calculate_loss(activations, encoded, reconstructed, config.l1Coefficient)

                if config.ghostGradPenalty:
                    ghost_penalty = calculate_ghost_penalty(encoded, sae.encoder.weight)
                    loss += ghost_penalty

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Calculate metrics every 10 steps
            if step % 10 == 0:
                l0_sparsity = (encoded > 0.01).float().sum(dim=-1).mean().item()
                dead_neurons = count_dead_neurons(encoded)

                # Save metrics to DB
                save_metrics(training_id, step, {
                    'loss': loss.item(),
                    'sparsity': l0_sparsity,
                    'dead_neurons': dead_neurons,
                    'learning_rate': scheduler.get_last_lr()[0]
                })

                # Update denormalized fields
                update_training(training_id, {
                    'current_step': step,
                    'latest_loss': loss.item(),
                    'latest_sparsity': l0_sparsity,
                    'latest_dead_neurons': dead_neurons
                })

                # Emit WebSocket event
                emit_websocket('training:progress', {
                    'training_id': training_id,
                    'current_step': step,
                    'progress': (step / config.trainingSteps) * 100,
                    'latest_loss': loss.item(),
                    'latest_sparsity': l0_sparsity,
                    'latest_dead_neurons': dead_neurons
                })

            # Auto-save checkpoints
            if config.autoSave and step % config.autoSaveInterval == 0:
                save_checkpoint(training_id, step, sae, optimizer, scheduler)

            # Manual checkpoint save signal
            if redis.get(f'training:{training_id}:save_checkpoint'):
                save_checkpoint(training_id, step, sae, optimizer, scheduler)
                redis.delete(f'training:{training_id}:save_checkpoint')

            # Clear GPU cache
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                # Handle OOM error
                config.batchSize = max(16, config.batchSize // 2)
                torch.cuda.empty_cache()
                logger.warning(f"OOM error, reduced batch size to {config.batchSize}")
                continue
            else:
                raise

    # 9. Training complete
    save_checkpoint(training_id, config.trainingSteps, sae, optimizer, scheduler, final=True)
    update_training_status(training_id, 'completed')
    emit_websocket('training:completed', {'training_id': training_id})
```

---

## 7. State Management

### Application State Organization

**Global State (Zustand Store):**
```typescript
interface TrainingStore {
  trainings: Training[];
  selectedConfig: TrainingConfig;
  loading: boolean;
  error: string | null;

  // Actions
  fetchTrainings: () => Promise<void>;
  createTraining: (config: TrainingConfig) => Promise<Training>;
  pauseTraining: (trainingId: string) => Promise<void>;
  resumeTraining: (trainingId: string) => Promise<void>;
  stopTraining: (trainingId: string) => Promise<void>;
  updateTrainingStatus: (trainingId: string, updates: Partial<Training>) => void;

  // WebSocket subscription
  subscribeToTrainingUpdates: () => void;
}
```

**Component State (React useState):**
- `showAdvancedConfig`: Boolean for collapsible advanced hyperparameters
- `showMetrics`: Boolean for live metrics panel visibility
- `showCheckpoints`: Boolean for checkpoint management panel
- `autoSave`: Boolean for auto-save toggle
- `autoSaveInterval`: Number for auto-save interval (steps)

### State Flow Patterns

**Training Start Flow:**
```
1. User clicks "Start Training"
   → Component calls trainingStore.createTraining(selectedConfig)

2. Store dispatches API call POST /api/trainings
   → API creates DB record, enqueues Celery task, returns training

3. Store adds training to trainings[] array
   → Component re-renders with new training in list

4. WebSocket receives 'training:status_changed' event (queued → initializing)
   → Store calls updateTrainingStatus(trainingId, {status: 'initializing'})

5. WebSocket receives 'training:status_changed' event (initializing → training)
   → Store updates status='training'

6. WebSocket receives 'training:progress' events every 10 steps
   → Store updates progress, metrics
   → TrainingCard re-renders with updated progress bar and metrics

7. Training completes, WebSocket receives 'training:completed'
   → Store updates status='completed', progress=100
```

**Checkpoint Save Flow:**
```
1. User clicks "Save Now" in checkpoint panel
   → Component calls saveCheckpoint(trainingId)

2. Component dispatches API call POST /api/trainings/:id/checkpoints
   → API sets Redis save signal

3. Worker detects signal, saves checkpoint
   → Worker emits WebSocket 'checkpoint:created' event

4. Store receives event, adds checkpoint to checkpoints[trainingId]
   → CheckpointManagement component re-renders with new checkpoint
```

### Side Effects Handling

**WebSocket Connection:**
```typescript
useEffect(() => {
  const socket = io('/ws');

  trainings.forEach(training => {
    if (['training', 'paused', 'initializing'].includes(training.status)) {
      socket.emit('subscribe', { channel: `training:${training.id}` });
    }
  });

  socket.on('training:progress', (data) => {
    trainingStore.updateTrainingStatus(data.training_id, data);
  });

  socket.on('checkpoint:created', (data) => {
    // Add checkpoint to store
  });

  return () => socket.disconnect();
}, [trainings]);
```

**Polling Fallback:**
- If WebSocket fails, poll `/api/trainings/:id` every 5 seconds
- Only poll for trainings with active status
- Stop polling when status becomes terminal (completed/stopped/failed)

---

## 8. Security Considerations

### Authentication & Authorization

**JWT-Based Authentication:**
- All endpoints require valid JWT token
- Token validated by FastAPI dependency middleware

**Permission Levels:**
- `training:read`: View trainings and metrics
- `training:write`: Create, pause, resume, stop trainings
- `training:admin`: Delete trainings, manage checkpoints

### Input Validation

**Pydantic Schema Validation:**
- All hyperparameters validated against ranges (see section 4)
- Dataset and model IDs validated for existence and ready status
- Encoder type must be one of: sparse, skip, transcoder

**Redis Signal Validation:**
- Signals only set by authenticated API requests
- Worker validates training_id exists before processing signals

### Rate Limiting

- Training creation: Max 10 trainings per hour per user
- Checkpoint save: Max 20 manual saves per training

---

## 9. Performance & Scalability

### Performance Optimization

**Training Throughput:**
- Target: >10 steps/second on Jetson Orin Nano (batch=256, expansion=8x)
- Optimizations:
  - Mixed precision training (40% memory reduction, 1.5x speedup)
  - Batch size tuning (32-256 range for optimal GPU utilization)
  - Clear GPU cache between steps (prevents memory fragmentation)

**Memory Optimization:**
- Dynamic batch size reduction on OOM (reduce by 50%)
- Gradient accumulation for small batches (maintain effective batch size)
- Activation checkpointing for large models (trades compute for memory)

**Database Query Optimization:**
- Index on `training_metrics(training_id, step)` for fast time-series queries
- Denormalized latest_* fields in trainings table (avoid JOIN on every request)
- Pagination for metrics (default 1000 records, max 10000)

### Scalability Considerations

**Horizontal Scaling:**
- Multiple Celery workers (1 per GPU)
- Stateless API servers (FastAPI can scale to N instances)

**Resource Limits:**
- Max 2 simultaneous trainings on single Jetson Orin Nano (6GB GPU limit)
- Max 10,000 metrics per training (500KB storage)
- Max 20 checkpoints per training (configurable retention policy)

---

## 10. Testing Strategy

### Unit Tests

**Training Configuration Validation:**
```python
def test_hyperparameters_validation():
    # Valid config
    config = HyperparametersSchema(
        learningRate=0.001,
        batchSize=256,
        l1Coefficient=0.0001,
        expansionFactor=8,
        trainingSteps=10000,
        optimizer="AdamW",
        lrSchedule="cosine",
        ghostGradPenalty=True
    )
    assert config.learningRate == 0.001

    # Invalid learning rate (too high)
    with pytest.raises(ValidationError):
        HyperparametersSchema(learningRate=1.0, ...)
```

**SAE Model Forward Pass:**
```python
def test_sae_forward_pass():
    sae = SAEModel(input_dim=768, hidden_dim=768*8, encoder_type='sparse')
    activations = torch.randn(32, 768)

    encoded, reconstructed = sae(activations)

    assert encoded.shape == (32, 768*8)
    assert reconstructed.shape == (32, 768)
    assert not torch.isnan(encoded).any()
```

### Integration Tests

**End-to-End Training Flow:**
```python
def test_training_flow(client, auth_headers):
    # Create training
    response = client.post("/api/trainings", json={
        "dataset_id": "ds_test_123",
        "model_id": "m_test_456",
        "encoder_type": "sparse",
        "hyperparameters": {...}
    }, headers=auth_headers)

    assert response.status_code == 201
    training = response.json()
    assert training["status"] == "queued"

    # Wait for training to start
    time.sleep(5)

    # Check status
    response = client.get(f"/api/trainings/{training['id']}", headers=auth_headers)
    training = response.json()
    assert training["status"] in ["initializing", "training"]
```

### Performance Tests

**Training Throughput Benchmark:**
```python
def test_training_throughput():
    # Run training for 100 steps, measure time
    start = time.time()
    train_sae(training_id, steps=100)
    elapsed = time.time() - start

    steps_per_second = 100 / elapsed
    assert steps_per_second > 10, f"Too slow: {steps_per_second} steps/sec"
```

---

## 11. Deployment & DevOps

### Deployment Pipeline

**CI/CD Stages:**
1. **Test Stage:** Run unit + integration tests
2. **Build Stage:** Build Docker image with PyTorch + CUDA
3. **Deploy Staging:** Deploy to Jetson Orin Nano dev board
4. **Manual Approval:** QA validation
5. **Deploy Production:** Deploy to production Jetson

**Environment Configuration:**
```bash
# .env.production
DATABASE_URL=postgresql://user:pass@localhost:5432/mistudio
REDIS_URL=redis://localhost:6379/0
GPU_MEMORY_LIMIT_GB=6
TRAINING_DATA_DIR=/data/trainings
LOG_LEVEL=INFO
```

### Monitoring & Logging

**Metrics:**
- Training throughput (steps/sec)
- GPU memory usage (alert if >90%)
- Training success rate (target: >95%)
- Checkpoint save time (target: <5 seconds)

**Logging:**
- Application logs: JSON structured logs to stdout
- Training logs: Saved to `/data/trainings/{training_id}/logs.txt`
- Error logs: Sentry integration for error tracking

---

## 12. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **GPU Memory Exhaustion** | High | Critical | Dynamic batch size reduction, gradient accumulation, mixed precision, memory monitoring |
| **Training Instability (NaN loss)** | Medium | High | Gradient clipping, hyperparameter validation, default configs tested on target hardware |
| **Checkpoint Corruption** | Low | Critical | safetensors format, atomic saves (write to temp, then rename), integrity validation |
| **WebSocket Connection Loss** | Medium | Low | Automatic reconnection, polling fallback, continue training even if disconnected |
| **Slow Training on Edge Device** | High | Medium | Mixed precision (1.5x speedup), batch size tuning, clear GPU cache, optimize dataloader |

### Dependencies & Blockers

**Dependency 1: Model Management (002_FPRD)**
- **Blocker Severity:** Critical
- **Requirement:** Loaded models with activation extraction
- **Status:** Complete (002_FTDD finished)

**Dependency 2: Dataset Management (001_FPRD)**
- **Blocker Severity:** Critical
- **Requirement:** Tokenized datasets for training data
- **Status:** Complete (001_FTDD finished)

---

## 13. Development Phases

### Phase 1: Core Training Loop (Weeks 1-2)

**Objectives:**
- Implement trainings database schema and API endpoints
- Implement SAE model architectures (sparse, skip, transcoder)
- Implement basic training loop (forward, backward, optimizer step)
- Implement Celery task: train_sae_task

**Deliverables:**
- POST /api/trainings, GET /api/trainings endpoints
- SAEModel class with forward pass and loss calculation
- Celery task with training loop
- Basic WebSocket progress updates

**Acceptance Criteria:**
- Can create training job via API
- Training loop executes for N steps
- Loss decreases during training
- Progress updates emitted via WebSocket

**Estimated Effort:** 70-90 hours (2 developers × 2 weeks)

---

### Phase 2: Checkpoint Management (Week 3)

**Objectives:**
- Implement checkpoint save/load functionality
- Implement pause/resume/stop controls
- Implement checkpoint retention policy
- Frontend: CheckpointManagement component

**Deliverables:**
- CheckpointService with save/load/delete
- Pause/resume/stop API endpoints
- Redis signal handling in worker
- Frontend checkpoint UI

**Acceptance Criteria:**
- Can save checkpoint during training
- Can pause training (saves checkpoint, exits loop)
- Can resume training (loads checkpoint, continues)
- Retention policy keeps correct checkpoints

**Estimated Effort:** 40-50 hours (2 developers × 1 week)

---

### Phase 3: Metrics & Visualization (Week 4)

**Objectives:**
- Implement training_metrics table and storage
- Implement GET /api/trainings/:id/metrics endpoint
- Frontend: LiveMetrics component (loss curve, sparsity curve, logs)
- Real-time metrics streaming via WebSocket

**Deliverables:**
- Metrics storage (DB insert every 10 steps)
- Metrics retrieval API with filtering
- Frontend charts (bar charts for loss/sparsity)
- Training logs display

**Acceptance Criteria:**
- Metrics saved every 10 steps
- Can query metrics by step range
- Charts update in real-time
- Training logs display timestamped entries

**Estimated Effort:** 30-40 hours (2 developers × 1 week)

---

### Phase 4: Memory Optimization & Error Handling (Week 5)

**Objectives:**
- Implement dynamic batch size reduction on OOM
- Implement gradient accumulation
- Implement ghost gradient penalty
- Error handling and retry logic

**Deliverables:**
- OOM error handling (catch, reduce batch size, retry)
- Gradient accumulation for small batches
- Ghost gradient penalty implementation
- Comprehensive error handling

**Acceptance Criteria:**
- Training survives OOM errors (reduces batch size)
- Gradient accumulation maintains effective batch size
- Ghost gradient penalty reduces dead neurons
- Errors logged with actionable messages

**Estimated Effort:** 30-40 hours (2 developers × 1 week)

---

### Phase 5: Testing & Polish (Week 6)

**Objectives:**
- Write unit tests (70% coverage)
- Write integration tests (end-to-end flows)
- Performance testing on Jetson Orin Nano
- UI polish to match Mock design exactly

**Deliverables:**
- Unit test suite (pytest)
- Integration test suite (API tests)
- Performance benchmarks
- Polished UI matching Mock-embedded-interp-ui.tsx

**Acceptance Criteria:**
- Unit test coverage >70%
- All integration tests passing
- Training throughput >10 steps/sec
- UI matches Mock design (colors, layout, behavior)

**Estimated Effort:** 30-40 hours (2 developers × 1 week)

---

### Total Estimated Timeline: 6 weeks (2 developers)

**Critical Path:**
- Phase 1 (training loop) blocks all other phases
- Phase 2 (checkpoints) blocks pause/resume functionality
- Phase 3 (metrics) blocks visualization

**Milestone Definitions:**
- **M1 (Week 2):** Basic training functional, loss decreases
- **M2 (Week 3):** Pause/resume functional with checkpoints
- **M3 (Week 4):** Real-time metrics and visualization working
- **M4 (Week 5):** Memory optimization complete, handles OOM
- **M5 (Week 6):** Feature complete, tested, polished

---

## 14. Appendix

### A. SAE Architecture Variants

**Sparse Autoencoder (Standard):**
```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed
```

**Skip Autoencoder (with residual connection):**
```python
class SkipAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        reconstructed = self.decoder(encoded) + x  # Skip connection
        return encoded, reconstructed
```

**Transcoder (cross-layer mapping):**
```python
class Transcoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed
```

### B. Checkpoint File Format

**safetensors Checkpoint Structure:**
```python
checkpoint = {
    'training_id': 'tr_abc123',
    'step': 4520,
    'encoder_state_dict': sae.encoder.state_dict(),
    'decoder_state_dict': sae.decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'hyperparameters': {
        'learningRate': 0.001,
        'batchSize': 256,
        # ... all hyperparameters
    },
    'metrics': {
        'loss': 0.0234,
        'sparsity': 52.3,
        'dead_neurons': 45
    },
    'rng_states': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all()
    }
}

# Save with safetensors
save_file(checkpoint, checkpoint_path)
```

### C. Memory Budget Breakdown

**Example: GPT-2 Medium (355M params) + SAE (8x expansion)**
```
Model: 355M × 1 byte (INT8) = 355 MB
SAE Encoder: 1024 × 8192 × 4 bytes = 33 MB
SAE Decoder: 8192 × 1024 × 4 bytes = 33 MB
Activations (batch=256): 256 × 1024 × 1024 × 4 bytes = 1 GB
Gradients: ~66 MB (encoder + decoder)
Optimizer State (AdamW): 66 MB × 2 = 132 MB (momentum + variance)
PyTorch Overhead: ~200 MB
Total: ~1.8 GB (fits in 6GB limit)
```

### D. Training Loop Pseudocode (Complete)

```python
def training_loop(training_id, config):
    # Setup
    model = load_model(config.model_id)
    dataset = load_dataset(config.dataset_id)
    sae = create_sae(model.hidden_size, config.expansionFactor, config.encoder_type)
    optimizer = create_optimizer(config.optimizer, sae.parameters(), config.learningRate)
    scheduler = create_scheduler(config.lrSchedule, optimizer, config.trainingSteps)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for step in range(config.trainingSteps):
        # Check signals
        if check_pause_signal(training_id):
            save_checkpoint(training_id, step, sae, optimizer, scheduler)
            update_status(training_id, 'paused')
            return

        # Training step
        batch = dataset.get_batch(config.batchSize)
        activations = extract_activations(model, batch)

        with torch.cuda.amp.autocast():
            encoded, reconstructed = sae(activations)
            loss = calculate_loss(activations, encoded, reconstructed, config)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics
        if step % 10 == 0:
            save_metrics(training_id, step, loss, encoded)
            emit_progress(training_id, step, loss)

        # Checkpoints
        if config.autoSave and step % config.autoSaveInterval == 0:
            save_checkpoint(training_id, step, sae, optimizer, scheduler)

        torch.cuda.empty_cache()

    # Complete
    save_checkpoint(training_id, config.trainingSteps, sae, optimizer, scheduler, final=True)
    update_status(training_id, 'completed')
```

---

**Document End**
**Total Sections:** 14
**Estimated Implementation Time:** 6 weeks (2 developers)
**Review Status:** Pending stakeholder review
