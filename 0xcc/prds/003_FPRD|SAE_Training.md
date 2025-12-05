# Feature PRD: SAE Training

**Document ID:** 003_FPRD|SAE_Training
**Version:** 1.0 (MVP Complete)
**Last Updated:** 2025-12-05
**Status:** Implemented
**Priority:** P0 (Core Feature)

---

## 1. Overview

### 1.1 Purpose
Enable users to train Sparse Autoencoders on transformer model activations with multiple architecture options and real-time progress monitoring.

### 1.2 User Problem
Researchers need to train SAEs but face challenges with:
- Complex hyperparameter configuration
- Long training times without feedback
- Different SAE architecture requirements
- Checkpoint management and recovery

### 1.3 Solution
A comprehensive training system with multiple architectures, real-time metrics streaming, and training templates for reproducibility.

---

## 2. Functional Requirements

### 2.1 Training Configuration
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | Select dataset and tokenization | Implemented |
| FR-1.2 | Select model and layer for activation extraction | Implemented |
| FR-1.3 | Configure all hyperparameters (see §4) | Implemented |
| FR-1.4 | Select SAE architecture (Standard, JumpReLU, Skip, Transcoder) | Implemented |
| FR-1.5 | Set checkpoint interval | Implemented |

### 2.2 Training Execution
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | Queue training jobs in Celery | Implemented |
| FR-2.2 | Stream metrics via WebSocket in real-time | Implemented |
| FR-2.3 | Auto-checkpoint at configured intervals | Implemented |
| FR-2.4 | Stop training gracefully | Implemented |
| FR-2.5 | Resume from checkpoint | Partial |

### 2.3 Metrics & Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | Display loss curve in real-time | Implemented |
| FR-3.2 | Display L0 sparsity metric | Implemented |
| FR-3.3 | Display reconstruction error | Implemented |
| FR-3.4 | Display FVU (Fraction of Variance Unexplained) | Implemented |
| FR-3.5 | Dead neuron count | Implemented |

### 2.4 Training Management
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | List all training jobs with status | Implemented |
| FR-4.2 | Delete completed/failed trainings | Implemented |
| FR-4.3 | Bulk delete trainings | Implemented |
| FR-4.4 | Retry failed trainings | Implemented |
| FR-4.5 | View hyperparameters modal | Implemented |

### 2.5 Training Templates (Sub-feature)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-5.1 | Save training configuration as template | Implemented |
| FR-5.2 | Load template to populate form | Implemented |
| FR-5.3 | Template favorites | Implemented |
| FR-5.4 | Template import/export (JSON) | Implemented |
| FR-5.5 | Duplicate templates | Implemented |

---

## 3. SAE Architectures

### 3.1 Standard SAE
```python
# Forward pass
z = ReLU(W_enc @ x_norm + b_enc)
x_hat = W_dec @ z + b_dec
# Loss
L = MSE(x_hat, x_norm) + l1_alpha * L1(z)
```

### 3.2 JumpReLU SAE (Gemma Scope Style)
```python
# Forward pass with learnable threshold
z = JumpReLU(W_enc @ x_norm + b_enc, threshold)
x_hat = W_dec @ z + b_dec
# Loss with L0 penalty
L = MSE(x_hat, x_norm) + l0_alpha * L0(z)
```

### 3.3 Skip SAE
```python
# Forward with residual
z = ReLU(W_enc @ x_norm + b_enc)
x_hat = W_dec @ z + b_dec + x_norm  # Skip connection
```

### 3.4 Transcoder SAE
```python
# Layer-to-layer transcoding
z = ReLU(W_enc @ layer_n + b_enc)
layer_n_plus_1_hat = W_dec @ z + b_dec
```

---

## 4. Hyperparameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `hidden_dim` | Model hidden dimension | Auto | Model-dependent |
| `latent_dim` | SAE latent dimension | 8x hidden | 2x - 32x |
| `l1_alpha` | L1 sparsity penalty | 0.001 | 0.0001 - 0.1 |
| `learning_rate` | Adam learning rate | 1e-4 | 1e-5 - 1e-3 |
| `batch_size` | Training batch size | 4096 | 512 - 16384 |
| `num_steps` | Total training steps | 100000 | 1000 - 1000000 |
| `normalize_activations` | Normalization method | constant_norm | none, layer_norm, constant_norm |
| `top_k_sparsity` | Top-K percentage | null | 0.1 - 10.0 |
| `checkpoint_interval` | Steps between checkpoints | 10000 | 1000 - 100000 |
| `dead_neuron_threshold` | Threshold for dead detection | 0.0 | 0.0 - 1e-6 |
| `resample_steps` | Steps for dead neuron resampling | null | null or integer |

---

## 5. User Interface

### 5.1 Training Panel
```
┌─────────────────────────────────────────────────────────────┐
│ Training                              [+ Start Training]    │
├─────────────────────────────────────────────────────────────┤
│ [Templates ▾]                         [Bulk Delete] [Retry] │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Training Card (Running)                                 │ │
│ │ Name: gemma-2b-layer12-8x                              │ │
│ │ Progress: ████████░░ 75% (75000/100000 steps)          │ │
│ │ Loss: 0.0234 | L0: 45.2 | FVU: 0.012                   │ │
│ │ ┌─────────────────────────────────────────────────────┐ │
│ │ │ [Loss Chart]                                        │ │
│ │ └─────────────────────────────────────────────────────┘ │
│ │ [Stop] [Hyperparameters]                               │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Start Training Modal
- Dataset/tokenization selector
- Model/layer selector
- Architecture selector
- Hyperparameter form with collapsible advanced section
- Template load/save
- Estimated training time

### 5.3 Hyperparameters Modal
- Organized sections: Basic, Sparsity, Optimization, Checkpoints
- Read-only display for completed trainings
- Copy configuration button

---

## 6. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/trainings` | GET | List all trainings |
| `/api/v1/trainings` | POST | Start new training |
| `/api/v1/trainings/{id}` | GET | Get training details |
| `/api/v1/trainings/{id}` | DELETE | Delete training |
| `/api/v1/trainings/{id}/stop` | POST | Stop training |
| `/api/v1/trainings/{id}/retry` | POST | Retry failed training |
| `/api/v1/trainings/{id}/metrics` | GET | Get metrics history |
| `/api/v1/trainings/{id}/checkpoints` | GET | List checkpoints |
| `/api/v1/training-templates` | GET/POST | Template CRUD |
| `/api/v1/training-templates/{id}` | GET/PUT/DELETE | Template by ID |

---

## 7. Data Model

### 7.1 Training Table
```sql
CREATE TABLE trainings (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    model_id UUID REFERENCES models(id),
    dataset_id UUID REFERENCES datasets(id),
    tokenization_id UUID REFERENCES dataset_tokenizations(id),
    layer INTEGER NOT NULL,
    architecture VARCHAR(50),  -- standard, jumprelu, skip, transcoder
    hyperparameters JSONB NOT NULL,
    status VARCHAR(50),  -- pending, running, completed, failed, stopped
    current_step INTEGER DEFAULT 0,
    final_loss FLOAT,
    sae_path VARCHAR(500),
    error_message TEXT,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

### 7.2 TrainingMetric Table
```sql
CREATE TABLE training_metrics (
    id UUID PRIMARY KEY,
    training_id UUID REFERENCES trainings(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    loss FLOAT,
    l0 FLOAT,
    l1 FLOAT,
    reconstruction_loss FLOAT,
    fvu FLOAT,
    dead_neurons INTEGER,
    created_at TIMESTAMP,
    UNIQUE(training_id, step)
);
```

### 7.3 TrainingTemplate Table
```sql
CREATE TABLE training_templates (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    hyperparameters JSONB NOT NULL,
    is_favorite BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## 8. WebSocket Channels

| Channel | Events | Payload |
|---------|--------|---------|
| `training/{id}` | `progress` | `{step, loss, l0, fvu, progress_pct}` |
| `training/{id}` | `completed` | `{sae_path, final_loss}` |
| `training/{id}` | `failed` | `{error, step}` |
| `training/{id}` | `checkpoint` | `{checkpoint_path, step}` |

---

## 9. Key Files

### Backend
- `backend/src/services/training_service.py` - Training orchestration
- `backend/src/ml/sparse_autoencoder.py` - Standard SAE implementation
- `backend/src/ml/jumprelu_sae.py` - JumpReLU implementation
- `backend/src/workers/training_tasks.py` - Celery training task
- `backend/src/api/v1/endpoints/trainings.py` - API routes

### Frontend
- `frontend/src/components/panels/TrainingPanel.tsx` - Main panel
- `frontend/src/components/training/TrainingCard.tsx` - Card with charts
- `frontend/src/components/training/StartTrainingModal.tsx` - Config modal
- `frontend/src/components/trainingTemplates/` - Template components
- `frontend/src/stores/trainingsStore.ts` - State management

---

## 10. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| Dataset Management | Requires tokenized dataset |
| Model Management | Requires model for activation extraction |
| SAE Management | Creates SAE records |
| Feature Discovery | Provides trained SAE |

---

## 11. Testing Checklist

- [x] Start Standard SAE training
- [x] Start JumpReLU training
- [x] Real-time metrics streaming
- [x] Stop training gracefully
- [x] Checkpoint creation
- [x] Delete training
- [x] Retry failed training
- [x] Template save/load
- [x] Template import/export
- [x] Hyperparameters modal

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/003_FTDD|SAE_Training.md) | [TID](../tids/003_FTID|SAE_Training.md)*
