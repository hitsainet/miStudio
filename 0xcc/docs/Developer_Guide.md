# miStudio Developer Guide

**MechInterp Studio (miStudio)** - An open-source platform for Sparse Autoencoder (SAE) training, feature discovery, and model steering.

**Version:** MVP (December 2025)
**Total Commits:** 444+ since October 2025
**Database Migrations:** 53

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Features](#core-features)
4. [Technology Stack](#technology-stack)
5. [Database Schema](#database-schema)
6. [ML Pipeline](#ml-pipeline)
7. [External Integrations](#external-integrations)
8. [Real-time Communication](#real-time-communication)
9. [API Reference](#api-reference)
10. [Development Setup](#development-setup)
11. [Testing](#testing)

---

## Project Overview

### Vision
miStudio is a comprehensive mechanistic interpretability workbench that enables researchers to:
- Train Sparse Autoencoders (SAEs) on transformer model activations
- Extract and analyze interpretable features from trained SAEs
- Apply feature-based steering to control model behavior
- Export findings to Neuronpedia for community sharing

### Goals
1. **Accessibility**: Make SAE research accessible to researchers without ML infrastructure expertise
2. **End-to-End Workflow**: Support the complete interpretability pipeline from data to steering
3. **Interoperability**: Compatible with HuggingFace, Neuronpedia, and SAELens ecosystems
4. **Real-time Feedback**: Provide immediate progress updates for long-running operations

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐ │
│  │Datasets │ Models  │Training │Features │  SAEs   │Steering │ │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘ │
│       │         │         │         │         │         │       │
│  ┌────┴─────────┴─────────┴─────────┴─────────┴─────────┴────┐ │
│  │              Zustand Stores + WebSocket Hooks              │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
└────────────────────────────┼────────────────────────────────────┘
                             │ HTTP/WebSocket
┌────────────────────────────┼────────────────────────────────────┐
│                         Nginx                                   │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                    Backend (FastAPI)                            │
│  ┌─────────────────────────┴─────────────────────────────────┐ │
│  │                     API Endpoints                          │ │
│  │  /datasets  /models  /trainings  /features  /saes  /steering│
│  └────────────────────────┬──────────────────────────────────┘ │
│                           │                                     │
│  ┌────────────────────────┴──────────────────────────────────┐ │
│  │                    Service Layer                           │ │
│  │  DatasetService  ModelService  TrainingService  ...        │ │
│  └────────────────────────┬──────────────────────────────────┘ │
│                           │                                     │
│  ┌────────────────────────┴──────────────────────────────────┐ │
│  │                     ML Pipeline                            │ │
│  │  SparseAutoencoder  JumpReLUSAE  SteeringService          │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────┐                    ┌─────────────────────────┐
│   PostgreSQL    │                    │     Celery Workers      │
│  ┌───────────┐  │                    │  ┌─────────────────┐   │
│  │  Models   │  │                    │  │ Training Tasks  │   │
│  │ Datasets  │  │                    │  │ Extraction Tasks│   │
│  │ Trainings │  │                    │  │ Labeling Tasks  │   │
│  │ Features  │  │                    │  │ Download Tasks  │   │
│  │   SAEs    │  │                    │  │ Monitoring      │   │
│  └───────────┘  │                    │  └─────────────────┘   │
└─────────────────┘                    └───────────┬─────────────┘
                                                   │
                                       ┌───────────┴───────────┐
                                       │        Redis          │
                                       │  (Message Broker +    │
                                       │   Result Backend)     │
                                       └───────────────────────┘
```

### Directory Structure

```
miStudio/
├── backend/
│   ├── src/
│   │   ├── api/v1/endpoints/     # REST API endpoints (17 modules)
│   │   ├── core/                 # Config, database, Celery app
│   │   ├── models/               # SQLAlchemy models (21 models)
│   │   ├── schemas/              # Pydantic schemas (17 modules)
│   │   ├── services/             # Business logic (30+ services)
│   │   ├── workers/              # Celery tasks (11 modules)
│   │   ├── ml/                   # ML implementations
│   │   └── utils/                # Utilities
│   ├── alembic/versions/         # Database migrations (53)
│   └── data/                     # Local storage
├── frontend/
│   ├── src/
│   │   ├── api/                  # API client modules (17)
│   │   ├── components/           # React components (14 directories)
│   │   │   ├── panels/           # Main navigation panels
│   │   │   ├── features/         # Feature discovery components
│   │   │   ├── steering/         # Model steering UI
│   │   │   ├── training/         # SAE training components
│   │   │   └── ...
│   │   ├── stores/               # Zustand state stores (17)
│   │   ├── hooks/                # React hooks (15, mostly WebSocket)
│   │   ├── types/                # TypeScript definitions (14)
│   │   └── utils/                # Frontend utilities
└── 0xcc/                         # Project documentation framework
```

---

## Core Features

### 1. Dataset Management
**Purpose**: Ingest and prepare training data for SAE training.

**Capabilities**:
- Download datasets from HuggingFace Hub
- Tokenize with configurable parameters
- Multi-tokenization support (different vocab sizes, filters)
- Token filtering (minimum length, special tokens, stop words)
- Statistics visualization (vocabulary distribution, sequence lengths)
- Sample browser with pagination

**Key Components**:
- `DatasetService`, `TokenizationService`
- `DatasetsPanel`, `DatasetCard`, `DownloadForm`
- WebSocket progress tracking for downloads/tokenization

### 2. Model Management
**Purpose**: Download and manage transformer models for analysis.

**Capabilities**:
- Download models from HuggingFace Hub
- Support for gated models with authentication
- Quantization (4-bit, 8-bit via bitsandbytes)
- Model architecture viewer
- Memory estimation before download

**Key Components**:
- `ModelService`, `model_tasks.py`
- `ModelsPanel`, `ModelCard`, `ModelDownloadForm`

### 3. SAE Training
**Purpose**: Train Sparse Autoencoders on model activations.

**SAE Architectures**:
| Architecture | Description | Use Case |
|-------------|-------------|----------|
| `Standard` | Classic SAE with L1 sparsity | General feature learning |
| `JumpReLU` | Gemma Scope-style with L0 penalty | State-of-the-art sparsity |
| `Skip` | Residual connections | Better reconstruction |
| `Transcoder` | Layer-to-layer | Activation transcoding |

**Training Features**:
- Real-time metrics (loss, L0, reconstruction error, FVU)
- Checkpoint management with auto-save
- Dead neuron detection and resampling
- Top-K sparsity for guaranteed sparsity levels
- Training templates for reproducibility
- Hyperparameter optimization hints

**Key Hyperparameters**:
```python
{
    "hidden_dim": 2304,        # Model hidden size
    "latent_dim": 18432,       # SAE width (8x expansion)
    "l1_alpha": 0.001,         # L1 penalty coefficient
    "learning_rate": 1e-4,
    "batch_size": 4096,
    "num_steps": 100000,
    "normalize_activations": "constant_norm_rescale",
    "top_k_sparsity": null,    # Optional Top-K (percentage)
}
```

**Key Components**:
- `TrainingService`, `SparseAutoencoder`, `JumpReLUSAE`
- `TrainingPanel`, `TrainingCard`, `StartTrainingModal`
- WebSocket progress streaming

### 4. Feature Discovery
**Purpose**: Extract and analyze interpretable features from trained SAEs.

**Capabilities**:
- Batch extraction with GPU optimization
- Activation statistics (frequency, max, mean)
- Interpretability scoring
- Context window capture (before/after tokens)
- Token filtering during extraction
- Example export to JSON

**Dual Labeling System**:
1. **Semantic Labels**: Human-readable descriptions
2. **Category Labels**: Taxonomy classification

**Auto-Labeling**:
- GPT-4o integration via OpenAI API
- Configurable prompt templates
- Confidence scoring

**Key Components**:
- `ExtractionService`, `FeatureService`, `LabelingService`
- `FeaturesPanel`, `FeatureDetailModal`, `TokenHighlight`
- Extraction templates for reproducibility

### 5. SAE Management
**Purpose**: Manage both trained and external SAEs.

**Sources**:
- **Trained**: SAEs trained within miStudio
- **HuggingFace**: Download from model hub
- **Gemma Scope**: Pre-trained Google SAEs

**Format Support**:
- Community Standard (SAELens-compatible)
- miStudio native format
- Automatic format detection and conversion

**Key Components**:
- `SAEManagerService`, `HuggingFaceSAEService`, `SAEConverter`
- `SAEsPanel`, `SAECard`, `DownloadFromHF`

### 6. Model Steering
**Purpose**: Control model behavior via feature interventions.

**Steering Types**:
- **Activation**: Add/subtract feature directions
- **Suppression**: Reduce specific feature activations

**Capabilities**:
- Multi-feature steering (combine multiple interventions)
- Strength sweep (test multiple intensities)
- Comparison mode (steered vs. unsteered)
- Neuronpedia-compatible calibration
- Prompt templates for experiments
- Export results for analysis

**Key Components**:
- `SteeringService`, `forward_hooks.py`
- `SteeringPanel`, `FeatureBrowser`, `ComparisonResults`

### 7. Neuronpedia Export
**Purpose**: Share SAE findings with the research community.

**Export Contents**:
- Feature activation examples
- Logit lens data (promoted/suppressed tokens)
- Activation histograms
- Explanations/labels
- SAELens-compatible weights

**Output Format**:
```
export.zip/
├── metadata.json           # SAE configuration
├── README.md               # Documentation
├── features/               # Individual feature JSONs
│   ├── 0.json
│   ├── 1.json
│   └── ...
├── explanations/
│   └── explanations.json   # Feature labels
└── saelens/
    ├── cfg.json            # SAELens config
    └── sae_weights.safetensors
```

**Key Components**:
- `NeuronpediaExportService`, `LogitLensService`
- `ExportToNeuronpedia` modal

### 8. System Monitoring
**Purpose**: Track resource utilization during operations.

**Metrics**:
- GPU utilization, memory, temperature, power
- CPU per-core utilization
- RAM and swap usage
- Disk I/O rates
- Network I/O rates

**Key Components**:
- `SystemMonitorService`, `system_monitor_tasks.py`
- `SystemMonitor` dashboard
- WebSocket streaming (2-second intervals)

---

## Technology Stack

### Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| FastAPI | 0.100+ | REST API framework |
| PostgreSQL | 14+ | Primary database |
| Redis | 7+ | Message broker & cache |
| Celery | 5.x | Distributed task queue |
| SQLAlchemy | 2.0 | ORM with async support |
| Alembic | 1.x | Database migrations |
| PyTorch | 2.0+ | ML framework |
| Transformers | 4.x | HuggingFace models |
| bitsandbytes | 0.41+ | Quantization |
| Socket.IO | 5.x | WebSocket server |

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18+ | UI framework |
| TypeScript | 5.x | Type safety |
| Vite | 5.x | Build tool |
| Zustand | 4.x | State management |
| Tailwind CSS | 3.x | Styling |
| Recharts | 2.x | Charting |
| Lucide React | - | Icons |
| Socket.IO Client | 4.x | WebSocket client |

### Infrastructure
| Technology | Purpose |
|-----------|---------|
| Docker Compose | Development environment |
| Nginx | Reverse proxy |
| Celery Beat | Scheduled tasks |

---

## Database Schema

### Core Entities

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Dataset     │     │      Model      │     │    Training     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id              │     │ id              │     │ id              │
│ name            │     │ name            │     │ name            │
│ repo_id         │     │ repo_id         │     │ model_id (FK)   │
│ file_path       │     │ file_path       │     │ dataset_id (FK) │
│ status          │     │ status          │     │ status          │
│ metadata        │     │ architecture    │     │ hyperparameters │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
         │                                                │
         │ 1:N                                           │ 1:N
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│DatasetTokenization│                            │ TrainingMetric  │
├─────────────────┤                              ├─────────────────┤
│ id              │                              │ id              │
│ dataset_id (FK) │                              │ training_id (FK)│
│ tokenizer_name  │                              │ step            │
│ num_tokens      │                              │ loss, l0, l1    │
│ vocab_size      │                              │ fvu             │
└─────────────────┘                              └─────────────────┘
```

### Feature Entities

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ExternalSAE   │     │     Feature     │     │FeatureActivation│
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id              │     │ id              │     │ id              │
│ name            │     │ neuron_index    │     │ feature_id (FK) │
│ model_name      │     │ training_id(FK) │     │ activation_value│
│ layer           │     │ external_sae_id │     │ token           │
│ n_features      │     │ label           │     │ context_before  │
│ architecture    │     │ category        │     │ context_after   │
│ local_path      │     │ statistics      │     │ position        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │
         │ 1:N                   │ 1:1
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  ExtractionJob  │     │FeatureDashboard │
├─────────────────┤     │     Data        │
│ id              │     ├─────────────────┤
│ external_sae_id │     │ feature_id (FK) │
│ status          │     │ logit_lens_data │
│ progress        │     │ histogram_data  │
│ features_found  │     │ top_tokens      │
└─────────────────┘     └─────────────────┘
```

### Template Entities

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│TrainingTemplate │  │ExtractionTemplate│ │PromptTemplate   │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ id, name        │  │ id, name        │  │ id, name        │
│ hyperparameters │  │ config          │  │ template_type   │
│ is_favorite     │  │ filters         │  │ content         │
│ usage_count     │  │ is_favorite     │  │ variables       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## ML Pipeline

### SAE Training Pipeline

```
1. Data Preparation
   Dataset → Tokenization → Activation Buffer

2. Activation Extraction
   Model(tokens) → hook_resid_post → activations[batch, seq, hidden]

3. SAE Training Loop
   for batch in activations:
       # Normalize
       x_norm = normalize(x)

       # Encode
       z = ReLU(W_enc @ x_norm + b_enc)  # Latent features

       # Top-K (optional)
       if top_k:
           z = topk_mask(z, k)

       # Decode
       x_hat = W_dec @ z + b_dec

       # Loss
       L_recon = MSE(x_hat, x_norm)
       L_sparse = l1_alpha * L1(z)  # or L0 for JumpReLU
       loss = L_recon + L_sparse

       # Update
       optimizer.step()

4. Checkpoint Saving
   Save: weights, optimizer, step, config
```

### Feature Extraction Pipeline

```
1. Load trained SAE
   sae = load_sae_auto_detect(path)

2. Extract activations from model
   for batch in dataset:
       with register_hook(model, layer):
           outputs = model(batch)
           activations = hook.activations

3. Encode through SAE
   features = sae.encode(activations)  # [batch, seq, n_features]

4. Find top activations per feature
   for feature_idx in range(n_features):
       top_k = topk(features[:, :, feature_idx], k=100)
       store_activations(feature_idx, top_k, tokens)

5. Compute statistics
   frequency = count_nonzero(features) / total
   max_val = features.max()
   interpretability = compute_interpretability_score(features)
```

### Model Steering Pipeline

```
1. Load SAE and Model
   sae = load_sae(path)
   model = load_model(name)

2. Register Steering Hook
   def steering_hook(module, input, output):
       # Decode to feature space
       features = sae.encode(output)

       # Modify feature activation
       features[:, :, target_idx] *= (1 + strength)

       # Re-encode to residual stream
       modified = sae.decode(features)
       return modified

   handle = model.layers[layer].register_forward_hook(steering_hook)

3. Generate
   steered_output = model.generate(prompt)

4. Compare
   baseline_output = model.generate(prompt)  # Without hook
   compute_metrics(steered_output, baseline_output)
```

---

## External Integrations

### HuggingFace Hub

**Datasets**:
```python
from datasets import load_dataset
dataset = load_dataset(repo_id, split=split, streaming=True)
```

**Models**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
```

**SAEs**:
```python
from huggingface_hub import hf_hub_download
files = hf_hub_download(repo_id, filename="*.safetensors")
```

### Neuronpedia

**Export Format Compatibility**:
- JSON format matches Neuronpedia upload spec
- SAELens-compatible cfg.json and weights
- Feature explanations in standardized schema

**Logit Lens Integration**:
```python
# Compute promoted/suppressed tokens
logits = sae.decoder.weight @ model.lm_head.weight.T
top_positive = topk(logits, k=20)
top_negative = topk(-logits, k=20)
```

### OpenAI API (Auto-labeling)

```python
from openai import OpenAI
client = OpenAI(api_key=settings.openai_api_key)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": labeling_prompt},
        {"role": "user", "content": feature_examples}
    ]
)
```

---

## Real-time Communication

### WebSocket Architecture

**Pattern**: Socket.IO with room-based pub/sub

**Channel Naming Convention**:
```
{entity_type}/{entity_id}/{event_type}
```

**Implemented Channels**:

| Channel | Events | Purpose |
|---------|--------|---------|
| `training/{id}` | progress, completed, failed | Training job updates |
| `extraction/{id}` | progress, completed, failed | Extraction updates |
| `model/{id}` | download_progress, completed | Model downloads |
| `dataset/{id}` | progress, completed, failed | Dataset operations |
| `labeling/{id}` | progress, results | Auto-labeling progress |
| `system/gpu/{id}` | metrics | GPU monitoring |
| `system/cpu` | metrics | CPU monitoring |
| `system/memory` | metrics | Memory monitoring |

**Backend Emission** (`websocket_emitter.py`):
```python
async def emit_training_progress(training_id: str, progress: dict):
    await emit_to_channel(f"training/{training_id}", "progress", progress)
```

**Frontend Subscription** (React hooks):
```typescript
export function useTrainingWebSocket(trainingId: string) {
  useEffect(() => {
    socket.emit('subscribe', `training/${trainingId}`);
    socket.on('progress', handleProgress);
    return () => socket.emit('unsubscribe', `training/${trainingId}`);
  }, [trainingId]);
}
```

**Fallback Pattern**:
- WebSocket disconnection triggers HTTP polling
- Automatic reconnection stops polling
- Managed in Zustand stores

---

## API Reference

### Endpoints Overview

| Prefix | Purpose | Key Operations |
|--------|---------|----------------|
| `/api/v1/datasets` | Dataset management | CRUD, download, tokenize |
| `/api/v1/models` | Model management | CRUD, download |
| `/api/v1/trainings` | SAE training | Create, monitor, checkpoints |
| `/api/v1/features` | Feature discovery | Query, update labels, search |
| `/api/v1/saes` | SAE management | CRUD, download, convert |
| `/api/v1/steering` | Model steering | Compare, sweep, generate |
| `/api/v1/neuronpedia` | Export | Start export, status, download |
| `/api/v1/system` | System monitoring | Metrics, GPU info |
| `/api/v1/task-queue` | Task management | List, cancel, cleanup |

### Key API Patterns

**Async Operations**:
```python
# Start async job
POST /api/v1/trainings
Response: {"id": "...", "status": "pending"}

# Poll status
GET /api/v1/trainings/{id}
Response: {"id": "...", "status": "running", "progress": 45.2}

# Or subscribe via WebSocket
socket.emit('subscribe', 'training/{id}')
```

**Pagination**:
```
GET /api/v1/features?page=1&limit=50&training_id=xxx
Response: {
  "items": [...],
  "total": 1000,
  "page": 1,
  "limit": 50
}
```

**Filtering**:
```
GET /api/v1/features?search=love&min_activation=0.5&has_label=true
```

---

## Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker and Docker Compose
- CUDA-capable GPU (recommended)

### Quick Start

```bash
# 1. Clone repository
git clone <repo-url>
cd miStudio

# 2. Add domain to hosts
sudo bash -c 'echo "127.0.0.1 mistudio.mcslab.io" >> /etc/hosts'

# 3. Start all services
./start-mistudio.sh

# Access at http://mistudio.mcslab.io
```

### Service Management

```bash
# Start all
./start-mistudio.sh

# Stop all
./stop-mistudio.sh

# Individual services
docker compose up -d          # Infrastructure (postgres, redis, nginx)
cd backend && source venv/bin/activate
python -m uvicorn src.main:app --reload  # Backend
cd frontend && npm run dev    # Frontend
celery -A src.core.celery_app worker     # Celery worker
```

### Environment Variables

```bash
# Backend (.env)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/mistudio
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=sk-...          # For auto-labeling
HF_TOKEN=hf_...                # For gated models
DATA_DIR=/path/to/data

# Frontend
VITE_API_URL=                  # Empty for same-origin
```

---

## Testing

### Backend Tests

```bash
cd backend
source venv/bin/activate
pytest                                    # All tests
pytest tests/unit/                        # Unit tests only
pytest tests/integration/                 # Integration tests
pytest --cov=src --cov-report=html       # With coverage
```

### Frontend Tests

```bash
cd frontend
npm test                                  # Run Vitest
npm run test:coverage                     # With coverage
```

### Test Structure

```
backend/tests/
├── unit/
│   ├── test_sparse_autoencoder.py
│   ├── test_extraction_service.py
│   └── ...
└── integration/
    ├── test_training_flow.py
    └── ...

frontend/src/
├── **/*.test.ts               # Co-located tests
├── stores/*.test.ts
└── hooks/*.test.ts
```

---

## Key Decisions & Design Patterns

### 1. WebSocket-First for Progress
All long-running operations emit progress via WebSocket for immediate UI feedback.

### 2. Celery for Background Tasks
CPU/GPU-intensive operations (training, extraction, downloads) run in Celery workers to avoid blocking the API.

### 3. Community Standard Format
SAEs are stored in SAELens-compatible format for ecosystem interoperability.

### 4. Graceful Degradation
Optional operations (logit lens, histograms) fail gracefully without blocking core functionality.

### 5. Multi-Architecture Support
Single codebase supports Standard, JumpReLU, Skip, and Transcoder SAE architectures.

---

## Feature Evolution Timeline

Based on git commit history (444+ commits):

| Phase | Focus | Key Commits |
|-------|-------|-------------|
| Oct 2025 | Foundation | Backend/frontend infrastructure, dataset management |
| Early Nov | Training | SAE training, checkpoints, metrics visualization |
| Mid Nov | Features | Feature extraction, labeling, dual-label system |
| Late Nov | SAEs | External SAE support, Gemma Scope, format conversion |
| Early Dec | Steering | Feature steering, comparison, Neuronpedia export |

---

## Contributing

### Code Style
- **Python**: Black formatter, Ruff linter, Google docstrings
- **TypeScript**: Prettier, ESLint (Airbnb config)

### Commit Convention
```
type(scope): description

Types: feat, fix, docs, style, refactor, test, chore
Example: feat(steering): add multi-strength testing
```

### Branch Strategy
- `main`: Production-ready
- `develop`: Integration branch
- `feature/*`: New features
- `bugfix/*`: Bug fixes

---

## Resources

- **Project PRD**: `0xcc/prds/000_PPRD|miStudio.md`
- **Architecture Decision Record**: `0xcc/adrs/000_PADR|miStudio.md`
- **Mock UI Reference**: `0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
- **API Documentation**: http://localhost:8000/docs (Swagger UI)

---

*Generated: December 2025*
*miStudio MVP - MechInterp Studio*
