# Project PRD: MechInterp Studio (miStudio)

**Document ID:** 000_PPRD|miStudio
**Version:** 2.1 (Post-MVP Enhancements)
**Last Updated:** 2025-12-16
**Status:** Active

---

## Executive Summary

MechInterp Studio (miStudio) is an open-source platform for Sparse Autoencoder (SAE) research that provides an end-to-end workflow for training SAEs, discovering interpretable features, and applying feature-based steering to transformer models. The MVP delivers 8 fully implemented features with a 9th (Multi-GPU Scalability) planned for post-MVP enhancement.

### Project Metrics (as of December 2025)
- **Total Commits:** 485+
- **Database Migrations:** 52
- **Backend Services:** 35+
- **Frontend Components:** 85+
- **Development Period:** October - December 2025

---

## 1. Vision & Goals

### 1.1 Vision
Democratize mechanistic interpretability research by providing a comprehensive, user-friendly workbench that enables researchers to train SAEs, discover features, and steer model behavior without requiring ML infrastructure expertise.

### 1.2 Goals
| Goal | Description | Status |
|------|-------------|--------|
| **Accessibility** | Make SAE research accessible to researchers without DevOps expertise | Achieved |
| **End-to-End Workflow** | Support complete pipeline from data to steering | Achieved |
| **Interoperability** | Compatible with HuggingFace, Neuronpedia, SAELens | Achieved |
| **Real-time Feedback** | Immediate progress updates for long-running operations | Achieved |
| **Scalability** | Multi-GPU support with aggregated/per-GPU monitoring | Planned |

### 1.3 Success Criteria
- [x] Users can download datasets from HuggingFace and tokenize them
- [x] Users can download and quantize models from HuggingFace
- [x] Users can train SAEs with multiple architectures (Standard, JumpReLU, Skip, Transcoder)
- [x] Users can extract and browse interpretable features
- [x] Users can apply feature-based steering with comparison mode
- [x] Users can export to Neuronpedia-compatible format
- [x] All long-running operations provide real-time progress via WebSocket
- [ ] Users can distribute training across multiple GPUs (planned)

---

## 2. Feature Inventory

### 2.1 MVP Features (Implemented)

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | Dataset Management | HuggingFace download, tokenization, statistics | Complete |
| 2 | Model Management | Model download, quantization, architecture viewer | Complete |
| 3 | SAE Training | Multi-architecture training with real-time metrics | Complete |
| 4 | Feature Discovery | Extraction, labeling, auto-labeling, search | Complete |
| 5 | SAE Management | Trained & external SAE management, format conversion | Complete |
| 6 | Model Steering | Feature interventions, comparison, export | Complete |
| 7 | Neuronpedia Export | Community-format export with logit lens data | Complete |
| 8 | System Monitoring | GPU/CPU/Memory/Disk/Network monitoring | Complete |
| 9 | Multi-GPU Scalability | Distributed training, aggregated monitoring | Planned |

### 2.2 Template Systems (Sub-features)
Templates are documented within their parent features:

| Template Type | Parent Feature | Purpose |
|---------------|----------------|---------|
| Training Templates | SAE Training | Save/load training configurations |
| Extraction Templates | Feature Discovery | Save/load extraction configurations |
| Labeling Prompt Templates | Feature Discovery | Customize auto-labeling prompts |
| Prompt Templates | Model Steering | Save/load steering experiment prompts |

---

## 3. Feature Details

### 3.1 Dataset Management

**Purpose:** Ingest and prepare training data for SAE training.

**Capabilities:**
- Download datasets from HuggingFace Hub
- Tokenize with configurable parameters (max_length, stride, truncation)
- Multi-tokenization support (create multiple tokenizations per dataset)
- Token filtering (minimum length, special tokens, stop words)
- Statistics visualization (vocabulary distribution, sequence lengths)
- Sample browser with pagination
- Real-time progress via WebSocket

**Key Files:**
- Backend: `dataset_service.py`, `tokenization_service.py`, `dataset_tasks.py`
- Frontend: `DatasetsPanel.tsx`, `DownloadForm.tsx`, `TokenizationStatsModal.tsx`

**API Endpoints:**
- `GET/POST /api/v1/datasets` - CRUD operations
- `POST /api/v1/datasets/{id}/download` - Start download
- `POST /api/v1/datasets/{id}/tokenize` - Start tokenization
- `GET /api/v1/datasets/{id}/statistics` - Get tokenization stats

---

### 3.2 Model Management

**Purpose:** Download and manage transformer models for analysis.

**Capabilities:**
- Download models from HuggingFace Hub
- Support for gated models with HF token authentication
- Quantization options (4-bit, 8-bit via bitsandbytes)
- Model architecture viewer (layers, parameters)
- Memory estimation before download
- Real-time download progress via WebSocket

**Key Files:**
- Backend: `model_service.py`, `model_tasks.py`
- Frontend: `ModelsPanel.tsx`, `ModelDownloadForm.tsx`, `ModelPreviewModal.tsx`

**API Endpoints:**
- `GET/POST /api/v1/models` - CRUD operations
- `POST /api/v1/models/{id}/download` - Start download
- `GET /api/v1/models/{id}/architecture` - Get architecture info

---

### 3.3 SAE Training

**Purpose:** Train Sparse Autoencoders on model activations.

**SAE Architectures:**
| Architecture | Description | Key Feature |
|-------------|-------------|-------------|
| Standard | Classic SAE with L1 sparsity | General purpose |
| JumpReLU | Gemma Scope-style with L0 penalty | State-of-the-art sparsity |
| Skip | Residual connections | Better reconstruction |
| Transcoder | Layer-to-layer mapping | Activation transcoding |

**Capabilities:**
- Real-time metrics streaming (loss, L0, L1, reconstruction error, FVU)
- Checkpoint management with configurable intervals
- Dead neuron detection and optional resampling
- Top-K sparsity for guaranteed sparsity levels
- Training templates for reproducibility
- Retry failed trainings with same configuration
- Bulk delete for cleanup

**Training Hyperparameters:**
```
hidden_dim, latent_dim, l1_alpha, learning_rate, batch_size,
num_steps, normalize_activations, top_k_sparsity, checkpoint_interval,
dead_neuron_threshold, resample_steps
```

**Key Files:**
- Backend: `training_service.py`, `sparse_autoencoder.py`, `jumprelu_sae.py`, `training_tasks.py`
- Frontend: `TrainingPanel.tsx`, `StartTrainingModal.tsx`, `TrainingCard.tsx`
- Templates: `TrainingTemplatesPanel.tsx`, `TrainingTemplateForm.tsx`

**API Endpoints:**
- `GET/POST /api/v1/trainings` - CRUD operations
- `GET /api/v1/trainings/{id}/metrics` - Get training metrics
- `GET /api/v1/trainings/{id}/checkpoints` - List checkpoints
- `POST /api/v1/trainings/{id}/stop` - Stop training
- `POST /api/v1/trainings/{id}/retry` - Retry failed training

---

### 3.4 Feature Discovery

**Purpose:** Extract and analyze interpretable features from trained SAEs.

**Capabilities:**
- Batch extraction with GPU optimization
- Activation statistics (frequency, max, mean, interpretability score)
- Context window capture (tokens before/after activation)
- Token filtering during extraction
- Feature search by label, category, statistics
- Example export to JSON
- NLP analysis of top-activating tokens
- BPE token reconstruction for human-readable text

**NLP Analysis (Added Dec 2025):**
- Automatic linguistic analysis of top-activating tokens
- POS tagging, lemmatization, semantic grouping
- Named entity recognition for pattern detection
- Integration with spaCy for linguistic processing

**BPE Reconstruction:**
- Merge adjacent BPE tokens for readability
- Display both raw tokens and reconstructed text
- Context-aware token grouping

**Dual Labeling System:**
1. **Semantic Labels:** Human-readable descriptions (e.g., "love and affection")
2. **Category Labels:** Taxonomy classification (e.g., "Emotion > Positive")

**Auto-Labeling:**
- GPT-4o integration via OpenAI API
- Local LLM support via Ollama (Added Dec 2025)
- Configurable prompt templates (Labeling Prompt Templates)
- Confidence scoring
- Batch processing with progress tracking

**Extraction Templates:**
- Save extraction configurations for reproducibility
- Filter settings: activation threshold, context window, token filters
- Favorites and usage tracking

**Key Files:**
- Backend: `extraction_service.py`, `feature_service.py`, `labeling_service.py`, `openai_labeling_service.py`
- Frontend: `FeaturesPanel.tsx`, `FeatureDetailModal.tsx`, `StartExtractionModal.tsx`
- Templates: `ExtractionTemplatesPanel.tsx`, `LabelingPromptTemplatesPanel.tsx`

**API Endpoints:**
- `GET /api/v1/features` - List features with filtering
- `PATCH /api/v1/features/{id}` - Update labels
- `POST /api/v1/features/extraction` - Start extraction job
- `POST /api/v1/features/labeling` - Start auto-labeling job

---

### 3.5 SAE Management

**Purpose:** Manage both trained and external SAEs.

**SAE Sources:**
- **Trained:** SAEs trained within miStudio (linked to training record)
- **HuggingFace:** Download from model hub
- **Gemma Scope:** Pre-trained Google SAEs (special download flow)

**Format Support:**
- **Community Standard:** SAELens-compatible (cfg.json + sae_weights.safetensors)
- **miStudio Native:** Internal format with extended metadata
- Automatic format detection and conversion

**Capabilities:**
- List all SAEs with filtering by source, model, layer
- Download SAEs from HuggingFace
- Convert between formats
- Link/unlink from training records
- View SAE configuration and statistics

**Key Files:**
- Backend: `sae_manager_service.py`, `huggingface_sae_service.py`, `sae_converter.py`
- Frontend: `SAEsPanel.tsx`, `SAECard.tsx`, `DownloadFromHF.tsx`

**API Endpoints:**
- `GET/POST /api/v1/saes` - CRUD operations
- `POST /api/v1/saes/download-hf` - Download from HuggingFace
- `POST /api/v1/saes/{id}/convert` - Convert format

---

### 3.6 Model Steering

**Purpose:** Control model behavior via feature interventions.

**Steering Types:**
- **Activation:** Add/subtract feature directions to the residual stream
- **Suppression:** Reduce specific feature activations toward zero

**Capabilities:**
- Multi-feature selection (select multiple features for steering)
- Combined multi-feature generation (apply all features in single pass) [Planned]
- Strength sweep (test multiple intensities in one run)
- Comparison mode (steered vs. unsteered side-by-side)
- Neuronpedia-compatible calibration
- Prompt templates for repeatable experiments
- Export results to JSON

**Prompt Templates:**
- Save prompts for steering experiments
- Variable substitution for batch testing
- Favorites and organization

**Key Files:**
- Backend: `steering_service.py`, `forward_hooks.py`
- Frontend: `SteeringPanel.tsx`, `FeatureBrowser.tsx`, `ComparisonResults.tsx`, `SelectedFeatureCard.tsx`
- Templates: `PromptTemplatesPanel.tsx`, `PromptListEditor.tsx`

**API Endpoints:**
- `POST /api/v1/steering/generate` - Generate with steering
- `POST /api/v1/steering/compare` - Compare steered vs. baseline
- `POST /api/v1/steering/sweep` - Multi-strength test
- `POST /api/v1/steering/combined` - Combined multi-feature generation [Planned]

---

### 3.7 Neuronpedia Export

**Purpose:** Share SAE findings with the research community.

**Export Contents:**
- Feature activation examples (top activating tokens with context)
- Logit lens data (promoted/suppressed tokens per feature)
- Activation histograms
- Feature explanations/labels
- SAELens-compatible weights

**Output Format:**
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

**Configuration Options:**
- Feature selection: all, extracted only, custom indices
- Include/exclude: logit lens, histograms, top tokens, explanations
- SAELens format inclusion

**Key Files:**
- Backend: `neuronpedia_export_service.py`, `logit_lens_service.py`
- Frontend: `ExportToNeuronpedia.tsx`

**API Endpoints:**
- `POST /api/v1/neuronpedia/export` - Start export job
- `GET /api/v1/neuronpedia/export/{id}` - Get job status
- `GET /api/v1/neuronpedia/export/{id}/download` - Download archive

---

### 3.8 System Monitoring

**Purpose:** Track resource utilization during operations.

**Metrics:**
| Category | Metrics |
|----------|---------|
| GPU | Utilization %, memory used/total, temperature, power draw |
| CPU | Per-core utilization % |
| Memory | RAM used/total, swap used/total |
| Disk | Read/write I/O rates (MB/s) |
| Network | Upload/download I/O rates (MB/s) |

**Implementation:**
- WebSocket streaming (2-second intervals via Celery Beat)
- Fallback to HTTP polling on WebSocket disconnect
- 1-hour rolling history with chart visualization
- Combined GPU utilization + temperature chart

**Key Files:**
- Backend: `system_monitor_service.py`, `system_monitor_tasks.py`, `websocket_emitter.py`
- Frontend: `SystemMonitor.tsx`, `UtilizationChart.tsx`, `useSystemMonitorWebSocket.ts`

**API Endpoints:**
- `GET /api/v1/system/metrics` - Current metrics
- `GET /api/v1/system/history` - Historical data

**WebSocket Channels:**
- `system/gpu/{id}` - Per-GPU metrics
- `system/cpu` - CPU metrics
- `system/memory` - Memory metrics
- `system/disk` - Disk I/O
- `system/network` - Network I/O

---

### 3.9 Multi-GPU Scalability (Planned)

**Purpose:** Enable distributed training and enhanced multi-GPU monitoring.

**Planned Capabilities:**
- Distributed SAE training across multiple GPUs
- Data parallelism with gradient synchronization
- Aggregated vs. per-GPU monitoring toggle
- Separate meters for each GPU's VRAM and utilization
- GPU selection for training jobs

**Status:** Not implemented - planned for post-MVP

---

## 4. Technology Stack

### 4.1 Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| FastAPI | 0.100+ | REST API framework |
| PostgreSQL | 14+ | Primary database |
| Redis | 7+ | Message broker & cache |
| Celery | 5.x | Distributed task queue |
| SQLAlchemy | 2.0 | ORM with async support |
| Alembic | 1.x | Database migrations (53) |
| PyTorch | 2.0+ | ML framework |
| Transformers | 4.x | HuggingFace models |
| bitsandbytes | 0.41+ | Quantization |
| Socket.IO | 5.x | WebSocket server |

### 4.2 Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18+ | UI framework |
| TypeScript | 5.x | Type safety |
| Vite | 5.x | Build tool |
| Zustand | 4.x | State management |
| Tailwind CSS | 3.x | Styling (slate dark theme) |
| Recharts | 2.x | Data visualization |
| Lucide React | - | Icon library |
| Socket.IO Client | 4.x | WebSocket client |

### 4.3 Infrastructure
| Technology | Purpose |
|-----------|---------|
| Docker Compose | Development environment |
| Nginx | Reverse proxy |
| Celery Beat | Scheduled tasks (monitoring) |

---

## 5. Architecture Highlights

### 5.1 WebSocket-First Real-time Updates
All long-running operations emit progress via WebSocket for immediate UI feedback:
- Channel pattern: `{entity_type}/{entity_id}`
- Automatic fallback to HTTP polling on disconnect
- Celery tasks emit via internal HTTP endpoint

### 5.2 Celery Task Queue
Background processing for CPU/GPU-intensive operations:
- Queues: `default`, `sae`, `processing`
- Priority routing for training vs. extraction
- Celery Beat for periodic system monitoring

### 5.3 SAELens Compatibility
Community Standard format ensures interoperability:
- `cfg.json` with SAELens-compatible configuration
- `sae_weights.safetensors` for model weights
- Automatic format detection and conversion

---

## 6. External Integrations

| Integration | Purpose | Status |
|-------------|---------|--------|
| HuggingFace Hub | Dataset/model/SAE downloads | Complete |
| Neuronpedia | Export format compatibility | Complete |
| SAELens | Weight format compatibility | Complete |
| OpenAI API | GPT-4o auto-labeling | Complete |
| Ollama | Local LLM auto-labeling | Complete |
| spaCy | NLP analysis for features | Complete |

---

## 7. Data Storage

### 7.1 Database Schema
- **21 SQLAlchemy models** across core entities and templates
- **53 Alembic migrations** for schema evolution
- JSONB columns for flexible metadata storage

### 7.2 File Storage
- Local filesystem at configurable `DATA_DIR`
- Organized by entity type: `models/`, `datasets/`, `saes/`, `exports/`
- Safetensors format for model/SAE weights

---

## 8. Development & Deployment

### 8.1 Development Setup
```bash
# Add domain to hosts
sudo bash -c 'echo "127.0.0.1 mistudio.mcslab.io" >> /etc/hosts'

# Start all services
./start-mistudio.sh

# Access at http://mistudio.mcslab.io
```

### 8.2 Service Components
1. Docker Compose (PostgreSQL, Redis, Nginx)
2. Backend (FastAPI on port 8000)
3. Frontend (Vite on port 3000)
4. Celery Worker (background tasks)
5. Celery Beat (scheduled tasks)

---

## 9. Related Documents

| Document | Path | Description |
|----------|------|-------------|
| Architecture Decision Record | `0xcc/adrs/000_PADR\|miStudio.md` | Technical decisions |
| Developer Guide | `0xcc/docs/Developer_Guide.md` | Implementation details |
| Feature PRDs | `0xcc/prds/001-009_FPRD\|*.md` | Individual feature specs |
| Technical Design Docs | `0xcc/tdds/*.md` | Design specifications |
| Implementation Docs | `0xcc/tids/*.md` | Implementation guidance |
| Task Lists | `0xcc/tasks/*.md` | Development tracking |

---

## 10. Recent Infrastructure Improvements (Dec 2025)

### 10.1 Celery Resilience
- Improved task routing with dedicated queues
- Better error handling in background tasks
- Graceful shutdown with signal handlers
- Task retry logic with exponential backoff

### 10.2 WebSocket Reliability
- Fixed emission inconsistencies in Celery workers
- Proper event name standardization
- Connection state management with automatic fallback

### 10.3 Data Handling
- Bytes-safe dataset samples endpoint (handles HuggingFace binary data)
- UTF-8/Latin-1 fallback encoding for non-text data
- Improved error messages for debugging

### 10.4 Background Monitoring
- Improved system monitor with stable metrics collection
- Fixed GPU memory reporting for multi-GPU systems
- Optimized polling intervals for reduced overhead

---

## 11. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-05 | Initial project vision and feature breakdown |
| 2.0 | 2025-12-05 | MVP complete - reflects actual implementation |
| 2.1 | 2025-12-16 | Post-MVP: NLP analysis, Ollama integration, infrastructure improvements |

---

*Generated: 2025-12-16*
*MechInterp Studio - Post-MVP Enhancements*
