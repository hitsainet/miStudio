# Architecture Decision Record: MechInterp Studio (miStudio)

**Version:** 1.0
**Created:** 2025-10-05
**Status:** Active
**Document Type:** Project-Level Architecture Decision Record

---

## Table of Contents
1. [Overview](#overview)
2. [Technology Stack Decisions](#technology-stack-decisions)
3. [System Architecture](#system-architecture)
4. [Data Architecture](#data-architecture)
5. [API Design](#api-design)
6. [Frontend Architecture](#frontend-architecture)
7. [ML/AI Pipeline Architecture](#mlai-pipeline-architecture)
8. [Edge Optimization Strategy](#edge-optimization-strategy)
9. [Security & Privacy](#security--privacy)
10. [Development Standards](#development-standards)
11. [Testing Strategy](#testing-strategy)
12. [Deployment Architecture](#deployment-architecture)
13. [Monitoring & Observability](#monitoring--observability)
14. [Project Standards Summary](#project-standards-summary)

---

## Overview

### Purpose
This ADR establishes the foundational technology choices, architectural patterns, and development standards for MechInterp Studio. All decisions prioritize **edge deployment**, **privacy preservation**, and **single-user desktop application** architecture.

### Key Architectural Principles
1. **Edge-First**: Optimize for resource-constrained hardware (Jetson Orin Nano 8GB)
2. **Privacy by Design**: All data and computation stays local
3. **Single-User Focus**: No multi-tenancy, no authentication (unless optional)
4. **Offline-Capable**: Core features work without internet
5. **Developer Experience**: Clear patterns, comprehensive documentation

### Decision Authority
- **Backend/ML Decisions**: Technical Lead (ML Engineering)
- **Frontend Decisions**: Technical Lead (Frontend Engineering)
- **Infrastructure Decisions**: DevOps Lead
- **Security Decisions**: Technical Lead + Security Review

---

## Technology Stack Decisions

### Decision 1: Backend Framework

**Decision:** Use **FastAPI (Python 3.10+)** for backend API

**Rationale:**
- **ML Integration**: Native Python integration with PyTorch, HuggingFace, NumPy
- **Async Support**: Built-in async/await for WebSocket and long-running jobs
- **Performance**: Fast execution, comparable to Node.js for I/O-bound tasks
- **Type Safety**: Pydantic models provide runtime validation and type hints
- **Documentation**: Automatic OpenAPI/Swagger documentation generation
- **Edge Compatibility**: Runs well on Jetson (ARM64) with minimal overhead

**Alternatives Considered:**
- ❌ **NestJS (TypeScript)**: Better for web services, but Python ML ecosystem is superior
- ❌ **Flask**: Less modern, lacks native async support, no automatic validation
- ❌ **Django**: Too heavy for single-user desktop app, unnecessary features

**Trade-offs:**
- ✅ Best ML library support
- ✅ Rapid development for ML workflows
- ❌ Slightly slower cold start than compiled languages
- ❌ GIL limitations (mitigated by using process-based workers)

---

### Decision 2: Database

**Decision:** Use **PostgreSQL 14+** for metadata storage

**Rationale:**
- **JSONB Support**: Store flexible metadata (hyperparameters, feature data) without schema migrations
- **Full-Text Search**: Search features, datasets, models efficiently
- **Reliability**: ACID compliance, proven stability
- **Performance**: Handles time-series data (training metrics) efficiently with proper indexing
- **Partitioning**: Support for large tables (feature_activations) via table partitioning
- **Edge Deployment**: Runs well on Jetson, low memory footprint

**Alternatives Considered:**
- ❌ **SQLite**: Limited concurrency, no advanced features, harder to scale
- ❌ **MongoDB**: Overkill for single-user, less mature on ARM, no ACID guarantees
- ❌ **DuckDB**: Great for analytics but not ideal for transactional workloads

**Schema Design Principles:**
- Use JSONB for flexible metadata (hyperparameters, configurations)
- Time-series data in dedicated tables with indexes on (training_id, step)
- Partition large tables (feature_activations) by feature_id
- Foreign keys with CASCADE deletes for data integrity

---

### Decision 3: Caching Layer

**Decision:** Use **Redis 7+** for caching and pub/sub

**Rationale:**
- **Real-Time Updates**: Pub/sub for WebSocket message distribution
- **Session State**: Store temporary job state for resume capability
- **Rate Limiting**: Token bucket algorithm for API rate limiting
- **Caching**: Cache expensive computations (feature statistics, model metadata)
- **Low Overhead**: Minimal memory footprint (~50MB base)

**Alternatives Considered:**
- ❌ **In-Memory (Python dict)**: No persistence, lost on restart
- ❌ **Memcached**: No pub/sub support, less feature-rich

**Usage Patterns:**
- WebSocket pub/sub channels per training job
- Cache model metadata for 1 hour
- Cache dataset statistics for 24 hours
- Job queue state storage (complement to Celery)

---

### Decision 4: Job Queue

**Decision:** Use **Celery** with **Redis** as broker/backend

**Rationale:**
- **Python Native**: Seamless integration with FastAPI and PyTorch
- **Priority Queues**: Different priorities for training vs. extraction jobs
- **Task Chaining**: Complex workflows (extract → train → analyze)
- **Result Storage**: Store job results in Redis or database
- **Monitoring**: Flower dashboard for job monitoring

**Alternatives Considered:**
- ❌ **RQ (Redis Queue)**: Simpler but less feature-rich
- ❌ **BullMQ (Node.js)**: Requires separate Node.js runtime

**Queue Configuration:**
- **training_queue**: Priority 10, 1 concurrent worker (GPU limitation)
- **extraction_queue**: Priority 5, 1-2 concurrent workers
- **analysis_queue**: Priority 3, 2-4 concurrent workers (CPU-bound)

---

### Decision 5: Frontend Framework

**Decision:** Use **React 18+ with TypeScript** for frontend

**Rationale:**
- **Component-Based**: Matches Mock UI component structure perfectly
- **Type Safety**: TypeScript prevents runtime errors, improves DX
- **Ecosystem**: Rich library ecosystem (charting, state management)
- **Performance**: Virtual DOM for efficient updates
- **Developer Experience**: Hot reload, excellent tooling

**Alternatives Considered:**
- ❌ **Vue 3**: Smaller ecosystem, less TypeScript support
- ❌ **Svelte**: Less mature ecosystem, harder to find contributors
- ❌ **Angular**: Too heavy, steep learning curve

**Build Tool:** **Vite** (fast builds, hot reload, optimized for React)

---

### Decision 6: State Management

**Decision:** Use **Zustand** for global state management

**Rationale:**
- **Simplicity**: Minimal boilerplate compared to Redux
- **Performance**: Only re-renders components that use changed state
- **TypeScript Support**: Excellent type inference
- **DevTools**: Compatible with Redux DevTools
- **Size**: ~1KB minified

**Alternatives Considered:**
- ❌ **Redux Toolkit**: More boilerplate, overkill for single-user app
- ❌ **React Context**: Performance issues with frequent updates
- ❌ **Jotai/Recoil**: More atomic but less mature

**State Organization:**
- `datasetStore`: Dataset management state
- `modelStore`: Model management state
- `trainingStore`: Training jobs and metrics
- `featureStore`: Discovered features and analysis
- `steeringStore`: Steering configurations and results
- `uiStore`: UI state (active tab, modals, notifications)

---

### Decision 7: UI Component Library

**Decision:** **Custom Components** (no external UI library)

**Rationale:**
- **Exact Mock UI Match**: Mock UI specifies exact styling and behavior
- **Dark Theme Control**: Full control over slate dark theme
- **Performance**: No unused library code
- **Consistency**: All components follow exact Mock UI specification
- **Learning**: Simpler for contributors (no library-specific patterns)

**Component Library:**
- **Lucide React** for icons (matches Mock UI exactly)
- **Tailwind CSS** for styling (matches Mock UI class names)
- **Custom components** for all UI elements

**Alternatives Considered:**
- ❌ **Material-UI**: Doesn't match Mock UI style
- ❌ **Ant Design**: Heavy, opinionated design system
- ❌ **Chakra UI**: Closer but still requires significant customization

---

### Decision 8: Real-Time Communication

**Decision:** Use **Socket.IO** for WebSocket connections

**Rationale:**
- **Fallback Support**: Automatically falls back to polling if WebSocket unavailable
- **Room Support**: Isolate training job updates by room
- **Reconnection**: Automatic reconnection with exponential backoff
- **Binary Support**: Efficient transmission of training metrics
- **Python Integration**: python-socketio library for FastAPI

**Alternatives Considered:**
- ❌ **Native WebSockets**: No automatic reconnection, no fallback
- ❌ **Server-Sent Events**: One-way only, no binary support

**Channel Design:**
- `/ws/training/{trainingId}`: Training progress updates
- `/ws/extraction/{jobId}`: Extraction progress updates
- `/ws/system`: System-wide notifications (errors, warnings)

---

### Decision 9: ML Framework

**Decision:** Use **PyTorch 2.0+** for all ML operations

**Rationale:**
- **Industry Standard**: Dominant framework for research and production
- **Edge Support**: TorchScript and TensorRT for Jetson optimization
- **Dynamic Graphs**: Easier debugging and experimentation
- **HuggingFace Integration**: Native integration with transformers library
- **Jetson Support**: Official NVIDIA support for ARM64

**Alternatives Considered:**
- ❌ **TensorFlow**: Less popular in interpretability research
- ❌ **JAX**: Less mature ecosystem, harder to deploy

**Optimization Strategy:**
- Mixed precision training (torch.cuda.amp)
- TorchScript compilation for inference
- TensorRT optimization for Jetson deployment
- ONNX export for cross-platform compatibility (future)

---

### Decision 10: Model Loading & Inference

**Decision:** Use **HuggingFace Transformers** for model management

**Rationale:**
- **Standardized API**: Consistent interface for all model architectures
- **Quantization Support**: Built-in INT8/INT4 quantization via bitsandbytes
- **Automatic Downloads**: Seamless model downloads from HuggingFace Hub
- **Architecture Support**: GPT-2, LLaMA, Pythia, Phi, Gemma, Qwen families
- **Edge Optimization**: Device mapping and offloading for memory management

**Quantization Library:** **bitsandbytes** for CUDA-based quantization

**Alternatives Considered:**
- ❌ **llama.cpp**: Requires separate runtime, less Python-friendly
- ❌ **Custom loaders**: Reinventing the wheel, maintenance burden

---

### Decision 11: Dataset Management

**Decision:** Use **HuggingFace Datasets** library

**Rationale:**
- **Streaming Support**: Stream large datasets without full download
- **Apache Arrow**: Memory-mapped files for zero-copy access
- **Preprocessing**: Built-in tokenization and transformation pipelines
- **Caching**: Automatic caching of processed data
- **HuggingFace Hub Integration**: Direct dataset downloads

**Alternatives Considered:**
- ❌ **Custom Dataset Loaders**: Significant development effort
- ❌ **Pandas**: Memory-inefficient for large datasets

---

### Decision 12: Visualization

**Decision:** Use **D3.js + Recharts** for visualizations

**Rationale:**
- **D3.js**: Custom visualizations (heatmaps, correlation matrices, UMAP)
- **Recharts**: Standard charts (line charts, bar charts) with React integration
- **Performance**: Canvas-based rendering for large datasets
- **Customization**: Full control over styling to match Mock UI

**Alternatives Considered:**
- ❌ **Plotly.js**: Heavier bundle size, less customizable
- ❌ **Chart.js**: Less powerful for complex visualizations

---

### Decision 13: Development Environment

**Decision:** Use **Docker Compose with Nginx Reverse Proxy** for development and production

**Rationale:**
- **Consistency**: Same environment across team members and environments
- **Services**: Easy management of PostgreSQL, Redis, backend, frontend, nginx
- **Edge Testing**: Test on x86_64, easily portable to ARM64 (Jetson)
- **Isolation**: No conflicts with host system
- **Future HTTPS**: Nginx ready for SSL/TLS termination (Let's Encrypt)
- **Single Entry Point**: All traffic through nginx on port 80 (HTTP) / 443 (HTTPS future)

**Public Access:**
- **Base URL**: `http://mistudio.mcslab.io` (port 80)
- **Future**: HTTPS on port 443 with SSL certificate

**Compose Services:**
```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      # Future: "443:443" for HTTPS
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      # Future: - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
      - frontend

  postgres:
    image: postgres:14-alpine
    ports: ["5432:5432"]
    # Internal only, not exposed to host in production

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    # Internal only, not exposed to host in production

  backend:
    build: ./backend
    expose: ["8000"]  # Internal only, proxied by nginx
    depends_on: [postgres, redis]
    environment:
      - API_BASE_URL=http://mistudio.mcslab.io

  frontend:
    build: ./frontend
    expose: ["3000"]  # Internal only, proxied by nginx
    environment:
      - VITE_API_URL=http://mistudio.mcslab.io/api

  celery-worker:
    build: ./backend
    command: celery -A app.workers.celery_app worker --loglevel=info
    depends_on: [redis, postgres]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Nginx Configuration:**
```nginx
# /nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=download_limit:10m rate=10r/h;

    server {
        listen 80;
        server_name mistudio.mcslab.io;

        # Future HTTPS redirect
        # return 301 https://$server_name$request_uri;

        # API requests
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Rate limiting
            limit_req zone=api_limit burst=20 nodelay;

            # Timeouts for long-running operations
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        # WebSocket connections
        location /ws/ {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            # WebSocket timeouts
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
        }

        # Frontend application
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check endpoint (no rate limiting)
        location /api/v1/health {
            proxy_pass http://backend;
            access_log off;
        }
    }

    # Future HTTPS configuration
    # server {
    #     listen 443 ssl http2;
    #     server_name mistudio.mcslab.io;
    #
    #     ssl_certificate /etc/nginx/ssl/mistudio.mcslab.io.crt;
    #     ssl_certificate_key /etc/nginx/ssl/mistudio.mcslab.io.key;
    #     ssl_protocols TLSv1.2 TLSv1.3;
    #     ssl_ciphers HIGH:!aNULL:!MD5;
    #
    #     # ... (same location blocks as above)
    # }
}
```

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Single-User Desktop / Server                     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐         │
│  │                    Nginx Reverse Proxy                 │         │
│  │            http://mistudio.mcslab.io (Port 80)         │         │
│  │         Future: https://mistudio.mcslab.io (Port 443)  │         │
│  └────────┬──────────────────────────────────┬────────────┘         │
│           │                                  │                      │
│           │ /api/* + /ws/*                   │ /*                   │
│           ↓                                  ↓                      │
│  ┌────────────────┐                    ┌──────────────┐            │
│  │    Backend     │                    │   Frontend   │            │
│  │   (FastAPI)    │                    │   (React)    │            │
│  │  Port 8000     │◄──────Internal────►│  Port 3000   │            │
│  │  (Internal)    │                    │  (Internal)  │            │
│  └────────┬───────┘                    └──────────────┘            │
│                                                   │          │
│                   ┌──────────────────────────────┼─────┐    │
│                   │                              │     │    │
│              ┌────▼────┐    ┌──────▼──────┐  ┌──▼───────┐  │
│              │PostgreSQL│    │    Redis    │  │  Celery  │  │
│              │  (Meta)  │    │(Cache/Queue)│  │  Worker  │  │
│              │Port 5432 │    │  Port 6379  │  │ (GPU/CPU)│  │
│              └─────────┘    └─────────────┘  └──────────┘  │
│                                                              │
│              ┌────────────────────────────────────┐         │
│              │      Local Filesystem Storage       │         │
│              │  /data/models/                     │         │
│              │  /data/datasets/                   │         │
│              │  /data/activations/                │         │
│              │  /data/checkpoints/                │         │
│              └────────────────────────────────────┘         │
│                                                              │
│              ┌────────────────────────────────────┐         │
│              │         GPU (CUDA/TensorRT)         │         │
│              │  Jetson Orin Nano / RTX 3060+      │         │
│              └────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**Frontend (React):**
- Render UI per Mock UI specification
- Handle user interactions
- Manage WebSocket connections for real-time updates
- Display visualizations (charts, heatmaps, UMAP)
- Client-side validation and error handling

**Backend API (FastAPI):**
- Expose REST API for CRUD operations
- Handle WebSocket connections for real-time updates
- Enqueue long-running jobs to Celery
- Serve static files (if needed)
- Input validation and error handling

**Celery Worker:**
- Execute long-running jobs (training, extraction, analysis)
- Publish progress updates via Redis pub/sub
- Access GPU for training and inference
- Handle job failures and retries

**PostgreSQL:**
- Store metadata (datasets, models, trainings, features)
- Store time-series training metrics
- Store user configurations (templates, presets)

**Redis:**
- Pub/sub for WebSocket message distribution
- Cache frequently accessed data
- Job queue backend for Celery
- Rate limiting state

**Local Filesystem:**
- Store large binary files (models, datasets, activations, checkpoints)
- Organized directory structure for easy management

---

## Data Architecture

### Database Schema

#### Core Tables

**datasets**
```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    source VARCHAR(50) NOT NULL, -- 'HuggingFace', 'Local', 'Custom'
    repo_id VARCHAR(255), -- HuggingFace repo (e.g., 'roneneldan/TinyStories')
    size_bytes BIGINT,
    status VARCHAR(20) NOT NULL, -- 'downloading', 'ingesting', 'ready', 'error'
    progress FLOAT, -- 0-100
    error_message TEXT,
    file_path VARCHAR(512), -- Path to local storage
    metadata JSONB, -- Flexible metadata (splits, features, stats)
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_datasets_status (status),
    INDEX idx_datasets_source (source)
);
```

**models**
```sql
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    repo_id VARCHAR(255), -- HuggingFace repo
    architecture VARCHAR(100), -- 'GPT-2', 'LLaMA', 'Phi', etc.
    params_count BIGINT, -- Number of parameters
    quantization VARCHAR(20), -- 'FP32', 'FP16', 'Q8', 'Q4', 'Q2'
    memory_req_bytes BIGINT,
    status VARCHAR(20) NOT NULL, -- 'downloading', 'loading', 'ready', 'error'
    progress FLOAT,
    error_message TEXT,
    file_path VARCHAR(512),
    metadata JSONB, -- Architecture details, layer info
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_models_status (status),
    INDEX idx_models_architecture (architecture)
);
```

**trainings**
```sql
CREATE TABLE trainings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    encoder_type VARCHAR(20) NOT NULL, -- 'sparse', 'skip', 'transcoder'
    hyperparameters JSONB NOT NULL, -- Full hyperparameter dict
    status VARCHAR(20) NOT NULL, -- 'initializing', 'training', 'paused', 'stopped', 'completed', 'error'
    current_step INT DEFAULT 0,
    total_steps INT NOT NULL,
    progress FLOAT DEFAULT 0, -- 0-100
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_trainings_status (status),
    INDEX idx_trainings_model (model_id),
    INDEX idx_trainings_dataset (dataset_id)
);
```

**training_metrics** (time-series)
```sql
CREATE TABLE training_metrics (
    id BIGSERIAL PRIMARY KEY,
    training_id UUID NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
    step INT NOT NULL,
    loss FLOAT,
    sparsity FLOAT, -- L0 sparsity
    reconstruction_error FLOAT,
    dead_neurons INT,
    explained_variance FLOAT,
    gpu_utilization FLOAT,
    memory_used_bytes BIGINT,
    timestamp TIMESTAMP DEFAULT NOW(),
    INDEX idx_metrics_training_step (training_id, step),
    INDEX idx_metrics_timestamp (timestamp)
);
```

**checkpoints**
```sql
CREATE TABLE checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_id UUID NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
    step INT NOT NULL,
    loss FLOAT,
    file_path VARCHAR(512) NOT NULL,
    file_size_bytes BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_checkpoints_training (training_id),
    INDEX idx_checkpoints_step (training_id, step)
);
```

**features**
```sql
CREATE TABLE features (
    id BIGSERIAL PRIMARY KEY,
    training_id UUID NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
    neuron_index INT NOT NULL,
    layer INT,
    name VARCHAR(255), -- User-defined or auto-generated
    description TEXT, -- LLM-generated or user-provided
    activation_frequency FLOAT, -- 0-1
    interpretability_score FLOAT, -- 0-1
    is_favorite BOOLEAN DEFAULT FALSE,
    metadata JSONB, -- Max activating examples, logit lens, etc.
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_features_training (training_id),
    INDEX idx_features_activation (activation_frequency DESC),
    INDEX idx_features_interpretability (interpretability_score DESC),
    INDEX idx_features_favorite (is_favorite)
);
```

**feature_activations** (large table, partitioned)
```sql
CREATE TABLE feature_activations (
    id BIGSERIAL,
    feature_id BIGINT NOT NULL REFERENCES features(id) ON DELETE CASCADE,
    sample_text TEXT,
    tokens JSONB, -- Array of tokens
    activations JSONB, -- Array of activation values per token
    max_activation FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
) PARTITION BY HASH (feature_id);

-- Create 8 partitions for parallelism
CREATE TABLE feature_activations_p0 PARTITION OF feature_activations FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE feature_activations_p1 PARTITION OF feature_activations FOR VALUES WITH (MODULUS 8, REMAINDER 1);
-- ... (continue for p2-p7)

CREATE INDEX idx_activations_feature ON feature_activations (feature_id);
CREATE INDEX idx_activations_max ON feature_activations (max_activation DESC);
```

**training_templates**
```sql
CREATE TABLE training_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_id UUID REFERENCES models(id) ON DELETE SET NULL, -- NULL = generic template
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    encoder_type VARCHAR(20) NOT NULL,
    hyperparameters JSONB NOT NULL,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_templates_favorite (is_favorite)
);
```

**extraction_templates**
```sql
CREATE TABLE extraction_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    layers INT[], -- PostgreSQL array of layer indices
    hook_types VARCHAR(50)[], -- ['residual', 'mlp', 'attention']
    max_samples INT,
    top_k_examples INT,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**steering_presets**
```sql
CREATE TABLE steering_presets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_id UUID NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    features JSONB NOT NULL, -- [{"feature_id": 42, "coefficient": 3.5}, ...]
    intervention_layer INT NOT NULL,
    temperature FLOAT DEFAULT 1.0,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_presets_training (training_id),
    INDEX idx_presets_favorite (is_favorite)
);
```

### Filesystem Organization

```
/data/
├── models/
│   ├── {model_id}/
│   │   ├── config.json
│   │   ├── model.safetensors (or pytorch_model.bin)
│   │   ├── tokenizer.json
│   │   └── metadata.json
│   └── ...
├── datasets/
│   ├── {dataset_id}/
│   │   ├── raw/                # Original downloads
│   │   ├── tokenized/          # Arrow format tokenized data
│   │   └── metadata.json
│   └── ...
├── activations/
│   ├── {model_id}_{dataset_id}_{layer}/
│   │   ├── activations.npy     # Memory-mapped NumPy array
│   │   ├── indices.npy         # Sample indices
│   │   └── metadata.json
│   └── ...
├── checkpoints/
│   ├── {training_id}/
│   │   ├── step_{step}/
│   │   │   ├── encoder.pt
│   │   │   ├── decoder.pt
│   │   │   ├── optimizer.pt
│   │   │   └── metadata.json
│   │   └── ...
│   └── ...
└── cache/
    └── ... (temporary files, cleared on restart)
```

---

## API Design

### REST API Structure

**Base URL (Public):** `http://mistudio.mcslab.io/api/v1`
**Base URL (Development):** `http://localhost/api/v1` or `http://localhost:8000/api/v1` (direct backend)

**Note:** All production traffic goes through nginx reverse proxy on port 80. Backend runs on internal port 8000.

### API Design Principles

1. **RESTful Conventions:**
   - GET: Retrieve resources (idempotent)
   - POST: Create resources or trigger actions
   - PUT: Replace entire resource (idempotent)
   - PATCH: Partial update (idempotent)
   - DELETE: Remove resource (idempotent)

2. **Response Format:**
   ```json
   // Success
   {
     "data": { ... },
     "meta": {
       "timestamp": "2025-10-05T12:34:56Z",
       "request_id": "req_abc123"
     }
   }

   // Error
   {
     "error": {
       "code": "VALIDATION_ERROR",
       "message": "Invalid learning rate: must be between 1e-6 and 1e-2",
       "details": {
         "field": "learningRate",
         "value": 0.5,
         "constraints": {"min": 1e-6, "max": 1e-2}
       },
       "retryable": false,
       "timestamp": "2025-10-05T12:34:56Z"
     }
   }
   ```

3. **Status Codes:**
   - 200: Success (GET, PUT, DELETE)
   - 201: Created (POST)
   - 202: Accepted (long-running job queued)
   - 400: Bad Request (validation error)
   - 404: Not Found
   - 409: Conflict (duplicate resource)
   - 429: Rate Limit Exceeded
   - 500: Internal Server Error
   - 503: Service Unavailable (GPU busy)

4. **Pagination:**
   - Query params: `?page=1&limit=50`
   - Response: `{"data": [...], "pagination": {"page": 1, "total": 234, "has_next": true}}`

5. **Filtering/Sorting:**
   - Query params: `?search=query&sortBy=field&order=desc`

### Core API Endpoints

#### Datasets

```
GET    /api/v1/datasets
GET    /api/v1/datasets/:id
POST   /api/v1/datasets/download
       Body: { repoId: string, accessToken?: string, split?: string }
       Returns: 202 { job_id, estimated_duration_seconds }
DELETE /api/v1/datasets/:id

GET    /api/v1/datasets/:id/samples?page=1&limit=50&split=train&search=query
GET    /api/v1/datasets/:id/stats
```

#### Models

```
GET    /api/v1/models
GET    /api/v1/models/:id
GET    /api/v1/models/:id/architecture
POST   /api/v1/models/download
       Body: { repoId: string, quantization: string, accessToken?: string }
       Returns: 202 { job_id }
DELETE /api/v1/models/:id
```

#### Training

```
GET    /api/v1/trainings?status=training&page=1&limit=20
GET    /api/v1/trainings/:id
POST   /api/v1/trainings
       Body: TrainingConfig (model_id, dataset_id, encoder_type, hyperparameters)
       Returns: 201 { training_id, status: 'initializing' }
POST   /api/v1/trainings/:id/pause
POST   /api/v1/trainings/:id/resume
POST   /api/v1/trainings/:id/stop
DELETE /api/v1/trainings/:id

GET    /api/v1/trainings/:id/metrics?start_step=0&end_step=1000
GET    /api/v1/trainings/:id/checkpoints
POST   /api/v1/trainings/:id/checkpoints
DELETE /api/v1/trainings/:id/checkpoints/:checkpointId
POST   /api/v1/trainings/:id/checkpoints/:checkpointId/load
```

#### Features

```
GET    /api/v1/features?training_id=xxx&page=1&limit=50&sortBy=activation&order=desc
GET    /api/v1/features/:id
PATCH  /api/v1/features/:id
       Body: { name?, description?, is_favorite? }
GET    /api/v1/features/:id/activations?top_k=100
GET    /api/v1/features/:id/logit-lens
GET    /api/v1/features/:id/correlations?top_k=20

POST   /api/v1/features/extract
       Body: { training_id: string, template_id?: string }
       Returns: 202 { job_id }
```

#### Steering

```
POST   /api/v1/steering/generate
       Body: {
         model_id: string,
         prompt: string,
         features: Array<{ id: number, coefficient: number }>,
         intervention_layer: number,
         temperature: number
       }
       Returns: 200 { unsteered_output, steered_output, metrics }

GET    /api/v1/steering/presets?training_id=xxx
POST   /api/v1/steering/presets
       Body: SteeringPreset
PUT    /api/v1/steering/presets/:id
DELETE /api/v1/steering/presets/:id
```

#### Templates

```
GET    /api/v1/templates/training?model_id=xxx&is_favorite=true
POST   /api/v1/templates/training
PUT    /api/v1/templates/training/:id
DELETE /api/v1/templates/training/:id

GET    /api/v1/templates/extraction
POST   /api/v1/templates/extraction
PUT    /api/v1/templates/extraction/:id
DELETE /api/v1/templates/extraction/:id
```

#### System

```
GET    /api/v1/health
       Returns: 200 { status: 'ok', gpu: {...}, db: 'connected', redis: 'connected' }
GET    /api/v1/health/ready
GET    /api/v1/health/live

GET    /api/v1/system/info
       Returns: { version, platform, gpu_info, memory_info }
```

### WebSocket Protocol

**Connection:** `ws://localhost:8000/ws/training/{trainingId}`

**Client → Server:**
```json
{ "type": "subscribe", "trainingId": "tr_abc123" }
{ "type": "ping" }
```

**Server → Client:**
```json
{
  "type": "training.progress",
  "data": {
    "step": 1234,
    "total_steps": 10000,
    "progress": 12.34,
    "metrics": { "loss": 0.342, "sparsity": 12.4 },
    "estimated_remaining_seconds": 427
  },
  "timestamp": "2025-10-05T12:34:56Z"
}

{ "type": "pong" }

{
  "type": "training.completed",
  "data": { "final_loss": 0.234, "total_time_seconds": 3600 }
}

{
  "type": "training.error",
  "error": { "code": "GPU_OOM", "message": "GPU out of memory" }
}
```

**Heartbeat:** Client sends ping every 30s, server responds with pong
**Reconnection:** Exponential backoff: 1s, 2s, 4s, 8s, max 30s

### Rate Limiting

**Global Limits (stored in Redis):**
- API requests: 100/minute per IP
- Training jobs: 1 concurrent (GPU limitation)
- Dataset downloads: 10/hour
- Model downloads: 10/hour
- Steering generation: 20/hour

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 47
X-RateLimit-Reset: 1633024800 (Unix timestamp)
Retry-After: 60 (seconds, on 429 responses)
```

---

## Frontend Architecture

### Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── datasets/
│   │   │   ├── DatasetsPanel.tsx
│   │   │   ├── DatasetCard.tsx
│   │   │   ├── DatasetDetailModal.tsx
│   │   │   └── HuggingFaceDownloader.tsx
│   │   ├── models/
│   │   │   ├── ModelsPanel.tsx
│   │   │   ├── ModelCard.tsx
│   │   │   ├── ModelArchitectureViewer.tsx
│   │   │   └── ActivationExtractionConfig.tsx
│   │   ├── training/
│   │   │   ├── TrainingPanel.tsx
│   │   │   ├── TrainingConfig.tsx
│   │   │   ├── TrainingCard.tsx
│   │   │   ├── TrainingMetrics.tsx
│   │   │   └── CheckpointManager.tsx
│   │   ├── features/
│   │   │   ├── FeaturesPanel.tsx
│   │   │   ├── FeatureBrowser.tsx
│   │   │   ├── FeatureCard.tsx
│   │   │   ├── FeatureDetailModal.tsx
│   │   │   └── FeatureVisualizations.tsx
│   │   ├── steering/
│   │   │   ├── SteeringPanel.tsx
│   │   │   ├── FeatureSelector.tsx
│   │   │   ├── ComparativeGeneration.tsx
│   │   │   └── SteeringMetrics.tsx
│   │   ├── common/
│   │   │   ├── Header.tsx
│   │   │   ├── Navigation.tsx
│   │   │   ├── ProgressBar.tsx
│   │   │   ├── StatusBadge.tsx
│   │   │   ├── Modal.tsx
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Select.tsx
│   │   │   └── Toast.tsx
│   │   └── visualizations/
│   │       ├── LineChart.tsx
│   │       ├── Heatmap.tsx
│   │       ├── ScatterPlot.tsx
│   │       └── CorrelationMatrix.tsx
│   ├── stores/
│   │   ├── datasetStore.ts
│   │   ├── modelStore.ts
│   │   ├── trainingStore.ts
│   │   ├── featureStore.ts
│   │   ├── steeringStore.ts
│   │   └── uiStore.ts
│   ├── services/
│   │   ├── api.ts          # Axios client with interceptors
│   │   ├── websocket.ts    # Socket.IO client
│   │   ├── datasetService.ts
│   │   ├── modelService.ts
│   │   ├── trainingService.ts
│   │   ├── featureService.ts
│   │   └── steeringService.ts
│   ├── hooks/
│   │   ├── useDatasets.ts
│   │   ├── useModels.ts
│   │   ├── useTraining.ts
│   │   ├── useFeatures.ts
│   │   ├── useSteering.ts
│   │   └── useWebSocket.ts
│   ├── types/
│   │   ├── dataset.types.ts
│   │   ├── model.types.ts
│   │   ├── training.types.ts
│   │   ├── feature.types.ts
│   │   └── steering.types.ts
│   ├── utils/
│   │   ├── format.ts       # Number/date formatting
│   │   ├── validation.ts   # Input validation
│   │   └── constants.ts    # App constants
│   ├── styles/
│   │   └── globals.css     # Tailwind + custom styles
│   ├── App.tsx
│   └── main.tsx
├── public/
│   └── index.html
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts
```

### Component Design Patterns

**Component Guidelines (from Mock UI):**
1. **Functional Components**: Use function components with hooks (no class components)
2. **TypeScript**: All components strictly typed
3. **Props Interface**: Define explicit interface for each component
4. **Styling**: Tailwind utility classes matching Mock UI exactly
5. **State**: Local state for UI, Zustand for global state
6. **Side Effects**: useEffect with proper dependency arrays

**Example Component Pattern:**
```typescript
interface TrainingCardProps {
  training: Training;
  onPause: (id: string) => void;
  onResume: (id: string) => void;
  onStop: (id: string) => void;
}

export function TrainingCard({ training, onPause, onResume, onStop }: TrainingCardProps) {
  const [showMetrics, setShowMetrics] = useState(false);

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      {/* Component implementation matching Mock UI exactly */}
    </div>
  );
}
```

### State Management Pattern

**Zustand Store Example:**
```typescript
interface TrainingStore {
  trainings: Training[];
  activeTraining: Training | null;

  fetchTrainings: () => Promise<void>;
  startTraining: (config: TrainingConfig) => Promise<string>;
  pauseTraining: (id: string) => Promise<void>;
  resumeTraining: (id: string) => Promise<void>;
  stopTraining: (id: string) => Promise<void>;
  updateTrainingProgress: (id: string, progress: TrainingProgress) => void;
}

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  trainings: [],
  activeTraining: null,

  fetchTrainings: async () => {
    const trainings = await trainingService.list();
    set({ trainings });
  },

  startTraining: async (config) => {
    const training = await trainingService.start(config);
    set((state) => ({ trainings: [training, ...state.trainings] }));
    return training.id;
  },

  // ... other actions
}));
```

### Styling Standards

**Tailwind Configuration (matching Mock UI):**
```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        slate: {
          950: '#0f1629', // Custom darkest slate
        },
      },
    },
  },
  // ... rest of config
};
```

**Common Class Patterns (from Mock UI):**
- Background: `bg-slate-950`, `bg-slate-900/50`, `bg-slate-900`
- Borders: `border-slate-800`, `border-slate-700`
- Text: `text-slate-100`, `text-slate-400`, `text-emerald-400`
- Buttons: `bg-emerald-600 hover:bg-emerald-700`
- Transitions: `transition-colors`, `transition-all duration-500`

---

## ML/AI Pipeline Architecture

### Activation Extraction Pipeline

**Process:**
1. Load model with quantization (bitsandbytes)
2. Register forward hooks on target layers
3. Stream dataset samples through model
4. Collect activations in memory-mapped arrays
5. Store to disk with metadata

**Code Pattern:**
```python
class ActivationExtractor:
    def __init__(self, model, layers, device='cuda'):
        self.model = model
        self.activations = {}
        self.hooks = []

        for layer_idx in layers:
            hook = self._register_hook(layer_idx)
            self.hooks.append(hook)

    def _register_hook(self, layer_idx):
        def hook_fn(module, input, output):
            self.activations[layer_idx].append(output.detach().cpu())

        layer = self.model.transformer.h[layer_idx]
        return layer.register_forward_hook(hook_fn)

    def extract(self, dataloader, output_path):
        # Stream through dataset, collect activations
        # Save to memory-mapped NumPy array
        pass
```

### SAE Training Pipeline

**Architecture:**
```python
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae, l1_coefficient=1e-3):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.l1_coefficient = l1_coefficient

        # Initialize decoder columns to unit norm
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x):
        # Encode
        z = F.relu(self.encoder(x))

        # Decode
        x_recon = self.decoder(z)

        # Loss
        mse_loss = F.mse_loss(x_recon, x)
        l1_loss = z.abs().mean()
        loss = mse_loss + self.l1_coefficient * l1_loss

        return x_recon, loss, z
```

**Training Loop:**
```python
async def train_sae(training_id, config):
    # Load activations
    activations = load_activations(config.activation_path)

    # Initialize SAE
    sae = SparseAutoencoder(d_model=768, d_sae=768*8)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-4)

    for step in range(config.total_steps):
        # Sample batch
        batch = sample_batch(activations, batch_size=256)

        # Forward pass
        x_recon, loss, z = sae(batch)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Metrics
        sparsity = (z > 0).float().mean().item()
        dead_neurons = (z.max(dim=0).values == 0).sum().item()

        # Publish progress (every 10 steps)
        if step % 10 == 0:
            await publish_progress(training_id, step, loss.item(), sparsity, dead_neurons)

        # Checkpoint (every 1000 steps)
        if step % 1000 == 0:
            save_checkpoint(training_id, step, sae, optimizer)
```

### Feature Discovery Pipeline

**Process:**
1. Load trained SAE
2. Run evaluation dataset through SAE
3. Record activation statistics per neuron
4. Find max-activating examples
5. Compute interpretability scores
6. Store features to database

**Code Pattern:**
```python
def discover_features(training_id, sae, eval_dataset):
    features = []

    for neuron_idx in range(sae.d_sae):
        # Collect activations for this neuron
        activations = []
        examples = []

        for batch in eval_dataset:
            _, _, z = sae(batch)
            neuron_acts = z[:, neuron_idx]

            # Store top-k activating examples
            top_k_indices = neuron_acts.topk(k=100).indices
            activations.extend(neuron_acts[top_k_indices])
            examples.extend([eval_dataset[i] for i in top_k_indices])

        # Compute statistics
        activation_frequency = (torch.tensor(activations) > 0).float().mean()
        interpretability_score = compute_interpretability(examples)

        # Create feature record
        feature = Feature(
            training_id=training_id,
            neuron_index=neuron_idx,
            activation_frequency=activation_frequency,
            interpretability_score=interpretability_score,
            max_activating_examples=examples[:100]
        )
        features.append(feature)

    return features
```

### Model Steering Pipeline

**Intervention Strategy:**
```python
class SteeringHook:
    def __init__(self, layer_idx, steering_vector):
        self.layer_idx = layer_idx
        self.steering_vector = steering_vector

    def __call__(self, module, input, output):
        # Add steering vector to residual stream
        return output + self.steering_vector

def apply_steering(model, features, coefficients, intervention_layer):
    # Build steering vector
    steering_vector = torch.zeros(model.config.hidden_size)
    for feature_id, coefficient in zip(features, coefficients):
        feature_vector = load_feature_vector(feature_id)
        steering_vector += coefficient * feature_vector

    # Register hook
    hook = SteeringHook(intervention_layer, steering_vector)
    handle = model.transformer.h[intervention_layer].register_forward_hook(hook)

    return handle
```

---

## Edge Optimization Strategy

### Memory Management

**Techniques:**
1. **Mixed Precision Training**: Use torch.cuda.amp for FP16/FP32 mixed precision
2. **Gradient Accumulation**: Simulate larger batch sizes without memory overhead
3. **Memory-Mapped Files**: Zero-copy access to large datasets and activations
4. **Streaming Inference**: Process data in chunks, don't load full dataset
5. **Model Quantization**: INT8/INT4 quantization via bitsandbytes

**Memory Budget (Jetson Orin Nano 8GB):**
- System: 1GB
- Model (quantized): 1-2GB
- SAE: 100-500MB
- Activations (cached): 2-3GB
- Training batch: 500MB
- Overhead: 1GB

### GPU Optimization

**TensorRT Integration:**
```python
import torch_tensorrt

# Optimize model for Jetson
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 512), dtype=torch.int32)],
    enabled_precisions={torch.float16},
    workspace_size=1 << 30  # 1GB
)
```

**CUDA Optimization:**
- Use pinned memory for faster GPU transfers
- Batch operations when possible
- Use torch.compile() for JIT compilation (PyTorch 2.0+)

### Inference Optimization

**Techniques:**
1. **KV Cache**: Cache key-value pairs for autoregressive generation
2. **Batching**: Batch multiple requests when possible
3. **Early Stopping**: Stop generation when max_new_tokens reached
4. **Speculative Decoding**: Use smaller model for draft, large model for verification (future)

---

## Security & Privacy

### Data Security

**Principles:**
1. **No External Transmission**: All data stays local
2. **No Telemetry**: No usage tracking or analytics
3. **User Control**: User explicitly initiates all downloads
4. **Secure Storage**: Proper file permissions on data directories

**Implementation:**
- File permissions: 0700 for data directories
- No cloud sync by default
- Optional encryption for sensitive models (future)

### API Security

**Input Validation:**
- Pydantic models for all request bodies
- Sanitize file paths to prevent traversal attacks
- Validate numeric ranges (learning rate, batch size, etc.)
- Reject oversized requests (max file upload: 10GB)

**Rate Limiting:**
- Token bucket algorithm via Redis
- Per-endpoint limits
- 429 responses with Retry-After header

### GPU Protection

**Resource Monitoring:**
- Check GPU memory before accepting training jobs
- Reject jobs if GPU >90% utilized
- Automatic job preemption for critical failures
- OOM handler to gracefully fail jobs

---

## Development Standards

### Code Style

**Python (Backend):**
- **Formatter**: Black (line length 100)
- **Linter**: Ruff (replaces Flake8, isort, pylint)
- **Type Checker**: MyPy (strict mode)
- **Docstrings**: Google style

**TypeScript (Frontend):**
- **Formatter**: Prettier
- **Linter**: ESLint (with TypeScript rules)
- **Style Guide**: Airbnb TypeScript

### Naming Conventions

**Python:**
- Functions/methods: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

**TypeScript:**
- Functions/methods: `camelCase`
- Classes/Components: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Interfaces: `PascalCase` (no `I` prefix)
- Types: `PascalCase`

### File Organization

**Backend:**
```
backend/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── datasets.py
│   │   │   │   ├── models.py
│   │   │   │   ├── trainings.py
│   │   │   │   ├── features.py
│   │   │   │   └── steering.py
│   │   │   └── router.py
│   │   └── deps.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── db/
│   │   ├── base.py
│   │   ├── session.py
│   │   └── models/
│   │       ├── dataset.py
│   │       ├── model.py
│   │       ├── training.py
│   │       └── feature.py
│   ├── schemas/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── training.py
│   │   └── feature.py
│   ├── services/
│   │   ├── dataset_service.py
│   │   ├── model_service.py
│   │   ├── training_service.py
│   │   └── feature_service.py
│   ├── ml/
│   │   ├── activation_extraction.py
│   │   ├── sae_training.py
│   │   ├── feature_discovery.py
│   │   └── steering.py
│   ├── workers/
│   │   ├── celery_app.py
│   │   └── tasks.py
│   ├── websocket/
│   │   └── manager.py
│   └── main.py
├── tests/
│   ├── api/
│   ├── services/
│   └── ml/
├── alembic/
│   └── versions/
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

### Error Handling

**Backend Pattern:**
```python
from fastapi import HTTPException

class DatasetNotFoundError(Exception):
    pass

class GPUOutOfMemoryError(Exception):
    pass

@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    try:
        dataset = await dataset_service.get(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        return {"data": dataset}
    except GPUOutOfMemoryError as e:
        raise HTTPException(status_code=503, detail="GPU out of memory")
    except Exception as e:
        logger.error(f"Error getting dataset: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

**Frontend Pattern:**
```typescript
async function fetchDataset(id: string): Promise<Dataset> {
  try {
    const response = await api.get(`/datasets/${id}`);
    return response.data.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.response?.status === 404) {
        throw new Error(`Dataset ${id} not found`);
      }
      if (error.response?.status === 503) {
        throw new Error('GPU out of memory, please try again later');
      }
    }
    throw new Error('Failed to fetch dataset');
  }
}
```

### Logging

**Backend (Python):**
```python
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)

# Structured logging
logger.info(
    "Training started",
    extra={
        "training_id": training_id,
        "model_id": config.model_id,
        "dataset_id": config.dataset_id,
    }
)
```

**Log Levels:**
- DEBUG: Detailed diagnostic information
- INFO: General informational messages
- WARNING: Something unexpected happened
- ERROR: Error occurred, but app continues
- CRITICAL: Serious error, app may crash

---

## Testing Strategy

### Backend Testing

**Test Pyramid:**
- Unit Tests: 70% (services, ML pipelines)
- Integration Tests: 20% (API endpoints, database)
- E2E Tests: 10% (full workflows)

**Test Framework:** pytest

**Coverage Target:** >80% overall, >90% for critical paths

**Example Unit Test:**
```python
import pytest
from app.ml.sae_training import SparseAutoencoder

def test_sae_forward_pass():
    sae = SparseAutoencoder(d_model=768, d_sae=768*8)
    x = torch.randn(32, 768)

    x_recon, loss, z = sae(x)

    assert x_recon.shape == x.shape
    assert z.shape == (32, 768*8)
    assert loss.item() > 0
    assert (z >= 0).all()  # ReLU activation
```

**Example Integration Test:**
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_create_training(client):
    response = client.post("/api/v1/trainings", json={
        "model_id": "test-model",
        "dataset_id": "test-dataset",
        "encoder_type": "sparse",
        "hyperparameters": {
            "learning_rate": 1e-4,
            "batch_size": 256,
            # ... other params
        }
    })
    assert response.status_code == 201
    assert "training_id" in response.json()["data"]
```

### Frontend Testing

**Test Framework:** Vitest + React Testing Library

**Test Types:**
- Component Tests: Render, interaction, state
- Hook Tests: Custom hooks logic
- Integration Tests: Multiple components together

**Example Component Test:**
```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { TrainingCard } from './TrainingCard';

test('pauses training when pause button clicked', async () => {
  const onPause = vi.fn();
  const training = { id: 'tr1', status: 'training', progress: 50 };

  render(<TrainingCard training={training} onPause={onPause} />);

  const pauseButton = screen.getByRole('button', { name: /pause/i });
  fireEvent.click(pauseButton);

  expect(onPause).toHaveBeenCalledWith('tr1');
});
```

### E2E Testing

**Framework:** Playwright

**Critical Paths:**
1. Download dataset → Download model → Start training → View features → Test steering
2. Resume training after pause
3. Load checkpoint and continue training

---

## Deployment Architecture

### Development Environment

**Docker Compose Setup:**
```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      # Future: - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      # Future: - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
      - frontend
    restart: unless-stopped

  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_PASSWORD: devpassword
      POSTGRES_DB: mistudio
    # Internal only - not exposed to host in production
    ports:
      - "5432:5432"  # Comment out for production
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    # Internal only - not exposed to host in production
    ports:
      - "6379:6379"  # Comment out for production
    restart: unless-stopped

  backend:
    build: ./backend
    expose:
      - "8000"  # Internal only, proxied by nginx
    environment:
      DATABASE_URL: postgresql://postgres:devpassword@postgres:5432/mistudio
      REDIS_URL: redis://redis:6379
      API_BASE_URL: http://mistudio.mcslab.io
      ALLOWED_ORIGINS: http://mistudio.mcslab.io,http://localhost
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app
      - /data:/data
    restart: unless-stopped

  celery-worker:
    build: ./backend
    command: celery -A app.workers.celery_app worker --loglevel=info --concurrency=1
    environment:
      DATABASE_URL: postgresql://postgres:devpassword@postgres:5432/mistudio
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app
      - /data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  frontend:
    build: ./frontend
    expose:
      - "3000"  # Internal only, proxied by nginx
    volumes:
      - ./frontend:/app
    environment:
      VITE_API_URL: http://mistudio.mcslab.io/api
    restart: unless-stopped

volumes:
  postgres_data:
```

### Production Deployment (Jetson)

**Installation:**
1. Install JetPack SDK (includes CUDA, TensorRT)
2. Install Docker with NVIDIA Container Runtime
3. Run Docker Compose setup
4. Configure systemd service for auto-start

**Systemd Service:**
```ini
[Unit]
Description=MechInterp Studio
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/mistudio
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down

[Install]
WantedBy=multi-user.target
```

### Native Installation (Alternative)

**For users who prefer native installation:**
1. Install PostgreSQL 14+
2. Install Redis 7+
3. Install Python 3.10+ with pip
4. Install Node.js 18+ with npm
5. Install backend: `pip install -r requirements.txt`
6. Install frontend: `npm install`
7. Start services: `./scripts/start.sh`

---

## Monitoring & Observability

### Metrics Collection

**Prometheus Metrics (Optional):**
- API request latency histogram
- Training job duration histogram
- GPU utilization gauge
- Memory usage gauge
- Database query latency histogram
- WebSocket connection count gauge

**Implementation:**
```python
from prometheus_client import Counter, Histogram, Gauge

api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])
training_duration = Histogram('training_duration_seconds', 'Training duration')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization')
```

### Logging

**Structured Logging Format:**
```json
{
  "timestamp": "2025-10-05T12:34:56Z",
  "level": "INFO",
  "logger": "app.api.trainings",
  "message": "Training started",
  "training_id": "tr_abc123",
  "model_id": "m1",
  "dataset_id": "ds1",
  "request_id": "req_xyz789"
}
```

**Log Aggregation:**
- Development: Console output
- Production: File rotation (logrotate)
- Optional: Loki for centralized logging

### Health Checks

**Endpoints:**
```
GET /api/v1/health        # Overall health
GET /api/v1/health/ready  # Ready to accept requests
GET /api/v1/health/live   # Process is alive
```

**Health Check Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-10-05T12:34:56Z",
  "components": {
    "database": "connected",
    "redis": "connected",
    "gpu": {
      "available": true,
      "name": "NVIDIA Jetson Orin Nano",
      "memory_total": "8GB",
      "memory_used": "2.3GB",
      "utilization": "45%"
    }
  }
}
```

---

## Project Standards Summary

### Technology Stack

**Backend:**
- Framework: FastAPI (Python 3.10+)
- Database: PostgreSQL 14+
- Cache: Redis 7+
- Queue: Celery with Redis backend
- ML: PyTorch 2.0+, HuggingFace Transformers/Datasets
- Quantization: bitsandbytes
- Edge: TensorRT for Jetson

**Frontend:**
- Framework: React 18+ with TypeScript
- Build Tool: Vite
- State: Zustand
- Styling: Tailwind CSS (matching Mock UI)
- Icons: Lucide React
- Charts: D3.js + Recharts
- Real-time: Socket.IO

**Infrastructure:**
- Containerization: Docker + Docker Compose
- Orchestration: systemd (production)
- Monitoring: Prometheus + Grafana (optional)

### Development Workflow

**Branch Strategy:**
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches

**Commit Convention:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: feat, fix, docs, style, refactor, test, chore

**Example:**
```
feat(training): add checkpoint auto-save

Implement automatic checkpoint saving every N steps during training.
Configurable interval with default of 1000 steps.

Closes #123
```

### Code Review Standards

**Requirements:**
- All code reviewed by at least one other developer
- Automated tests pass (CI/CD)
- No decrease in test coverage
- Follows style guide (Black/Prettier)
- Documentation updated

### Documentation Standards

**Required Documentation:**
- API endpoints (OpenAPI/Swagger)
- Database schema (ERD + migrations)
- Architecture diagrams (system, sequence)
- User guide (getting started, tutorials)
- Developer guide (setup, contribution, architecture)

---

## Decision Log

### Major Decisions

| Decision | Date | Rationale |
|----------|------|-----------|
| FastAPI for backend | 2025-10-05 | Best ML integration, async support, type safety |
| PostgreSQL for database | 2025-10-05 | JSONB support, reliability, performance |
| React + TypeScript | 2025-10-05 | Matches Mock UI, strong ecosystem, type safety |
| Zustand for state | 2025-10-05 | Minimal boilerplate, performance, simplicity |
| Custom UI components | 2025-10-05 | Exact Mock UI match, full control, lightweight |
| Docker Compose | 2025-10-05 | Consistency, easy setup, portable |
| PyTorch + HuggingFace | 2025-10-05 | Industry standard, edge support, best ecosystem |
| Nginx reverse proxy | 2025-10-05 | Single entry point, HTTPS termination, rate limiting, production-ready |

### Open Questions

- [ ] Exact Jetson TensorRT optimization strategy (defer to implementation)
- [ ] Multi-GPU support (future enhancement)
- [ ] ONNX export for cross-platform (future consideration)
- [ ] Optional cloud sync for backups (future feature)

---

## References

**External Documentation:**
- FastAPI: https://fastapi.tiangolo.com/
- PyTorch: https://pytorch.org/docs/stable/index.html
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- React: https://react.dev/
- Tailwind CSS: https://tailwindcss.com/docs
- PostgreSQL: https://www.postgresql.org/docs/14/
- Redis: https://redis.io/docs/

**Project Documents:**
- Project PRD: `0xcc/prds/000_PPRD|miStudio.md`
- Mock UI Specification: `0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
- Technical Specification: `0xcc/project-specs/core/miStudio_Specification.md`

---

## Appendix: Project Standards for CLAUDE.md

**Copy the following section to CLAUDE.md:**

```markdown
## Project Standards

### Technology Stack

**Backend:**
- Python 3.10+, FastAPI, PostgreSQL 14+, Redis 7+, Celery
- PyTorch 2.0+, HuggingFace (transformers, datasets), bitsandbytes
- TensorRT for Jetson optimization

**Frontend:**
- React 18+ with TypeScript, Vite, Zustand
- Tailwind CSS (slate dark theme per Mock UI)
- Lucide React icons, D3.js + Recharts
- Socket.IO for real-time updates

**Infrastructure:**
- Docker Compose for development
- systemd for production (Jetson)
- Local filesystem storage (/data/)

### Coding Standards

**Python:**
- Formatter: Black (line length 100)
- Linter: Ruff
- Type Checker: MyPy (strict)
- Docstrings: Google style

**TypeScript:**
- Formatter: Prettier
- Linter: ESLint (Airbnb)
- All components strictly typed

### Naming Conventions

**Python:** `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
**TypeScript:** `camelCase` functions, `PascalCase` components/types, `UPPER_SNAKE_CASE` constants

### Testing

**Backend:** pytest (>80% coverage target)
**Frontend:** Vitest + React Testing Library
**E2E:** Playwright for critical paths

### Git Workflow

**Branches:** `main` (production), `develop` (integration), `feature/*`, `bugfix/*`
**Commits:** Conventional commits (`feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`)
**Review:** All code reviewed, tests pass, no coverage decrease

### File Organization

**Backend:** `app/api/`, `app/services/`, `app/ml/`, `app/db/`, `app/workers/`
**Frontend:** `src/components/`, `src/stores/`, `src/services/`, `src/hooks/`, `src/types/`

### Error Handling

- Backend: FastAPI HTTPException with proper status codes
- Frontend: Try-catch with axios error handling
- Structured error responses with `error.code`, `error.message`, `error.details`

### API Design

- RESTful conventions (GET, POST, PUT, PATCH, DELETE)
- Response format: `{ data, meta }` or `{ error }`
- Status codes: 200, 201, 202, 400, 404, 409, 429, 500, 503
- Pagination: `?page=1&limit=50`
- WebSocket: Socket.IO with rooms per training job

### UI/UX Standards

**PRIMARY REFERENCE:** `0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`

- Dark theme: slate color palette (bg-slate-950, 900, 800)
- Emerald accents: buttons, success states
- Tailwind utility classes matching Mock UI exactly
- Functional components with TypeScript
- Zustand for global state, local state for UI

### Database Schema

- PostgreSQL with JSONB for flexible metadata
- Time-series metrics in dedicated tables with indexes
- Partitioned tables for large data (feature_activations)
- Foreign keys with CASCADE for data integrity

### Edge Optimization

- Mixed precision training (FP16)
- Gradient accumulation for large effective batches
- Memory-mapped files for datasets/activations
- TensorRT optimization for Jetson inference
- INT8/INT4 quantization via bitsandbytes

### Deployment

**Development:** Docker Compose (nginx, postgres, redis, backend, frontend, celery)
**Production:** systemd service on Jetson with Docker Compose + nginx reverse proxy
**Base URL:** http://mistudio.mcslab.io (port 80)
**Future HTTPS:** Port 443 with SSL certificate
**Alternative:** Native installation (Nginx + PostgreSQL + Redis + Python + Node.js)
```

---

**Document Control:**

**Version:** 1.0
**Created By:** AI Dev Tasks Framework (XCC)
**Created Date:** 2025-10-05
**Last Updated:** 2025-10-05
**Status:** Active - Approved for Development

**Review and Approval:**
- [ ] Technical Lead (Backend/ML) Review
- [ ] Technical Lead (Frontend) Review
- [ ] DevOps Lead Review
- [ ] Development Team Acknowledgment

**Related Documents:**
- Project PRD: `0xcc/prds/000_PPRD|miStudio.md`
- Mock UI Reference: `0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
- Technical Specification: `0xcc/project-specs/core/miStudio_Specification.md`
- CLAUDE.md: `CLAUDE.md` (update with Project Standards)

**Next Steps:**
1. Update CLAUDE.md with Project Standards section (copy from Appendix)
2. Begin Feature PRD creation starting with Dataset Management
3. Set up development environment (Docker Compose)
4. Initialize backend and frontend projects

---

*This ADR establishes the foundational architecture for MechInterp Studio. All implementation decisions should reference this document for consistency and alignment with project goals.*
