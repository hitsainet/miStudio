# Technical Design Document: Feature Discovery

**Document ID:** 004_FTDD|Feature_Discovery
**Feature:** Interpretable Feature Extraction and Analysis from Trained SAEs
**PRD Reference:** 004_FPRD|Feature_Discovery.md
**ADR Reference:** 000_PADR|miStudio.md
**Status:** Draft
**Created:** 2025-10-06
**Last Updated:** 2025-10-06
**Owner:** miStudio Development Team

---

## 1. Executive Summary

This Technical Design Document defines the architecture and implementation approach for the Feature Discovery feature of miStudio. The feature enables users to extract, browse, analyze, and interpret learned features from trained Sparse Autoencoder (SAE) models, transforming abstract neural network representations into human-interpretable insights.

**Business Objective:** Make trained SAE models actionable by extracting interpretable features that explain model behavior, enabling mechanistic interpretability research and identifying specific features for downstream steering tasks.

**Technical Approach:**
- FastAPI backend with Celery for background feature extraction jobs
- PyTorch-based SAE inference on evaluation samples to identify max-activating examples
- PostgreSQL storage with JSONB for token sequences and activation values
- Full-text search (GIN indexes) for feature browsing
- Token-level activation highlighting with intensity-based color gradients
- React frontend matching Mock-embedded-interp-ui.tsx FeaturesPanel (lines 2159-2584) and FeatureDetailModal (lines 2587-2725) **exactly**
- Advanced analysis: logit lens, feature correlations, ablation studies

**Key Design Decisions:**
1. Store top-K (100) max-activating examples per feature (not all activations) to manage storage
2. Use JSONB for tokens and activations arrays (flexible, indexed, queryable)
3. Implement full-text search with PostgreSQL GIN indexes (not Elasticsearch for MVP)
4. Calculate token highlight intensity as `activation / max_activation` (normalized 0-1)
5. Cache expensive analyses (logit lens, correlations, ablation) in `feature_analysis_cache` table
6. Auto-generate feature labels using simple pattern matching (not LLM for MVP)

**Success Metrics:**
- Feature extraction completes in <5 minutes for 10,000 samples on Jetson Orin Nano
- Feature browser loads 16,384 features in <1 second
- Search filtering responds within 300ms (debounced)
- Token highlighting clearly shows activation intensity (validated by user testing)

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │FeaturesPanel │  │FeatureDetail │  │MaxActivatingExamples │ │
│  │(lines 2159-  │  │Modal         │  │(token highlighting)  │ │
│  │ 2584)        │  │(lines 2587-  │  │                      │ │
│  │              │  │ 2725)        │  │                      │ │
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
│  │              API Routes (/api/trainings/:id/features)     │   │
│  │  POST /trainings/:id/extract-features                     │   │
│  │  GET /trainings/:id/features (list with search/sort)      │   │
│  │  GET /features/:id (detail)                               │   │
│  │  PATCH /features/:id (update label)                       │   │
│  │  POST /features/:id/favorite                              │   │
│  │  GET /features/:id/examples (max-activating)              │   │
│  │  GET /features/:id/logit-lens                             │   │
│  │  GET /features/:id/correlations                           │   │
│  │  GET /features/:id/ablation                               │   │
│  └─────────────┬────────────────────────────────────────────┘   │
│                │                                                 │
│  ┌─────────────┴────────────────────────────────────────────┐   │
│  │         Services Layer                                    │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐   │   │
│  │  │ExtractionService│  │ AnalysisService              │   │   │
│  │  │- Run SAE        │  │ - Logit lens calculation     │   │   │
│  │  │- Identify top-K │  │ - Correlation computation    │   │   │
│  │  │- Auto-label     │  │ - Ablation analysis          │   │   │
│  │  └────────┬────────┘  └──────────┬───────────────────┘   │   │
│  └───────────┼────────────────────────┼─────────────────────┘   │
│              │                        │                          │
│  ┌───────────┴────────────────────────┴─────────────────────┐   │
│  │         Celery Workers (Background Extraction)            │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │ extract_features_task(training_id, config)          │ │   │
│  │  │   1. Load trained SAE checkpoint                    │ │   │
│  │  │   2. Load evaluation samples from dataset           │ │   │
│  │  │   3. For each sample:                               │ │   │
│  │  │      - Extract model activations                    │ │   │
│  │  │      - Pass through SAE encoder                     │ │   │
│  │  │   4. For each feature (SAE neuron):                 │ │   │
│  │  │      - Calculate activation_frequency               │ │   │
│  │  │      - Calculate interpretability_score             │ │   │
│  │  │      - Identify top-K max-activating examples       │ │   │
│  │  │   5. Auto-generate feature labels                   │ │   │
│  │  │   6. Store features + activations in database       │ │   │
│  │  │   7. Emit WebSocket progress events                 │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └───────────┬────────────────────────────────────────────┘   │
└──────────────┼───────────────────────────────────────────────┘
               │
┌──────────────┼───────────────────────────────────────────────┐
│         Data Layer                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  PostgreSQL     │  │  Redis       │  │  Filesystem     │  │
│  │  - extraction_  │  │  - Celery    │  │  - SAE          │  │
│  │    jobs         │  │  - Progress  │  │    checkpoints  │  │
│  │  - features     │  │              │  │                 │  │
│  │  - feature_     │  │              │  │                 │  │
│  │    activations  │  │              │  │                 │  │
│  │  - feature_     │  │              │  │                 │  │
│  │    analysis_    │  │              │  │                 │  │
│  │    cache        │  │              │  │                 │  │
│  └─────────────────┘  └──────────────┘  └─────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Feature Extraction Flow Diagram

```
User selects completed training
    ↓
Frontend: Display extraction configuration (evaluation_samples, top_k_examples)
    ↓
User clicks "Extract Features"
    ↓
Frontend: POST /api/trainings/:id/extract-features
    ↓
FastAPI: Create extraction_jobs record (status='queued')
    ↓
FastAPI: Enqueue Celery task → extract_features_task.delay(training_id)
    ↓
Celery Worker: Load SAE checkpoint from training
    ↓
Worker: Load dataset samples (evaluation_samples count)
    ↓
Worker: Update status='extracting', progress=0
    ↓
Worker: Feature extraction loop START
    ├─ FOR each sample in evaluation_samples:
    │   ├─ Tokenize sample
    │   ├─ Extract model activations (via Model Management)
    │   ├─ Pass activations through SAE encoder
    │   ├─ Store per-feature activations: encoded[neuron_index]
    │   ├─ Update progress: (sample_idx / total_samples) * 100
    │   └─ Every 5% progress: emit WebSocket 'extraction:progress'
    ├─
    ├─ FOR each feature (SAE neuron_index):
    │   ├─ Calculate activation_frequency: count(activations > 0.01) / total_samples
    │   ├─ Calculate max_activation_value: max(activations)
    │   ├─ Select top-K max-activating examples (sort by activation desc, take top-K)
    │   ├─ Calculate interpretability_score (heuristic based on consistency)
    │   ├─ Auto-generate feature label (pattern matching on tokens)
    │   ├─ Create feature record in database
    │   ├─ Store top-K examples in feature_activations table
    │   │   └─ Each example: {tokens: [], activations: [], max_activation, sample_index}
    │   └─ Update progress
    └─ Feature extraction loop END
    ↓
Worker: Update extraction status='completed', progress=100
    ↓
Worker: Emit WebSocket 'extraction:completed' with stats (total_features, avg_interpretability)
    ↓
Frontend: Receives completion event, transitions to feature browser
    ↓
Frontend: Display feature statistics (Features Found, Interpretable %, Activation Rate)
```

### Component Relationships

**Frontend Components (React):**
- **FeaturesPanel (Lines 2159-2584):** Training selector, extraction UI, feature browser with search/sort, pagination
- **FeatureDetailModal (Lines 2587-2725):** Feature details, editable label, statistics, tabs (Examples/Logit Lens/Correlations/Ablation)
- **MaxActivatingExamples (Lines 2728-2800+):** Token sequences with per-token activation highlighting
- **LogitLensView, FeatureCorrelations, AblationAnalysis:** Analysis tab content components

**Backend Services:**
- **ExtractionService:** Orchestrates feature extraction, calculates statistics, auto-generates labels
- **AnalysisService:** Performs expensive analyses (logit lens, correlations, ablation), caches results
- **FeatureService:** CRUD operations for features, search/filter/sort logic

**Celery Tasks:**
- **extract_features_task:** Main extraction job (long-running, 5+ minutes)

### Integration Points

**With SAE Training (003_FPRD - Upstream):**
- Requires completed training jobs with trained SAE checkpoints
- Loads SAE checkpoint file from `/data/trainings/{training_id}/checkpoints/`
- Uses training.hyperparameters for SAE architecture configuration

**With Dataset Management (001_FPRD - Upstream):**
- Requires tokenized datasets for evaluation samples
- Loads samples from dataset.storage_path

**With Model Management (002_FPRD - Upstream):**
- Requires models loaded in GPU memory for activation extraction
- Uses ModelRegistry.get_model() and ActivationExtractor

**With Model Steering (005_FPRD - Downstream):**
- Provides discovered features for intervention
- Features become selectable for steering operations

---

## 3. Technical Stack

### Core Technologies

**Backend:**
- **FastAPI 0.104+:** Async API framework
- **Celery 5.3+:** Distributed task queue for extraction
- **Redis 7.0+:** Message broker, progress tracking
- **PyTorch 2.0+:** SAE inference, activation extraction
- **NumPy 1.24+:** Statistical calculations (correlations)

**Frontend:**
- **React 18+:** UI framework
- **Zustand 4.4+:** State management
- **WebSocket (socket.io-client):** Real-time extraction progress
- **Lucide React:** Icons

**Database:**
- **PostgreSQL 14+:** Features, activations, analysis cache
- **JSONB:** Token and activation arrays (flexible, indexed)
- **GIN Indexes:** Full-text search on feature names/descriptions

### Technology Justifications

| Technology | Justification | Alternative Considered | Why Rejected |
|------------|--------------|------------------------|--------------|
| **JSONB for tokens/activations** | Flexible schema, efficient indexing, native PostgreSQL support | Separate normalized tables | Too many rows (billions), slower queries |
| **GIN indexes for search** | Native PostgreSQL full-text search, good performance for <100K features | Elasticsearch | Overkill for MVP, adds deployment complexity |
| **Top-K storage only** | Reduces storage from ~50GB to ~3GB per training | Store all activations | Impractical storage size, slower queries |
| **Client-side search debouncing** | Reduces API calls, feels instant | Server-side only | Slow UX, high backend load |
| **Simple auto-labeling** | Fast, no API costs, deterministic | GPT-4 API labeling | Expensive, slow, API dependency |
| **Analysis caching** | Analyses are expensive (30s+), rarely change | Real-time computation | Too slow for UX |

### Dependencies and Versions

```python
# requirements.txt additions for Feature Discovery
numpy==1.24.3
scipy==1.11.3  # For correlation calculations
```

**Version Requirements:**
- NumPy 1.24+ required for advanced indexing features
- SciPy required for Pearson correlation calculations

---

## 4. Data Design

### Database Schema Expansion

#### extraction_jobs Table

```sql
CREATE TABLE extraction_jobs (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'ext_abc123'
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Configuration
    config JSONB NOT NULL,  -- {evaluation_samples, top_k_examples}

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
        -- Values: 'queued', 'extracting', 'completed', 'failed'
    progress FLOAT DEFAULT 0,
    error_message TEXT,

    -- Statistics (populated on completion)
    total_features INTEGER,
    avg_interpretability FLOAT,
    avg_activation_freq FLOAT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT extraction_jobs_status_check CHECK (status IN
        ('queued', 'extracting', 'completed', 'failed'))
);

CREATE INDEX idx_extraction_jobs_training_id ON extraction_jobs(training_id);
CREATE INDEX idx_extraction_jobs_status ON extraction_jobs(status);
CREATE UNIQUE INDEX idx_extraction_jobs_training_unique
    ON extraction_jobs(training_id) WHERE status IN ('queued', 'extracting');
-- Ensures only one active extraction per training
```

**Config JSONB Structure:**
```json
{
  "evaluation_samples": 10000,
  "top_k_examples": 100
}
```

#### features Table (from 003_SPEC|Postgres lines 385-477)

```sql
CREATE TABLE features (
    id BIGSERIAL PRIMARY KEY,
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Feature identification
    neuron_index INTEGER NOT NULL,  -- Index in SAE layer (0-16383)
    layer INTEGER,  -- Transformer layer (if multi-layer SAE)

    -- Metadata
    name VARCHAR(500),  -- User-assigned or auto-generated name
    description TEXT,  -- User-provided or LLM-generated description
    label_source VARCHAR(50) DEFAULT 'auto',  -- 'auto' or 'user'

    -- Statistics
    activation_frequency FLOAT NOT NULL,  -- 0-1, how often activates > 0.01
    interpretability_score FLOAT,  -- 0-1, automated interpretability score
    max_activation_value FLOAT,  -- Maximum observed activation

    -- User flags
    is_favorite BOOLEAN DEFAULT FALSE,
    is_hidden BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite unique constraint
    CONSTRAINT features_unique_neuron UNIQUE (training_id, neuron_index, layer)
);

CREATE INDEX idx_features_training_id ON features(training_id);
CREATE INDEX idx_features_activation_freq ON features(training_id, activation_frequency DESC);
CREATE INDEX idx_features_interpretability ON features(training_id, interpretability_score DESC NULLS LAST);
CREATE INDEX idx_features_favorite ON features(is_favorite) WHERE is_favorite = TRUE;
CREATE INDEX idx_features_layer ON features(layer);

-- Full-text search on feature names and descriptions
CREATE INDEX idx_features_search ON features
    USING GIN(to_tsvector('english',
        COALESCE(name, '') || ' ' || COALESCE(description, '')));
```

**Storage Estimate:** ~1KB per feature × 16,384 features = ~16MB per training

#### feature_activations Table (from 003_SPEC|Postgres lines 437-477)

```sql
CREATE TABLE feature_activations (
    id BIGSERIAL PRIMARY KEY,
    feature_id BIGINT NOT NULL REFERENCES features(id) ON DELETE CASCADE,

    -- Dataset sample reference
    dataset_sample_id VARCHAR(255),  -- Reference to specific sample in dataset
    sample_index INTEGER,  -- Index in dataset

    -- Token-level data (stored as JSONB)
    tokens JSONB NOT NULL,  -- ["The", "quick", "brown", "fox", ...]
    activations JSONB NOT NULL,  -- [0.0, 0.2, 0.8, 0.3, ...]

    -- Max activation in this example
    max_activation FLOAT NOT NULL,

    -- Context (optional, for display)
    context_before TEXT,  -- Text before the max-activating token
    context_after TEXT,   -- Text after the max-activating token

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Critical: Partition this table by feature_id ranges for performance
-- See "Database Partitioning Strategy" section below

-- Indexes (on each partition)
CREATE INDEX idx_feature_activations_feature ON feature_activations(feature_id);
CREATE INDEX idx_feature_activations_max_activation
    ON feature_activations(feature_id, max_activation DESC);
```

**Storage Estimate:** ~2KB per activation example × 100 examples × 16,384 features = ~3.2GB per training

**JSONB Example:**
```json
{
  "tokens": ["The", "quick", "brown", "fox"],
  "activations": [0.01, 0.12, 0.89, 0.23]
}
```

#### feature_analysis_cache Table

```sql
CREATE TABLE feature_analysis_cache (
    id BIGSERIAL PRIMARY KEY,
    feature_id BIGINT NOT NULL REFERENCES features(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,  -- 'logit_lens', 'correlations', 'ablation'

    -- Cached results (JSONB)
    results JSONB NOT NULL,

    -- Cache metadata
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days'),

    CONSTRAINT feature_analysis_cache_unique UNIQUE (feature_id, analysis_type)
);

CREATE INDEX idx_feature_analysis_cache_feature ON feature_analysis_cache(feature_id);
CREATE INDEX idx_feature_analysis_cache_expires ON feature_analysis_cache(expires_at);

-- Cleanup job: DELETE FROM feature_analysis_cache WHERE expires_at < NOW();
```

**Results JSONB Examples:**

**Logit Lens:**
```json
{
  "top_tokens": ["the", "a", "an", "this", "that"],
  "probabilities": [0.23, 0.18, 0.15, 0.12, 0.09],
  "interpretation": "determiners and articles"
}
```

**Correlations:**
```json
{
  "correlations": [
    {"feature_id": 89, "name": "Sentence Start", "correlation": 0.87},
    {"feature_id": 137, "name": "Noun Phrases", "correlation": 0.76}
  ]
}
```

**Ablation:**
```json
{
  "perplexity_delta": 2.3,
  "impact_score": 0.76,
  "baseline_perplexity": 12.4,
  "ablated_perplexity": 14.7
}
```

### Database Partitioning Strategy

**Problem:** `feature_activations` table can grow to billions of rows (16K features × 100 examples × 100 trainings = 164M rows)

**Solution:** Partition by feature_id ranges (1000 features per partition)

```sql
-- Create partitioned table
CREATE TABLE feature_activations (
    -- ... columns as above
) PARTITION BY RANGE (feature_id);

-- Create partitions (adjust ranges based on feature count)
CREATE TABLE feature_activations_p0 PARTITION OF feature_activations
    FOR VALUES FROM (0) TO (10000);

CREATE TABLE feature_activations_p1 PARTITION OF feature_activations
    FOR VALUES FROM (10000) TO (20000);

-- ... continue for expected feature count

-- Create indexes on each partition automatically via template
```

**Benefits:**
- Queries filtered by feature_id only scan relevant partition
- Faster inserts (parallel writes to different partitions)
- Easier maintenance (drop old partitions to delete old trainings)

### Data Validation Strategy

**Feature Extraction Validation:**
- Validate training.status='completed' before extraction
- Validate SAE checkpoint file exists at expected path
- Validate evaluation_samples count > 0 and <= 100,000
- Validate top_k_examples > 0 and <= 1,000

**Feature Data Validation:**
- activation_frequency must be 0-1
- interpretability_score must be 0-1 or NULL
- neuron_index must be 0 to (SAE hidden_dim - 1)
- tokens and activations JSONB arrays must have same length

**Search Query Validation:**
- Sanitize search query to prevent SQL injection (use parameterized queries)
- Limit search query length to 500 characters
- Validate sort_by is one of: activation_freq, interpretability, feature_id

---

## 5. API Design

### RESTful API Conventions

**Base Path:** `/api/trainings/:id/features` and `/api/features/:id`

**HTTP Method Semantics:**
- GET: Retrieve feature(s), analysis results
- POST: Start extraction, toggle favorite
- PATCH: Update feature metadata
- DELETE: Unfavorite feature

**Status Codes:**
- 200 OK: Successful GET/PATCH/POST (actions)
- 201 Created: Successful POST (extraction start)
- 400 Bad Request: Validation error
- 404 Not Found: Feature/training not found
- 409 Conflict: Extraction already in progress
- 500 Internal Server Error: Unexpected error

### API Endpoint Specifications

#### GET /api/trainings/:id/extraction-status
**Purpose:** Get extraction status for training

**Response:** 200 OK
```json
{
  "extraction_id": "ext_abc123",
  "status": "extracting",
  "progress": 67.3,
  "started_at": "2025-10-06T10:00:00Z",
  "config": {
    "evaluation_samples": 10000,
    "top_k_examples": 100
  }
}
```

**Response (not started):** 200 OK
```json
{
  "extraction_id": null,
  "status": "not_started"
}
```

#### POST /api/trainings/:id/extract-features
**Purpose:** Start feature extraction

**Request Body:**
```json
{
  "evaluation_samples": 10000,
  "top_k_examples": 100
}
```

**Response:** 201 Created
```json
{
  "extraction_id": "ext_abc123",
  "status": "queued",
  "created_at": "2025-10-06T10:00:00Z"
}
```

**Error Response (already extracting):** 409 Conflict
```json
{
  "error": {
    "code": "EXTRACTION_IN_PROGRESS",
    "message": "Feature extraction already in progress for this training"
  }
}
```

#### GET /api/trainings/:id/features
**Purpose:** List features for training with search/filter/sort

**Query Parameters:**
- `search` (string): Filter by name (case-insensitive, full-text search)
- `sort_by` (enum): activation_freq | interpretability | feature_id (default: activation_freq)
- `sort_order` (enum): asc | desc (default: desc)
- `limit` (int): Max results (default 50, max 500)
- `offset` (int): Pagination offset (default 0)
- `is_favorite` (bool): Filter favorites only (optional)

**Response:** 200 OK
```json
{
  "features": [
    {
      "id": 1234,
      "neuron_index": 42,
      "name": "Sentiment Positive",
      "activation_frequency": 0.23,
      "interpretability_score": 0.94,
      "max_activation_value": 0.89,
      "is_favorite": false,
      "example_context": {
        "tokens": ["I", "really", "love", "this"],
        "activations": [0.02, 0.05, 0.89, 0.03]
      }
    }
  ],
  "total": 2048,
  "limit": 50,
  "offset": 0,
  "statistics": {
    "total_features": 2048,
    "interpretable_percentage": 87.3,
    "avg_activation_frequency": 0.124
  }
}
```

**Implementation Notes:**
- Use PostgreSQL `to_tsquery` for full-text search on `name` and `description`
- Include one example context per feature (first max-activating example) for table display
- Calculate statistics once on extraction completion, cache in extraction_jobs table

#### GET /api/features/:id
**Purpose:** Get feature details

**Response:** 200 OK
```json
{
  "id": 1234,
  "training_id": "tr_abc123",
  "neuron_index": 42,
  "layer": 12,
  "name": "Sentiment Positive",
  "description": null,
  "label_source": "auto",
  "activation_frequency": 0.23,
  "interpretability_score": 0.94,
  "max_activation_value": 0.89,
  "is_favorite": false,
  "active_samples": 2300,
  "created_at": "2025-10-06T10:30:00Z",
  "updated_at": "2025-10-06T10:30:00Z"
}
```

**Calculated Field:**
- `active_samples` = `activation_frequency * total_evaluation_samples`

#### PATCH /api/features/:id
**Purpose:** Update feature metadata

**Request Body:**
```json
{
  "name": "Updated Label",
  "description": "Custom description"
}
```

**Response:** 200 OK (updated feature object)

**Implementation Notes:**
- Set `label_source='user'` when user edits name
- Update `updated_at` timestamp

#### POST /api/features/:id/favorite
**Purpose:** Mark feature as favorite

**Response:** 200 OK
```json
{
  "is_favorite": true
}
```

#### DELETE /api/features/:id/favorite
**Purpose:** Unfavorite feature

**Response:** 200 OK
```json
{
  "is_favorite": false
}
```

#### GET /api/features/:id/examples
**Purpose:** Get max-activating examples for feature

**Query Parameters:**
- `limit` (int): Max examples (default 100, max 1000)

**Response:** 200 OK
```json
{
  "examples": [
    {
      "id": 5678,
      "tokens": ["The", "cat", "sat", "on", "the", "mat"],
      "activations": [0.01, 0.12, 0.08, 0.03, 0.02, 0.05],
      "max_activation": 0.12,
      "sample_index": 4521,
      "context_before": "Once upon a time,",
      "context_after": "in a sunny day."
    }
  ],
  "total": 100
}
```

**Implementation Notes:**
- Examples ordered by max_activation DESC
- Limit to top-K examples stored during extraction

#### GET /api/features/:id/logit-lens
**Purpose:** Get logit lens analysis (cached if available)

**Response:** 200 OK
```json
{
  "top_tokens": ["the", "a", "an", "this", "that"],
  "probabilities": [0.23, 0.18, 0.15, 0.12, 0.09],
  "interpretation": "determiners and articles",
  "computed_at": "2025-10-06T10:30:00Z"
}
```

**Implementation Notes:**
- Check `feature_analysis_cache` table first
- If not cached or expired, compute and cache
- Computation takes ~5-10 seconds

#### GET /api/features/:id/correlations
**Purpose:** Get correlated features (cached if available)

**Response:** 200 OK
```json
{
  "correlations": [
    {
      "feature_id": 89,
      "neuron_index": 89,
      "name": "Sentence Start",
      "correlation": 0.87
    },
    {
      "feature_id": 137,
      "neuron_index": 137,
      "name": "Noun Phrases",
      "correlation": 0.76
    }
  ],
  "computed_at": "2025-10-06T10:30:00Z"
}
```

**Implementation Notes:**
- Returns top 10 correlated features (correlation > 0.5)
- Pearson correlation calculated across evaluation set activations
- Expensive computation (~30 seconds for 16K features)

#### GET /api/features/:id/ablation
**Purpose:** Get ablation analysis (cached if available)

**Response:** 200 OK
```json
{
  "perplexity_delta": 2.3,
  "impact_score": 0.76,
  "baseline_perplexity": 12.4,
  "ablated_perplexity": 14.7,
  "computed_at": "2025-10-06T10:30:00Z"
}
```

**Implementation Notes:**
- Runs model inference twice (with/without feature)
- Expensive computation (~20 seconds)
- Impact score normalized 0-1: `perplexity_delta / max_perplexity_delta_observed`

### WebSocket Event Protocol

**Connection:** `socket.io` connection to `/ws`

**Event Subscription:**
```javascript
const socket = io('/ws');
socket.emit('subscribe', { channel: `extraction:${extractionId}` });
```

**Events:**

**extraction:progress** (every 5% progress)
```json
{
  "extraction_id": "ext_abc123",
  "progress": 67.3,
  "features_extracted": 11000,
  "total_features": 16384,
  "current_phase": "extracting_activations"
}
```

**extraction:completed**
```json
{
  "extraction_id": "ext_abc123",
  "total_features": 16384,
  "avg_interpretability": 0.873,
  "avg_activation_freq": 0.124,
  "interpretable_count": 14300,
  "completed_at": "2025-10-06T10:35:00Z"
}
```

**extraction:failed**
```json
{
  "extraction_id": "ext_abc123",
  "error_code": "EXTRACTION_FAILED",
  "error_message": "SAE checkpoint file not found",
  "failed_at": "2025-10-06T10:35:00Z"
}
```

---

## 6. Component Architecture

### Frontend Component Hierarchy

```
FeaturesPanel (Lines 2159-2584)
├── Training Selector
│   ├── Training Dropdown (completed trainings only)
│   └── Training Summary Cards (3-column: Model, Dataset, Encoder)
├── Feature Extraction & Analysis Panel
│   ├── Extraction Configuration (if not extracted)
│   │   ├── Evaluation Samples Input (default 10000)
│   │   ├── Top-K Examples Input (default 100)
│   │   └── Extract Features Button (lightning icon)
│   ├── Extraction Progress (if extracting)
│   │   ├── Progress Bar (emerald gradient, animated)
│   │   └── Status Message ("Processing activation patterns...")
│   └── Feature Browser (if extracted)
│       ├── Feature Statistics (3-column grid)
│       │   ├── Features Found (emerald)
│       │   ├── Interpretable % (blue)
│       │   └── Activation Rate % (purple)
│       ├── Search and Sort Controls
│       │   ├── Search Input (debounced 300ms)
│       │   ├── Sort By Dropdown
│       │   └── Sort Order Toggle (arrow rotates 180°)
│       ├── Feature Table
│       │   ├── Header Row (ID, Label, Example Context, Activation Freq, Interpretability, Actions)
│       │   └── Feature Rows[] (clickable, hover effect)
│       │       ├── Neuron Index (#42, monospace)
│       │       ├── Feature Name
│       │       ├── Example Context (tokens with highlighting)
│       │       ├── Activation Freq (emerald, 2 decimals)
│       │       ├── Interpretability (blue, 1 decimal)
│       │       └── Favorite Star (toggle)
│       └── Pagination
│           ├── Info ("Showing X of Y features")
│           └── Buttons (Previous, Next)
└── Feature Detail Modal (if feature selected)

FeatureDetailModal (Lines 2587-2725)
├── Modal Overlay (fixed inset-0, black/50 backdrop)
└── Modal Content (max-w-6xl, max-h-90vh, bg-slate-900)
    ├── Header
    │   ├── Title ("Feature #42")
    │   ├── Editable Label Input (saves on blur)
    │   ├── Close Button (X icon, top-right)
    │   └── Statistics Grid (4-column)
    │       ├── Activation Frequency (emerald)
    │       ├── Interpretability (blue)
    │       ├── Max Activation (purple)
    │       └── Active Samples (yellow)
    ├── Tabs
    │   ├── Examples Tab (active: emerald border-b-2)
    │   ├── Logit Lens Tab
    │   ├── Correlations Tab
    │   └── Ablation Tab
    └── Content Area (p-6, overflow-y-auto, flex-1)
        ├── Examples Tab Content
        │   └── MaxActivatingExamples Component
        ├── Logit Lens Tab Content
        │   └── LogitLensView Component
        ├── Correlations Tab Content
        │   └── FeatureCorrelations Component
        └── Ablation Tab Content
            └── AblationAnalysis Component

MaxActivatingExamples Component (Lines 2728-2800+)
├── Header ("Showing N examples")
└── Example Cards[]
    ├── Example Number ("Example 1")
    ├── Max Activation Value (emerald, 3 decimals)
    └── Token Sequence (flex-wrap, monospace)
        └── Token Spans[] (each with highlighting)
            ├── Background: rgba(16, 185, 129, intensity * 0.4)
            ├── Text Color: white if intensity > 0.6, else slate-300
            ├── Border: 1px emerald if intensity > 0.7
            └── Tooltip: "Activation: {value}" (3 decimals)
```

### Backend Service Architecture

```
ExtractionService
├── start_extraction(training_id, config) -> extraction_id
│   ├── Validate training status='completed'
│   ├── Check no active extraction exists
│   ├── Create extraction_jobs record
│   ├── Enqueue Celery task
│   └── Return extraction_id
├── extract_features_for_training(training_id, config)
│   ├── Load SAE checkpoint
│   ├── Load dataset samples
│   ├── Extract activations for all samples
│   ├── Calculate per-feature statistics
│   ├── Identify top-K examples
│   ├── Auto-generate labels
│   └── Store features + activations in DB
└── auto_label_feature(feature_activations) -> label
    ├── Analyze top 5 max-activating examples
    ├── Pattern matching (punctuation, questions, code, sentiment)
    └── Return descriptive label or "Feature {neuron_index}"

AnalysisService
├── calculate_logit_lens(feature_id) -> LogitLensResult
│   ├── Load feature
│   ├── Activate feature in SAE (high value, others zero)
│   ├── Decode to get reconstructed activation
│   ├── Pass through model to get output logits
│   ├── Apply softmax, get top 10 tokens
│   └── Cache result
├── calculate_correlations(feature_id) -> CorrelationResult
│   ├── Load all features for training
│   ├── Load activation vectors for all features
│   ├── Calculate Pearson correlation
│   ├── Return top 10 (correlation > 0.5)
│   └── Cache result
└── calculate_ablation(feature_id) -> AblationResult
    ├── Load feature and examples
    ├── Run model with feature active (baseline)
    ├── Run model with feature ablated (set to zero)
    ├── Calculate perplexity delta
    ├── Normalize to impact score
    └── Cache result

FeatureService
├── list_features(training_id, filters, sort, pagination) -> FeatureList
│   ├── Build query with search filter (GIN index)
│   ├── Apply sort (activation_freq, interpretability, feature_id)
│   ├── Apply pagination (limit, offset)
│   ├── Include one example_context per feature
│   └── Return paginated results + total count
├── get_feature_detail(feature_id) -> Feature
│   ├── Load feature record
│   ├── Calculate active_samples
│   └── Return detailed feature object
├── update_feature(feature_id, updates) -> Feature
│   ├── Validate updates
│   ├── Set label_source='user' if name changed
│   ├── Update updated_at timestamp
│   └── Return updated feature
└── toggle_favorite(feature_id, is_favorite) -> Boolean
    ├── Update is_favorite field
    └── Return new value
```

### Celery Task: extract_features_task

**Pseudocode:**
```python
@celery_app.task(bind=True)
def extract_features_task(self, training_id: str, config: dict):
    # 1. Setup
    training = db.query(Training).filter(Training.id == training_id).first()
    extraction = db.query(ExtractionJob).filter(
        ExtractionJob.training_id == training_id,
        ExtractionJob.status == 'queued'
    ).first()

    update_extraction_status(extraction.id, 'extracting', 0)

    # 2. Load SAE and dataset
    sae_checkpoint = load_checkpoint(training.final_checkpoint_id)
    sae = create_sae_from_checkpoint(sae_checkpoint)
    sae.eval()  # Set to evaluation mode

    dataset = load_dataset(training.dataset_id)
    samples = dataset.get_samples(count=config['evaluation_samples'])

    model = model_registry.get_model(training.model_id)

    # 3. Extract activations for all samples
    all_activations = {}  # {neuron_index: [activations_per_sample]}

    for sample_idx, sample in enumerate(samples):
        # Extract model activations
        input_ids = tokenize(sample)
        with torch.no_grad():
            model_activations = extract_activations(model, input_ids)

        # Pass through SAE encoder
        with torch.no_grad():
            encoded = sae.encoder(model_activations)  # Shape: (seq_len, hidden_dim)

        # Store per-neuron activations
        for neuron_idx in range(encoded.shape[-1]):
            if neuron_idx not in all_activations:
                all_activations[neuron_idx] = []

            all_activations[neuron_idx].append({
                'sample_idx': sample_idx,
                'tokens': input_ids.tolist(),
                'activations': encoded[:, neuron_idx].cpu().numpy().tolist(),
                'max_activation': encoded[:, neuron_idx].max().item()
            })

        # Update progress every 5%
        progress = (sample_idx / len(samples)) * 100
        if progress % 5 == 0:
            update_extraction_status(extraction.id, 'extracting', progress)
            emit_websocket('extraction:progress', {
                'extraction_id': extraction.id,
                'progress': progress
            })

    # 4. Process each feature
    total_features = len(all_activations)
    for neuron_idx, activations_list in all_activations.items():
        # Calculate statistics
        all_activations_flat = [act['max_activation'] for act in activations_list]
        activation_frequency = sum(1 for a in all_activations_flat if a > 0.01) / len(all_activations_flat)
        max_activation_value = max(all_activations_flat)

        # Calculate interpretability score (heuristic)
        interpretability_score = calculate_interpretability(activations_list)

        # Select top-K max-activating examples
        top_k_examples = sorted(activations_list, key=lambda x: x['max_activation'], reverse=True)[:config['top_k_examples']]

        # Auto-generate label
        label = auto_label_feature(top_k_examples)

        # Create feature record
        feature = Feature(
            training_id=training_id,
            neuron_index=neuron_idx,
            name=label,
            label_source='auto',
            activation_frequency=activation_frequency,
            interpretability_score=interpretability_score,
            max_activation_value=max_activation_value
        )
        db.add(feature)
        db.flush()  # Get feature.id

        # Store top-K examples
        for example in top_k_examples:
            activation_example = FeatureActivation(
                feature_id=feature.id,
                sample_index=example['sample_idx'],
                tokens=example['tokens'],
                activations=example['activations'],
                max_activation=example['max_activation']
            )
            db.add(activation_example)

        db.commit()

    # 5. Calculate statistics
    all_features = db.query(Feature).filter(Feature.training_id == training_id).all()
    total_features = len(all_features)
    avg_interpretability = np.mean([f.interpretability_score for f in all_features if f.interpretability_score])
    avg_activation_freq = np.mean([f.activation_frequency for f in all_features])
    interpretable_count = sum(1 for f in all_features if f.interpretability_score and f.interpretability_score > 0.8)

    # 6. Complete extraction
    update_extraction_status(extraction.id, 'completed', 100)
    update_extraction_stats(extraction.id, {
        'total_features': total_features,
        'avg_interpretability': avg_interpretability,
        'avg_activation_freq': avg_activation_freq
    })

    emit_websocket('extraction:completed', {
        'extraction_id': extraction.id,
        'total_features': total_features,
        'avg_interpretability': avg_interpretability,
        'avg_activation_freq': avg_activation_freq,
        'interpretable_count': interpretable_count
    })
```

---

## 7. State Management

### Application State Organization

**Global State (Zustand Store):**
```typescript
interface FeaturesStore {
  selectedTraining: string | null;
  extractionStatus: Record<string, ExtractionStatus>;
  features: Feature[];
  selectedFeature: Feature | null;
  searchQuery: string;
  sortBy: 'activation_freq' | 'interpretability' | 'feature_id';
  sortOrder: 'asc' | 'desc';
  favoritedFeatures: Set<number>;
  loading: boolean;

  // Actions
  selectTraining: (trainingId: string) => void;
  startExtraction: (trainingId: string, config: ExtractionConfig) => Promise<void>;
  fetchFeatures: (trainingId: string, filters: FeatureFilters) => Promise<void>;
  selectFeature: (feature: Feature) => void;
  updateFeatureLabel: (featureId: number, name: string) => Promise<void>;
  toggleFavorite: (featureId: number) => Promise<void>;
  setSearchQuery: (query: string) => void;
  setSortBy: (sortBy: string) => void;
  toggleSortOrder: () => void;

  // WebSocket subscription
  subscribeToExtractionUpdates: (extractionId: string) => void;
}
```

**Component State (React useState):**
- `activeTab`: 'examples' | 'logit-lens' | 'correlations' | 'ablation' (in FeatureDetailModal)
- `featureLabel`: Controlled input for editable label
- `showMetrics`: Boolean for collapsible sections

### State Flow Patterns

**Feature Extraction Flow:**
```
1. User selects completed training
   → Component calls featuresStore.selectTraining(trainingId)
   → Store checks extraction status via API

2. User configures extraction and clicks "Extract Features"
   → Component calls featuresStore.startExtraction(trainingId, config)
   → Store dispatches API call POST /api/trainings/:id/extract-features

3. API creates extraction job, returns extraction_id
   → Store subscribes to WebSocket channel: `extraction:${extractionId}`

4. Celery worker starts extraction
   → WebSocket receives 'extraction:progress' events (every 5%)
   → Store updates extractionStatus[trainingId].progress

5. Extraction completes
   → WebSocket receives 'extraction:completed' event
   → Store updates extractionStatus[trainingId].status='completed'
   → Store triggers fetchFeatures() to load feature list

6. Component re-renders
   → Shows feature statistics panel
   → Shows feature browser with search/sort
```

**Feature Search Flow:**
```
1. User types in search input
   → Component debounces input (300ms)
   → Component calls featuresStore.setSearchQuery(query)

2. Store updates searchQuery state
   → Triggers fetchFeatures() with updated filters

3. API returns filtered results
   → Store updates features[] array
   → Component re-renders table with filtered features

4. No results
   → Component shows "No features match your search" message
```

**Feature Favorite Toggle Flow:**
```
1. User clicks star icon in feature table
   → Component calls toggleFavorite(featureId)
   → Event propagation stopped (e.stopPropagation())

2. Store dispatches optimistic update
   → Immediately adds/removes featureId from favoritedFeatures Set
   → UI updates instantly (star fills/hollows)

3. Store dispatches API call POST/DELETE /api/features/:id/favorite
   → API updates database

4. If API fails
   → Store rolls back optimistic update
   → Shows error toast
```

### Side Effects Handling

**WebSocket Connection:**
```typescript
useEffect(() => {
  const socket = io('/ws');

  Object.keys(extractionStatus).forEach(trainingId => {
    const status = extractionStatus[trainingId];
    if (status.status === 'extracting') {
      socket.emit('subscribe', { channel: `extraction:${status.extraction_id}` });
    }
  });

  socket.on('extraction:progress', (data) => {
    featuresStore.updateExtractionProgress(data.extraction_id, data.progress);
  });

  socket.on('extraction:completed', (data) => {
    featuresStore.completeExtraction(data.extraction_id, data);
    featuresStore.fetchFeatures(selectedTraining);
  });

  return () => socket.disconnect();
}, [extractionStatus, selectedTraining]);
```

**Search Debouncing:**
```typescript
useEffect(() => {
  const timeoutId = setTimeout(() => {
    if (searchQuery !== prevSearchQuery) {
      featuresStore.fetchFeatures(selectedTraining, { search: searchQuery });
    }
  }, 300);

  return () => clearTimeout(timeoutId);
}, [searchQuery]);
```

---

## 8. Security Considerations

### Authentication & Authorization

**JWT-Based Authentication:**
- All endpoints require valid JWT token
- Token validated by FastAPI dependency middleware

**Permission Levels:**
- `features:read`: View features and analysis
- `features:write`: Update feature labels, toggle favorites
- `features:extract`: Start feature extraction jobs

### Input Validation

**Pydantic Schema Validation:**
```python
class ExtractionConfigRequest(BaseModel):
    evaluation_samples: int = Field(..., ge=1000, le=100000)
    top_k_examples: int = Field(..., ge=10, le=1000)

class FeatureSearchRequest(BaseModel):
    search: str = Field('', max_length=500)
    sort_by: Literal['activation_freq', 'interpretability', 'feature_id']
    sort_order: Literal['asc', 'desc']
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)
```

**Search Query Sanitization:**
- Use parameterized queries (no string concatenation)
- Limit search query length to 500 characters
- Use PostgreSQL `to_tsquery` with proper escaping

### Rate Limiting

- Feature extraction: Max 1 concurrent extraction per training
- Feature list API: Max 100 requests per minute per user
- Analysis endpoints: Max 10 requests per minute per user (expensive operations)

---

## 9. Performance & Scalability

### Performance Optimization

**Feature Extraction Speed:**
- Batch processing: Process 32 samples at once (maximize GPU utilization)
- Target: 10,000 samples in <5 minutes on Jetson Orin Nano
- Optimizations:
  - Use `torch.no_grad()` for inference (no gradients needed)
  - Clear GPU cache between batches
  - Use efficient tensor operations (avoid Python loops)

**Feature Browser Load Time:**
- Index on `activation_frequency`, `interpretability_score` for fast sorting
- GIN index on `to_tsvector(name || description)` for full-text search
- Pagination (default 50 features per page)
- Include only one example_context per feature (not all 100 examples)
- Target: Load 16,384 features in <1 second

**Search Response Time:**
- Client-side debouncing (300ms) to reduce API calls
- PostgreSQL full-text search with GIN index (fast for <100K features)
- Target: Search results within 300ms

**Analysis Caching:**
- Cache logit lens, correlations, ablation in `feature_analysis_cache` table
- Cache expiration: 7 days
- Recompute only if feature is updated or cache expired
- Target: Cached analysis loads in <100ms, computed in <30 seconds

### Database Query Optimization

**Efficient Queries:**
```sql
-- Feature list with search (uses GIN index)
SELECT * FROM features
WHERE training_id = 'tr_abc123'
  AND to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, ''))
      @@ to_tsquery('english', 'sentiment')
ORDER BY activation_frequency DESC
LIMIT 50 OFFSET 0;

-- Max-activating examples (uses index on feature_id, max_activation DESC)
SELECT * FROM feature_activations
WHERE feature_id = 1234
ORDER BY max_activation DESC
LIMIT 100;
```

**Connection Pooling:**
- Min connections: 5
- Max connections: 20
- Connection timeout: 30 seconds

### Scalability Considerations

**Storage Growth:**
- `features` table: ~16MB per training (16K features × 1KB)
- `feature_activations` table: ~3.2GB per training (16K features × 100 examples × 2KB)
- For 100 trainings: ~320GB storage (partitioning recommended)

**Partitioning Strategy:**
- Partition `feature_activations` by feature_id ranges (1000 features per partition)
- Drop old partitions when trainings are deleted
- Parallel queries across partitions

**Horizontal Scaling:**
- Stateless API servers (FastAPI can scale to N instances)
- Multiple Celery workers (1 per GPU for extraction)
- PostgreSQL read replicas for feature list queries

---

## 10. Testing Strategy

### Unit Tests

**Feature Statistics Calculation:**
```python
def test_activation_frequency_calculation():
    activations = [0.0, 0.02, 0.001, 0.05, 0.0, 0.03]
    threshold = 0.01
    frequency = calculate_activation_frequency(activations, threshold)
    assert frequency == 0.5  # 3 out of 6 above threshold

def test_interpretability_score():
    # High consistency = high interpretability
    consistent_examples = [
        {'tokens': ['not', 'good'], 'activations': [0.9, 0.2]},
        {'tokens': ['not', 'bad'], 'activations': [0.89, 0.18]},
        {'tokens': ['not', 'great'], 'activations': [0.91, 0.22]}
    ]
    score = calculate_interpretability_score(consistent_examples)
    assert score > 0.8

def test_token_highlight_intensity():
    activations = [0.1, 0.5, 0.9, 0.3]
    max_activation = 0.9
    intensities = [a / max_activation for a in activations]
    assert intensities == [0.111, 0.556, 1.0, 0.333]
```

**Auto-Labeling:**
```python
def test_auto_label_punctuation():
    examples = [
        {'tokens': ['.', ',', '!'], 'activations': [0.9, 0.85, 0.88]},
        {'tokens': [',', ';', '.'], 'activations': [0.92, 0.87, 0.90]}
    ]
    label = auto_label_feature(examples)
    assert label == "Punctuation"

def test_auto_label_fallback():
    examples = [
        {'tokens': ['random', 'tokens'], 'activations': [0.5, 0.4]}
    ]
    label = auto_label_feature(examples, neuron_index=42)
    assert label == "Feature 42"
```

### Integration Tests

**End-to-End Extraction Flow:**
```python
def test_extraction_flow(client, auth_headers):
    # Start extraction
    response = client.post("/api/trainings/tr_test_123/extract-features", json={
        "evaluation_samples": 1000,
        "top_k_examples": 10
    }, headers=auth_headers)

    assert response.status_code == 201
    extraction_id = response.json()["extraction_id"]

    # Wait for completion (or mock Celery task)
    # ... polling or WebSocket mock

    # Verify features created
    response = client.get(f"/api/trainings/tr_test_123/features", headers=auth_headers)
    assert response.status_code == 200
    features = response.json()["features"]
    assert len(features) > 0
```

**Feature Search:**
```python
def test_feature_search(client, auth_headers):
    response = client.get("/api/trainings/tr_test_123/features", params={
        "search": "sentiment",
        "sort_by": "interpretability",
        "sort_order": "desc",
        "limit": 10
    }, headers=auth_headers)

    assert response.status_code == 200
    features = response.json()["features"]
    assert all("sentiment" in f["name"].lower() for f in features)
    # Verify sorted by interpretability descending
    interpretabilities = [f["interpretability_score"] for f in features]
    assert interpretabilities == sorted(interpretabilities, reverse=True)
```

### Performance Tests

**Extraction Speed Benchmark:**
```python
def test_extraction_speed():
    start = time.time()
    extract_features_task('tr_test_123', {'evaluation_samples': 10000, 'top_k_examples': 100})
    elapsed = time.time() - start

    assert elapsed < 300, f"Extraction took {elapsed}s (>5 minutes)"
```

**Feature Browser Load Time:**
```python
def test_feature_list_load_time():
    start = time.time()
    response = client.get("/api/trainings/tr_test_123/features?limit=50")
    elapsed = time.time() - start

    assert elapsed < 1.0, f"Feature list load took {elapsed}s (>1 second)"
```

---

## 11. Deployment & DevOps

### Deployment Pipeline

**CI/CD Stages:**
1. **Test Stage:** Run unit + integration tests
2. **Build Stage:** Build Docker image with PyTorch + PostgreSQL drivers
3. **Deploy Staging:** Deploy to Jetson Orin Nano dev board
4. **Manual Approval:** QA validation (test feature extraction)
5. **Deploy Production:** Deploy to production Jetson

**Environment Configuration:**
```bash
# .env.production
DATABASE_URL=postgresql://user:pass@localhost:5432/mistudio
REDIS_URL=redis://localhost:6379/0
MAX_CONCURRENT_EXTRACTIONS=1
FEATURE_EXTRACTION_TIMEOUT=600  # 10 minutes
ANALYSIS_CACHE_EXPIRY_DAYS=7
LOG_LEVEL=INFO
```

### Monitoring & Logging

**Metrics:**
- Extraction completion rate (target: >95%)
- Extraction duration (target: <5 minutes for 10K samples)
- Feature browser load time (target: <1 second)
- Search latency (p95: <300ms)
- Analysis cache hit rate (target: >80%)

**Logging:**
- Application logs: JSON structured logs to stdout
- Extraction logs: Progress, errors, statistics
- Analysis logs: Computation time, cache hits/misses

**Alerts:**
- Extraction failure rate >5%: Error alert
- Feature browser load time >3 seconds: Warning alert
- Analysis computation time >60 seconds: Warning alert

---

## 12. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **Extraction Too Slow** | Medium | High | Batch processing, GPU optimization, progress updates to show work happening |
| **Storage Exhaustion** | High | Medium | Retention policy, partition old data, allow users to delete features |
| **Search Performance Degradation** | Medium | Medium | GIN indexes, pagination, client-side debouncing, consider Elasticsearch post-MVP |
| **Auto-Labels Not Useful** | High | Medium | Easy label editing, show auto vs user labels, consider LLM labeling post-MVP |
| **Analysis Computation Too Slow** | Medium | High | Aggressive caching (7 days), show loading spinner, compute on-demand only |

### Dependencies & Blockers

**Dependency 1: SAE Training (003_FPRD)**
- **Blocker Severity:** Critical
- **Requirement:** Completed training jobs with trained SAE checkpoints
- **Status:** Complete (003_FTDD finished)

**Dependency 2: Dataset Management (001_FPRD)**
- **Blocker Severity:** Critical
- **Requirement:** Tokenized datasets for evaluation samples
- **Status:** Complete (001_FTDD finished)

**Dependency 3: Model Management (002_FPRD)**
- **Blocker Severity:** Critical
- **Requirement:** Models loaded in GPU memory for activation extraction
- **Status:** Complete (002_FTDD finished)

---

## 13. Development Phases

### Phase 1: Core Feature Extraction (Weeks 1-2)

**Objectives:**
- Implement extraction_jobs, features, feature_activations database tables
- Implement feature extraction Celery task
- Implement training selector and extraction configuration UI
- Implement WebSocket progress updates

**Deliverables:**
- POST /api/trainings/:id/extract-features endpoint
- extract_features_task Celery worker
- FeaturesPanel training selector (lines 2159-2351)
- Extraction configuration UI (lines 2359-2414)
- Progress bar with WebSocket updates

**Acceptance Criteria:**
- Can select completed training
- Can configure extraction (evaluation_samples, top_k_examples)
- Can start extraction, see progress bar
- Extraction completes, features stored in database

**Estimated Effort:** 60-70 hours (2 developers × 1.5-2 weeks)

---

### Phase 2: Feature Browser (Week 3)

**Objectives:**
- Implement feature list API with search/filter/sort
- Implement feature browser UI with table
- Implement token highlighting visualization
- Implement favorite toggle

**Deliverables:**
- GET /api/trainings/:id/features endpoint
- Feature browser UI (lines 2434-2567)
- Token highlighting (lines 2498-2518)
- Favorite star toggle (lines 2531-2539)
- Pagination UI (lines 2554-2566)

**Acceptance Criteria:**
- Features display in table with search/sort
- Token highlighting shows activation intensity clearly
- Can favorite/unfavorite features
- Pagination works correctly

**Estimated Effort:** 40-50 hours (2 developers × 1 week)

---

### Phase 3: Feature Detail Modal (Week 4)

**Objectives:**
- Implement feature detail API endpoints
- Implement FeatureDetailModal UI with tabs
- Implement MaxActivatingExamples component
- Implement editable label

**Deliverables:**
- GET /api/features/:id endpoint
- PATCH /api/features/:id endpoint
- GET /api/features/:id/examples endpoint
- FeatureDetailModal component (lines 2587-2725)
- MaxActivatingExamples component (lines 2728-2800+)

**Acceptance Criteria:**
- Can click feature row to open modal
- Modal shows feature details and statistics
- Can edit feature label, saves on blur
- Examples tab shows max-activating examples with token highlighting
- Can close modal (X button, Escape key, backdrop click)

**Estimated Effort:** 35-45 hours (2 developers × 1 week)

---

### Phase 4: Advanced Analysis (Week 5)

**Objectives:**
- Implement logit lens calculation and caching
- Implement feature correlation calculation
- Implement ablation analysis
- Implement analysis tab UIs

**Deliverables:**
- GET /api/features/:id/logit-lens endpoint
- GET /api/features/:id/correlations endpoint
- GET /api/features/:id/ablation endpoint
- feature_analysis_cache table
- LogitLensView, FeatureCorrelations, AblationAnalysis components

**Acceptance Criteria:**
- Logit lens shows top predicted tokens
- Correlations show top correlated features
- Ablation shows perplexity delta and impact score
- Results cached for performance

**Estimated Effort:** 30-40 hours (2 developers × 1 week)

---

### Phase 5: Testing & Polish (Week 6)

**Objectives:**
- Write unit tests (70% coverage)
- Write integration tests
- Performance testing and optimization
- UI polish to match Mock design exactly

**Deliverables:**
- Unit test suite (pytest)
- Integration test suite (API tests)
- Performance benchmarks
- Polished UI matching Mock-embedded-interp-ui.tsx

**Acceptance Criteria:**
- Unit test coverage >70%
- All integration tests passing
- Extraction speed <5 minutes for 10K samples
- Feature browser loads in <1 second
- UI matches Mock design exactly (colors, layout, behavior)

**Estimated Effort:** 30-40 hours (2 developers × 1 week)

---

### Total Estimated Timeline: 6 weeks (2 developers)

**Critical Path:**
- Phase 1 (extraction) blocks all other phases
- Phase 2 (browser) blocks Phase 3 (modal)
- Phase 4 (analysis) can run parallel to Phase 3

**Milestone Definitions:**
- **M1 (Week 2):** Feature extraction functional, features stored in DB
- **M2 (Week 3):** Feature browser functional, search/sort/pagination working
- **M3 (Week 4):** Feature detail modal functional, examples displayed
- **M4 (Week 5):** Advanced analysis functional, all tabs working
- **M5 (Week 6):** Feature complete, tested, polished, matches Mock UI exactly

---

## 14. Appendix

### A. Auto-Labeling Heuristics

**Pattern Matching Algorithm:**
```python
def auto_label_feature(top_examples, neuron_index):
    # Extract tokens from top 5 examples
    all_tokens = []
    for example in top_examples[:5]:
        tokens = example['tokens']
        activations = example['activations']
        max_activation = max(activations)

        # Get high-activation tokens (intensity > 0.7)
        high_activation_tokens = [
            tokens[i] for i, a in enumerate(activations)
            if a / max_activation > 0.7
        ]
        all_tokens.extend(high_activation_tokens)

    # Pattern matching
    if all(t in string.punctuation for t in all_tokens[:10]):
        return "Punctuation"

    if any(t.lower() in ['what', 'how', 'why', 'when', 'where', 'who'] for t in all_tokens):
        return "Question Pattern"

    if any(t.lower() in ['def', 'function', 'class', 'import', 'return'] for t in all_tokens):
        return "Code Syntax"

    if any(t.lower() in ['love', 'great', 'amazing', 'wonderful', 'excellent'] for t in all_tokens):
        return "Sentiment Positive"

    if any(t.lower() in ['hate', 'terrible', 'awful', 'horrible', 'bad'] for t in all_tokens):
        return "Sentiment Negative"

    if any(t.lower() in ['not', 'no', 'never', "n't", 'neither'] for t in all_tokens):
        return "Negation Logic"

    if any(t.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our'] for t in all_tokens):
        return "Pronouns First Person"

    # Fallback
    return f"Feature {neuron_index}"
```

### B. Token Highlighting Visualization Algorithm

**JavaScript Implementation:**
```javascript
function renderTokenHighlighting(tokens, activations) {
  const maxActivation = Math.max(...activations);

  return tokens.map((token, idx) => {
    const activation = activations[idx];
    const intensity = Math.min(activation / maxActivation, 1.0);

    const style = {
      backgroundColor: `rgba(16, 185, 129, ${intensity * 0.4})`, // emerald-500
      color: intensity > 0.6 ? '#fff' : '#cbd5e1', // white or slate-300
      border: intensity > 0.7 ? '1px solid rgba(16, 185, 129, 0.5)' : 'none',
      padding: '2px 4px',
      borderRadius: '4px',
      fontFamily: 'monospace',
      fontSize: '12px'
    };

    return (
      <span
        key={idx}
        style={style}
        title={`Activation: ${activation.toFixed(3)}`}
      >
        {token}
      </span>
    );
  });
}
```

### C. Interpretability Score Calculation

**Heuristic Algorithm:**
```python
def calculate_interpretability_score(top_examples):
    """
    Calculate interpretability score based on:
    1. Consistency of activation patterns
    2. Sparsity of activation (not too sparse, not too dense)
    """
    # 1. Consistency: How similar are the top examples?
    activation_patterns = [example['activations'] for example in top_examples[:10]]
    consistency = calculate_pattern_consistency(activation_patterns)

    # 2. Sparsity: What fraction of tokens activate?
    all_activations = [a for example in top_examples for a in example['activations']]
    sparsity = sum(1 for a in all_activations if a > 0.01) / len(all_activations)

    # Ideal sparsity: 10-30% (too sparse = too specific, too dense = too general)
    sparsity_score = 1.0 - abs(0.2 - sparsity) / 0.8

    # Combine scores
    interpretability = (consistency * 0.7) + (sparsity_score * 0.3)
    return max(0.0, min(1.0, interpretability))
```

### D. Logit Lens Calculation

**PyTorch Implementation:**
```python
def calculate_logit_lens(feature_id):
    feature = db.query(Feature).filter(Feature.id == feature_id).first()
    training = feature.training
    sae = load_sae(training.final_checkpoint_id)
    model = model_registry.get_model(training.model_id)

    # 1. Activate feature in SAE encoder output
    feature_vector = torch.zeros(sae.hidden_dim)
    feature_vector[feature.neuron_index] = 10.0  # High activation

    # 2. Pass through SAE decoder
    with torch.no_grad():
        reconstructed_activation = sae.decoder(feature_vector)

    # 3. Pass through model to get output logits
    with torch.no_grad():
        logits = model.lm_head(reconstructed_activation)

    # 4. Apply softmax, get top 10 tokens
    probabilities = F.softmax(logits, dim=-1)
    top_probs, top_indices = probabilities.topk(10)

    top_tokens = [model.tokenizer.decode(idx) for idx in top_indices]
    top_probs = top_probs.tolist()

    # 5. Generate interpretation (simple heuristic)
    interpretation = generate_interpretation(top_tokens)

    return {
        'top_tokens': top_tokens,
        'probabilities': top_probs,
        'interpretation': interpretation
    }
```

---

**Document End**
**Total Sections:** 14
**Estimated Implementation Time:** 6 weeks (2 developers)
**Review Status:** Pending stakeholder review
