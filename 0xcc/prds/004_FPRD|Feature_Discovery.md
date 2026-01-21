# Feature PRD: Feature Discovery

**Document ID:** 004_FPRD|Feature_Discovery
**Version:** 1.2 (Batch Extraction & Live Metrics)
**Last Updated:** 2026-01-21
**Status:** Implemented
**Priority:** P0 (Core Feature)

---

## 1. Overview

### 1.1 Purpose
Enable users to extract, browse, and label interpretable features from trained SAEs, including automated labeling via GPT-4o.

### 1.2 User Problem
Researchers need to understand SAE features but face challenges with:
- Extracting features from large datasets efficiently
- Finding patterns across thousands of features
- Creating meaningful labels for interpretation
- Organizing features for analysis

### 1.3 Solution
A comprehensive feature discovery system with batch extraction, statistical analysis, dual labeling (semantic + category), and GPT-4o auto-labeling.

---

## 2. Functional Requirements

### 2.1 Feature Extraction
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | Extract top activations per feature | Implemented |
| FR-1.2 | Batch processing with GPU optimization | Implemented |
| FR-1.3 | Configurable activation threshold | Implemented |
| FR-1.4 | Context window capture (tokens before/after) | Implemented |
| FR-1.5 | Token filtering during extraction | Implemented |
| FR-1.6 | Real-time progress via WebSocket | Implemented |
| FR-1.7 | Live progress metrics (samples/second, ETA) | Implemented |
| FR-1.8 | Time-based progress emission (every 2 seconds) | Implemented |
| FR-1.9 | Features in heap count for progress graphs | Implemented |
| FR-1.10 | Heap examples count for collection rate graphs | Implemented |

### 2.2 Feature Browser
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | List features with pagination | Implemented |
| FR-2.2 | Search by label, category, statistics | Implemented |
| FR-2.3 | Sort by activation frequency, max, mean | Implemented |
| FR-2.4 | Filter by label status, category | Implemented |
| FR-2.5 | View feature detail modal | Implemented |

### 2.3 Feature Statistics
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | Activation frequency | Implemented |
| FR-3.2 | Max/mean activation value | Implemented |
| FR-3.3 | Interpretability score | Implemented |
| FR-3.4 | Token distribution analysis | Implemented |

### 2.4 Dual Labeling System
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | Semantic label (free-text description) | Implemented |
| FR-4.2 | Category label (hierarchical taxonomy) | Implemented |
| FR-4.3 | Manual label editing | Implemented |
| FR-4.4 | Label confidence score | Implemented |
| FR-4.5 | Label source tracking (manual/auto/imported) | Implemented |

### 2.5 Auto-Labeling (Sub-feature)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-5.1 | GPT-4o integration via OpenAI API | Implemented |
| FR-5.2 | Batch auto-labeling with progress | Implemented |
| FR-5.3 | Configurable prompt templates | Implemented |
| FR-5.4 | Labeling prompt template management | Implemented |
| FR-5.5 | Stop/resume labeling job | Implemented |

### 2.6 Extraction Templates (Sub-feature)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-6.1 | Save extraction configuration | Implemented |
| FR-6.2 | Load template to populate form | Implemented |
| FR-6.3 | Template favorites and usage count | Implemented |
| FR-6.4 | Import/export templates (JSON) | Implemented |
| FR-6.5 | "Save as Template" button in extraction modal | Implemented |
| FR-6.6 | Auto-generated template names from config | Implemented |

### 2.7 Labeling Prompt Templates (Sub-feature)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-7.1 | Create custom labeling prompts | Implemented |
| FR-7.2 | Variable substitution in prompts | Implemented |
| FR-7.3 | System/user prompt separation | Implemented |
| FR-7.4 | Template favorites | Implemented |

### 2.8 NLP Analysis (Sub-feature - Added Dec 2025)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-8.1 | Part-of-speech tagging for tokens | Implemented |
| FR-8.2 | Lemmatization for semantic grouping | Implemented |
| FR-8.3 | Named entity recognition | Implemented |
| FR-8.4 | BPE token reconstruction | Implemented |
| FR-8.5 | spaCy integration | Implemented |

### 2.9 Local LLM Integration (Sub-feature - Added Dec 2025)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-9.1 | Ollama backend for auto-labeling | Implemented |
| FR-9.2 | Configurable LLM provider selection | Implemented |
| FR-9.3 | Same prompt template compatibility | Implemented |
| FR-9.4 | Offline operation support | Implemented |

### 2.10 Batch Extraction (Sub-feature - Added Jan 2026)
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-10.1 | Select multiple SAEs for batch extraction | Implemented |
| FR-10.2 | Sequential processing (one SAE at a time) | Implemented |
| FR-10.3 | Auto-continue after NLP analysis completion | Implemented |
| FR-10.4 | Batch progress tracking (position/total) | Implemented |
| FR-10.5 | Continue batch even if one job's NLP fails | Implemented |
| FR-10.6 | Batch ID linking related extraction jobs | Implemented |

---

## 3. Feature Detail Modal

### 3.1 Components
- **Summary**: Feature index, activation stats, labels
- **Top Activations**: Examples with context highlighting
- **Token Analysis**: Distribution of activating tokens
- **Edit Labels**: Manual label input

### 3.2 Token Highlighting
```
Context: "The cat sat on the [mat] and looked around"
                                 ^^^
         Activation: 4.52 at position 6
```

---

## 4. User Interface

### 4.1 Features Panel
```
┌─────────────────────────────────────────────────────────────┐
│ Features                    [Extract] [Auto-Label] [Export] │
├─────────────────────────────────────────────────────────────┤
│ SAE: [gemma-2b-layer12 ▾]  Search: [________]  Filter: [▾] │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Feature #42                                    [Edit]   │ │
│ │ Label: "expressions of romantic love"                   │ │
│ │ Category: Emotion > Positive > Love                     │ │
│ │ Freq: 0.023% | Max: 8.4 | Mean: 2.1                    │ │
│ │ Top tokens: love, heart, romance, dear, beloved         │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Feature #156                                   [Edit]   │ │
│ │ Label: [unlabeled]                                      │ │
│ │ Freq: 0.045% | Max: 6.2 | Mean: 1.8                    │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Extraction Modal
- SAE selection
- Dataset/tokenization selection
- Activation threshold
- Context window size
- Token filters
- Template load/save

### 4.3 Auto-Label Modal
- Feature selection (all, unlabeled, custom range)
- Prompt template selection
- Batch size configuration
- Progress display
- Stop/resume controls

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/features` | GET | List features with filtering |
| `/api/v1/features/{id}` | GET | Get feature details |
| `/api/v1/features/{id}` | PATCH | Update labels |
| `/api/v1/features/extraction` | POST | Start extraction job |
| `/api/v1/features/extraction/{id}` | GET | Get extraction status |
| `/api/v1/features/labeling` | POST | Start auto-labeling |
| `/api/v1/features/labeling/{id}` | GET | Get labeling status |
| `/api/v1/features/export` | POST | Export to JSON |
| `/api/v1/extraction-templates` | GET/POST | Template CRUD |
| `/api/v1/labeling-prompt-templates` | GET/POST | Prompt CRUD |

---

## 6. Data Model

### 6.1 Feature Table
```sql
CREATE TABLE features (
    id UUID PRIMARY KEY,
    training_id UUID REFERENCES trainings(id),
    external_sae_id UUID REFERENCES external_saes(id),
    neuron_index INTEGER NOT NULL,
    label VARCHAR(500),  -- Semantic label
    category VARCHAR(255),  -- Category label
    label_confidence FLOAT,
    label_source VARCHAR(50),  -- manual, auto, imported
    statistics JSONB,  -- frequency, max, mean, interpretability
    top_tokens JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 6.2 FeatureActivation Table
```sql
CREATE TABLE feature_activations (
    id UUID PRIMARY KEY,
    feature_id UUID REFERENCES features(id) ON DELETE CASCADE,
    activation_value FLOAT NOT NULL,
    token VARCHAR(255),
    token_id INTEGER,
    context_before TEXT,
    context_after TEXT,
    position INTEGER,
    sample_index INTEGER,
    created_at TIMESTAMP
);
```

### 6.3 ExtractionJob Table
```sql
CREATE TABLE extraction_jobs (
    id UUID PRIMARY KEY,
    sae_id UUID,  -- training_id or external_sae_id
    dataset_id UUID REFERENCES datasets(id),
    config JSONB NOT NULL,
    status VARCHAR(50),
    progress FLOAT,
    features_found INTEGER,
    error_message TEXT,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

---

## 7. WebSocket Channels

| Channel | Events | Payload |
|---------|--------|---------|
| `extraction/{id}` | `extraction:progress` | See detailed payload below |
| `extraction/{id}` | `extraction:completed` | `{features_count, nlp_status?}` |
| `extraction/{id}` | `extraction:failed` | `{error_message}` |
| `labeling/{id}` | `progress` | `{progress, labeled_count}` |
| `labeling/{id}` | `results` | `{feature_id, label, confidence}` |
| `labeling/{id}` | `completed` | `{total_labeled}` |

### 7.1 Extraction Progress Payload (Jan 2026)
```json
{
  "extraction_id": "uuid",
  "status": "extracting",
  "sae_id": "uuid",
  "progress": 0.45,
  "features_extracted": 7372,
  "total_features": 16384,
  "current_batch": 23,
  "total_batches": 50,
  "samples_processed": 23000,
  "total_samples": 50000,
  "samples_per_second": 125.5,
  "eta_seconds": 216,
  "features_in_heap": 14521,
  "heap_examples_count": 72605
}
```

---

## 8. Key Files

### Backend
- `backend/src/services/extraction_service.py` - Extraction logic
- `backend/src/services/feature_service.py` - Feature management
- `backend/src/services/labeling_service.py` - Labeling orchestration
- `backend/src/services/openai_labeling_service.py` - GPT-4o integration
- `backend/src/workers/extraction_tasks.py` - Celery extraction task
- `backend/src/api/v1/endpoints/features.py` - API routes

### Frontend
- `frontend/src/components/panels/FeaturesPanel.tsx` - Main panel
- `frontend/src/components/features/FeatureDetailModal.tsx` - Detail view
- `frontend/src/components/features/StartExtractionModal.tsx` - Extraction config
- `frontend/src/components/features/TokenHighlight.tsx` - Context display
- `frontend/src/components/panels/ExtractionTemplatesPanel.tsx` - Templates
- `frontend/src/components/panels/LabelingPromptTemplatesPanel.tsx` - Prompts

---

## 9. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| SAE Training | Provides trained SAE |
| SAE Management | Can extract from external SAEs |
| Dataset Management | Provides extraction data |
| Model Steering | Features used for steering |
| Neuronpedia Export | Features exported |

---

## 10. Testing Checklist

- [x] Extract features from trained SAE
- [x] Extract from external SAE
- [x] Feature browser pagination
- [x] Search features by label
- [x] Filter by statistics
- [x] Manual label editing
- [x] Auto-labeling with GPT-4o
- [x] Labeling prompt templates
- [x] Extraction templates
- [x] Token highlighting
- [x] Export features to JSON
- [x] Batch extraction with multiple SAEs (Jan 2026)
- [x] Live progress metrics display (Jan 2026)
- [x] Save as Template from extraction modal (Jan 2026)
- [x] Sequential batch processing with NLP continuation (Jan 2026)

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/004_FTDD|Feature_Discovery.md) | [TID](../tids/004_FTID|Feature_Discovery.md)*
