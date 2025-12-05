# Feature PRD: Neuronpedia Export

**Document ID:** 007_FPRD|Neuronpedia_Export
**Version:** 1.0 (MVP Complete)
**Last Updated:** 2025-12-05
**Status:** Implemented
**Priority:** P1 (Important Feature)

---

## 1. Overview

### 1.1 Purpose
Enable users to export SAE findings in Neuronpedia-compatible format for community sharing and collaboration.

### 1.2 User Problem
Researchers need to share SAE findings but face challenges with:
- Creating Neuronpedia-compatible export packages
- Computing logit lens data for features
- Generating activation histograms
- Packaging weights in SAELens format

### 1.3 Solution
A comprehensive export system that creates complete Neuronpedia-compatible packages with configurable data inclusion.

---

## 2. Functional Requirements

### 2.1 Export Configuration
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | Select features to export (all, extracted, custom) | Implemented |
| FR-1.2 | Include/exclude logit lens data | Implemented |
| FR-1.3 | Include/exclude activation histograms | Implemented |
| FR-1.4 | Include/exclude top tokens | Implemented |
| FR-1.5 | Include/exclude explanations | Implemented |
| FR-1.6 | Include SAELens-compatible weights | Implemented |

### 2.2 Data Computation
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | Compute logit lens (promoted/suppressed tokens) | Implemented |
| FR-2.2 | Generate activation histograms | Implemented |
| FR-2.3 | Aggregate top activating tokens | Implemented |
| FR-2.4 | Package feature explanations | Implemented |

### 2.3 Export Job Management
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | Queue export jobs | Implemented |
| FR-3.2 | Real-time progress via WebSocket | Implemented |
| FR-3.3 | Download completed export | Implemented |
| FR-3.4 | Cancel in-progress export | Implemented |
| FR-3.5 | List export history | Implemented |

---

## 3. Export Package Structure

### 3.1 Complete Package
```
export.zip/
├── metadata.json           # SAE configuration and export info
├── README.md               # Human-readable documentation
├── features/               # Individual feature JSONs
│   ├── 0.json
│   ├── 1.json
│   ├── 2.json
│   └── ...
├── explanations/
│   └── explanations.json   # All feature labels
└── saelens/
    ├── cfg.json            # SAELens-compatible config
    └── sae_weights.safetensors  # Model weights
```

### 3.2 metadata.json
```json
{
  "sae_id": "uuid",
  "sae_name": "gemma-2b-layer12-8x",
  "model_name": "google/gemma-2-2b",
  "layer": 12,
  "d_in": 2304,
  "d_sae": 18432,
  "architecture": "standard",
  "export_date": "2025-12-05T00:00:00Z",
  "feature_count": 993,
  "includes": {
    "logit_lens": true,
    "histograms": true,
    "top_tokens": true,
    "explanations": true,
    "saelens_weights": true
  }
}
```

### 3.3 Feature JSON (features/0.json)
```json
{
  "feature_index": 0,
  "label": "expressions of romantic love",
  "category": "Emotion > Positive > Love",
  "statistics": {
    "activation_frequency": 0.023,
    "max_activation": 8.4,
    "mean_activation": 2.1
  },
  "logit_lens": {
    "top_positive": [
      {"token": "love", "value": 0.85},
      {"token": "heart", "value": 0.72}
    ],
    "top_negative": [
      {"token": "hate", "value": -0.45}
    ]
  },
  "histogram": {
    "bin_edges": [0, 0.5, 1.0, ...],
    "counts": [100, 50, 25, ...]
  },
  "top_tokens": [
    {"token": "love", "count": 45, "mean_activation": 3.2}
  ],
  "activations": [
    {
      "token": "love",
      "value": 4.52,
      "context_before": "I really ",
      "context_after": " you so much"
    }
  ]
}
```

---

## 4. User Interface

### 4.1 Export Modal
```
┌─────────────────────────────────────────────────────────────┐
│ Export to Neuronpedia                              [x]      │
├─────────────────────────────────────────────────────────────┤
│ SAE: gemma-2b-layer12-8x                                    │
│ Features Available: 993 extracted                           │
├─────────────────────────────────────────────────────────────┤
│ Feature Selection                                           │
│ (●) All features (993)                                      │
│ ( ) Extracted features only                                 │
│ ( ) Custom range: [___] to [___]                           │
├─────────────────────────────────────────────────────────────┤
│ Include Data                                                │
│ [✓] Logit lens data (promoted/suppressed tokens)           │
│     Top K: [20]                                             │
│ [✓] Activation histograms                                  │
│     Bins: [50]                                              │
│ [✓] Top activating tokens                                  │
│     Top K: [50]                                             │
│ [✓] Feature explanations/labels                            │
│ [✓] SAELens-compatible weights                             │
├─────────────────────────────────────────────────────────────┤
│ Estimated Size: ~150 MB                                     │
│                                                             │
│                              [Cancel] [Start Export]        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Progress View
```
┌─────────────────────────────────────────────────────────────┐
│ Exporting...                                                │
│ ████████████░░░░░░░░ 60%                                   │
│                                                             │
│ Stage: Computing logit lens (593/993 features)              │
│ Elapsed: 2:34                                               │
│                                                             │
│                                         [Cancel]            │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/neuronpedia/export` | POST | Start export job |
| `/api/v1/neuronpedia/export/{id}` | GET | Get export status |
| `/api/v1/neuronpedia/export/{id}/download` | GET | Download archive |
| `/api/v1/neuronpedia/export/{id}/cancel` | POST | Cancel export |
| `/api/v1/neuronpedia/exports` | GET | List export history |

---

## 6. Data Model

### 6.1 NeuronpediaExportJob Table
```sql
CREATE TABLE neuronpedia_export_jobs (
    id UUID PRIMARY KEY,
    sae_id UUID,  -- training_id or external_sae_id
    config JSONB NOT NULL,
    status VARCHAR(50),  -- pending, computing, packaging, completed, failed
    progress FLOAT DEFAULT 0,
    current_stage VARCHAR(100),
    feature_count INTEGER,
    output_path VARCHAR(500),
    file_size_bytes BIGINT,
    error_message TEXT,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

### 6.2 FeatureDashboardData Table
```sql
CREATE TABLE feature_dashboard_data (
    id UUID PRIMARY KEY,
    feature_id UUID REFERENCES features(id) ON DELETE CASCADE,
    logit_lens_data JSONB,
    histogram_data JSONB,
    top_tokens JSONB,
    computed_at TIMESTAMP,
    UNIQUE(feature_id)
);
```

---

## 7. WebSocket Channels

| Channel | Events | Payload |
|---------|--------|---------|
| `export/{id}` | `progress` | `{progress, stage, feature_count}` |
| `export/{id}` | `completed` | `{output_path, file_size}` |
| `export/{id}` | `failed` | `{error}` |

---

## 8. Key Files

### Backend
- `backend/src/services/neuronpedia_export_service.py` - Export orchestration
- `backend/src/services/logit_lens_service.py` - Logit lens computation
- `backend/src/workers/export_tasks.py` - Celery export task
- `backend/src/api/v1/endpoints/neuronpedia.py` - API routes
- `backend/src/schemas/neuronpedia.py` - Request/response schemas

### Frontend
- `frontend/src/components/saes/ExportToNeuronpedia.tsx` - Export modal
- `frontend/src/types/neuronpedia.ts` - TypeScript types
- `frontend/src/api/neuronpedia.ts` - API client

---

## 9. Logit Lens Computation

### 9.1 Algorithm
```python
def compute_logit_lens(sae, model, feature_idx, k=20):
    # Get feature direction (decoder weight)
    feature_direction = sae.W_dec[:, feature_idx]

    # Project through model's unembedding
    logits = feature_direction @ model.lm_head.weight.T

    # Get top promoted tokens
    top_positive = topk(logits, k)

    # Get top suppressed tokens
    top_negative = topk(-logits, k)

    return {
        "top_positive": decode_tokens(top_positive),
        "top_negative": decode_tokens(top_negative)
    }
```

---

## 10. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| SAE Management | Provides SAE to export |
| Feature Discovery | Provides extracted features |
| Model Management | Provides model for logit lens |

---

## 11. Testing Checklist

- [x] Export with all options enabled
- [x] Export extracted features only
- [x] Export custom feature range
- [x] Logit lens computation
- [x] Histogram generation
- [x] SAELens weights packaging
- [x] Download completed export
- [x] Cancel in-progress export
- [x] WebSocket progress updates
- [x] Validate export format compatibility

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/007_FTDD|Neuronpedia_Export.md) | [TID](../tids/007_FTID|Neuronpedia_Export.md)*
