# Feature PRD: Dataset Management

**Document ID:** 001_FPRD|Dataset_Management
**Version:** 1.0 (MVP Complete)
**Last Updated:** 2025-12-05
**Status:** Implemented
**Priority:** P0 (Core Feature)

---

## 1. Overview

### 1.1 Purpose
Enable users to download, prepare, and manage datasets for SAE training, including tokenization with configurable parameters and statistical analysis.

### 1.2 User Problem
Researchers need efficient data preparation for SAE training but face challenges with:
- Downloading large datasets from HuggingFace
- Tokenizing text with appropriate parameters
- Understanding dataset characteristics before training
- Managing multiple tokenizations of the same dataset

### 1.3 Solution
A comprehensive dataset management system with HuggingFace integration, flexible tokenization, and detailed statistics visualization.

---

## 2. Functional Requirements

### 2.1 Dataset Download
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | Download datasets from HuggingFace Hub by repo ID | Implemented |
| FR-1.2 | Support dataset configuration selection (splits, subsets) | Implemented |
| FR-1.3 | Display real-time download progress via WebSocket | Implemented |
| FR-1.4 | Support resumable downloads on interruption | Partial |
| FR-1.5 | Cancel in-progress downloads | Implemented |

### 2.2 Tokenization
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | Tokenize datasets using model tokenizers | Implemented |
| FR-2.2 | Configurable max_length for sequence length | Implemented |
| FR-2.3 | Configurable stride for overlapping sequences | Implemented |
| FR-2.4 | Token filtering: minimum length, special tokens | Implemented |
| FR-2.5 | Stop words filtering | Implemented |
| FR-2.6 | Multiple tokenizations per dataset | Implemented |
| FR-2.7 | Real-time tokenization progress | Implemented |

### 2.3 Statistics & Visualization
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | Display vocabulary distribution histogram | Implemented |
| FR-3.2 | Display sequence length distribution | Implemented |
| FR-3.3 | Show total tokens, unique tokens, sequences | Implemented |
| FR-3.4 | Sample browser with pagination | Implemented |
| FR-3.5 | Export statistics to JSON | Planned |

### 2.4 Management
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | List all datasets with metadata | Implemented |
| FR-4.2 | Delete datasets and associated tokenizations | Implemented |
| FR-4.3 | Search/filter datasets | Implemented |
| FR-4.4 | View tokenization history per dataset | Implemented |

---

## 3. Non-Functional Requirements

| Requirement | Target | Status |
|-------------|--------|--------|
| NFR-1 | Handle datasets up to 100GB | Tested to 10GB |
| NFR-2 | Tokenization throughput: 100K tokens/sec | Achieved |
| NFR-3 | Download progress updates every 500ms | Achieved |
| NFR-4 | Statistics calculation < 5s for typical datasets | Achieved |

---

## 4. User Interface

### 4.1 Datasets Panel
```
┌─────────────────────────────────────────────────────────────┐
│ Datasets                                    [+ Download]    │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Dataset Card                                            │ │
│ │ Name: monology/pile-uncopyrighted                      │ │
│ │ Size: 2.3GB | Samples: 1,234,567                       │ │
│ │ Status: Ready                                          │ │
│ │ Tokenizations: 2                                       │ │
│ │ [Tokenize] [Statistics] [Delete]                       │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Dataset Card (Downloading)                             │ │
│ │ Name: HuggingFaceFW/fineweb                           │ │
│ │ Progress: ████████░░░░░░░░ 52%                        │ │
│ │ Downloaded: 5.2GB / 10GB                              │ │
│ │ [Cancel]                                               │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Download Modal
- HuggingFace repo ID input with autocomplete
- Configuration options (split, subset)
- Memory/disk estimation
- Start/Cancel buttons

### 4.3 Tokenization Modal
- Tokenizer selection (from available models)
- Max length input (default: 1024)
- Stride input (default: 512)
- Token filter checkboxes
- Stop words toggle

### 4.4 Statistics Modal
- Vocabulary distribution chart (bar chart)
- Sequence length histogram
- Summary statistics table
- Sample browser with pagination

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/datasets` | GET | List all datasets |
| `/api/v1/datasets` | POST | Create dataset record |
| `/api/v1/datasets/{id}` | GET | Get dataset details |
| `/api/v1/datasets/{id}` | DELETE | Delete dataset |
| `/api/v1/datasets/{id}/download` | POST | Start download |
| `/api/v1/datasets/{id}/tokenize` | POST | Start tokenization |
| `/api/v1/datasets/{id}/statistics` | GET | Get statistics |
| `/api/v1/datasets/{id}/samples` | GET | Get sample data |
| `/api/v1/datasets/{id}/cancel` | POST | Cancel operation |

---

## 6. Data Model

### 6.1 Dataset Table
```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    repo_id VARCHAR(255),
    file_path VARCHAR(500),
    status VARCHAR(50),  -- pending, downloading, ready, failed
    metadata JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 6.2 DatasetTokenization Table
```sql
CREATE TABLE dataset_tokenizations (
    id UUID PRIMARY KEY,
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    tokenizer_name VARCHAR(255),
    max_length INTEGER,
    stride INTEGER,
    num_tokens BIGINT,
    num_sequences INTEGER,
    vocab_size INTEGER,
    token_file_path VARCHAR(500),
    statistics JSONB,
    created_at TIMESTAMP
);
```

---

## 7. WebSocket Channels

| Channel | Events | Payload |
|---------|--------|---------|
| `dataset/{id}` | `download_progress` | `{progress: float, downloaded: int, total: int}` |
| `dataset/{id}` | `download_completed` | `{file_path: string}` |
| `dataset/{id}` | `download_failed` | `{error: string}` |
| `dataset/{id}` | `tokenization_progress` | `{progress: float, tokens_processed: int}` |
| `dataset/{id}` | `tokenization_completed` | `{tokenization_id: string}` |

---

## 8. Key Files

### Backend
- `backend/src/services/dataset_service.py` - Core business logic
- `backend/src/services/tokenization_service.py` - Tokenization logic
- `backend/src/workers/dataset_tasks.py` - Celery tasks
- `backend/src/api/v1/endpoints/datasets.py` - API routes
- `backend/src/models/dataset.py` - SQLAlchemy models
- `backend/src/schemas/dataset.py` - Pydantic schemas

### Frontend
- `frontend/src/components/panels/DatasetsPanel.tsx` - Main panel
- `frontend/src/components/datasets/DatasetCard.tsx` - Card component
- `frontend/src/components/datasets/DownloadForm.tsx` - Download modal
- `frontend/src/components/datasets/TokenizationStatsModal.tsx` - Statistics
- `frontend/src/stores/datasetsStore.ts` - Zustand store
- `frontend/src/hooks/useDatasetWebSocket.ts` - WebSocket hook

---

## 9. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| Model Management | Requires tokenizers from models |
| SAE Training | Provides training data |
| Feature Discovery | Provides context for activations |

---

## 10. Testing Checklist

- [x] Download dataset from HuggingFace
- [x] Cancel in-progress download
- [x] Tokenize with custom parameters
- [x] Multiple tokenizations per dataset
- [x] View statistics modal
- [x] Delete dataset with cascade
- [x] WebSocket progress updates
- [x] Error handling for invalid repo IDs
- [x] Token filtering options

---

## 11. Known Limitations

1. **Large datasets**: Streaming mode not fully implemented for 100GB+ datasets
2. **Resume**: Download resume requires HuggingFace Hub support
3. **Export**: Statistics export to JSON not yet implemented

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/001_FTDD|Dataset_Management.md) | [TID](../tids/001_FTID|Dataset_Management.md)*
