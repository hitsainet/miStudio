# Technical Design Document: Dataset Management

**Document ID:** 001_FTDD|Dataset_Management
**Version:** 1.1
**Last Updated:** 2025-12-16
**Status:** Implemented
**Related PRD:** [001_FPRD|Dataset_Management](../prds/001_FPRD|Dataset_Management.md)

---

## 1. System Architecture

### 1.1 Component Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │DatasetsPanel│  │DownloadForm │  │TokenizationStatsModal│ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│  ┌──────┴────────────────┴─────────────────────┴──────────┐ │
│  │                  datasetsStore (Zustand)                │ │
│  └─────────────────────────┬───────────────────────────────┘ │
└────────────────────────────┼────────────────────────────────┘
                             │ HTTP + WebSocket
┌────────────────────────────┼────────────────────────────────┐
│                    Backend (FastAPI)                        │
│  ┌─────────────────────────┴───────────────────────────────┐│
│  │              /api/v1/datasets endpoints                 ││
│  └─────────────────────────┬───────────────────────────────┘│
│                            │                                │
│  ┌─────────────────────────┴───────────────────────────────┐│
│  │                   DatasetService                        ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ││
│  │  │download_from_│  │tokenize_     │  │get_statistics│  ││
│  │  │huggingface() │  │dataset()     │  │()            │  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  ││
│  └─────────────────────────┬───────────────────────────────┘│
│                            │                                │
│  ┌─────────────────────────┴───────────────────────────────┐│
│  │                  Celery Workers                         ││
│  │  ┌──────────────────┐  ┌──────────────────────────────┐││
│  │  │download_dataset_ │  │tokenize_dataset_task         │││
│  │  │task              │  │                              │││
│  │  └──────────────────┘  └──────────────────────────────┘││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

#### Download Flow
```
User → DownloadForm → POST /datasets/{id}/download
                              ↓
                    DatasetService.start_download()
                              ↓
                    Celery: download_dataset_task.delay()
                              ↓
                    HuggingFace Hub API (streaming)
                              ↓
                    WebSocket emit: dataset/{id}/progress
                              ↓
                    Save to DATA_DIR/datasets/{repo_id}/
                              ↓
                    Update dataset.status = 'ready'
```

#### Tokenization Flow
```
User → TokenizeButton → POST /datasets/{id}/tokenize
                              ↓
                    DatasetService.start_tokenization()
                              ↓
                    Celery: tokenize_dataset_task.delay()
                              ↓
                    Load model tokenizer
                              ↓
                    Process samples in batches
                              ↓
                    Apply filters (length, special, stop words)
                              ↓
                    Save tokens to .arrow file
                              ↓
                    Compute statistics
                              ↓
                    WebSocket emit: completed
```

---

## 2. Database Schema

### 2.1 Entity Relationship
```
┌─────────────────┐         ┌─────────────────────┐
│     Dataset     │ 1     N │ DatasetTokenization │
├─────────────────┤─────────├─────────────────────┤
│ id (PK)         │         │ id (PK)             │
│ name            │         │ dataset_id (FK)     │
│ repo_id         │         │ tokenizer_name      │
│ file_path       │         │ max_length          │
│ status          │         │ stride              │
│ metadata (JSON) │         │ num_tokens          │
│ created_at      │         │ num_sequences       │
│ updated_at      │         │ vocab_size          │
└─────────────────┘         │ token_file_path     │
                            │ statistics (JSON)   │
                            │ created_at          │
                            └─────────────────────┘
```

### 2.2 Status State Machine
```
                    ┌─────────┐
                    │ pending │
                    └────┬────┘
                         │ start_download()
                         ▼
                  ┌─────────────┐
            ┌─────│ downloading │─────┐
            │     └─────────────┘     │
     error  │                         │ success
            ▼                         ▼
       ┌────────┐               ┌───────┐
       │ failed │               │ ready │
       └────────┘               └───────┘
```

---

## 3. API Design

### 3.1 Endpoints

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| GET | `/datasets` | `?page&limit&status` | `{items: Dataset[], total}` |
| POST | `/datasets` | `{repo_id, name}` | `Dataset` |
| GET | `/datasets/{id}` | - | `Dataset` |
| DELETE | `/datasets/{id}` | - | `204 No Content` |
| POST | `/datasets/{id}/download` | `{split?, subset?}` | `{job_id}` |
| POST | `/datasets/{id}/tokenize` | `TokenizeConfig` | `{tokenization_id}` |
| GET | `/datasets/{id}/statistics` | `?tokenization_id` | `Statistics` |
| GET | `/datasets/{id}/samples` | `?page&limit&tokenization_id` | `{samples[]}` |
| POST | `/datasets/{id}/cancel` | - | `204` |

### 3.2 Request/Response Schemas

```python
class TokenizeConfig(BaseModel):
    tokenizer_name: str
    max_length: int = 1024
    stride: int = 512
    min_token_length: int = 3
    filter_special_tokens: bool = True
    filter_stop_words: bool = True

class DatasetStatistics(BaseModel):
    total_tokens: int
    unique_tokens: int
    total_sequences: int
    vocab_distribution: List[VocabItem]
    length_distribution: List[int]
```

---

## 4. Service Layer Design

### 4.1 DatasetService

```python
class DatasetService:
    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis

    async def create_dataset(self, repo_id: str, name: str) -> Dataset:
        """Create dataset record for tracking."""

    async def start_download(self, dataset_id: UUID, config: DownloadConfig) -> str:
        """Queue download task, return job_id."""

    async def start_tokenization(self, dataset_id: UUID, config: TokenizeConfig) -> UUID:
        """Queue tokenization task, return tokenization_id."""

    async def get_statistics(self, tokenization_id: UUID) -> DatasetStatistics:
        """Compute or retrieve cached statistics."""

    async def get_samples(self, tokenization_id: UUID, page: int, limit: int) -> List[Sample]:
        """Retrieve sample data with pagination."""
```

### 4.3 Data Sanitization (Added Dec 2025)

HuggingFace datasets may contain binary data (bytes) that cannot be JSON serialized.
The samples endpoint implements a sanitization layer to handle this:

```python
def sanitize_value(value):
    """
    Recursively convert bytes and other non-JSON-serializable types to strings.
    Uses UTF-8 decoding with Latin-1 fallback (Latin-1 accepts any byte sequence).
    """
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return value.decode('latin-1')  # Always succeeds
    elif isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_value(item) for item in value]
    elif isinstance(value, tuple):
        return tuple(sanitize_value(item) for item in value)
    else:
        return value
```

**Use Case:** The Pile dataset includes `bytes` objects in the `repetitions` field which caused 500 errors before this fix.

### 4.2 TokenizationService

```python
class TokenizationService:
    def tokenize_batch(self, texts: List[str], tokenizer, config: TokenizeConfig) -> TokenBatch:
        """Tokenize batch with filtering."""

    def apply_filters(self, tokens: List[int], tokenizer, config: TokenizeConfig) -> List[int]:
        """Apply token filters: length, special, stop words."""

    def compute_statistics(self, token_file: Path) -> Statistics:
        """Compute vocab distribution and length stats."""
```

---

## 5. Celery Task Design

### 5.1 Download Task

```python
@celery_app.task(bind=True, queue='processing')
def download_dataset_task(self, dataset_id: str, config: dict):
    """
    Download dataset from HuggingFace.

    Progress emission every 1%:
    - emit_dataset_progress(dataset_id, progress, downloaded_bytes, total_bytes)

    On completion:
    - Update dataset.status = 'ready'
    - emit_dataset_completed(dataset_id)
    """
```

### 5.2 Tokenization Task

```python
@celery_app.task(bind=True, queue='processing')
def tokenize_dataset_task(self, dataset_id: str, tokenization_id: str, config: dict):
    """
    Tokenize dataset samples.

    Steps:
    1. Load tokenizer from model
    2. Stream samples from dataset
    3. Tokenize in batches (1000 samples)
    4. Apply filters per batch
    5. Write to Arrow file
    6. Compute final statistics

    Progress emission every 1000 samples.
    """
```

---

## 6. WebSocket Events

### 6.1 Channel: `dataset/{dataset_id}`

| Event | Payload | Trigger |
|-------|---------|---------|
| `download_progress` | `{progress: float, downloaded: int, total: int}` | Every 1% |
| `download_completed` | `{file_path: string}` | Download done |
| `download_failed` | `{error: string}` | Download error |
| `tokenization_progress` | `{progress: float, tokens: int, sequences: int}` | Every 1000 samples |
| `tokenization_completed` | `{tokenization_id: string, statistics: object}` | Tokenization done |
| `tokenization_failed` | `{error: string}` | Tokenization error |

---

## 7. File Storage Structure

```
DATA_DIR/
└── datasets/
    └── {repo_id_normalized}/
        ├── raw/
        │   └── dataset.arrow           # Original HuggingFace data
        └── tokenized/
            └── {tokenization_id}/
                ├── tokens.arrow        # Tokenized sequences
                └── metadata.json       # Tokenization config + stats
```

---

## 8. Frontend State Management

### 8.1 datasetsStore (Zustand)

```typescript
interface DatasetsState {
  datasets: Dataset[];
  selectedDataset: Dataset | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchDatasets: () => Promise<void>;
  createDataset: (repoId: string, name: string) => Promise<Dataset>;
  deleteDataset: (id: string) => Promise<void>;
  startDownload: (id: string, config: DownloadConfig) => Promise<void>;
  startTokenization: (id: string, config: TokenizeConfig) => Promise<void>;

  // WebSocket handlers
  updateDownloadProgress: (id: string, progress: ProgressData) => void;
  setDownloadComplete: (id: string) => void;
}
```

---

## 9. Error Handling

### 9.1 Error Categories

| Category | HTTP Code | Handling |
|----------|-----------|----------|
| Invalid repo_id | 400 | Validate before download |
| Dataset not found | 404 | Return null/404 |
| Download failed | 500 | Retry up to 3 times |
| Tokenization OOM | 500 | Reduce batch size, retry |
| Disk full | 507 | Alert user, cleanup |

### 9.2 Retry Strategy

```python
@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError)
)
def download_dataset_task(self, ...):
    ...
```

---

## 10. Performance Considerations

### 10.1 Optimizations
- **Streaming downloads**: Use HuggingFace streaming to avoid memory issues
- **Batch tokenization**: Process 1000 samples per batch
- **Arrow format**: Memory-mapped for zero-copy access
- **Statistics caching**: Cache computed stats in Redis (1 hour TTL)

### 10.2 Limits
- Max dataset size: 100GB (configurable)
- Max tokenization batch: 10,000 samples
- WebSocket throttle: 500ms between progress updates

---

*Related: [PRD](../prds/001_FPRD|Dataset_Management.md) | [TID](../tids/001_FTID|Dataset_Management.md) | [FTASKS](../tasks/001_FTASKS|Dataset_Management.md)*
