# Technical Design Document: Dataset Management

**Document ID:** 001_FTDD|Dataset_Management
**Feature:** Dataset Management System
**Status:** Draft
**Created:** 2025-10-06
**Last Updated:** 2025-10-06
**Owner:** miStudio Development Team

**Related Documents:**
- PRD: `001_FPRD|Dataset_Management.md`
- ADR: `000_PADR|miStudio.md`
- UI Reference: `Mock-embedded-interp-ui.tsx` (lines 1108-1202, DatasetsPanel)

---

## 1. Executive Summary

This Technical Design Document outlines the architecture and implementation approach for the Dataset Management feature, which enables users to download, tokenize, browse, and manage datasets from HuggingFace for SAE training. The system provides a React-based frontend interface matching the Mock UI exactly (DatasetsPanel component), backed by a FastAPI async backend with Celery workers for background job processing, PostgreSQL for metadata storage, and local filesystem for dataset files.

**Key Technical Decisions:**
- **Frontend:** React 18+ with TypeScript, Zustand for state management, exact replication of Mock UI styling
- **Backend:** FastAPI with async/await, Celery + Redis for background downloads
- **Storage:** PostgreSQL 14+ (metadata), Local filesystem `/data/datasets/` (files)
- **Real-time Updates:** WebSocket connections for download/tokenization progress

**Architecture Alignment:** This design follows the Component-based architecture defined in the ADR, with clear separation between presentation (React components), business logic (API services), and data layer (database + filesystem).

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  DatasetsPanel (React Component)                          │  │
│  │  - Matches Mock UI lines 1108-1202 exactly               │  │
│  │  - Uses Zustand store for state management               │  │
│  │  - WebSocket connection for real-time updates            │  │
│  └──────────────────────────────────────────────────────────┘  │
│           ▲                             ▲                        │
│           │ HTTP/REST                   │ WebSocket             │
│           │                             │                        │
└───────────┼─────────────────────────────┼────────────────────────┘
            │                             │
┌───────────▼─────────────────────────────▼────────────────────────┐
│                       Backend Layer (FastAPI)                     │
│  ┌──────────────────┐   ┌──────────────────┐                    │
│  │   API Routes     │   │  WebSocket       │                    │
│  │  /api/datasets   │   │  Manager         │                    │
│  └────────┬─────────┘   └──────┬───────────┘                    │
│           │                     │                                │
│  ┌────────▼─────────────────────▼───────────┐                   │
│  │       Service Layer (Business Logic)     │                   │
│  │  - DatasetService                        │                   │
│  │  - TokenizationService                   │                   │
│  │  - StatisticsService                     │                   │
│  └────────┬────────────────────┬────────────┘                   │
│           │                    │                                 │
└───────────┼────────────────────┼─────────────────────────────────┘
            │                    │
┌───────────▼────────────────────▼─────────────────────────────────┐
│                    Background Workers (Celery)                    │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │  Download Worker │  │  Tokenize Worker │                     │
│  │  - HuggingFace   │  │  - Tokenizer     │                     │
│  │  - Progress      │  │  - Statistics    │                     │
│  └────────┬─────────┘  └────────┬─────────┘                     │
│           │                     │                                │
└───────────┼─────────────────────┼─────────────────────────────────┘
            │                     │
┌───────────▼─────────────────────▼─────────────────────────────────┐
│                      Data Layer                                    │
│  ┌───────────────┐  ┌────────────────────────────────────┐      │
│  │  PostgreSQL   │  │  Filesystem (/data/datasets/)      │      │
│  │  - datasets   │  │  - ds_{id}/                        │      │
│  │    table      │  │    - raw/                          │      │
│  │               │  │    - tokenized/                    │      │
│  └───────────────┘  └────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────┘
```

### Component Relationships

**Frontend Components (React):**
```
DatasetsPanel (Main Container)
├── DatasetCard[] (List of datasets)
│   ├── StatusBadge (download/ready/error)
│   ├── ProgressBar (if downloading/tokenizing)
│   └── Actions (Delete button)
├── DatasetDetailModal (on card click)
│   ├── Overview Tab
│   ├── Samples Tab (DatasetSamplesBrowser)
│   ├── Tokenization Tab (TokenizationSettings)
│   └── Statistics Tab (DatasetStatistics)
└── DownloadModal (HuggingFace download form)
```

**Backend Services:**
```
API Layer (FastAPI)
├── DatasetRouter → DatasetService
│   ├── list_datasets()
│   ├── get_dataset(id)
│   ├── download_dataset() → enqueue Celery task
│   ├── delete_dataset()
│   └── get_dataset_samples(id)
│
Celery Workers
├── download_dataset_task
│   ├── Load HuggingFace datasets library
│   ├── Download with progress tracking
│   ├── Emit WebSocket progress events
│   └── Auto-trigger tokenization on completion
│
└── tokenize_dataset_task
    ├── Load tokenizer
    ├── Process samples in batches
    ├── Calculate statistics
    └── Update database status
```

### Data Flow: Dataset Download

```
1. User clicks "Download Dataset" (Frontend)
   ↓
2. DatasetsPanel → POST /api/datasets/download
   {dataset_id: "pile", source: "huggingface"}
   ↓
3. FastAPI creates database record (status='downloading', progress=0)
   Returns {id: "ds_abc123", status: "downloading"}
   ↓
4. FastAPI enqueues Celery task: download_dataset_task.delay(ds_abc123)
   ↓
5. Celery worker picks up task
   - Calls datasets.load_dataset("pile")
   - Tracks progress via callback
   - Emits WebSocket events: ws://mistudio/datasets/ds_abc123/progress
   ↓
6. Frontend receives WebSocket events
   - Updates Zustand store: datasets[ds_abc123].progress = 45.2
   - ProgressBar component re-renders automatically
   ↓
7. Download completes
   - Worker saves files to /data/datasets/ds_abc123/raw/
   - Updates database: status='processing', file_path='/data/datasets/...'
   - Auto-triggers tokenization task
   ↓
8. Tokenization completes
   - Updates database: status='ready', num_samples, num_tokens, etc.
   - Emits WebSocket event: {type: 'completed'}
   - Frontend shows green "Ready" badge
```

---

## 3. Technical Stack

### Frontend Technologies

| Technology | Version | Purpose | Justification |
|-----------|---------|---------|---------------|
| **React** | 18.3+ | UI framework | Component-based, excellent TypeScript support, matches Mock UI patterns |
| **TypeScript** | 5.5+ | Type safety | Prevents runtime errors, better IDE support, enforces interface contracts |
| **Zustand** | 4.5+ | State management | Lightweight, simple API, perfect for real-time WebSocket updates |
| **Vite** | 5.4+ | Build tool | Fast HMR, optimized production builds, native ESM support |
| **Tailwind CSS** | 3.4+ | Styling | Utility-first CSS matching Mock UI exactly, responsive design |
| **Lucide React** | 0.436+ | Icons | Consistent icon library used in Mock UI |
| **socket.io-client** | 4.7+ | WebSocket | Real-time progress updates from backend |

### Backend Technologies

| Technology | Version | Purpose | Justification |
|-----------|---------|---------|---------------|
| **FastAPI** | 0.115+ | Web framework | Async/await support, automatic OpenAPI docs, Python type hints |
| **Celery** | 5.4+ | Task queue | Distributed background jobs for downloads/tokenization |
| **Redis** | 7.4+ | Message broker | Celery task queue, WebSocket pub/sub |
| **PostgreSQL** | 14+ | Database | JSONB support, full-text search (GIN indexes), ACID guarantees |
| **SQLAlchemy** | 2.0+ | ORM | Async support, type-safe queries, migration management |
| **Alembic** | 1.13+ | Migrations | Database version control, team collaboration |
| **HuggingFace Datasets** | 3.0+ | Dataset loading | Official library for HuggingFace Hub integration |
| **Transformers** | 4.45+ | Tokenizers | Pre-trained tokenizers (GPT-2, Llama, etc.) |
| **Pydantic** | 2.9+ | Validation | Request/response validation, settings management |

### Dependencies Justification

**Why Celery + Redis:**
- Downloads can take minutes to hours (large datasets)
- FastAPI request timeout would kill long-running downloads
- Need progress tracking and cancellation support
- Redis provides fast pub/sub for WebSocket events

**Why PostgreSQL + SQLAlchemy:**
- JSONB fields for flexible metadata (statistics, split info)
- Full-text search with GIN indexes for dataset name/description search
- ACID transactions for dataset deletion (metadata + filesystem cleanup)
- SQLAlchemy async support matches FastAPI async patterns

**Why HuggingFace Datasets Library:**
- Official library with streaming support
- Built-in caching and download management
- Automatic format conversion (Parquet, Arrow, JSON)
- Progress callbacks for tracking

---

## 4. Data Design

### Database Schema

```sql
CREATE TABLE datasets (
    -- Primary key
    id VARCHAR(255) PRIMARY KEY,  -- Format: ds_{uuid}

    -- Basic metadata
    name VARCHAR(500) NOT NULL,
    source VARCHAR(50) NOT NULL,  -- 'huggingface', 'local', 'custom'

    -- File storage
    file_path VARCHAR(1000),           -- /data/datasets/{id}/raw/
    tokenized_path VARCHAR(1000),      -- /data/datasets/{id}/tokenized/
    size_bytes BIGINT,                 -- Total disk size

    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'downloading',
        -- States: downloading, processing, tokenizing, ready, error
    progress FLOAT DEFAULT 0,          -- 0-100 for downloads/tokenization
    error_message TEXT,                -- Populated on error

    -- Dataset statistics (populated after tokenization)
    num_samples INTEGER,               -- Total number of samples
    num_tokens BIGINT,                 -- Total tokens across all samples
    vocab_size INTEGER,                -- Unique tokens
    avg_sequence_length FLOAT,         -- Average tokens per sample
    split_info JSONB,                  -- {"train": 9000, "val": 500, "test": 500}

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT datasets_status_check CHECK (status IN
        ('downloading', 'processing', 'tokenizing', 'ready', 'error'))
);

-- Indexes for performance
CREATE INDEX idx_datasets_status ON datasets(status);
CREATE INDEX idx_datasets_source ON datasets(source);
CREATE INDEX idx_datasets_created_at ON datasets(created_at DESC);

-- Full-text search index
CREATE INDEX idx_datasets_search ON datasets
    USING GIN(to_tsvector('english', name));
```

### Data Validation Strategy

**Input Validation (Pydantic Models):**
```python
from pydantic import BaseModel, Field, validator

class DatasetDownloadRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=500)
    source: Literal['huggingface'] = 'huggingface'
    split: Optional[str] = None  # 'train', 'test', 'validation', or None (all)

    @validator('dataset_id')
    def validate_dataset_id(cls, v):
        # Must be valid HuggingFace dataset ID
        if not re.match(r'^[a-zA-Z0-9-_/]+$', v):
            raise ValueError('Invalid dataset ID format')
        return v

class DatasetResponse(BaseModel):
    id: str
    name: str
    source: str
    status: str
    progress: float
    size_bytes: Optional[int] = None
    num_samples: Optional[int] = None
    created_at: datetime
```

**Database Consistency:**
- Use SQLAlchemy `session.commit()` for atomic updates
- Cascade deletes: When dataset deleted, remove filesystem files in transaction
- Status transitions validated: Can't go from 'ready' to 'downloading'
- Foreign key constraints: Models/trainings reference datasets (ON DELETE RESTRICT)

### Filesystem Organization

```
/data/datasets/
├── ds_abc123/                    # Dataset ID
│   ├── raw/                      # Original downloaded files
│   │   ├── dataset_info.json     # HuggingFace metadata
│   │   ├── data-00000.parquet    # Data files
│   │   ├── data-00001.parquet
│   │   └── ...
│   ├── tokenized/                # Tokenized versions
│   │   ├── train.pt              # PyTorch tensors
│   │   ├── val.pt
│   │   └── test.pt
│   └── metadata.json             # miStudio metadata
│
├── ds_def456/
│   └── ...
```

**File Naming Convention:**
- Dataset ID: `ds_{uuid}` (UUID v4 for uniqueness)
- Raw files: Preserve HuggingFace naming (Parquet, Arrow, JSON)
- Tokenized files: `{split}.pt` (PyTorch tensors saved with `torch.save()`)

---

## 5. API Design

### RESTful Endpoints

**Endpoint:** `GET /api/datasets`
- **Purpose:** List all datasets with optional filtering
- **Query Params:** `?status=ready&source=huggingface&limit=50&offset=0`
- **Response:** Paginated list of datasets
- **Caching:** Cache for 30 seconds (datasets don't change frequently)

**Endpoint:** `GET /api/datasets/:id`
- **Purpose:** Get single dataset details
- **Response:** Full dataset object with statistics
- **Error Handling:** 404 if not found, 500 on database error

**Endpoint:** `POST /api/datasets/download`
- **Purpose:** Initiate dataset download from HuggingFace
- **Request Body:** `{dataset_id, source, split?}`
- **Response:** 201 Created with dataset object (status='downloading')
- **Side Effects:** Enqueues Celery download task
- **Error Handling:** 400 if invalid dataset_id, 409 if already exists

**Endpoint:** `GET /api/datasets/:id/samples`
- **Purpose:** Retrieve dataset samples for browsing
- **Query Params:** `?limit=10&offset=0&split=train&search=query`
- **Response:** Paginated samples with full-text search
- **Performance:** Use database LIMIT/OFFSET, cache results

**Endpoint:** `DELETE /api/datasets/:id`
- **Purpose:** Delete dataset (metadata + files)
- **Response:** 204 No Content
- **Side Effects:** Removes /data/datasets/{id}/ directory recursively
- **Error Handling:** 409 if dataset in use by training, 404 if not found
- **Transaction:** Database delete + filesystem delete in try/finally

### API Error Handling Pattern

```python
from fastapi import HTTPException

class DatasetNotFoundError(HTTPException):
    def __init__(self, dataset_id: str):
        super().__init__(
            status_code=404,
            detail=f"Dataset {dataset_id} not found"
        )

class DatasetInUseError(HTTPException):
    def __init__(self, dataset_id: str):
        super().__init__(
            status_code=409,
            detail=f"Dataset {dataset_id} is in use by active trainings"
        )

# Usage in route
@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    dataset = await DatasetService.get(db, dataset_id)
    if not dataset:
        raise DatasetNotFoundError(dataset_id)

    # Check if in use
    trainings = await Training.get_by_dataset(db, dataset_id)
    if trainings:
        raise DatasetInUseError(dataset_id)

    await DatasetService.delete(db, dataset_id)
    return Response(status_code=204)
```

### WebSocket Protocol

**Channel:** `ws://mistudio.mcslab.io/ws/datasets/{dataset_id}`
- **Events:** `download_progress`, `tokenization_progress`, `completed`, `error`
- **Authentication:** JWT token in query param: `?token=xxx`
- **Reconnection:** Client implements exponential backoff (1s, 2s, 4s, max 30s)

**Example Event Payloads:**
```javascript
// Download progress
{
  type: 'download_progress',
  dataset_id: 'ds_abc123',
  progress: 45.2,
  bytes_downloaded: 1024000000,
  total_bytes: 2048000000,
  status: 'downloading'
}

// Tokenization progress
{
  type: 'tokenization_progress',
  dataset_id: 'ds_abc123',
  progress: 67.8,
  samples_processed: 6780,
  total_samples: 10000,
  status: 'tokenizing'
}

// Completed
{
  type: 'completed',
  dataset_id: 'ds_abc123',
  status: 'ready',
  num_samples: 10000,
  num_tokens: 4870000,
  vocab_size: 50257
}

// Error
{
  type: 'error',
  dataset_id: 'ds_abc123',
  error_message: 'Failed to download: Network timeout',
  status: 'error'
}
```

---

## 6. Component Architecture

### Frontend Component Hierarchy

```typescript
// DatasetsPanel.tsx - Main container component
export const DatasetsPanel: React.FC = () => {
  // Zustand store hooks
  const datasets = useDatasetStore((state) => state.datasets);
  const fetchDatasets = useDatasetStore((state) => state.fetchDatasets);

  // WebSocket connection
  const { subscribe, unsubscribe } = useWebSocket();

  // Local state
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);

  useEffect(() => {
    fetchDatasets();

    // Subscribe to all dataset updates
    datasets.forEach(ds => {
      if (ds.status !== 'ready' && ds.status !== 'error') {
        subscribe(`datasets/${ds.id}`, handleDatasetUpdate);
      }
    });

    return () => {
      datasets.forEach(ds => unsubscribe(`datasets/${ds.id}`));
    };
  }, [datasets]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Datasets</h2>
        <button
          onClick={() => setShowDownloadModal(true)}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded-lg"
        >
          <Download className="w-4 h-4 inline mr-2" />
          Download Dataset
        </button>
      </div>

      {datasets.length === 0 ? (
        <EmptyState message="No datasets yet. Download from HuggingFace to get started." />
      ) : (
        <div className="grid grid-cols-2 gap-6">
          {datasets.map(dataset => (
            <DatasetCard
              key={dataset.id}
              dataset={dataset}
              onClick={() => setSelectedDataset(dataset)}
            />
          ))}
        </div>
      )}

      {showDownloadModal && (
        <DownloadDatasetModal onClose={() => setShowDownloadModal(false)} />
      )}

      {selectedDataset && (
        <DatasetDetailModal
          dataset={selectedDataset}
          onClose={() => setSelectedDataset(null)}
        />
      )}
    </div>
  );
};
```

### Zustand Store Design

```typescript
// stores/datasetStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface Dataset {
  id: string;
  name: string;
  source: string;
  status: 'downloading' | 'processing' | 'tokenizing' | 'ready' | 'error';
  progress: number;
  size_bytes?: number;
  num_samples?: number;
  num_tokens?: number;
  created_at: string;
}

interface DatasetStore {
  datasets: Dataset[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchDatasets: () => Promise<void>;
  addDataset: (dataset: Dataset) => void;
  updateDataset: (id: string, updates: Partial<Dataset>) => void;
  deleteDataset: (id: string) => Promise<void>;
}

export const useDatasetStore = create<DatasetStore>()(
  devtools((set, get) => ({
    datasets: [],
    loading: false,
    error: null,

    fetchDatasets: async () => {
      set({ loading: true, error: null });
      try {
        const response = await fetch('/api/datasets');
        const data = await response.json();
        set({ datasets: data.datasets, loading: false });
      } catch (error) {
        set({ error: error.message, loading: false });
      }
    },

    addDataset: (dataset) => {
      set((state) => ({
        datasets: [...state.datasets, dataset]
      }));
    },

    updateDataset: (id, updates) => {
      set((state) => ({
        datasets: state.datasets.map(ds =>
          ds.id === id ? { ...ds, ...updates } : ds
        )
      }));
    },

    deleteDataset: async (id) => {
      try {
        await fetch(`/api/datasets/${id}`, { method: 'DELETE' });
        set((state) => ({
          datasets: state.datasets.filter(ds => ds.id !== id)
        }));
      } catch (error) {
        set({ error: error.message });
      }
    }
  }))
);
```

### WebSocket Hook

```typescript
// hooks/useWebSocket.ts
import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';

export const useWebSocket = () => {
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    socketRef.current = io('ws://mistudio.mcslab.io', {
      transports: ['websocket'],
      query: {
        token: localStorage.getItem('auth_token')
      }
    });

    return () => {
      socketRef.current?.disconnect();
    };
  }, []);

  const subscribe = (channel: string, handler: (data: any) => void) => {
    socketRef.current?.on(channel, handler);
  };

  const unsubscribe = (channel: string) => {
    socketRef.current?.off(channel);
  };

  return { subscribe, unsubscribe };
};
```

---

## 7. State Management

### Application State Organization

```typescript
// Global State (Zustand stores)
stores/
├── datasetStore.ts      # Dataset list, CRUD operations
├── authStore.ts         # User authentication, JWT token
├── notificationStore.ts # Toast notifications, alerts
└── websocketStore.ts    # WebSocket connection status

// Component Local State (useState)
- Modal open/close state
- Form input values (dataset_id, split selection)
- UI interactions (hover, focus states)
```

### State Flow Pattern

```
User Action (e.g., "Download Dataset")
    ↓
Component Event Handler
    ↓
API Call (async function)
    ↓
Optimistic Update (Zustand store)
    ↓
Backend Processing (Celery)
    ↓
WebSocket Event Received
    ↓
Store Update (actual progress)
    ↓
Component Re-render (automatic via Zustand)
```

### Real-time Update Strategy

**Problem:** How to update UI in real-time as downloads/tokenization progress?

**Solution:** WebSocket subscription with Zustand integration

```typescript
// In DatasetsPanel component
useEffect(() => {
  const { subscribe } = useWebSocket();
  const updateDataset = useDatasetStore((state) => state.updateDataset);

  // Subscribe to all active datasets
  datasets
    .filter(ds => ds.status !== 'ready' && ds.status !== 'error')
    .forEach(ds => {
      subscribe(`datasets/${ds.id}`, (event) => {
        updateDataset(ds.id, {
          progress: event.progress,
          status: event.status,
          num_samples: event.num_samples,
          num_tokens: event.num_tokens
        });
      });
    });
}, [datasets]);
```

**Benefits:**
- Automatic UI updates (no polling)
- Multiple users see same updates
- Works across browser tabs

---

## 8. Security Considerations

### Authentication Strategy

**Current Scope (MVP):** Single-user system running locally on Jetson Orin Nano
- No authentication required (localhost only)
- nginx reverse proxy provides external access security

**Future (Multi-user):**
- JWT tokens for API authentication
- WebSocket authentication via token query param
- Role-based access control (RBAC)

### Input Validation

**Frontend Validation:**
```typescript
// In DownloadDatasetModal
const validateDatasetId = (id: string): string | null => {
  if (!id || id.trim().length === 0) {
    return "Dataset ID is required";
  }
  if (!/^[a-zA-Z0-9-_/]+$/.test(id)) {
    return "Invalid dataset ID format";
  }
  if (id.length > 500) {
    return "Dataset ID too long";
  }
  return null;
};
```

**Backend Validation (Pydantic):**
```python
class DatasetDownloadRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=500, regex=r'^[a-zA-Z0-9-_/]+$')
    source: Literal['huggingface'] = 'huggingface'
    split: Optional[str] = Field(None, regex=r'^(train|test|validation|all)$')
```

### Filesystem Security

**Path Traversal Prevention:**
```python
import os
from pathlib import Path

def get_dataset_path(dataset_id: str) -> Path:
    # Validate dataset_id format
    if not re.match(r'^ds_[a-f0-9-]+$', dataset_id):
        raise ValueError("Invalid dataset ID")

    base_path = Path("/data/datasets")
    dataset_path = (base_path / dataset_id).resolve()

    # Ensure path is within base_path (prevent traversal)
    if not str(dataset_path).startswith(str(base_path)):
        raise ValueError("Path traversal detected")

    return dataset_path
```

### Data Sanitization

**Database Injection Prevention:**
- Use SQLAlchemy parameterized queries (never string concatenation)
- Validate all inputs with Pydantic before database operations

**XSS Prevention:**
- React automatically escapes HTML in JSX
- Use `dangerouslySetInnerHTML` only when absolutely necessary (never for user input)

#### 8.5 Metadata Validation Strategy

**Pydantic V2 Schemas (Task 13.3):**

The system uses Pydantic V2 for comprehensive metadata validation:

```python
# backend/src/schemas/metadata.py
from pydantic import BaseModel, Field, field_validator

class TokenizationMetadata(BaseModel):
    """Tokenization statistics and configuration."""
    tokenizer_name: str
    text_column_used: str
    max_length: int = Field(ge=1, le=8192)
    stride: int = Field(ge=0)
    num_tokens: int = Field(ge=0)
    avg_seq_length: float = Field(ge=0)
    min_seq_length: int = Field(ge=0)
    max_seq_length: int = Field(ge=0)

    @field_validator('max_seq_length')
    @classmethod
    def validate_max_seq_length(cls, v: int, info: ValidationInfo) -> int:
        min_seq = info.data.get('min_seq_length')
        if min_seq is not None and v < min_seq:
            raise ValueError('max_seq_length must be >= min_seq_length')
        return v

class DatasetMetadata(BaseModel):
    """Complete dataset metadata container."""
    schema: Optional[SchemaMetadata] = None
    tokenization: Optional[TokenizationMetadata] = None
    download: Optional[DownloadMetadata] = None
```

**Benefits:**
- Type-safe metadata access
- Automatic validation on API requests
- Clear error messages for malformed data
- Cross-field validation (e.g., max >= min)

#### 8.6 Deep Merge Pattern

**Metadata Preservation Across Operations:**

To preserve download metadata when adding tokenization metadata, the system uses a deep merge algorithm:

```python
# backend/src/services/dataset_service.py
def deep_merge_metadata(existing: dict, new: dict) -> dict:
    """Recursively merge metadata, preserving existing data."""
    merged = existing.copy() if existing else {}

    for key, value in new.items():
        if value is None:
            continue  # Skip None values to preserve existing

        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_metadata(merged[key], value)
        else:
            merged[key] = value

    return merged
```

**Example:**
```python
# Download creates: {"download": {"split": "train", "config": "en"}}
# Tokenization adds: {"tokenization": {"tokenizer_name": "gpt2", ...}}
# Result: {"download": {...}, "tokenization": {...}}
```

---

## 9. Performance & Scalability

### Performance Optimization

**Frontend:**
1. **Code Splitting:** Lazy load DetailModal components
```typescript
const DatasetDetailModal = lazy(() => import('./DatasetDetailModal'));
```

2. **Memoization:** Prevent unnecessary re-renders
```typescript
const DatasetCard = React.memo(({ dataset, onClick }) => {
  // Component logic
});
```

3. **Virtualization:** Use react-window for large dataset lists (future enhancement)

**Backend:**
1. **Database Indexing:**
   - B-tree index on `created_at` for sorting
   - GIN index on `name` for full-text search
   - Index on `status` for filtering

2. **Query Optimization:**
```python
# Bad: N+1 query problem
for dataset in datasets:
    samples = await get_samples(dataset.id)  # Query per dataset

# Good: Single query with JOIN
datasets_with_samples = await db.execute(
    select(Dataset, Sample)
    .join(Sample, Dataset.id == Sample.dataset_id)
    .limit(50)
)
```

3. **Caching Strategy:**
```python
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache

@router.get("/datasets")
@cache(expire=30)  # Cache for 30 seconds
async def list_datasets():
    return await DatasetService.list_all()
```

#### 10.4 NumPy Vectorization (Task 13.9)

**Statistics Calculation Optimization:**

The system uses NumPy vectorized operations for 10x performance improvement:

```python
# backend/src/services/tokenization_service.py
import numpy as np

@staticmethod
def calculate_statistics(tokenized_dataset: HFDataset) -> Dict[str, Any]:
    """Calculate token statistics using vectorized operations."""
    # Convert to numpy array for vectorization
    input_ids = tokenized_dataset["input_ids"]
    seq_lengths = np.array([len(ids) for ids in input_ids])

    return {
        "num_tokens": int(seq_lengths.sum()),
        "avg_seq_length": float(seq_lengths.mean()),
        "min_seq_length": int(seq_lengths.min()),
        "max_seq_length": int(seq_lengths.max()),
    }
```

**Performance:**
- Before: ~10s for 1M samples (Python loop)
- After: ~1s for 1M samples (NumPy vectorization)
- 10x speedup for large datasets

### Scalability Considerations

**Current Scope (Edge Device):**
- Single Jetson Orin Nano (8GB RAM, 6GB GPU)
- Expect 10-50 datasets (< 500GB total)
- 1-2 simultaneous downloads

**Scaling Approach:**
1. **Horizontal Celery Scaling:** Add worker nodes for parallel downloads
2. **Database Replication:** PostgreSQL read replicas for query scaling
3. **Distributed Storage:** NFS/S3 for dataset file storage (future)

**Performance Targets:**
- Dataset list load: < 500ms
- Sample retrieval (10 samples): < 200ms
- WebSocket latency: < 100ms
- Database query time: < 50ms

---

## 10. Testing Strategy

### Unit Testing

**Frontend (Vitest + React Testing Library):**
```typescript
// DatasetsPanel.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { DatasetsPanel } from './DatasetsPanel';

describe('DatasetsPanel', () => {
  it('displays empty state when no datasets', () => {
    render(<DatasetsPanel />);
    expect(screen.getByText(/No datasets yet/i)).toBeInTheDocument();
  });

  it('displays dataset cards when datasets exist', async () => {
    // Mock Zustand store
    useDatasetStore.setState({
      datasets: [
        { id: 'ds_1', name: 'The Pile', status: 'ready', progress: 100 }
      ]
    });

    render(<DatasetsPanel />);
    await waitFor(() => {
      expect(screen.getByText('The Pile')).toBeInTheDocument();
    });
  });

  it('opens download modal on button click', () => {
    render(<DatasetsPanel />);
    fireEvent.click(screen.getByText(/Download Dataset/i));
    expect(screen.getByText(/HuggingFace Dataset ID/i)).toBeInTheDocument();
  });
});
```

**Backend (pytest + pytest-asyncio):**
```python
# tests/test_dataset_service.py
import pytest
from app.services.dataset_service import DatasetService

@pytest.mark.asyncio
async def test_create_dataset(db_session):
    dataset = await DatasetService.create(
        db_session,
        dataset_id="pile",
        source="huggingface"
    )
    assert dataset.id.startswith("ds_")
    assert dataset.status == "downloading"
    assert dataset.progress == 0

@pytest.mark.asyncio
async def test_delete_dataset(db_session, tmp_path):
    dataset = await DatasetService.create(db_session, "pile", "huggingface")
    dataset.file_path = str(tmp_path / dataset.id)

    await DatasetService.delete(db_session, dataset.id)

    # Verify database record deleted
    result = await DatasetService.get(db_session, dataset.id)
    assert result is None

    # Verify filesystem cleaned up
    assert not (tmp_path / dataset.id).exists()
```

### Integration Testing

**API Endpoint Tests:**
```python
# tests/test_api_datasets.py
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_list_datasets(client: AsyncClient):
    response = await client.get("/api/datasets")
    assert response.status_code == 200
    assert "datasets" in response.json()

@pytest.mark.asyncio
async def test_download_dataset(client: AsyncClient):
    response = await client.post(
        "/api/datasets/download",
        json={"dataset_id": "pile", "source": "huggingface"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "downloading"
    assert data["id"].startswith("ds_")
```

### End-to-End Testing

**Playwright Tests:**
```typescript
// e2e/datasets.spec.ts
import { test, expect } from '@playwright/test';

test('download dataset flow', async ({ page }) => {
  await page.goto('http://localhost:5173');

  // Click download button
  await page.click('text=Download Dataset');

  // Fill form
  await page.fill('input[name="dataset_id"]', 'pile');
  await page.click('text=Start Download');

  // Wait for dataset card to appear
  await expect(page.locator('text=The Pile')).toBeVisible({ timeout: 5000 });

  // Verify progress bar exists
  await expect(page.locator('[role="progressbar"]')).toBeVisible();
});
```

### Test Coverage Goals

- **Frontend:** 80% code coverage (components, hooks, utils)
- **Backend:** 90% code coverage (services, routes, workers)
- **E2E:** Cover all critical user paths (download, browse, delete)

---

## 11. Deployment & DevOps

### Docker Configuration

**Backend Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run migrations and start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "5173:80"
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/mistudio
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/data
    depends_on:
      - postgres
      - redis

  celery-worker:
    build: ./backend
    command: celery -A app.workers.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/mistudio
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/data
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mistudio
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Environment Configuration

```bash
# .env.production
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/mistudio
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=http://mistudio.mcslab.io
DATA_PATH=/data
```

### Monitoring & Logging

**Logging Strategy:**
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('/var/log/mistudio/app.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info(f"Dataset {dataset_id} download started")
logger.error(f"Dataset {dataset_id} download failed: {error}")
```

**Metrics Collection:**
- Prometheus endpoint: `/metrics`
- Track: API response times, Celery task durations, database query times
- Grafana dashboard for visualization

---

## 12. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **HuggingFace API rate limiting** | Medium | High | Implement exponential backoff, cache dataset metadata locally |
| **Large dataset download failures** | High | Medium | Implement resume capability, save partial downloads, retry logic |
| **Filesystem fills up** | Medium | High | Monitor disk usage, implement cleanup policies, warn user at 80% |
| **Database connection exhaustion** | Low | High | Use SQLAlchemy connection pooling (max 10 connections), async queries |
| **WebSocket connection drops** | Medium | Low | Implement client-side reconnection with exponential backoff |

### Mitigation Strategies

**HuggingFace Rate Limiting:**
```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)
async def download_with_retry(dataset_id):
    try:
        dataset = load_dataset(dataset_id, streaming=True)
        return dataset
    except RateLimitError:
        logger.warning(f"Rate limited on {dataset_id}, retrying...")
        raise
```

**Partial Download Resume:**
```python
def resume_download(dataset_id, last_checkpoint):
    # HuggingFace datasets library supports streaming
    dataset = load_dataset(
        dataset_id,
        streaming=True,
        split=f"train[{last_checkpoint}:]"  # Resume from checkpoint
    )
    return dataset
```

**Disk Space Monitoring:**
```python
import shutil

def check_disk_space(path: str, required_bytes: int) -> bool:
    stat = shutil.disk_usage(path)
    free_bytes = stat.free

    if free_bytes < required_bytes:
        raise InsufficientDiskSpaceError(
            f"Required: {required_bytes}, Available: {free_bytes}"
        )

    if free_bytes < stat.total * 0.2:  # Less than 20% free
        logger.warning(f"Low disk space: {free_bytes / 1e9:.2f} GB remaining")

    return True
```

---

## 13. Development Phases

### Phase 1: Core Infrastructure (Week 1)
**Goal:** Set up project structure, database, basic API

**Tasks:**
1. Initialize project repositories (frontend, backend)
2. Set up Docker Compose development environment
3. Create database schema with Alembic migrations
4. Implement basic FastAPI routes (CRUD operations)
5. Create Zustand store for dataset management
6. Implement basic React components (DatasetsPanel, DatasetCard)

**Deliverables:**
- Running Docker Compose environment
- Database migrations
- Basic API endpoints (GET, POST, DELETE)
- Simple UI showing dataset list

**Dependencies:** None
**Estimated Effort:** 40 hours (1 developer)

---

### Phase 2: Download & Background Jobs (Week 2)
**Goal:** Implement HuggingFace download with Celery workers

**Tasks:**
1. Set up Celery + Redis
2. Implement download_dataset_task worker
3. Add progress tracking via database updates
4. Implement WebSocket server for real-time updates
5. Add WebSocket client in frontend
6. Update UI to show download progress

**Deliverables:**
- Working Celery task queue
- Dataset download from HuggingFace
- Real-time progress bar in UI
- WebSocket connection management

**Dependencies:** Phase 1
**Estimated Effort:** 50 hours (1 developer)

---

### Phase 3: Tokenization & Statistics (Week 3)
**Goal:** Tokenize datasets and calculate statistics

**Tasks:**
1. Implement tokenize_dataset_task worker
2. Add tokenizer selection logic (GPT-2, Llama, etc.)
3. Calculate dataset statistics (num_tokens, vocab_size, etc.)
4. Create DatasetStatistics component
5. Implement sample browsing with pagination
6. Add full-text search on samples

**Deliverables:**
- Tokenization worker functional
- Statistics calculation
- Sample browser UI
- Search functionality

**Dependencies:** Phase 2
**Estimated Effort:** 45 hours (1 developer)

---

### Phase 4: Dataset Detail Modal (Week 4)
**Goal:** Complete detailed dataset view

**Tasks:**
1. Create DatasetDetailModal component structure
2. Implement Overview tab
3. Implement Samples tab with DatasetSamplesBrowser
4. Implement Tokenization tab with settings
5. Implement Statistics tab with visualizations
6. Add modal animations and transitions

**Deliverables:**
- Complete DatasetDetailModal matching Mock UI
- All 4 tabs functional
- Sample browsing with search/filter

**Dependencies:** Phase 3
**Estimated Effort:** 35 hours (1 developer)

---

### Phase 5: Testing & Polish (Week 5)
**Goal:** Comprehensive testing and bug fixes

**Tasks:**
1. Write unit tests (frontend components, backend services)
2. Write integration tests (API endpoints, Celery tasks)
3. Write E2E tests (critical user flows)
4. Fix bugs discovered during testing
5. Performance optimization (caching, indexing)
6. Documentation (API docs, deployment guide)

**Deliverables:**
- 80% frontend test coverage
- 90% backend test coverage
- E2E tests passing
- Performance benchmarks met
- Deployment documentation

**Dependencies:** Phase 4
**Estimated Effort:** 40 hours (1 developer)

---

### Total Timeline
**Estimated Duration:** 5 weeks (210 hours)
**Team Size:** 1 full-time developer
**Risk Buffer:** +20% (1 additional week for unknowns)

### Milestones
- **Week 1:** Basic infrastructure running
- **Week 2:** Download functionality working
- **Week 3:** Tokenization complete
- **Week 4:** UI feature complete
- **Week 5:** Production ready

---

## 14. Appendix

### Technology Decision Matrix

| Consideration | Zustand | Redux Toolkit | Jotai |
|--------------|---------|---------------|-------|
| Learning Curve | Low | Medium | Low |
| Boilerplate | Minimal | Medium | Minimal |
| DevTools | Yes | Excellent | Limited |
| Real-time Updates | Excellent | Good | Excellent |
| **Decision** | ✅ **Selected** | | |

**Rationale:** Zustand chosen for simplicity, minimal boilerplate, excellent TypeScript support, and easy integration with WebSocket updates.

### Alternative Approaches Considered

**File Storage Alternatives:**
1. **PostgreSQL BYTEA:** Store files in database
   - ❌ Rejected: Poor performance for large files, database bloat
2. **S3-compatible Storage:** Use MinIO or AWS S3
   - ✅ Future enhancement: Better for distributed systems
3. **Local Filesystem:** Current approach
   - ✅ Selected: Simple, fast, sufficient for edge device

**Background Job Alternatives:**
1. **FastAPI BackgroundTasks:** Built-in background task execution
   - ❌ Rejected: No distribution, no monitoring, limited to single process
2. **RQ (Redis Queue):** Simpler alternative to Celery
   - ❌ Rejected: Less features, smaller ecosystem
3. **Celery:** Current approach
   - ✅ Selected: Mature, distributed, excellent monitoring tools

### References

- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **Celery Documentation:** https://docs.celeryproject.org/
- **HuggingFace Datasets:** https://huggingface.co/docs/datasets/
- **Zustand Documentation:** https://docs.pmnd.rs/zustand/
- **SQLAlchemy Async:** https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- **Mock UI Reference:** `Mock-embedded-interp-ui.tsx` lines 1108-1202

---

**Document End**
**Total Sections:** 14
**Estimated Implementation Time:** 5 weeks (1 developer)
**Next Step:** Create Technical Implementation Document (TID) with specific coding hints and file organization

