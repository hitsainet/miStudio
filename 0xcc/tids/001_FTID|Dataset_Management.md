# Technical Implementation Document: Dataset Management

**Document ID:** 001_FTID|Dataset_Management
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related TDD:** [001_FTDD|Dataset_Management](../tdds/001_FTDD|Dataset_Management.md)

---

## 1. Implementation Order

### Phase 1: Backend Foundation
1. Database migration (dataset table)
2. SQLAlchemy model
3. Pydantic schemas
4. Dataset service layer
5. API endpoints

### Phase 2: HuggingFace Integration
1. HuggingFace service for downloads
2. Celery task for background downloads
3. WebSocket progress emission
4. Download resumption logic

### Phase 3: Frontend
1. Zustand store
2. API client functions
3. DatasetCard component
4. DatasetDownloadForm component
5. DatasetsPanel integration

---

## 2. File-by-File Implementation

### 2.1 Backend Files

#### `backend/src/models/dataset.py`
```python
from sqlalchemy import Column, String, Integer, BigInteger, Enum, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from src.db.base import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    source = Column(String(50), nullable=False)  # huggingface, local
    repo_id = Column(String(255))  # HuggingFace repo ID
    local_path = Column(String(500))
    status = Column(String(50), default="pending")
    total_samples = Column(Integer)
    downloaded_bytes = Column(BigInteger, default=0)
    total_bytes = Column(BigInteger)
    config = Column(JSONB)  # subset, split, etc.
    error_message = Column(String)
    created_at = Column(TIMESTAMP, server_default="now()")
    updated_at = Column(TIMESTAMP, onupdate="now()")
```

**Key Implementation Notes:**
- Use `UUID(as_uuid=True)` for proper PostgreSQL UUID handling
- JSONB for flexible config storage (subset, split, revision)
- Status enum: pending, downloading, processing, ready, failed

#### `backend/src/schemas/dataset.py`
```python
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime

class DatasetCreate(BaseModel):
    name: str
    source: str = "huggingface"
    repo_id: Optional[str] = None
    config: Optional[dict] = None

class DatasetResponse(BaseModel):
    id: UUID
    name: str
    source: str
    repo_id: Optional[str]
    local_path: Optional[str]
    status: str
    total_samples: Optional[int]
    downloaded_bytes: int
    total_bytes: Optional[int]
    config: Optional[dict]
    error_message: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class DatasetDownloadRequest(BaseModel):
    repo_id: str
    name: Optional[str] = None
    subset: Optional[str] = None
    split: Optional[str] = "train"
    max_samples: Optional[int] = None
```

**Key Implementation Notes:**
- Use `from_attributes = True` (Pydantic v2) instead of `orm_mode`
- Optional fields allow flexibility for different dataset sources

#### `backend/src/services/dataset_service.py`
```python
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from src.models.dataset import Dataset
from src.schemas.dataset import DatasetCreate, DatasetDownloadRequest

class DatasetService:
    def __init__(self, db: Session):
        self.db = db

    def create(self, data: DatasetCreate) -> Dataset:
        dataset = Dataset(**data.model_dump())
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset

    def get_by_id(self, dataset_id: UUID) -> Optional[Dataset]:
        return self.db.query(Dataset).filter(Dataset.id == dataset_id).first()

    def list_all(self, status: Optional[str] = None) -> List[Dataset]:
        query = self.db.query(Dataset)
        if status:
            query = query.filter(Dataset.status == status)
        return query.order_by(Dataset.created_at.desc()).all()

    def update_status(self, dataset_id: UUID, status: str, **kwargs):
        dataset = self.get_by_id(dataset_id)
        if dataset:
            dataset.status = status
            for key, value in kwargs.items():
                setattr(dataset, key, value)
            self.db.commit()
        return dataset

    def delete(self, dataset_id: UUID) -> bool:
        dataset = self.get_by_id(dataset_id)
        if dataset:
            self.db.delete(dataset)
            self.db.commit()
            return True
        return False
```

**Key Implementation Notes:**
- Always call `db.refresh()` after commit to get updated values
- Use `model_dump()` (Pydantic v2) instead of `dict()`
- Return None for not found, let API handle 404

#### `backend/src/services/huggingface_dataset_service.py`
```python
import os
from pathlib import Path
from datasets import load_dataset, DownloadConfig
from huggingface_hub import snapshot_download, hf_hub_download
from src.core.config import settings

class HuggingFaceDatasetService:
    def __init__(self):
        self.cache_dir = Path(settings.data_dir) / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(
        self,
        repo_id: str,
        subset: str = None,
        split: str = "train",
        max_samples: int = None,
        progress_callback=None
    ):
        """Download dataset from HuggingFace."""
        local_path = self.cache_dir / repo_id.replace("/", "_")

        # Load with streaming first to get size
        ds = load_dataset(
            repo_id,
            subset,
            split=split,
            streaming=True
        )

        # Now download fully
        ds = load_dataset(
            repo_id,
            subset,
            split=split,
            cache_dir=str(local_path)
        )

        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        # Save to disk in efficient format
        ds.save_to_disk(str(local_path / "processed"))

        return {
            "local_path": str(local_path / "processed"),
            "total_samples": len(ds),
            "columns": ds.column_names
        }

    def get_dataset_info(self, repo_id: str):
        """Get dataset metadata without downloading."""
        from huggingface_hub import dataset_info
        info = dataset_info(repo_id)
        return {
            "id": info.id,
            "description": info.description,
            "citation": info.citation,
            "size_categories": info.size_categories
        }
```

**Key Implementation Notes:**
- Use `streaming=True` first to get size info without full download
- Store in Arrow format via `save_to_disk()` for fast loading
- Replace `/` with `_` in repo_id for filesystem paths

#### `backend/src/workers/dataset_tasks.py`
```python
from celery import shared_task
from src.db.session import SessionLocal
from src.services.dataset_service import DatasetService
from src.services.huggingface_dataset_service import HuggingFaceDatasetService
from src.workers.websocket_emitter import emit_dataset_progress

@shared_task(bind=True, queue='default')
def download_dataset_task(self, dataset_id: str, config: dict):
    """Background task for dataset download."""
    db = SessionLocal()
    try:
        service = DatasetService(db)
        hf_service = HuggingFaceDatasetService()

        # Update status to downloading
        service.update_status(dataset_id, "downloading")
        emit_dataset_progress(dataset_id, 0, "Starting download...")

        def progress_callback(progress, message):
            emit_dataset_progress(dataset_id, progress, message)

        # Download
        result = hf_service.download_dataset(
            repo_id=config["repo_id"],
            subset=config.get("subset"),
            split=config.get("split", "train"),
            max_samples=config.get("max_samples"),
            progress_callback=progress_callback
        )

        # Update with results
        service.update_status(
            dataset_id,
            "ready",
            local_path=result["local_path"],
            total_samples=result["total_samples"]
        )
        emit_dataset_progress(dataset_id, 100, "Complete", completed=True)

    except Exception as e:
        service.update_status(dataset_id, "failed", error_message=str(e))
        emit_dataset_progress(dataset_id, 0, str(e), failed=True)
    finally:
        db.close()
```

**Key Implementation Notes:**
- Always use `bind=True` for access to `self.request.id`
- Create new DB session for each task (Celery workers are separate processes)
- Always close DB session in finally block
- Emit WebSocket progress at key milestones

#### `backend/src/api/v1/endpoints/datasets.py`
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from src.db.session import get_db
from src.services.dataset_service import DatasetService
from src.schemas.dataset import DatasetResponse, DatasetDownloadRequest
from src.workers.dataset_tasks import download_dataset_task

router = APIRouter()

@router.get("", response_model=List[DatasetResponse])
def list_datasets(
    status: str = None,
    db: Session = Depends(get_db)
):
    service = DatasetService(db)
    return service.list_all(status=status)

@router.post("/download", response_model=DatasetResponse, status_code=status.HTTP_202_ACCEPTED)
def download_dataset(
    request: DatasetDownloadRequest,
    db: Session = Depends(get_db)
):
    service = DatasetService(db)

    # Create dataset record
    dataset = service.create({
        "name": request.name or request.repo_id.split("/")[-1],
        "source": "huggingface",
        "repo_id": request.repo_id,
        "status": "pending",
        "config": {
            "subset": request.subset,
            "split": request.split,
            "max_samples": request.max_samples
        }
    })

    # Queue download task
    download_dataset_task.delay(str(dataset.id), dataset.config)

    return dataset

@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: UUID, db: Session = Depends(get_db)):
    service = DatasetService(db)
    dataset = service.get_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: UUID, db: Session = Depends(get_db)):
    service = DatasetService(db)
    if not service.delete(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
```

**Key Implementation Notes:**
- Use 202 Accepted for async operations (download)
- Use 204 No Content for successful deletes
- Pass UUID directly to service, FastAPI handles parsing

### 2.2 Frontend Files

#### `frontend/src/types/dataset.ts`
```typescript
export interface Dataset {
  id: string;
  name: string;
  source: 'huggingface' | 'local';
  repo_id?: string;
  local_path?: string;
  status: 'pending' | 'downloading' | 'processing' | 'ready' | 'failed';
  total_samples?: number;
  downloaded_bytes: number;
  total_bytes?: number;
  config?: {
    subset?: string;
    split?: string;
    max_samples?: number;
  };
  error_message?: string;
  created_at: string;
}

export interface DatasetDownloadRequest {
  repo_id: string;
  name?: string;
  subset?: string;
  split?: string;
  max_samples?: number;
}
```

#### `frontend/src/api/datasets.ts`
```typescript
import axios from 'axios';
import { Dataset, DatasetDownloadRequest } from '../types/dataset';

const API_BASE = '/api/v1/datasets';

export const datasetsApi = {
  list: async (status?: string): Promise<Dataset[]> => {
    const params = status ? { status } : {};
    const { data } = await axios.get(API_BASE, { params });
    return data;
  },

  download: async (request: DatasetDownloadRequest): Promise<Dataset> => {
    const { data } = await axios.post(`${API_BASE}/download`, request);
    return data;
  },

  get: async (id: string): Promise<Dataset> => {
    const { data } = await axios.get(`${API_BASE}/${id}`);
    return data;
  },

  delete: async (id: string): Promise<void> => {
    await axios.delete(`${API_BASE}/${id}`);
  }
};
```

#### `frontend/src/stores/datasetsStore.ts`
```typescript
import { create } from 'zustand';
import { Dataset, DatasetDownloadRequest } from '../types/dataset';
import { datasetsApi } from '../api/datasets';

interface DatasetsState {
  datasets: Dataset[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchDatasets: () => Promise<void>;
  downloadDataset: (request: DatasetDownloadRequest) => Promise<Dataset>;
  deleteDataset: (id: string) => Promise<void>;
  updateDatasetProgress: (id: string, progress: Partial<Dataset>) => void;
}

export const useDatasetsStore = create<DatasetsState>((set, get) => ({
  datasets: [],
  loading: false,
  error: null,

  fetchDatasets: async () => {
    set({ loading: true, error: null });
    try {
      const datasets = await datasetsApi.list();
      set({ datasets, loading: false });
    } catch (error) {
      set({ error: 'Failed to fetch datasets', loading: false });
    }
  },

  downloadDataset: async (request) => {
    const dataset = await datasetsApi.download(request);
    set(state => ({
      datasets: [dataset, ...state.datasets]
    }));
    return dataset;
  },

  deleteDataset: async (id) => {
    await datasetsApi.delete(id);
    set(state => ({
      datasets: state.datasets.filter(d => d.id !== id)
    }));
  },

  updateDatasetProgress: (id, progress) => {
    set(state => ({
      datasets: state.datasets.map(d =>
        d.id === id ? { ...d, ...progress } : d
      )
    }));
  }
}));
```

**Key Implementation Notes:**
- Use `create` from zustand (not `createStore`)
- Optimistic updates for better UX
- `updateDatasetProgress` for WebSocket updates

#### `frontend/src/components/datasets/DownloadForm.tsx`
```typescript
import React, { useState } from 'react';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { DatasetDownloadRequest } from '../../types/dataset';

export function DownloadForm({ onClose }: { onClose: () => void }) {
  const downloadDataset = useDatasetsStore(s => s.downloadDataset);
  const [form, setForm] = useState<DatasetDownloadRequest>({
    repo_id: '',
    split: 'train'
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await downloadDataset(form);
      onClose();
    } catch (error) {
      console.error('Download failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm text-slate-400 mb-1">
          HuggingFace Repository
        </label>
        <input
          type="text"
          value={form.repo_id}
          onChange={e => setForm({ ...form, repo_id: e.target.value })}
          placeholder="username/dataset-name"
          className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
          required
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-slate-400 mb-1">
            Subset (optional)
          </label>
          <input
            type="text"
            value={form.subset || ''}
            onChange={e => setForm({ ...form, subset: e.target.value || undefined })}
            className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-sm text-slate-400 mb-1">
            Split
          </label>
          <select
            value={form.split}
            onChange={e => setForm({ ...form, split: e.target.value })}
            className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
          >
            <option value="train">train</option>
            <option value="test">test</option>
            <option value="validation">validation</option>
          </select>
        </div>
      </div>

      <div className="flex justify-end gap-2 pt-4">
        <button
          type="button"
          onClick={onClose}
          className="px-4 py-2 text-slate-400 hover:text-white"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={loading || !form.repo_id}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded disabled:opacity-50"
        >
          {loading ? 'Starting...' : 'Download'}
        </button>
      </div>
    </form>
  );
}
```

---

## 3. Common Patterns

### 3.1 WebSocket Progress Hook
```typescript
// frontend/src/hooks/useDatasetWebSocket.ts
import { useEffect } from 'react';
import { socket } from '../api/websocket';
import { useDatasetsStore } from '../stores/datasetsStore';

export function useDatasetWebSocket(datasetId: string | null) {
  const updateProgress = useDatasetsStore(s => s.updateDatasetProgress);

  useEffect(() => {
    if (!datasetId) return;

    const channel = `dataset/${datasetId}`;
    socket.emit('join', channel);

    const handleProgress = (data: any) => {
      updateProgress(datasetId, {
        downloaded_bytes: data.downloaded_bytes,
        status: data.status
      });
    };

    socket.on('progress', handleProgress);

    return () => {
      socket.emit('leave', channel);
      socket.off('progress', handleProgress);
    };
  }, [datasetId, updateProgress]);
}
```

### 3.2 Progress Bar Component
```typescript
// frontend/src/components/common/ProgressBar.tsx
interface ProgressBarProps {
  progress: number;  // 0-100
  label?: string;
  variant?: 'default' | 'success' | 'error';
}

export function ProgressBar({ progress, label, variant = 'default' }: ProgressBarProps) {
  const colors = {
    default: 'bg-emerald-500',
    success: 'bg-green-500',
    error: 'bg-red-500'
  };

  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-sm text-slate-400 mb-1">
          <span>{label}</span>
          <span>{progress.toFixed(0)}%</span>
        </div>
      )}
      <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${colors[variant]} transition-all duration-300`}
          style={{ width: `${Math.min(100, progress)}%` }}
        />
      </div>
    </div>
  );
}
```

---

## 4. Testing Strategy

### 4.1 Backend Tests
```python
# backend/tests/test_dataset_service.py
import pytest
from src.services.dataset_service import DatasetService

def test_create_dataset(db_session):
    service = DatasetService(db_session)
    dataset = service.create({
        "name": "test-dataset",
        "source": "huggingface",
        "repo_id": "test/dataset"
    })
    assert dataset.id is not None
    assert dataset.status == "pending"

def test_list_datasets_by_status(db_session):
    service = DatasetService(db_session)
    # Create test data...
    ready = service.list_all(status="ready")
    assert all(d.status == "ready" for d in ready)
```

### 4.2 Frontend Tests
```typescript
// frontend/src/components/datasets/DownloadForm.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { DownloadForm } from './DownloadForm';

test('submits form with repo_id', async () => {
  const onClose = vi.fn();
  render(<DownloadForm onClose={onClose} />);

  fireEvent.change(screen.getByPlaceholderText(/username/), {
    target: { value: 'test/dataset' }
  });
  fireEvent.click(screen.getByText('Download'));

  // Assert API called...
});
```

---

## 5. Common Pitfalls

### Pitfall 1: Database Session Lifecycle
```python
# WRONG - Session leak
@router.post("/download")
def download(request: Request):
    db = SessionLocal()  # Never closed!
    # ...

# RIGHT - Use dependency injection
@router.post("/download")
def download(request: Request, db: Session = Depends(get_db)):
    # Session managed by FastAPI
    # ...
```

### Pitfall 2: Celery Task Serialization
```python
# WRONG - Passing SQLAlchemy object
download_task.delay(dataset)  # Can't serialize!

# RIGHT - Pass serializable data
download_task.delay(str(dataset.id), dataset.config)
```

### Pitfall 3: WebSocket Room Naming
```python
# WRONG - Inconsistent naming
emit_to_room(f"dataset-{id}")  # Hyphen
socket.emit('join', `dataset/${id}`)  # Slash

# RIGHT - Consistent convention
# Backend and frontend must use same pattern
channel = f"dataset/{dataset_id}"  # Always slash
```

---

## 6. Performance Tips

1. **Batch Database Operations**
   ```python
   # Instead of N queries
   for id in dataset_ids:
       dataset = db.query(Dataset).get(id)

   # Do 1 query
   datasets = db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()
   ```

2. **Stream Large Downloads**
   ```python
   # Use streaming to avoid memory issues
   ds = load_dataset(repo_id, streaming=True)
   for batch in ds.iter(batch_size=1000):
       process(batch)
   ```

3. **Debounce Progress Updates**
   ```python
   # Don't emit every byte
   last_emit = 0
   if progress - last_emit >= 5:  # Every 5%
       emit_progress(progress)
       last_emit = progress
   ```

---

*Related: [PRD](../prds/001_FPRD|Dataset_Management.md) | [TDD](../tdds/001_FTDD|Dataset_Management.md) | [FTASKS](../tasks/001_FTASKS|Dataset_Management.md)*
