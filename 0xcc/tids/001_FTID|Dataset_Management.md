# Technical Implementation Document: Dataset Management

**Document ID:** 001_FTID|Dataset_Management
**Feature:** Dataset Management System
**PRD Reference:** 001_FPRD|Dataset_Management.md
**TDD Reference:** 001_FTDD|Dataset_Management.md
**Status:** Ready for Implementation
**Created:** 2025-10-06
**Last Updated:** 2025-10-06

---

## 1. Implementation Overview

This Technical Implementation Document provides specific, actionable guidance for implementing the Dataset Management feature. The implementation follows the architecture defined in the TDD and strictly adheres to the Mock UI design (lines 1108-1202) as the PRIMARY reference.

**Key Implementation Principles:**
- **Mock UI is PRIMARY:** All frontend components must match Mock-embedded-interp-ui.tsx exactly (styling, behavior, structure)
- **No Mocking in Production:** All functionality must be genuinely functional (no mock data in production code)
- **TypeScript Strict Mode:** All TypeScript files use strict type checking
- **Async/Await:** All asynchronous operations use async/await (no callbacks)
- **Error Handling:** Every API call, file operation, and external service call has proper error handling

**Integration Points:**
- Frontend: React components in `src/components/panels/`
- Backend: FastAPI routes in `src/api/routes/datasets.py`
- State Management: Zustand store in `src/stores/datasetsStore.ts`
- Database: SQLAlchemy models in `src/models/dataset.py`
- Background Jobs: Celery tasks in `src/workers/dataset_tasks.py`

---

## 2. File Structure and Organization

### Directory Organization

```
miStudio/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── panels/
│   │   │   │   ├── DatasetsPanel.tsx                 # Main panel (lines 1108-1201)
│   │   │   │   └── DatasetsPanel.test.tsx
│   │   │   ├── datasets/
│   │   │   │   ├── DatasetCard.tsx                   # Dataset card component
│   │   │   │   ├── DatasetCard.test.tsx
│   │   │   │   ├── DatasetDetailModal.tsx            # Detail modal
│   │   │   │   ├── DatasetDetailModal.test.tsx
│   │   │   │   ├── DownloadForm.tsx                  # HF download form (lines 1118-1163)
│   │   │   │   ├── DownloadForm.test.tsx
│   │   │   │   ├── DatasetSamplesBrowser.tsx         # Samples tab
│   │   │   │   ├── DatasetStatistics.tsx             # Statistics tab
│   │   │   │   └── TokenizationSettings.tsx          # Tokenization tab
│   │   │   ├── common/
│   │   │   │   ├── StatusBadge.tsx                   # Reusable status badge
│   │   │   │   └── ProgressBar.tsx                   # Reusable progress bar
│   │   ├── stores/
│   │   │   ├── datasetsStore.ts                      # Zustand store
│   │   │   └── datasetsStore.test.ts
│   │   ├── api/
│   │   │   ├── datasets.ts                           # API client functions
│   │   │   └── websocket.ts                          # WebSocket client
│   │   ├── types/
│   │   │   └── dataset.ts                            # TypeScript interfaces
│   │   └── utils/
│   │       ├── formatters.ts                         # Format file size, dates, etc.
│   │       └── validators.ts                         # Input validation helpers
│   │
├── backend/
│   ├── src/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── datasets.py                       # FastAPI router
│   │   │       └── __init__.py
│   │   ├── services/
│   │   │   ├── dataset_service.py                    # Business logic
│   │   │   ├── tokenization_service.py               # Tokenization logic
│   │   │   └── statistics_service.py                 # Statistics calculation
│   │   ├── models/
│   │   │   ├── dataset.py                            # SQLAlchemy model
│   │   │   └── __init__.py
│   │   ├── schemas/
│   │   │   ├── dataset.py                            # Pydantic schemas
│   │   │   └── __init__.py
│   │   ├── workers/
│   │   │   ├── dataset_tasks.py                      # Celery tasks
│   │   │   └── __init__.py
│   │   ├── core/
│   │   │   ├── config.py                             # Configuration
│   │   │   ├── database.py                           # DB connection
│   │   │   ├── celery_app.py                         # Celery app
│   │   │   └── websocket.py                          # WebSocket manager
│   │   └── utils/
│   │       ├── file_utils.py                         # File operations
│   │       └── hf_utils.py                           # HuggingFace helpers
│   │
│   └── tests/
│       ├── unit/
│       │   ├── test_dataset_service.py
│       │   ├── test_tokenization_service.py
│       │   └── test_dataset_tasks.py
│       └── integration/
│           ├── test_dataset_api.py
│           └── test_dataset_workflow.py
```

### File Naming Conventions

**Frontend:**
- **Components:** PascalCase with `.tsx` extension (e.g., `DatasetsPanel.tsx`)
- **Tests:** Component name + `.test.tsx` (e.g., `DatasetsPanel.test.tsx`)
- **Stores:** camelCase + `Store.ts` (e.g., `datasetsStore.ts`)
- **Types:** camelCase + `.ts` (e.g., `dataset.ts`)
- **API Clients:** camelCase + `.ts` (e.g., `datasets.ts`)

**Backend:**
- **Routes:** lowercase with underscores (e.g., `datasets.py`)
- **Services:** lowercase with underscores + `_service.py` (e.g., `dataset_service.py`)
- **Models:** lowercase with underscores (e.g., `dataset.py`)
- **Schemas:** lowercase with underscores (e.g., `dataset.py`)
- **Tasks:** lowercase with underscores + `_tasks.py` (e.g., `dataset_tasks.py`)
- **Tests:** `test_` prefix + file name (e.g., `test_dataset_service.py`)

### Import Patterns

**Frontend Imports (Order):**
```typescript
// 1. React and external libraries
import React, { useState, useEffect } from 'react';
import { Download, Database, CheckCircle, Loader, Activity } from 'lucide-react';

// 2. Internal stores and hooks
import { useDatasetsStore } from '@/stores/datasetsStore';
import { useWebSocket } from '@/hooks/useWebSocket';

// 3. Internal components
import { DatasetCard } from '@/components/datasets/DatasetCard';
import { StatusBadge } from '@/components/common/StatusBadge';

// 4. API clients and utilities
import { downloadDataset, getDatasets } from '@/api/datasets';
import { formatFileSize, formatDate } from '@/utils/formatters';

// 5. Types
import type { Dataset, DatasetStatus } from '@/types/dataset';

// 6. Styles (if any)
import './DatasetsPanel.css';
```

**Backend Imports (Order):**
```python
# 1. Standard library
import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

# 2. Third-party libraries
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from celery import Task

# 3. Internal core
from src.core.database import get_db
from src.core.config import settings

# 4. Internal models and schemas
from src.models.dataset import Dataset
from src.schemas.dataset import DatasetCreate, DatasetResponse

# 5. Internal services
from src.services.dataset_service import DatasetService

# 6. Internal utilities
from src.utils.file_utils import ensure_dir, get_file_size
```

---

## 3. Component Implementation Hints

### DatasetsPanel Component

**File:** `frontend/src/components/panels/DatasetsPanel.tsx`

**Purpose:** Main container component for dataset management (PRIMARY REFERENCE: Mock UI lines 1108-1201)

**Key Implementation Details:**

```typescript
import React, { useState, useEffect } from 'react';
import { Download, Database, CheckCircle, Loader, Activity } from 'lucide-react';
import { useDatasetsStore } from '@/stores/datasetsStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { DatasetCard } from '@/components/datasets/DatasetCard';
import { DatasetDetailModal } from '@/components/datasets/DatasetDetailModal';
import { DownloadForm } from '@/components/datasets/DownloadForm';
import type { Dataset } from '@/types/dataset';

interface DatasetsPanelProps {
  // No props needed - data comes from Zustand store
}

export const DatasetsPanel: React.FC<DatasetsPanelProps> = () => {
  // State management
  const datasets = useDatasetsStore((state) => state.datasets);
  const fetchDatasets = useDatasetsStore((state) => state.fetchDatasets);
  const downloadDataset = useDatasetsStore((state) => state.downloadDataset);

  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);

  // WebSocket subscription for real-time updates
  const { subscribe, unsubscribe } = useWebSocket();

  // Fetch datasets on mount
  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  // Subscribe to WebSocket updates for downloading/processing datasets
  useEffect(() => {
    datasets.forEach(ds => {
      if (ds.status === 'downloading' || ds.status === 'processing') {
        subscribe(`datasets/${ds.id}/progress`, (data) => {
          // Update handled by Zustand store
        });
      }
    });

    // Cleanup subscriptions
    return () => {
      datasets.forEach(ds => {
        unsubscribe(`datasets/${ds.id}/progress`);
      });
    };
  }, [datasets, subscribe, unsubscribe]);

  // Handle download submission
  const handleDownload = async (repo: string, token: string) => {
    try {
      await downloadDataset(repo, token);
    } catch (error) {
      console.error('Download failed:', error);
      // Error handling in Zustand store
    }
  };

  return (
    <div className="space-y-6">
      {/* Title - EXACTLY as Mock UI line 1115 */}
      <h2 className="text-2xl font-semibold">Dataset Management</h2>

      {/* Download Form - EXACTLY as Mock UI lines 1118-1163 */}
      <DownloadForm onDownload={handleDownload} />

      {/* Datasets List - EXACTLY as Mock UI lines 1165-1191 */}
      <div className="grid gap-4">
        {datasets.map((ds) => (
          <DatasetCard
            key={ds.id}
            dataset={ds}
            onClick={() => ds.status === 'ready' && setSelectedDataset(ds)}
          />
        ))}
      </div>

      {/* Detail Modal - EXACTLY as Mock UI lines 1194-1199 */}
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

**Styling Notes:**
- Background: `space-y-6` (consistent vertical spacing)
- Title: `text-2xl font-semibold` (matches Mock UI)
- Grid: `grid gap-4` (4-unit spacing between cards)
- All styling MUST match Mock UI exactly

---

### DatasetCard Component

**File:** `frontend/src/components/datasets/DatasetCard.tsx`

**Purpose:** Individual dataset card display (Mock UI lines 1167-1189)

```typescript
import React from 'react';
import { Database, CheckCircle, Loader, Activity } from 'lucide-react';
import type { Dataset } from '@/types/dataset';

interface DatasetCardProps {
  dataset: Dataset;
  onClick: () => void;
}

export const DatasetCard: React.FC<DatasetCardProps> = ({ dataset, onClick }) => {
  const isReady = dataset.status === 'ready';
  const isProcessing = dataset.status === 'downloading' || dataset.status === 'processing';

  // Status icon mapping
  const statusIcon = {
    ready: <CheckCircle className="w-5 h-5 text-emerald-400" />,
    downloading: <Loader className="w-5 h-5 text-blue-400 animate-spin" />,
    processing: <Activity className="w-5 h-5 text-yellow-400" />,
    error: <Activity className="w-5 h-5 text-red-400" />
  }[dataset.status] || null;

  return (
    <div
      onClick={isReady ? onClick : undefined}
      className={`bg-slate-900/50 border border-slate-800 rounded-lg p-6 ${
        isReady ? 'cursor-pointer hover:bg-slate-900/70 transition-colors' : ''
      }`}
    >
      <div className="flex items-center justify-between">
        {/* Left side: Icon + Info */}
        <div className="flex items-center gap-4">
          <Database className="w-8 h-8 text-blue-400" />
          <div>
            <h3 className="font-semibold text-lg">{dataset.name}</h3>
            <p className="text-sm text-slate-400">
              Source: {dataset.source} • Size: {dataset.size}
            </p>
          </div>
        </div>

        {/* Right side: Status Icon + Badge */}
        <div className="flex items-center gap-3">
          {statusIcon}
          <span className="text-sm capitalize px-3 py-1 bg-slate-800 rounded-full">
            {dataset.status}
          </span>
        </div>
      </div>

      {/* Progress bar for downloading/processing - Add if needed */}
      {isProcessing && dataset.progress !== undefined && (
        <div className="mt-4">
          <div className="flex justify-between text-sm text-slate-400 mb-2">
            <span>{dataset.status === 'downloading' ? 'Downloading' : 'Processing'}</span>
            <span>{dataset.progress.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-2">
            <div
              className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${dataset.progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
};
```

**Styling Checklist:**
- ✅ Background: `bg-slate-900/50` with `border border-slate-800`
- ✅ Hover: `hover:bg-slate-900/70` (only for ready datasets)
- ✅ Icon sizes: `w-8 h-8` for Database, `w-5 h-5` for status icons
- ✅ Icon colors: blue-400 (Database), emerald-400 (ready), blue-400 (downloading), yellow-400 (processing)
- ✅ Status badge: `bg-slate-800 rounded-full px-3 py-1`

---

### DownloadForm Component

**File:** `frontend/src/components/datasets/DownloadForm.tsx`

**Purpose:** HuggingFace download form (Mock UI lines 1118-1163)

```typescript
import React, { useState } from 'react';
import { Download } from 'lucide-react';

interface DownloadFormProps {
  onDownload: (repo: string, token: string) => Promise<void>;
}

export const DownloadForm: React.FC<DownloadFormProps> = ({ onDownload }) => {
  const [hfRepo, setHfRepo] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!hfRepo) return;

    setIsSubmitting(true);
    try {
      await onDownload(hfRepo, accessToken);
      // Clear form on success
      setHfRepo('');
      setAccessToken('');
    } catch (error) {
      // Error handled by parent
      console.error('Download submission failed:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <h3 className="text-lg font-semibold mb-4">Download from HuggingFace</h3>

      <div className="space-y-4">
        {/* Repository Input - EXACTLY Mock UI lines 1122-1133 */}
        <div>
          <label
            htmlFor="dataset-repo-input"
            className="block text-sm font-medium text-slate-300 mb-2"
          >
            Dataset Repository
          </label>
          <input
            id="dataset-repo-input"
            type="text"
            placeholder="e.g., roneneldan/TinyStories"
            value={hfRepo}
            onChange={(e) => setHfRepo(e.target.value)}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
          />
        </div>

        {/* Access Token Input - EXACTLY Mock UI lines 1135-1147 */}
        <div>
          <label
            htmlFor="dataset-token-input"
            className="block text-sm font-medium text-slate-300 mb-2"
          >
            Access Token (optional, for gated datasets)
          </label>
          <input
            id="dataset-token-input"
            type="password"
            placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
            value={accessToken}
            onChange={(e) => setAccessToken(e.target.value)}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 font-mono text-sm"
          />
        </div>

        {/* Download Button - EXACTLY Mock UI lines 1149-1161 */}
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!hfRepo || isSubmitting}
          className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
        >
          <Download className="w-5 h-5" />
          {isSubmitting ? 'Downloading...' : 'Download Dataset from HuggingFace'}
        </button>
      </div>
    </div>
  );
};
```

**Form Validation:**
- Repository field: Required, non-empty
- Access token: Optional (only for gated datasets)
- Button disabled when: empty repo OR already submitting

---

### DatasetDetailModal Component

**File:** `frontend/src/components/datasets/DatasetDetailModal.tsx`

**Purpose:** Full-screen modal showing dataset details, samples, statistics, and tokenization

```typescript
import React, { useState } from 'react';
import { X, FileText, BarChart, Settings } from 'lucide-react';
import { DatasetSamplesBrowser } from './DatasetSamplesBrowser';
import { DatasetStatistics } from './DatasetStatistics';
import { TokenizationSettings } from './TokenizationSettings';
import type { Dataset } from '@/types/dataset';

interface DatasetDetailModalProps {
  dataset: Dataset;
  onClose: () => void;
}

type Tab = 'samples' | 'statistics' | 'tokenization';

export const DatasetDetailModal: React.FC<DatasetDetailModalProps> = ({
  dataset,
  onClose
}) => {
  const [activeTab, setActiveTab] = useState<Tab>('samples');

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-slate-900 border border-slate-800 rounded-lg w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div>
            <h2 className="text-2xl font-semibold">{dataset.name}</h2>
            <p className="text-sm text-slate-400 mt-1">
              {dataset.num_samples?.toLocaleString()} samples • {dataset.size}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200"
            aria-label="Close modal"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-800 px-6">
          <button
            onClick={() => setActiveTab('samples')}
            className={`px-4 py-3 border-b-2 transition-colors ${
              activeTab === 'samples'
                ? 'border-emerald-500 text-emerald-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileText className="w-4 h-4 inline mr-2" />
            Samples
          </button>
          <button
            onClick={() => setActiveTab('statistics')}
            className={`px-4 py-3 border-b-2 transition-colors ${
              activeTab === 'statistics'
                ? 'border-emerald-500 text-emerald-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart className="w-4 h-4 inline mr-2" />
            Statistics
          </button>
          <button
            onClick={() => setActiveTab('tokenization')}
            className={`px-4 py-3 border-b-2 transition-colors ${
              activeTab === 'tokenization'
                ? 'border-emerald-500 text-emerald-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Settings className="w-4 h-4 inline mr-2" />
            Tokenization
          </button>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'samples' && <DatasetSamplesBrowser dataset={dataset} />}
          {activeTab === 'statistics' && <DatasetStatistics dataset={dataset} />}
          {activeTab === 'tokenization' && <TokenizationSettings dataset={dataset} />}
        </div>
      </div>
    </div>
  );
};
```

**Modal Styling:**
- Backdrop: `fixed inset-0 bg-black/50` (50% opacity black)
- Container: `bg-slate-900 border border-slate-800 rounded-lg`
- Size: `max-w-6xl max-h-[90vh]`
- Tabs: Active tab has `border-emerald-500 text-emerald-400`

---

## 4. Database Implementation Approach

### SQLAlchemy Model

**File:** `backend/src/models/dataset.py`

```python
from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Enum
from sqlalchemy.sql import func
from src.core.database import Base
import enum

class DatasetStatus(str, enum.Enum):
    """Dataset processing status."""
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class Dataset(Base):
    """Dataset model for storing dataset metadata."""
    __tablename__ = "datasets"

    # Primary key
    id = Column(String(255), primary_key=True)  # Format: ds_{uuid}

    # Basic info
    name = Column(String(500), nullable=False)
    source = Column(String(100), nullable=False)  # "huggingface", "local"
    hf_repo_id = Column(String(500), nullable=True)  # HuggingFace repo ID

    # Status and progress
    status = Column(Enum(DatasetStatus), nullable=False, default=DatasetStatus.DOWNLOADING)
    progress = Column(Float, default=0.0)  # 0-100
    error_message = Column(String(1000), nullable=True)

    # File paths
    raw_path = Column(String(1000), nullable=True)  # /data/datasets/ds_{id}/raw/
    tokenized_path = Column(String(1000), nullable=True)  # /data/datasets/ds_{id}/tokenized/

    # Statistics
    num_samples = Column(Integer, nullable=True)
    num_tokens = Column(Integer, nullable=True)
    avg_seq_length = Column(Float, nullable=True)
    vocab_size = Column(Integer, nullable=True)
    size_bytes = Column(Integer, nullable=True)

    # Additional metadata (JSONB for flexibility)
    metadata = Column(JSON, nullable=True)  # {"splits": {"train": 1000, "val": 200}, ...}

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Dataset(id={self.id}, name={self.name}, status={self.status})>"
```

**Key Fields Explained:**
- `id`: Primary key, format `ds_{uuid}` for consistency
- `status`: Enum (downloading, processing, ready, error)
- `progress`: Float 0-100 for download/processing progress
- `metadata`: JSONB for flexible additional data (splits, config, etc.)
- `size_bytes`: Raw size in bytes (format to human-readable in frontend)

### Database Migration

**File:** `backend/alembic/versions/001_create_datasets_table.py`

```python
"""Create datasets table

Revision ID: 001
Revises:
Create Date: 2025-10-06

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'datasets',
        sa.Column('id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('source', sa.String(100), nullable=False),
        sa.Column('hf_repo_id', sa.String(500), nullable=True),
        sa.Column('status', sa.Enum('downloading', 'processing', 'ready', 'error', name='dataset_status'), nullable=False),
        sa.Column('progress', sa.Float, default=0.0),
        sa.Column('error_message', sa.String(1000), nullable=True),
        sa.Column('raw_path', sa.String(1000), nullable=True),
        sa.Column('tokenized_path', sa.String(1000), nullable=True),
        sa.Column('num_samples', sa.Integer, nullable=True),
        sa.Column('num_tokens', sa.Integer, nullable=True),
        sa.Column('avg_seq_length', sa.Float, nullable=True),
        sa.Column('vocab_size', sa.Integer, nullable=True),
        sa.Column('size_bytes', sa.Integer, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now())
    )

    # Indexes for performance
    op.create_index('idx_datasets_status', 'datasets', ['status'])
    op.create_index('idx_datasets_source', 'datasets', ['source'])
    op.create_index('idx_datasets_created_at', 'datasets', ['created_at'])

def downgrade():
    op.drop_index('idx_datasets_created_at')
    op.drop_index('idx_datasets_source')
    op.drop_index('idx_datasets_status')
    op.drop_table('datasets')
    op.execute('DROP TYPE dataset_status')
```

**Index Strategy:**
- `idx_datasets_status`: Fast filtering by status (e.g., show only "ready" datasets)
- `idx_datasets_source`: Fast filtering by source (HuggingFace vs local)
- `idx_datasets_created_at`: Sort by creation date

---

## 5. API Implementation Strategy

### FastAPI Router

**File:** `backend/src/api/routes/datasets.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from src.core.database import get_db
from src.services.dataset_service import DatasetService
from src.schemas.dataset import (
    DatasetCreate,
    DatasetResponse,
    DatasetDownloadRequest,
    DatasetListResponse
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    status: Optional[str] = Query(None, description="Filter by status"),
    source: Optional[str] = Query(None, description="Filter by source"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List all datasets with optional filtering.

    Query Parameters:
    - status: Filter by status (downloading, processing, ready, error)
    - source: Filter by source (huggingface, local)
    - limit: Max results (default 100)
    - offset: Pagination offset (default 0)
    """
    service = DatasetService(db)
    datasets = await service.list_datasets(
        status=status,
        source=source,
        limit=limit,
        offset=offset
    )

    return {"datasets": datasets, "total": len(datasets)}

@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get single dataset by ID."""
    service = DatasetService(db)
    dataset = await service.get_dataset(dataset_id)

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )

    return dataset

@router.post("/download", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def download_dataset(
    request: DatasetDownloadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Download dataset from HuggingFace.

    Creates dataset record and enqueues background download task.
    """
    service = DatasetService(db)

    # Create dataset record (status='downloading')
    dataset = await service.create_dataset_from_hf(
        hf_repo_id=request.hf_repo_id,
        access_token=request.access_token
    )

    # Enqueue Celery task
    from src.workers.dataset_tasks import download_dataset_task
    download_dataset_task.delay(dataset.id, request.hf_repo_id, request.access_token)

    return dataset

@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete dataset and all associated files.

    Checks for dependencies (trainings using this dataset) before deletion.
    """
    service = DatasetService(db)

    # Check dependencies
    has_dependencies = await service.check_dependencies(dataset_id)
    if has_dependencies:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset is referenced by existing trainings. Cannot delete."
        )

    # Delete files and database record
    await service.delete_dataset(dataset_id)

    return None

@router.get("/{dataset_id}/samples", response_model=List[dict])
async def get_dataset_samples(
    dataset_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None, description="Full-text search"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated dataset samples with optional search.

    Query Parameters:
    - limit: Max samples per page (default 50)
    - offset: Pagination offset
    - search: Full-text search query
    """
    service = DatasetService(db)

    # Verify dataset exists
    dataset = await service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )

    # Load samples from file
    samples = await service.get_samples(
        dataset_id=dataset_id,
        limit=limit,
        offset=offset,
        search=search
    )

    return samples
```

**Error Handling Pattern:**
- Use FastAPI's `HTTPException` for all errors
- Standard status codes: 404 (Not Found), 409 (Conflict), 500 (Internal Error)
- Descriptive error messages for frontend display

---

### Pydantic Schemas

**File:** `backend/src/schemas/dataset.py`

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

class DatasetDownloadRequest(BaseModel):
    """Request schema for downloading dataset from HuggingFace."""
    hf_repo_id: str = Field(..., description="HuggingFace repository ID", example="roneneldan/TinyStories")
    access_token: Optional[str] = Field(None, description="HuggingFace access token for gated datasets")

    @validator('hf_repo_id')
    def validate_repo_id(cls, v):
        """Validate HuggingFace repo ID format."""
        if not v or '/' not in v:
            raise ValueError("Invalid HuggingFace repo ID. Format: username/dataset-name")
        return v

class DatasetResponse(BaseModel):
    """Response schema for dataset."""
    id: str
    name: str
    source: str
    hf_repo_id: Optional[str]
    status: str
    progress: float
    error_message: Optional[str]
    raw_path: Optional[str]
    tokenized_path: Optional[str]
    num_samples: Optional[int]
    num_tokens: Optional[int]
    avg_seq_length: Optional[float]
    vocab_size: Optional[int]
    size_bytes: Optional[int]
    size: str  # Human-readable size (computed field)
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: Optional[datetime]

    @validator('size', always=True)
    def compute_size(cls, v, values):
        """Compute human-readable size from size_bytes."""
        size_bytes = values.get('size_bytes')
        if not size_bytes:
            return "Unknown"

        # Convert bytes to human-readable
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    class Config:
        orm_mode = True

class DatasetListResponse(BaseModel):
    """Response schema for list of datasets."""
    datasets: List[DatasetResponse]
    total: int
```

**Validation Rules:**
- `hf_repo_id`: Must contain "/" (username/dataset format)
- `size`: Computed field (human-readable from size_bytes)
- `orm_mode = True`: Enables direct conversion from SQLAlchemy models

---

## 6. Frontend Implementation Approach

### Zustand Store

**File:** `frontend/src/stores/datasetsStore.ts`

```typescript
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { Dataset } from '@/types/dataset';
import * as api from '@/api/datasets';

interface DatasetsStore {
  // State
  datasets: Dataset[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchDatasets: () => Promise<void>;
  getDataset: (id: string) => Dataset | undefined;
  downloadDataset: (repo: string, token: string) => Promise<void>;
  deleteDataset: (id: string) => Promise<void>;
  updateDatasetProgress: (id: string, progress: number) => void;
  updateDatasetStatus: (id: string, status: string, error?: string) => void;
}

export const useDatasetsStore = create<DatasetsStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      datasets: [],
      loading: false,
      error: null,

      // Fetch all datasets
      fetchDatasets: async () => {
        set({ loading: true, error: null });
        try {
          const response = await api.getDatasets();
          set({ datasets: response.datasets, loading: false });
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to fetch datasets',
            loading: false
          });
        }
      },

      // Get single dataset by ID
      getDataset: (id: string) => {
        return get().datasets.find(ds => ds.id === id);
      },

      // Download dataset from HuggingFace
      downloadDataset: async (repo: string, token: string) => {
        try {
          const dataset = await api.downloadDataset(repo, token);

          // Add to datasets list
          set(state => ({
            datasets: [...state.datasets, dataset]
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Download failed'
          });
          throw error;
        }
      },

      // Delete dataset
      deleteDataset: async (id: string) => {
        try {
          await api.deleteDataset(id);

          // Remove from datasets list
          set(state => ({
            datasets: state.datasets.filter(ds => ds.id !== id)
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Delete failed'
          });
          throw error;
        }
      },

      // Update progress (from WebSocket)
      updateDatasetProgress: (id: string, progress: number) => {
        set(state => ({
          datasets: state.datasets.map(ds =>
            ds.id === id ? { ...ds, progress } : ds
          )
        }));
      },

      // Update status (from WebSocket)
      updateDatasetStatus: (id: string, status: string, error?: string) => {
        set(state => ({
          datasets: state.datasets.map(ds =>
            ds.id === id
              ? { ...ds, status, error_message: error || null }
              : ds
          )
        }));
      }
    }),
    { name: 'DatasetsStore' }
  )
);
```

**Store Patterns:**
- All async operations use `async/await`
- Loading and error states tracked for UI feedback
- WebSocket updates handled via `updateDatasetProgress` and `updateDatasetStatus`
- Devtools middleware for debugging

---

### API Client Functions

**File:** `frontend/src/api/datasets.ts`

```typescript
import type { Dataset, DatasetListResponse } from '@/types/dataset';

const API_BASE = '/api/datasets';

// Helper for auth token
function getAuthToken(): string {
  return localStorage.getItem('auth_token') || '';
}

// Helper for API calls
async function fetchAPI<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${getAuthToken()}`,
      ...options.headers
    }
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'API request failed');
  }

  return response.json();
}

/**
 * Get all datasets with optional filters.
 */
export async function getDatasets(params?: {
  status?: string;
  source?: string;
  limit?: number;
  offset?: number;
}): Promise<DatasetListResponse> {
  const queryParams = new URLSearchParams();
  if (params?.status) queryParams.set('status', params.status);
  if (params?.source) queryParams.set('source', params.source);
  if (params?.limit) queryParams.set('limit', params.limit.toString());
  if (params?.offset) queryParams.set('offset', params.offset.toString());

  const url = `${API_BASE}?${queryParams.toString()}`;
  return fetchAPI<DatasetListResponse>(url);
}

/**
 * Get single dataset by ID.
 */
export async function getDataset(id: string): Promise<Dataset> {
  return fetchAPI<Dataset>(`${API_BASE}/${id}`);
}

/**
 * Download dataset from HuggingFace.
 */
export async function downloadDataset(repo: string, token: string): Promise<Dataset> {
  return fetchAPI<Dataset>(`${API_BASE}/download`, {
    method: 'POST',
    body: JSON.stringify({
      hf_repo_id: repo,
      access_token: token || undefined
    })
  });
}

/**
 * Delete dataset.
 */
export async function deleteDataset(id: string): Promise<void> {
  await fetchAPI<void>(`${API_BASE}/${id}`, {
    method: 'DELETE'
  });
}

/**
 * Get dataset samples with pagination.
 */
export async function getDatasetSamples(
  id: string,
  params?: {
    limit?: number;
    offset?: number;
    search?: string;
  }
): Promise<any[]> {
  const queryParams = new URLSearchParams();
  if (params?.limit) queryParams.set('limit', params.limit.toString());
  if (params?.offset) queryParams.set('offset', params.offset.toString());
  if (params?.search) queryParams.set('search', params.search);

  const url = `${API_BASE}/${id}/samples?${queryParams.toString()}`;
  return fetchAPI<any[]>(url);
}
```

**API Client Patterns:**
- Single `fetchAPI` helper for all requests
- Automatic auth token injection
- Automatic error handling
- Type-safe with TypeScript generics

---

## 7. Business Logic Implementation Hints

### DatasetService

**File:** `backend/src/services/dataset_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional
import uuid
import os
from src.models.dataset import Dataset, DatasetStatus
from src.utils.file_utils import ensure_dir, get_directory_size

class DatasetService:
    """Service for dataset business logic."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_datasets(
        self,
        status: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dataset]:
        """List datasets with optional filtering."""
        query = select(Dataset)

        # Apply filters
        filters = []
        if status:
            filters.append(Dataset.status == status)
        if source:
            filters.append(Dataset.source == source)

        if filters:
            query = query.where(and_(*filters))

        # Pagination and sorting
        query = query.order_by(Dataset.created_at.desc()).limit(limit).offset(offset)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Get dataset by ID."""
        result = await self.db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        return result.scalar_one_or_none()

    async def create_dataset_from_hf(
        self,
        hf_repo_id: str,
        access_token: Optional[str] = None
    ) -> Dataset:
        """
        Create dataset record for HuggingFace download.

        Creates database entry with status='downloading' and returns immediately.
        Background Celery task will handle actual download.
        """
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"

        # Extract name from repo ID
        name = hf_repo_id.split('/')[-1]

        # Create dataset record
        dataset = Dataset(
            id=dataset_id,
            name=name,
            source="huggingface",
            hf_repo_id=hf_repo_id,
            status=DatasetStatus.DOWNLOADING,
            progress=0.0,
            metadata={"access_token_provided": bool(access_token)}
        )

        self.db.add(dataset)
        await self.db.commit()
        await self.db.refresh(dataset)

        return dataset

    async def update_progress(
        self,
        dataset_id: str,
        progress: float,
        status: Optional[DatasetStatus] = None
    ) -> None:
        """Update dataset download/processing progress."""
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        dataset.progress = progress
        if status:
            dataset.status = status

        await self.db.commit()

    async def delete_dataset(self, dataset_id: str) -> None:
        """
        Delete dataset record and files.

        Removes:
        - Database record
        - Raw files
        - Tokenized files
        """
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Delete files
        if dataset.raw_path and os.path.exists(dataset.raw_path):
            import shutil
            shutil.rmtree(dataset.raw_path)

        if dataset.tokenized_path and os.path.exists(dataset.tokenized_path):
            import shutil
            shutil.rmtree(dataset.tokenized_path)

        # Delete database record
        await self.db.delete(dataset)
        await self.db.commit()

    async def check_dependencies(self, dataset_id: str) -> bool:
        """
        Check if dataset is referenced by any trainings.

        Returns True if dependencies exist (cannot delete).
        """
        # TODO: Query trainings table
        # For now, return False (no dependencies)
        return False
```

**Service Pattern:**
- All database operations in service layer
- Services receive `AsyncSession` via dependency injection
- Business logic isolated from API routes
- Reusable across different contexts (API, CLI, background jobs)

---

### Celery Task: Download Dataset

**File:** `backend/src/workers/dataset_tasks.py`

```python
from celery import Task
from src.core.celery_app import celery_app
from src.core.database import AsyncSessionLocal
from src.services.dataset_service import DatasetService
from src.models.dataset import DatasetStatus
from datasets import load_dataset
import os

@celery_app.task(bind=True, max_retries=3)
def download_dataset_task(
    self: Task,
    dataset_id: str,
    hf_repo_id: str,
    access_token: str = None
):
    """
    Download dataset from HuggingFace.

    Background task that:
    1. Downloads dataset using HuggingFace datasets library
    2. Saves to /data/datasets/{dataset_id}/raw/
    3. Updates progress via WebSocket
    4. Auto-triggers tokenization on completion
    """

    # Setup database session
    async def run_download():
        async with AsyncSessionLocal() as db:
            service = DatasetService(db)

            try:
                # Update status
                await service.update_progress(dataset_id, 0.0, DatasetStatus.DOWNLOADING)

                # Create directory
                raw_path = f"/data/datasets/{dataset_id}/raw"
                os.makedirs(raw_path, exist_ok=True)

                # Download from HuggingFace
                # Note: progress tracking not directly supported, use periodic updates
                dataset = load_dataset(
                    hf_repo_id,
                    token=access_token if access_token else None,
                    cache_dir=raw_path
                )

                # Save to disk
                dataset.save_to_disk(raw_path)

                # Calculate statistics
                num_samples = sum(len(split) for split in dataset.values())
                size_bytes = get_directory_size(raw_path)

                # Update dataset record
                ds = await service.get_dataset(dataset_id)
                ds.raw_path = raw_path
                ds.num_samples = num_samples
                ds.size_bytes = size_bytes
                ds.status = DatasetStatus.READY
                ds.progress = 100.0
                await db.commit()

                # Emit WebSocket completion event
                from src.core.websocket import emit_event
                emit_event(f"datasets/{dataset_id}/progress", {
                    "type": "completed",
                    "progress": 100.0,
                    "status": "ready"
                })

            except Exception as e:
                # Handle error
                await service.update_progress(
                    dataset_id,
                    0.0,
                    DatasetStatus.ERROR
                )

                # Update error message
                ds = await service.get_dataset(dataset_id)
                ds.error_message = str(e)
                await db.commit()

                # Emit error event
                from src.core.websocket import emit_event
                emit_event(f"datasets/{dataset_id}/progress", {
                    "type": "error",
                    "error": str(e)
                })

                # Retry if possible
                if self.request.retries < self.max_retries:
                    raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))

    # Run async function
    import asyncio
    asyncio.run(run_download())

def get_directory_size(path: str) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total
```

**Celery Task Patterns:**
- `bind=True`: Access task instance as `self`
- `max_retries=3`: Retry up to 3 times on failure
- Async database operations wrapped in `asyncio.run()`
- WebSocket events for real-time UI updates
- Comprehensive error handling with retries

---

## 8. Testing Implementation Approach

### Unit Test Example

**File:** `backend/tests/unit/test_dataset_service.py`

```python
import pytest
from unittest.mock import Mock, AsyncMock
from src.services.dataset_service import DatasetService
from src.models.dataset import Dataset, DatasetStatus

@pytest.mark.asyncio
async def test_list_datasets_with_filter():
    """Test listing datasets with status filter."""
    # Mock database session
    mock_db = AsyncMock()
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = [
        Dataset(id="ds_1", name="Test Dataset", status=DatasetStatus.READY)
    ]
    mock_db.execute.return_value = mock_result

    # Create service
    service = DatasetService(mock_db)

    # List datasets
    datasets = await service.list_datasets(status="ready")

    # Assertions
    assert len(datasets) == 1
    assert datasets[0].status == DatasetStatus.READY

@pytest.mark.asyncio
async def test_create_dataset_from_hf():
    """Test creating dataset from HuggingFace repo."""
    mock_db = AsyncMock()

    service = DatasetService(mock_db)

    # Create dataset
    dataset = await service.create_dataset_from_hf(
        hf_repo_id="roneneldan/TinyStories",
        access_token=None
    )

    # Assertions
    assert dataset.id.startswith("ds_")
    assert dataset.name == "TinyStories"
    assert dataset.source == "huggingface"
    assert dataset.status == DatasetStatus.DOWNLOADING
    assert dataset.progress == 0.0
```

### Integration Test Example

**File:** `backend/tests/integration/test_dataset_api.py`

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_list_datasets():
    """Test GET /api/datasets endpoint."""
    response = client.get("/api/datasets")

    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert "total" in data

def test_download_dataset():
    """Test POST /api/datasets/download endpoint."""
    response = client.post("/api/datasets/download", json={
        "hf_repo_id": "roneneldan/TinyStories",
        "access_token": None
    })

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "downloading"
    assert data["source"] == "huggingface"

def test_delete_dataset():
    """Test DELETE /api/datasets/{id} endpoint."""
    # First create a dataset
    create_response = client.post("/api/datasets/download", json={
        "hf_repo_id": "test/dataset"
    })
    dataset_id = create_response.json()["id"]

    # Delete it
    delete_response = client.delete(f"/api/datasets/{dataset_id}")
    assert delete_response.status_code == 204

    # Verify deletion
    get_response = client.get(f"/api/datasets/{dataset_id}")
    assert get_response.status_code == 404
```

---

## 9. Configuration and Environment Strategy

### Environment Variables

**File:** `backend/.env`

```bash
# Database
DATABASE_URL=postgresql+asyncpg://mistudio:password@localhost/mistudio

# Redis (Celery broker)
REDIS_URL=redis://localhost:6379/0

# Data directories
DATA_DIR=/data
DATASETS_DIR=/data/datasets

# HuggingFace
HF_HOME=/data/huggingface_cache

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# WebSocket
WS_ENABLED=true
WS_PORT=8001
```

### Configuration Class

**File:** `backend/src/core/config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str

    # Redis
    redis_url: str

    # Data directories
    data_dir: str = "/data"
    datasets_dir: str = "/data/datasets"

    # HuggingFace
    hf_home: str = "/data/huggingface_cache"

    # Celery
    celery_broker_url: str
    celery_result_backend: str

    # WebSocket
    ws_enabled: bool = True
    ws_port: int = 8001

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 10. Integration Strategy

### WebSocket Integration

**File:** `frontend/src/hooks/useWebSocket.ts`

```typescript
import { useEffect, useCallback } from 'react';
import io, { Socket } from 'socket.io-client';
import { useDatasetsStore } from '@/stores/datasetsStore';

let socket: Socket | null = null;

export function useWebSocket() {
  useEffect(() => {
    // Initialize socket connection
    if (!socket) {
      socket = io('ws://localhost:8001', {
        transports: ['websocket']
      });

      socket.on('connect', () => {
        console.log('WebSocket connected');
      });

      socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
      });
    }

    return () => {
      // Don't disconnect on unmount (keep connection alive)
    };
  }, []);

  const subscribe = useCallback((channel: string, callback: (data: any) => void) => {
    if (!socket) return;

    socket.on(channel, callback);
  }, []);

  const unsubscribe = useCallback((channel: string) => {
    if (!socket) return;

    socket.off(channel);
  }, []);

  return { subscribe, unsubscribe };
}

// Pre-configured hook for dataset progress updates
export function useDatasetProgress(datasetId: string) {
  const { updateDatasetProgress, updateDatasetStatus } = useDatasetsStore();
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    const channel = `datasets/${datasetId}/progress`;

    const handleUpdate = (data: any) => {
      if (data.type === 'progress') {
        updateDatasetProgress(datasetId, data.progress);
      } else if (data.type === 'completed') {
        updateDatasetStatus(datasetId, 'ready');
      } else if (data.type === 'error') {
        updateDatasetStatus(datasetId, 'error', data.error);
      }
    };

    subscribe(channel, handleUpdate);

    return () => {
      unsubscribe(channel);
    };
  }, [datasetId, subscribe, unsubscribe, updateDatasetProgress, updateDatasetStatus]);
}
```

---

## 11. Utilities and Helpers Design

### File Utilities

**File:** `backend/src/utils/file_utils.py`

```python
import os
import shutil
from typing import Optional

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def get_directory_size(path: str) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total

def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(path) if os.path.exists(path) else 0

def delete_directory(path: str) -> None:
    """Recursively delete directory."""
    if os.path.exists(path):
        shutil.rmtree(path)

def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
```

### Frontend Formatters

**File:** `frontend/src/utils/formatters.ts`

```typescript
export function formatFileSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

export function formatDate(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
}

export function formatDateTime(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}
```

---

## 12. Implementation Patterns

### Pattern: Deep Metadata Merge

**Function:** `deep_merge_metadata()`
**Location:** `backend/src/services/dataset_service.py:176-188`

**Purpose:** Preserve existing metadata when adding new sections

**Implementation:**
```python
def deep_merge_metadata(existing: Optional[dict], new: Optional[dict]) -> dict:
    """
    Deep merge two metadata dictionaries, preserving existing data.

    Rules:
    - None values in new dict are skipped (preserve existing)
    - Nested dicts are merged recursively
    - Non-dict values are overwritten
    """
    if not existing:
        return new if new else {}
    if not new:
        return existing

    merged = existing.copy()
    for key, value in new.items():
        if value is None:
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_metadata(merged[key], value)
        else:
            merged[key] = value
    return merged
```

**Usage:**
```python
# In DatasetService.update_dataset()
if updates.metadata:
    merged_metadata = deep_merge_metadata(
        dataset.extra_metadata,
        updates.metadata
    )
    dataset.extra_metadata = merged_metadata
```

---

### Pattern: Pydantic Metadata Validation

**Schema:** `TokenizationMetadata`
**Location:** `backend/src/schemas/metadata.py:15-50`

**Purpose:** Validate tokenization statistics structure and constraints

**Key Features:**
```python
class TokenizationMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    tokenizer_name: str
    max_length: int = Field(ge=1, le=8192)  # Bounds checking
    num_tokens: int = Field(ge=0)  # Non-negative

    @field_validator('avg_seq_length')
    @classmethod
    def validate_avg_seq_length(cls, v, info):
        # Cross-field validation
        if v > info.data['max_length']:
            raise ValueError('avg cannot exceed max_length')
        return v
```

**Integration:**
```python
# In DatasetUpdate schema
class DatasetUpdate(BaseModel):
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('metadata')
    @classmethod
    def validate_metadata_structure(cls, v):
        if v and 'tokenization' in v:
            TokenizationMetadata(**v['tokenization'])
        return v
```

---

## 13. Error Handling and Logging Strategy

### Error Handling Pattern

**Backend:**
```python
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

async def handle_dataset_operation():
    try:
        # Operation
        result = await some_operation()
        return result
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file not found: {e}"
        )
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access dataset"
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

**Frontend:**
```typescript
async function handleAPIError(error: unknown): Promise<string> {
  if (error instanceof Error) {
    // Parse FastAPI error response
    try {
      const errorData = JSON.parse(error.message);
      return errorData.detail || 'An error occurred';
    } catch {
      return error.message;
    }
  }
  return 'An unknown error occurred';
}

// Usage in component
try {
  await downloadDataset(repo, token);
} catch (error) {
  const message = await handleAPIError(error);
  setError(message);
  // Optionally show toast notification
}
```

---

## 13. Performance Implementation Hints

### Database Query Optimization

```python
# BAD: N+1 query problem
for dataset in datasets:
    samples = await get_samples(dataset.id)  # Individual query for each dataset

# GOOD: Use eager loading or batch queries
datasets_with_samples = await db.execute(
    select(Dataset)
    .options(selectinload(Dataset.samples))
    .where(Dataset.status == 'ready')
)
```

### Frontend Performance

```typescript
// Use React.memo for expensive components
export const DatasetCard = React.memo<DatasetCardProps>(({ dataset, onClick }) => {
  // Component implementation
}, (prevProps, nextProps) => {
  // Custom comparison (only re-render if dataset changed)
  return prevProps.dataset.id === nextProps.dataset.id &&
         prevProps.dataset.status === nextProps.dataset.status &&
         prevProps.dataset.progress === nextProps.dataset.progress;
});

// Use useMemo for expensive computations
const sortedDatasets = useMemo(() => {
  return datasets.sort((a, b) =>
    new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );
}, [datasets]);
```

---

## 14. Code Quality and Standards

### TypeScript Standards

```typescript
// ALWAYS use strict types (no 'any')
interface Dataset {
  id: string;
  name: string;
  status: 'downloading' | 'processing' | 'ready' | 'error';  // Literal types
  progress: number;
  created_at: Date;
}

// ALWAYS handle null/undefined explicitly
function getDatasetName(dataset: Dataset | null): string {
  return dataset?.name ?? 'Unknown Dataset';
}

// ALWAYS use async/await (not .then())
async function fetchData() {
  try {
    const data = await api.getDatasets();
    return data;
  } catch (error) {
    console.error('Fetch failed:', error);
    throw error;
  }
}
```

### Python Standards

```python
# ALWAYS use type hints
from typing import List, Optional

async def get_datasets(
    status: Optional[str] = None,
    limit: int = 100
) -> List[Dataset]:
    """Get datasets with optional filter."""
    # Implementation

# ALWAYS use async/await for I/O operations
async def download_dataset(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.content

# ALWAYS use f-strings (not % or .format())
logger.info(f"Downloaded dataset {dataset_id} ({size_bytes} bytes)")
```

---

**Document End**
**Status:** Ready for Task Generation
**Next Step:** Use this TID to generate detailed task list via @0xcc/instruct/006_generate-tasks.md
**Total Sections:** 14
**Estimated Size:** ~40KB
