# Technical Implementation Document: Model Management

**Document ID:** 002_FTID|Model_Management
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related TDD:** [002_FTDD|Model_Management](../tdds/002_FTDD|Model_Management.md)

---

## 1. Implementation Order

### Phase 1: Backend Foundation
1. Database migration (models table)
2. SQLAlchemy model
3. Pydantic schemas
4. Model service layer
5. API endpoints

### Phase 2: HuggingFace Integration
1. HuggingFace download service
2. Celery task for background downloads
3. WebSocket progress emission
4. Model loading utilities

### Phase 3: Quantization
1. bitsandbytes integration
2. Quantization service
3. Memory estimation utilities

### Phase 4: Frontend
1. Zustand store
2. API client functions
3. ModelCard component
4. ModelDownloadForm component
5. ModelsPanel integration

---

## 2. File-by-File Implementation

### 2.1 Backend Files

#### `backend/src/models/model.py`
```python
from sqlalchemy import Column, String, Integer, BigInteger, Boolean, Enum, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from src.db.base import Base

class Model(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    model_id = Column(String(255), nullable=False)  # HuggingFace model ID
    revision = Column(String(100))
    local_path = Column(String(500))
    status = Column(String(50), default="pending")

    # Model info
    architecture = Column(String(100))
    hidden_size = Column(Integer)
    num_layers = Column(Integer)
    num_heads = Column(Integer)
    vocab_size = Column(Integer)

    # Quantization
    is_quantized = Column(Boolean, default=False)
    quantization_type = Column(String(50))  # 4bit, 8bit, none

    # Size tracking
    size_bytes = Column(BigInteger)
    downloaded_bytes = Column(BigInteger, default=0)

    # Metadata
    config = Column(JSONB)
    error_message = Column(String)
    created_at = Column(TIMESTAMP, server_default="now()")
    updated_at = Column(TIMESTAMP, onupdate="now()")
```

**Key Implementation Notes:**
- Store both `name` (display) and `model_id` (HuggingFace ID)
- Track model architecture details for SAE compatibility checking
- Quantization fields for bitsandbytes support

#### `backend/src/services/model_service.py`
```python
from pathlib import Path
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from src.models.model import Model
from src.core.config import settings

class ModelService:
    def __init__(self, db: Session):
        self.db = db
        self.models_dir = Path(settings.data_dir) / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def create(self, data: dict) -> Model:
        model = Model(**data)
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model

    def get_by_id(self, model_id: UUID) -> Optional[Model]:
        return self.db.query(Model).filter(Model.id == model_id).first()

    def get_by_model_id(self, model_id: str) -> Optional[Model]:
        """Get by HuggingFace model ID."""
        return self.db.query(Model).filter(Model.model_id == model_id).first()

    def list_ready(self) -> List[Model]:
        """List only ready-to-use models."""
        return self.db.query(Model).filter(
            Model.status == "ready"
        ).order_by(Model.created_at.desc()).all()

    def list_all(self) -> List[Model]:
        return self.db.query(Model).order_by(Model.created_at.desc()).all()

    def update_status(self, model_id: UUID, status: str, **kwargs):
        model = self.get_by_id(model_id)
        if model:
            model.status = status
            for key, value in kwargs.items():
                setattr(model, key, value)
            self.db.commit()
        return model

    def delete(self, model_id: UUID) -> bool:
        model = self.get_by_id(model_id)
        if model:
            # Clean up local files
            if model.local_path:
                import shutil
                shutil.rmtree(model.local_path, ignore_errors=True)
            self.db.delete(model)
            self.db.commit()
            return True
        return False
```

#### `backend/src/services/huggingface_model_service.py`
```python
import os
from pathlib import Path
from typing import Callable, Optional
from huggingface_hub import snapshot_download, HfApi
from transformers import AutoConfig
from src.core.config import settings

class HuggingFaceModelService:
    def __init__(self):
        self.cache_dir = Path(settings.data_dir) / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()

    def get_model_info(self, model_id: str) -> dict:
        """Get model metadata without downloading."""
        try:
            config = AutoConfig.from_pretrained(model_id)
            info = self.api.model_info(model_id)

            return {
                "model_id": model_id,
                "architecture": config.architectures[0] if config.architectures else None,
                "hidden_size": getattr(config, "hidden_size", None),
                "num_layers": getattr(config, "num_hidden_layers", None),
                "num_heads": getattr(config, "num_attention_heads", None),
                "vocab_size": getattr(config, "vocab_size", None),
                "size_bytes": sum(s.size for s in info.siblings if s.rfilename.endswith(('.bin', '.safetensors')))
            }
        except Exception as e:
            raise ValueError(f"Failed to get model info: {e}")

    def download_model(
        self,
        model_id: str,
        revision: str = None,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Download model files from HuggingFace."""
        local_dir = self.cache_dir / model_id.replace("/", "_")

        # Download with progress tracking
        path = snapshot_download(
            model_id,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        return path

    def estimate_memory(self, model_id: str, quantization: str = None) -> dict:
        """Estimate memory requirements."""
        info = self.get_model_info(model_id)
        size_gb = info["size_bytes"] / (1024**3)

        if quantization == "4bit":
            vram_gb = size_gb * 0.3  # ~30% of original
        elif quantization == "8bit":
            vram_gb = size_gb * 0.55  # ~55% of original
        else:
            vram_gb = size_gb * 1.2  # ~120% for inference overhead

        return {
            "model_size_gb": round(size_gb, 2),
            "estimated_vram_gb": round(vram_gb, 2),
            "quantization": quantization
        }
```

**Key Implementation Notes:**
- Use `snapshot_download` for complete model download
- `local_dir_use_symlinks=False` ensures actual files are downloaded
- `resume_download=True` supports interrupted downloads

#### `backend/src/services/quantization_service.py`
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional

class QuantizationService:
    @staticmethod
    def get_quantization_config(quantization_type: str) -> Optional[BitsAndBytesConfig]:
        """Get bitsandbytes config for quantization."""
        if quantization_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantization_type == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True
            )
        return None

    @staticmethod
    def load_model(
        model_path: str,
        quantization_type: str = None,
        device_map: str = "auto"
    ):
        """Load model with optional quantization."""
        config = QuantizationService.get_quantization_config(quantization_type)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=config,
            device_map=device_map,
            torch_dtype=torch.float16 if not config else None,
            trust_remote_code=True
        )

        return model
```

**Key Implementation Notes:**
- Use `nf4` (NormalFloat4) for best quality 4-bit quantization
- `use_double_quant=True` further reduces memory
- `device_map="auto"` handles multi-GPU distribution

#### `backend/src/workers/model_tasks.py`
```python
from celery import shared_task
from src.db.session import SessionLocal
from src.services.model_service import ModelService
from src.services.huggingface_model_service import HuggingFaceModelService
from src.workers.websocket_emitter import emit_model_download_progress

@shared_task(bind=True, queue='default')
def download_model_task(self, model_id: str, config: dict):
    """Background task for model download."""
    db = SessionLocal()
    try:
        service = ModelService(db)
        hf_service = HuggingFaceModelService()

        # Update status
        service.update_status(model_id, "downloading")
        emit_model_download_progress(model_id, 0, "Starting download...")

        # Get model info first
        info = hf_service.get_model_info(config["model_id"])
        service.update_status(
            model_id,
            "downloading",
            architecture=info["architecture"],
            hidden_size=info["hidden_size"],
            num_layers=info["num_layers"],
            num_heads=info["num_heads"],
            vocab_size=info["vocab_size"],
            size_bytes=info["size_bytes"]
        )

        # Download
        local_path = hf_service.download_model(
            config["model_id"],
            revision=config.get("revision")
        )

        # Update final status
        service.update_status(
            model_id,
            "ready",
            local_path=local_path,
            downloaded_bytes=info["size_bytes"]
        )
        emit_model_download_progress(model_id, 100, "Complete", completed=True)

    except Exception as e:
        service.update_status(model_id, "failed", error_message=str(e))
        emit_model_download_progress(model_id, 0, str(e), failed=True)
    finally:
        db.close()
```

### 2.2 Frontend Files

#### `frontend/src/types/model.ts`
```typescript
export interface Model {
  id: string;
  name: string;
  model_id: string;
  revision?: string;
  local_path?: string;
  status: 'pending' | 'downloading' | 'ready' | 'failed';

  // Architecture info
  architecture?: string;
  hidden_size?: number;
  num_layers?: number;
  num_heads?: number;
  vocab_size?: number;

  // Quantization
  is_quantized: boolean;
  quantization_type?: '4bit' | '8bit' | 'none';

  // Size
  size_bytes?: number;
  downloaded_bytes: number;

  error_message?: string;
  created_at: string;
}

export interface ModelDownloadRequest {
  model_id: string;
  name?: string;
  revision?: string;
  quantization_type?: '4bit' | '8bit' | 'none';
}

export interface ModelInfo {
  model_id: string;
  architecture?: string;
  hidden_size?: number;
  num_layers?: number;
  size_bytes?: number;
}
```

#### `frontend/src/stores/modelsStore.ts`
```typescript
import { create } from 'zustand';
import { Model, ModelDownloadRequest } from '../types/model';
import { modelsApi } from '../api/models';

interface ModelsState {
  models: Model[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchModels: () => Promise<void>;
  downloadModel: (request: ModelDownloadRequest) => Promise<Model>;
  deleteModel: (id: string) => Promise<void>;
  updateModelProgress: (id: string, progress: Partial<Model>) => void;

  // Selectors
  getReadyModels: () => Model[];
  getModelById: (id: string) => Model | undefined;
}

export const useModelsStore = create<ModelsState>((set, get) => ({
  models: [],
  loading: false,
  error: null,

  fetchModels: async () => {
    set({ loading: true, error: null });
    try {
      const models = await modelsApi.list();
      set({ models, loading: false });
    } catch (error) {
      set({ error: 'Failed to fetch models', loading: false });
    }
  },

  downloadModel: async (request) => {
    const model = await modelsApi.download(request);
    set(state => ({
      models: [model, ...state.models]
    }));
    return model;
  },

  deleteModel: async (id) => {
    await modelsApi.delete(id);
    set(state => ({
      models: state.models.filter(m => m.id !== id)
    }));
  },

  updateModelProgress: (id, progress) => {
    set(state => ({
      models: state.models.map(m =>
        m.id === id ? { ...m, ...progress } : m
      )
    }));
  },

  getReadyModels: () => {
    return get().models.filter(m => m.status === 'ready');
  },

  getModelById: (id) => {
    return get().models.find(m => m.id === id);
  }
}));
```

#### `frontend/src/components/models/ModelDownloadForm.tsx`
```typescript
import React, { useState, useEffect } from 'react';
import { useModelsStore } from '../../stores/modelsStore';
import { modelsApi } from '../../api/models';
import { ModelDownloadRequest, ModelInfo } from '../../types/model';
import { formatBytes } from '../../utils/formatters';

export function ModelDownloadForm({ onClose }: { onClose: () => void }) {
  const downloadModel = useModelsStore(s => s.downloadModel);
  const [form, setForm] = useState<ModelDownloadRequest>({
    model_id: '',
    quantization_type: 'none'
  });
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [infoLoading, setInfoLoading] = useState(false);

  // Fetch model info when model_id changes (debounced)
  useEffect(() => {
    if (!form.model_id || form.model_id.length < 3) {
      setModelInfo(null);
      return;
    }

    const timer = setTimeout(async () => {
      setInfoLoading(true);
      try {
        const info = await modelsApi.getModelInfo(form.model_id);
        setModelInfo(info);
      } catch {
        setModelInfo(null);
      } finally {
        setInfoLoading(false);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [form.model_id]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await downloadModel(form);
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
          HuggingFace Model ID
        </label>
        <input
          type="text"
          value={form.model_id}
          onChange={e => setForm({ ...form, model_id: e.target.value })}
          placeholder="google/gemma-2-2b"
          className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
          required
        />
      </div>

      {/* Model Info Preview */}
      {modelInfo && (
        <div className="bg-slate-800/50 rounded p-3 text-sm">
          <div className="text-slate-400">Model Info:</div>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <div>Architecture: {modelInfo.architecture}</div>
            <div>Hidden Size: {modelInfo.hidden_size}</div>
            <div>Layers: {modelInfo.num_layers}</div>
            <div>Size: {formatBytes(modelInfo.size_bytes || 0)}</div>
          </div>
        </div>
      )}

      <div>
        <label className="block text-sm text-slate-400 mb-1">
          Quantization
        </label>
        <select
          value={form.quantization_type}
          onChange={e => setForm({ ...form, quantization_type: e.target.value as any })}
          className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2"
        >
          <option value="none">None (Full Precision)</option>
          <option value="8bit">8-bit (bitsandbytes)</option>
          <option value="4bit">4-bit (bitsandbytes NF4)</option>
        </select>
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
          disabled={loading || !form.model_id}
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

### 3.1 Model Loading with Caching
```python
# backend/src/ml/model_loader.py
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ModelLoader:
    _instance = None
    _loaded_models = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_path: str, quantization: str = None):
        """Load model with caching."""
        cache_key = f"{model_path}_{quantization}"

        if cache_key not in self._loaded_models:
            model = self._load_model_impl(model_path, quantization)
            self._loaded_models[cache_key] = model

        return self._loaded_models[cache_key]

    def unload_model(self, model_path: str, quantization: str = None):
        """Unload model to free memory."""
        cache_key = f"{model_path}_{quantization}"
        if cache_key in self._loaded_models:
            del self._loaded_models[cache_key]
            torch.cuda.empty_cache()
```

### 3.2 Memory Monitoring During Download
```typescript
// frontend/src/hooks/useModelDownloadWebSocket.ts
import { useEffect, useState } from 'react';
import { socket } from '../api/websocket';
import { useModelsStore } from '../stores/modelsStore';

export function useModelDownloadWebSocket(modelId: string | null) {
  const updateProgress = useModelsStore(s => s.updateModelProgress);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!modelId) return;

    const channel = `model/${modelId}`;
    socket.emit('join', channel);

    const handleProgress = (data: any) => {
      setProgress(data.progress);
      updateProgress(modelId, {
        downloaded_bytes: data.downloaded_bytes,
        status: data.status
      });
    };

    const handleComplete = () => {
      updateProgress(modelId, { status: 'ready' });
    };

    socket.on('download_progress', handleProgress);
    socket.on('download_completed', handleComplete);

    return () => {
      socket.emit('leave', channel);
      socket.off('download_progress', handleProgress);
      socket.off('download_completed', handleComplete);
    };
  }, [modelId, updateProgress]);

  return { progress };
}
```

---

## 4. Testing Strategy

### 4.1 Backend Tests
```python
# backend/tests/test_model_service.py
import pytest
from unittest.mock import patch, MagicMock

def test_get_model_info():
    service = HuggingFaceModelService()
    with patch('transformers.AutoConfig.from_pretrained') as mock_config:
        mock_config.return_value = MagicMock(
            architectures=['GemmaForCausalLM'],
            hidden_size=2048,
            num_hidden_layers=18
        )
        info = service.get_model_info('google/gemma-2-2b')
        assert info['architecture'] == 'GemmaForCausalLM'

def test_quantization_config():
    config = QuantizationService.get_quantization_config('4bit')
    assert config.load_in_4bit == True
    assert config.bnb_4bit_quant_type == 'nf4'
```

### 4.2 Frontend Tests
```typescript
// frontend/src/components/models/ModelDownloadForm.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ModelDownloadForm } from './ModelDownloadForm';

test('fetches model info on input', async () => {
  render(<ModelDownloadForm onClose={() => {}} />);

  fireEvent.change(screen.getByPlaceholderText(/gemma/i), {
    target: { value: 'google/gemma-2-2b' }
  });

  await waitFor(() => {
    expect(screen.getByText(/Architecture/)).toBeInTheDocument();
  });
});
```

---

## 5. Common Pitfalls

### Pitfall 1: CUDA Out of Memory
```python
# WRONG - Loading without memory check
model = AutoModelForCausalLM.from_pretrained(path)

# RIGHT - Check available memory first
import torch
available = torch.cuda.get_device_properties(0).total_memory
if model_size > available * 0.8:
    raise MemoryError("Insufficient GPU memory")
```

### Pitfall 2: Token Length Mismatch
```python
# WRONG - Using tokenizer without model config
tokenizer = AutoTokenizer.from_pretrained(model_id)

# RIGHT - Ensure tokenizer matches model
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    model_max_length=model.config.max_position_embeddings
)
```

### Pitfall 3: Concurrent Model Loading
```python
# WRONG - Multiple processes loading same model
# Causes GPU memory fragmentation

# RIGHT - Use singleton pattern with locks
import threading

class ModelLoader:
    _lock = threading.Lock()

    def load_model(self, path):
        with self._lock:
            # Thread-safe loading
            return self._load_impl(path)
```

---

## 6. Performance Tips

1. **Use SafeTensors Format**
   ```python
   # Faster loading than .bin files
   model = AutoModelForCausalLM.from_pretrained(
       path,
       use_safetensors=True
   )
   ```

2. **Lazy Loading for Model Info**
   ```python
   # Don't load full model for metadata
   config = AutoConfig.from_pretrained(model_id)  # Fast
   # vs
   model = AutoModelForCausalLM.from_pretrained(model_id)  # Slow
   ```

3. **GPU Memory Management**
   ```python
   # Clean up between model loads
   def cleanup():
       import gc
       gc.collect()
       torch.cuda.empty_cache()
   ```

---

*Related: [PRD](../prds/002_FPRD|Model_Management.md) | [TDD](../tdds/002_FTDD|Model_Management.md) | [FTASKS](../tasks/002_FTASKS|Model_Management.md)*
