# Technical Implementation Document: Model Management

**Document ID:** 002_FTID|Model_Management
**Feature:** Language Model Loading, Configuration, and Activation Extraction
**PRD Reference:** 002_FPRD|Model_Management.md
**TDD Reference:** 002_FTDD|Model_Management.md
**Status:** Ready for Implementation
**Created:** 2025-10-06

---

## 1. Implementation Overview

This TID provides implementation guidance for the Model Management feature, focusing on PyTorch model loading with quantization, activation extraction via forward hooks, and React components that match Mock UI lines 1204-1625 EXACTLY.

**Key Implementation Principles:**
- **Mock UI is PRIMARY:** Match lines 1204-1625 exactly (ModelsPanel, ModelArchitectureViewer, ActivationExtractionConfig)
- **No Mock Data in Production:** All model loading and extraction genuinely functional
- **Memory Optimization:** INT4/INT8 quantization for 6GB GPU VRAM constraint (Jetson Orin Nano)
- **Forward Hooks:** Non-invasive activation extraction without model architecture modification

**Integration Points:**
- Frontend: `src/components/panels/ModelsPanel.tsx` (lines 1204-1343)
- Backend: `src/api/routes/models.py`, `src/services/model_service.py`
- State: `src/stores/modelsStore.ts`
- ML: PyTorch model loading, bitsandbytes quantization, forward hooks

---

## 2. File Structure and Organization

```
miStudio/
├── frontend/src/
│   ├── components/
│   │   ├── panels/
│   │   │   └── ModelsPanel.tsx                    # Lines 1204-1343
│   │   ├── models/
│   │   │   ├── ModelCard.tsx                      # Individual model card
│   │   │   ├── ModelArchitectureViewer.tsx        # Architecture modal (lines 1346-1437)
│   │   │   ├── ActivationExtractionConfig.tsx     # Extraction config (lines 1440-1625)
│   │   │   └── LayerSelector.tsx                  # Layer grid selector (lines 1538-1554)
│   ├── stores/
│   │   └── modelsStore.ts                         # Zustand store
│   ├── api/
│   │   └── models.ts                              # API client
│   └── types/
│       └── model.ts                               # TypeScript interfaces
│
├── backend/src/
│   ├── api/routes/
│   │   └── models.py                              # FastAPI router
│   ├── services/
│   │   ├── model_service.py                       # Model lifecycle management
│   │   ├── quantization_service.py                # Quantization logic
│   │   └── activation_service.py                  # Activation extraction
│   ├── models/
│   │   └── model.py                               # SQLAlchemy model
│   ├── schemas/
│   │   └── model.py                               # Pydantic schemas
│   └── ml/
│       ├── model_loader.py                        # PyTorch model loading
│       ├── quantize.py                            # bitsandbytes integration
│       └── forward_hooks.py                       # Hook registration
```

---

## 3. Component Implementation Hints

### ModelsPanel Component

**File:** `frontend/src/components/panels/ModelsPanel.tsx`

**PRIMARY REFERENCE:** Mock UI lines 1204-1343

```typescript
import React, { useState, useEffect } from 'react';
import { Download, Cpu, CheckCircle, Loader, Activity } from 'lucide-react';
import { useModelsStore } from '@/stores/modelsStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { ModelCard } from '@/components/models/ModelCard';
import { ModelArchitectureViewer } from '@/components/models/ModelArchitectureViewer';
import { ActivationExtractionConfig } from '@/components/models/ActivationExtractionConfig';
import type { Model } from '@/types/model';

export const ModelsPanel: React.FC = () => {
  // State from store
  const models = useModelsStore((state) => state.models);
  const downloadModel = useModelsStore((state) => state.downloadModel);
  const fetchModels = useModelsStore((state) => state.fetchModels);

  // Local state - EXACTLY as Mock UI lines 1206-1210
  const [hfModelRepo, setHfModelRepo] = useState('');
  const [quantization, setQuantization] = useState<'FP16' | 'Q8' | 'Q4' | 'Q2'>('Q4');
  const [accessToken, setAccessToken] = useState('');
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [showExtractionConfig, setShowExtractionConfig] = useState(false);

  // Fetch models on mount
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // WebSocket subscription for download progress
  const { subscribe, unsubscribe } = useWebSocket();
  useEffect(() => {
    models.forEach(model => {
      if (model.status === 'downloading' || model.status === 'quantizing') {
        subscribe(`models/${model.id}/progress`, () => {
          // Store handles updates
        });
      }
    });
    return () => {
      models.forEach(model => unsubscribe(`models/${model.id}/progress`));
    };
  }, [models, subscribe, unsubscribe]);

  // Handle download - EXACTLY as Mock UI lines 1264-1268
  const handleDownload = async () => {
    if (!hfModelRepo) return;
    await downloadModel(hfModelRepo, quantization, accessToken);
    setHfModelRepo('');
    setAccessToken('');
  };

  return (
    <div className="space-y-6">
      {/* Header & Download Form - EXACTLY lines 1214-1276 */}
      <div>
        <h2 className="text-2xl font-semibold mb-4">Model Management</h2>
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
          {/* Model Repo + Quantization (2-column grid) */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                HuggingFace Model Repository
              </label>
              <input
                type="text"
                placeholder="e.g., TinyLlama/TinyLlama-1.1B"
                value={hfModelRepo}
                onChange={(e) => setHfModelRepo(e.target.value)}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              />
            </div>
            <div>
              <label htmlFor="model-quantization" className="block text-sm font-medium text-slate-300 mb-2">
                Quantization Format
              </label>
              <select
                id="model-quantization"
                value={quantization}
                onChange={(e) => setQuantization(e.target.value as any)}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              >
                <option value="FP16">FP16 (Full Precision)</option>
                <option value="Q8">Q8 (8-bit)</option>
                <option value="Q4">Q4 (4-bit)</option>
                <option value="Q2">Q2 (2-bit)</option>
              </select>
            </div>
          </div>

          {/* Access Token */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Access Token <span className="text-slate-500">(optional, for gated models)</span>
            </label>
            <input
              type="password"
              placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
              value={accessToken}
              onChange={(e) => setAccessToken(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 font-mono text-sm"
            />
            <p className="mt-1 text-xs text-slate-500">
              Required for gated models like Llama, Gemma, or other restricted access models
            </p>
          </div>

          {/* Download Button */}
          <button
            type="button"
            onClick={handleDownload}
            disabled={!hfModelRepo}
            className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
          >
            <Download className="w-5 h-5" />
            Download Model from HuggingFace
          </button>
        </div>
      </div>

      {/* Models List - EXACTLY lines 1278-1332 */}
      <div className="grid gap-4">
        {models.map((model) => (
          <ModelCard
            key={model.id}
            model={model}
            onViewArchitecture={() => setSelectedModel(model)}
            onExtractActivations={() => setShowExtractionConfig(true)}
          />
        ))}
      </div>

      {/* Modals - EXACTLY lines 1334-1340 */}
      {selectedModel && (
        <ModelArchitectureViewer
          model={selectedModel}
          onClose={() => setSelectedModel(null)}
        />
      )}

      {showExtractionConfig && (
        <ActivationExtractionConfig
          onClose={() => setShowExtractionConfig(false)}
        />
      )}
    </div>
  );
};
```

**Styling Checklist:**
- ✅ Title: `text-2xl font-semibold mb-4`
- ✅ Form: `bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4`
- ✅ Grid: `grid grid-cols-2 gap-4` for repo + quantization
- ✅ Button: `bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700`
- ✅ Models list: `grid gap-4` (not grid-cols-* - vertical stacking)

---

### ModelCard Component

**File:** `frontend/src/components/models/ModelCard.tsx`

**PRIMARY REFERENCE:** Mock UI lines 1280-1330

```typescript
import React from 'react';
import { Cpu, CheckCircle, Loader, Activity } from 'lucide-react';
import type { Model } from '@/types/model';

interface ModelCardProps {
  model: Model;
  onViewArchitecture: () => void;
  onExtractActivations: () => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  onViewArchitecture,
  onExtractActivations
}) => {
  const isReady = model.status === 'ready';
  const isProcessing = model.status === 'downloading' || model.status === 'quantizing';

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <div className="flex items-center justify-between">
        {/* Left: Icon + Info (clickable for architecture viewer) */}
        <div
          className="flex items-center gap-4 cursor-pointer hover:opacity-80 transition-opacity"
          onClick={onViewArchitecture}
        >
          <Cpu className="w-8 h-8 text-purple-400" />
          <div>
            <h3 className="font-semibold text-lg">{model.name}</h3>
            <p className="text-sm text-slate-400">
              {model.params} params • {model.quantized} quantization • {model.memReq} memory
            </p>
          </div>
        </div>

        {/* Right: Extract Button + Status */}
        <div className="flex items-center gap-3">
          {isReady && (
            <button
              type="button"
              onClick={onExtractActivations}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm font-medium transition-colors"
            >
              Extract Activations
            </button>
          )}
          {model.status === 'ready' && <CheckCircle className="w-5 h-5 text-emerald-400" />}
          {model.status === 'downloading' && <Loader className="w-5 h-5 text-blue-400 animate-spin" />}
          {model.status === 'quantizing' && <Activity className="w-5 h-5 text-yellow-400" />}
          <span className={`px-3 py-1 rounded-full text-sm ${
            model.status === 'ready'
              ? 'bg-emerald-900/30 text-emerald-400'
              : 'bg-slate-800 text-slate-300 capitalize'
          }`}>
            {model.status || 'Edge-Ready'}
          </span>
        </div>
      </div>

      {/* Progress Bar - EXACTLY lines 1316-1329 */}
      {isProcessing && model.progress !== undefined && (
        <div className="mt-4 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">
              {model.status === 'downloading' ? 'Download' : 'Quantization'} Progress
            </span>
            <span className="text-emerald-400 font-medium">{model.progress.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-500"
              style={{ width: `${model.progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
};
```

**Key Styling:**
- Purple theme: `text-purple-400` for Cpu icon, `bg-purple-600` for Extract button, `from-purple-500 to-purple-400` for progress
- Status badge: Green (`bg-emerald-900/30 text-emerald-400`) for ready, gray for others
- Progress bar: `h-2` height with gradient

---

## 4. Database Implementation Approach

### SQLAlchemy Model

**File:** `backend/src/models/model.py`

```python
from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Enum, Boolean
from sqlalchemy.sql import func
from src.core.database import Base
import enum

class ModelStatus(str, enum.Enum):
    DOWNLOADING = "downloading"
    QUANTIZING = "quantizing"
    READY = "ready"
    ERROR = "error"

class Model(Base):
    __tablename__ = "models"

    id = Column(String(255), primary_key=True)  # Format: m_{uuid}
    name = Column(String(500), nullable=False)
    hf_repo_id = Column(String(500), nullable=False)

    # Configuration
    quantization = Column(String(50), nullable=False)  # FP16, Q8, Q4, Q2
    params = Column(String(50), nullable=True)  # "124M", "1.3B", etc.
    num_layers = Column(Integer, nullable=True)
    hidden_dim = Column(Integer, nullable=True)
    num_heads = Column(Integer, nullable=True)
    vocab_size = Column(Integer, nullable=True)
    max_position = Column(Integer, nullable=True)

    # Status
    status = Column(Enum(ModelStatus), nullable=False, default=ModelStatus.DOWNLOADING)
    progress = Column(Float, default=0.0)
    error_message = Column(String(1000), nullable=True)

    # File paths
    model_path = Column(String(1000), nullable=True)  # /data/models/m_{id}/
    config_path = Column(String(1000), nullable=True)

    # Memory
    size_bytes = Column(Integer, nullable=True)
    mem_req_bytes = Column(Integer, nullable=True)  # Estimated GPU VRAM

    # Metadata
    metadata = Column(JSON, nullable=True)  # Architecture details, config, etc.
    is_loaded = Column(Boolean, default=False)  # Currently in GPU memory

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

**Key Fields:**
- `quantization`: FP16, Q8, Q4, Q2 (matches dropdown options exactly)
- `is_loaded`: Track if model currently in GPU memory (for memory management)
- `mem_req_bytes`: Estimated VRAM requirement (important for Jetson Orin Nano's 6GB limit)

---

## 5. API Implementation Strategy

### FastAPI Router

**File:** `backend/src/api/routes/models.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from src.core.database import get_db
from src.services.model_service import ModelService
from src.schemas.model import ModelDownloadRequest, ModelResponse

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("/", response_model=List[ModelResponse])
async def list_models(
    status: str = None,
    db: AsyncSession = Depends(get_db)
):
    """List all models with optional status filter."""
    service = ModelService(db)
    models = await service.list_models(status=status)
    return models

@router.post("/download", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def download_model(
    request: ModelDownloadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Download and quantize model from HuggingFace.
    Enqueues background Celery task.
    """
    service = ModelService(db)

    # Create model record
    model = await service.create_model_from_hf(
        hf_repo_id=request.hf_repo_id,
        quantization=request.quantization,
        access_token=request.access_token
    )

    # Enqueue download + quantization task
    from src.workers.model_tasks import download_and_quantize_model_task
    download_and_quantize_model_task.delay(
        model.id,
        request.hf_repo_id,
        request.quantization,
        request.access_token
    )

    return model

@router.get("/{model_id}/architecture")
async def get_model_architecture(
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get model architecture details (layers, dimensions, config)."""
    service = ModelService(db)
    model = await service.get_model(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Extract architecture from metadata
    architecture = await service.get_architecture_details(model)
    return architecture

@router.post("/{model_id}/extract-activations")
async def extract_activations(
    model_id: str,
    request: dict,  # {dataset_id, layers, activation_type, batch_size, max_samples}
    db: AsyncSession = Depends(get_db)
):
    """
    Extract activations from model on dataset.
    Enqueues background Celery task.
    """
    service = ModelService(db)

    # Verify model exists and is ready
    model = await service.get_model(model_id)
    if not model or model.status != 'ready':
        raise HTTPException(status_code=400, detail="Model not ready")

    # Enqueue extraction task
    from src.workers.activation_tasks import extract_activations_task
    task = extract_activations_task.delay(model_id, request)

    return {"task_id": task.id, "status": "queued"}
```

---

## 6. Business Logic Implementation Hints

### Model Loading with Quantization

**File:** `backend/src/ml/model_loader.py`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple

def load_model_with_quantization(
    hf_repo_id: str,
    quantization: str,
    access_token: str = None
) -> Tuple[torch.nn.Module, any]:
    """
    Load model from HuggingFace with quantization.

    Args:
        hf_repo_id: HuggingFace repository ID (e.g., "TinyLlama/TinyLlama-1.1B")
        quantization: Quantization format ("FP16", "Q8", "Q4", "Q2")
        access_token: HuggingFace access token for gated models

    Returns:
        (model, tokenizer) tuple
    """

    # Configure quantization
    if quantization == "FP16":
        # Full precision (FP16)
        model = AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=access_token
        )
    elif quantization == "Q8":
        # 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            quantization_config=quantization_config,
            device_map="auto",
            token=access_token
        )
    elif quantization == "Q4":
        # 4-bit quantization (recommended for Jetson Orin Nano)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            quantization_config=quantization_config,
            device_map="auto",
            token=access_token
        )
    elif quantization == "Q2":
        # 2-bit quantization (experimental, max compression)
        # Note: Requires custom quantization, bitsandbytes doesn't support 2-bit natively
        raise NotImplementedError("2-bit quantization not yet supported")
    else:
        raise ValueError(f"Invalid quantization: {quantization}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id, token=access_token)

    # Set model to eval mode
    model.eval()

    return model, tokenizer

def get_model_memory_usage(model: torch.nn.Module) -> int:
    """Calculate model memory usage in bytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_size + buffer_size
```

**Quantization Strategy:**
- **FP16:** Full precision, largest memory (use for models <1B params on Jetson)
- **Q8:** 8-bit quantization, ~50% memory reduction, minimal accuracy loss
- **Q4:** 4-bit quantization, ~75% memory reduction, slight accuracy loss (RECOMMENDED for Jetson)
- **Q2:** 2-bit quantization, ~87.5% memory reduction (not yet implemented)

---

### Forward Hooks for Activation Extraction

**File:** `backend/src/ml/forward_hooks.py`

```python
import torch
from typing import Dict, List, Callable

class ActivationExtractor:
    """Extract activations from model layers using forward hooks."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self, layer_indices: List[int], activation_type: str = "residual"):
        """
        Register forward hooks on specified layers.

        Args:
            layer_indices: List of layer indices to extract (e.g., [0, 5, 11])
            activation_type: Type of activation ("residual", "mlp", "attention")
        """
        for layer_idx in layer_indices:
            # Get layer module (GPT-2 architecture)
            layer = self.model.transformer.h[layer_idx]

            # Define hook function
            def create_hook(idx: int):
                def hook(module, input, output):
                    if activation_type == "residual":
                        # Residual stream (layer output)
                        activations = output[0] if isinstance(output, tuple) else output
                    elif activation_type == "mlp":
                        # MLP output
                        activations = module.mlp(module.ln_2(output[0]))
                    elif activation_type == "attention":
                        # Attention output
                        activations = module.attn(module.ln_1(input[0]))[0]
                    else:
                        raise ValueError(f"Invalid activation_type: {activation_type}")

                    # Store activations (detach from computation graph)
                    self.activations[idx] = activations.detach().cpu()

                return hook

            # Register hook
            handle = layer.register_forward_hook(create_hook(layer_idx))
            self.hooks.append(handle)

    def extract_from_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[int, torch.Tensor]:
        """
        Run forward pass and extract activations.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dict mapping layer_idx -> activations tensor [batch_size, seq_len, hidden_dim]
        """
        # Clear previous activations
        self.activations.clear()

        # Forward pass (with hooks active)
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Return extracted activations
        return self.activations.copy()

    def cleanup(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.activations.clear()
```

**Hook Usage Pattern:**
```python
# Initialize extractor
extractor = ActivationExtractor(model)

# Register hooks on layers [0, 5, 11]
extractor.register_hooks(layer_indices=[0, 5, 11], activation_type="residual")

try:
    # Extract activations for batch
    activations = extractor.extract_from_batch(input_ids, attention_mask)

    # Process activations
    for layer_idx, acts in activations.items():
        print(f"Layer {layer_idx}: {acts.shape}")  # [batch_size, seq_len, hidden_dim]
        # Save to database or file

finally:
    # CRITICAL: Always cleanup hooks
    extractor.cleanup()
```

---

## 7. Frontend State Management

### Zustand Store

**File:** `frontend/src/stores/modelsStore.ts`

```typescript
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { Model } from '@/types/model';
import * as api from '@/api/models';

interface ModelsStore {
  models: Model[];
  loading: boolean;
  error: string | null;

  fetchModels: () => Promise<void>;
  getModel: (id: string) => Model | undefined;
  downloadModel: (repo: string, quantization: string, token: string) => Promise<void>;
  updateModelProgress: (id: string, progress: number) => void;
  updateModelStatus: (id: string, status: string) => void;
}

export const useModelsStore = create<ModelsStore>()(
  devtools(
    (set, get) => ({
      models: [],
      loading: false,
      error: null,

      fetchModels: async () => {
        set({ loading: true, error: null });
        try {
          const models = await api.getModels();
          set({ models, loading: false });
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to fetch models',
            loading: false
          });
        }
      },

      getModel: (id: string) => {
        return get().models.find(m => m.id === id);
      },

      downloadModel: async (repo: string, quantization: string, token: string) => {
        try {
          const model = await api.downloadModel(repo, quantization, token);
          set(state => ({
            models: [...state.models, model]
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Download failed'
          });
          throw error;
        }
      },

      updateModelProgress: (id: string, progress: number) => {
        set(state => ({
          models: state.models.map(m =>
            m.id === id ? { ...m, progress } : m
          )
        }));
      },

      updateModelStatus: (id: string, status: string) => {
        set(state => ({
          models: state.models.map(m =>
            m.id === id ? { ...m, status, progress: status === 'ready' ? 100 : m.progress } : m
          )
        }));
      }
    }),
    { name: 'ModelsStore' }
  )
);
```

---

## 8. Testing Implementation Approach

### Unit Test: Model Loading

**File:** `backend/tests/unit/test_model_loader.py`

```python
import pytest
import torch
from src.ml.model_loader import load_model_with_quantization

@pytest.mark.asyncio
async def test_load_fp16_model():
    """Test loading model with FP16 precision."""
    model, tokenizer = load_model_with_quantization(
        hf_repo_id="gpt2",
        quantization="FP16"
    )

    assert model is not None
    assert tokenizer is not None
    assert model.dtype == torch.float16

@pytest.mark.asyncio
async def test_load_q4_model():
    """Test loading model with 4-bit quantization."""
    model, tokenizer = load_model_with_quantization(
        hf_repo_id="gpt2",
        quantization="Q4"
    )

    assert model is not None
    # Check quantization config
    assert hasattr(model, 'quantization_config')
```

### Unit Test: Forward Hooks

**File:** `backend/tests/unit/test_forward_hooks.py`

```python
import pytest
import torch
from src.ml.forward_hooks import ActivationExtractor
from src.ml.model_loader import load_model_with_quantization

@pytest.mark.asyncio
async def test_activation_extraction():
    """Test activation extraction via forward hooks."""
    # Load small model for testing
    model, tokenizer = load_model_with_quantization("gpt2", "FP16")

    # Initialize extractor
    extractor = ActivationExtractor(model)

    try:
        # Register hooks on layers 0, 5, 11
        extractor.register_hooks(layer_indices=[0, 5, 11], activation_type="residual")

        # Prepare input
        text = "The cat sat on the mat"
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Extract activations
        activations = extractor.extract_from_batch(input_ids)

        # Assertions
        assert len(activations) == 3  # 3 layers
        assert 0 in activations
        assert 5 in activations
        assert 11 in activations

        # Check shape [batch_size, seq_len, hidden_dim]
        assert activations[0].shape[0] == 1  # batch_size
        assert activations[0].shape[1] == input_ids.shape[1]  # seq_len
        assert activations[0].shape[2] == 768  # hidden_dim for GPT-2

    finally:
        extractor.cleanup()
```

---

## 9. Performance Implementation Hints

### Memory Management for Jetson Orin Nano

```python
import torch
import gc

def manage_gpu_memory(model: torch.nn.Module) -> dict:
    """
    Monitor and manage GPU memory on Jetson Orin Nano (6GB VRAM).

    Returns memory stats dict.
    """
    if torch.cuda.is_available():
        # Get current memory usage
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB

        stats = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "available_gb": round(max_memory - allocated, 2),
            "max_gb": round(max_memory, 2),
            "utilization_pct": round((allocated / max_memory) * 100, 1)
        }

        # Warn if usage > 90%
        if stats["utilization_pct"] > 90:
            print(f"⚠️  High GPU memory usage: {stats['utilization_pct']}%")
            # Try to free up memory
            torch.cuda.empty_cache()
            gc.collect()

        return stats
    else:
        return {"error": "CUDA not available"}

# Usage before loading model
stats = manage_gpu_memory(None)
if stats["available_gb"] < 2.0:
    raise MemoryError(f"Insufficient GPU memory: {stats['available_gb']}GB available, need at least 2GB")
```

**Memory Optimization Strategies:**
- Unload unused models before loading new ones
- Use `torch.cuda.empty_cache()` after operations
- Monitor memory before large operations (inference, extraction)
- Recommend Q4 quantization for models >1B params

---

## 10. Code Quality and Standards

### TypeScript Type Definitions

**File:** `frontend/src/types/model.ts`

```typescript
export interface Model {
  id: string;
  name: string;
  hf_repo_id: string;
  quantization: 'FP16' | 'Q8' | 'Q4' | 'Q2';
  params: string;  // "124M", "1.3B", etc.
  num_layers?: number;
  hidden_dim?: number;
  num_heads?: number;
  vocab_size?: number;
  max_position?: number;
  status: 'downloading' | 'quantizing' | 'ready' | 'error';
  progress: number;  // 0-100
  error_message?: string;
  model_path?: string;
  size_bytes?: number;
  mem_req_bytes?: number;
  memReq: string;  // Human-readable (e.g., "1.2 GB")
  is_loaded: boolean;
  created_at: Date;
  updated_at?: Date;
}

export interface ModelArchitecture {
  total_layers: number;
  hidden_dimension: number;
  attention_heads: number;
  parameters: string;
  layers: ModelLayer[];
  config: ModelConfig;
}

export interface ModelLayer {
  type: string;
  size?: string;
  attention?: string;
  mlp?: string;
}

export interface ModelConfig {
  vocab_size: number;
  max_position: number;
  mlp_ratio: number;
  architecture: string;
}
```

---

## 10. Extraction Template Management Implementation

This section provides implementation guidance for the extraction template save/load/favorite/export/import functionality (FR-4A from PRD).

### Component: Saved Templates Section

**Location:** Inside `ActivationExtractionConfig` modal (Mock UI lines 1440-1625)

**Implementation Pattern:**

```typescript
// Add to ActivationExtractionConfig component state
const [templates, setTemplates] = useState<ExtractionTemplate[]>([]);
const [showTemplates, setShowTemplates] = useState(false);
const [templateName, setTemplateName] = useState('');
const [templateDescription, setTemplateDescription] = useState('');

// Fetch templates on modal open
useEffect(() => {
  if (isOpen) {
    fetchExtractionTemplates();
  }
}, [isOpen]);

const fetchExtractionTemplates = async () => {
  try {
    const response = await fetch('/api/templates/extraction?limit=50');
    const data = await response.json();
    setTemplates(data.templates);
  } catch (error) {
    console.error('Failed to fetch templates:', error);
  }
};

// Auto-generate template name
const generateTemplateName = () => {
  const timestamp = new Date().toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit'
  }).replace(':', '');

  const hookType = hookTypes.join('_');

  if (selectedLayers.length === 1) {
    return `${hookType}_layer${selectedLayers[0]}_${maxSamples || 'all'}samples_${timestamp}`;
  } else {
    const min = Math.min(...selectedLayers);
    const max = Math.max(...selectedLayers);
    return `${hookType}_layers${min}-${max}_${maxSamples || 'all'}samples_${timestamp}`;
  }
};

// Save template handler
const handleSaveTemplate = async () => {
  const name = templateName || generateTemplateName();

  const template = {
    name,
    description: templateDescription,
    layers: selectedLayers,
    hook_types: hookTypes,
    max_samples: maxSamples,
    top_k_examples: 100,
    is_favorite: false
  };

  try {
    const response = await fetch('/api/templates/extraction', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(template)
    });

    if (response.ok) {
      await fetchExtractionTemplates();
      setTemplateName('');
      setTemplateDescription('');
      // Show success toast
    }
  } catch (error) {
    console.error('Failed to save template:', error);
    // Show error toast
  }
};

// Load template handler
const handleLoadTemplate = (template: ExtractionTemplate) => {
  setSelectedLayers(template.layers);
  setHookTypes(template.hook_types);
  setMaxSamples(template.max_samples);
  // Show success toast: "Loaded template: {name}"
};

// Delete template handler
const handleDeleteTemplate = async (templateId: string) => {
  if (!confirm('Delete this template? This cannot be undone.')) {
    return;
  }

  try {
    const response = await fetch(`/api/templates/extraction/${templateId}`, {
      method: 'DELETE'
    });

    if (response.ok) {
      await fetchExtractionTemplates();
      // Show success toast
    }
  } catch (error) {
    console.error('Failed to delete template:', error);
    // Show error toast
  }
};

// Toggle favorite handler
const handleToggleFavorite = async (templateId: string, isFavorite: boolean) => {
  try {
    const response = await fetch(`/api/templates/extraction/${templateId}/favorite`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ is_favorite: !isFavorite })
    });

    if (response.ok) {
      await fetchExtractionTemplates();
    }
  } catch (error) {
    console.error('Failed to toggle favorite:', error);
  }
};
```

**UI Structure:**

```typescript
// Add collapsible section in ActivationExtractionConfig modal
<div className="mb-4">
  <button
    onClick={() => setShowTemplates(!showTemplates)}
    className="flex items-center justify-between w-full px-3 py-2 text-sm font-medium text-gray-300 hover:text-white bg-gray-800 rounded"
  >
    <span>Saved Templates ({templates.length})</span>
    <ChevronDown className={`w-4 h-4 transition-transform ${showTemplates ? 'rotate-180' : ''}`} />
  </button>

  {showTemplates && (
    <div className="mt-2 space-y-2">
      {/* Save New Template Form */}
      <div className="p-3 bg-gray-800 rounded">
        <input
          type="text"
          placeholder="Template name (auto-generated)"
          value={templateName}
          onChange={(e) => setTemplateName(e.target.value)}
          className="w-full px-2 py-1 mb-2 text-sm bg-gray-700 border border-gray-600 rounded"
        />
        <textarea
          placeholder="Description (optional)"
          value={templateDescription}
          onChange={(e) => setTemplateDescription(e.target.value)}
          maxLength={500}
          rows={2}
          className="w-full px-2 py-1 mb-2 text-sm bg-gray-700 border border-gray-600 rounded resize-none"
        />
        <button
          onClick={handleSaveTemplate}
          disabled={selectedLayers.length === 0 || hookTypes.length === 0}
          className="w-full px-3 py-1 text-sm font-medium text-white bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Save className="inline w-4 h-4 mr-1" />
          Save Current Configuration
        </button>
      </div>

      {/* Template List */}
      <div className="space-y-1 max-h-64 overflow-y-auto">
        {templates.map((template) => (
          <div
            key={template.id}
            className="flex items-center justify-between p-2 bg-gray-800 rounded hover:bg-gray-750"
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleToggleFavorite(template.id, template.is_favorite)}
                  className="text-gray-400 hover:text-yellow-400"
                >
                  <Star className={`w-4 h-4 ${template.is_favorite ? 'fill-yellow-400 text-yellow-400' : ''}`} />
                </button>
                <span className="text-sm font-medium text-white truncate">{template.name}</span>
              </div>
              {template.description && (
                <p className="mt-1 text-xs text-gray-400 truncate">{template.description}</p>
              )}
              <div className="flex gap-2 mt-1 text-xs text-gray-500">
                <span>{template.layers.length} layers</span>
                <span>•</span>
                <span>{template.hook_types.join(', ')}</span>
                {template.max_samples && (
                  <>
                    <span>•</span>
                    <span>{template.max_samples} samples</span>
                  </>
                )}
              </div>
            </div>
            <div className="flex gap-1 ml-2">
              <button
                onClick={() => handleLoadTemplate(template)}
                className="p-1 text-gray-400 hover:text-blue-400"
                title="Load template"
              >
                <Upload className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleDeleteTemplate(template.id)}
                className="p-1 text-gray-400 hover:text-red-400"
                title="Delete template"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        ))}

        {templates.length === 0 && (
          <div className="py-8 text-center text-gray-500">
            <p className="text-sm">No saved templates</p>
            <p className="text-xs">Configure extraction settings and save as a template</p>
          </div>
        )}
      </div>

      {/* Export/Import Buttons */}
      <div className="flex gap-2">
        <button
          onClick={handleExportTemplates}
          className="flex-1 px-3 py-1 text-sm font-medium text-gray-300 bg-gray-700 rounded hover:bg-gray-600"
        >
          <Download className="inline w-4 h-4 mr-1" />
          Export All
        </button>
        <button
          onClick={() => document.getElementById('import-templates-input')?.click()}
          className="flex-1 px-3 py-1 text-sm font-medium text-gray-300 bg-gray-700 rounded hover:bg-gray-600"
        >
          <Upload className="inline w-4 h-4 mr-1" />
          Import
        </button>
        <input
          id="import-templates-input"
          type="file"
          accept=".json"
          onChange={handleImportTemplates}
          className="hidden"
        />
      </div>
    </div>
  )}
</div>
```

### Backend: Extraction Template Endpoints

**File:** `backend/src/api/routes/templates.py`

**Implementation Pattern:**

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json

from ..database import get_db
from ..models.extraction_template import ExtractionTemplate
from ..schemas.extraction_template import (
    ExtractionTemplateCreate,
    ExtractionTemplateUpdate,
    ExtractionTemplateResponse,
    ExtractionTemplatesListResponse,
    ToggleFavoriteRequest
)

router = APIRouter(prefix="/api/templates", tags=["templates"])

# GET /api/templates/extraction
@router.get("/extraction", response_model=ExtractionTemplatesListResponse)
async def list_extraction_templates(
    is_favorite: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List extraction templates with optional filtering."""
    query = db.query(ExtractionTemplate)

    if is_favorite is not None:
        query = query.filter(ExtractionTemplate.is_favorite == is_favorite)

    total = query.count()
    templates = query.order_by(ExtractionTemplate.updated_at.desc()).offset(offset).limit(limit).all()

    return {
        "templates": templates,
        "total": total,
        "limit": limit,
        "offset": offset
    }

# POST /api/templates/extraction
@router.post("/extraction", response_model=ExtractionTemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_extraction_template(
    template: ExtractionTemplateCreate,
    db: Session = Depends(get_db)
):
    """Create new extraction template."""
    # Check for name conflicts
    existing = db.query(ExtractionTemplate).filter(ExtractionTemplate.name == template.name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Template with name '{template.name}' already exists")

    # Validate layers and hook_types not empty
    if not template.layers or len(template.layers) == 0:
        raise HTTPException(status_code=400, detail="layers array cannot be empty")
    if not template.hook_types or len(template.hook_types) == 0:
        raise HTTPException(status_code=400, detail="hook_types array cannot be empty")

    # Create template
    db_template = ExtractionTemplate(**template.dict())
    db.add(db_template)
    db.commit()
    db.refresh(db_template)

    return db_template

# PUT /api/templates/extraction/:id
@router.put("/extraction/{template_id}", response_model=ExtractionTemplateResponse)
async def update_extraction_template(
    template_id: str,
    template: ExtractionTemplateUpdate,
    db: Session = Depends(get_db)
):
    """Update existing extraction template."""
    db_template = db.query(ExtractionTemplate).filter(ExtractionTemplate.id == template_id).first()
    if not db_template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Check for name conflicts (if name is being updated)
    if template.name and template.name != db_template.name:
        existing = db.query(ExtractionTemplate).filter(ExtractionTemplate.name == template.name).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Template with name '{template.name}' already exists")

    # Update fields
    for key, value in template.dict(exclude_unset=True).items():
        setattr(db_template, key, value)

    db.commit()
    db.refresh(db_template)

    return db_template

# DELETE /api/templates/extraction/:id
@router.delete("/extraction/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_extraction_template(
    template_id: str,
    db: Session = Depends(get_db)
):
    """Delete extraction template."""
    db_template = db.query(ExtractionTemplate).filter(ExtractionTemplate.id == template_id).first()
    if not db_template:
        raise HTTPException(status_code=404, detail="Template not found")

    db.delete(db_template)
    db.commit()

    return None

# PATCH /api/templates/extraction/:id/favorite
@router.patch("/extraction/{template_id}/favorite")
async def toggle_extraction_template_favorite(
    template_id: str,
    request: ToggleFavoriteRequest,
    db: Session = Depends(get_db)
):
    """Toggle favorite status for extraction template."""
    db_template = db.query(ExtractionTemplate).filter(ExtractionTemplate.id == template_id).first()
    if not db_template:
        raise HTTPException(status_code=404, detail="Template not found")

    db_template.is_favorite = request.is_favorite
    db.commit()
    db.refresh(db_template)

    return {
        "id": db_template.id,
        "is_favorite": db_template.is_favorite,
        "updated_at": db_template.updated_at
    }
```

### Export/Import Implementation

**Export Handler:**

```typescript
const handleExportTemplates = async () => {
  try {
    const response = await fetch('/api/templates/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        include_training: true,
        include_extraction: true,
        include_steering: true
      })
    });

    if (response.ok) {
      const data = await response.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `miStudio-templates-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }
  } catch (error) {
    console.error('Failed to export templates:', error);
  }
};
```

**Import Handler:**

```typescript
const handleImportTemplates = async (event: React.ChangeEvent<HTMLInputElement>) => {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    const text = await file.text();
    const data = JSON.parse(text);

    const response = await fetch('/api/templates/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (response.ok) {
      const result = await response.json();
      await fetchExtractionTemplates();

      // Show summary toast
      const total = result.imported.extraction_templates;
      const skipped = result.skipped.extraction_templates;
      alert(`Import complete: ${total} templates imported, ${skipped} skipped`);

      if (result.errors.length > 0) {
        console.warn('Import errors:', result.errors);
      }
    }
  } catch (error) {
    console.error('Failed to import templates:', error);
    alert('Import failed. Please check the file format.');
  }

  // Reset file input
  event.target.value = '';
};
```

### Database Migration

**File:** `backend/alembic/versions/XXXX_add_extraction_templates.py`

```python
"""add extraction_templates table

Revision ID: XXXX
Revises: YYYY
Create Date: 2025-10-07

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'XXXX'
down_revision = 'YYYY'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'extraction_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('layers', postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column('hook_types', postgresql.ARRAY(sa.String(50)), nullable=False),
        sa.Column('max_samples', sa.Integer(), nullable=True),
        sa.Column('top_k_examples', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('is_favorite', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.CheckConstraint("LENGTH(name) > 0", name='extraction_templates_name_not_empty'),
        sa.CheckConstraint("array_length(layers, 1) > 0", name='extraction_templates_layers_not_empty'),
        sa.CheckConstraint("array_length(hook_types, 1) > 0", name='extraction_templates_hooks_not_empty')
    )

    op.create_index('idx_extraction_templates_favorite', 'extraction_templates', ['is_favorite'])
    op.create_index('idx_extraction_templates_updated_at', 'extraction_templates', ['updated_at'], postgresql_ops={'updated_at': 'DESC'})
    op.create_index('idx_extraction_templates_name', 'extraction_templates', ['name'])

    # Create trigger function for auto-updating updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_extraction_template_timestamp()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER trigger_extraction_template_updated_at
        BEFORE UPDATE ON extraction_templates
        FOR EACH ROW
        EXECUTE FUNCTION update_extraction_template_timestamp();
    """)

def downgrade():
    op.execute("DROP TRIGGER IF EXISTS trigger_extraction_template_updated_at ON extraction_templates")
    op.execute("DROP FUNCTION IF EXISTS update_extraction_template_timestamp")
    op.drop_index('idx_extraction_templates_name')
    op.drop_index('idx_extraction_templates_updated_at')
    op.drop_index('idx_extraction_templates_favorite')
    op.drop_table('extraction_templates')
```

### TypeScript Interfaces

**File:** `frontend/src/types/extractionTemplate.ts`

```typescript
export interface ExtractionTemplate {
  id: string;
  name: string;
  description?: string;
  layers: number[];
  hook_types: string[];
  max_samples?: number;
  top_k_examples: number;
  is_favorite: boolean;
  created_at: string;
  updated_at: string;
}

export interface ExtractionTemplateCreate {
  name: string;
  description?: string;
  layers: number[];
  hook_types: string[];
  max_samples?: number;
  top_k_examples?: number;
  is_favorite?: boolean;
}

export interface ExtractionTemplatesListResponse {
  templates: ExtractionTemplate[];
  total: number;
  limit: number;
  offset: number;
}
```

### Implementation Checklist

- [ ] Add extraction_templates table migration
- [ ] Implement all 7 API endpoints with validation
- [ ] Add "Saved Templates" collapsible section to ActivationExtractionConfig modal
- [ ] Implement auto-generated template naming
- [ ] Implement save/load/delete/favorite functionality
- [ ] Implement export/import with file download/upload
- [ ] Add TypeScript interfaces for extraction templates
- [ ] Add unit tests for API endpoints
- [ ] Add integration tests for template workflow
- [ ] Test name conflict handling
- [ ] Test import validation and error handling

---

**Document End**
**Status:** Ready for Task Generation
**Total Sections:** 11 (focused implementation guide with extraction templates)
**Estimated Size:** ~45KB
**Next:** 003_FTID|SAE_Training.md
