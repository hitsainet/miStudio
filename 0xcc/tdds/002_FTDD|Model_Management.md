# Technical Design Document: Model Management

**Document ID:** 002_FTDD|Model_Management
**Version:** 1.0
**Last Updated:** 2025-12-05
**Status:** Implemented
**Related PRD:** [002_FPRD|Model_Management](../prds/002_FPRD|Model_Management.md)

---

## 1. System Architecture

### 1.1 Component Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                         │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │  ModelsPanel  │  │ModelDownloadForm│  │ModelPreviewModal│
│  └───────┬───────┘  └────────┬────────┘  └───────┬───────┘ │
│          │                   │                    │         │
│  ┌───────┴───────────────────┴────────────────────┴───────┐ │
│  │                   modelsStore (Zustand)                 │ │
│  └────────────────────────────┬────────────────────────────┘ │
└───────────────────────────────┼─────────────────────────────┘
                                │ HTTP + WebSocket
┌───────────────────────────────┼─────────────────────────────┐
│                    Backend (FastAPI)                        │
│  ┌────────────────────────────┴────────────────────────────┐│
│  │               /api/v1/models endpoints                  ││
│  └────────────────────────────┬────────────────────────────┘│
│                               │                             │
│  ┌────────────────────────────┴────────────────────────────┐│
│  │                     ModelService                        ││
│  └────────────────────────────┬────────────────────────────┘│
│                               │                             │
│  ┌────────────────────────────┴────────────────────────────┐│
│  │    Celery Worker: download_model_task                   ││
│  │    ┌──────────────────────────────────────────────┐     ││
│  │    │  HuggingFace Hub ──→ Local Storage           │     ││
│  │    │  AutoModelForCausalLM + bitsandbytes         │     ││
│  │    └──────────────────────────────────────────────┘     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Download Flow
```
User Input: repo_id + quantization
              ↓
POST /models/{id}/download
              ↓
ModelService.start_download()
              ↓
Celery: download_model_task.delay()
              ↓
┌─────────────────────────────────────┐
│ HuggingFace Hub                     │
│  - Validate access (gated check)    │
│  - Get model config                 │
│  - Download shards                  │
└─────────────────────────────────────┘
              ↓
WebSocket progress: model/{id}/download_progress
              ↓
┌─────────────────────────────────────┐
│ If quantization requested:          │
│  - Load with bitsandbytes           │
│  - Save quantized weights           │
└─────────────────────────────────────┘
              ↓
Update model.status = 'ready'
WebSocket: download_completed
```

---

## 2. Database Schema

### 2.1 Model Entity
```sql
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    repo_id VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    architecture JSONB,  -- {hidden_dim, num_layers, vocab_size, ...}
    quantization VARCHAR(50),  -- none, bnb-4bit, bnb-8bit
    size_bytes BIGINT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT models_status_check CHECK (
        status IN ('pending', 'downloading', 'ready', 'failed')
    )
);

CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_repo_id ON models(repo_id);
```

### 2.2 Architecture JSONB Structure
```json
{
  "model_type": "gemma2",
  "hidden_size": 2304,
  "intermediate_size": 9216,
  "num_hidden_layers": 26,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "vocab_size": 256000,
  "max_position_embeddings": 8192,
  "torch_dtype": "bfloat16",
  "hook_points": [
    "model.layers.0.hook_resid_post",
    "model.layers.12.hook_resid_post",
    ...
  ]
}
```

---

## 3. Quantization Design

### 3.1 Supported Modes
| Mode | Library | Config | Memory Reduction |
|------|---------|--------|------------------|
| `none` | - | Full precision (fp16/bf16) | 0% |
| `bnb-4bit` | bitsandbytes | NF4, double quant | ~75% |
| `bnb-8bit` | bitsandbytes | INT8 | ~50% |

### 3.2 Implementation
```python
def load_model_with_quantization(repo_id: str, quantization: str):
    if quantization == 'bnb-4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization == 'bnb-8bit':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return model
```

---

## 4. API Design

### 4.1 Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/models` | List all models |
| POST | `/models` | Create model record |
| GET | `/models/{id}` | Get model details |
| DELETE | `/models/{id}` | Delete model + files |
| POST | `/models/{id}/download` | Start download |
| GET | `/models/{id}/architecture` | Get layer info |
| POST | `/models/preview` | Preview before download |

### 4.2 Download Request
```python
class ModelDownloadRequest(BaseModel):
    quantization: Literal['none', 'bnb-4bit', 'bnb-8bit'] = 'none'
    hf_token: Optional[str] = None  # For gated models
```

### 4.3 Preview Response
```python
class ModelPreview(BaseModel):
    repo_id: str
    model_type: str
    parameters: str  # "2.6B"
    hidden_size: int
    num_layers: int
    estimated_size_gb: float
    estimated_quantized_size_gb: Optional[float]
    requires_auth: bool
    license: Optional[str]
```

---

## 5. Service Layer

### 5.1 ModelService
```python
class ModelService:
    async def preview_model(self, repo_id: str, hf_token: str = None) -> ModelPreview:
        """Fetch model info from HuggingFace without downloading."""
        config = AutoConfig.from_pretrained(repo_id, token=hf_token)
        return ModelPreview(
            repo_id=repo_id,
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            ...
        )

    async def start_download(self, model_id: UUID, config: DownloadConfig) -> str:
        """Queue download task."""
        task = download_model_task.delay(str(model_id), config.dict())
        return task.id

    async def get_architecture(self, model_id: UUID) -> ModelArchitecture:
        """Return hook points and layer structure."""
        model = await self.get_model(model_id)
        return self._parse_architecture(model.architecture)
```

---

## 6. Celery Task

### 6.1 Download Task
```python
@celery_app.task(bind=True, queue='processing', max_retries=2)
def download_model_task(self, model_id: str, config: dict):
    """
    Download and optionally quantize model.

    Steps:
    1. Validate HuggingFace access
    2. Download model shards with progress
    3. Apply quantization if requested
    4. Extract architecture metadata
    5. Update database record
    """
    model_record = get_model(model_id)
    model_record.status = 'downloading'
    save_model(model_record)

    try:
        # Progress callback for WebSocket
        def progress_callback(progress, downloaded, total):
            emit_model_progress(model_id, progress, downloaded, total)

        # Download with progress tracking
        model_path = download_with_progress(
            repo_id=config['repo_id'],
            token=config.get('hf_token'),
            callback=progress_callback
        )

        # Quantize if needed
        if config['quantization'] != 'none':
            model = load_and_quantize(model_path, config['quantization'])
            save_quantized(model, model_path)

        # Extract architecture
        architecture = extract_architecture(model_path)

        # Update record
        model_record.status = 'ready'
        model_record.file_path = str(model_path)
        model_record.architecture = architecture
        save_model(model_record)

        emit_model_completed(model_id)

    except Exception as e:
        model_record.status = 'failed'
        model_record.error_message = str(e)
        save_model(model_record)
        emit_model_failed(model_id, str(e))
        raise
```

---

## 7. WebSocket Events

### 7.1 Channel: `model/{model_id}`

| Event | Payload | Description |
|-------|---------|-------------|
| `download_progress` | `{progress, downloaded_gb, total_gb, speed_mbps}` | During download |
| `download_completed` | `{file_path, size_bytes}` | Download success |
| `download_failed` | `{error}` | Download error |

---

## 8. File Storage

```
DATA_DIR/
└── models/
    └── {repo_id_normalized}/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── model.safetensors          # Or sharded
        ├── model-00001-of-00002.safetensors
        └── quantized/                  # If quantized
            └── model_4bit.safetensors
```

---

## 9. Frontend State

### 9.1 modelsStore
```typescript
interface ModelsState {
  models: Model[];
  downloadProgress: Record<string, DownloadProgress>;
  isLoading: boolean;

  fetchModels: () => Promise<void>;
  previewModel: (repoId: string, token?: string) => Promise<ModelPreview>;
  startDownload: (id: string, config: DownloadConfig) => Promise<void>;
  deleteModel: (id: string) => Promise<void>;
  updateProgress: (id: string, progress: DownloadProgress) => void;
}
```

---

## 10. Error Handling

| Error | Cause | Handling |
|-------|-------|----------|
| 401 Unauthorized | Gated model, no token | Prompt for HF token |
| 404 Not Found | Invalid repo_id | Validate in preview |
| Disk Full | Insufficient space | Check before download |
| CUDA OOM | Quantization failure | Reduce batch, retry |

---

*Related: [PRD](../prds/002_FPRD|Model_Management.md) | [TID](../tids/002_FTID|Model_Management.md) | [FTASKS](../tasks/002_FTASKS|Model_Management.md)*
