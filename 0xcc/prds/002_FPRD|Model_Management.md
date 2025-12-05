# Feature PRD: Model Management

**Document ID:** 002_FPRD|Model_Management
**Version:** 1.0 (MVP Complete)
**Last Updated:** 2025-12-05
**Status:** Implemented
**Priority:** P0 (Core Feature)

---

## 1. Overview

### 1.1 Purpose
Enable users to download, configure, and manage transformer models from HuggingFace Hub for SAE training and inference.

### 1.2 User Problem
Researchers need access to various transformer models but face challenges with:
- Memory constraints on edge devices
- Managing multiple model versions
- Understanding model architecture for hook placement
- Handling gated models requiring authentication

### 1.3 Solution
A model management system with HuggingFace integration, quantization support, and architecture visualization.

---

## 2. Functional Requirements

### 2.1 Model Download
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | Download models from HuggingFace Hub by repo ID | Implemented |
| FR-1.2 | Support gated models with HF token authentication | Implemented |
| FR-1.3 | Display real-time download progress via WebSocket | Implemented |
| FR-1.4 | Memory estimation before download | Implemented |
| FR-1.5 | Cancel in-progress downloads | Implemented |

### 2.2 Quantization
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | 4-bit quantization via bitsandbytes (bnb-4bit) | Implemented |
| FR-2.2 | 8-bit quantization via bitsandbytes (bnb-8bit) | Implemented |
| FR-2.3 | Full precision (fp16/fp32) option | Implemented |
| FR-2.4 | Display quantized model size | Implemented |

### 2.3 Architecture Viewer
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | Display model architecture (layer types, dimensions) | Implemented |
| FR-3.2 | Show hook points for activation extraction | Implemented |
| FR-3.3 | Display parameter count per layer | Implemented |
| FR-3.4 | Show hidden dimension, vocab size, layers | Implemented |

### 2.4 Management
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | List all downloaded models | Implemented |
| FR-4.2 | Delete models and free disk space | Implemented |
| FR-4.3 | Filter by model family, size, quantization | Implemented |
| FR-4.4 | Display model metadata and capabilities | Implemented |

---

## 3. Non-Functional Requirements

| Requirement | Target | Status |
|-------------|--------|--------|
| NFR-1 | Support models up to 70B parameters | Tested to 8B |
| NFR-2 | Download progress updates every 1s | Achieved |
| NFR-3 | Quantized model load time < 60s | Achieved |
| NFR-4 | Memory estimation accuracy within 10% | Achieved |

---

## 4. User Interface

### 4.1 Models Panel
```
┌─────────────────────────────────────────────────────────────┐
│ Models                                      [+ Download]    │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Model Card                                              │ │
│ │ Name: google/gemma-2-2b                                │ │
│ │ Parameters: 2B | Hidden: 2304 | Layers: 26             │ │
│ │ Quantization: 4-bit | Size: 1.5GB                      │ │
│ │ Status: Ready                                          │ │
│ │ [Architecture] [Delete]                                │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Download Modal
- HuggingFace repo ID input with model preview
- Quantization selection (None, 4-bit, 8-bit)
- HF Token input for gated models
- Memory requirement display
- Start/Cancel buttons

### 4.3 Architecture Modal
- Layer tree view
- Hook point selector
- Parameter count summary
- Model config display

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/models` | GET | List all models |
| `/api/v1/models` | POST | Create model record |
| `/api/v1/models/{id}` | GET | Get model details |
| `/api/v1/models/{id}` | DELETE | Delete model |
| `/api/v1/models/{id}/download` | POST | Start download |
| `/api/v1/models/{id}/architecture` | GET | Get architecture info |
| `/api/v1/models/preview` | POST | Preview model before download |

---

## 6. Data Model

```sql
CREATE TABLE models (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    repo_id VARCHAR(255),
    file_path VARCHAR(500),
    status VARCHAR(50),  -- pending, downloading, ready, failed
    architecture JSONB,  -- layer info, hidden_dim, etc.
    quantization VARCHAR(50),  -- none, bnb-4bit, bnb-8bit
    size_bytes BIGINT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## 7. WebSocket Channels

| Channel | Events | Payload |
|---------|--------|---------|
| `model/{id}` | `download_progress` | `{progress: float, downloaded_gb: float}` |
| `model/{id}` | `download_completed` | `{file_path: string}` |
| `model/{id}` | `download_failed` | `{error: string}` |

---

## 8. Key Files

### Backend
- `backend/src/services/model_service.py` - Core business logic
- `backend/src/workers/model_tasks.py` - Celery download tasks
- `backend/src/api/v1/endpoints/models.py` - API routes
- `backend/src/models/model.py` - SQLAlchemy model
- `backend/src/schemas/model.py` - Pydantic schemas

### Frontend
- `frontend/src/components/panels/ModelsPanel.tsx` - Main panel
- `frontend/src/components/models/ModelCard.tsx` - Card component
- `frontend/src/components/models/ModelDownloadForm.tsx` - Download modal
- `frontend/src/components/models/ModelPreviewModal.tsx` - Preview/architecture
- `frontend/src/stores/modelsStore.ts` - Zustand store

---

## 9. Supported Model Families

| Family | Example Models | Tested |
|--------|----------------|--------|
| Gemma | gemma-2-2b, gemma-2-9b | Yes |
| LLaMA | llama-2-7b, llama-3-8b | Yes |
| Pythia | pythia-70m, pythia-410m | Yes |
| GPT-2 | gpt2, gpt2-medium | Yes |
| Phi | phi-2, phi-3-mini | Yes |
| Qwen | qwen-1.5-0.5b | Yes |

---

## 10. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| Dataset Management | Uses model tokenizers |
| SAE Training | Provides base model for training |
| Model Steering | Provides model for inference |
| Feature Discovery | Provides hook points |

---

## 11. Testing Checklist

- [x] Download model from HuggingFace
- [x] Download gated model with token
- [x] 4-bit quantization
- [x] 8-bit quantization
- [x] Cancel in-progress download
- [x] View architecture modal
- [x] Delete model
- [x] Memory estimation accuracy

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/002_FTDD|Model_Management.md) | [TID](../tids/002_FTID|Model_Management.md)*
