# Feature PRD: SAE Management

**Document ID:** 005_FPRD|SAE_Management
**Version:** 1.0 (MVP Complete)
**Last Updated:** 2025-12-05
**Status:** Implemented
**Priority:** P0 (Core Feature)

---

## 1. Overview

### 1.1 Purpose
Enable users to manage both internally trained and externally sourced SAEs, including downloads from HuggingFace and Gemma Scope.

### 1.2 User Problem
Researchers need to work with SAEs from multiple sources:
- Internally trained SAEs from miStudio
- Pre-trained SAEs from HuggingFace (Gemma Scope, community)
- Different SAE formats requiring conversion

### 1.3 Solution
A unified SAE management system supporting multiple sources, automatic format detection, and SAELens-compatible storage.

---

## 2. Functional Requirements

### 2.1 SAE Sources
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | List SAEs trained in miStudio | Implemented |
| FR-1.2 | Download SAEs from HuggingFace | Implemented |
| FR-1.3 | Download Gemma Scope SAEs | Implemented |
| FR-1.4 | Upload local SAE files | Partial |

### 2.2 Format Support
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | SAELens Community Standard format | Implemented |
| FR-2.2 | miStudio native format | Implemented |
| FR-2.3 | Automatic format detection | Implemented |
| FR-2.4 | Format conversion | Implemented |

### 2.3 SAE Browser
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | List all SAEs with source indicator | Implemented |
| FR-3.2 | Filter by model, layer, source | Implemented |
| FR-3.3 | Search SAEs | Implemented |
| FR-3.4 | View SAE configuration | Implemented |

### 2.4 Management
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | Delete SAEs | Implemented |
| FR-4.2 | Link SAE to training record | Implemented |
| FR-4.3 | View SAE statistics | Implemented |

---

## 3. SAE Format Specifications

### 3.1 Community Standard (SAELens)
```
sae_directory/
├── cfg.json                    # Configuration
└── sae_weights.safetensors     # Weights
```

**cfg.json Fields:**
```json
{
  "d_in": 2304,
  "d_sae": 18432,
  "dtype": "float32",
  "model_name": "google/gemma-2-2b",
  "hook_name": "blocks.12.hook_resid_post",
  "architecture": "standard"
}
```

### 3.2 miStudio Native
```
sae_directory/
├── config.json                 # Extended configuration
├── model.safetensors           # Weights
└── training_metadata.json      # Training history (optional)
```

---

## 4. User Interface

### 4.1 SAEs Panel
```
┌─────────────────────────────────────────────────────────────┐
│ SAEs                                    [+ Download from HF] │
├─────────────────────────────────────────────────────────────┤
│ Filter: [All Sources ▾] [All Models ▾]  Search: [________]  │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [Trained] gemma-2b-layer12-8x                          │ │
│ │ Model: google/gemma-2-2b | Layer: 12                   │ │
│ │ Dimensions: 2304 → 18432 | Architecture: Standard      │ │
│ │ Features: 993 extracted                                │ │
│ │ [Extract] [Steering] [Export] [Delete]                 │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [HuggingFace] gemma-scope-2b-pt-res                    │ │
│ │ Model: google/gemma-2-2b | Layer: 12                   │ │
│ │ Dimensions: 2304 → 16384 | Architecture: JumpReLU     │ │
│ │ Source: google/gemma-scope-2b-pt-res                   │ │
│ │ [Extract] [Steering] [Delete]                          │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Download from HF Modal
- Repository ID input
- Model layer selection (for Gemma Scope)
- SAE configuration preview
- Memory estimation
- Download progress

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/saes` | GET | List all SAEs |
| `/api/v1/saes` | POST | Create SAE record |
| `/api/v1/saes/{id}` | GET | Get SAE details |
| `/api/v1/saes/{id}` | DELETE | Delete SAE |
| `/api/v1/saes/download-hf` | POST | Download from HuggingFace |
| `/api/v1/saes/{id}/convert` | POST | Convert format |
| `/api/v1/saes/{id}/config` | GET | Get SAE configuration |

---

## 6. Data Model

### 6.1 ExternalSAE Table
```sql
CREATE TABLE external_saes (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    layer INTEGER NOT NULL,
    hook_name VARCHAR(255),
    d_in INTEGER NOT NULL,
    d_sae INTEGER NOT NULL,
    architecture VARCHAR(50),  -- standard, jumprelu, skip, transcoder
    source VARCHAR(50),  -- huggingface, gemma_scope, upload
    repo_id VARCHAR(255),
    local_path VARCHAR(500),
    format VARCHAR(50),  -- community, mistudio
    metadata JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## 7. Key Files

### Backend
- `backend/src/services/sae_manager_service.py` - SAE management logic
- `backend/src/services/huggingface_sae_service.py` - HF downloads
- `backend/src/ml/sae_converter.py` - Format conversion
- `backend/src/ml/community_format.py` - SAELens format handling
- `backend/src/api/v1/endpoints/saes.py` - API routes
- `backend/src/models/external_sae.py` - SQLAlchemy model

### Frontend
- `frontend/src/components/panels/SAEsPanel.tsx` - Main panel
- `frontend/src/components/saes/SAECard.tsx` - Card component
- `frontend/src/components/saes/DownloadFromHF.tsx` - Download modal
- `frontend/src/stores/saesStore.ts` - Zustand store

---

## 8. Gemma Scope Support

### 8.1 Repository Structure
```
google/gemma-scope-2b-pt-res/
├── layer_0/
│   ├── width_16k/
│   │   ├── average_l0_82/
│   │   │   ├── cfg.json
│   │   │   └── sae_weights.safetensors
│   │   └── ...
│   └── ...
├── layer_12/
└── ...
```

### 8.2 Download Flow
1. User selects Gemma Scope repo
2. System parses layer/width/l0 options
3. User selects specific SAE variant
4. Download and store in community format

---

## 9. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| SAE Training | Creates SAE records |
| Feature Discovery | Provides SAE for extraction |
| Model Steering | Provides SAE for steering |
| Neuronpedia Export | Exports SAE data |

---

## 10. Testing Checklist

- [x] List trained SAEs
- [x] Download from HuggingFace
- [x] Download Gemma Scope SAE
- [x] Format detection
- [x] View SAE configuration
- [x] Delete SAE
- [x] Filter by source/model
- [x] Link to training record

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/005_FTDD|SAE_Management.md) | [TID](../tids/005_FTID|SAE_Management.md)*
