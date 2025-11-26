# Task List: SAEs Tab & Steering Interface

**Feature:** SAE Management + Feature-Based Model Steering
**PRD Reference:** 0xcc/project-specs/steering/SAE_Steering_FPRD.md
**Mock UI Reference:** 0xcc/project-specs/steering/SAE_Steering_Final_v4.jsx
**Screenshots:** SAEs-2025.11.25.jpg, Steering-Top-2025.11.25.jpg, Steering-Bot-2025.11.25.jpg
**Previous Tasks:** 005_FTASKS|Model_Steering.md (superseded), 006_FTASKS|SAE_Sharing_Integration.md (merged)
**Created:** 2025-11-26
**Status:** Ready for Implementation

---

## Overview

This task list implements two new navigation tabs:
1. **SAEs Tab** - Download from HuggingFace, upload trained SAEs, manage local SAEs
2. **Steering Tab** - Feature-based text generation with comparative analysis

---

## Architecture Summary

### SAEs Tab Flow
```
User → Downloads SAE from HuggingFace → Converts to miStudio format → Available in "Your SAEs"
User → Trains SAE locally → Appears in "Your SAEs" → Can upload to HuggingFace
User → Clicks "Use in Steering →" → Navigates to Steering tab with SAE pre-selected
```

### Steering Tab Flow
```
User → Selects Active SAE → Searches/selects features (up to 4)
User → Configures strength (-100 to +300) and layer per feature
User → Enters prompt and generation parameters
User → Clicks "Generate Comparison" → Gets unsteered + steered outputs
User → Reviews metrics (perplexity, coherence, behavioral score)
```

---

## Relevant Files

### Backend - New Files
- `backend/src/services/sae_manager_service.py` - SAE download, upload, conversion, listing
- `backend/src/services/steering_service.py` - Forward hooks, generation, metrics
- `backend/src/services/huggingface_service.py` - HF Hub API client
- `backend/src/services/sae_converter.py` - SAELens ↔ miStudio format conversion
- `backend/src/models/external_sae.py` - Imported SAE tracking model
- `backend/src/schemas/sae_management.py` - Schemas for SAE operations
- `backend/src/schemas/steering.py` - Schemas for steering operations
- `backend/src/api/v1/endpoints/saes.py` - SAE management endpoints
- `backend/src/api/v1/endpoints/steering.py` - Steering endpoints
- `backend/src/workers/sae_tasks.py` - Celery tasks for async SAE operations
- `backend/alembic/versions/xxx_create_external_saes_table.py` - Migration

### Backend - Files to Modify
- `backend/src/core/config.py` - Add HF_TOKEN, steering configs
- `backend/requirements.txt` - Add huggingface_hub, sentence-transformers

### Frontend - New Files
- `frontend/src/components/panels/SAEsPanel.tsx` - SAEs tab main component
- `frontend/src/components/panels/SteeringPanel.tsx` - Steering tab main component
- `frontend/src/components/saes/DownloadFromHF.tsx` - Download form
- `frontend/src/components/saes/UploadToHF.tsx` - Upload form
- `frontend/src/components/saes/SAECard.tsx` - SAE list item
- `frontend/src/components/steering/FeatureSelector.tsx` - Left sidebar feature management
- `frontend/src/components/steering/StrengthSlider.tsx` - Strength control with warning zones
- `frontend/src/components/steering/GenerationConfig.tsx` - Right side config panel
- `frontend/src/components/steering/ComparisonResults.tsx` - Output display
- `frontend/src/stores/saesStore.ts` - SAE state management
- `frontend/src/stores/steeringStore.ts` - Steering state management
- `frontend/src/types/sae.ts` - SAE TypeScript types
- `frontend/src/types/steering.ts` - Steering TypeScript types
- `frontend/src/api/saes.ts` - SAE API client
- `frontend/src/api/steering.ts` - Steering API client

### Frontend - Files to Modify
- `frontend/src/App.tsx` - Add SAEs and Steering tabs to navigation
- `frontend/src/components/layout/Navigation.tsx` - Add tab links

---

## Phase 1: Database & Backend Foundation

### 1.0 Database Schema for External SAEs
- [x] 1.1 Create migration `xxx_create_external_saes_table.py`
- [x] 1.2 Define `external_saes` table:
  ```sql
  id UUID PRIMARY KEY
  source_type VARCHAR(50)  -- 'huggingface', 'local_upload'
  repository_id VARCHAR(255)  -- e.g., 'google/gemma-scope-2b-pt-res'
  sae_id VARCHAR(255)  -- e.g., 'layer_12/width_16k/canonical'
  local_path VARCHAR(500)  -- Path to converted checkpoint
  model_name VARCHAR(255)  -- Target model (e.g., 'gemma-2b')
  layer INTEGER
  latent_dim INTEGER  -- Number of features
  hidden_dim INTEGER
  architecture VARCHAR(50)  -- 'standard', 'gated', 'jumprelu'
  metadata JSONB  -- Additional config from source
  status VARCHAR(50)  -- 'downloading', 'converting', 'ready', 'failed'
  error_message TEXT
  created_at TIMESTAMP
  updated_at TIMESTAMP
  ```
- [x] 1.3 Add indexes: `idx_external_saes_repository`, `idx_external_saes_status`
- [x] 1.4 Create SQLAlchemy model in `backend/src/models/external_sae.py`
- [x] 1.5 Run migration and verify table

### 1.1 Configuration Updates
- [x] 1.1.1 Add to `config.py`:
  - `hf_token: Optional[str]` - HuggingFace API token
  - `hf_cache_dir: str` - Cache directory for downloads
  - `steering_timeout_seconds: int = 30`
  - `max_steering_features: int = 4`
- [x] 1.1.2 Add to `requirements.txt`:
  - `huggingface_hub>=0.20.0`
  - `sentence-transformers>=2.2.0`
  - `safetensors>=0.4.0`

---

## Phase 2: HuggingFace Service

### 2.0 HuggingFace Service Class
- [x] 2.1 Create `backend/src/services/huggingface_service.py`
- [x] 2.2 Implement `HuggingFaceService` class with HfApi client
- [x] 2.3 Implement `preview_repository(repo_id: str)` method:
  - Fetch repository metadata (name, description, files)
  - List available SAE paths within repository
  - Return preview info without downloading
- [x] 2.4 Implement `download_sae(repo_id: str, path: str, cache_dir: str)`:
  - Use `hf_hub_download()` for checkpoint files
  - Download cfg.json or config.json
  - Handle authentication for private repos
  - Return local paths to downloaded files
- [x] 2.5 Implement `upload_sae(local_dir: str, repo_id: str, commit_message: str)`:
  - Create repo if not exists via `create_repo()`
  - Upload checkpoint and config via `upload_folder()`
  - Generate README.md model card
  - Return repository URL
- [x] 2.6 Add progress callback support for WebSocket updates
- [x] 2.7 Add error handling: network failures, auth errors, rate limits
- [ ] 2.8 Write unit tests for HuggingFace service

---

## Phase 3: SAELens Format Converter

### 3.0 SAE Converter Service
- [ ] 3.1 Create `backend/src/services/sae_converter.py`
- [ ] 3.2 Study SAELens format:
  - Directory structure: `{hook_name}/`
  - Checkpoint: `sae_weights.safetensors` or similar
  - Config: `cfg.json` with architecture, d_in, d_sae, activation_fn
- [ ] 3.3 Implement `saelens_to_mistudio(source_dir: str, target_dir: str)`:
  - Load SAELens config and weights
  - Map architecture names: `standard` → `TopK`, etc.
  - Convert checkpoint format
  - Generate miStudio-compatible metadata
  - Return converted checkpoint path
- [ ] 3.4 Implement `mistudio_to_saelens(training_id: str, target_dir: str)`:
  - Load miStudio training checkpoint
  - Generate SAELens directory structure
  - Create cfg.json with required fields
  - Generate hook_name from layer info
  - Return export directory path
- [ ] 3.5 Implement `infer_model_from_dimensions(hidden_dim: int)`:
  - Map hidden dimensions to model names
  - 768 → GPT-2 small, 2048 → Gemma-2B, etc.
- [ ] 3.6 Write roundtrip tests: miStudio → SAELens → miStudio

---

## Phase 4: SAE Manager Service

### 4.0 SAE Manager Service Class
- [x] 4.1 Create `backend/src/services/sae_manager_service.py`
- [x] 4.2 Implement `list_local_saes()`:
  - Query external_saes table for imported SAEs
  - Query trainings table for locally trained SAEs (status=completed)
  - Merge and return unified list with source indicator
- [x] 4.3 Implement `import_from_huggingface(repo_id: str, path: str)`:
  - Create external_sae record with status='downloading'
  - Call HuggingFaceService.download_sae()
  - Update status to 'converting'
  - Call SAEConverter.saelens_to_mistudio()
  - Update status to 'ready' with local_path
  - Emit WebSocket progress updates
- [x] 4.4 Implement `export_to_huggingface(training_id: str, repo_id: str)`:
  - Validate training has completed checkpoint
  - Call SAEConverter.mistudio_to_saelens()
  - Call HuggingFaceService.upload_sae()
  - Return repository URL
- [x] 4.5 Implement `get_sae_for_steering(sae_id: str)`:
  - Load SAE checkpoint (external or trained)
  - Return encoder and decoder tensors
  - Cache loaded SAEs for performance
- [x] 4.6 Implement `delete_external_sae(sae_id: str)`:
  - Delete checkpoint files
  - Delete database record
- [ ] 4.7 Write integration tests

---

## Phase 5: Steering Service

### 5.0 Steering Service Foundation
- [ ] 5.1 Create `backend/src/services/steering_service.py`
- [ ] 5.2 Initialize with:
  - `self.hook_handles = []` for cleanup
  - `self.sae_manager = SAEManagerService()`
  - `self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')`
- [ ] 5.3 Implement `_register_steering_hook(model, layer, encoder, decoder, feature_configs)`:
  ```python
  def steering_hook(module, input, output):
      activations = output[0] if isinstance(output, tuple) else output
      feature_acts = torch.relu(encoder(activations))
      for feature_id, strength in feature_configs.items():
          # strength: -100 to +300, convert to multiplier
          multiplier = 1 + (strength / 100)  # -100 → 0x, 0 → 1x, +100 → 2x
          feature_acts[:, :, feature_id] *= multiplier
      steered = decoder(feature_acts)
      return (steered,) if isinstance(output, tuple) else steered
  ```
- [ ] 5.4 Register on target layer: `model.transformer.h[layer].register_forward_hook()`
- [ ] 5.5 Store handle for cleanup

### 5.1 Multi-Layer Hook Support
- [ ] 5.1.1 Support per-feature layer targeting:
  - Each feature can specify its own layer (L0-L21)
  - Register separate hook per unique layer
  - Track which features apply at each layer
- [ ] 5.1.2 Group features by layer for efficient hook registration
- [ ] 5.1.3 Implement `_cleanup_hooks()`:
  - Remove all registered hooks
  - Clear hook handles list
  - Use in try/finally for guaranteed cleanup

### 5.2 Generation Methods
- [ ] 5.2.1 Implement `_generate_unsteered(model, tokenizer, prompt, config)`:
  - Tokenize prompt
  - Generate with return_dict_in_generate=True, output_scores=True
  - Return (text, logprobs)
- [ ] 5.2.2 Implement `_generate_steered(model, tokenizer, sae_encoder, sae_decoder, prompt, features, config)`:
  - Register hooks for each unique layer
  - Generate with same parameters as unsteered
  - Cleanup hooks in finally block
  - Return (text, logprobs)
- [ ] 5.2.3 Implement `generate_comparison(request: SteeringRequest)`:
  - Load model and SAE
  - Generate unsteered output
  - Generate steered output with same seed
  - Calculate metrics
  - Return comparison response

### 5.3 Metrics Calculation
- [ ] 5.3.1 Implement `_calculate_perplexity(logprobs)`:
  - Compute NLL from logprobs
  - Return exp(mean_nll)
- [ ] 5.3.2 Implement `_calculate_coherence(text)`:
  - Use sentence embeddings for sentence-to-sentence similarity
  - Average pairwise similarities
  - Return 0-1 score
- [ ] 5.3.3 Implement `_calculate_behavioral_score(unsteered_text, steered_text, target_features)`:
  - Measure how well steering achieved intended behavior
  - Use feature-specific heuristics or embeddings
  - Return 0-1 score

---

## Phase 6: Pydantic Schemas

### 6.0 SAE Management Schemas
- [x] 6.1 Create `backend/src/schemas/sae_management.py`
- [x] 6.2 Define `SAEPreviewRequest`:
  - `repository_id: str`
  - `path: Optional[str]`
  - `access_token: Optional[str]`
- [x] 6.3 Define `SAEPreviewResponse`:
  - `repository_id: str`
  - `available_saes: List[SAEInfo]`
  - `model_card: Optional[str]`
- [x] 6.4 Define `SAEDownloadRequest`:
  - `repository_id: str`
  - `path: Optional[str]`
  - `access_token: Optional[str]`
- [x] 6.5 Define `SAEUploadRequest`:
  - `training_id: str`  # miStudio training to export
  - `target_repository: str`
  - `access_token: str`
- [x] 6.6 Define `SAEListItem`:
  - `id: str`
  - `name: str`
  - `source: Literal['local', 'huggingface']`
  - `model_name: str`
  - `layer: int`
  - `latent_dim: int`
  - `status: str`
  - `created_at: datetime`

### 6.1 Steering Schemas
- [x] 6.1.1 Create `backend/src/schemas/steering.py`
- [x] 6.1.2 Define `SelectedFeature`:
  - `feature_id: int`
  - `strength: float`  # -100 to +300
  - `layer: int`  # L0-L21
- [x] 6.1.3 Define `SteeringGenerateRequest`:
  - `sae_id: str`
  - `prompt: str`
  - `features: List[SelectedFeature]`  # max 4
  - `max_tokens: int = 100`
  - `temperature: float = 0.7`
  - `top_p: float = 0.9`
  - `samples_per_config: int = 2`
  - `random_seed: Optional[int]`
- [x] 6.1.4 Define `GenerationOutput`:
  - `samples: List[str]`
  - `perplexity: float`
  - `coherence: float`
  - `avg_length: int`
- [x] 6.1.5 Define `SteeredOutput`:
  - `feature_id: int`
  - `feature_name: str`
  - `strength: float`
  - `layer: int`
  - `samples: List[str]`
  - `perplexity: float`
  - `coherence: float`
  - `behavioral_score: float`
  - `avg_length: int`
- [x] 6.1.6 Define `SteeringGenerateResponse`:
  - `unsteered: GenerationOutput`
  - `steered: List[SteeredOutput]`
  - `generation_time_ms: int`

---

## Phase 7: FastAPI Endpoints

### 7.0 SAE Management Endpoints
- [x] 7.1 Create `backend/src/api/v1/endpoints/saes.py`
- [x] 7.2 Implement `POST /api/v1/saes/preview`:
  - Accept SAEPreviewRequest
  - Call HuggingFaceService.preview_repository()
  - Return available SAEs without downloading
- [x] 7.3 Implement `POST /api/v1/saes/download`:
  - Accept SAEDownloadRequest
  - Enqueue Celery task for async download
  - Return 202 Accepted with job_id
- [x] 7.4 Implement `POST /api/v1/saes/upload`:
  - Accept SAEUploadRequest
  - Validate training exists with checkpoint
  - Enqueue Celery task for async upload
  - Return 202 Accepted with job_id
- [x] 7.5 Implement `GET /api/v1/saes`:
  - Return list of all SAEs (local + imported)
  - Include status, source, feature count
- [x] 7.6 Implement `GET /api/v1/saes/{sae_id}`:
  - Return detailed SAE info
- [x] 7.7 Implement `DELETE /api/v1/saes/{sae_id}`:
  - Delete imported SAE (not locally trained)
- [x] 7.8 Register router in main app

### 7.1 Steering Endpoints
- [x] 7.1.1 Create `backend/src/api/v1/endpoints/steering.py`
- [x] 7.1.2 Implement `GET /api/v1/steering/saes/{sae_id}/features`:
  - Return features for selected SAE
  - Support search query parameter
  - Include feature name, ID, activation frequency, top tokens
- [x] 7.1.3 Implement `POST /api/v1/steering/generate`:
  - Accept SteeringGenerateRequest
  - Validate SAE exists and ready
  - Validate features exist for SAE
  - Call SteeringService.generate_comparison()
  - Return SteeringGenerateResponse
- [ ] 7.1.4 Add timeout handling: 408 if exceeds 30 seconds
- [ ] 7.1.5 Add rate limiting: 5 requests per minute
- [x] 7.1.6 Register router in main app

---

## Phase 8: Celery Tasks

### 8.0 SAE Import/Export Tasks
- [ ] 8.1 Create `backend/src/workers/sae_tasks.py`
- [ ] 8.2 Implement `@celery_app.task download_sae_task(repo_id, path, access_token)`:
  - Create external_sae record
  - Call SAEManagerService.import_from_huggingface()
  - Emit WebSocket progress: downloading (0-50%), converting (50-90%), ready (100%)
  - Handle errors and update status
- [ ] 8.3 Implement `@celery_app.task upload_sae_task(training_id, repo_id, access_token)`:
  - Call SAEManagerService.export_to_huggingface()
  - Emit WebSocket progress: converting (0-50%), uploading (50-100%)
  - Return repository URL on success
- [ ] 8.4 Add retry logic with exponential backoff
- [ ] 8.5 Add to Celery autodiscovery

### 8.1 WebSocket Events
- [ ] 8.1.1 Define SAE events:
  - `sae/download/{job_id}` - Download progress
  - `sae/upload/{job_id}` - Upload progress
- [ ] 8.1.2 Emit events from Celery tasks via websocket_emitter

---

## Phase 9: Frontend Types & API Client

### 9.0 TypeScript Types
- [x] 9.1 Create `frontend/src/types/sae.ts`:
  ```typescript
  interface SAE {
    id: string;
    name: string;
    source: 'local' | 'huggingface';
    modelName: string;
    layer: number;
    latentDim: number;
    status: 'ready' | 'downloading' | 'converting' | 'failed';
    createdAt: string;
  }
  ```
- [x] 9.2 Create `frontend/src/types/steering.ts`:
  ```typescript
  interface SelectedFeature {
    featureId: number;
    name: string;
    strength: number;  // -100 to +300
    layer: number;
    color: 'teal' | 'blue' | 'purple' | 'amber';
  }

  interface SteeringConfig {
    saeId: string;
    prompt: string;
    features: SelectedFeature[];
    maxTokens: number;
    temperature: number;
    topP: number;
    samplesPerConfig: number;
  }
  ```

### 9.1 API Clients
- [x] 9.1.1 Create `frontend/src/api/saes.ts`:
  - `previewSAE(repoId, path?)` → SAEPreviewResponse
  - `downloadSAE(repoId, path?, token?)` → { jobId }
  - `uploadSAE(trainingId, repoId, token)` → { jobId }
  - `listSAEs()` → SAE[]
  - `deleteSAE(saeId)` → void
- [x] 9.1.2 Create `frontend/src/api/steering.ts`:
  - `getFeatures(saeId, search?)` → Feature[]
  - `generateComparison(config)` → SteeringResponse

---

## Phase 10: SAEs Tab - Zustand Store

### 10.0 SAEs Store
- [x] 10.1 Create `frontend/src/stores/saesStore.ts`
- [x] 10.2 Define state:
  ```typescript
  interface SAEsState {
    saes: SAE[];
    downloadJobs: Map<string, DownloadJob>;
    uploadJobs: Map<string, UploadJob>;
    loading: boolean;
    error: string | null;
  }
  ```
- [x] 10.3 Implement actions:
  - `fetchSAEs()` - Load all SAEs
  - `previewRepository(repoId, path?)` - Get preview before download
  - `startDownload(repoId, path?, token?)` - Initiate download
  - `startUpload(trainingId, repoId, token)` - Initiate upload
  - `updateDownloadProgress(jobId, progress)` - From WebSocket
  - `updateUploadProgress(jobId, progress)` - From WebSocket
  - `deleteSAE(saeId)` - Remove imported SAE
- [x] 10.4 Add WebSocket subscription for progress events

---

## Phase 11: SAEs Tab - UI Components

### 11.0 SAEs Panel Main Component
- [x] 11.1 Create `frontend/src/components/panels/SAEsPanel.tsx`
- [x] 11.2 Layout: Page header + 2-column grid (Download | Upload) + Your SAEs list
- [x] 11.3 Page header:
  ```jsx
  <div className="mb-6">
    <h1 className="text-2xl font-bold">SAE Management</h1>
    <p className="text-slate-400">Download SAEs from HuggingFace or upload your trained SAEs</p>
  </div>
  ```

### 11.1 Download from HuggingFace Component
- [x] 11.1.1 Create `frontend/src/components/saes/DownloadFromHF.tsx`
- [x] 11.1.2 Card with teal border accent (from mock)
- [x] 11.1.3 Form fields:
  - Repository ID (text input, placeholder: "google/gemma-scope-2b-pt-res")
  - Path (optional text input, placeholder: "layer_12/width_16k/canonical")
  - Access Token (optional password input, placeholder: "hf_...")
  - Help text: "Required for private or gated models"
- [x] 11.1.4 Buttons: "Preview" (slate), "Download" (emerald gradient)
- [x] 11.1.5 Preview modal: show available SAEs in repository
- [x] 11.1.6 Download progress indicator

### 11.2 Upload to HuggingFace Component
- [x] 11.2.1 Create `frontend/src/components/saes/UploadToHF.tsx`
- [x] 11.2.2 Card with matching style
- [x] 11.2.3 Form fields:
  - Select Local SAE (dropdown of completed trainings)
  - Target Repository (text input, placeholder: "your-username/sae-name")
  - Access Token (required password input)
- [x] 11.2.4 Button: "Upload SAE" (emerald gradient)
- [x] 11.2.5 Upload progress indicator

### 11.3 Your SAEs List Component
- [x] 11.3.1 Create `frontend/src/components/saes/SAECard.tsx`
- [x] 11.3.2 Card layout per mock:
  - Icon (document icon)
  - Name (bold)
  - Subtitle: "{model} • Layer {N} • {features} features"
  - Status badge: "Ready" (emerald), "Downloading" (amber), etc.
  - Action button: "Use in Steering →" (slate, arrow icon)
- [x] 11.3.3 List container with "Local: N" and "HuggingFace: N" badges
- [x] 11.3.4 Empty state when no SAEs available

---

## Phase 12: Steering Tab - Zustand Store

### 12.0 Steering Store
- [x] 12.1 Create `frontend/src/stores/steeringStore.ts`
- [x] 12.2 Define state:
  ```typescript
  interface SteeringState {
    // SAE Selection
    activeSAE: SAE | null;

    // Feature Selection
    selectedFeatures: SelectedFeature[];  // max 4
    featureSearch: string;
    availableFeatures: Feature[];

    // Generation Config
    prompt: string;
    maxTokens: number;
    temperature: number;
    topP: number;
    samplesPerConfig: number;

    // Advanced Config
    showAdvanced: boolean;
    steeringMethod: 'direct_decoder' | 'sae_ts';
    layerStrategy: 'single' | 'multi';
    randomSeed: number | null;

    // Results
    isGenerating: boolean;
    unsteeredResults: GenerationOutput | null;
    steeredResults: SteeredOutput[];
    error: string | null;
  }
  ```
- [x] 12.3 Implement actions:
  - `setActiveSAE(sae)` - Set SAE, clear features
  - `searchFeatures(query)` - Search features for active SAE
  - `addFeature(feature)` - Add to selection (max 4)
  - `removeFeature(featureId)` - Remove from selection
  - `updateFeatureStrength(featureId, strength)` - Update strength
  - `updateFeatureLayer(featureId, layer)` - Update target layer
  - `applyPreset(preset)` - Apply Subtle/Moderate/Strong
  - `setPrompt(prompt)` - Update prompt
  - `generateComparison()` - Execute generation
  - `saveExperiment()` - Save current config
  - `exportResults()` - Export to JSON/CSV
  - `reset()` - Clear all state

---

## Phase 13: Steering Tab - Left Sidebar Components

### 13.0 Feature Selector Component
- [x] 13.1 Create `frontend/src/components/steering/FeatureSelector.tsx`
- [x] 13.2 Fixed width: 272px (per mock)
- [x] 13.3 Sections:
  - Active SAE dropdown (full width)
  - Selected Features header with "+ Add" button
  - Selected feature cards (color-coded)
  - Strength Presets row
  - Feature Browser with search

### 13.1 Active SAE Dropdown
- [x] 13.1.1 Dropdown showing: "{name} - Layer {N} ({features} features)"
- [x] 13.1.2 Status indicator dot (emerald for Ready)
- [x] 13.1.3 Subtitle: "{latent_dim} features • Layer {N} • Ready"

### 13.2 Selected Feature Card
- [x] 13.2.1 Create per-feature card with:
  - Color dot (teal/blue/purple/amber based on index)
  - Feature ID and name
  - Strength value (right side, e.g., "-33", "141")
- [x] 13.2.2 Strength slider component (see 13.3)
- [x] 13.2.3 Warning banner for extreme values:
  - Yellow: "High strength - monitor coherence" (|α| > 100)
  - Red: "Negative values suppress this feature" (α < 0)
- [x] 13.2.4 Layer dropdown (L0-L21) with chevron
- [x] 13.2.5 Action buttons: View (eye icon), External link (arrow icon)

### 13.3 Strength Slider Component
- [x] 13.3.1 Create `frontend/src/components/steering/StrengthSlider.tsx`
- [x] 13.3.2 Range: -100 to +300
- [x] 13.3.3 Visual zones on slider track:
  - Red zone: -100 to -50 (suppression)
  - Amber zone: -50 to 0 (mild suppression)
  - Normal zone: 0 to 100
  - Amber zone: 100 to 200 (high amplification)
  - Red zone: 200 to 300 (extreme)
- [x] 13.3.4 Thumb follows zone colors
- [x] 13.3.5 Tick marks at: -100, 0, 100, 200, 300

### 13.4 Strength Presets Row
- [ ] 13.4.1 Three buttons: "Subtle" (10), "Moderate" (50), "Strong" (100)
- [ ] 13.4.2 Style: bg-slate-800 hover:bg-slate-700 rounded px-3 py-1

### 13.5 Feature Browser
- [x] 13.5.1 Search input: "Search features..."
- [x] 13.5.2 Scrollable list of available features
- [x] 13.5.3 Each feature shows:
  - Feature ID (e.g., "#32")
  - Name (e.g., "nuclear_program")
  - Activation frequency (e.g., "82.0%")
  - Top tokens as chips (e.g., "Iran", "nuclear", "program")
- [x] 13.5.4 Click to add feature to selection (if < 4 selected)
- [x] 13.5.5 Disable already-selected features

---

## Phase 14: Steering Tab - Main Content Area

### 14.0 Generation Config Component
- [x] 14.1 Create `frontend/src/components/steering/GenerationConfig.tsx`
- [x] 14.2 Prompt section:
  - Large textarea with placeholder
  - Quick prompt buttons below: "In the beginning...", "The capital of Fra...", "def fibonacci(n):...", "Once upon a time..."
- [x] 14.3 Generation Parameters section:
  - Grid layout: Max Tokens | Temperature | Top-p | Samples per Config
  - "Show Advanced" toggle (chevron)
- [x] 14.4 Advanced section (collapsed by default):
  - Steering Method dropdown: Direct Decoder / SAE-TS
  - Layer Strategy dropdown: Single / Multi
  - Random Seed input

### 14.1 Comparison Configuration
- [ ] 14.1.1 Visual preview cards:
  - Unsteered card (gray dot): "Baseline generation"
  - Selected feature cards (colored dots): show name, α value, layer
- [ ] 14.1.2 "+ Add Feature" placeholder card (dashed border)
- [ ] 14.1.3 Summary text: "Will generate 2 samples × 3 configs = 6 total outputs"
- [ ] 14.1.4 Estimated time: "Est. time: ~5s"

### 14.2 Generate Button
- [x] 14.2.1 Large centered button: "Generate Comparison" with play icon
- [x] 14.2.2 Gradient: teal-500 to emerald-500
- [x] 14.2.3 Disabled when: no SAE, no features, no prompt, generating
- [x] 14.2.4 Loading state: spinner + "Generating..."

---

## Phase 15: Steering Tab - Results Display

### 15.0 Comparison Results Component
- [x] 15.1 Create `frontend/src/components/steering/ComparisonResults.tsx`
- [x] 15.2 Header: "Comparison Results" + action buttons (Regenerate, Save, Export)
- [x] 15.3 Conditional render: only show when results exist

### 15.1 Unsteered Results Section
- [x] 15.1.1 Header: gray dot + "UNSTEERED" + "Baseline" badge
- [x] 15.1.2 Sample cards:
  - "Sample 1", "Sample 2" labels
  - Generated text in card
  - Copy button (top right of each sample)
- [x] 15.1.3 Metrics bar:
  - Perplexity: numeric value
  - Coherence: 0-1 value
  - Avg Length: token count

### 15.2 Steered Results Section
- [x] 15.2.1 Per-feature result blocks
- [x] 15.2.2 Header: colored dot + feature ID + name + "α = {strength}" + "Layer {N}"
- [x] 15.2.3 Sample cards (same as unsteered)
- [x] 15.2.4 Metrics bar with additional:
  - Perplexity (amber if > 30)
  - Coherence
  - Behavioral: 0-1 score
  - Avg Length

### 15.3 Metrics Visualization
- [x] 15.3.1 Color coding:
  - Perplexity: white (normal), amber (> 30), red (> 50)
  - Coherence: emerald (> 0.8), amber (0.5-0.8), red (< 0.5)
  - Behavioral: emerald (> 0.8), amber (0.5-0.8), red (< 0.5)

---

## Phase 16: Navigation Integration

### 16.0 Add SAEs and Steering Tabs
- [x] 16.1 Update `frontend/src/App.tsx`:
  - Add SAEs route: `/saes` → SAEsPanel
  - Add Steering route: `/steering` → SteeringPanel
- [x] 16.2 Update navigation component:
  - Add "SAEs" tab after "Training"
  - Add "Steering" tab after "Labeling"
- [x] 16.3 Update tab order: Datasets | Models | Training | SAEs | Extractions | Labeling | Steering | Templates | Monitor

### 16.1 Cross-Tab Navigation
- [x] 16.1.1 "Use in Steering →" button on SAE cards:
  - Navigate to /steering
  - Pre-select the SAE in steering store
- [ ] 16.1.2 Feature detail modal "Steer with this feature" button:
  - Navigate to /steering
  - Pre-select SAE and add feature

---

## Phase 17: WebSocket Integration

### 17.0 SAE Progress WebSocket Hook
- [ ] 17.1 Create `frontend/src/hooks/useSAEWebSocket.ts`
- [ ] 17.2 Subscribe to channels:
  - `sae/download/{jobId}` for download progress
  - `sae/upload/{jobId}` for upload progress
- [ ] 17.3 Update saesStore on progress events
- [ ] 17.4 Auto-refresh SAE list on completion

### 17.1 Steering Progress (Optional)
- [ ] 17.1.1 For long generations, emit progress events
- [ ] 17.1.2 Update UI with generation progress

---

## Phase 18: Testing

### 18.0 Backend Tests
- [ ] 18.1 Unit tests for SAEConverter (roundtrip conversion)
- [ ] 18.2 Unit tests for HuggingFaceService (mocked API calls)
- [ ] 18.3 Unit tests for SAEManagerService
- [ ] 18.4 Unit tests for SteeringService (hook registration, metrics)
- [ ] 18.5 Integration tests for SAE endpoints
- [ ] 18.6 Integration tests for Steering endpoints

### 18.1 Frontend Tests
- [ ] 18.1.1 Unit tests for saesStore
- [ ] 18.1.2 Unit tests for steeringStore
- [ ] 18.1.3 Component tests for SAEsPanel
- [ ] 18.1.4 Component tests for SteeringPanel
- [ ] 18.1.5 Component tests for StrengthSlider

### 18.2 E2E Tests
- [ ] 18.2.1 Download SAE from HuggingFace flow
- [ ] 18.2.2 Upload SAE to HuggingFace flow
- [ ] 18.2.3 Complete steering workflow: select SAE → add features → generate → view results

---

## Phase 19: Documentation

### 19.0 User Documentation
- [ ] 19.1 SAEs tab user guide: how to download/upload SAEs
- [ ] 19.2 Steering tab user guide: how to run steering experiments
- [ ] 19.3 Strength parameter guide: what different values mean
- [ ] 19.4 Troubleshooting guide: common errors and solutions

### 19.1 API Documentation
- [ ] 19.1.1 Document SAE management endpoints in OpenAPI
- [ ] 19.1.2 Document Steering endpoints in OpenAPI
- [ ] 19.1.3 Add example requests/responses

---

## Implementation Priority

### Critical Path (P0)
1. Phase 1: Database schema
2. Phase 4: SAE Manager Service (local SAE listing)
3. Phase 5: Steering Service
4. Phase 6-7: Schemas and endpoints
5. Phase 10-11: SAEs tab UI
6. Phase 12-15: Steering tab UI
7. Phase 16: Navigation integration

### Secondary (P1)
1. Phase 2-3: HuggingFace integration
2. Phase 8: Celery tasks for async operations
3. Phase 17: WebSocket progress

### Polish (P2)
1. Phase 18: Testing
2. Phase 19: Documentation

---

## Estimated Effort

| Phase | Tasks | Estimated Hours |
|-------|-------|-----------------|
| Phase 1: Database | 5 | 2h |
| Phase 2: HuggingFace Service | 8 | 6h |
| Phase 3: SAE Converter | 6 | 8h |
| Phase 4: SAE Manager | 7 | 4h |
| Phase 5: Steering Service | 15 | 12h |
| Phase 6: Schemas | 8 | 3h |
| Phase 7: Endpoints | 10 | 6h |
| Phase 8: Celery Tasks | 5 | 3h |
| Phase 9: Frontend Types | 4 | 2h |
| Phase 10: SAEs Store | 4 | 2h |
| Phase 11: SAEs UI | 10 | 6h |
| Phase 12: Steering Store | 3 | 2h |
| Phase 13: Sidebar UI | 15 | 8h |
| Phase 14: Config UI | 8 | 4h |
| Phase 15: Results UI | 8 | 4h |
| Phase 16: Navigation | 4 | 1h |
| Phase 17: WebSocket | 4 | 2h |
| Phase 18: Testing | 15 | 8h |
| Phase 19: Documentation | 6 | 3h |
| **Total** | **145** | **~86h** |

---

## Notes

- **Steering strength range**: -100 to +300 (per mock UI) vs previous -5 to +5
  - Conversion: new_strength / 100 = old_coefficient
  - -100 → feature zeroed, 0 → unchanged, +100 → doubled, +300 → 4x
- **Per-feature layers**: Each feature can target different layers (new requirement)
- **Max 4 features**: Hard limit for comparison clarity
- **Color coding**: teal (feature 1), blue (2), purple (3), amber (4)
- **Warning zones**: Visual feedback for extreme steering values

---

**Last Updated:** 2025-11-26
**Status:** In Progress - Core UI Complete, Backend Steering Service Pending
**Supersedes:** 005_FTASKS|Model_Steering.md (core steering), 006_FTASKS|SAE_Sharing_Integration.md (HF integration)

---

## Progress Summary (2025-11-26)

### Completed Phases:
- **Phase 1:** Database & Backend Foundation - All tasks complete
- **Phase 2:** HuggingFace Service - Implementation complete, tests pending
- **Phase 4:** SAE Manager Service - Implementation complete, tests pending
- **Phase 6:** Pydantic Schemas - All schemas defined
- **Phase 7:** FastAPI Endpoints - All endpoints implemented (timeout/rate limiting pending)
- **Phase 9:** Frontend Types & API Clients - Complete
- **Phase 10:** SAEs Tab Zustand Store - Complete
- **Phase 11:** SAEs Tab UI Components - Complete
- **Phase 12:** Steering Tab Zustand Store - Complete
- **Phase 13:** Steering Tab Left Sidebar - Complete (strength presets pending)
- **Phase 14:** Steering Tab Main Content - Mostly complete (preview cards pending)
- **Phase 15:** Steering Tab Results Display - UI complete
- **Phase 16:** Navigation Integration - Complete

### Pending/Deferred:
- **Phase 3:** SAELens Format Converter - Not yet implemented
- **Phase 5:** Steering Service (Backend) - Deferred, requires model integration
- **Phase 8:** Celery Tasks for SAE - Not yet implemented
- **Phase 17-19:** WebSocket, Testing, Documentation - Not started

### Bug Fixes Applied:
- Fixed feature labels not showing in selected feature cards (steeringStore explicit property assignment)
- Fixed layer detection showing Lnull/L0 (backend now returns actual layer from training config)
- Added auto-select layer dropdown from selected SAE
- Rearranged feature browser layout (Layer/Add above Feature Index)
