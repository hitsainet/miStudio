# Feature PRD: Model Management

**Feature ID:** 002_FPRD|Model_Management
**Feature Name:** Model Management Panel
**Priority:** P0 (Blocker for MVP)
**Status:** Draft
**Created:** 2025-10-05
**Last Updated:** 2025-10-05

---

## 1. Feature Overview

### Feature Description
The Model Management Panel provides complete lifecycle management for language models used in Sparse Autoencoder (SAE) training and interpretability research. Users can download pre-trained models from HuggingFace, quantize them for efficient edge deployment, inspect model architectures, extract activations from specific layers, and manage model storage—all optimized for resource-constrained Jetson Orin Nano devices.

### Problem Statement
SAE training and mechanistic interpretability research require access to language models, but deploying large models on edge devices presents critical challenges:
- **Memory Constraints**: Full-precision models (FP32) exceed Jetson's 8GB RAM
- **Storage Limitations**: Original model weights (1-13GB each) consume limited SSD space
- **Inference Speed**: Unoptimized models are too slow for real-time activation extraction
- **Model Compatibility**: Users need visibility into architecture details (layers, hidden dimensions) before training
- **Activation Extraction**: Complex setup required to hook into transformer layers for SAE training

### Feature Goals
1. **Enable Edge Deployment**: Quantize models to fit in Jetson's 8GB RAM and accelerate inference
2. **Simplify Model Acquisition**: One-click downloads from HuggingFace with automatic quantization
3. **Provide Architecture Transparency**: Detailed layer-by-layer inspection for research planning
4. **Support Activation Extraction**: Configure and extract activations from specific layers/hooks
5. **Optimize Storage**: Efficient storage with quantization (4-bit reduces size by 87.5%)

### Connection to Project Objectives
This feature directly enables the project's core mission of edge-based mechanistic interpretability:
- **Edge-First Design**: Quantization and optimization specifically for Jetson Orin Nano
- **Researcher-Friendly**: Architecture viewer demystifies transformer internals
- **Training Enablement**: Activation extraction is prerequisite for SAE training
- **Reproducible Research**: Version-controlled models with documented architectures

---

## 2. User Stories & Scenarios

### Primary User Stories

#### US-1: Download Model from HuggingFace
**As a** ML researcher
**I want to** download pre-trained models directly from HuggingFace with automatic quantization
**So that** I can quickly deploy models on my edge device without manual conversion

**Acceptance Criteria:**
- User can enter HuggingFace repository ID (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
- User can select quantization format: Q2, Q4, Q8, FP16, FP32
- System validates repository exists and contains model files
- System displays estimated download size and quantized size
- System handles gated models requiring access tokens
- Download progress shows percentage, speed, and ETA
- After download, system automatically quantizes model if format != FP32
- Quantization progress displays with GPU utilization metrics
- Model appears in list with "ready" status after quantization complete
- System gracefully handles OOM errors during quantization with fallback to higher precision

#### US-2: Inspect Model Architecture
**As a** ML researcher
**I want to** view detailed model architecture (layers, attention heads, hidden dimensions)
**So that** I can plan SAE training and understand where to extract activations

**Acceptance Criteria:**
- User can click on any "ready" model to open architecture viewer modal
- Modal displays model overview stats:
  - Total layers (e.g., 12 transformer blocks)
  - Hidden dimension (e.g., 768)
  - Attention heads (e.g., 12)
  - Total parameters (e.g., 1.1B)
  - Vocabulary size (e.g., 50257)
  - Max sequence length (e.g., 1024)
- Modal displays layer-by-layer breakdown with indices
- Each layer shows: layer type (Embedding, TransformerBlock, LayerNorm, Output), dimensions, sub-components
- TransformerBlock layers expand to show: attention (heads × dims), MLP (input → hidden → output)
- Modal displays model configuration JSON (architecture-specific hyperparameters)
- Modal renders within 100ms of click
- Architecture data cached in database for instant retrieval

#### US-3: Extract Model Activations
**As a** ML researcher
**I want to** configure activation extraction from specific layers and hook points
**So that** I can collect training data for my sparse autoencoder

**Acceptance Criteria:**
- User clicks "Extract Activations" button on model card
- System opens extraction configuration modal with:
  - Dataset selector (dropdown of ready datasets)
  - Layer selector (checkboxes for each layer: L0, L1, ..., L11)
  - Hook type selector (checkboxes: residual stream, MLP output, attention output)
  - Max samples slider (limit samples for testing, default: all)
  - Batch size input (default: 32, optimized for Jetson)
- User can select "all layers" or "deselect all" with bulk actions
- System validates at least one layer and one hook type selected
- User clicks "Start Extraction" to queue background job
- System displays extraction progress: percentage, samples processed, ETA
- Extracted activations save to `/data/activations/extraction_{id}/`
- Each layer+hook combination creates separate .npy file (e.g., `layer_4_residual.npy`)
- System creates metadata JSON with extraction config and dataset reference
- Extraction completes with success message showing storage location

#### US-4: View Model Memory Requirements
**As a** ML researcher
**I want to** see estimated memory requirements before downloading
**So that** I can ensure model fits in my device's available RAM

**Acceptance Criteria:**
- Download form displays calculated memory estimates for each quantization format:
  - FP32: {X} GB RAM (original precision)
  - FP16: {Y} GB RAM (~50% of FP32)
  - Q8: {Z} GB RAM (~25% of FP32)
  - Q4: {W} GB RAM (~12.5% of FP32)
  - Q2: {V} GB RAM (~6.25% of FP32)
- System highlights recommended format (Q4) for Jetson Orin Nano
- System displays warning if selected format exceeds available RAM (>6GB for 8GB device)
- After model download, model card displays actual memory usage (measured)
- Memory estimates account for both model weights and inference overhead (~500MB)

#### US-5: Save and Reuse Extraction Configuration Templates
**As a** ML researcher
**I want to** save my activation extraction configurations as reusable templates
**So that** I can quickly apply the same extraction settings across different models and experiments

**Acceptance Criteria:**
- User can save current extraction configuration (layers + hook types + sampling) as a template
- System auto-generates descriptive template name with layer range and timestamp
- User can provide custom name and optional description
- User can mark templates as favorites for quick access
- User can load saved template to instantly populate all extraction fields
- User can delete unwanted templates with confirmation
- Templates persist across sessions in database
- User can export all templates to JSON file for backup
- User can import templates from JSON file
- Templates are included in combined export with training templates and steering presets

#### US-6: Delete Unused Models
**As a** ML researcher
**I want to** delete models I no longer need
**So that** I can free up storage space for new models

**Acceptance Criteria:**
- User can click delete button on model card
- System displays confirmation modal showing:
  - Model name, parameter count, and size
  - Warning if model is referenced by any trainings (prevent deletion)
  - Storage space to be freed (raw + quantized sizes)
- After confirmation, system deletes:
  - Raw model files from `/data/models/raw/{model_id}/`
  - Quantized files from `/data/models/quantized/{model_id}_q{X}/`
  - Database metadata record (CASCADE deletes related extractions)
  - Tokenizer files from cache
- System prevents deletion if model is actively used by running training
- UI updates immediately with fade-out animation
- System logs deletion event with model ID and timestamp

### Secondary User Scenarios

#### SC-1: Quantize Existing Model to Different Format
**Scenario:** User has FP16 model but wants Q4 for better performance

**Flow:**
1. User clicks "Re-quantize" button on model card
2. System opens quantization dialog with format selector
3. User selects Q4 format
4. System queues quantization background job
5. Progress bar shows quantization progress (0-100%)
6. New model variant appears in list with suffix (e.g., "TinyLlama-1.1B-Q4")
7. Original model remains unchanged

**Acceptance Criteria:**
- Re-quantization creates new model record, doesn't modify original
- User can have multiple quantization formats of same model
- System prevents duplicate quantizations (same model + format)
- Quantization uses GPU acceleration (bitsandbytes library)

#### SC-2: Compare Model Architectures
**Scenario:** User wants to compare two models side-by-side for research

**Flow:**
1. User opens architecture viewer for Model A (e.g., GPT-2)
2. User clicks "Compare with..." button
3. System shows model selector
4. User selects Model B (e.g., TinyLlama)
5. System displays side-by-side comparison table:
   - Total layers: 12 vs 22
   - Hidden dim: 768 vs 2048
   - Attention heads: 12 vs 32
   - Parameters: 124M vs 1.1B
   - Max sequence: 1024 vs 2048
6. User can toggle between models in single viewer

**Acceptance Criteria:**
- Comparison highlights differences in red, similarities in green
- User can compare up to 2 models at once
- Comparison persists across page reloads (saved in local storage)

#### SC-3: Resume Interrupted Model Download
**Scenario:** Network interruption during large model download (13GB)

**Flow:**
1. Model shows "downloading" status with progress at 67%
2. Network connection lost
3. System detects failure after 30-second timeout
4. Model transitions to "error" status with message: "Download interrupted"
5. User clicks "Retry Download" button
6. System resumes from last completed chunk (HTTP Range requests)
7. Download continues from 67% to completion
8. Model transitions to "quantizing" status

**Acceptance Criteria:**
- System uses resumable downloads (wget or curl with -C flag)
- Partial downloads stored in `/data/temp/models/`
- Maximum 3 automatic retry attempts with exponential backoff
- User can manually trigger retry unlimited times
- System validates file integrity with checksums

### Edge Cases and Error Scenarios

#### EC-1: Insufficient GPU Memory for Quantization
**Scenario:** User attempts Q4 quantization but GPU OOM occurs

**Handling:**
- System catches OOM error from bitsandbytes library
- Display error: "Quantization failed: Insufficient GPU memory. Trying FP16 fallback..."
- System automatically retries with FP16 (less memory intensive)
- If FP16 also fails, display: "Model too large for device. Please try Q8 or use smaller model."
- Provide link to recommended models for Jetson (<2B parameters)

#### EC-2: Corrupted Model Files
**Scenario:** Downloaded model files fail integrity check

**Handling:**
- System validates checksums against HuggingFace metadata
- If mismatch detected, display: "Model files corrupted. Re-downloading..."
- System deletes corrupted files and retries download (auto-retry count: 1)
- If retry fails, display: "Unable to download model. Please check network connection and try again."
- Provide "Delete and Retry" button to clean up and start fresh

#### EC-3: Incompatible Model Architecture
**Scenario:** User downloads model with unsupported architecture (e.g., encoder-only BERT)

**Handling:**
- System detects architecture from config.json during download
- If unsupported, display warning: "This model architecture (bert-base) is not supported for SAE training. miStudio supports decoder-only models: GPT-2, Llama, Phi, Pythia."
- System completes download but marks model as "incompatible" status
- User can view architecture but cannot extract activations or train SAEs
- Provide link to supported model architectures documentation

#### EC-4: Gated Model Without Access Token
**Scenario:** User attempts to download Llama-2 (gated) without providing token

**Handling:**
- HuggingFace API returns 403 Forbidden
- Display error: "This model requires authentication. Please provide a HuggingFace access token with model access permissions."
- Highlight access token input field with red border
- Provide link to HuggingFace token generation page
- Show example of valid token format: "hf_xxxxxxxxxxxxxxxxxxxx"

### User Journey Flows

#### Journey 1: First Model Download
```
1. User opens Model Management panel (empty state)
   ↓
2. User reads "No models yet. Download from HuggingFace to get started."
   ↓
3. User enters "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in repository field
   ↓
4. System validates repository and displays size estimate: 2.1GB → 270MB (Q4)
   ↓
5. User selects Q4 quantization format
   ↓
6. User clicks "Download Model from HuggingFace"
   ↓
7. System starts download, progress bar shows 0% → 100% (2 minutes)
   ↓
8. Status transitions to "quantizing" with GPU utilization graph
   ↓
9. Quantization completes in 30 seconds
   ↓
10. Status transitions to "ready" with checkmark
    ↓
11. User clicks on model card to inspect architecture
    ↓
12. Architecture viewer opens showing 22 layers, 2048 hidden dim
    ↓
13. User closes viewer, proceeds to activation extraction
```

#### Journey 2: Extract Activations for SAE Training
```
1. User has TinyLlama model (ready) and OpenWebText dataset (ready)
   ↓
2. User clicks "Extract Activations" button on TinyLlama card
   ↓
3. Extraction configuration modal opens
   ↓
4. User selects OpenWebText dataset from dropdown
   ↓
5. User selects layers: L0, L5, L10, L15, L20 (every 5th layer)
   ↓
6. User selects hook types: Residual Stream, MLP Output
   ↓
7. User sets max samples: 10,000 (for testing)
   ↓
8. User clicks "Start Extraction"
   ↓
9. System validates configuration (success)
   ↓
10. Background job starts, progress shows 0% → 100% (5 minutes)
    ↓
11. Success message: "Activations extracted to /data/activations/extraction_abc123/"
    ↓
12. User navigates to Training panel to configure SAE training using extracted activations
```

---

## 3. Functional Requirements

### FR-1: HuggingFace Model Download
1. **FR-1.1**: System shall provide input field for HuggingFace repository ID (format: "org/model-name")
2. **FR-1.2**: System shall provide quantization format selector (dropdown: FP32, FP16, Q8, Q4, Q2)
3. **FR-1.3**: System shall provide optional access token input (password-masked) for gated models
4. **FR-1.4**: System shall validate repository exists via HuggingFace API before download
5. **FR-1.5**: System shall calculate and display estimated sizes:
   - Original size (from repo metadata)
   - Quantized size (calculated: params × bytes_per_param × quantization_factor)
   - Memory requirement (size × 1.2 for inference overhead)
6. **FR-1.6**: System shall display warning if memory requirement > 6GB (Jetson constraint)
7. **FR-1.7**: System shall initiate download via HuggingFace Transformers library
8. **FR-1.8**: System shall display real-time download progress: percentage, MB downloaded, speed (MB/s), ETA
9. **FR-1.9**: System shall support resume capability for interrupted downloads (HTTP Range requests)
10. **FR-1.10**: System shall store raw model files in `/data/models/raw/{model_id}/`
11. **FR-1.11**: System shall create database record with status "downloading" before starting
12. **FR-1.12**: System shall transition to "loading" status after download complete
13. **FR-1.13**: System shall load model config and validate architecture compatibility
14. **FR-1.14**: System shall transition to "quantizing" status if format != FP32
15. **FR-1.15**: System shall perform quantization using bitsandbytes library (4-bit, 8-bit) or manual FP16 conversion
16. **FR-1.16**: System shall store quantized model in `/data/models/quantized/{model_id}_{format}/`
17. **FR-1.17**: System shall transition to "ready" status after successful quantization
18. **FR-1.18**: System shall transition to "error" status on failure with descriptive error message
19. **FR-1.19**: System shall implement automatic retry (3 attempts) with exponential backoff for transient failures
20. **FR-1.20**: System shall perform download and quantization as Celery background tasks

### FR-2: Model Architecture Inspection
1. **FR-2.1**: System shall parse model config.json during download to extract architecture details
2. **FR-2.2**: System shall store architecture metadata in database as JSONB field containing:
   - Model type (gpt2, llama, phi, pythia, etc.)
   - Number of layers
   - Hidden dimension size
   - Number of attention heads
   - Head dimension (hidden_size / num_heads)
   - Intermediate size (MLP hidden dimension)
   - Vocabulary size
   - Max position embeddings (sequence length)
   - Activation function
   - Layer norm epsilon
3. **FR-2.3**: System shall provide architecture viewer modal accessible by clicking model card
4. **FR-2.4**: Architecture viewer shall display overview stats in grid layout (4 columns):
   - Total Layers
   - Hidden Dimension
   - Attention Heads
   - Total Parameters
5. **FR-2.5**: Architecture viewer shall display layer-by-layer breakdown with:
   - Layer index (0, 1, 2, ...)
   - Layer type (Embedding, TransformerBlock_0, LayerNorm, Output)
   - Dimensions/shapes for each layer
6. **FR-2.6**: TransformerBlock entries shall expand to show sub-components:
   - Attention: {num_heads} heads × {head_dim} dims
   - MLP: {hidden_size} → {intermediate_size} → {hidden_size}
7. **FR-2.7**: Architecture viewer shall display model configuration JSON in collapsible section
8. **FR-2.8**: System shall cache architecture data in database for instant retrieval (<100ms)
9. **FR-2.9**: Architecture viewer shall support keyboard navigation (Tab, Enter, Esc)
10. **FR-2.10**: Architecture viewer shall render within 100ms of modal open

### FR-3: Model Quantization
1. **FR-3.1**: System shall support quantization formats: Q2, Q4, Q8, FP16, FP32
2. **FR-3.2**: System shall use bitsandbytes library for 4-bit and 8-bit quantization
3. **FR-3.3**: System shall use PyTorch native for FP16 quantization (`model.half()`)
4. **FR-3.4**: System shall skip quantization if format == FP32 (no quantization needed)
5. **FR-3.5**: System shall calculate quantization factors:
   - FP32: 1.0 (baseline)
   - FP16: 0.5 (2 bytes vs 4 bytes per param)
   - Q8: 0.25 (1 byte vs 4 bytes per param)
   - Q4: 0.125 (0.5 bytes vs 4 bytes per param)
   - Q2: 0.0625 (0.25 bytes vs 4 bytes per param)
6. **FR-3.6**: System shall display quantization progress with GPU utilization monitoring
7. **FR-3.7**: System shall validate quantized model by running test forward pass (1 batch)
8. **FR-3.8**: System shall store quantization metadata in database: format, size_bytes, compression_ratio
9. **FR-3.9**: System shall handle OOM errors during quantization:
   - Catch `torch.cuda.OutOfMemoryError`
   - Log error with GPU memory usage
   - Fallback to next higher precision (Q2→Q4→Q8→FP16)
   - Display error message to user with fallback format
10. **FR-3.10**: System shall support re-quantization (creating new model variant from existing)
11. **FR-3.11**: System shall prevent duplicate quantizations (same model + format)

### FR-4: Activation Extraction Configuration
1. **FR-4.1**: System shall provide "Extract Activations" button on model cards with status "ready"
2. **FR-4.2**: System shall open extraction configuration modal with form fields:
   - Dataset selector (dropdown of ready datasets)
   - Layer selector (multi-select checkboxes: L0, L1, ..., L{N-1})
   - Hook type selector (multi-select checkboxes: residual, mlp, attention)
   - Max samples input (integer, optional, default: null = all samples)
   - Batch size input (integer, default: 32, range: 1-512)
   - Top-k examples per feature input (integer, default: 100, range: 10-1000)
3. **FR-4.3**: System shall provide "Select All Layers" and "Deselect All Layers" bulk actions
4. **FR-4.4**: System shall validate extraction configuration:
   - At least 1 layer selected
   - At least 1 hook type selected
   - Dataset must be in "ready" status
   - Batch size must be power of 2 (optimization requirement)
5. **FR-4.5**: System shall calculate estimated extraction time and storage:
   - Time: (num_samples / batch_size) × (num_layers × num_hooks) × 0.5 seconds per batch
   - Storage: num_samples × seq_len × hidden_dim × 4 bytes (FP32) × num_layers × num_hooks
6. **FR-4.6**: System shall display warnings:
   - If estimated time > 1 hour: "This extraction may take over 1 hour. Consider reducing max samples or layers."
   - If estimated storage > 50GB: "This extraction requires 50GB+ storage. Ensure sufficient disk space."
7. **FR-4.7**: System shall queue extraction as Celery background task
8. **FR-4.8**: System shall return job_id and extraction_id to client
9. **FR-4.9**: System shall create database record for extraction with status "initializing"

### FR-4A: Extraction Template Management
1. **FR-4A.1**: System shall provide "Saved Templates" collapsible section within Activation Extraction modal
2. **FR-4A.2**: System shall display template count in section header (e.g., "Saved Templates (5)")
3. **FR-4A.3**: System shall provide "Save as Template" form with fields:
   - Template name input (auto-generated default: `{type}_layers{min}-{max}_{samples}samples_{HHMM}`)
   - Optional description textarea (max 500 characters)
   - "Save Template" submit button
4. **FR-4A.4**: System shall auto-generate template names with format:
   - Single layer: `{activation_type}_layer{N}_{samples}samples_{HHMM}`
   - Multiple layers: `{activation_type}_layers{min}-{max}_{samples}samples_{HHMM}`
   - Examples: `residual_layers0-11_1000samples_1430`, `mlp_layer6_5000samples_0925`
   - Timestamp format: HHMM (24-hour, e.g., 1430 = 2:30 PM)
5. **FR-4A.5**: System shall save extraction templates to `extraction_templates` database table with fields:
   - id (UUID primary key)
   - name (VARCHAR 255, required)
   - description (TEXT, optional)
   - layers (INTEGER[], PostgreSQL array)
   - hook_types (VARCHAR(50)[], array: ['residual', 'mlp', 'attention'])
   - max_samples (INTEGER, nullable)
   - top_k_examples (INTEGER, default 100)
   - is_favorite (BOOLEAN, default false)
   - created_at, updated_at (TIMESTAMP)
6. **FR-4A.6**: System shall display template list with template cards showing:
   - Template name (bold, clickable)
   - Description (if provided, gray text, truncated to 100 chars)
   - Layer range badge (e.g., "Layers 0-11" or "Layer 6")
   - Hook types badges (e.g., "Residual", "MLP")
   - Sample count (if specified, e.g., "1000 samples")
   - Favorite star icon (gold if favorited, gray if not)
   - "Load" button (emerald green)
   - "Delete" button (red)
7. **FR-4A.7**: System shall implement "Load Template" action:
   - Populate layer checkboxes from template.layers array
   - Select hook type checkboxes from template.hook_types array
   - Set max_samples input from template.max_samples
   - Set top_k_examples input from template.top_k_examples
   - Show success toast: "Template '{name}' loaded"
8. **FR-4A.8**: System shall implement "Delete Template" action:
   - Show confirmation dialog: "Delete template '{name}'? This cannot be undone."
   - On confirm: DELETE /api/v1/templates/extraction/:id
   - Remove template card from UI with fade-out animation
   - Show success toast: "Template '{name}' deleted"
9. **FR-4A.9**: System shall implement "Toggle Favorite" action:
   - Click star icon to toggle is_favorite flag
   - Send PATCH /api/v1/templates/extraction/:id/favorite { is_favorite: boolean }
   - Update star icon immediately (optimistic UI update)
   - Move favorited templates to top of list
10. **FR-4A.10**: System shall support template export/import:
    - Export: POST /api/v1/templates/export (downloads combined JSON file with all template types)
    - Import: POST /api/v1/templates/import (uploads JSON file, validates, imports all templates)
    - Export format includes version field for future compatibility
    - Import validates structure and skips invalid templates with warning
11. **FR-4A.11**: System shall prevent duplicate template names:
    - Check for existing name before save
    - If duplicate found, suggest: "extraction_template_{name}_{counter}" (e.g., residual_layers0-11_1000samples_1430_2)
12. **FR-4A.12**: System shall sort templates by:
    - Favorites first (is_favorite = true)
    - Then by updated_at DESC (most recently modified first)
13. **FR-4A.13**: System shall validate template configuration before save:
    - At least 1 layer must be selected
    - At least 1 hook type must be selected
    - Layer indices must be valid for current architecture
    - Name must not be empty and must be ≤255 characters

### FR-5: Activation Extraction Execution
1. **FR-5.1**: System shall implement PyTorch forward hooks for activation capture:
   - Residual stream: Hook after layer norm (output of transformer block)
   - MLP output: Hook after MLP feed-forward layer
   - Attention output: Hook after attention layer before residual addition
2. **FR-5.2**: System shall load model in inference mode (`model.eval()`, `torch.no_grad()`)
3. **FR-5.3**: System shall load dataset in batches using PyTorch DataLoader
4. **FR-5.4**: System shall register forward hooks on specified layers
5. **FR-5.5**: System shall run forward passes batch-by-batch, collecting activations
6. **FR-5.6**: System shall store activations as NumPy arrays:
   - Shape: [num_samples, seq_len, hidden_dim]
   - Dtype: float32
   - File format: .npy (efficient, memory-mappable)
   - Filename: `layer_{idx}_{hook_type}.npy`
7. **FR-5.7**: System shall save activations to `/data/activations/extraction_{id}/`
8. **FR-5.8**: System shall create metadata JSON file with:
   - Model ID and architecture
   - Dataset ID and sample count
   - Layers and hook types extracted
   - Timestamp and duration
   - File paths and sizes
9. **FR-5.9**: System shall update extraction progress every 100 batches (or 1%, whichever is larger)
10. **FR-5.10**: System shall transition extraction status: initializing → extracting → completed
11. **FR-5.11**: System shall handle extraction errors:
    - Catch OOM errors (reduce batch size automatically and retry)
    - Catch model loading errors (report incompatible model)
    - Catch dataset errors (report corrupted dataset)
    - Transition to "error" status with detailed error message
12. **FR-5.12**: System shall compute activation statistics:
    - Mean activation magnitude per layer
    - Max activation magnitude per layer
    - Activation sparsity (percentage of near-zero activations)
13. **FR-5.13**: System shall log extraction performance metrics:
    - Samples per second
    - GPU utilization
    - Memory usage
    - Total duration

### FR-6: Model Listing and Status Display
1. **FR-6.1**: System shall display model list with cards showing:
   - Model name (from HuggingFace repo)
   - Parameter count (human-readable: 135M, 1.1B, 2.7B)
   - Quantization format (Q4, FP16, etc.)
   - Memory requirement (GB)
   - Status badge (downloading, loading, quantizing, ready, error)
   - Progress bar (if downloading/quantizing)
2. **FR-6.2**: System shall use polling (every 2 seconds) to fetch status updates for active operations
3. **FR-6.3**: System shall display status-specific icons:
   - Downloading: Animated spinner (Loader icon)
   - Quantizing: Activity icon with pulse animation
   - Ready: Checkmark (CheckCircle icon)
   - Error: X icon with red color
4. **FR-6.4**: System shall display "Extract Activations" button only for ready models
5. **FR-6.5**: System shall enable card click to open architecture viewer only for ready models
6. **FR-6.6**: System shall display progress bar with gradient fill for visual appeal
7. **FR-6.7**: System shall show tooltips on hover for status badges explaining current state
8. **FR-6.8**: System shall sort models: in-progress first, then by created_at DESC

### FR-7: Model Deletion and Cleanup
1. **FR-7.1**: System shall provide delete button on each model card
2. **FR-7.2**: System shall check for training dependencies before deletion:
   - Query `trainings` table for references to model_id
   - If active trainings exist (status != completed/stopped), block deletion
3. **FR-7.3**: System shall display confirmation modal with:
   - Model name, parameter count, and size
   - Warning if trainings reference this model
   - Storage space to be freed (raw + quantized + tokenizer)
   - List of affected trainings (if any)
4. **FR-7.4**: Upon confirmation, system shall:
   - Delete raw model files from `/data/models/raw/{model_id}/`
   - Delete quantized files from `/data/models/quantized/{model_id}_{format}/`
   - Delete tokenizer files from cache
   - Delete database record (CASCADE deletes related extractions)
   - Update UI immediately (fade-out animation)
5. **FR-7.5**: System shall prevent deletion if model is actively loading or quantizing
6. **FR-7.6**: System shall log deletion events with model ID, size freed, timestamp
7. **FR-7.7**: System shall provide "Delete All Unused Models" bulk action (models not in trainings)

### FR-8: Memory and Performance Monitoring
1. **FR-8.1**: System shall track GPU memory usage during quantization and extraction
2. **FR-8.2**: System shall display GPU utilization graph (0-100%) during quantization
3. **FR-8.3**: System shall monitor disk space before operations:
   - Check available space before model download
   - Check available space before activation extraction
   - Display warning if < 10GB free space
4. **FR-8.4**: System shall estimate and display operation durations:
   - Model download: size_mb / network_speed_mbps (estimated from first 10%)
   - Quantization: params_count / 1e9 × 30 seconds (empirical estimate)
   - Extraction: samples / batch_size × layers × hooks × 0.5 seconds per batch
5. **FR-8.5**: System shall track actual operation durations and improve estimates over time
6. **FR-8.6**: System shall display memory requirement warnings prominently:
   - Yellow warning: 4-6GB (caution, monitor closely)
   - Red warning: >6GB (risky, may cause OOM)
   - Green: <4GB (safe for Jetson)

---

## 4. User Experience Requirements

### UI/UX Specifications

#### Primary Reference: Mock UI
All UI/UX specifications reference the PRIMARY authoritative specification:
**@0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx**

Specifically:
- **ModelsPanel component** (lines 1204-1343) - Main panel layout and model cards
- **ModelArchitectureViewer component** (lines 1346-1437) - Architecture inspection modal
- **ActivationExtractionConfig component** (lines 1440-1625) - Extraction configuration modal

#### Visual Design Standards
- **Theme**: Dark mode (slate-950 background, slate-100 text) matching Dataset Management
- **Primary Color**: Purple-400 for model-related elements (differentiates from dataset emerald-400)
- **Secondary Colors**:
  - Blue-400 for downloading states
  - Yellow-400 for quantizing/processing states
  - Emerald-400 for success/ready states
  - Red-400 for error states
- **Typography**: System font stack, consistent with application standards
- **Spacing**: 4px grid system (gap-2, gap-4, gap-6)
- **Borders**: border-slate-800 for divisions
- **Shadows**: Minimal, backdrop-blur for modals

#### Layout Structure
```
┌─────────────────────────────────────────────────────┐
│ Model Management                                    │
│ [Download from HuggingFace card]                   │
│                                                     │
│ ┌─────────────────────────────────────────────┐   │
│ │ HuggingFace Model Repository:               │   │
│ │ [TinyLlama/TinyLlama-1.1B        ]          │   │
│ │                                              │   │
│ │ Quantization Format:  Access Token:         │   │
│ │ [Q4 ▼]                [••••••••••]          │   │
│ │                                              │   │
│ │ Est. Size: 2.1GB → 270MB (Q4)               │   │
│ │ Memory Req: 1.5GB ✓ Safe for Jetson        │   │
│ │                                              │   │
│ │ [Download Model from HuggingFace]           │   │
│ └─────────────────────────────────────────────┘   │
│                                                     │
│ ┌─────────────────────────────────────────────┐   │
│ │ [Cpu icon] TinyLlama-1.1B                    │   │
│ │ 1.1B params • Q4 quantization • 1.2GB memory│   │
│ │ [Extract Activations] [✓] Edge-Ready        │   │
│ └─────────────────────────────────────────────┘   │
│                                                     │
│ ┌─────────────────────────────────────────────┐   │
│ │ [Cpu icon] Phi-2                             │   │
│ │ 2.7B params • Q4 quantization • 2.1GB memory│   │
│ │ [Extract Activations] [✓] Edge-Ready        │   │
│ └─────────────────────────────────────────────┘   │
│                                                     │
│ ┌─────────────────────────────────────────────┐   │
│ │ [Cpu icon] SmolLM-135M                       │   │
│ │ 135M params • FP16 quantization • 270MB     │   │
│ │ [Progress: 45%]           [⟳] Quantizing    │   │
│ └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

#### Interactive Patterns

1. **Download Form**
   - Repository input: `bg-slate-900 border-slate-700 rounded-lg focus:border-purple-500`
   - Quantization dropdown: Styled select with options Q2, Q4, Q8, FP16, FP32
   - Access token: `type="password"` with password masking
   - Size estimates: Dynamic calculation displayed below inputs
   - Submit button: `bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700`
   - Disabled when repository field empty

2. **Model Cards**
   - Clickable cards: `hover:bg-slate-900/70 transition-colors cursor-pointer`
   - Only "ready" models clickable
   - Card layout: Icon (Cpu) + metadata + action buttons + status
   - Extract Activations button: `bg-purple-600 hover:bg-purple-700` (prominent CTA)

3. **Architecture Viewer Modal**
   - Full-screen modal with close button (X icon top-right)
   - Header: Model name in emerald-400, subtitle with params + quantization
   - Stats grid: 4-column layout with bg-slate-800/50 cards
   - Layer list: Scrollable with hover effects on each layer entry
   - Expandable TransformerBlock entries show attention and MLP sub-components
   - Model config JSON: Collapsible section with syntax highlighting

4. **Extraction Configuration Modal**
   - Modal width: max-w-3xl (wider than architecture viewer)
   - Form sections: Dataset, Layers, Hook Types, Settings
   - Layer selector: Grid layout (6 columns) with L0, L1, L2... buttons
   - Selected layers: `bg-emerald-600` background
   - Unselected layers: `bg-slate-800` background
   - Bulk actions: "Select All" and "Deselect All" buttons (text-xs)
   - Settings: 2-column grid for batch size and max samples inputs

5. **Progress Indicators**
   - Download progress: Full-width bar with gradient fill
   - Quantization progress: Same style as download, includes GPU utilization text
   - Extraction progress: Displayed in modal with samples processed count

6. **Status Indicators** (from lucide-react)
   - Ready: `<CheckCircle className="w-5 h-5 text-emerald-400" />`
   - Downloading: `<Loader className="w-5 h-5 text-blue-400 animate-spin" />`
   - Quantizing: `<Activity className="w-5 h-5 text-yellow-400" />`
   - Error: `<X className="w-5 h-5 text-red-400" />`
   - Model icon: `<Cpu className="w-8 h-8 text-purple-400" />`

#### Responsive Design
- Container: `max-w-7xl mx-auto px-6 py-8` (matches dataset panel)
- Model cards: Full-width on mobile, consistent on all breakpoints
- Architecture viewer: Responsive grid (grid-cols-2 on mobile, grid-cols-4 on desktop)
- Extraction modal: Stack sections vertically on mobile

#### Accessibility Requirements
- **Keyboard Navigation**: All interactive elements Tab/Enter/Esc accessible
- **ARIA Labels**:
  - `<button aria-label="Extract activations from model">` for Extract button
  - `<button aria-label="Close architecture viewer">` for modal close
  - `<select id="model-quantization" aria-describedby="quantization-help">`
- **Focus Indicators**: Purple focus ring (`focus:ring-purple-500`) for model-related inputs
- **Screen Reader Support**:
  - Progress bars include `aria-valuenow`, `aria-valuemin`, `aria-valuemax`
  - Status badges announce state changes
  - Modal dialogs use `role="dialog" aria-modal="true"`
- **Color Contrast**: All text meets WCAG AA (4.5:1 contrast ratio)
- **Tooltips**: Hover tooltips for status badges and action buttons

#### Performance Expectations
- **Initial Load**: Model list renders within 200ms
- **Status Polling**: Updates every 2 seconds without UI flicker
- **Architecture Viewer**: Modal opens within 100ms, layer list renders within 500ms
- **Extraction Modal**: Opens within 100ms, calculations update within 50ms of input change
- **Smooth Animations**: 60fps transitions for progress bars, card hover effects

---

## 5. Data Requirements

### Data Models

#### Model Model (PostgreSQL)
Based on **@0xcc/project-specs/infrastructure/003_SPEC|Postgres_Usecase_Details_and_Guidance.md** (lines 180-218)

```sql
CREATE TABLE models (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'm_abc123'
    name VARCHAR(500) NOT NULL,
    architecture VARCHAR(100) NOT NULL,  -- 'llama', 'gpt2', 'pythia', 'phi', etc.
    params_count BIGINT NOT NULL,  -- Total parameter count
    quantization VARCHAR(50) NOT NULL,  -- 'Q4', 'Q8', 'FP16', 'FP32', 'INT8'
    status VARCHAR(50) NOT NULL DEFAULT 'downloading',
        -- Values: 'downloading', 'loading', 'quantizing', 'ready', 'error'
    progress FLOAT,  -- 0-100
    error_message TEXT,
    file_path VARCHAR(1000),  -- /data/models/raw/{id}/
    quantized_path VARCHAR(1000),  -- /data/models/quantized/{id}/

    -- Architecture details (JSONB for flexibility)
    architecture_config JSONB,  -- { "num_layers": 12, "hidden_size": 768, ... }

    -- Resource requirements
    memory_required_bytes BIGINT,
    disk_size_bytes BIGINT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT models_status_check CHECK (status IN
        ('downloading', 'loading', 'quantizing', 'ready', 'error'))
);
```

#### Model Architecture Configuration (JSONB field)
Stored in `architecture_config` JSONB column:

```typescript
interface ArchitectureConfig {
  model_type: string;  // "gpt2", "llama", "phi", "pythia"
  num_hidden_layers: number;  // 12, 22, 32, etc.
  hidden_size: number;  // 768, 2048, 4096, etc.
  num_attention_heads: number;  // 12, 32, etc.
  intermediate_size: number;  // MLP hidden dimension
  max_position_embeddings: number;  // Max sequence length
  vocab_size: number;  // Vocabulary size
  hidden_act: string;  // "gelu", "silu", etc.
  layer_norm_epsilon: number;  // e.g., 1e-5
  // Architecture-specific fields (optional)
  num_key_value_heads?: number;  // For GQA (Llama-2)
  rope_theta?: number;  // For RoPE embeddings
  attention_dropout?: number;
  hidden_dropout?: number;
}
```

### Data Validation Rules

1. **Repository ID Validation**
   - Format: `^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$`
   - Example: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   - Cannot be empty or whitespace-only
   - Max length: 500 characters

2. **Quantization Format Validation**
   - Must be one of: Q2, Q4, Q8, FP16, FP32
   - Case-insensitive (normalize to uppercase)

3. **Access Token Validation**
   - Optional field (can be empty)
   - Format: `^hf_[a-zA-Z0-9]{20,}$`
   - Not stored in database

4. **File Path Validation**
   - Must be absolute path starting with `/data/models/`
   - Prevent path traversal: No `..` sequences
   - Max length: 1000 characters

5. **Parameter Count Validation**
   - Integer >= 0
   - Display human-readable: 135M, 1.1B, 2.7B, 13B
   - Calculation: params_count / 1e9 (billions) or / 1e6 (millions)

6. **Memory Requirement Validation**
   - Integer >= 0 (bytes)
   - Display human-readable: 270MB, 1.2GB, 2.1GB
   - Calculation depends on quantization:
     - FP32: params_count × 4 bytes × 1.2 (overhead)
     - FP16: params_count × 2 bytes × 1.2
     - Q8: params_count × 1 byte × 1.2
     - Q4: params_count × 0.5 bytes × 1.2
     - Q2: params_count × 0.25 bytes × 1.2

7. **Status Validation**
   - Must be one of: downloading, loading, quantizing, ready, error
   - Status transitions must follow state machine

### Data Persistence

#### File Storage Structure
Based on **@0xcc/project-specs/infrastructure/001_SPEC|Folder_File_Details.md** (lines 670-694)

```
/data/models/
├── raw/                    # Original model weights
│   ├── model_m_abc123/
│   │   ├── pytorch_model.bin  or  model.safetensors
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── special_tokens_map.json
│   │
│   └── model_m_xyz456/
│       └── ...
│
├── quantized/              # Quantized model variants
│   ├── model_m_abc123_q4/
│   │   ├── model.safetensors  (quantized weights)
│   │   ├── config.json  (same as raw)
│   │   └── quantization_config.json  (quantization metadata)
│   │
│   └── model_m_xyz456_fp16/
│       └── ...
│
└── metadata/               # Model metadata
    ├── m_abc123_architecture.json
    └── m_xyz456_architecture.json
```

#### Activation Storage Structure

```
/data/activations/
├── extraction_ext_abc123/
│   ├── model_info.json     # Model and dataset references
│   ├── metadata.json       # Extraction parameters
│   ├── statistics.json     # Activation statistics
│   │
│   ├── layer_0_residual.npy    # Shape: [n_samples, seq_len, hidden_dim]
│   ├── layer_0_mlp.npy
│   ├── layer_4_residual.npy
│   ├── layer_4_mlp.npy
│   ├── layer_8_residual.npy
│   └── ...
│
└── extraction_ext_xyz456/
    └── ...
```

#### Database Indexes
```sql
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_architecture ON models(architecture);
CREATE INDEX idx_models_created_at ON models(created_at DESC);

-- Full-text search index
CREATE INDEX idx_models_search ON models
    USING GIN(to_tsvector('english', name || ' ' || architecture));

-- JSONB index for architecture queries
CREATE INDEX idx_models_architecture_config ON models USING GIN(architecture_config);
```

### Data Migration Considerations
- **Initial Migration**: Create `models` table with all columns and constraints
- **V2 Migration** (if needed): Add `architecture_config` JSONB column for detailed architecture data
- **V3 Migration** (if needed): Add `memory_required_bytes` column for memory tracking
- **Backward Compatibility**: Old models without architecture_config display "Unknown" in viewer

---

## 6. Technical Constraints

### Technology Stack Constraints (from ADR)

#### Backend Framework
- **FastAPI**: Async API endpoints for model operations
- **Pydantic**: Data validation and serialization
- **SQLAlchemy**: ORM for database operations
- **Alembic**: Database migrations

#### ML/AI Libraries
- **PyTorch**: Model loading and inference
- **HuggingFace Transformers**: Download models, load tokenizers, model architectures
- **bitsandbytes**: 4-bit and 8-bit quantization
- **safetensors**: Efficient model weight storage format
- **accelerate**: GPU memory optimization and mixed precision

#### Job Queue
- **Celery**: Background processing for downloads, quantization, extraction
- **Redis**: Celery broker and result backend

#### Storage
- **Local Filesystem**: `/data/models/` for model weights
- **PostgreSQL**: Metadata storage with JSONB for architecture

### ADR Decision References

#### Decision 1: Backend Framework (FastAPI)
- **Constraint**: All model operations must be async-compatible
- **Impact**: Use `async def` for download/quantization endpoints

#### Decision 3: Database (PostgreSQL with JSONB)
- **Constraint**: Architecture metadata stored as JSONB, not separate columns
- **Impact**: Query architecture using PostgreSQL JSON operators

#### Decision 6: Job Queue (Celery + Redis)
- **Constraint**: Downloads, quantization, and extraction must be Celery tasks
- **Impact**: API returns 202 Accepted with job_id, client polls for status

#### Decision 7: ML/AI Stack (PyTorch + HuggingFace)
- **Constraint**: Must use HuggingFace Transformers for model loading
- **Impact**: Limited to architectures supported by HuggingFace (GPT-2, Llama, Phi, Pythia, etc.)

#### Decision 8: Storage (Local Filesystem)
- **Constraint**: All models stored in `/data/models/` directory
- **Impact**: Disk space checks required before downloads

#### Decision 11: Edge Optimization (Jetson Orin Nano)
- **Constraints**:
  - Max model memory: 6GB (leave 2GB for OS and application)
  - Quantization required for models >2B parameters
  - GPU memory monitoring during quantization
  - Efficient activation storage (memory-mapped NumPy arrays)
- **Impact**: Automatic fallback to higher precision on OOM

### Performance Requirements

#### Response Times (p95)
- **GET /api/models**: <100ms (list all models)
- **POST /api/models/download**: <500ms (start background job, return job_id)
- **GET /api/models/:id**: <100ms (get single model metadata)
- **GET /api/models/:id/architecture**: <100ms (cached architecture from database)
- **POST /api/models/:id/extract**: <500ms (queue extraction job)

#### Throughput
- **Concurrent Downloads**: 1 (network and disk bandwidth limitation)
- **Concurrent Quantizations**: 1 (GPU limitation)
- **Concurrent Extractions**: 1 (GPU memory limitation)
- **API Requests**: 100/sec (nginx + FastAPI on Jetson)

#### Storage
- **Model Sizes**: 270MB - 5GB each (quantized)
- **Total Model Storage Budget**: ~200GB (user-managed)
- **Activation Storage**: 10GB - 100GB per extraction
- **Database Metadata**: ~2KB per model (<2MB for 1000 models)

#### Memory
- **Model Loading**: 270MB - 6GB depending on quantization
- **Quantization Overhead**: +2GB temporary (for FP32 → Q4 conversion)
- **Extraction Overhead**: +1GB for activation buffers
- **Available RAM**: 8GB total, budget 6GB for ML operations

#### Network
- **Download Speed**: 10-100 Mbps typical on edge
- **Resume Capability**: HTTP Range requests for interrupted downloads
- **Bandwidth Warning**: If download exceeds 5GB

### Security Requirements

#### Input Validation
- Sanitize repository IDs (prevent command injection)
- Validate file paths (prevent path traversal)
- Validate quantization formats (whitelist only)
- Rate limit download requests (10/hour per endpoint)

#### Authentication
- HuggingFace access tokens:
  - Never stored in database
  - Masked in UI (password input)
  - Passed only to HuggingFace API via HTTPS

#### File System Security
- Restrict operations to `/data/models/` directory
- Validate model files before loading (check magic bytes)
- Set appropriate file permissions (chmod 644)
- Prevent overwriting existing models without confirmation

#### Model Loading Security
- Load models in isolated process (Celery worker)
- Use `torch.load` with `weights_only=True` to prevent arbitrary code execution
- Validate model architecture before loading (check config.json)
- Timeout for hung model loads (5 minutes)

---

## 7. API/Integration Specifications

### Backend API Endpoints

Based on **@0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx** (lines 336-355)

#### Endpoint 1: List Models
```http
GET /api/models
```

**Query Parameters:**
- `page` (optional, integer, default=1): Page number
- `limit` (optional, integer, default=50): Items per page
- `status` (optional, string): Filter by status
- `architecture` (optional, string): Filter by architecture (gpt2, llama, etc.)
- `search` (optional, string): Full-text search query

**Response (200 OK):**
```json
{
  "data": [
    {
      "id": "m_tinyllama_20251005",
      "name": "TinyLlama-1.1B-Chat-v1.0",
      "params": "1.1B",
      "params_count": 1100000000,
      "quantized": "Q4",
      "memReq": "1.2GB",
      "memory_required_bytes": 1288490188,
      "status": "ready",
      "progress": null,
      "architecture": "llama",
      "disk_size_bytes": 687194767,
      "created_at": "2025-10-05T10:00:00Z",
      "updated_at": "2025-10-05T10:03:30Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 3,
    "has_next": false,
    "has_prev": false
  }
}
```

---

#### Endpoint 2: Get Model Details with Architecture
```http
GET /api/models/:id
```

**Path Parameters:**
- `id` (required, string): Model ID

**Response (200 OK):**
```json
{
  "data": {
    "id": "m_tinyllama_20251005",
    "name": "TinyLlama-1.1B-Chat-v1.0",
    "architecture": "llama",
    "params_count": 1100000000,
    "quantization": "Q4",
    "status": "ready",
    "file_path": "/data/models/raw/m_tinyllama_20251005/",
    "quantized_path": "/data/models/quantized/m_tinyllama_20251005_q4/",
    "memory_required_bytes": 1288490188,
    "disk_size_bytes": 687194767,
    "architecture_config": {
      "model_type": "llama",
      "num_hidden_layers": 22,
      "hidden_size": 2048,
      "num_attention_heads": 32,
      "num_key_value_heads": 4,
      "intermediate_size": 5632,
      "max_position_embeddings": 2048,
      "vocab_size": 32000,
      "hidden_act": "silu",
      "layer_norm_epsilon": 1e-5,
      "rope_theta": 10000
    },
    "created_at": "2025-10-05T10:00:00Z",
    "updated_at": "2025-10-05T10:03:30Z"
  }
}
```

---

#### Endpoint 3: Download Model from HuggingFace
```http
POST /api/models/download
```

**Request Body:**
```json
{
  "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "quantization": "Q4",
  "access_token": "hf_xxxxxxxxxxxxxxxxxxxx"  // optional
}
```

**Request Validation:**
- `repo_id`: Required, string, format `org/model-name`
- `quantization`: Required, string, enum ["Q2", "Q4", "Q8", "FP16", "FP32"]
- `access_token`: Optional, string, format `hf_[a-zA-Z0-9]{20,}`

**Response (202 Accepted):**
```json
{
  "data": {
    "model_id": "m_tinyllama_20251005",
    "status": "downloading",
    "message": "Model download started"
  },
  "meta": {
    "job_id": "job_model_abc123",
    "estimated_duration_seconds": 180,
    "estimated_size_bytes": 2147483648,
    "quantized_size_bytes": 268435456
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid repo_id or quantization format
- `404 Not Found`: HuggingFace repository does not exist
- `403 Forbidden`: Access token required for gated model
- `409 Conflict`: Model already exists
- `429 Rate Limit Exceeded`: Too many download requests
- `503 Service Unavailable`: Insufficient disk space

---

#### Endpoint 4: Start Activation Extraction
```http
POST /api/models/:id/extract
```

**Path Parameters:**
- `id` (required, string): Model ID

**Request Body:**
```json
{
  "dataset_id": "ds_openwebtext_20251005",
  "layers": [0, 5, 10, 15, 20],
  "hook_types": ["residual", "mlp"],
  "max_samples": 10000,
  "batch_size": 32,
  "top_k_examples": 100
}
```

**Request Validation:**
- `dataset_id`: Required, string, must reference existing ready dataset
- `layers`: Required, array of integers, at least 1 layer
- `hook_types`: Required, array of strings, enum ["residual", "mlp", "attention"]
- `max_samples`: Optional, integer, null = all samples
- `batch_size`: Optional, integer, default 32, must be power of 2
- `top_k_examples`: Optional, integer, default 100, range 10-1000

**Response (202 Accepted):**
```json
{
  "data": {
    "extraction_id": "ext_abc123",
    "status": "initializing",
    "message": "Activation extraction started"
  },
  "meta": {
    "job_id": "job_extract_xyz789",
    "estimated_duration_seconds": 300,
    "estimated_storage_bytes": 53687091200
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid extraction configuration
- `404 Not Found`: Model or dataset ID does not exist
- `409 Conflict`: Extraction already in progress for this model
- `503 Service Unavailable`: GPU busy or insufficient disk space

---

#### Endpoint 5: Delete Model
```http
DELETE /api/models/:id
```

**Path Parameters:**
- `id` (required, string): Model ID

**Response (200 OK):**
```json
{
  "data": {
    "message": "Model deleted successfully",
    "model_id": "m_old_model_123",
    "freed_bytes": 2684354560
  }
}
```

**Error Responses:**
- `404 Not Found`: Model ID does not exist
- `409 Conflict`: Model referenced by active training
- `500 Internal Server Error`: File deletion failed

---

### Internal Service Integration

#### HuggingFace Transformers Model Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Load configuration first (lightweight)
config = AutoConfig.from_pretrained(
    repo_id,
    cache_dir=f"/data/models/raw/{model_id}",
    token=access_token
)

# Download and load model
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    config=config,
    cache_dir=f"/data/models/raw/{model_id}",
    token=access_token,
    torch_dtype=torch.float32,  # Will quantize separately
    device_map="cpu"  # Load to CPU first
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    repo_id,
    cache_dir=f"/data/models/raw/{model_id}",
    token=access_token
)
```

#### Model Quantization with bitsandbytes
```python
import torch
from transformers import BitsAndBytesConfig

# 4-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # Normal Float 4-bit
)

# Quantize model
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="cuda:0"  # Load directly to GPU
)

# Save quantized model
quantized_model.save_pretrained(
    f"/data/models/quantized/{model_id}_q4/",
    safe_serialization=True  # Use safetensors format
)
```

#### Activation Extraction with PyTorch Hooks
```python
import torch
import numpy as np

# Storage for captured activations
activations = {}

def create_hook(layer_name):
    def hook_fn(module, input, output):
        # Store activation (detach from computation graph)
        activations[layer_name] = output.detach().cpu().numpy()
    return hook_fn

# Register hooks on model layers
model.eval()
hooks = []

for idx in selected_layers:
    if "residual" in hook_types:
        layer = model.model.layers[idx]
        hook = layer.register_forward_hook(create_hook(f"layer_{idx}_residual"))
        hooks.append(hook)

    if "mlp" in hook_types:
        layer = model.model.layers[idx].mlp
        hook = layer.register_forward_hook(create_hook(f"layer_{idx}_mlp"))
        hooks.append(hook)

# Run forward pass with hooks
with torch.no_grad():
    for batch in dataloader:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        outputs = model(**inputs)

        # Save activations to disk
        for layer_name, activation in activations.items():
            np.save(f"{output_dir}/{layer_name}.npy", activation)

# Remove hooks
for hook in hooks:
    hook.remove()
```

---

## 8. Non-Functional Requirements

### NFR-1: Performance

#### Load Time
- Model list render: <200ms from API request to UI display
- Architecture viewer modal: <100ms open time, <500ms layer list render
- Extraction modal: <100ms open time, <50ms calculation updates

#### Throughput
- API: 100 requests/second on Jetson
- Concurrent downloads: 1 (enforced by job queue)
- Concurrent quantizations: 1 (GPU limitation)
- Concurrent extractions: 1 (GPU memory limitation)

#### Scalability
- Support 100+ models without performance degradation
- Architecture viewer handles models with 100+ layers
- Extraction handles datasets with 1M+ samples

### NFR-2: Reliability

#### Availability
- Model API endpoints: 99.5% uptime
- Graceful degradation: If GPU unavailable, queue operations for later

#### Error Handling
- Structured error responses (code, message, details)
- Retry logic for transient errors (exponential backoff, max 3 retries)
- Resume capability for interrupted downloads
- Automatic fallback to higher precision on OOM

#### Data Integrity
- Checksum validation for downloaded models
- Database foreign key constraints prevent orphaned records
- Atomic file operations (write to temp, then rename)

### NFR-3: Usability

#### Learnability
- First-time users can download model in <3 minutes without docs
- Architecture viewer immediately understandable for ML researchers
- Extraction configuration has sensible defaults

#### Efficiency
- Keyboard navigation for power users
- Bulk actions (select all layers, deselect all)
- Recently used models appear at top

#### Error Prevention
- Confirmation modal for destructive actions
- Memory warnings before downloads
- Disk space checks before operations
- Clear status indicators

### NFR-4: Maintainability

#### Code Organization
- `ModelService`, `ModelRepository`, `ModelSchema` (ADR naming)
- All model logic in `backend/app/services/model_service.py`
- API endpoints in `backend/app/api/v1/endpoints/models.py`
- Frontend components in `frontend/src/components/models/`

#### Testing
- Unit test coverage: ≥80% for business logic
- Integration tests: All API endpoints
- E2E tests: Complete download → quantization → extraction flow

#### Logging
- Structured JSON logs
- Log levels: DEBUG (file ops), INFO (status changes), ERROR (failures)
- Include: model_id, operation, duration, GPU usage, error details

#### Documentation
- API endpoints in OpenAPI spec
- Code comments for quantization algorithms
- README for model architecture support

---

## 9. Feature Boundaries (Non-Goals)

### Explicitly Out of Scope

1. **Model Fine-Tuning**
   - **Not Included**: Fine-tune models on custom datasets
   - **Rationale**: Focus on interpretability, not training; requires separate infrastructure
   - **Future Consideration**: Phase 4 (fine-tuning support)

2. **Model Merging/Ensembling**
   - **Not Included**: Combine multiple models or create model ensembles
   - **Rationale**: Adds complexity; research use case is rare on edge devices

3. **Custom Architecture Training**
   - **Not Included**: Train models from scratch or define custom architectures
   - **Rationale**: Computationally prohibitive on Jetson; focus on pre-trained models

4. **Multi-Modal Models**
   - **Not Included**: Vision models (CLIP, LLaVA) or audio models (Whisper)
   - **Rationale**: Text-only focus for MVP; vision support in Phase 4

5. **Model Serving/Inference API**
   - **Not Included**: Expose models as REST API for general inference
   - **Rationale**: miStudio is for interpretability research, not production serving

6. **Model Version Control**
   - **Not Included**: Track multiple versions of same model with diffs
   - **Rationale**: Users manage versions manually via HuggingFace

7. **Distributed Model Loading**
   - **Not Included**: Load model across multiple GPUs or devices
   - **Rationale**: Single Jetson device constraint

8. **Model Benchmarking**
   - **Not Included**: Automated benchmarks (perplexity, accuracy, speed)
   - **Rationale**: Focus on interpretability, not performance evaluation

9. **GGUF/llama.cpp Integration**
   - **Not Included**: Support GGUF format or llama.cpp quantization
   - **Rationale**: PyTorch + bitsandbytes sufficient for MVP; consider Phase 3

10. **Model Export to Other Formats**
    - **Not Included**: Export to ONNX, TensorRT, CoreML
    - **Rationale**: miStudio operates on PyTorch models exclusively

### Related Features Handled Separately

1. **Model Tokenizers**
   - Tokenizers downloaded with models (automatic)
   - Used by Dataset Management for tokenization
   - Model Management provides tokenizers, Dataset Management consumes them

2. **SAE Training on Activations**
   - Handled by Training feature (003_FPRD|SAE_Training.md)
   - Model Management provides activations, Training feature trains SAEs

3. **Feature Discovery on Trained SAEs**
   - Handled by Feature Discovery feature (004_FPRD|Feature_Discovery.md)
   - Model Management enables activation extraction, Feature Discovery analyzes SAEs

4. **Model Steering**
   - Handled by Steering feature (005_FPRD|Model_Steering.md)
   - Model Management provides models, Steering feature applies interventions

### Technical Limitations Accepted

1. **No Streaming Model Loading**: Entire model loaded into memory (no progressive loading)
2. **No Dynamic Quantization**: Quantization format fixed at download time (re-quantization creates new model)
3. **No Partial Activation Extraction**: All selected layers extracted in single pass (no incremental extraction)
4. **Fixed Hook Points**: Only residual, MLP, attention supported (no custom hook locations)

---

## 10. Dependencies

### Feature Dependencies

#### Hard Dependencies (Blockers)
1. **Database Schema**: PostgreSQL `models` table must exist
   - **Owner**: Database Migration (001_initial_schema.py)
   - **Status**: Pending
   - **Impact**: Cannot store model metadata without schema

2. **File Storage Structure**: `/data/models/` directory structure
   - **Owner**: Infrastructure Setup (docker-compose.yml volume mounts)
   - **Status**: Pending
   - **Impact**: Cannot store model files

3. **Celery Worker with GPU Access**: Worker must have CUDA access
   - **Owner**: Deployment Configuration (docker-compose.yml celery service)
   - **Status**: Pending
   - **Impact**: Quantization and extraction will fail

4. **Redis**: Celery broker
   - **Owner**: Infrastructure Setup (docker-compose.yml redis service)
   - **Status**: Pending
   - **Impact**: Cannot queue background jobs

#### Soft Dependencies (Recommended)
1. **Dataset Management**: Activation extraction depends on datasets
   - **Owner**: 001_FPRD|Dataset_Management feature
   - **Status**: Complete
   - **Impact**: Extraction feature disabled until datasets available; rest of Model Management works independently

2. **Training Configuration**: Training consumes models from this feature
   - **Owner**: 003_FPRD|SAE_Training feature
   - **Status**: Pending
   - **Impact**: No impact on Model Management; one-way dependency

### External Service Dependencies

#### HuggingFace Hub
- **Service**: HuggingFace API (https://huggingface.co)
- **Purpose**: Download models and validate repository IDs
- **Availability**: 99.9% uptime
- **Failure Handling**: Retry with exponential backoff; fallback to error status
- **Rate Limits**: 1000 requests/hour (authenticated)
- **Authentication**: Optional access token for gated models

### Library Dependencies

```python
# Backend requirements.txt
fastapi>=0.104.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
celery>=5.3.0
redis>=5.0.0
torch>=2.0.0
transformers>=4.30.0
bitsandbytes>=0.41.0
safetensors>=0.3.0
accelerate>=0.20.0
sentencepiece>=0.1.99  # For tokenizers
numpy>=1.24.0
```

```json
// Frontend package.json
{
  "dependencies": {
    "react": "^18.2.0",
    "typescript": "^5.0.0",
    "zustand": "^4.4.0",
    "lucide-react": "^0.292.0",
    "recharts": "^2.9.0",
    "axios": "^1.6.0"
  }
}
```

### Infrastructure Dependencies

1. **CUDA Toolkit**: Required for GPU quantization and extraction
   - Version: 11.8 or 12.1 (compatible with PyTorch 2.0)
   - Jetson comes with CUDA pre-installed

2. **cuDNN**: Deep learning primitives
   - Version: 8.x (compatible with CUDA version)

3. **TensorRT** (optional): Inference optimization
   - Version: 8.6 for Jetson Orin Nano

### Timeline Dependencies

#### Phase 1 (MVP - Week 1-2)
- Database schema created
- Celery worker with GPU access deployed
- HuggingFace download implemented and tested

#### Phase 2 (Enhancement - Week 3-4)
- Dataset Management complete (for extraction feature)
- Quantization tested with all formats (Q2, Q4, Q8, FP16)
- Architecture viewer fully functional

#### Phase 3 (Polish - Week 5-6)
- Activation extraction working end-to-end
- Performance optimization (caching, memory management)
- Error recovery (OOM handling, resume downloads)

---

## 11. Success Criteria

### Quantitative Success Metrics

#### Performance Metrics
1. **Download Throughput**
   - Target: Download 2GB model in <15 minutes on 10 Mbps connection
   - Measurement: Log download start/end times

2. **Quantization Speed**
   - Target: Q4 quantization of 1.1B model in <60 seconds on Jetson GPU
   - Measurement: Log quantization duration

3. **Extraction Speed**
   - Target: Extract activations from 10K samples in <5 minutes (5 layers, 2 hooks)
   - Measurement: Log extraction duration

4. **API Response Times (p95)**
   - Target: <100ms for GET endpoints, <500ms for POST endpoints
   - Measurement: FastAPI middleware logging

#### Reliability Metrics
1. **Download Success Rate**
   - Target: >95% of downloads complete successfully
   - Measurement: Log outcomes

2. **Quantization Success Rate**
   - Target: >90% quantizations succeed (some OOM expected for large models)
   - Measurement: Track quantization attempts vs successes

3. **Extraction Success Rate**
   - Target: >95% extractions complete successfully
   - Measurement: Track extraction job outcomes

#### Usability Metrics
1. **Time to First Model**
   - Target: <5 minutes (from UI open to "ready" status for small model)
   - Measurement: User testing sessions

2. **Architecture Viewer Usage**
   - Target: >50% of users open architecture viewer before extraction
   - Measurement: Log viewer opens vs extractions started

### Qualitative Success Indicators

#### User Satisfaction
1. **Ease of Use**: Users download and quantize models without documentation
2. **Error Clarity**: Users understand OOM errors and choose appropriate quantization
3. **Visual Polish**: UI matches Mock UI reference exactly

#### Developer Feedback
1. **Code Quality**: Code review approval without major revisions
2. **Test Coverage**: ≥80% coverage
3. **Documentation**: All endpoints in OpenAPI spec

### Business Impact Measurements

#### Adoption Metrics
1. **Model Usage**: Average 3 models downloaded per user within first week
2. **Quantization Formats**: >70% of users select Q4 (recommended format)
3. **Extraction Usage**: >60% of users extract activations within first week

---

## 12. Testing Requirements

### Unit Testing

#### Backend Unit Tests (Pytest)

**Test Suite 1: Model Service**
- `test_create_model_record()`
- `test_update_model_status()`
- `test_compute_memory_requirements()`
- `test_validate_quantization_format()`
- `test_parse_architecture_config()`

**Test Suite 2: Model Repository**
- `test_get_model_by_id()`
- `test_list_models_with_filters()`
- `test_delete_model_with_trainings()`

**Test Suite 3: Model Download Task**
- `test_download_task_success()`
- `test_download_task_oom_fallback()`
- `test_download_task_resume()`

**Test Suite 4: Activation Extraction**
- `test_register_forward_hooks()`
- `test_extract_activations_batch()`
- `test_save_activations_to_disk()`

**Coverage Target**: ≥80%

#### Frontend Unit Tests (Vitest)

**Test Suite 1: ModelsPanel Component**
- `test_render_model_list()`
- `test_download_form_submission()`
- `test_quantization_selector()`

**Test Suite 2: ModelArchitectureViewer**
- `test_display_architecture_overview()`
- `test_layer_list_render()`
- `test_expandable_transformer_blocks()`

**Test Suite 3: ActivationExtractionConfig**
- `test_layer_selection()`
- `test_hook_type_selection()`
- `test_extraction_validation()`

**Coverage Target**: ≥70%

### Integration Testing

**Test Scenario 1: Download and Quantize**
1. POST /api/models/download with Q4 format
2. Poll status until "ready"
3. Verify model files exist
4. Verify quantization metadata in database

**Test Scenario 2: Extract Activations**
1. POST /api/models/:id/extract
2. Poll status until "completed"
3. Verify activation files exist (.npy)
4. Verify activation shapes correct

**Test Scenario 3: Delete Model**
1. Create model
2. DELETE /api/models/:id
3. Verify files deleted
4. Verify database record removed

### User Acceptance Testing

**UAT Scenario 1: First Model Download**
- User downloads TinyLlama with Q4 quantization
- Model ready in <5 minutes
- User opens architecture viewer

**UAT Scenario 2: Extract Activations**
- User configures extraction (5 layers, residual + MLP)
- Extraction completes in <5 minutes for 10K samples
- User navigates to training with extracted activations

---

## 13. Implementation Considerations

### Complexity Assessment

#### High Complexity Components (8/10)
1. **Activation Extraction with PyTorch Hooks**
   - **Challenges**: Complex hook registration, memory management, multi-layer coordination
   - **Mitigation**: Reference existing SAE libraries (SAELens, TransformerLens)
   - **Effort**: 20 hours (2.5 days)

2. **Quantization with OOM Handling**
   - **Challenges**: bitsandbytes integration, GPU memory monitoring, fallback logic
   - **Mitigation**: Test with multiple model sizes, implement robust error handling
   - **Effort**: 16 hours (2 days)

#### Medium Complexity Components (6/10)
1. **Architecture Viewer**
   - **Challenges**: Parse diverse config formats, render nested layers
   - **Mitigation**: Use mock data for testing, handle unknown architectures gracefully
   - **Effort**: 12 hours (1.5 days)

2. **Model Download with Resume**
   - **Challenges**: HTTP Range requests, partial file management
   - **Mitigation**: Use HuggingFace Transformers built-in resume support
   - **Effort**: 8 hours (1 day)

### Risk Factors

**Risk 1: GPU OOM During Quantization**
- **Probability**: High (60%)
- **Mitigation**: Automatic fallback to higher precision, clear error messages

**Risk 2: Activation Extraction Memory Overflow**
- **Probability**: Medium (40%)
- **Mitigation**: Streaming extraction, batch size auto-tuning

**Risk 3: Unsupported Model Architecture**
- **Probability**: Medium (30%)
- **Mitigation**: Validate architecture before download, display compatibility warnings

### Recommended Implementation Approach

#### Phase 1: Core Download (Week 1)
1. Database schema + API endpoints
2. HuggingFace download with Celery
3. Basic quantization (FP16, Q4)

#### Phase 2: Architecture & Extraction (Week 2)
1. Architecture viewer modal
2. Activation extraction configuration
3. PyTorch hooks implementation

#### Phase 3: Polish (Week 3)
1. Error handling and OOM recovery
2. Model deletion
3. Performance optimization

---

## 14. Open Questions

1. **Q: Should we support GGUF format (llama.cpp)?**
   - **Impact**: Enables compatibility with llama.cpp ecosystem
   - **Options**: A) Yes, add GGUF support B) No, PyTorch only (MVP)
   - **Decision Needed**: Week 1
   - **Recommendation**: Option B (defer to Phase 3)

2. **Q: Should extraction save activations in HDF5 instead of NumPy?**
   - **Impact**: HDF5 more efficient for large arrays, but adds dependency
   - **Options**: A) HDF5 B) NumPy (simpler)
   - **Decision Needed**: Week 2
   - **Recommendation**: Option B for MVP

3. **Q: Should we auto-delete models after extraction completes?**
   - **Impact**: Saves storage but may surprise users
   - **Options**: A) Auto-delete B) Keep all models C) Prompt user
   - **Decision Needed**: Week 2
   - **Recommendation**: Option B (user manages manually)

---

**Document Version:** 1.0
**Status:** Draft
**Next Review:** After stakeholder feedback
**Related Documents:**
- @0xcc/prds/000_PPRD|miStudio.md (Project PRD)
- @0xcc/prds/001_FPRD|Dataset_Management.md (Dataset Management PRD)
- @0xcc/adrs/000_PADR|miStudio.md (Architecture Decision Record)
- @0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx (PRIMARY UI/UX Reference)
- @0xcc/project-specs/infrastructure/001_SPEC|Folder_File_Details.md (File Structure)
- @0xcc/project-specs/infrastructure/003_SPEC|Postgres_Usecase_Details_and_Guidance.md (Database Schema)
