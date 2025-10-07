# Feature PRD: SAE Training

**Document ID:** 003_FPRD|SAE_Training
**Feature:** Sparse Autoencoder (SAE) Training Configuration and Execution
**Status:** Draft
**Created:** 2025-10-06
**Last Updated:** 2025-10-06
**Owner:** miStudio Development Team

---

## 1. Feature Overview

### Purpose
Enable users to configure, launch, and monitor Sparse Autoencoder (SAE) training jobs that extract interpretable features from language model activations. This feature bridges the gap between raw model activations (extracted via Model Management) and interpretable features (analyzed in Feature Discovery), forming the core of miStudio's mechanistic interpretability workflow.

### Business Value
- **Primary Value:** Transforms uninterpretable model activations into structured, interpretable feature representations
- **Research Value:** Enables discovery of specific circuits and features within language models
- **Educational Value:** Provides transparent, real-time visibility into SAE training dynamics
- **Efficiency Value:** Optimized for edge deployment on Jetson Orin Nano with memory-aware batch sizing

### Success Criteria
- Users can successfully configure and launch SAE training with appropriate hyperparameters
- Training jobs execute reliably with progress tracking and real-time metrics visualization
- Checkpoint system enables pause/resume functionality without data loss
- Edge optimization keeps GPU memory usage within 6GB limits during training
- Training completes with <5% dead neurons and L0 sparsity within target range (30-70)

### Target Users
- **ML Researchers:** Investigating model interpretability through SAE methods
- **Students:** Learning about sparse autoencoders and mechanistic interpretability
- **Practitioners:** Building interpretable AI systems with feature-level understanding

---

## 2. Feature Context

### Relationship to Project PRD
This feature implements **Core Feature #3: SAE Training** from the Project PRD (000_PPRD|miStudio.md), specifically:
- Priority: **P0 (Critical MVP feature)**
- Dependencies: Dataset Management (for training data), Model Management (for activation extraction)
- Enables: Feature Discovery (requires trained SAEs), Model Steering (requires interpretable features)

### Integration with Existing Features

**Upstream Dependencies:**
- **Dataset Management (001_FPRD):** Provides tokenized datasets for activation extraction during training
- **Model Management (002_FPRD):** Provides models and activation extraction capabilities needed for SAE training input

**Downstream Consumers:**
- **Feature Discovery (004_FPRD - Pending):** Analyzes features from trained SAE checkpoints
- **Model Steering (005_FPRD - Pending):** Uses SAE features for model behavior modification

### UI/UX Reference
**PRIMARY REFERENCE:** `@0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
- **TrainingPanel Component:** Lines 1628-1842 (configuration, hyperparameters, job initiation)
- **TrainingCard Component:** Lines 1845-2156 (progress tracking, metrics, checkpoints, controls)
- **Advanced Hyperparameters Section:** Lines 1704-1814 (collapsible advanced configuration)
- **Checkpoint Management UI:** Lines 1941-2027 (checkpoint save/load/delete, auto-save)
- **Live Metrics Visualization:** Lines 2031-2086 (loss curves, L0 sparsity, training logs)

---

## 3. User Stories & Scenarios

### US-1: Configure SAE Training Job (Priority: P0)
**As a** ML researcher
**I want to** configure a new SAE training job with appropriate hyperparameters
**So that** I can extract interpretable features from my language model

**Acceptance Criteria:**
- Given I'm on the Training panel, I can select a ready dataset and model from dropdowns
- When I select an encoder type (Sparse/Skip/Transcoder), appropriate default hyperparameters load
- When I expand "Advanced Hyperparameters", I can configure: learning rate, batch size, L1 coefficient, expansion factor, training steps, optimizer, LR schedule, ghost gradient penalty toggle
- When I click "Start Training" with valid configuration, a training job is created and begins execution
- When dataset or model is not selected, "Start Training" button is disabled
- Given the training is configured, I can see configuration summary before starting

**UI Reference:** Mock-embedded-interp-ui.tsx TrainingPanel (lines 1628-1825)

---

### US-2: Monitor Training Progress (Priority: P0)
**As a** user with an active training job
**I want to** view real-time training progress and key metrics
**So that** I can assess training quality and convergence

**Acceptance Criteria:**
- Given a training job is running, I see a progress bar showing completion percentage (0-100%)
- When training progresses, I see live updates for: Loss, L0 Sparsity, Dead Neurons count, GPU Utilization percentage
- When I click "Show Live Metrics", I see visualizations for: Loss curve (last 20 steps), L0 Sparsity curve (last 20 steps), training logs with timestamped entries
- When training enters different states, the status badge updates: "initializing" (blue spinner), "training" (green pulse), "paused" (yellow), "completed" (green checkmark)
- Given training is active, metrics update automatically every 1-2 seconds

**UI Reference:** Mock-embedded-interp-ui.tsx TrainingCard (lines 1845-2088)

---

### US-3: Manage Training Lifecycle (Priority: P0)
**As a** user with a training job
**I want to** pause, resume, stop, or retry training
**So that** I can manage resource allocation and handle errors

**Acceptance Criteria:**
- When training status is "training", I can click "Pause" to suspend training (saves checkpoint automatically)
- When training status is "paused", I can click "Resume" to continue from last checkpoint
- When training status is "training" or "paused", I can click "Stop" to terminate training permanently
- When training status is "stopped" or "error", I can click "Retry" to restart training from beginning
- When I pause training, current progress and metrics are preserved
- When I resume training, training continues from exact step where it was paused

**UI Reference:** Mock-embedded-interp-ui.tsx TrainingCard controls (lines 2090-2153)

---

### US-4: Save and Load Training Checkpoints (Priority: P1)
**As a** ML researcher
**I want to** save intermediate training checkpoints and load them later
**So that** I can resume training from specific points or analyze intermediate models

**Acceptance Criteria:**
- When training is active, I can click "Checkpoints (N)" to view checkpoint management panel
- When I click "Save Now", a new checkpoint is created at current training step
- Given checkpoints exist, I see a list showing: Step number, Loss value, Timestamp, Load/Delete actions
- When I click "Load" on a checkpoint, training resumes from that checkpoint's step
- When I click "Delete" on a checkpoint, the checkpoint file is removed (with confirmation)
- When I enable "Auto-save every N steps" toggle, checkpoints are saved automatically at specified intervals
- When I configure auto-save interval, I can set values from 100-10000 steps

**UI Reference:** Mock-embedded-interp-ui.tsx Checkpoint Management (lines 1941-2027)

---

### US-5: Configure Advanced Hyperparameters (Priority: P1)
**As an** advanced ML researcher
**I want to** fine-tune SAE training hyperparameters
**So that** I can optimize training for my specific use case

**Acceptance Criteria:**
- When I click "Advanced Hyperparameters", the section expands to show all configurable parameters
- Given advanced parameters are visible, I can configure:
  - Learning Rate (number input, step 0.00001, typical range 0.0001-0.01)
  - Batch Size (dropdown: 64, 128, 256, 512, 1024, 2048)
  - L1 Coefficient (number input, step 0.00001, typical range 0.00001-0.001)
  - Expansion Factor (dropdown: 4x, 8x, 16x, 32x)
  - Training Steps (number input, min 1000)
  - Optimizer (dropdown: Adam, AdamW, SGD)
  - LR Schedule (dropdown: constant, linear, cosine, exponential)
  - Ghost Gradient Penalty (toggle switch, on/off)
- When I change a hyperparameter, the configuration updates immediately
- When I select higher batch sizes or expansion factors, I see memory warnings if exceeding edge device capacity

**UI Reference:** Mock-embedded-interp-ui.tsx Advanced Hyperparameters (lines 1704-1814)

---

### SC-1: View Training History (Priority: P2)
**As a** user
**I want to** view a list of all training jobs (active and completed)
**So that** I can track my training experiments over time

**Acceptance Criteria:**
- Given I'm on the Training panel, I see a "Training Jobs" section below configuration
- When no trainings exist, I see a message: "No training jobs yet. Configure and start training above."
- When trainings exist, I see cards for each training showing: Model name, Dataset name, Encoder type, Start time, Current status, Progress metrics (if active/completed)
- Given multiple trainings, they are sorted by creation time (most recent first)
- When I scroll through training list, older trainings load (if paginated)

**UI Reference:** Mock-embedded-interp-ui.tsx TrainingPanel job list (lines 1827-1842)

---

### US-6: Save and Reuse Training Templates (Priority: P0 - MVP Enhancement)
**As a** ML researcher
**I want to** save my successful training configurations as reusable templates
**So that** I can quickly configure similar training jobs without re-entering all hyperparameters

**Acceptance Criteria:**
- When I'm on the Training panel, I see a collapsible "Saved Templates" section showing template count (e.g., "Saved Templates (3)")
- When I click "Save as Template", I see a form with auto-generated name: `{encoder}_{expansion}x_{steps}steps_{HHMM}` (e.g., "sparse_8x_10000steps_1430")
- When I save a template, system stores: encoder type, hyperparameters (including trainingLayers array), model_id (optional), dataset_id (optional), name, description, is_favorite flag
- When templates exist, I see template cards showing: name, description, encoder type badge, hyperparameter summary (expansion, steps), favorite star, Load/Delete buttons
- When I click "Load" on a template, all configuration fields populate: encoder type, all hyperparameters including training layers, model (if template has model_id), dataset (if template has dataset_id)
- When I click star icon on template, it toggles favorite status and moves to top of list
- When I click "Delete" on template, I see confirmation: "Delete template '{name}'? This cannot be undone."
- When I click "Export Templates", system downloads JSON file with all training templates (included in combined export with extraction templates and steering presets)
- When I click "Import Templates", I can upload JSON file to restore templates
- When I load a template, I can still modify individual parameters before starting training

**UI Reference:** Mock-embedded-interp-ui.tsx Training Templates (lines 2285-2455)

---

### US-7: Train SAEs on Multiple Layers Simultaneously (Priority: P0 - MVP Enhancement)
**As a** ML researcher
**I want to** train sparse autoencoders on multiple transformer layers in a single training job
**So that** I can efficiently analyze feature emergence across layers without running separate training jobs

**Acceptance Criteria:**
- When I expand "Advanced Hyperparameters", I see a "Training Layers" section with an 8-column checkbox grid
- When model is selected, grid displays checkboxes for each layer (e.g., L0-L21 for TinyLlama with 22 layers)
- When I click "Select All", all layer checkboxes become checked
- When I click "Clear All", all layer checkboxes become unchecked
- When I select multiple layers (e.g., L0, L5, L10, L15), the label updates: "Training Layers (4 selected)"
- When I start training with multiple layers, system trains separate SAE for each selected layer using same hyperparameters
- When training progresses, metrics aggregate across all layers (average loss, average sparsity)
- When training completes, checkpoints include SAE states for all trained layers (directory structure: `checkpoint_{step}/layer_{idx}/encoder.pt`)
- When I view model architecture, I see layer count dynamically (TinyLlama: 22 layers, Phi-2: 32 layers)
- When I exceed recommended layer count (>4 on 8GB Jetson), I see warning: "Training >4 layers simultaneously may exceed available memory"
- When template includes trainingLayers array, loading template restores layer selections

**UI Reference:** Mock-embedded-interp-ui.tsx Multi-Layer Training UI (lines 2175-2236)

---

### SC-2: Export Training Results for Publication (Priority: P2)
**As a** researcher preparing a publication
**I want to** export training metrics, checkpoints, and configuration
**So that** I can include results in papers and ensure reproducibility

**Acceptance Criteria:**
- When viewing completed training, I can click "Export Results"
- When exporting, system creates ZIP file containing: training configuration JSON, all checkpoints, metrics CSV, training logs TXT
- Given exported ZIP, collaborators can reproduce training by importing configuration
- When I import configuration, all hyperparameters and settings restore exactly

---

## 4. Functional Requirements

### FR-1: Training Configuration (17 requirements)

**FR-1.1:** System SHALL provide dropdown selection for target dataset (filtered to show only datasets with status='ready')
**FR-1.2:** System SHALL provide dropdown selection for target model (filtered to show only models with status='ready')
**FR-1.3:** System SHALL provide encoder type selection with three options: "Sparse Autoencoder", "Skip Autoencoder", "Transcoder"
**FR-1.4:** System SHALL load default hyperparameters when encoder type is selected:
- Learning Rate: 0.001
- Batch Size: 256
- L1 Coefficient: 0.0001
- Expansion Factor: 8
- Training Steps: 10000
- Optimizer: AdamW
- LR Schedule: cosine
- Ghost Gradient Penalty: true

**FR-1.5:** System SHALL provide collapsible "Advanced Hyperparameters" section (default: collapsed)
**FR-1.6:** System SHALL validate that dataset and model are selected before enabling "Start Training" button
**FR-1.7:** System SHALL calculate estimated training time based on: training steps, batch size, model size, dataset size
**FR-1.8:** System SHALL display memory warning if batch_size * expansion_factor * model_hidden_size exceeds available GPU memory (6GB threshold for Jetson Orin Nano)
**FR-1.9:** System SHALL persist hyperparameter changes in browser session storage for recovery on page refresh
**FR-1.10:** System SHALL provide "Reset to Defaults" button to restore default hyperparameters
**FR-1.11:** System SHALL validate hyperparameter ranges before training submission:
- Learning Rate: 0.000001 to 0.1
- Batch Size: 64 to 2048 (power of 2 preferred)
- L1 Coefficient: 0.00001 to 0.01
- Expansion Factor: 4 to 64
- Training Steps: 1000 to 1000000

**FR-1.12:** System SHALL generate unique training ID with format: `tr_{uuid}_{timestamp}`
**FR-1.13:** System SHALL create database record in `trainings` table with status='queued' on job submission
**FR-1.14:** System SHALL store complete hyperparameter configuration in JSONB `hyperparameters` field
**FR-1.15:** System SHALL calculate total_steps based on: training_steps hyperparameter
**FR-1.16:** System SHALL assign training to available GPU (gpu_id field) and Celery worker (worker_id field)
**FR-1.17:** System SHALL emit WebSocket event `training:created` with training metadata to connected clients

---

### FR-1A: Training Template Management (20 requirements)

**FR-1A.1:** System SHALL provide collapsible "Saved Templates" section in Training panel below configuration form
**FR-1A.2:** System SHALL display template count in section header (e.g., "Saved Templates (5)")
**FR-1A.3:** System SHALL provide "Save as Template" form with fields:
- Template name input (with auto-generated default)
- Optional description textarea (max 500 characters)
- "Save Template" submit button

**FR-1A.4:** System SHALL auto-generate training template names with format:
- Pattern: `{encoder}_{expansion}x_{steps}steps_{HHMM}`
- Encoder: `sparse`, `skip`, or `transcoder`
- Expansion: expansion factor value (e.g., 8)
- Steps: training_steps value (e.g., 10000)
- Timestamp: HHMM in 24-hour format (e.g., 1430 = 2:30 PM)
- Example: `sparse_8x_10000steps_1430`

**FR-1A.5:** System SHALL save training templates to `training_templates` database table with schema:
```sql
CREATE TABLE training_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    encoder_type VARCHAR(20) NOT NULL,
    hyperparameters JSONB NOT NULL,  -- includes trainingLayers: number[]
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**FR-1A.6:** System SHALL store complete hyperparameter configuration in JSONB field including:
- learningRate, batchSize, l1Coefficient, expansionFactor, trainingSteps
- trainingLayers (number[] array for multi-layer support)
- optimizer, lrSchedule, ghostGradPenalty

**FR-1A.7:** System SHALL display template list with template cards showing:
- Template name (bold, 18px, clickable)
- Description (if provided, text-slate-400, 14px, truncated to 150 chars)
- Encoder type badge (bg-purple-500/20, text-purple-400)
- Hyperparameter summary badges: "{expansion}x expansion", "{steps} steps"
- Training layers badge (if multi-layer): "Layers {min}-{max}" or "Layer {N}"
- Model/Dataset names (if associated): "Model: {name} • Dataset: {name}"
- Favorite star icon (gold text-yellow-400 if true, gray text-slate-500 if false)
- "Load" button (bg-emerald-600 hover:bg-emerald-700)
- "Delete" button (text-red-400 hover:text-red-300)

**FR-1A.8:** System SHALL implement "Load Template" action:
- Populate encoder_type selection from template.encoder_type
- Populate all hyperparameter inputs from template.hyperparameters
- Restore trainingLayers array selections in layer checkbox grid
- Set model dropdown to template.model_id (if not null and model still exists)
- Set dataset dropdown to template.dataset_id (if not null and dataset still exists)
- Show success toast: "Template '{name}' loaded successfully"
- Log event: `template_loaded` with template_id

**FR-1A.9:** System SHALL implement "Delete Template" action:
- Display confirmation dialog: "Delete training template '{name}'? This action cannot be undone."
- On confirm: Send DELETE /api/v1/templates/training/:id
- Remove template card from UI with 300ms fade-out animation
- Show success toast: "Template '{name}' deleted"
- Update template count in section header
- Log event: `template_deleted` with template_id

**FR-1A.10:** System SHALL implement "Toggle Favorite" action:
- On star icon click: Toggle is_favorite flag in database
- Send PATCH /api/v1/templates/training/:id/favorite with { is_favorite: boolean }
- Update star icon immediately (optimistic UI update)
- Re-sort template list to move favorited templates to top
- Show subtle toast: "Template marked as favorite" or "Template unfavorited"

**FR-1A.11:** System SHALL sort templates by:
1. Favorites first (is_favorite = true, ordered by updated_at DESC)
2. Non-favorites (is_favorite = false, ordered by updated_at DESC)

**FR-1A.12:** System SHALL support combined template export/import:
- Export: POST /api/v1/templates/export returns JSON with structure:
```json
{
  "version": "1.0",
  "exported_at": "2025-10-07T12:34:56Z",
  "training_templates": [...],
  "extraction_templates": [...],
  "steering_presets": [...]
}
```
- Import: POST /api/v1/templates/import accepts FormData with JSON file
- Import validates template structure and skips invalid entries with warnings
- Import generates new UUIDs for all imported templates
- Import preserves is_favorite flags

**FR-1A.13:** System SHALL prevent duplicate template names:
- Check for existing name before save
- If duplicate found, append counter: `{name}_2`, `{name}_3`, etc.
- Show warning toast: "Template name exists, saved as '{new_name}'"

**FR-1A.14:** System SHALL validate template configuration before save:
- Name must not be empty and must be ≤255 characters
- Encoder type must be one of: 'sparse', 'skip', 'transcoder'
- Hyperparameters JSONB must include all required fields
- trainingLayers array must have at least 1 layer
- If model_id provided, model must exist and status='ready'
- If dataset_id provided, dataset must exist and status='ready'

**FR-1A.15:** System SHALL provide "Export Templates" button:
- Button label: "Export All Templates"
- On click: Triggers combined export of all template types
- Downloads file: `mistudio_templates_{timestamp}.json`
- Shows success toast: "Exported {count} templates"

**FR-1A.16:** System SHALL provide "Import Templates" button:
- Button label: "Import Templates"
- On click: Opens file picker (accept: .json)
- Validates JSON structure before import
- Shows progress dialog during import
- Shows summary after import: "{count} templates imported, {errors} errors"
- Lists any validation errors with template names

**FR-1A.17:** System SHALL handle template loading edge cases:
- If template references deleted model: Clear model_id, show warning: "Associated model no longer exists"
- If template references deleted dataset: Clear dataset_id, show warning: "Associated dataset no longer exists"
- If template has invalid layer indices for current model: Show error: "Template layer selections incompatible with current model architecture"

**FR-1A.18:** System SHALL provide template search/filter (future enhancement):
- Filter by encoder type
- Filter by favorites only
- Search by template name
- Sort by: name, created_at, updated_at

**FR-1A.19:** System SHALL log all template operations for audit:
- template_created: {template_id, name, user_action}
- template_loaded: {template_id, name, training_id}
- template_deleted: {template_id, name}
- template_exported: {count, timestamp}
- template_imported: {count, errors, timestamp}

**FR-1A.20:** System SHALL update template.updated_at timestamp on:
- Template metadata edit (name or description change)
- Toggle favorite status

---

### FR-1B: Multi-Layer Training Support (18 requirements)

**FR-1B.1:** System SHALL provide "Training Layers" section within Advanced Hyperparameters with:
- Section label: "Training Layers ({N} selected)"
- 8-column checkbox grid
- "Select All" button
- "Clear All" button
- Dynamic grid generation based on model.num_layers

**FR-1B.2:** System SHALL dynamically generate layer checkboxes based on selected model's architecture:
- Query model.num_layers from database
- Generate checkbox grid with layers 0 to (num_layers - 1)
- Layout: 8 columns, rows as needed
- Example: TinyLlama (22 layers) = 3 rows, Phi-2 (32 layers) = 4 rows

**FR-1B.3:** System SHALL require model architecture metadata in models table:
```sql
ALTER TABLE models ADD COLUMN num_layers INT;
ALTER TABLE models ADD COLUMN hidden_dim INT;
ALTER TABLE models ADD COLUMN num_heads INT;
```

**FR-1B.4:** System SHALL extract and store architecture metadata during model download:
- Parse config.json for: n_layer (or num_hidden_layers), hidden_size (or n_embd), n_head (or num_attention_heads)
- Map architecture-specific names to standard fields
- Store in models table for fast UI generation

**FR-1B.5:** System SHALL style layer checkboxes per Mock UI specification:
- Unchecked: border-slate-700 bg-slate-900 text-slate-400
- Checked: border-emerald-500 bg-emerald-500/20 text-emerald-400
- Hover: border-emerald-400
- Label format: "L{idx}" (e.g., L0, L1, L2, ...)

**FR-1B.6:** System SHALL implement "Select All" action:
- Check all layer checkboxes
- Update trainingLayers array to [0, 1, 2, ..., num_layers-1]
- Update label: "Training Layers ({num_layers} selected)"
- Enable visual feedback (all checkboxes highlighted)

**FR-1B.7:** System SHALL implement "Clear All" action:
- Uncheck all layer checkboxes
- Update trainingLayers array to empty []
- Update label: "Training Layers (0 selected)"
- Show validation warning if user tries to start training with 0 layers

**FR-1B.8:** System SHALL validate training layer selection:
- At least 1 layer must be selected before training can start
- Show error message: "Please select at least one training layer"
- Disable "Start Training" button if trainingLayers.length === 0

**FR-1B.9:** System SHALL store trainingLayers array in hyperparameters JSONB field:
```json
{
  "learningRate": 0.001,
  "batchSize": 256,
  "trainingLayers": [0, 5, 10, 15, 20],
  ...other hyperparameters
}
```

**FR-1B.10:** System SHALL calculate memory requirements for multi-layer training:
- Memory per layer: hidden_size × expansion_factor × 4 bytes (FP32) × 2 (encoder + decoder)
- Total memory: memory_per_layer × num_selected_layers + base_memory
- Show warning if total_memory > 6GB: "Training {count} layers may exceed available memory. Recommend ≤4 layers on 8GB devices."

**FR-1B.11:** System SHALL initialize separate SAE instance for each selected layer:
```python
saes = {}
for layer_idx in config.hyperparameters['trainingLayers']:
    saes[layer_idx] = SparseAutoencoder(
        d_model=model.hidden_size,
        d_sae=model.hidden_size * expansion_factor,
        l1_coefficient=config.hyperparameters['l1Coefficient']
    )
```

**FR-1B.12:** System SHALL extract activations from all selected layers during training:
- Register forward hooks on layers specified in trainingLayers array
- Collect activations in dictionary: {layer_idx: activations_tensor}
- Process all layers in single forward pass for efficiency

**FR-1B.13:** System SHALL train each layer's SAE independently:
- Separate optimizer for each SAE (or shared optimizer with parameter groups)
- Calculate loss per layer: recon_loss[layer_idx] + l1_penalty[layer_idx]
- Aggregate losses: total_loss = sum(losses.values()) / num_layers

**FR-1B.14:** System SHALL aggregate metrics across layers for display:
- Average loss: mean(loss per layer)
- Average L0 sparsity: mean(sparsity per layer)
- Total dead neurons: sum(dead_neurons per layer)
- Per-layer metrics stored in training_metrics table with layer_idx column

**FR-1B.15:** System SHALL save multi-layer checkpoints with directory structure:
```
/data/trainings/{training_id}/checkpoints/checkpoint_{step}/
    ├── layer_0/
    │   ├── encoder.pt
    │   ├── decoder.pt
    │   └── optimizer.pt
    ├── layer_5/
    │   ├── encoder.pt
    │   ├── decoder.pt
    │   └── optimizer.pt
    └── metadata.json  # includes trainingLayers array
```

**FR-1B.16:** System SHALL load multi-layer checkpoints on resume:
- Parse metadata.json to get trainingLayers array
- Load SAE state for each layer from checkpoint/{layer_idx}/ subdirectory
- Validate checkpoint structure before loading
- Show error if checkpoint incomplete: "Checkpoint missing states for layers: {missing_layers}"

**FR-1B.17:** System SHALL display multi-layer training in UI:
- Training card shows: "Training {count} layers: L{min}-L{max}"
- Metrics display average across layers with tooltip showing per-layer breakdown
- Progress bar reflects steps across all layers (same step count for all)

**FR-1B.18:** System SHALL handle multi-layer training errors:
- If OOM error during initialization: Reduce number of layers automatically and retry
- If OOM during training: Reduce batch size for all layers
- Log which layer caused error for debugging
- Show user-friendly error: "GPU memory exceeded. Try training fewer layers or reducing batch size."

---

### FR-2: Training Initialization (12 requirements)

**FR-2.1:** System SHALL transition training status from 'queued' to 'initializing' when Celery worker picks up job
**FR-2.2:** System SHALL load target model into GPU memory with configured quantization
**FR-2.3:** System SHALL load tokenized dataset from filesystem path specified in dataset record
**FR-2.4:** System SHALL create SAE model architecture based on encoder_type:
- **Sparse Autoencoder:** Standard encoder-decoder with L1 sparsity penalty
- **Skip Autoencoder:** Includes residual skip connections
- **Transcoder:** Cross-layer activation mapping

**FR-2.5:** System SHALL calculate SAE architecture dimensions:
- Input dimension: model.hidden_size (from model architecture config)
- Hidden dimension: model.hidden_size * expansion_factor
- Output dimension: model.hidden_size (reconstruction target)

**FR-2.6:** System SHALL initialize SAE weights using Xavier/Glorot initialization for encoder/decoder
**FR-2.7:** System SHALL create optimizer instance (Adam/AdamW/SGD) with configured learning rate
**FR-2.8:** System SHALL create learning rate scheduler (constant/linear/cosine/exponential) based on configuration
**FR-2.9:** System SHALL create training output directory: `/data/trainings/{training_id}/`
**FR-2.10:** System SHALL set up checkpoint directory: `/data/trainings/{training_id}/checkpoints/`
**FR-2.11:** System SHALL log initialization details to training logs: model loaded, dataset loaded, SAE architecture created, optimizer configured
**FR-2.12:** System SHALL transition status to 'training' after successful initialization, or 'error' on failure

---

### FR-3: Training Loop Execution (15 requirements)

**FR-3.1:** System SHALL execute training loop for total_steps iterations
**FR-3.2:** System SHALL perform the following per training step:
1. Extract batch of samples from dataset
2. Tokenize batch and create input_ids tensor
3. Extract model activations from specified layer(s) using forward hooks
4. Pass activations through SAE encoder
5. Calculate L1 sparsity penalty: `l1_penalty = l1_coefficient * abs(encoder_output).sum()`
6. Pass encoder output through SAE decoder
7. Calculate reconstruction loss: `recon_loss = mse(decoder_output, activations)`
8. Calculate total loss: `total_loss = recon_loss + l1_penalty`
9. Backpropagate gradients
10. Update optimizer step
11. Update learning rate scheduler
12. Clear GPU cache

**FR-3.3:** System SHALL calculate L0 sparsity metric per step: `l0_sparsity = (encoder_output > threshold).float().sum(dim=-1).mean()`
**FR-3.4:** System SHALL track dead neurons: neurons with activation count < 1% across last 1000 steps
**FR-3.5:** System SHALL update `current_step` field in database every 10 steps
**FR-3.6:** System SHALL store training metrics in `training_metrics` table every 10 steps with fields:
- step
- loss (total_loss)
- sparsity (l0_sparsity)
- reconstruction_error (recon_loss)
- dead_neurons (count)
- learning_rate (current_lr)
- timestamp

**FR-3.7:** System SHALL update denormalized metrics in `trainings` table every 10 steps:
- latest_loss
- latest_sparsity
- latest_dead_neurons

**FR-3.8:** System SHALL emit WebSocket event `training:progress` every 10 steps with metrics payload
**FR-3.9:** System SHALL monitor GPU memory usage and log warnings if utilization > 90%
**FR-3.10:** System SHALL implement gradient clipping (max_norm=1.0) to prevent exploding gradients
**FR-3.11:** System SHALL implement ghost gradient penalty if hyperparameters.ghostGradPenalty=true:
- Penalizes neurons that rarely activate
- Formula: `ghost_penalty = (1 - activation_frequency) * encoder_weights.norm()`

**FR-3.12:** System SHALL handle OOM errors gracefully:
- Catch CUDA OOM exception
- Reduce batch size by 50%
- Clear GPU cache
- Retry step with smaller batch

**FR-3.13:** System SHALL support pause operation:
- Set status='paused' in database
- Save current training state (optimizer, scheduler, step count) to checkpoint
- Emit WebSocket event `training:paused`
- Exit training loop gracefully

**FR-3.14:** System SHALL support stop operation:
- Set status='stopped' in database
- Save final checkpoint (marked as stopped)
- Emit WebSocket event `training:stopped`
- Clean up GPU memory
- Exit training loop

**FR-3.15:** System SHALL transition to 'completed' status when current_step reaches total_steps

---

### FR-4: Checkpoint Management (13 requirements)

**FR-4.1:** System SHALL save training checkpoints containing:
- SAE model state_dict (encoder and decoder weights)
- Optimizer state_dict
- Learning rate scheduler state_dict
- Current training step
- Latest metrics (loss, sparsity, dead_neurons)
- Training hyperparameters (for validation)
- Random number generator states (for reproducibility)

**FR-4.2:** System SHALL save checkpoint to filesystem: `/data/trainings/{training_id}/checkpoints/{checkpoint_id}.pt`
**FR-4.3:** System SHALL create checkpoint database record in `checkpoints` table with fields:
- id (format: `cp_{uuid}`)
- training_id
- step
- loss
- sparsity
- storage_path
- file_size_bytes
- created_at

**FR-4.4:** System SHALL support manual checkpoint save triggered by API POST request
**FR-4.5:** System SHALL support automatic checkpoint save at configured intervals (auto-save feature)
**FR-4.6:** System SHALL implement checkpoint retention policy:
- Keep all checkpoints if total < 10
- If total > 10, keep: first, last, every 1000 steps, best loss checkpoint

**FR-4.7:** System SHALL support checkpoint loading for training resume:
- Load SAE model state_dict
- Load optimizer state_dict
- Load scheduler state_dict
- Restore current_step
- Continue training from restored state

**FR-4.8:** System SHALL validate checkpoint compatibility before loading:
- Match training_id
- Verify hyperparameters match current configuration
- Check model architecture dimensions match

**FR-4.9:** System SHALL support checkpoint deletion:
- Remove checkpoint file from filesystem
- Remove checkpoint record from database
- Prevent deletion of most recent checkpoint if training is paused

**FR-4.10:** System SHALL calculate checkpoint file size after save and update database record
**FR-4.11:** System SHALL use PyTorch safetensors format for checkpoint serialization (safer than pickle)
**FR-4.12:** System SHALL emit WebSocket event `checkpoint:created` when checkpoint is saved
**FR-4.13:** System SHALL list checkpoints via API GET request, sorted by step descending

---

### FR-5: Progress Tracking and Metrics (11 requirements)

**FR-5.1:** System SHALL calculate progress percentage as: `(current_step / total_steps) * 100`
**FR-5.2:** System SHALL update `trainings.updated_at` timestamp on every database write
**FR-5.3:** System SHALL provide API endpoint `GET /api/trainings/:id/metrics` to retrieve time-series metrics
**FR-5.4:** System SHALL support metrics query with filters:
- start_step, end_step (range filtering)
- limit (max records returned, default 1000)
- interval (downsample to every Nth step)

**FR-5.5:** System SHALL calculate estimated time remaining:
- Formula: `(total_steps - current_step) * avg_step_time`
- avg_step_time calculated from last 100 steps

**FR-5.6:** System SHALL track and display GPU utilization percentage:
- Query from NVIDIA SMI or PyTorch CUDA APIs
- Update every 5 seconds
- Only display when status='training'

**FR-5.7:** System SHALL provide real-time training logs via WebSocket channel `training:{training_id}:logs`
**FR-5.8:** System SHALL store training logs to file: `/data/trainings/{training_id}/logs.txt`
**FR-5.9:** System SHALL log the following events:
- Training started/paused/resumed/stopped/completed
- Checkpoint saved/loaded
- OOM errors and batch size reductions
- Dead neuron warnings (if count > 5% of total)
- Convergence warnings (if loss plateaus for 1000 steps)

**FR-5.10:** System SHALL calculate training throughput: `steps_per_second = current_step / elapsed_time`
**FR-5.11:** System SHALL display metrics in TrainingCard component:
- Loss (4 decimal places)
- L0 Sparsity (1 decimal place)
- Dead Neurons (integer count)
- GPU Utilization (percentage)

---

### FR-6: Training Status Management (10 requirements)

**FR-6.1:** System SHALL support the following training statuses:
- `queued`: Job submitted, waiting for worker
- `initializing`: Loading model, dataset, creating SAE
- `training`: Active training loop executing
- `paused`: Temporarily suspended by user
- `stopped`: Permanently terminated by user
- `completed`: Successfully finished all training steps
- `failed`: Terminated due to error (with error details)
- `error`: System error during initialization or training

**FR-6.2:** System SHALL enforce valid status transitions:
- queued → initializing → training
- training → paused → training (resume)
- training/paused → stopped (permanent termination)
- training → completed (natural completion)
- Any status → failed/error (on exception)

**FR-6.3:** System SHALL store error_message and error_code in database on failure
**FR-6.4:** System SHALL implement retry_count tracking (max 3 retries on transient errors)
**FR-6.5:** System SHALL provide API endpoint `POST /api/trainings/:id/pause` to pause training
**FR-6.6:** System SHALL provide API endpoint `POST /api/trainings/:id/resume` to resume paused training
**FR-6.7:** System SHALL provide API endpoint `POST /api/trainings/:id/stop` to permanently stop training
**FR-6.8:** System SHALL provide API endpoint `POST /api/trainings/:id/retry` to restart failed training from beginning
**FR-6.9:** System SHALL emit WebSocket events on status changes: `training:status_changed` with payload: {training_id, old_status, new_status}
**FR-6.10:** System SHALL display appropriate status indicators in UI:
- initializing: Blue spinner icon
- training: Green pulsing Activity icon
- paused: Yellow pause icon
- completed: Green checkmark icon
- error/failed: Red X icon

---

### FR-7: Memory Optimization for Edge Deployment (9 requirements)

**FR-7.1:** System SHALL implement dynamic batch size adjustment:
- Start with configured batch_size
- If OOM occurs, reduce by 50%
- Minimum batch size: 16
- Log batch size changes

**FR-7.2:** System SHALL implement gradient accumulation if batch size falls below 64:
- Accumulate gradients over N micro-batches to maintain effective batch size
- Formula: `accumulation_steps = max(1, 64 // current_batch_size)`

**FR-7.3:** System SHALL use PyTorch automatic mixed precision (AMP) training:
- Use torch.cuda.amp.autocast() for forward and loss calculation
- Use GradScaler for gradient scaling
- Reduces memory usage by ~40%

**FR-7.4:** System SHALL clear GPU cache between batches: `torch.cuda.empty_cache()`
**FR-7.5:** System SHALL offload optimizer state to CPU if GPU memory > 90% utilized
**FR-7.6:** System SHALL implement activation checkpointing for large models (>1B parameters):
- Use torch.utils.checkpoint.checkpoint() for encoder forward pass
- Trades compute for memory (recomputes activations during backward pass)

**FR-7.7:** System SHALL validate memory requirements before training start:
- Calculate estimated memory: `batch_size * hidden_dim * expansion_factor * 4 bytes`
- Warn if estimated > 6GB (Jetson Orin Nano limit)
- Suggest reduced batch size or expansion factor

**FR-7.8:** System SHALL log memory statistics every 100 steps:
- GPU memory allocated
- GPU memory reserved
- GPU memory free
- Peak memory usage

**FR-7.9:** System SHALL implement early stopping on repeated OOM errors:
- If OOM occurs 3 times consecutively after batch size reduction
- Set status='failed' with error: "Insufficient GPU memory for training"

---

### FR-8: Training Visualization (8 requirements)

**FR-8.1:** System SHALL provide "Show Live Metrics" toggle in TrainingCard component
**FR-8.2:** System SHALL display Loss Curve visualization when metrics are shown:
- Bar chart showing last 20 steps
- Y-axis: Loss value (auto-scaled)
- X-axis: Training step
- Color: Emerald-500 gradient

**FR-8.3:** System SHALL display L0 Sparsity Curve visualization when metrics are shown:
- Bar chart showing last 20 steps
- Y-axis: L0 sparsity value (0-100)
- X-axis: Training step
- Color: Blue-500 gradient

**FR-8.4:** System SHALL display Training Logs panel when metrics are shown:
- Dark slate background with monospace font
- Last 10 log entries with timestamps
- Auto-scroll to newest entries
- "Live" indicator when training is active

**FR-8.5:** System SHALL update visualizations every 2 seconds when training is active
**FR-8.6:** System SHALL use WebSocket subscriptions for real-time updates (not polling)
**FR-8.7:** System SHALL display "—" placeholder for metrics before training reaches step 10
**FR-8.8:** System SHALL highlight concerning metrics:
- Loss > previous step: Yellow highlight
- Dead neurons > 5%: Red highlight
- GPU utilization > 95%: Red highlight

---

## 5. Non-Functional Requirements

### Performance Requirements
- **Training Throughput:** System SHALL achieve minimum 10 steps/second on Jetson Orin Nano with batch_size=256, expansion_factor=8
- **Metric Update Latency:** Progress updates SHALL appear in UI within 2 seconds of database write
- **Checkpoint Save Time:** Checkpoint save operation SHALL complete within 5 seconds for SAE models with <100M parameters
- **WebSocket Latency:** Real-time metrics SHALL be delivered via WebSocket within 500ms of emission

### Scalability Requirements
- **Concurrent Trainings:** System SHALL support up to 2 simultaneous training jobs on single Jetson Orin Nano (if memory allows)
- **Metrics Storage:** System SHALL handle 1,000,000 metric records (100 trainings × 10,000 steps) without query performance degradation
- **Checkpoint Storage:** System SHALL efficiently store up to 1,000 checkpoints (100 trainings × 10 checkpoints) totaling ~50GB

### Reliability Requirements
- **Training Resume:** System SHALL successfully resume training from checkpoint with <0.1% divergence in loss trajectory
- **Error Recovery:** System SHALL automatically retry transient errors (OOM, CUDA errors) up to 3 times before failing
- **Data Integrity:** System SHALL ensure atomic checkpoint saves (no partial/corrupted checkpoint files)

### Usability Requirements
- **Configuration Time:** Users SHALL be able to configure and start training within 30 seconds using default parameters
- **Metric Interpretation:** All metrics SHALL include tooltips with plain-language explanations
- **Error Messages:** Failure errors SHALL provide actionable troubleshooting steps

---

## 6. Technical Requirements

### Architecture Constraints (from ADR)
- **Backend Framework:** FastAPI with async/await patterns for non-blocking training operations
- **Background Jobs:** Celery + Redis for distributed training job execution
- **Database:** PostgreSQL 14+ with JSONB for hyperparameter storage
- **ML Framework:** PyTorch 2.0+ with CUDA support
- **Model Serialization:** safetensors format for checkpoint saves (safer than pickle)

### Edge Deployment Constraints
- **Hardware:** NVIDIA Jetson Orin Nano (8GB RAM, 6GB GPU VRAM, CUDA 11.4+)
- **Memory Budget:** Training SHALL not exceed 6GB GPU memory (includes model + SAE + activations + gradients)
- **Thermal Management:** Training SHALL monitor GPU temperature and throttle if >80°C

### API Specifications

#### POST /api/trainings
**Purpose:** Create and start new training job
**Authentication:** Required
**Request Body:**
```json
{
  "dataset_id": "ds_pile_123abc",
  "model_id": "m_gpt2_medium_456def",
  "encoder_type": "sparse",
  "hyperparameters": {
    "learningRate": 0.001,
    "batchSize": 256,
    "l1Coefficient": 0.0001,
    "expansionFactor": 8,
    "trainingSteps": 10000,
    "optimizer": "AdamW",
    "lrSchedule": "cosine",
    "ghostGradPenalty": true
  }
}
```
**Response:** 201 Created
```json
{
  "id": "tr_abc123def456",
  "status": "queued",
  "created_at": "2025-10-06T10:30:00Z",
  "dataset": { "id": "ds_pile_123abc", "name": "The Pile" },
  "model": { "id": "m_gpt2_medium_456def", "name": "GPT-2 Medium" },
  "encoder_type": "sparse",
  "hyperparameters": { ... },
  "progress": 0,
  "current_step": 0,
  "total_steps": 10000
}
```

#### GET /api/trainings
**Purpose:** List all training jobs
**Query Parameters:**
- `status` (optional): Filter by status (e.g., "training,paused,completed")
- `model_id` (optional): Filter by model
- `dataset_id` (optional): Filter by dataset
- `limit` (default: 50): Max results
- `offset` (default: 0): Pagination offset

**Response:** 200 OK
```json
{
  "trainings": [
    {
      "id": "tr_abc123",
      "dataset_id": "ds_pile_123",
      "model_id": "m_gpt2_456",
      "encoder_type": "sparse",
      "status": "training",
      "progress": 45.2,
      "current_step": 4520,
      "total_steps": 10000,
      "latest_loss": 0.0234,
      "latest_sparsity": 52.3,
      "latest_dead_neurons": 45,
      "created_at": "2025-10-06T10:00:00Z",
      "started_at": "2025-10-06T10:05:00Z",
      "updated_at": "2025-10-06T10:45:00Z"
    }
  ],
  "total": 15,
  "limit": 50,
  "offset": 0
}
```

#### GET /api/trainings/:id
**Purpose:** Get single training job details
**Response:** 200 OK (same structure as POST response, but with current metrics)

#### POST /api/trainings/:id/pause
**Purpose:** Pause active training
**Response:** 200 OK
```json
{
  "status": "paused",
  "paused_at": "2025-10-06T10:50:00Z",
  "checkpoint_id": "cp_pause_abc123"
}
```

#### POST /api/trainings/:id/resume
**Purpose:** Resume paused training
**Response:** 200 OK
```json
{
  "status": "training",
  "resumed_at": "2025-10-06T11:00:00Z"
}
```

#### POST /api/trainings/:id/stop
**Purpose:** Permanently stop training
**Response:** 200 OK
```json
{
  "status": "stopped",
  "stopped_at": "2025-10-06T11:05:00Z",
  "final_checkpoint_id": "cp_final_abc123"
}
```

#### POST /api/trainings/:id/retry
**Purpose:** Retry failed training from beginning
**Response:** 201 Created (new training instance with same config)

#### GET /api/trainings/:id/metrics
**Purpose:** Retrieve time-series training metrics
**Query Parameters:**
- `start_step` (optional): Start of step range
- `end_step` (optional): End of step range
- `interval` (optional): Downsample to every Nth step
- `limit` (default: 1000): Max records

**Response:** 200 OK
```json
{
  "training_id": "tr_abc123",
  "metrics": [
    {
      "step": 100,
      "loss": 0.0456,
      "sparsity": 48.3,
      "reconstruction_error": 0.0412,
      "dead_neurons": 78,
      "learning_rate": 0.001,
      "timestamp": "2025-10-06T10:10:00Z"
    },
    {
      "step": 200,
      "loss": 0.0398,
      "sparsity": 51.2,
      "reconstruction_error": 0.0354,
      "dead_neurons": 62,
      "learning_rate": 0.00095,
      "timestamp": "2025-10-06T10:15:00Z"
    }
  ],
  "total": 4520
}
```

#### POST /api/trainings/:id/checkpoints
**Purpose:** Manually save checkpoint
**Request Body:** (optional)
```json
{
  "note": "Before hyperparameter change"
}
```
**Response:** 201 Created
```json
{
  "id": "cp_manual_abc123",
  "training_id": "tr_abc123",
  "step": 4520,
  "loss": 0.0234,
  "sparsity": 52.3,
  "storage_path": "/data/trainings/tr_abc123/checkpoints/cp_manual_abc123.pt",
  "file_size_bytes": 45670234,
  "created_at": "2025-10-06T10:50:00Z"
}
```

#### GET /api/trainings/:id/checkpoints
**Purpose:** List checkpoints for training
**Response:** 200 OK
```json
{
  "checkpoints": [
    {
      "id": "cp_manual_abc123",
      "step": 4520,
      "loss": 0.0234,
      "sparsity": 52.3,
      "file_size_bytes": 45670234,
      "created_at": "2025-10-06T10:50:00Z"
    }
  ]
}
```

#### POST /api/trainings/:id/checkpoints/:checkpoint_id/load
**Purpose:** Load checkpoint and resume training from that point
**Response:** 200 OK
```json
{
  "status": "training",
  "loaded_checkpoint": "cp_manual_abc123",
  "resumed_from_step": 4520
}
```

#### DELETE /api/trainings/:id/checkpoints/:checkpoint_id
**Purpose:** Delete checkpoint
**Response:** 204 No Content

---

### Database Schema

#### trainings Table (Expansion from 003_SPEC|Postgres_Usecase_Details_and_Guidance.md)
```sql
CREATE TABLE trainings (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'tr_abc123'

    -- Foreign keys
    model_id VARCHAR(255) NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    dataset_id VARCHAR(255) NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,

    -- Configuration
    encoder_type VARCHAR(50) NOT NULL,  -- 'sparse', 'skip', 'transcoder'
    hyperparameters JSONB NOT NULL,  -- Complete hyperparameter set

    -- State
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
        -- Values: 'queued', 'initializing', 'training', 'paused',
        --         'stopped', 'completed', 'failed', 'error'

    -- Progress tracking
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER NOT NULL,
    progress FLOAT GENERATED ALWAYS AS (
        CASE WHEN total_steps > 0
        THEN (current_step::FLOAT / total_steps * 100)
        ELSE 0 END
    ) STORED,

    -- Latest metrics (denormalized for quick access)
    latest_loss FLOAT,
    latest_sparsity FLOAT,
    latest_dead_neurons INTEGER,

    -- Error handling
    error_message TEXT,
    error_code VARCHAR(100),
    retry_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    paused_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Resource allocation
    gpu_id INTEGER,
    worker_id VARCHAR(100),

    CONSTRAINT trainings_status_check CHECK (status IN
        ('queued', 'initializing', 'training', 'paused', 'stopped',
         'completed', 'failed', 'error'))
);

CREATE INDEX idx_trainings_status ON trainings(status);
CREATE INDEX idx_trainings_model_id ON trainings(model_id);
CREATE INDEX idx_trainings_dataset_id ON trainings(dataset_id);
CREATE INDEX idx_trainings_created_at ON trainings(created_at DESC);
CREATE INDEX idx_trainings_completed_at ON trainings(completed_at DESC NULLS LAST);

-- Composite index for filtering active trainings
CREATE INDEX idx_trainings_active ON trainings(status, updated_at)
    WHERE status IN ('queued', 'initializing', 'training', 'paused');
```

**Hyperparameters JSONB Structure:**
```json
{
  "learningRate": 0.001,
  "batchSize": 256,
  "l1Coefficient": 0.0001,
  "expansionFactor": 8,
  "trainingSteps": 10000,
  "optimizer": "AdamW",
  "lrSchedule": "cosine",
  "ghostGradPenalty": true
}
```

#### training_metrics Table
```sql
CREATE TABLE training_metrics (
    id BIGSERIAL PRIMARY KEY,
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Metrics
    step INTEGER NOT NULL,
    loss FLOAT NOT NULL,
    sparsity FLOAT,  -- L0 sparsity (average active features)
    reconstruction_error FLOAT,
    dead_neurons INTEGER,
    explained_variance FLOAT,

    -- Learning rate (for debugging)
    learning_rate FLOAT,

    -- Timestamp
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite unique constraint (one metric per step per training)
    CONSTRAINT training_metrics_unique_step UNIQUE (training_id, step)
);

-- Critical index for time-series queries
CREATE INDEX idx_training_metrics_training_step ON training_metrics(training_id, step);
CREATE INDEX idx_training_metrics_timestamp ON training_metrics(training_id, timestamp);
```

**Storage Estimate:** ~50 bytes per metric × 10,000 steps = 500KB per training

#### checkpoints Table
```sql
CREATE TABLE checkpoints (
    id VARCHAR(255) PRIMARY KEY,  -- e.g., 'cp_abc123'
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,

    -- Checkpoint details
    step INTEGER NOT NULL,
    loss FLOAT,
    sparsity FLOAT,

    -- File storage
    storage_path VARCHAR(1000) NOT NULL,  -- /data/trainings/{training_id}/checkpoints/{id}.pt
    file_size_bytes BIGINT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite unique constraint
    CONSTRAINT checkpoints_unique_step UNIQUE (training_id, step)
);

CREATE INDEX idx_checkpoints_training_id ON checkpoints(training_id);
CREATE INDEX idx_checkpoints_step ON checkpoints(training_id, step DESC);
CREATE INDEX idx_checkpoints_created_at ON checkpoints(created_at DESC);
```

**Storage Estimate:** ~500 bytes per checkpoint × 10 checkpoints = 5KB per training (metadata only, checkpoint files stored on filesystem)

---

### WebSocket Events

**Channel:** `training:{training_id}`
**Events:**
- `training:created` - New training job created
- `training:status_changed` - Status transition occurred
- `training:progress` - Progress update (every 10 steps)
- `training:paused` - Training paused
- `training:resumed` - Training resumed
- `training:stopped` - Training stopped
- `training:completed` - Training completed
- `training:error` - Training error occurred
- `checkpoint:created` - Checkpoint saved
- `checkpoint:loaded` - Checkpoint loaded
- `checkpoint:deleted` - Checkpoint deleted

**Example Event Payload (training:progress):**
```json
{
  "training_id": "tr_abc123",
  "current_step": 4520,
  "progress": 45.2,
  "latest_loss": 0.0234,
  "latest_sparsity": 52.3,
  "latest_dead_neurons": 45,
  "gpu_utilization": 78.5,
  "timestamp": "2025-10-06T10:45:00Z"
}
```

---

### File Storage Structure

```
/data/trainings/
├── tr_abc123/                       # Training directory
│   ├── config.json                  # Saved training configuration
│   ├── logs.txt                     # Training logs
│   ├── checkpoints/                 # Checkpoint directory
│   │   ├── cp_manual_abc123.pt      # Manual checkpoint
│   │   ├── cp_step_1000.pt          # Auto-save checkpoint
│   │   ├── cp_step_2000.pt          # Auto-save checkpoint
│   │   └── cp_final.pt              # Final checkpoint
│   └── metrics/                     # Optional: Exported metrics CSV
│       └── metrics.csv
└── tr_def456/
    └── ...
```

**Checkpoint File Format (.pt):**
```python
torch.save({
    'training_id': 'tr_abc123',
    'step': 4520,
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'hyperparameters': { ... },
    'metrics': {
        'loss': 0.0234,
        'sparsity': 52.3,
        'dead_neurons': 45
    },
    'rng_states': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all()
    }
}, checkpoint_path)
```

---

## 7. UI/UX Specifications

### Training Configuration Panel (TrainingPanel Component)

**Layout:** Single-column form with sections
**Color Scheme:** Emerald-600 for primary actions, slate-900/slate-800 for backgrounds
**Reference:** Mock-embedded-interp-ui.tsx lines 1628-1842

**Components:**
1. **Header:** "Training Configuration" (text-2xl font-semibold)
2. **Configuration Form:** (bg-slate-900/50 border border-slate-800 rounded-lg p-6)
   - **Dataset Dropdown:**
     - Label: "Dataset" (text-sm font-medium text-slate-300)
     - Dropdown: Shows datasets with status='ready'
     - Empty state: "Select dataset..."
   - **Model Dropdown:**
     - Label: "Model"
     - Dropdown: Shows models with status='ready'
     - Empty state: "Select model..."
   - **Encoder Type Dropdown:**
     - Label: "Encoder Type"
     - Options: "Sparse Autoencoder", "Skip Autoencoder", "Transcoder"

3. **Advanced Hyperparameters Section:** (collapsible)
   - **Toggle Button:** "Advanced Hyperparameters" with chevron-down icon (rotates 180° when expanded)
   - **Expanded Content:** (border-t border-slate-700 pt-4)
     - Grid layout: 2 columns
     - **Learning Rate:** number input, step=0.00001
     - **Batch Size:** dropdown (64, 128, 256, 512, 1024, 2048)
     - **L1 Coefficient:** number input, step=0.00001
     - **Expansion Factor:** dropdown (4x, 8x, 16x, 32x)
     - **Training Steps:** number input, min=1000
     - **Optimizer:** dropdown (Adam, AdamW, SGD)
     - **LR Schedule:** dropdown (constant, linear, cosine, exponential)
     - **Ghost Gradient Penalty:** toggle switch (emerald-600 when on, slate-600 when off)

4. **Start Training Button:**
   - Full width, bg-emerald-600 hover:bg-emerald-700
   - Disabled state: bg-slate-700 cursor-not-allowed (when dataset or model not selected)
   - Icon: Play icon (Lucide React)
   - Text: "Start Training"

5. **Training Jobs Section:**
   - Header: "Training Jobs" (text-xl font-semibold)
   - Empty state: "No training jobs yet. Configure and start training above." (centered, py-12, text-slate-400)
   - Job list: TrainingCard components for each training

---

### Training Card Component (TrainingCard)

**Layout:** Card with sections
**Reference:** Mock-embedded-interp-ui.tsx lines 1845-2156

**Structure:**

1. **Header Section:**
   - **Left:**
     - Model name + Dataset name (font-semibold text-lg)
     - Subtitle: "Encoder: {type} • Started: {time}" (text-sm text-slate-400)
   - **Right:**
     - Status icon (Activity/CheckCircle/Pause/Loader based on status)
     - Status badge (capitalize, px-3 py-1 bg-slate-800 rounded-full text-sm)

2. **Progress Section:** (only shown for training/completed/paused)
   - **Progress Bar:**
     - Label: "Training Progress" with percentage (text-slate-400 / text-emerald-400)
     - Bar: h-2 bg-slate-800 rounded-full with emerald gradient fill
   - **Metrics Grid:** (4 columns, gap-3, bg-slate-800/50 rounded-lg p-3)
     - **Loss:** emerald-400, 4 decimal places
     - **L0 Sparsity:** blue-400, 1 decimal place
     - **Dead Neurons:** red-400, integer
     - **GPU Util:** purple-400, percentage

3. **Action Buttons:** (grid-cols-2 gap-2)
   - **Show/Hide Live Metrics:** bg-slate-800 hover:bg-slate-700 with Activity icon
   - **Checkpoints (N):** bg-slate-800 hover:bg-slate-700 with download icon

4. **Live Metrics Section:** (shown when "Show Live Metrics" toggled)
   - **Loss Curve:** Bar chart visualization (h-24, emerald-500 bars)
   - **L0 Sparsity Curve:** Bar chart visualization (h-24, blue-500 bars)
   - **Training Logs:** Dark panel (bg-slate-950 rounded-lg p-4)
     - Header: "Training Logs" with "Live" indicator (text-emerald-400)
     - Log entries: monospace font, timestamped, last 10 entries
     - Auto-scroll to newest

5. **Checkpoint Management Section:** (shown when "Checkpoints" toggled)
   - **Header:** "Checkpoint Management" with "Save Now" button (bg-emerald-600 hover:bg-emerald-700)
   - **Checkpoint List:** (max-h-48 overflow-y-auto)
     - Each checkpoint: Step number, Loss value, Timestamp
     - Actions: Load button, Delete button (text-red-400)
   - **Auto-save Configuration:**
     - Toggle: "Auto-save every N steps" with switch
     - Input: Number input for interval (when enabled, 100-10000 step range)

6. **Control Buttons:** (border-t border-slate-700 pt-4)
   - **When training:**
     - Pause button (bg-yellow-600 hover:bg-yellow-700 with pause icon)
     - Stop button (bg-red-600 hover:bg-red-700 with stop icon)
   - **When paused:**
     - Resume button (bg-emerald-600 hover:bg-emerald-700 with Play icon)
     - Stop button (bg-red-600 hover:bg-red-700 with stop icon)
   - **When stopped/failed:**
     - Retry button (bg-blue-600 hover:bg-blue-700 with retry icon)

---

### Interaction Patterns

**Real-time Updates:**
- WebSocket connection subscribes to `training:{training_id}` channel on component mount
- Progress bar animates smoothly with CSS transitions
- Metrics update every 2 seconds (via WebSocket, not polling)
- Chart bars animate with height transitions

**User Feedback:**
- Button clicks trigger immediate UI state change (optimistic updates)
- Failed actions show error toast notification
- Successful actions show success toast (e.g., "Checkpoint saved successfully")

**Responsive Behavior:**
- Metrics grid collapses to 2 columns on medium screens
- Training list shows 3 cards initially, load more on scroll
- Charts scale responsively to container width

---

## 8. Testing Requirements

### Unit Tests
- Test hyperparameter validation logic (ranges, types, required fields)
- Test SAE architecture initialization (dimensions, weight initialization)
- Test training step calculation (loss, sparsity, dead neurons)
- Test checkpoint save/load roundtrip (state preservation)
- Test status transition logic (valid/invalid transitions)
- Test memory estimation calculations

### Integration Tests
- Test end-to-end training flow: configure → start → progress → complete
- Test pause/resume functionality with checkpoint integrity
- Test OOM error handling and batch size reduction
- Test WebSocket event emission and client reception
- Test concurrent training jobs (resource allocation)
- Test checkpoint retention policy (automatic cleanup)

### Edge Case Tests
- Test training with minimal batch size (16) and gradient accumulation
- Test training with maximum expansion factor (32x) near memory limit
- Test training with empty dataset (should fail gracefully)
- Test training with corrupted checkpoint (should detect and fail)
- Test training with manual stop at step 0 (no metrics available)
- Test training with extremely long training_steps (1,000,000) for UI display

### Performance Tests
- Benchmark training throughput (steps/second) on Jetson Orin Nano
- Measure checkpoint save time for various model sizes
- Measure WebSocket latency for metrics updates
- Measure database query performance for metrics retrieval (10,000 steps)

---

## 9. Dependencies

### Upstream Dependencies (Must be complete before implementation)
- **Dataset Management (001_FPRD):** Requires tokenized datasets for training
- **Model Management (002_FPRD):** Requires models and activation extraction capability

### Downstream Dependents (Blocked until this feature is complete)
- **Feature Discovery (004_FPRD):** Requires trained SAE models to extract features
- **Model Steering (005_FPRD):** Requires interpretable features from trained SAEs

### External Dependencies
- **PyTorch:** 2.0+ with CUDA support for GPU training
- **bitsandbytes:** For quantization and memory-efficient optimizers
- **safetensors:** For secure checkpoint serialization
- **Celery:** For distributed training job execution
- **Redis:** For Celery task queue
- **WebSocket:** For real-time progress updates

---

## 10. Risks & Mitigations

### Technical Risks

**Risk 1: GPU Memory Exhaustion on Jetson Orin Nano**
- **Likelihood:** High
- **Impact:** Critical (training cannot proceed)
- **Mitigation:**
  - Implement dynamic batch size reduction on OOM
  - Validate memory requirements before training start
  - Use automatic mixed precision (AMP) to reduce memory 40%
  - Implement gradient accumulation for small batches
  - Clear GPU cache between batches
  - Provide clear warnings when user selects memory-intensive configurations

**Risk 2: Training Instability (Divergence, NaN Loss)**
- **Likelihood:** Medium
- **Impact:** High (wasted training time)
- **Mitigation:**
  - Implement gradient clipping (max_norm=1.0)
  - Validate hyperparameter ranges before training
  - Monitor for NaN/Inf in loss and terminate early
  - Provide tested default hyperparameters
  - Log warnings for concerning metric trends (loss plateau, excessive dead neurons)

**Risk 3: Checkpoint Corruption**
- **Likelihood:** Low
- **Impact:** Critical (loss of training progress)
- **Mitigation:**
  - Use safetensors format (safer than pickle)
  - Implement atomic checkpoint saves (write to temp file, then rename)
  - Validate checkpoint integrity before deletion of previous checkpoint
  - Test checkpoint save/load roundtrip in unit tests

**Risk 4: WebSocket Connection Loss**
- **Likelihood:** Medium
- **Impact:** Low (metrics not displayed, training continues)
- **Mitigation:**
  - Implement automatic WebSocket reconnection with exponential backoff
  - Poll for latest metrics on reconnection to fill gaps
  - Display connection status in UI
  - Continue training even if WebSocket is disconnected

---

### Usability Risks

**Risk 5: Users Configure Unrealistic Hyperparameters**
- **Likelihood:** High (especially for beginners)
- **Impact:** Medium (wasted time, poor results)
- **Mitigation:**
  - Provide sensible defaults tested on target hardware
  - Display warnings for extreme configurations (e.g., batch_size=2048 on edge device)
  - Show estimated memory usage before training start
  - Include tooltips explaining each hyperparameter's effect
  - Consider implementing hyperparameter presets/templates

**Risk 6: Training Progress is Unclear**
- **Likelihood:** Medium
- **Impact:** Medium (user abandons training prematurely)
- **Mitigation:**
  - Display estimated time remaining
  - Show clear progress percentage
  - Provide real-time metrics with trend visualizations
  - Include explanations of what each metric means
  - Log key milestones (10%, 50%, 90% complete)

---

## 11. Future Enhancements

### Post-MVP Features (Not included in initial implementation)

1. **Distributed Training Across Multiple GPUs**
   - Support for multi-GPU training with data parallelism
   - Automatic load balancing across available GPUs
   - Priority: P3, Timeline: Q2 2026

2. **Hyperparameter Auto-tuning**
   - Bayesian optimization for hyperparameter search
   - Automated L1 coefficient tuning for target sparsity
   - Priority: P2, Timeline: Q3 2026

3. **Training Comparison Dashboard**
   - Side-by-side metric comparison for multiple trainings
   - Export comparison charts as images
   - Priority: P2, Timeline: Q2 2026

4. **Resume from External Checkpoint**
   - Import checkpoint files from filesystem
   - Resume training from externally trained SAE
   - Priority: P3, Timeline: Q4 2026

5. **Advanced SAE Architectures**
   - Gated SAE variant
   - Top-K activation function
   - Multi-layer SAE stacking
   - Priority: P2, Timeline: Q3 2026

6. **Training Template Library**
   - Pre-configured templates for common model/dataset combinations
   - Community-shared templates
   - Priority: P2, Timeline: Q2 2026

---

## 12. Open Questions

### Questions for Stakeholders

1. **Checkpoint Retention:** Should the system automatically delete old checkpoints beyond a certain limit, or require manual deletion?
   - **Recommendation:** Implement retention policy (keep first, last, every 1000 steps, best loss) but allow user override

2. **Multi-GPU Support:** Should MVP support multi-GPU training, or defer to post-MVP?
   - **Recommendation:** Defer to post-MVP (single GPU sufficient for edge deployment)

3. **Training Pause Timeout:** Should paused trainings auto-stop after a certain period (e.g., 24 hours) to free resources?
   - **Recommendation:** Yes, auto-stop after 24 hours with notification

4. **Error Notification:** Should training errors send email/notification to user, or just update UI?
   - **Recommendation:** MVP uses UI-only notifications, defer email to post-MVP

5. **Checkpoint Size Limits:** Should the system enforce maximum checkpoint storage per training (e.g., 10 checkpoints max)?
   - **Recommendation:** Yes, enforce limit of 20 checkpoints per training (configurable in settings)

---

## 13. Success Metrics

### Feature Adoption Metrics
- **Training Job Creation Rate:** Number of training jobs created per week
- **Training Completion Rate:** Percentage of trainings that complete successfully (target: >90%)
- **Checkpoint Usage Rate:** Percentage of trainings that use manual checkpoints (target: >30%)

### Quality Metrics
- **Dead Neuron Rate:** Percentage of trained SAEs with <5% dead neurons (target: >85%)
- **L0 Sparsity Range:** Percentage of trainings achieving L0 sparsity 30-70 (target: >80%)
- **Training Stability:** Percentage of trainings that complete without OOM or NaN errors (target: >95%)

### Performance Metrics
- **Training Throughput:** Average steps/second on Jetson Orin Nano (target: >10 steps/sec)
- **Checkpoint Save Time:** Average time to save checkpoint (target: <5 seconds)
- **Metrics Update Latency:** Average latency from database write to UI update (target: <2 seconds)

### User Experience Metrics
- **Configuration Time:** Average time from opening Training panel to starting training (target: <2 minutes)
- **Error Resolution Rate:** Percentage of failed trainings successfully resolved by retry (target: >70%)
- **User Satisfaction:** Post-feature survey rating (target: >4/5 stars)

---

## 14. Appendices

### A. Glossary

- **SAE (Sparse Autoencoder):** Neural network trained to reconstruct input activations with sparsity constraint
- **L0 Sparsity:** Average number of active (non-zero) neurons in SAE encoding
- **L1 Coefficient:** Hyperparameter controlling sparsity penalty strength (higher = sparser)
- **Expansion Factor:** Ratio of SAE hidden dimension to model hidden dimension (e.g., 8x means SAE has 8× more neurons)
- **Dead Neurons:** SAE neurons that rarely activate (<1% of samples)
- **Ghost Gradient Penalty:** Regularization technique to prevent dead neurons
- **Reconstruction Loss:** Mean squared error between SAE output and original activations
- **Checkpoint:** Saved snapshot of model weights and training state for resume/analysis
- **Learning Rate Schedule:** Strategy for adjusting learning rate during training (constant/linear/cosine/exponential)

### B. References

**Research Papers:**
- Anthropic (2023): "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
- Bricken et al. (2023): "Towards Automated Circuit Discovery for Mechanistic Interpretability"

**Technical Documentation:**
- PyTorch Mixed Precision Training: https://pytorch.org/docs/stable/amp.html
- PyTorch Checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- Celery Documentation: https://docs.celeryproject.org/

**UI/UX Reference:**
- Mock-embedded-interp-ui.tsx (PRIMARY REFERENCE)
- Material Design Progress Indicators
- Tailwind CSS Dark Theme Patterns

### C. Implementation Example: SAE Training Loop

```python
# Simplified training loop pseudocode

def train_sae(training_id, config):
    # Initialize
    model = load_model(config.model_id)
    dataset = load_dataset(config.dataset_id)
    sae = create_sae(
        input_dim=model.hidden_size,
        hidden_dim=model.hidden_size * config.expansion_factor,
        encoder_type=config.encoder_type
    )
    optimizer = create_optimizer(config.optimizer, lr=config.learning_rate)
    scheduler = create_scheduler(config.lr_schedule, optimizer, config.training_steps)

    # Update status
    update_training_status(training_id, 'training')

    # Training loop
    for step in range(config.training_steps):
        # Check for pause/stop signals
        if check_pause_signal(training_id):
            save_checkpoint(training_id, step, sae, optimizer, scheduler)
            update_training_status(training_id, 'paused')
            return

        # Get batch
        batch = dataset.get_batch(config.batch_size)
        input_ids = tokenize(batch)

        # Extract activations
        with torch.no_grad():
            activations = extract_activations(model, input_ids, layer=config.layer)

        # Forward pass
        with torch.cuda.amp.autocast():  # Mixed precision
            encoded = sae.encoder(activations)
            reconstructed = sae.decoder(encoded)

            # Calculate losses
            recon_loss = F.mse_loss(reconstructed, activations)
            l1_penalty = config.l1_coefficient * encoded.abs().sum(dim=-1).mean()

            if config.ghost_grad_penalty:
                ghost_penalty = calculate_ghost_penalty(encoded, sae.encoder.weight)
                total_loss = recon_loss + l1_penalty + ghost_penalty
            else:
                total_loss = recon_loss + l1_penalty

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Calculate metrics
        l0_sparsity = (encoded > 0.01).float().sum(dim=-1).mean().item()
        dead_neurons = count_dead_neurons(encoded)

        # Log metrics every 10 steps
        if step % 10 == 0:
            save_metrics(training_id, step, {
                'loss': total_loss.item(),
                'sparsity': l0_sparsity,
                'reconstruction_error': recon_loss.item(),
                'dead_neurons': dead_neurons,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            emit_websocket_event('training:progress', {
                'training_id': training_id,
                'step': step,
                'progress': (step / config.training_steps) * 100,
                'metrics': { ... }
            })

        # Auto-save checkpoints
        if config.auto_save and step % config.auto_save_interval == 0:
            save_checkpoint(training_id, step, sae, optimizer, scheduler)

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Training complete
    save_checkpoint(training_id, config.training_steps, sae, optimizer, scheduler, final=True)
    update_training_status(training_id, 'completed')
    emit_websocket_event('training:completed', {'training_id': training_id})
```

---

**Document End**
**Total Sections:** 14
**Total Functional Requirements:** 95 (across FR-1 through FR-8)
**Estimated Implementation Time:** 4-6 weeks (2 developers)
**Review Status:** Pending stakeholder review
