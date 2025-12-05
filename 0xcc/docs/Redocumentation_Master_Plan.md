# miStudio Redocumentation Master Plan

**Purpose**: Archive accumulated development artifacts and create clean XCC-compliant documentation that accurately reflects the implemented MVP.

**Date**: December 2025
**Status**: Planning

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Archive Plan](#archive-plan)
3. [New Documentation Structure](#new-documentation-structure)
4. [Implemented Features Inventory](#implemented-features-inventory)
5. [Document Creation Sequence](#document-creation-sequence)
6. [Execution Plan](#execution-plan)

---

## Current State Analysis

### Documentation Inventory (To Archive)

#### Tasks Directory (34 files, ~970KB)
Files accumulated during development that should be archived:

| Category | Count | Examples |
|----------|-------|----------|
| Original FTASKS | 8 | `001_FTASKS|Dataset_Management.md`, `003_FTASKS|SAE_Training.md` |
| Enhancement FTASKS | 9 | `ENH_FTASKS|*.md`, `*-ENH_0*.md` |
| Supplemental FTASKS | 5 | `SUPP_TASKS|*.md` |
| UX FTASKS | 2 | `UX-001_FTASKS|Theme_Toggle.md` |
| Bugfix/Issues | 4 | `BUGFIX_*.md`, `FIX_*.md`, `ISSUES.md` |
| Plans/Summaries | 6 | `*_PLAN_*.md`, `*_SUMMARY_*.md` |

#### Docs Directory (70+ files, ~1.5MB)
Development notes, session summaries, and investigation documents:

| Category | Count | Archive? |
|----------|-------|----------|
| Session Summaries | 8 | Yes |
| Investigation/Debug | 15 | Yes |
| Implementation Plans | 12 | Yes |
| Guides (keep some) | 10 | Selective |
| Phase Summaries | 5 | Yes |
| Bug Fix Notes | 8 | Yes |
| Comparison/Analysis | 6 | Yes |
| **Developer_Guide.md** | 1 | **KEEP** |

#### Images Directory (40+ files, ~9.5MB)
UI screenshots from development - all should be archived.

#### Project-Specs Directory
Legacy specifications that predate implementation - archive entirely.

#### TDDs/TIDs (14 files)
Original design documents from October - outdated, need fresh versions.

#### PRDs (7 files)
Original PRDs - some may be updatable, most need rewriting.

---

## Archive Plan

### Archive Directory Structure

```
0xcc/archive/                        # New archive root
├── legacy_tasks/                    # All old task files
├── legacy_docs/                     # Session summaries, investigations
├── legacy_prds/                     # Original PRDs
├── legacy_tdds/                     # Original TDDs
├── legacy_tids/                     # Original TIDs
├── legacy_project_specs/            # Original project specs
├── legacy_images/                   # Development screenshots
├── legacy_transcripts/              # Session transcripts
├── legacy_session_summaries/        # Session summary files
└── README.md                        # Archive index/explanation
```

### Files to Keep (Not Archive)

```
0xcc/
├── instruct/                        # XCC framework instructions (KEEP)
├── docs/
│   └── Developer_Guide.md           # Recent comprehensive guide (KEEP)
├── scripts/                         # Utility scripts (KEEP)
└── adrs/
    └── 000_PADR|miStudio.md         # Update in place
```

---

## New Documentation Structure

### XCC-Compliant Structure

```
0xcc/
├── prds/                            # Product Requirements Documents
│   ├── 000_PPRD|miStudio.md         # Project PRD (REWRITE)
│   ├── 001_FPRD|Dataset_Management.md
│   ├── 002_FPRD|Model_Management.md
│   ├── 003_FPRD|SAE_Training.md
│   ├── 004_FPRD|Feature_Discovery.md
│   ├── 005_FPRD|SAE_Management.md
│   ├── 006_FPRD|Model_Steering.md
│   ├── 007_FPRD|Neuronpedia_Export.md
│   └── 008_FPRD|System_Monitoring.md
│
├── adrs/                            # Architecture Decision Records
│   └── 000_PADR|miStudio.md         # (UPDATE in place)
│
├── tdds/                            # Technical Design Documents
│   ├── 001_FTDD|Dataset_Management.md
│   ├── 002_FTDD|Model_Management.md
│   ├── 003_FTDD|SAE_Training.md
│   ├── 004_FTDD|Feature_Discovery.md
│   ├── 005_FTDD|SAE_Management.md
│   ├── 006_FTDD|Model_Steering.md
│   ├── 007_FTDD|Neuronpedia_Export.md
│   └── 008_FTDD|System_Monitoring.md
│
├── tids/                            # Technical Implementation Documents
│   ├── 001_FTID|Dataset_Management.md
│   ├── 002_FTID|Model_Management.md
│   ├── 003_FTID|SAE_Training.md
│   ├── 004_FTID|Feature_Discovery.md
│   ├── 005_FTID|SAE_Management.md
│   ├── 006_FTID|Model_Steering.md
│   ├── 007_FTID|Neuronpedia_Export.md
│   └── 008_FTID|System_Monitoring.md
│
├── tasks/                           # Implementation Task Lists
│   ├── 001_FTASKS|Dataset_Management.md
│   ├── 002_FTASKS|Model_Management.md
│   ├── 003_FTASKS|SAE_Training.md
│   ├── 004_FTASKS|Feature_Discovery.md
│   ├── 005_FTASKS|SAE_Management.md
│   ├── 006_FTASKS|Model_Steering.md
│   ├── 007_FTASKS|Neuronpedia_Export.md
│   └── 008_FTASKS|System_Monitoring.md
│
├── docs/                            # Reference Documentation
│   ├── Developer_Guide.md           # (KEEP - already created)
│   └── CHANGELOG.md                 # Feature changelog (NEW)
│
├── instruct/                        # XCC Framework (KEEP)
│
├── scripts/                         # Utility Scripts (KEEP)
│
└── archive/                         # Archived legacy docs (NEW)
```

---

## Implemented Features Inventory

Based on code review and git history, these are the **8 core implemented features**:

### 1. Dataset Management
**Status**: Complete (MVP)

**Implemented Capabilities**:
- HuggingFace dataset download with progress tracking
- Multi-tokenization support (multiple tokenizers per dataset)
- Token filtering (min length, special tokens, stop words)
- Statistics visualization (vocabulary distribution, sequence lengths)
- Sample browser with pagination
- Dataset deletion with file cleanup

**Backend Components**:
- `DatasetService`, `TokenizationService`
- `dataset_tasks.py` (Celery tasks)
- Models: `Dataset`, `DatasetTokenization`

**Frontend Components**:
- `DatasetsPanel`, `DatasetCard`, `DownloadForm`
- `datasetsStore.ts`

### 2. Model Management
**Status**: Complete (MVP)

**Implemented Capabilities**:
- HuggingFace model download with authentication
- Quantization support (4-bit, 8-bit via bitsandbytes)
- Memory estimation before download
- Model architecture viewer
- Model deletion

**Backend Components**:
- `ModelService`, `model_tasks.py`
- Models: `Model`

**Frontend Components**:
- `ModelsPanel`, `ModelCard`, `ModelDownloadForm`
- `modelsStore.ts`

### 3. SAE Training
**Status**: Complete (MVP)

**Implemented Capabilities**:
- 4 SAE architectures (Standard, JumpReLU, Skip, Transcoder)
- Real-time metrics (loss, L0, L1, FVU, reconstruction error)
- Checkpoint management with auto-save
- Dead neuron detection and resampling
- Top-K sparsity for guaranteed sparsity
- Training templates for reproducibility
- Hyperparameter optimization hints

**Backend Components**:
- `TrainingService`, `SparseAutoencoder`, `JumpReLUSAE`
- `training_tasks.py`
- Models: `Training`, `TrainingMetric`, `Checkpoint`, `TrainingTemplate`

**Frontend Components**:
- `TrainingPanel`, `TrainingCard`, `StartTrainingModal`
- `trainingsStore.ts`, `trainingTemplatesStore.ts`

### 4. Feature Discovery
**Status**: Complete (MVP)

**Implemented Capabilities**:
- Batch activation extraction with GPU optimization
- Feature statistics (frequency, max, mean, interpretability)
- Context window capture (before/after tokens)
- Token filtering during extraction
- Dual labeling (semantic + category)
- GPT-4o auto-labeling integration
- Labeling prompt templates
- Feature search and filtering
- Example export to JSON

**Backend Components**:
- `ExtractionService`, `FeatureService`, `LabelingService`
- `extraction_tasks.py`, `labeling_tasks.py`
- Models: `Feature`, `FeatureActivation`, `ExtractionJob`, `LabelingJob`

**Frontend Components**:
- `ExtractionsPanel`, `FeaturesPanel`, `FeatureDetailModal`
- `featuresStore.ts`, `labelingStore.ts`

### 5. SAE Management
**Status**: Complete (MVP)

**Implemented Capabilities**:
- Trained SAE management (from training jobs)
- External SAE download from HuggingFace
- Gemma Scope SAE support
- Community Standard format (SAELens-compatible)
- Format auto-detection and conversion
- Upload to HuggingFace

**Backend Components**:
- `SAEManagerService`, `HuggingFaceSAEService`, `SAEConverter`
- `sae_tasks.py`
- Models: `ExternalSAE`

**Frontend Components**:
- `SAEsPanel`, `SAECard`, `DownloadFromHF`, `UploadToHF`
- `saesStore.ts`

### 6. Model Steering
**Status**: Complete (MVP)

**Implemented Capabilities**:
- Multi-feature steering (combine multiple interventions)
- Activation steering and suppression
- Strength sweep (test multiple intensities)
- Comparison mode (steered vs. unsteered)
- Neuronpedia-compatible calibration
- Prompt templates for experiments
- Feature browser with search
- Results export

**Backend Components**:
- `SteeringService`, `forward_hooks.py`
- Schemas: `SteeringComparisonRequest`, etc.

**Frontend Components**:
- `SteeringPanel`, `FeatureBrowser`, `ComparisonResults`
- `steeringStore.ts`, `promptTemplatesStore.ts`

### 7. Neuronpedia Export
**Status**: Complete (MVP)

**Implemented Capabilities**:
- Feature activation examples export
- Logit lens data (promoted/suppressed tokens)
- Activation histograms
- Feature explanations/labels
- SAELens-compatible weights export
- README generation
- ZIP archive packaging
- Job progress tracking

**Backend Components**:
- `NeuronpediaExportService`, `LogitLensService`
- `neuronpedia_tasks.py`
- Models: `NeuronpediaExportJob`, `FeatureDashboardData`

**Frontend Components**:
- `ExportToNeuronpedia`
- `neuronpediaExportStore.ts`

### 8. System Monitoring
**Status**: Complete (MVP)

**Implemented Capabilities**:
- GPU metrics (utilization, memory, temperature, power)
- CPU per-core utilization
- RAM and swap usage
- Disk I/O rates
- Network I/O rates
- Real-time WebSocket streaming
- Historical data charts

**Backend Components**:
- `SystemMonitorService`, `system_monitor_tasks.py`
- Celery Beat scheduled task

**Frontend Components**:
- `SystemMonitor`, `UtilizationChart`
- `systemMonitorStore.ts`

---

## Document Creation Sequence

### Phase 1: Archive (1 task)
1. Create archive directory structure
2. Move all legacy files
3. Create archive README

### Phase 2: Project Level (2 documents)
1. **000_PPRD|miStudio.md** - Fresh Project PRD reflecting actual MVP
2. **000_PADR|miStudio.md** - Update ADR with actual decisions

### Phase 3: Feature Documentation (8 features x 4 docs = 32 documents)

For each feature, create in order:
1. **FPRD** - What was built (requirements as implemented)
2. **FTDD** - How it was designed (architecture)
3. **FTID** - How to implement/extend (guidance)
4. **FTASKS** - What was done (completed tasks)

**Suggested Order** (by complexity):
1. System Monitoring (simplest)
2. Dataset Management
3. Model Management
4. SAE Training
5. Feature Discovery
6. SAE Management
7. Model Steering
8. Neuronpedia Export

### Phase 4: Cleanup (1 task)
1. Update CLAUDE.md to reference new docs
2. Create CHANGELOG.md
3. Verify documentation completeness

---

## Execution Plan

### Step 1: Archive Legacy Documentation

```bash
# Create archive structure
mkdir -p 0xcc/archive/{legacy_tasks,legacy_docs,legacy_prds,legacy_tdds,legacy_tids,legacy_project_specs,legacy_images,legacy_transcripts,legacy_session_summaries}

# Move files (preserving structure)
mv 0xcc/tasks/* 0xcc/archive/legacy_tasks/
mv 0xcc/docs/* 0xcc/archive/legacy_docs/  # except Developer_Guide.md
mv 0xcc/prds/* 0xcc/archive/legacy_prds/
mv 0xcc/tdds/* 0xcc/archive/legacy_tdds/
mv 0xcc/tids/* 0xcc/archive/legacy_tids/
mv 0xcc/project-specs/* 0xcc/archive/legacy_project_specs/
mv 0xcc/img/* 0xcc/archive/legacy_images/
mv 0xcc/transcripts/* 0xcc/archive/legacy_transcripts/
mv 0xcc/session_summaries/* 0xcc/archive/legacy_session_summaries/
```

### Step 2: Create New Documents

Each document follows the XCC template structure from `0xcc/instruct/`.

**Estimated effort per feature**:
- FPRD: 30-60 min (reverse-engineer from code)
- FTDD: 60-90 min (document actual architecture)
- FTID: 30-45 min (implementation guidance)
- FTASKS: 45-60 min (document what was built)

**Total estimated effort**:
- Archive: 30 min
- Project docs: 2-3 hours
- Feature docs (8 x 4): 16-24 hours
- Cleanup: 1 hour
- **Total: 20-28 hours**

### Step 3: Validation

For each document:
1. Cross-reference with actual code
2. Verify all components are documented
3. Ensure another Claude Code instance could recreate the feature
4. Test that git history milestones are captured

---

## Success Criteria

1. **Archive Complete**: All legacy docs moved, zipped, gitignored
2. **Clean Structure**: Only XCC-compliant docs in active directories
3. **Accuracy**: Documents reflect actual implementation, not aspirational plans
4. **Reproducibility**: A new Claude Code instance could rebuild miStudio from docs
5. **Completeness**: All 8 features have full doc sets (FPRD, FTDD, FTID, FTASKS)
6. **Traceability**: Git commits can be traced to documented features

---

## Next Steps

**Immediate Action Required**:

1. **Approve this plan** or request modifications
2. **Execute archive** (can be done immediately)
3. **Begin document creation** (start with Project PRD)

**Questions for User**:

1. Should we keep any specific docs from the legacy set?
2. Are there features I missed in the inventory?
3. Should templates/extraction templates be documented as separate features or sub-features?
4. Preference on archive format (keep unzipped for reference vs. zip immediately)?

---

*Plan created: December 2025*
*miStudio Redocumentation Initiative*
