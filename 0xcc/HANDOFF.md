# miStudio Documentation Handoff

**Created:** 2025-10-06
**Status:** In Progress - TDD/TID Creation Phase
**Next Session Focus:** Complete remaining 3 TDDs and 5 TIDs

---

## Work Completed (100%)

### ✅ Feature PRDs (5/5 Complete)

All Feature PRDs have been created following the XCC Framework structure with Mock UI as PRIMARY reference:

1. **001_FPRD|Dataset_Management.md** (~1,200 lines)
   - 16 user stories, 45 functional requirements
   - HuggingFace download, tokenization, browsing
   - UI Reference: Mock UI lines 1108-1202 (DatasetsPanel)

2. **002_FPRD|Model_Management.md** (~1,300 lines)
   - 5 user stories, 84 functional requirements
   - Model download, quantization, activation extraction
   - UI Reference: Mock UI lines 1204-1343 (ModelsPanel), 1346-1437 (ModelArchitectureViewer), 1440-1625 (ActivationExtractionConfig)

3. **003_FPRD|SAE_Training.md** (~1,500 lines)
   - 7 user stories, 95 functional requirements
   - Training configuration, execution, checkpoint management
   - UI Reference: Mock UI lines 1628-1842 (TrainingPanel), 1845-2156 (TrainingCard)

4. **004_FPRD|Feature_Discovery.md** (~1,400 lines)
   - 9 user stories, 76 functional requirements
   - Feature extraction, browsing, analysis (logit lens, correlations, ablation)
   - UI Reference: Mock UI lines 2159-2584 (FeaturesPanel), 2587-2725 (FeatureDetailModal)

5. **005_FPRD|Model_Steering.md** (~1,300 lines)
   - 7 user stories, 71 functional requirements
   - Feature selection, coefficient sliders, comparative generation
   - UI Reference: Mock UI lines 3512-3952 (SteeringPanel)

**Total PRD Lines:** ~6,700 lines across 5 comprehensive documents

---

### ✅ Technical Design Documents (2/5 Complete)

1. **001_FTDD|Dataset_Management.md** (43KB, ~600 lines)
   - Complete 14-section TDD structure
   - System architecture with component diagrams
   - FastAPI + Celery + Redis architecture
   - PostgreSQL schema, API design, WebSocket protocol
   - React component hierarchy with Zustand state management
   - 5-week development timeline

2. **002_FTDD|Model_Management.md** (56KB, ~1,647 lines)
   - Comprehensive TDD following same 14-section structure
   - PyTorch model loading with quantization (bitsandbytes)
   - Forward hooks for activation extraction
   - Caching strategy for performance
   - Memory optimization for Jetson Orin Nano
   - 6-week development timeline

---

## Work Remaining (In Priority Order)

### 🔲 TDD Documents (3 remaining)

#### 1. 003_FTDD|SAE_Training.md (HIGHEST PRIORITY)
**Complexity:** HIGH (Most complex feature)
**Estimated Size:** 60-70KB (~1,800 lines)
**Key Technical Areas:**
- Training loop architecture (PyTorch training pipeline)
- SAE model architecture (encoder-decoder with sparsity)
- Checkpoint management (save/load with safetensors)
- Memory optimization for edge device (gradient accumulation, mixed precision)
- Real-time metrics (WebSocket streaming)
- Celery worker for distributed training

**Reference Documents:**
- PRD: `003_FPRD|SAE_Training.md`
- Mock UI: Lines 1628-1842 (TrainingPanel), 1845-2156 (TrainingCard), 2031-2086 (metrics)
- Database Schema: `003_SPEC|Postgres_Usecase_Details_and_Guidance.md` lines 223-346 (trainings, training_metrics, checkpoints tables)

**Key Technical Decisions Needed:**
- Training loop implementation (PyTorch training step)
- SAE architecture variants (sparse, skip, transcoder)
- Checkpoint format (safetensors vs PyTorch .pt)
- GPU memory management (batch size reduction, gradient accumulation)
- Metrics calculation (L0 sparsity, dead neurons)

---

#### 2. 004_FTDD|Feature_Discovery.md
**Complexity:** MEDIUM-HIGH
**Estimated Size:** 50-60KB (~1,500 lines)
**Key Technical Areas:**
- Feature extraction pipeline (SAE encoder inference on eval samples)
- Max-activating examples identification (top-K selection)
- Token-level activation highlighting (React visualization)
- Analysis algorithms (logit lens, correlations, ablation)
- Full-text search with PostgreSQL GIN indexes
- Pagination and filtering

**Reference Documents:**
- PRD: `004_FPRD|Feature_Discovery.md`
- Mock UI: Lines 2159-2584 (FeaturesPanel), 2434-2567 (feature browser table), 2587-2725 (FeatureDetailModal), 2728-2800 (MaxActivatingExamples)
- Database Schema: `003_SPEC|Postgres_Usecase_Details_and_Guidance.md` lines 385-477 (features, feature_activations tables)

**Key Technical Decisions Needed:**
- Extraction algorithm (batch processing, progress tracking)
- Token highlighting calculation (intensity normalization)
- Analysis caching strategy (feature_analysis_cache table)
- Automated labeling heuristics

---

#### 3. 005_FTDD|Model_Steering.md
**Complexity:** MEDIUM
**Estimated Size:** 45-55KB (~1,400 lines)
**Key Technical Areas:**
- Forward hook implementation (PyTorch hook system)
- Steering algorithm (multiplicative vs additive coefficients)
- Side-by-side generation (unsteered vs steered)
- Comparison metrics (KL divergence, perplexity delta, semantic similarity)
- Real-time generation progress

**Reference Documents:**
- PRD: `005_FPRD|Model_Steering.md`
- Mock UI: Lines 3512-3952 (SteeringPanel), 3582-3706 (feature selection), 3820-3895 (comparative output), 3898-3947 (metrics)

**Key Technical Decisions Needed:**
- Steering coefficient application method (multiplicative recommended)
- Hook registration timing (before generation)
- Metrics calculation implementation (sentence transformers for similarity)
- Generation timeout and error handling

---

### 🔲 TID Documents (5 remaining)

All TIDs should follow the structure from `@0xcc/instruct/005_create-tid.md`:

1. **001_FTID|Dataset_Management.md** (~30-40KB)
   - File organization patterns
   - Component implementation hints (DatasetsPanel, DatasetCard, DatasetDetailModal)
   - Celery task implementation (download_dataset_task, tokenize_dataset_task)
   - Database operations (SQLAlchemy async patterns)
   - WebSocket emit patterns

2. **002_FTID|Model_Management.md** (~35-45KB)
   - Model loading implementation (transformers library)
   - Quantization setup (bitsandbytes config)
   - Forward hook registration (activation extraction)
   - Caching implementation (activation_cache table)
   - Memory monitoring (CUDA memory tracking)

3. **003_FTID|SAE_Training.md** (~40-50KB)
   - SAE model class implementation (encoder, decoder, loss functions)
   - Training loop structure (step-by-step pseudocode)
   - Checkpoint I/O (save/load with metadata)
   - Metrics calculation (L0 sparsity, dead neurons)
   - GPU memory optimization (mixed precision, gradient accumulation)

4. **004_FTID|Feature_Discovery.md** (~35-45KB)
   - Extraction pipeline implementation (batch processing loop)
   - Token highlighting logic (intensity calculation, CSS styles)
   - Analysis implementation (logit lens, correlations, ablation)
   - Database queries (PostgreSQL GIN search, pagination)

5. **005_FTID|Model_Steering.md** (~30-40KB)
   - Forward hook function implementation
   - Coefficient application logic
   - Generation comparison implementation (parallel or sequential)
   - Metrics calculation (KL divergence, perplexity, similarity)

---

## Key Patterns Established

### TDD Structure (14 sections - use as template)

```
1. Executive Summary
2. System Architecture (with ASCII diagrams)
3. Technical Stack (table format with justifications)
4. Data Design (PostgreSQL schema, validation patterns)
5. API Design (REST endpoints, WebSocket protocol)
6. Component Architecture (React hierarchy, Zustand stores)
7. State Management (patterns for real-time updates)
8. Security Considerations (auth, validation, sanitization)
9. Performance & Scalability (optimization strategies)
10. Testing Strategy (unit, integration, e2e)
11. Deployment & DevOps (Docker, CI/CD)
12. Risk Assessment (risks + mitigations table)
13. Development Phases (week-by-week timeline)
14. Appendix (technology decisions, alternatives, references)
```

### Common Technical Patterns Across Features

**Frontend (React + TypeScript):**
```typescript
// Component structure
export const FeaturePanel: React.FC = () => {
  const items = useFeatureStore((state) => state.items);
  const { subscribe } = useWebSocket();

  useEffect(() => {
    fetchItems();
    items.forEach(item => {
      if (item.status === 'processing') {
        subscribe(`feature/${item.id}`, handleUpdate);
      }
    });
  }, [items]);

  return (/* JSX matching Mock UI exactly */);
};

// Zustand store pattern
export const useFeatureStore = create<FeatureStore>()(
  devtools((set, get) => ({
    items: [],
    loading: false,
    fetchItems: async () => { /* ... */ },
    updateItem: (id, updates) => { /* ... */ }
  }))
);
```

**Backend (FastAPI + Celery):**
```python
# API route pattern
@router.post("/")
async def create_item(
    request: ItemCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    # Validate
    validate_input(request)

    # Create database record
    item = await ItemService.create(db, request)

    # Enqueue background task
    process_item_task.delay(item.id)

    return item

# Celery task pattern
@celery_app.task
def process_item_task(item_id: str):
    # Long-running processing
    for progress in range(0, 100):
        # Update database
        update_progress(item_id, progress)

        # Emit WebSocket event
        emit_websocket(f"item/{item_id}", {
            "type": "progress",
            "progress": progress
        })
```

**Database (PostgreSQL + SQLAlchemy):**
```sql
-- Standard table pattern
CREATE TABLE items (
    id VARCHAR(255) PRIMARY KEY,  -- Format: itm_{uuid}
    name VARCHAR(500) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress FLOAT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT items_status_check CHECK (status IN (...))
);

CREATE INDEX idx_items_status ON items(status);
CREATE INDEX idx_items_created_at ON items(created_at DESC);
```

---

## Critical References for Next Session

### Primary Documents
1. **Mock UI:** `/home/x-sean/app/miStudio/0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
   - PRIMARY source of truth for UI/UX
   - All components must match exactly (styling, behavior, layout)

2. **ADR:** `/home/x-sean/app/miStudio/0xcc/adrs/000_PADR|miStudio.md`
   - All technical standards and architectural decisions
   - Technology stack, coding patterns, deployment strategy

3. **Infrastructure Specs:**
   - `/home/x-sean/app/miStudio/0xcc/project-specs/infrastructure/001_SPEC|Folder_File_Details.md` (file structure)
   - `/home/x-sean/app/miStudio/0xcc/project-specs/infrastructure/003_SPEC|Postgres_Usecase_Details_and_Guidance.md` (database schema)

### Feature-Specific PRDs
- `001_FPRD|Dataset_Management.md` through `005_FPRD|Model_Steering.md`

### Completed TDD Templates
- `001_FTDD|Dataset_Management.md` (43KB) - Use as structure template
- `002_FTDD|Model_Management.md` (56KB) - Use as technical depth reference

### Framework Instructions
- `/home/x-sean/app/miStudio/0xcc/instruct/004_create-tdd.md` (TDD guidelines)
- `/home/x-sean/app/miStudio/0xcc/instruct/005_create-tid.md` (TID guidelines)

---

## Resume Instructions for Next Session

**Immediate Actions:**
1. Load this HANDOFF.md document
2. Review completed TDDs (001 and 002) for pattern reference
3. Start with highest priority: `003_FTDD|SAE_Training.md`

**Creation Order:**
```
Priority 1: 003_FTDD|SAE_Training.md      (Most complex, blocks TIDs)
Priority 2: 004_FTDD|Feature_Discovery.md (Medium complexity)
Priority 3: 005_FTDD|Model_Steering.md    (Medium complexity)

Then TIDs in order:
001_FTID|Dataset_Management.md
002_FTID|Model_Management.md
003_FTID|SAE_Training.md
004_FTID|Feature_Discovery.md
005_FTID|Model_Steering.md
```

**Quality Standards:**
- Each TDD: 14 sections, 50-70KB, comprehensive technical detail
- Each TID: Implementation-focused, 30-50KB, specific coding patterns
- All documents reference Mock UI with exact line numbers
- All documents follow ADR technical standards
- Database schemas reference PostgreSQL spec file
- Component patterns match established React/TypeScript conventions

**Completion Criteria:**
- All 5 TDDs created (3 remaining)
- All 5 TIDs created (5 remaining)
- Total: 8 documents to complete

---

## Technical Context for SAE Training TDD (Next Document)

### Key Challenges:
1. **Training Loop Complexity:**
   - Forward pass: model activations → SAE encoder → SAE decoder → reconstruction loss
   - Backward pass: Calculate gradients, apply optimizer, update LR scheduler
   - Sparsity penalty: L1 regularization on encoder activations
   - Ghost gradient penalty: Penalize rarely-activated neurons

2. **Memory Constraints:**
   - Jetson Orin Nano: 8GB RAM, 6GB GPU VRAM
   - Must support models up to 1B parameters with quantization
   - Dynamic batch size reduction on OOM
   - Gradient accumulation for effective larger batches
   - Mixed precision training (torch.cuda.amp)

3. **Checkpoint Management:**
   - Save frequency: Manual + auto-save every N steps
   - Retention policy: Keep first, last, every 1000 steps, best loss
   - Checkpoint contents: Model state, optimizer state, scheduler state, RNG states
   - Format: safetensors (safer than pickle)

4. **Real-time Metrics:**
   - Calculate every 10 steps: loss, L0 sparsity, dead neurons, learning rate
   - Store in training_metrics table
   - Emit WebSocket events for live UI updates
   - Update denormalized fields in trainings table

### Architecture Diagram Pattern (use in TDD):
```
User clicks "Start Training"
    ↓
Frontend: POST /api/trainings (config)
    ↓
FastAPI: Create training record (status='queued')
    ↓
Celery: Enqueue train_sae_task.delay(training_id)
    ↓
Worker: Initialize (load model, dataset, create SAE)
    ↓
Worker: Training loop
    ├─ Extract activations from model
    ├─ Pass through SAE encoder
    ├─ Calculate reconstruction loss + L1 penalty
    ├─ Backward pass (gradients)
    ├─ Optimizer step
    ├─ Every 10 steps: Calculate metrics, emit WebSocket
    ├─ Every N steps: Save checkpoint (if auto-save enabled)
    └─ Repeat until total_steps reached
    ↓
Worker: Save final checkpoint, update status='completed'
    ↓
Frontend: Receives WebSocket 'completed' event, displays results
```

---

## File Locations Summary

### Created Files (7 total):
```
0xcc/
├── prds/
│   ├── 001_FPRD|Dataset_Management.md        ✅ Complete
│   ├── 002_FPRD|Model_Management.md          ✅ Complete
│   ├── 003_FPRD|SAE_Training.md              ✅ Complete
│   ├── 004_FPRD|Feature_Discovery.md         ✅ Complete
│   └── 005_FPRD|Model_Steering.md            ✅ Complete
│
└── tdds/
    ├── 001_FTDD|Dataset_Management.md         ✅ Complete (43KB)
    └── 002_FTDD|Model_Management.md           ✅ Complete (56KB)
```

### Files to Create (8 total):
```
0xcc/
├── tdds/
│   ├── 003_FTDD|SAE_Training.md              🔲 Next (Priority 1)
│   ├── 004_FTDD|Feature_Discovery.md         🔲 Pending (Priority 2)
│   └── 005_FTDD|Model_Steering.md            🔲 Pending (Priority 3)
│
└── tids/
    ├── 001_FTID|Dataset_Management.md         🔲 Pending
    ├── 002_FTID|Model_Management.md           🔲 Pending
    ├── 003_FTID|SAE_Training.md               🔲 Pending
    ├── 004_FTID|Feature_Discovery.md          🔲 Pending
    └── 005_FTID|Model_Steering.md             🔲 Pending
```

---

## Success Metrics

**Completed Work:** 7/15 documents (46.7%)
- ✅ 5/5 PRDs (100%)
- ✅ 2/5 TDDs (40%)
- 🔲 0/5 TIDs (0%)

**Remaining Work:** 8/15 documents (53.3%)
- 3 TDDs (~170KB total)
- 5 TIDs (~185KB total)

**Estimated Time to Complete:**
- TDDs: ~8-10 hours (2-3 hours per document)
- TIDs: ~8-10 hours (1.5-2 hours per document)
- Total: ~16-20 hours of focused work

---

**Ready for Next Session!**
This handoff document provides everything needed to seamlessly continue the TDD/TID creation process with full context and clear priorities.
