# Task List: SAE Training Enhancements - Resource Management & Multi-Dataset Support

**Feature:** Enhanced SAE Training with GPU Memory Cleanup and Multi-Dataset Training
**Parent Task:** 003_FTASKS|SAE_Training.md
**Status:** Ready for Implementation
**Created:** 2025-10-26
**Priority:** High (GPU cleanup), Medium (Multi-dataset)

---

## Enhancement Overview

This enhancement adds three critical improvements to the SAE training system:

1. **GPU Memory Cleanup**: Explicit VRAM cleanup after training completion/failure
2. **Multi-Dataset Training**: Support for training SAEs across multiple datasets simultaneously
3. **Dataset Mixing Strategies**: Weighted sampling and curriculum learning support

### Rationale

**GPU Memory Cleanup:**
- Current implementation relies on Python garbage collection
- Long-running Celery workers accumulate memory across multiple jobs
- Models/optimizers remain in VRAM until GC runs
- Can cause OOM errors on subsequent trainings even with sufficient theoretical memory

**Multi-Dataset Training:**
- Enables comprehensive feature learning across diverse data distributions
- Supports domain-specific + general data mixing for balanced SAEs
- Allows curriculum learning (simple → complex data progression)
- Improves feature quality through exposure to varied activation patterns

### Evidence from Empirical Testing

Recent training comparisons demonstrated:
- l1_alpha=0.01 with 3.2M samples (100k steps × batch_size=32) achieved optimal L0=4.6%
- More diverse data → better feature coverage and reduced polysemantic features
- Targeted data → domain-specific interpretability improvements

---

## Relevant Files

### Backend (Existing - To Modify)
- `backend/src/workers/training_tasks.py` - Add GPU cleanup, multi-dataset data loading
- `backend/src/models/training.py` - Add multi-dataset relationship
- `backend/src/schemas/training.py` - Update schemas for multi-dataset support
- `backend/src/services/training_service.py` - Validate multi-dataset configurations
- `backend/alembic/versions/` - New migration for training_datasets junction table

### Backend (New Files)
- `backend/src/services/multi_dataset_loader.py` - Multi-dataset batch sampling logic
- `backend/src/schemas/dataset_mixing.py` - Dataset mixing strategy schemas

### Frontend (To Modify)
- `frontend/src/types/training.ts` - Add multi-dataset types
- `frontend/src/stores/trainingsStore.ts` - Support multi-dataset config
- `frontend/src/components/panels/TrainingPanel.tsx` - Multi-dataset selector UI

### Frontend (New Files)
- `frontend/src/components/training/DatasetMixingConfig.tsx` - Dataset weights configuration UI

### Tests
- `backend/tests/unit/test_multi_dataset_loader.py` - Multi-dataset sampling tests
- `backend/tests/unit/test_gpu_cleanup.py` - GPU memory cleanup verification
- `backend/tests/integration/test_multi_dataset_training.py` - End-to-end multi-dataset training
- `frontend/tests/DatasetMixingConfig.test.tsx` - Dataset mixing UI tests

---

## Tasks

### Phase 1: GPU Memory Cleanup (Quick Fix - Priority: High)
✅ COMPLETE - Implemented in commits 12900b3, f6abcfa

- [x] 1.0 Add Explicit GPU Memory Cleanup to Training Tasks
  - [x] 1.1 Add cleanup helper function `cleanup_training_resources(models, optimizers, device)`
    - Delete model references: `del models, optimizers`
    - Clear CUDA cache if available: `torch.cuda.empty_cache()`
    - Log memory before/after for monitoring
    - File: `backend/src/workers/training_tasks.py`

  - [x] 1.2 Call cleanup in training completion path
    - Location: After line 602 (after training:completed event emission)
    - Log: "Cleaning up GPU memory: {before}MB → {after}MB"
    - Verify models/optimizers are out of scope
    - File: `backend/src/workers/training_tasks.py` (lines 603-610)

  - [x] 1.3 Call cleanup in exception handler (training failure)
    - Location: After line 623 (after training:failed event emission)
    - Ensure cleanup happens before raising exception
    - Use try-except to prevent cleanup failures from masking original error
    - File: `backend/src/workers/training_tasks.py` (lines 610-636)

  - [x] 1.4 Add cleanup to OOM recovery path
    - Location: After line 411 (after reducing batch size)
    - Already has `torch.cuda.empty_cache()` but add model deletion if needed
    - File: `backend/src/workers/training_tasks.py` (lines 404-421)

  - [x] 1.5 Test GPU cleanup effectiveness
    - Create test: Start training → Complete → Check GPU memory released
    - Create test: Start training → Fail → Check GPU memory released
    - Verify memory released within 1 second of training end
    - File: `backend/tests/unit/test_gpu_cleanup.py`

### Phase 2: Database Schema for Multi-Dataset Support

- [ ] 2.0 Create Training-Datasets Junction Table
  - [ ] 2.1 Create Alembic migration `004_add_training_datasets_junction.py`
    - Table: `training_datasets`
    - Columns:
      - `id` (UUID, primary key)
      - `training_id` (UUID, foreign key → trainings.id, CASCADE on delete)
      - `dataset_id` (UUID, foreign key → datasets.id, RESTRICT on delete)
      - `weight` (Float, default 1.0, range 0.0-1.0) - Sampling weight
      - `order` (Integer, nullable) - For curriculum learning (NULL = simultaneous)
      - `created_at` (Timestamp with timezone)
    - Indexes:
      - `idx_training_datasets_training_id` on (training_id)
      - `idx_training_datasets_dataset_id` on (dataset_id)
      - Unique constraint on (training_id, dataset_id)
    - File: `backend/alembic/versions/004_add_training_datasets_junction.py`

  - [ ] 2.2 Update Training model to add relationship
    - Add relationship: `datasets = relationship("TrainingDataset", back_populates="training")`
    - Add property: `dataset_ids: List[str]` (computed from datasets relationship)
    - Maintain backward compatibility: `dataset_id` property returns first dataset
    - File: `backend/src/models/training.py`

  - [ ] 2.3 Create TrainingDataset model
    - SQLAlchemy model with columns matching migration
    - Relationships: `training`, `dataset`
    - Validation: weight in range [0.0, 1.0]
    - File: `backend/src/models/training_dataset.py` (new file)

### Phase 3: Dataset Mixing Schemas and Validation

- [ ] 3.0 Create Dataset Mixing Configuration Schemas
  - [ ] 3.1 Create `DatasetMixingStrategy` enum
    - Values: `UNIFORM`, `WEIGHTED`, `CURRICULUM`, `SEQUENTIAL`
    - UNIFORM: Equal sampling from all datasets
    - WEIGHTED: Sample proportionally by specified weights
    - CURRICULUM: Progress through datasets by order (phase-based)
    - SEQUENTIAL: One dataset at a time, switch at specified steps
    - File: `backend/src/schemas/dataset_mixing.py` (new file)

  - [ ] 3.2 Create `DatasetConfig` schema
    - Fields:
      - `dataset_id: str` (UUID of dataset)
      - `weight: float = 1.0` (sampling weight, 0.0-1.0)
      - `order: Optional[int] = None` (for curriculum/sequential)
      - `start_step: Optional[int] = None` (when to start using this dataset)
      - `end_step: Optional[int] = None` (when to stop using this dataset)
    - Validation:
      - weight in [0.0, 1.0]
      - If order specified, must be >= 0
      - start_step < end_step if both specified
    - File: `backend/src/schemas/dataset_mixing.py`

  - [ ] 3.3 Create `MultiDatasetConfig` schema
    - Fields:
      - `datasets: List[DatasetConfig]` (min 1, max 10 datasets)
      - `strategy: DatasetMixingStrategy = UNIFORM`
      - `normalize_weights: bool = True` (auto-normalize weights to sum to 1.0)
      - `shuffle_across_datasets: bool = True` (shuffle batch samples)
    - Validation:
      - At least one dataset
      - Weights sum to > 0 if WEIGHTED strategy
      - Orders are unique and sequential if CURRICULUM strategy
      - No overlapping step ranges if SEQUENTIAL strategy
    - File: `backend/src/schemas/dataset_mixing.py`

  - [ ] 3.4 Update `TrainingCreateRequest` schema
    - Add: `dataset_config: Optional[MultiDatasetConfig] = None`
    - Maintain: `dataset_id: Optional[str] = None` (backward compatibility)
    - Validation: Must specify EITHER dataset_id OR dataset_config (not both, not neither)
    - File: `backend/src/schemas/training.py`

### Phase 4: Multi-Dataset Data Loader Implementation

- [ ] 4.0 Implement Multi-Dataset Batch Sampler
  - [ ] 4.1 Create `MultiDatasetLoader` class
    - Constructor: `__init__(dataset_configs, strategy, batch_size, device)`
    - Load all datasets into memory or create data loaders
    - Normalize weights if configured
    - Initialize per-dataset sample indices
    - File: `backend/src/services/multi_dataset_loader.py` (new file)

  - [ ] 4.2 Implement `get_batch(step: int)` method for UNIFORM strategy
    - Divide batch_size equally across datasets
    - Handle remainders (distribute to datasets with highest weights)
    - Sample randomly from each dataset
    - Shuffle combined batch if configured
    - Return: `Dict[str, torch.Tensor]` with keys: input_ids, attention_mask, etc.
    - File: `backend/src/services/multi_dataset_loader.py`

  - [ ] 4.3 Implement `get_batch(step: int)` method for WEIGHTED strategy
    - Calculate samples_per_dataset: `batch_size * normalized_weights`
    - Round to integers (ensure sum = batch_size)
    - Sample from each dataset proportionally
    - Shuffle combined batch if configured
    - File: `backend/src/services/multi_dataset_loader.py`

  - [ ] 4.4 Implement `get_batch(step: int)` method for CURRICULUM strategy
    - Determine current curriculum phase based on step and total_steps
    - Calculate phase boundaries: `total_steps / num_datasets`
    - Use only datasets with `order <= current_phase`
    - Apply WEIGHTED strategy within active datasets
    - File: `backend/src/services/multi_dataset_loader.py`

  - [ ] 4.5 Implement `get_batch(step: int)` method for SEQUENTIAL strategy
    - Find active dataset based on step ranges (start_step, end_step)
    - Return batch from only the active dataset
    - Raise error if step has no active dataset (configuration bug)
    - File: `backend/src/services/multi_dataset_loader.py`

  - [ ] 4.6 Add dataset exhaustion handling
    - Track samples consumed per dataset
    - Reshuffle dataset when exhausted (epoch boundary)
    - Log: "Dataset {name} completed epoch {N}, reshuffling"
    - Ensure infinite iteration for training loop
    - File: `backend/src/services/multi_dataset_loader.py`

  - [ ] 4.7 Add logging and diagnostics
    - Log dataset mixing configuration at initialization
    - Log batch composition every 100 steps (DEBUG level)
    - Track samples seen per dataset (for statistics)
    - Method: `get_statistics() -> Dict[str, int]` (samples per dataset)
    - File: `backend/src/services/multi_dataset_loader.py`

### Phase 5: Integrate Multi-Dataset Support into Training Task

- [ ] 5.0 Update Training Task to Support Multi-Dataset Loading
  - [ ] 5.1 Modify `train_sae` task to detect multi-dataset configuration
    - Check if `training.datasets` relationship has multiple entries
    - If single dataset: Use existing data loading logic (backward compatibility)
    - If multiple datasets: Instantiate `MultiDatasetLoader`
    - File: `backend/src/workers/training_tasks.py` (around line 200)

  - [ ] 5.2 Replace single-dataset data loading with multi-dataset loader
    - Current: Load dataset from `training.dataset_id`
    - New: Build `DatasetConfig` list from `training.datasets` relationship
    - Instantiate: `loader = MultiDatasetLoader(configs, strategy, batch_size, device)`
    - Replace batch retrieval: `batch = loader.get_batch(step)`
    - File: `backend/src/workers/training_tasks.py` (lines 220-250)

  - [ ] 5.3 Log multi-dataset configuration at training start
    - Log dataset names, weights, strategy
    - Example: "Training with 3 datasets: WikiText-103 (40%), Code (40%), Books (20%)"
    - Log expected samples per dataset over full training
    - File: `backend/src/workers/training_tasks.py` (around line 260)

  - [ ] 5.4 Store multi-dataset statistics in training metadata
    - After training completion, call `loader.get_statistics()`
    - Store in `training.metadata['dataset_statistics']`
    - Include: samples_per_dataset, epochs_per_dataset
    - File: `backend/src/workers/training_tasks.py` (around line 600)

### Phase 6: Training Service Updates for Multi-Dataset

- [ ] 6.0 Update Training Service to Handle Multi-Dataset Creation
  - [ ] 6.1 Modify `create_training()` to handle dataset_config
    - If `dataset_config` provided: Create multiple `TrainingDataset` entries
    - If `dataset_id` provided: Create single `TrainingDataset` entry (legacy)
    - Validate all dataset IDs exist and are in READY status
    - Normalize weights if configured
    - File: `backend/src/services/training_service.py`

  - [ ] 6.2 Add validation: datasets must be compatible
    - Check: All datasets have same `tokenizer_id` (if applicable)
    - Check: All datasets have same `model_type` (if applicable)
    - Warn if datasets have very different sizes (may cause sampling imbalance)
    - File: `backend/src/services/training_service.py`

  - [ ] 6.3 Update `get_training()` to include dataset mixing info
    - Include `dataset_configs` in response
    - Include `mixing_strategy` in response
    - Maintain `dataset_id` for backward compatibility (first dataset)
    - File: `backend/src/services/training_service.py`

  - [ ] 6.4 Add helper: `estimate_multi_dataset_memory()`
    - Calculate memory based on largest dataset in mix
    - Account for additional overhead of multi-dataset loader (~10%)
    - Return memory estimate and per-dataset breakdown
    - File: `backend/src/services/training_service.py`

### Phase 7: Frontend - Multi-Dataset Selection UI

- [ ] 7.0 Create Dataset Mixing Configuration Component
  - [ ] 7.1 Create `DatasetMixingConfig.tsx` component
    - Props: `selectedDatasets`, `onDatasetsChange`, `availableDatasets`
    - State: `strategy`, `weights`, `showAdvanced`
    - Layout: Dataset list + strategy selector + optional weight sliders
    - File: `frontend/src/components/training/DatasetMixingConfig.tsx` (new file)

  - [ ] 7.2 Implement dataset multi-select with checkboxes
    - Show available datasets with checkboxes
    - Display dataset name, size, status
    - Disable datasets that are not READY
    - Allow selection of 1-10 datasets
    - Highlight first dataset as "primary" (for legacy dataset_id)
    - File: `frontend/src/components/training/DatasetMixingConfig.tsx`

  - [ ] 7.3 Implement strategy selector
    - Radio buttons: Uniform, Weighted, Curriculum, Sequential
    - Uniform: Default, no additional config
    - Weighted: Show weight sliders for each dataset
    - Curriculum: Show order selectors (drag-to-reorder)
    - Sequential: Show step range inputs per dataset
    - File: `frontend/src/components/training/DatasetMixingConfig.tsx`

  - [ ] 7.4 Implement weight sliders (for WEIGHTED strategy)
    - Slider per dataset: 0% - 100%
    - Real-time normalization display: "Dataset A: 40% (12,800 samples/step)"
    - Auto-normalize toggle: "Normalize weights to sum to 100%"
    - Visual indicator: Bar chart showing relative weights
    - File: `frontend/src/components/training/DatasetMixingConfig.tsx`

  - [ ] 7.5 Implement curriculum phase configuration (for CURRICULUM strategy)
    - Drag-and-drop reordering of datasets
    - Visual timeline showing phase progression
    - Display: "Phase 1 (steps 0-33k): Dataset A → Phase 2 (steps 33k-66k): A+B → Phase 3 (steps 66k-100k): A+B+C"
    - File: `frontend/src/components/training/DatasetMixingConfig.tsx`

  - [ ] 7.6 Implement step range configuration (for SEQUENTIAL strategy)
    - Step range inputs per dataset: [start_step, end_step]
    - Validation: No overlaps, no gaps
    - Visual timeline: Horizontal bars showing dataset usage over training
    - File: `frontend/src/components/training/DatasetMixingConfig.tsx`

### Phase 8: Frontend - Integrate Multi-Dataset UI into Training Panel

- [ ] 8.0 Update Training Panel for Multi-Dataset Support
  - [ ] 8.1 Replace single dataset dropdown with multi-dataset component
    - Location: `TrainingPanel.tsx` (around line 227-244)
    - Replace `<select>` with `<DatasetMixingConfig>`
    - Pass: available datasets, selected config, onChange handler
    - Show simplified view by default, "Configure Multi-Dataset" button for advanced
    - File: `frontend/src/components/panels/TrainingPanel.tsx`

  - [ ] 8.2 Update training config state to support multi-dataset
    - Add: `dataset_config: MultiDatasetConfig | null`
    - Maintain: `dataset_id: string` for backward compatibility
    - Logic: If multi-dataset selected, set dataset_config; else set dataset_id
    - File: `frontend/src/stores/trainingsStore.ts`

  - [ ] 8.3 Update training creation request builder
    - If `dataset_config` present: Include in request, omit `dataset_id`
    - If `dataset_id` present: Include in request, omit `dataset_config`
    - Validation: Must have one or the other
    - File: `frontend/src/stores/trainingsStore.ts`

  - [ ] 8.4 Add dataset mixing info to TrainingCard
    - Display: "Trained on 3 datasets" with tooltip showing breakdown
    - Tooltip: List datasets with weights: "WikiText (40%), Code (40%), Books (20%)"
    - Show strategy: "Mixing: Weighted" or "Mixing: Curriculum (3 phases)"
    - File: `frontend/src/components/training/TrainingCard.tsx`

  - [ ] 8.5 Update memory estimation for multi-dataset
    - Call: `estimateMultiDatasetMemory(configs, batch_size)`
    - Display: "Memory: 4.2 GB (max across datasets + 10% overhead)"
    - Warn if any dataset would cause OOM individually
    - File: `frontend/src/utils/memoryEstimation.ts`

### Phase 9: Testing and Validation

- [ ] 9.0 Unit Tests for Multi-Dataset Loader
  - [ ] 9.1 Test UNIFORM strategy sampling
    - Create 3 mock datasets with 1000 samples each
    - Request batch_size=30 → expect ~10 samples per dataset
    - Verify: All datasets represented in batch
    - Verify: Samples are shuffled (if configured)
    - File: `backend/tests/unit/test_multi_dataset_loader.py`

  - [ ] 9.2 Test WEIGHTED strategy sampling
    - Create 3 datasets with weights [0.5, 0.3, 0.2]
    - Request batch_size=100 → expect [50, 30, 20] samples
    - Verify: Actual sample counts match expected (within rounding)
    - Test edge case: weights [0.7, 0.2, 0.1] → [70, 20, 10]
    - File: `backend/tests/unit/test_multi_dataset_loader.py`

  - [ ] 9.3 Test CURRICULUM strategy progression
    - Create 3 datasets with orders [0, 1, 2]
    - Steps 0-33k: Only dataset 0
    - Steps 33k-66k: Datasets 0+1
    - Steps 66k-100k: Datasets 0+1+2
    - Verify: Correct datasets used at each phase
    - File: `backend/tests/unit/test_multi_dataset_loader.py`

  - [ ] 9.4 Test SEQUENTIAL strategy step ranges
    - Create 3 datasets with ranges: [0-30k], [30k-60k], [60k-100k]
    - Verify: Only correct dataset active at each step
    - Test edge case: Step exactly at boundary (e.g., 30,000)
    - File: `backend/tests/unit/test_multi_dataset_loader.py`

  - [ ] 9.5 Test dataset exhaustion and reshuffling
    - Create small dataset (100 samples)
    - Request batches until dataset exhausted
    - Verify: Dataset reshuffled, training continues
    - Verify: Samples not repeated within epoch
    - File: `backend/tests/unit/test_multi_dataset_loader.py`

- [ ] 10.0 Integration Tests for Multi-Dataset Training
  - [ ] 10.1 Test end-to-end multi-dataset training (UNIFORM)
    - Create 2 test datasets (100 samples each)
    - Start training with UNIFORM strategy
    - Run 100 steps (batch_size=10)
    - Verify: Training completes successfully
    - Verify: Both datasets used (check statistics)
    - File: `backend/tests/integration/test_multi_dataset_training.py`

  - [ ] 10.2 Test end-to-end multi-dataset training (WEIGHTED)
    - Create 2 test datasets with weights [0.7, 0.3]
    - Run 100 steps
    - Verify: ~70% samples from dataset 1, ~30% from dataset 2
    - Verify: Statistics stored in training metadata
    - File: `backend/tests/integration/test_multi_dataset_training.py`

  - [ ] 10.3 Test GPU memory cleanup after multi-dataset training
    - Record GPU memory before training
    - Start multi-dataset training (2 datasets)
    - Wait for completion
    - Check GPU memory after completion
    - Verify: Memory returned to pre-training level (±100MB)
    - File: `backend/tests/integration/test_multi_dataset_training.py`

  - [ ] 10.4 Test backward compatibility (single dataset)
    - Create training with `dataset_id` (not `dataset_config`)
    - Verify: Training works exactly as before
    - Verify: No errors, no warnings
    - File: `backend/tests/integration/test_multi_dataset_training.py`

- [ ] 11.0 Frontend Component Tests
  - [ ] 11.1 Test DatasetMixingConfig component rendering
    - Render with 3 available datasets
    - Verify: All datasets shown with checkboxes
    - Verify: Strategy selector visible
    - File: `frontend/tests/DatasetMixingConfig.test.tsx`

  - [ ] 11.2 Test dataset selection interaction
    - Select 2 datasets
    - Verify: `onDatasetsChange` called with correct IDs
    - Deselect 1 dataset
    - Verify: Updated selection propagated
    - File: `frontend/tests/DatasetMixingConfig.test.tsx`

  - [ ] 11.3 Test weight slider interaction (WEIGHTED strategy)
    - Select WEIGHTED strategy
    - Adjust dataset weight sliders
    - Verify: Weights normalized in real-time
    - Verify: onChange called with correct weight values
    - File: `frontend/tests/DatasetMixingConfig.test.tsx`

  - [ ] 11.4 Test TrainingPanel with multi-dataset config
    - Open TrainingPanel
    - Configure multi-dataset mixing (2 datasets, WEIGHTED)
    - Click "Start Training"
    - Verify: API called with correct `dataset_config` structure
    - Verify: Request does NOT include `dataset_id` field
    - File: `frontend/tests/TrainingPanel.test.tsx`

### Phase 10: Documentation and Finalization

- [ ] 12.0 Update Documentation
  - [ ] 12.1 Add multi-dataset training guide
    - Document: When to use multi-dataset training
    - Document: Choosing mixing strategies (with examples)
    - Document: Weight selection best practices
    - Document: Curriculum learning patterns
    - File: `docs/sae_training_multi_dataset.md` (new file)

  - [ ] 12.2 Update hyperparameter documentation tooltips
    - Add tooltip for "Mixing Strategy" dropdown
    - Add tooltip for "Dataset Weights" sliders
    - Explain: How dataset diversity affects feature quality
    - Explain: Trade-offs of each strategy
    - File: `frontend/src/config/hyperparameterDocs.ts`

  - [ ] 12.3 Update API documentation
    - Document: `dataset_config` request field
    - Document: Multi-dataset response structure
    - Add examples: Single dataset (legacy), multi-dataset (UNIFORM), multi-dataset (WEIGHTED)
    - File: Backend API docs (FastAPI auto-generated)

  - [ ] 12.4 Add migration guide for existing trainings
    - Document: Backward compatibility guarantees
    - Document: How to convert existing single-dataset configs to multi-dataset
    - Document: Performance considerations (memory, throughput)
    - File: `docs/migration_multi_dataset.md` (new file)

---

## Success Criteria

### Phase 1 (GPU Cleanup):
- [ ] GPU memory released within 1 second of training completion
- [ ] No memory leaks across 10 consecutive training jobs
- [ ] Memory usage returns to baseline (±100MB) after training

### Phase 2-6 (Multi-Dataset Backend):
- [ ] Support 1-10 datasets per training
- [ ] All 4 mixing strategies (UNIFORM, WEIGHTED, CURRICULUM, SEQUENTIAL) functional
- [ ] Backward compatibility: Single-dataset trainings work unchanged
- [ ] Dataset statistics stored in training metadata

### Phase 7-8 (Multi-Dataset Frontend):
- [ ] Intuitive multi-dataset selection UI
- [ ] Real-time weight normalization and validation
- [ ] Visual timeline for curriculum/sequential strategies
- [ ] Memory estimation accounts for multi-dataset overhead

### Phase 9 (Testing):
- [ ] 95%+ test coverage for multi-dataset loader
- [ ] End-to-end integration tests pass
- [ ] No performance degradation for single-dataset trainings
- [ ] GPU cleanup verified in integration tests

### Phase 10 (Documentation):
- [ ] Comprehensive multi-dataset training guide
- [ ] Updated API documentation with examples
- [ ] Migration guide for existing users

---

## Estimated Effort

- Phase 1 (GPU Cleanup): **2-3 hours** ⚡ Quick Win
- Phase 2-3 (Schema + Validation): **4-6 hours**
- Phase 4 (Multi-Dataset Loader): **8-12 hours**
- Phase 5-6 (Backend Integration): **6-8 hours**
- Phase 7-8 (Frontend UI): **10-14 hours**
- Phase 9 (Testing): **8-10 hours**
- Phase 10 (Documentation): **4-6 hours**

**Total Estimated Time:** 42-59 hours

**Recommended Approach:**
1. Start with Phase 1 (GPU Cleanup) - immediate benefit, minimal risk
2. Implement Phase 2-6 (Backend) - core multi-dataset functionality
3. Add Phase 7-8 (Frontend) - user-facing multi-dataset UI
4. Complete Phase 9-10 (Testing + Docs) - polish and productionize

---

## Notes

- **GPU Cleanup (Phase 1)** can be implemented and deployed immediately as a bug fix
- **Multi-Dataset Support (Phases 2-10)** is a larger feature that builds on existing architecture
- The junction table approach allows future expansion (e.g., per-dataset learning rates, per-dataset l1_alpha)
- Mixing strategies are inspired by curriculum learning research (Bengio et al., 2009) and multi-task learning practices
- All changes maintain backward compatibility with existing single-dataset trainings
