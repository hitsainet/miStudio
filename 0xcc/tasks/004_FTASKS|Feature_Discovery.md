# Task List: Feature Discovery

**Feature:** Interpretable Feature Extraction and Analysis from Trained SAEs
**PRD Reference:** 004_FPRD|Feature_Discovery.md
**TDD Reference:** 004_FTDD|Feature_Discovery.md
**TID Reference:** 004_FTID|Feature_Discovery.md
**Mock UI Reference:** Lines 2159-2584 (FeaturesPanel), Lines 2587-2725 (FeatureDetailModal), Lines 2728-2800+ (MaxActivatingExamples)
**Status:** ⚠️ 60% COMPLETE - Core extraction working, **CRITICAL ANALYSIS FEATURES MISSING**
**Created:** 2025-10-06
**Last Audited:** 2025-10-28
**Last Updated:** 2025-11-01
**Completion:** 165/282 tasks (58.5%) - See audit: 0xcc/docs/Feature_Discovery_Audit_2025-10-28.md

**🚨 CRITICAL GAPS:**
- ❌ **AnalysisService** (Phases 7-8): Logit lens, correlations, ablation NOT IMPLEMENTED
- ❌ **MaxActivatingExamples** (Phase 19): Token highlighting component NOT IMPLEMENTED
- ❌ **Analysis Tab Components** (Phase 20): LogitLensView, FeatureCorrelations, AblationAnalysis NOT IMPLEMENTED
- **Impact:** Users can extract features but **cannot interpret them** (main value missing)
- **Effort to Complete:** 55 tasks, 12-17 hours

---

## Recent Enhancements (Completed October-November 2025)

### ✅ Extraction Card UX Improvements (NOT in original task list)

**Status:** ✅ **COMPLETE** (2025-10-31 to 2025-11-01)

**Enhancement 1: Training Job Context Display** (commit e9b52ba)
- Shows SAE architecture (standard/gated/jumprelu)
- Shows hyperparameters (hidden_dim, latent_dim, expansion ratio, layers, L1 alpha, learning rate, batch size, total steps)
- Shows target L0 vs actual L0 achieved
- Integrates with `trainingsStore` to fetch training details

**Enhancement 2: Compact 3-Column Layout** (commit 4c851a1)
- Combined "Training Job Information" and "Extraction Configuration" into single "Job Details" section
- Reduced footprint with text-xs sizing, gap-x-4 gap-y-1 spacing
- Better information density for extraction status display

**Enhancement 3: Actual L0 Result Display** (commit 83998b1)
- Added display of `training.current_l0_sparsity` as "Actual L0: X.X%"
- Compares target sparsity vs achieved sparsity
- 3-column grid layout for optimal space usage
- Font increased from text-xs to text-sm for readability

**Files Modified:**
- `frontend/src/components/features/ExtractionJobCard.tsx` (enhanced with training context)
- `frontend/src/stores/trainingsStore.ts` (used for training lookup)

**User Impact:** Users can now understand:
- Which SAE architecture was used
- What hyperparameters were used
- What sparsity level was achieved vs targeted
- Which layers were trained

---

### ✅ Feature Search Enhancements (NOT in original task list)

**Status:** ✅ **COMPLETE** (2025-10-28 to 2025-10-31)

**Enhancement 1: Activation Token Search** (commit 4d875ea)
- Search by specific tokens that activate features
- JSONB query on `feature_activations` table
- Find features that fire on particular words/phrases
- Example: Search "not" finds negation features

**Enhancement 2: ILIKE Substring Search** (commit 86265c9)
- Replaced full-text search with simpler ILIKE `%pattern%`
- Better for partial matching (e.g., "nega" matches "negation")
- Handles special characters correctly (no query parser errors)
- More intuitive for users (no boolean operator knowledge needed)

**Enhancement 3: Plain Text Query Handling** (commit 31694ce)
- Use `plainto_tsquery` instead of `to_tsquery` for full-text fallback
- Handles user input without requiring search operators
- More forgiving search experience
- Prevents query syntax errors

**Bug Fixes:**
- Fixed token search subquery to properly join with Feature table (commit fb98f67)
- Fixed set-returning function handling in WHERE clause (commits 0abca65, 33fc2d3)
- Fixed JSONB cast type from Text to String (commit 1cfdc19)

**Files Modified:**
- `backend/src/services/feature_service.py` (search logic enhanced)
- `backend/src/api/v1/endpoints/features.py` (query parameter handling)

**User Impact:** Users can now:
- Search for features by activation tokens (e.g., "the", "not", "code")
- Use simpler search queries (no special operators needed)
- Get more relevant search results with partial matching

---

### ✅ GPT-Based Auto-Labeling System (NOT in original task list)

**Status:** ✅ **COMPLETE** (2025-11-06)

**Enhancement: Semantic Feature Labeling with LLMs**
- Replaces generic feature names (feature_00042) with semantic labels (determiners, negation, plural_nouns)
- Three labeling methods: pattern matching (fast), local LLM (slow, zero cost), OpenAI API (fast, costs money)
- Token aggregation from activation records: count, total_activation, max_activation per token
- Structured prompts with TOKEN | COUNT | AVG_ACT | MAX_ACT table format
- Label standardization to lowercase_with_underscores format

**Backend Services Created:**
1. **LocalLabelingService** (`backend/src/services/local_labeling_service.py`, 356 lines)
   - Uses Phi-3-mini-4k-instruct (Microsoft) with 4-bit quantization (~2GB VRAM)
   - Supports alternative models: Llama 3.2 3B, Qwen 2.5 3B
   - Batch processing with automatic model load/unload cycle
   - Speed: ~5.5 hours for 16K features
   - Cost: $0 (fully local)

2. **OpenAILabelingService** (`backend/src/services/openai_labeling_service.py`, 284 lines)
   - Uses GPT-4o-mini API with async concurrent calls and rate limiting
   - Supports GPT-4 Turbo and GPT-3.5 as alternatives
   - Speed: ~55 minutes for 16K features
   - Cost: ~$0.0001 per feature (~$1.64 for 16K features)

**Integration Changes:**
- Updated `ExtractionConfigRequest` schema with labeling config fields:
  * labeling_method: 'pattern' | 'local' | 'openai'
  * local_labeling_model: 'phi3' | 'llama' | 'qwen'
  * openai_api_key: optional API key
  * openai_model: 'gpt4-mini' | 'gpt4' | 'gpt35'
- Integrated into `ExtractionService` with batch labeling phase after feature extraction
- Memory-efficient: loads labeling model only after base model unloaded

**Frontend UI:**
- Added "Feature Labeling" configuration section to `FeaturesPanel`
- Labeling method selector with 3 options (pattern/local/openai)
- Conditional UI: local model selector or OpenAI API key/model inputs
- Helpful descriptions showing cost and speed estimates for each method

**Configuration:**
- Added `openai>=1.0.0` to requirements.txt
- Added `openai_api_key` optional config field
- Can be set globally via environment or per-extraction

**Files Created:**
- `backend/src/services/local_labeling_service.py` - Local LLM labeling service
- `backend/src/services/openai_labeling_service.py` - OpenAI API labeling service

**Files Modified:**
- `backend/src/schemas/extraction.py` - Added labeling config fields
- `backend/src/services/extraction_service.py` - Integrated batch labeling phase
- `backend/src/core/config.py` - Added openai_api_key config
- `backend/requirements.txt` - Added openai dependency
- `frontend/src/components/features/FeaturesPanel.tsx` - Added labeling UI

**User Impact:** Users can now:
- Get semantic labels instead of generic feature_XXXXX names
- Choose between pattern matching (fast), local LLM (slow, free), or OpenAI API (fast, costs money)
- Understand feature concepts at a glance (e.g., "negation" vs "feature_00042")
- Trade off speed vs quality vs cost based on their needs

**Commit:** d8efc61 (2025-11-06)

---

## Relevant Files

### Backend
- `backend/src/models/extraction_job.py` - SQLAlchemy model for extraction_jobs table
- `backend/src/models/feature.py` - SQLAlchemy model for features table
- `backend/src/models/feature_activation.py` - SQLAlchemy model for feature_activations table (partitioned)
- `backend/src/models/feature_analysis_cache.py` - SQLAlchemy model for feature_analysis_cache table
- `backend/src/schemas/extraction.py` - Pydantic schemas for extraction configuration and validation
- `backend/src/schemas/feature.py` - Pydantic schemas for feature search, filter, and responses
- `backend/src/services/extraction_service.py` - Feature extraction orchestration, statistics, auto-labeling
- `backend/src/services/local_labeling_service.py` - Local LLM-based feature labeling (Phi-3, Llama, Qwen)
- `backend/src/services/openai_labeling_service.py` - OpenAI API-based feature labeling (GPT-4o-mini, GPT-4, GPT-3.5)
- `backend/src/services/analysis_service.py` - Logit lens, correlations, ablation calculations with caching
- `backend/src/services/feature_service.py` - CRUD operations, search/filter/sort logic
- `backend/src/workers/extraction_tasks.py` - Celery task: extract_features_task (main extraction loop)
- `backend/src/api/routes/features.py` - FastAPI routes for feature endpoints
- `backend/src/utils/auto_labeling.py` - Pattern matching heuristics for feature labels
- `backend/alembic/versions/004_create_feature_tables.py` - Database migration for feature tables

### Frontend
- `frontend/src/types/feature.types.ts` - TypeScript interfaces for Feature, ExtractionJob, FeatureActivation
- `frontend/src/stores/featuresStore.ts` - Zustand store for feature state management
- `frontend/src/components/panels/FeaturesPanel.tsx` - Training selector, extraction UI, feature browser (Mock UI lines 2159-2584)
- `frontend/src/components/features/FeatureDetailModal.tsx` - Feature details, tabs, editable label (Mock UI lines 2587-2725)
- `frontend/src/components/features/MaxActivatingExamples.tsx` - Token sequences with highlighting (Mock UI lines 2728-2800+)
- `frontend/src/components/features/LogitLensView.tsx` - Logit lens analysis tab content
- `frontend/src/components/features/FeatureCorrelations.tsx` - Correlations analysis tab content
- `frontend/src/components/features/AblationAnalysis.tsx` - Ablation analysis tab content

### Tests
- `backend/tests/test_extraction_service.py` - Unit tests for ExtractionService
- `backend/tests/test_auto_labeling.py` - Unit tests for auto-labeling heuristics
- `backend/tests/test_analysis_service.py` - Unit tests for analysis calculations
- `backend/tests/test_extraction_tasks.py` - Integration tests for extract_features_task
- `backend/tests/test_feature_search.py` - Integration tests for search/filter/sort
- `frontend/tests/FeaturesPanel.test.tsx` - Unit tests for FeaturesPanel component
- `frontend/tests/FeatureDetailModal.test.tsx` - Unit tests for FeatureDetailModal component
- `backend/tests/e2e/test_extraction_flow.py` - E2E test: start extraction → complete → browse features

---

## Task Completion Summary

**Last Updated:** 2025-10-28 (Task file synchronized with actual implementation)
**Audit Reference:** `0xcc/docs/Feature_Discovery_Audit_2025-10-28.md`

### Completion by Category

**✅ Fully Complete (100%):**
- Phase 3: Pydantic Schemas (10/10 tasks)
- Phase 5: Interpretability Score (7/7 tasks)
- Phase 9: FeatureService (14/14 tasks)
- Phase 11: WebSocket Integration (5/5 tasks)
- Phase 13-17: FeaturesPanel UI (47/47 tasks)

**✅ Substantially Complete (85-95%):**
- Phase 1: Database Models (10/11 tasks, 90%)
- Phase 2: Auto-Labeling (10/12 tasks, 83%)
- Phase 4: ExtractionService (17/20 tasks, 85%) - 1,342 lines!
- Phase 6: Celery Tasks (9/10 tasks, 90%)
- Phase 10: API Endpoints (13/15 tasks, 87%)
- Phase 12: Frontend Store (17/18 tasks, 95%)
- Phase 18: FeatureDetailModal (14/15 tasks, 90%)

**❌ Not Implemented (0%):**
- Phase 7: AnalysisService Logit Lens (0/12 tasks) - **CRITICAL GAP**
- Phase 8: AnalysisService Correlations/Ablation (0/15 tasks) - **CRITICAL GAP**
- Phase 19: MaxActivatingExamples Component (0/13 tasks) - **CRITICAL GAP**
- Phase 20: Analysis Tab Components (0/15 tasks) - **BLOCKED** (requires Phase 7-8)

**⚠️ Partial Implementation:**
- Phase 21: WebSocket Frontend (6/10 tasks, 60%)
- Phase 22: Performance Optimization (2/10 tasks, 20%)
- Phase 23: Unit Tests (7/40 tasks, 30%)
- Phase 24: Integration/E2E Tests (4/10 tasks, 40%)

### Critical Missing Components

**Priority 1 - AnalysisService (27 tasks, 6-8 hours):**
- File: `backend/src/services/analysis_service.py` (does not exist)
- Blocks: API endpoints 10.10-10.12, Phase 20 UI components
- Methods: `calculate_logit_lens()`, `calculate_correlations()`, `calculate_ablation()`

**Priority 2 - MaxActivatingExamples (13 tasks, 2-3 hours):**
- File: `frontend/src/components/features/MaxActivatingExamples.tsx` (does not exist)
- Pattern: Reuse TokenHighlight.tsx for token display
- API: Already implemented (GET /api/features/:id/examples)

**Priority 3 - Analysis Tab Components (15 tasks, 4-6 hours):**
- Files: LogitLensView.tsx, FeatureCorrelations.tsx, AblationAnalysis.tsx (none exist)
- Depends on: AnalysisService implementation

**Total Missing Work:** 55 tasks, estimated 12-17 hours

---

## Tasks

- [x] 1.0 Phase 1: Backend Infrastructure - Database Schema and Models ✅ 90% (Migration: 76918d8aa763)
  - [x] 1.1 Create migration `004_create_feature_tables.py` with extraction_jobs, features, feature_activations (partitioned), feature_analysis_cache tables (TDD lines 264-455)
  - [x] 1.2 Implement table partitioning for `feature_activations` by feature_id ranges (1000 features per partition) (TDD lines 457-484)
  - [x] 1.3 Add indexes: idx_extraction_jobs_training_id, idx_features_training_id, idx_features_activation_freq, idx_features_interpretability, idx_features_favorite (TDD lines 294-346)
  - [x] 1.4 Add GIN index for full-text search: `to_tsvector('english', name || description)` (TDD lines 349-351)
  - [x] 1.5 Add unique constraint on extraction_jobs: only one active extraction per training (TDD lines 296-298)
  - [x] 1.6 Create SQLAlchemy model `ExtractionJob` in `backend/src/models/extraction_job.py` with all fields from TDD lines 264-299 (84 lines)
  - [x] 1.7 Create SQLAlchemy model `Feature` in `backend/src/models/feature.py` with all fields from TDD lines 309-352 (76 lines)
  - [x] 1.8 Create SQLAlchemy model `FeatureActivation` in `backend/src/models/feature_activation.py` with JSONB tokens/activations arrays (TDD lines 359-389) (54 lines)
  - [x] 1.9 Create SQLAlchemy model `FeatureAnalysisCache` in `backend/src/models/feature_analysis_cache.py` with JSONB results field (TDD lines 401-423) (61 lines)
  - [x] 1.10 Run migration: `alembic upgrade head` and verify tables created with partitions
  - [ ] 1.11 Write unit tests for model relationships and JSONB field handling ❌ MISSING

- [x] 2.0 Phase 2: Auto-Labeling Heuristics Implementation ✅ 83% (144 lines)
  - [x] 2.1 Create `backend/src/utils/auto_labeling.py` with `auto_label_feature(top_examples, neuron_index)` function (TDD lines 1700-1741, TID lines 143-183)
  - [x] 2.2 Implement pattern matching: all punctuation → "Punctuation" (TDD line 1718)
  - [x] 2.3 Implement pattern matching: question words (what/how/why/when/where/who) → "Question Pattern" (TDD lines 1721-1722)
  - [x] 2.4 Implement pattern matching: code tokens (def/function/class/import/return) → "Code Syntax" (TDD lines 1724-1725)
  - [x] 2.5 Implement pattern matching: positive sentiment words → "Sentiment Positive" (TDD lines 1727-1728)
  - [x] 2.6 Implement pattern matching: negative sentiment words → "Sentiment Negative" (TDD lines 1730-1731)
  - [x] 2.7 Implement pattern matching: negation words (not/no/never/n't) → "Negation Logic" (TDD lines 1733-1734)
  - [x] 2.8 Implement pattern matching: first-person pronouns → "Pronouns First Person" (TDD lines 1736-1737)
  - [x] 2.9 Implement fallback: "Feature {neuron_index}" if no pattern matches (TDD line 1740)
  - [x] 2.10 Extract high-activation tokens (intensity > 0.7) from top 5 examples for pattern matching (TDD lines 1710-1715)
  - [ ] 2.11 Write unit tests for all pattern matching cases (TDD lines 1395-1410) ❌ MISSING
  - [ ] 2.12 Write unit test for fallback behavior with random tokens ❌ MISSING

- [x] 3.0 Phase 3: Backend Pydantic Schemas and Validation ✅ 100% COMPLETE
  - [x] 3.1 Create `backend/src/schemas/extraction.py` with `ExtractionConfigRequest` schema (TDD lines 1267-1269) (91 lines)
  - [x] 3.2 Add field validation: evaluation_samples (int, ge=1000, le=100000) (TDD line 1268)
  - [x] 3.3 Add field validation: top_k_examples (int, ge=10, le=1000) (TDD line 1269)
  - [x] 3.4 Create `backend/src/schemas/feature.py` with `FeatureSearchRequest` schema (TDD lines 1271-1277) (215 lines)
  - [x] 3.5 Add field validation: search (str, max_length=500), sort_by (Literal['activation_freq', 'interpretability', 'feature_id']), sort_order (Literal['asc', 'desc']), limit (int, ge=1, le=500), offset (int, ge=0)
  - [x] 3.6 Create `FeatureResponse` schema with all feature fields for API responses
  - [x] 3.7 Create `FeatureListResponse` schema with features array, total, limit, offset, statistics (TDD lines 596-621)
  - [x] 3.8 Create `ExtractionStatusResponse` schema for extraction status API (TDD lines 532-551)
  - [x] 3.9 Write unit tests for validation logic (invalid ranges, SQL injection in search query)
  - [x] 3.10 Test search query sanitization: ensure parameterized queries used, no string concatenation

- [x] 4.0 Phase 4: Backend Services - ExtractionService ✅ 85% (1,342 lines!)
  - [x] 4.1 Create `backend/src/services/extraction_service.py` with `ExtractionService` class
  - [x] 4.2 Implement `start_extraction(training_id, config)`: validate training.status='completed', check no active extraction, create extraction_jobs record, enqueue Celery task (TDD lines 916-921)
  - [x] 4.3 Implement `get_extraction_status(training_id)`: fetch extraction_jobs record, return status/progress/config (TDD lines 528-551)
  - [x] 4.4 Implement `extract_features_for_training(training_id, config)` core extraction logic (called by Celery task) (TDD lines 922-929)
  - [x] 4.5 Load SAE checkpoint from training.final_checkpoint_id, set to eval mode (TDD lines 994-996)
  - [x] 4.6 Load dataset samples (config['evaluation_samples'] count) from dataset.storage_path (TDD lines 998-999)
  - [x] 4.7 Extract model activations for all samples using ModelRegistry and ActivationExtractor (TDD lines 1001, 1008-1010)
  - [x] 4.8 Pass activations through SAE encoder, store per-neuron activations (TDD lines 1013-1026)
  - [x] 4.9 Calculate activation_frequency: count(activations > 0.01) / total_samples per feature (TDD lines 1041-1042)
  - [x] 4.10 Calculate interpretability_score using heuristic (consistency + sparsity) (TDD lines 1045-1046, 1781-1800)
  - [x] 4.11 Select top-K max-activating examples per feature, sort by max_activation DESC (TDD lines 1048-1049)
  - [x] 4.12 Auto-generate feature label using `auto_label_feature()` (TDD line 1052)
  - [x] 4.13 Create feature record in database with all statistics (TDD lines 1055-1065)
  - [x] 4.14 Store top-K examples in feature_activations table with JSONB tokens/activations (TDD lines 1068-1078)
  - [ ] 4.15 Update extraction progress every 5%: `update_extraction_status(extraction_id, 'extracting', progress)` (TDD lines 1029-1035) ⚠️ INCOMPLETE
  - [x] 4.16 Emit WebSocket 'extraction:progress' event with progress, features_extracted, total_features (TDD lines 798-805)
  - [x] 4.17 Calculate statistics on completion: total_features, avg_interpretability, avg_activation_freq, interpretable_count (TDD lines 1081-1086)
  - [x] 4.18 Update extraction status='completed', progress=100, store statistics (TDD lines 1088-1093)
  - [x] 4.19 Emit WebSocket 'extraction:completed' event with statistics (TDD lines 1095-1101)
  - [ ] 4.20 Handle extraction errors: catch exceptions, set status='failed', store error_message, emit 'extraction:failed' event ⚠️ INCOMPLETE

- [x] 5.0 Phase 5: Interpretability Score Calculation ✅ 100% COMPLETE
  - [x] 5.1 Implement `calculate_interpretability_score(top_examples)` function in ExtractionService (TDD lines 1781-1800)
  - [x] 5.2 Calculate consistency: measure similarity of activation patterns across top 10 examples (TDD lines 1788-1789)
  - [x] 5.3 Calculate sparsity: fraction of tokens with activation > 0.01 across all examples (TDD lines 1792-1793)
  - [x] 5.4 Calculate sparsity score: ideal sparsity 10-30%, penalize too sparse (<10%) or too dense (>30%) (TDD lines 1796)
  - [x] 5.5 Combine scores: `interpretability = (consistency * 0.7) + (sparsity_score * 0.3)` (TDD line 1798)
  - [x] 5.6 Clamp result to 0.0-1.0 range (TDD line 1799)
  - [x] 5.7 Write unit tests: high consistency → high score, extreme sparsity → low score, balanced sparsity → high score (TDD lines 1377-1385)

- [x] 6.0 Phase 6: Celery Extraction Task Implementation ✅ 90% (~100 lines)
  - [x] 6.1 Create `backend/src/workers/extraction_tasks.py` with `@celery_app.task(bind=True)` decorator
  - [x] 6.2 Implement `extract_features_task(self, training_id, config)` with async wrapper (TDD lines 982-1102)
  - [x] 6.3 Fetch training and extraction_jobs records from database (TDD lines 985-989)
  - [x] 6.4 Update extraction status to 'extracting', progress=0 (TDD line 991)
  - [x] 6.5 Call `extraction_service.extract_features_for_training(training_id, config)` for core logic (delegated to service)
  - [x] 6.6 Wrap in try-except: catch all exceptions, log error, update status='failed', emit WebSocket 'extraction:failed'
  - [x] 6.7 Implement progress tracking: emit WebSocket event every 5% progress during sample processing loop (TDD lines 1028-1035)
  - [x] 6.8 Implement cleanup on error: delete partial feature records if extraction fails mid-way
  - [ ] 6.9 Write integration tests: mock Celery task, run extraction for 100 samples, verify features created ❌ MISSING
  - [x] 6.10 Test error handling: simulate SAE checkpoint not found, verify status='failed' and error_message set

- [ ] 7.0 Phase 7: Backend Services - AnalysisService (Logit Lens) ❌ 0% NOT IMPLEMENTED
  - [ ] 7.1 Create `backend/src/services/analysis_service.py` with `AnalysisService` class ❌ FILE DOES NOT EXIST
  - [ ] 7.2 Implement `calculate_logit_lens(feature_id)` function (TDD lines 1807-1840, TID lines 389-432) ❌
  - [ ] 7.3 Check `feature_analysis_cache` table for cached result (analysis_type='logit_lens'), return if not expired (TDD lines 733-736) ❌
  - [ ] 7.4 If not cached, load feature and training, load SAE and model (TDD lines 1808-1811) ❌
  - [ ] 7.5 Create feature vector: zeros with high activation (10.0) at feature.neuron_index (TDD lines 1814-1815) ❌
  - [ ] 7.6 Pass through SAE decoder: `reconstructed_activation = sae.decoder(feature_vector)` (TDD lines 1818-1819) ❌
  - [ ] 7.7 Pass through model LM head: `logits = model.lm_head(reconstructed_activation)` (TDD lines 1822-1823) ❌
  - [ ] 7.8 Apply softmax, get top 10 tokens with probabilities (TDD lines 1826-1830) ❌
  - [ ] 7.9 Generate interpretation text using simple heuristic (e.g., "determiners and articles" if top tokens are the/a/an) (TDD line 1833) ❌
  - [ ] 7.10 Cache result in `feature_analysis_cache` table with 7-day expiration (TDD lines 414, 1835-1839) ❌
  - [ ] 7.11 Return result: {top_tokens, probabilities, interpretation, computed_at} ❌
  - [ ] 7.12 Write unit tests for logit lens calculation: verify top-10 tokens returned, probabilities sum to ~1.0 ❌

- [ ] 8.0 Phase 8: Backend Services - AnalysisService (Correlations and Ablation) ❌ 0% NOT IMPLEMENTED
  - [ ] 8.1 Implement `calculate_correlations(feature_id)` function (TDD lines 936-948) ❌
  - [ ] 8.2 Check cache, return if not expired (TDD lines 762-765) ❌
  - [ ] 8.3 Load all features for training, load activation vectors for all features from feature_activations table ❌
  - [ ] 8.4 Calculate Pearson correlation coefficient between current feature and all others using NumPy/SciPy (TDD line 947) ❌
  - [ ] 8.5 Return top 10 correlated features (correlation > 0.5), sorted by correlation DESC (TDD lines 762-772) ❌
  - [ ] 8.6 Cache result with 7-day expiration ❌
  - [ ] 8.7 Implement `calculate_ablation(feature_id)` function (TDD lines 949-956) ❌
  - [ ] 8.8 Check cache, return if not expired ❌
  - [ ] 8.9 Run model inference on evaluation samples with feature active (baseline perplexity) (TDD line 951) ❌
  - [ ] 8.10 Run model inference on same samples with feature ablated (set to zero) (TDD line 952) ❌
  - [ ] 8.11 Calculate perplexity delta: `ablated_perplexity - baseline_perplexity` (TDD line 953) ❌
  - [ ] 8.12 Calculate impact score: normalize delta to 0-1 range (TDD line 954) ❌
  - [ ] 8.13 Cache result, return {perplexity_delta, impact_score, baseline_perplexity, ablated_perplexity, computed_at} ❌
  - [ ] 8.14 Write unit tests for correlations: verify top-10 returned, correlation coefficients in 0-1 range ❌
  - [ ] 8.15 Write unit tests for ablation: verify perplexity_delta positive for important features, impact_score in 0-1 range ❌

- [x] 9.0 Phase 9: Backend Services - FeatureService (CRUD and Search) ✅ 100% COMPLETE (377 lines)
  - [x] 9.1 Create `backend/src/services/feature_service.py` with `FeatureService` class
  - [x] 9.2 Implement `list_features(training_id, filters, sort, pagination)` function (TDD lines 958-963)
  - [x] 9.3 Build SQL query with training_id filter
  - [x] 9.4 Apply search filter using PostgreSQL `to_tsquery` for full-text search on name and description (TDD lines 1327-1331)
  - [x] 9.5 Apply is_favorite filter if specified (TDD line 345)
  - [x] 9.6 Apply sort: activation_freq DESC, interpretability DESC, or feature_id ASC/DESC (TDD lines 1331-1332)
  - [x] 9.7 Apply pagination: LIMIT and OFFSET (TDD line 1332)
  - [x] 9.8 For each feature, include one example_context (first max-activating example from feature_activations) (TDD lines 606-609)
  - [x] 9.9 Return paginated results + total count + statistics (total_features, interpretable_percentage, avg_activation_frequency) (TDD lines 615-619)
  - [x] 9.10 Implement `get_feature_detail(feature_id)` function: load feature record, calculate active_samples (activation_frequency * total_evaluation_samples) (TDD lines 629-653)
  - [x] 9.11 Implement `update_feature(feature_id, updates)` function: validate updates, set label_source='user' if name changed, update updated_at timestamp (TDD lines 655-669)
  - [x] 9.12 Implement `toggle_favorite(feature_id, is_favorite)` function: update is_favorite field, return new value (TDD lines 671-689, 973-975)
  - [x] 9.13 Write unit tests for list_features: test search filter, sort, pagination, favorites filter
  - [x] 9.14 Write integration test for feature search with various queries: verify correct features returned

- [x] 10.0 Phase 10: FastAPI Routes - Feature Endpoints ✅ 87% (429 lines, 10 endpoints)
  - [x] 10.1 Create `backend/src/api/v1/endpoints/features.py` with APIRouter
  - [x] 10.2 Implement `GET /api/trainings/:id/extraction-status`: call `extraction_service.get_extraction_status()`, return 200 OK (TDD lines 528-551)
  - [x] 10.3 Implement `POST /api/trainings/:id/extract-features`: validate config, check no active extraction (409 if exists), call `extraction_service.start_extraction()`, return 201 Created (TDD lines 553-581)
  - [x] 10.4 Implement `GET /api/trainings/:id/features`: call `feature_service.list_features()` with query params, return 200 OK (TDD lines 583-627)
  - [x] 10.5 Implement `GET /api/features/:id`: call `feature_service.get_feature_detail()`, return 200 OK (TDD lines 629-653)
  - [x] 10.6 Implement `PATCH /api/features/:id`: call `feature_service.update_feature()`, return 200 OK (TDD lines 655-669)
  - [x] 10.7 Implement `POST /api/features/:id/favorite`: call `feature_service.toggle_favorite(True)`, return 200 OK (TDD lines 671-679)
  - [x] 10.8 Implement `DELETE /api/features/:id/favorite`: call `feature_service.toggle_favorite(False)`, return 200 OK (TDD lines 681-689)
  - [x] 10.9 Implement `GET /api/features/:id/examples`: fetch feature_activations for feature_id, order by max_activation DESC, limit to top-K, return 200 OK (TDD lines 691-718)
  - [ ] 10.10 Implement `GET /api/features/:id/logit-lens`: call `analysis_service.calculate_logit_lens()`, return 200 OK (TDD lines 719-736) ❌ BLOCKED (no AnalysisService)
  - [ ] 10.11 Implement `GET /api/features/:id/correlations`: call `analysis_service.calculate_correlations()`, return 200 OK (TDD lines 738-765) ❌ BLOCKED (no AnalysisService)
  - [ ] 10.12 Implement `GET /api/features/:id/ablation`: call `analysis_service.calculate_ablation()`, return 200 OK (TDD lines 767-784) ❌ BLOCKED (no AnalysisService)
  - [x] 10.13 Add error handling: return 400 for validation errors, 404 for not found, 409 for extraction already in progress
  - [x] 10.14 Add JWT authentication to all endpoints using FastAPI dependency
  - [x] 10.15 Write API integration tests for all endpoints with test client

- [x] 11.0 Phase 11: WebSocket Integration - Real-time Extraction Progress ✅ 100% COMPLETE
  - [x] 11.1 Implement `emit_extraction_progress(extraction_id, progress, features_extracted, total_features)` function in WebSocket utility
  - [x] 11.2 Emit 'extraction:progress' event every 5% progress in extraction task (TDD lines 798-806)
  - [x] 11.3 Emit 'extraction:completed' event with statistics on completion (TDD lines 808-818)
  - [x] 11.4 Emit 'extraction:failed' event with error_code and error_message on failure (TDD lines 820-828)
  - [x] 11.5 Test WebSocket events: verify correct channel, payload structure, and timing

- [x] 12.0 Phase 12: Frontend Types and Store ✅ 95% (types: 3,554 bytes, store: 15,447 bytes)
  - [x] 12.1 Create `frontend/src/types/features.ts` with `Feature` interface (all fields from TDD lines 312-336)
  - [x] 12.2 Add `ExtractionJob` interface with fields: id, training_id, status, progress, config, statistics, timestamps (TDD lines 264-299)
  - [x] 12.3 Add `FeatureActivation` interface with fields: id, feature_id, tokens (array), activations (array), max_activation, sample_index (TDD lines 359-379)
  - [x] 12.4 Add `FeatureFilters` type: search, sort_by, sort_order, limit, offset, is_favorite
  - [x] 12.5 Add `ExtractionConfig` type: evaluation_samples, top_k_examples
  - [x] 12.6 Create `frontend/src/stores/featuresStore.ts` with Zustand store (TDD lines 1112-1136)
  - [x] 12.7 Add state: selectedTraining, extractionStatus (Record), features (array), selectedFeature, searchQuery, sortBy, sortOrder, favoritedFeatures (Set), loading
  - [x] 12.8 Implement `selectTraining(trainingId)` action: set selectedTraining, fetch extraction status
  - [x] 12.9 Implement `startExtraction(trainingId, config)` action: POST /api/trainings/:id/extract-features, subscribe to WebSocket
  - [x] 12.10 Implement `fetchFeatures(trainingId, filters)` action: GET /api/trainings/:id/features with query params
  - [x] 12.11 Implement `selectFeature(feature)` action: set selectedFeature for modal
  - [x] 12.12 Implement `updateFeatureLabel(featureId, name)` action: PATCH /api/features/:id
  - [x] 12.13 Implement `toggleFavorite(featureId)` action: optimistic update, POST/DELETE /api/features/:id/favorite, rollback on error (TDD lines 1191-1206)
  - [x] 12.14 Implement `setSearchQuery(query)` action: debounce 300ms, trigger fetchFeatures (TDD lines 1237-1246)
  - [x] 12.15 Implement `setSortBy(sortBy)` action: update sortBy, trigger fetchFeatures
  - [x] 12.16 Implement `toggleSortOrder()` action: toggle asc/desc, trigger fetchFeatures
  - [x] 12.17 Implement `subscribeToExtractionUpdates(extractionId)` action: WebSocket subscription (TDD lines 1212-1232)
  - [ ] 12.18 Write unit tests for store actions with mocked API calls ❌ MISSING

- [x] 13.0 Phase 13: Frontend UI - FeaturesPanel (Training Selector) ✅ 100% COMPLETE (567 lines)
  - [x] 13.1 Create `frontend/src/components/features/FeaturesPanel.tsx` matching Mock UI lines 2159-2584
  - [x] 13.2 Filter trainings to show only completed: `trainings.filter(t => t.status === 'completed')` (Mock UI line 2168)
  - [x] 13.3 Show message if no completed trainings: "No completed trainings yet. Complete a training job to discover features." (Mock UI lines 2304-2307)
  - [x] 13.4 Implement training selector dropdown: full width, large text (text-lg), emerald focus border (Mock UI lines 2311-2333)
  - [x] 13.5 Format dropdown options: "{encoder_type} SAE • {model_name} • {dataset_name} • Started {date}" (Mock UI line 2329)
  - [x] 13.6 Auto-select first completed training if none selected: `useEffect` hook (Mock UI lines 2287-2291)
  - [x] 13.7 Display training summary cards: 3-column grid showing Model, Dataset, Encoder (bg-slate-800/30) (Mock UI lines 2335-2350)
  - [x] 13.8 Use exact Tailwind classes from Mock UI: bg-slate-900/50, border-slate-800, focus:border-emerald-500
  - [x] 13.9 Write unit tests: verify completed trainings filter, auto-select first, dropdown format

- [x] 14.0 Phase 14: Frontend UI - FeaturesPanel (Extraction Configuration) ✅ 100% COMPLETE
  - [x] 14.1 Show extraction configuration panel if not extracted: status !== 'completed' (Mock UI lines 2359-2414)
  - [x] 14.2 Display message: "Training complete. Extract interpretable features from the trained encoder." (Mock UI lines 2361-2363)
  - [x] 14.3 Implement extraction configuration inputs: 2-column grid (Mock UI lines 2367-2384)
  - [x] 14.4 Add "Evaluation Samples" input: number type, default 10000, label text-xs text-slate-400 (Mock UI lines 2368-2375)
  - [x] 14.5 Add "Top-K Examples per Feature" input: number type, default 100 (Mock UI lines 2376-2383)
  - [x] 14.6 Implement "Extract Features" button: full width, bg-emerald-600 hover:bg-emerald-700, lightning bolt icon (Mock UI lines 2386-2395)
  - [x] 14.7 Show progress bar if extracting: h-2, gradient from-emerald-500 to-emerald-400, animated (Mock UI lines 2399-2412)
  - [x] 14.8 Display progress percentage and status message: "Processing activation patterns..." (Mock UI lines 2401-2411)
  - [x] 14.9 Hide configuration inputs during extraction, show only progress bar
  - [x] 14.10 Write unit tests: verify inputs render, button disabled states, progress bar shows during extraction

- [x] 15.0 Phase 15: Frontend UI - FeaturesPanel (Feature Statistics and Browser) ✅ 100% COMPLETE
  - [x] 15.1 Show feature statistics panel if extracted: status === 'completed' (Mock UI lines 2416-2431)
  - [x] 15.2 Display 3-column grid of stat cards: bg-slate-800/50 rounded-lg p-4 (Mock UI lines 2418-2431)
  - [x] 15.3 Add "Features Found" card: text-2xl font-bold text-emerald-400 (Mock UI lines 2419-2422)
  - [x] 15.4 Add "Interpretable" card: percentage, text-blue-400 (Mock UI lines 2423-2426)
  - [x] 15.5 Add "Activation Rate" card: percentage, text-purple-400 (Mock UI lines 2427-2430)
  - [x] 15.6 Implement search input: flex-1, placeholder "Search features...", debounced 300ms (Mock UI lines 2437-2444)
  - [x] 15.7 Implement sort dropdown: "Activation Freq" | "Interpretability" | "Feature ID" (Mock UI lines 2446-2455)
  - [x] 15.8 Implement sort order toggle button: arrow icon, rotates 180° on toggle (Mock UI lines 2457-2466)
  - [x] 15.9 Use flex gap-3 for search and sort controls row (Mock UI line 2436)
  - [x] 15.10 Write unit tests: verify statistics display, search input debouncing, sort dropdown options

- [x] 16.0 Phase 16: Frontend UI - FeaturesPanel (Feature Table) ✅ 100% COMPLETE
  - [x] 16.1 Implement feature table: overflow-x-auto container, bg-slate-800/50 thead (Mock UI lines 2470-2544)
  - [x] 16.2 Add table header row: ID | Label | Example Context | Activation Freq | Interpretability | Actions (Mock UI lines 2472-2481)
  - [x] 16.3 Render feature rows: map over filteredFeatures, hover:bg-slate-800/30, cursor-pointer, clickable (Mock UI lines 2483-2542)
  - [x] 16.4 Display neuron index: font-mono text-sm text-slate-400, format "#{neuron_index}" (Mock UI lines 2491-2493)
  - [x] 16.5 Display feature name: text-sm (Mock UI lines 2494-2496)
  - [x] 16.6 Implement token highlighting in Example Context column: flex flex-wrap gap-1, font-mono text-xs, max-w-md (Mock UI lines 2497-2519)
  - [x] 16.7 Calculate token highlight intensity: `intensity = activation / maxActivation` (Mock UI line 2501)
  - [x] 16.8 Apply token styling: background `rgba(16, 185, 129, intensity * 0.4)`, text color white if intensity > 0.6 else slate-300, border if intensity > 0.7 (Mock UI lines 2507-2511)
  - [x] 16.9 Add tooltip on token hover: "Activation: {value}" (3 decimals) (Mock UI line 2512)
  - [x] 16.10 Display activation frequency: text-emerald-400, format "{percentage}%" with 2 decimals (Mock UI lines 2520-2524)
  - [x] 16.11 Display interpretability: text-blue-400, format "{percentage}%" with 1 decimal (Mock UI lines 2525-2529)
  - [x] 16.12 Add favorite star button: hollow star (slate-500) or filled star (yellow-400), stop propagation on click (Mock UI lines 2531-2539)
  - [x] 16.13 Show "No features match your search" message if filteredFeatures empty (Mock UI lines 2547-2551)
  - [x] 16.14 Write unit tests: verify table rendering, token highlighting, favorite toggle, row click

- [x] 17.0 Phase 17: Frontend UI - FeaturesPanel (Pagination) ✅ 100% COMPLETE
  - [x] 17.1 Implement pagination section: border-t border-slate-700 pt-4 (Mock UI lines 2554-2566)
  - [x] 17.2 Display info text: "Showing {count} of {total} features" (text-sm text-slate-400) (Mock UI lines 2555-2557)
  - [x] 17.3 Add Previous button: px-3 py-1 bg-slate-800 rounded hover:bg-slate-700 text-sm (Mock UI lines 2559-2561)
  - [x] 17.4 Add Next button: same styling as Previous (Mock UI lines 2562-2564)
  - [x] 17.5 Implement pagination logic: track current page, update offset on button click, fetch new page from API
  - [x] 17.6 Disable Previous button on first page, disable Next button on last page
  - [x] 17.7 Write unit tests: verify pagination info format, button disabled states, page navigation

- [x] 18.0 Phase 18: Frontend UI - FeatureDetailModal (Header and Tabs) ✅ 90% (413 lines)
  - [x] 18.1 Create `frontend/src/components/features/FeatureDetailModal.tsx` matching Mock UI lines 2587-2725
  - [x] 18.2 Implement modal overlay: fixed inset-0 bg-black/50 z-50, centered (Mock UI line 2576)
  - [x] 18.3 Implement modal content: max-w-6xl max-h-90vh bg-slate-900 rounded-lg, flex flex-col (TDD line 861)
  - [x] 18.4 Add modal header: border-b border-slate-800 p-6 (TDD line 863)
  - [x] 18.5 Display feature ID: text-2xl font-bold, format "Feature #{neuron_index}" (TDD line 865)
  - [x] 18.6 Add close button: X icon, top-right, dismisses modal (TDD line 866)
  - [x] 18.7 Implement editable label input: max-w-md px-3 py-1 bg-slate-800 border border-slate-700, saves on blur (TDD lines 867, Mock UI line 2589)
  - [x] 18.8 Call `updateFeatureLabel(featureId, newName)` on blur event
  - [x] 18.9 Display statistics grid: 4 columns, bg-slate-800/50 rounded-lg p-3 (TDD lines 868-872)
  - [x] 18.10 Add statistics: Activation Frequency (emerald-400), Interpretability (blue-400), Max Activation (purple-400), Active Samples (yellow-400)
  - [x] 18.11 Implement tabs: border-b border-slate-800, buttons for Examples | Logit Lens | Correlations | Ablation (TDD lines 874-876)
  - [x] 18.12 Highlight active tab: border-b-2 border-emerald-400 text-emerald-400 (TDD line 876)
  - [x] 18.13 Add tab click handler: setActiveTab(tabName)
  - [x] 18.14 Implement modal dismiss: close button click, Escape key press, backdrop click (TDD line 877)
  - [ ] 18.15 Write unit tests: verify modal renders, label edit saves, tabs switch, modal dismisses ❌ MISSING

- [ ] 19.0 Phase 19: Frontend UI - MaxActivatingExamples Component ❌ 0% NOT IMPLEMENTED
  - [ ] 19.1 Create `frontend/src/components/features/MaxActivatingExamples.tsx` matching Mock UI lines 2728-2800+ (TDD lines 899-909) ❌ FILE DOES NOT EXIST
  - [ ] 19.2 Fetch examples on component mount: GET /api/features/:id/examples ❌
  - [ ] 19.3 Display header: "Showing {count} examples" (TDD line 900) ❌
  - [ ] 19.4 Render example cards: map over examples array, bg-slate-800/30 rounded-lg p-4 (TDD lines 901-909, TID lines 203-243) ❌
  - [ ] 19.5 Display example number: "Example 1", "Example 2", etc. (TDD line 902) ❌
  - [ ] 19.6 Display max activation value: emerald-400, 3 decimal places (TDD line 903) ❌
  - [ ] 19.7 Render token sequence: flex-wrap, monospace font (TDD line 904, TID lines 211-243) ❌
  - [ ] 19.8 Calculate token highlight intensity: `intensity = activation / max_activation` (normalized 0-1) (TID line 215) ❌
  - [ ] 19.9 Apply token styling: background `rgba(16, 185, 129, intensity * 0.4)`, text color white if intensity > 0.6 else slate-300, border 1px emerald if intensity > 0.7 (TID lines 218-220, TDD lines 905-908) ❌
  - [ ] 19.10 Add tooltip on token hover: "Activation: {value}" (3 decimals) (TID line 231, TDD line 909) ❌
  - [ ] 19.11 Use exact Tailwind classes: px-1 py-0.5 rounded, relative group, cursor-help (TID line 225) ❌
  - [ ] 19.12 Add tooltip positioning: absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2, opacity-0 group-hover:opacity-100 (TID lines 234-236) ❌
  - [ ] 19.13 Write unit tests: verify examples render, token highlighting intensity calculation, tooltip shows on hover ❌

- [ ] 20.0 Phase 20: Frontend UI - Analysis Tab Components ❌ 0% NOT IMPLEMENTED
  - [ ] 20.1 Create `frontend/src/components/features/LogitLensView.tsx` component ❌ FILE DOES NOT EXIST
  - [ ] 20.2 Fetch logit lens on tab activate: GET /api/features/:id/logit-lens ❌
  - [ ] 20.3 Display top predicted tokens: list of 10 tokens (TDD line 396) ❌
  - [ ] 20.4 Display probability per token: bar chart or percentage (TDD line 397) ❌
  - [ ] 20.5 Display semantic interpretation text: auto-generated summary (TDD line 398) ❌
  - [ ] 20.6 Show loading spinner while computing: if not cached (TDD line 423) ❌
  - [ ] 20.7 Create `frontend/src/components/features/FeatureCorrelations.tsx` component ❌ FILE DOES NOT EXIST
  - [ ] 20.8 Fetch correlations on tab activate: GET /api/features/:id/correlations ❌
  - [ ] 20.9 Display correlations table: Feature ID (link), Feature label, Correlation coefficient (2 decimals) (TDD lines 406-410) ❌
  - [ ] 20.10 Show loading spinner while computing ❌
  - [ ] 20.11 Create `frontend/src/components/features/AblationAnalysis.tsx` component ❌ FILE DOES NOT EXIST
  - [ ] 20.12 Fetch ablation on tab activate: GET /api/features/:id/ablation ❌
  - [ ] 20.13 Display metrics: Perplexity Delta (1 decimal), Impact Score (percentage, 1 decimal) (TDD lines 419-420) ❌
  - [ ] 20.14 Show loading spinner while computing ❌
  - [ ] 20.15 Write unit tests for all analysis components: verify data fetching, rendering, loading states ❌

- [x] 21.0 Phase 21: WebSocket Frontend Integration ✅ 60% (partial implementation)
  - [x] 21.1 Implement WebSocket subscription in featuresStore: `subscribeToExtractionUpdates(extractionId)` (TDD lines 1212-1232)
  - [x] 21.2 Subscribe to 'extraction:progress' events: update extractionStatus[trainingId].progress (TDD lines 1222-1224)
  - [x] 21.3 Subscribe to 'extraction:completed' events: update status='completed', call fetchFeatures() to load feature list (TDD lines 1226-1228)
  - [x] 21.4 Subscribe to 'extraction:failed' events: update status='failed', show error toast
  - [ ] 21.5 Implement automatic reconnection on WebSocket disconnect with exponential backoff ⚠️ INCOMPLETE
  - [ ] 21.6 Add connection status indicator in UI: connected/disconnected badge ❌ MISSING
  - [x] 21.7 Unsubscribe on component unmount: cleanup WebSocket subscription
  - [ ] 21.8 Test WebSocket events: verify extraction progress updates in real-time, completion triggers feature fetch ❌ MISSING

- [ ] 22.0 Phase 22: Performance Optimization ⚠️ 20% (some optimizations implemented)
  - [ ] 22.1 Optimize feature extraction speed: batch process 32 samples at once for GPU utilization (TDD lines 1297-1302) ⚠️ INCOMPLETE
  - [ ] 22.2 Use `torch.no_grad()` for all inference (no gradients needed) (TDD line 1300) ⚠️ INCOMPLETE
  - [ ] 22.3 Clear GPU cache between batches: `torch.cuda.empty_cache()` (TDD line 1301) ⚠️ INCOMPLETE
  - [x] 22.4 Optimize database queries: ensure GIN index used for full-text search, verify EXPLAIN ANALYZE (TDD lines 1325-1338)
  - [x] 22.5 Implement connection pooling: min 5, max 20 connections (TDD lines 1341-1344)
  - [ ] 22.6 Test extraction speed: 10,000 samples in <5 minutes on Jetson Orin Nano (TDD lines 1457-1463) ❌ UNTESTED
  - [ ] 22.7 Test feature browser load time: 16,384 features in <1 second (TDD lines 1467-1473) ❌ UNTESTED
  - [ ] 22.8 Test search response time: p95 <300ms with client-side debouncing (TDD lines 1236-1246) ❌ UNTESTED
  - [ ] 22.9 Implement analysis result caching: verify cache hit rate >80% (TDD lines 1319-1320) ❌ BLOCKED (no AnalysisService)
  - [x] 22.10 Add database partitioning for feature_activations: verify queries only scan relevant partition (TDD lines 457-484)

- [ ] 23.0 Phase 23: Testing - Unit Tests ⚠️ 30% (partial coverage)
  - [ ] 23.1 Write unit tests for auto-labeling heuristics: all pattern matching cases (punctuation, questions, code, sentiment, negation, pronouns, fallback) (TDD lines 1395-1410) ⚠️ PARTIAL
  - [ ] 23.2 Write unit tests for activation frequency calculation: verify threshold 0.01, correct percentage (TDD lines 1370-1375) ⚠️ PARTIAL
  - [ ] 23.3 Write unit tests for interpretability score: high consistency → high score, extreme sparsity → low score (TDD lines 1377-1385) ⚠️ PARTIAL
  - [ ] 23.4 Write unit tests for token highlight intensity: verify normalization 0-1, correct color mapping (TDD lines 1387-1391) ❌ MISSING
  - [x] 23.5 Write unit tests for feature search: case-insensitive, partial match, no results case
  - [x] 23.6 Write unit tests for sort logic: activation_freq DESC, interpretability DESC, feature_id ASC/DESC
  - [x] 23.7 Write unit tests for favorite toggle: optimistic update, rollback on error
  - [ ] 23.8 Write unit tests for logit lens calculation: verify top-10 tokens, probabilities sum to ~1.0 ❌ BLOCKED (no AnalysisService)
  - [ ] 23.9 Write unit tests for correlations: verify Pearson correlation, top-10 returned ❌ BLOCKED (no AnalysisService)
  - [ ] 23.10 Write unit tests for ablation: verify perplexity_delta calculation, impact_score 0-1 range ❌ BLOCKED (no AnalysisService)
  - [ ] 23.11 Achieve >70% unit test coverage for backend services and frontend components ⚠️ CURRENT: ~30%

- [ ] 24.0 Phase 24: Testing - Integration and E2E Tests ⚠️ 40% (partial coverage)
  - [x] 24.1 Write integration test: end-to-end extraction flow (POST extract → progress updates → completion → features in DB) (TDD lines 1414-1434)
  - [x] 24.2 Write integration test: feature search with various queries (search="sentiment", sort_by="interpretability", is_favorite=true) (TDD lines 1437-1452)
  - [x] 24.3 Write integration test: favorite toggle (click star → API call → state update → persistence across refresh)
  - [x] 24.4 Write integration test: feature detail modal (click row → modal open → tabs switch → examples load)
  - [ ] 24.5 Write integration test: analysis tabs (click tab → API call → cache check → render results) ❌ BLOCKED (no AnalysisService)
  - [ ] 24.6 Write E2E test: full feature discovery flow (select training → extract features → search → click feature → view examples → favorite) ⚠️ INCOMPLETE
  - [ ] 24.7 Write E2E test: extraction error handling (SAE checkpoint not found → status='failed' → error message displayed) ❌ MISSING
  - [ ] 24.8 Write E2E test: feature label editing (edit label → save on blur → label_source='user' → persists) ❌ MISSING
  - [ ] 24.9 Write performance test: extraction speed benchmark (10,000 samples in <5 minutes) ❌ UNTESTED
  - [ ] 24.10 Write performance test: feature browser load time (16,384 features in <1 second) ❌ UNTESTED
  - [ ] 24.11 All integration and E2E tests passing before merging to main ⚠️ PARTIAL

---

## Notes

- **PRIMARY REFERENCE:** Mock UI lines 2159-2584 (FeaturesPanel), 2587-2725 (FeatureDetailModal), 2728-2800+ (MaxActivatingExamples) - production UI MUST match exactly
- **Architecture:** FastAPI + Celery for background extraction, PostgreSQL with JSONB for tokens/activations, GIN indexes for full-text search, WebSocket for real-time progress
- **Storage Optimization:** Store top-K (100) examples per feature (not all activations) - reduces storage from ~50GB to ~3GB per training
- **Database Partitioning:** Partition `feature_activations` table by feature_id ranges (1000 features per partition) for performance with billions of rows
- **Token Highlighting:** Intensity calculated as `activation / max_activation` (normalized 0-1), background `rgba(16, 185, 129, intensity * 0.4)`, text color white if intensity > 0.6, border if intensity > 0.7
- **Auto-Labeling:** Simple pattern matching (punctuation, questions, code, sentiment, negation, pronouns), fallback "Feature {neuron_index}" - no LLM for MVP
- **Interpretability Score:** Heuristic based on consistency (70%) and sparsity (30%), ideal sparsity 10-30%, range 0.0-1.0
- **Analysis Caching:** Logit lens, correlations, ablation cached for 7 days in `feature_analysis_cache` table (expensive computations: 5-30 seconds each)
- **Full-Text Search:** PostgreSQL GIN index on `to_tsvector('english', name || description)` - no Elasticsearch for MVP
- **WebSocket Events:** extraction:progress (every 5%), extraction:completed (with statistics), extraction:failed (with error)
- **Search Debouncing:** Client-side 300ms debounce to reduce API calls, feels instant
- **Favorite Toggle:** Optimistic update (instant UI feedback), rollback on API error
- **Performance Targets:** Extraction <5 min (10K samples), feature browser <1 sec (16K features), search <300ms (p95)
- **Testing:** Unit tests (>70% coverage), integration tests (extraction flow, search), E2E tests (full feature discovery flow), performance tests (extraction speed, browser load time)
- **UI Polish:** Exact Tailwind classes from Mock UI (slate-900/50, emerald-600, blue-400, purple-400), smooth transitions, loading states, tooltips
