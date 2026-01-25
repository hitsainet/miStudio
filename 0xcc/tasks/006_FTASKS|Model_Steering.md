# Feature Tasks: Model Steering

**Document ID:** 006_FTASKS|Model_Steering
**Version:** 1.1 (Combined Mode Enhancement)
**Last Updated:** 2026-01-24
**Status:** Partially Implemented (Phase 7 Pending)
**Related PRD:** [006_FPRD|Model_Steering](../prds/006_FPRD|Model_Steering.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Hook Infrastructure | 3 tasks | ✅ Complete |
| Phase 2: Steering Service | 4 tasks | ✅ Complete |
| Phase 3: API Endpoints | 4 tasks | ✅ Complete |
| Phase 4: Prompt Templates | 4 tasks | ✅ Complete |
| Phase 5: Frontend Store | 3 tasks | ✅ Complete |
| Phase 6: UI Components | 6 tasks | ✅ Complete |
| Phase 7: Combined Multi-Feature Mode | 7 tasks | ⏳ Pending |

**Total: 31 tasks (24 complete, 7 pending)**

---

## Phase 1: Hook Infrastructure

### Task 1.1: Create Forward Hooks Module
- [x] Define SteeringConfig dataclass
- [x] Create SteeringHookManager class
- [x] Implement hook registration
- [x] Implement hook cleanup

**Files:**
- `backend/src/ml/forward_hooks.py`

### Task 1.2: Implement Steering Hook
- [x] Encode through SAE
- [x] Modify feature activation
- [x] Decode back to residual
- [x] Handle tuple outputs

### Task 1.3: Multi-Feature Support
- [x] Accept list of configs
- [x] Apply interventions sequentially
- [x] Filter by layer

---

## Phase 2: Steering Service

### Task 2.1: Create Steering Service
- [x] Initialize with model/SAE
- [x] Set up hook manager
- [x] Implement generate_steered()

**Files:**
- `backend/src/services/steering_service.py`

### Task 2.2: Implement Comparison Mode
- [x] Generate baseline (no steering)
- [x] Generate steered output
- [x] Return both results

### Task 2.3: Implement Strength Sweep
- [x] Accept strength list
- [x] Generate output per strength
- [x] Return results array

### Task 2.4: Implement Calibration
- [x] Compute calibration factor
- [x] Based on decoder weight norm
- [x] Cache calibration values

---

## Phase 3: API Endpoints

### Task 3.1: Create Router
- [x] Define router
- [x] Add to main router

**Files:**
- `backend/src/api/v1/endpoints/steering.py`

### Task 3.2: Generate Endpoint
- [x] POST /steering/generate
- [x] Accept feature configs
- [x] Accept generation params
- [x] Return generated text

### Task 3.3: Compare Endpoint
- [x] POST /steering/compare
- [x] Return baseline and steered
- [x] Include feature configs

### Task 3.4: Sweep Endpoint
- [x] POST /steering/sweep
- [x] Accept strength array
- [x] Return results array

---

## Phase 4: Prompt Templates

### Task 4.1: Create Template Migration
- [x] Create prompt_templates table
- [x] Add variables JSONB column
- [x] Add template_type column

**Files:**
- `backend/alembic/versions/xxx_create_prompt_templates.py`

### Task 4.2: Template Service
- [x] CRUD operations
- [x] Variable substitution
- [x] Favorites management

**Files:**
- `backend/src/services/prompt_template_service.py`

### Task 4.3: Template API
- [x] GET /prompt-templates
- [x] POST /prompt-templates
- [x] PUT /prompt-templates/{id}
- [x] DELETE /prompt-templates/{id}

**Files:**
- `backend/src/api/v1/endpoints/prompt_templates.py`

### Task 4.4: Import/Export
- [x] Export to JSON
- [x] Import from JSON
- [x] Validate structure

---

## Phase 5: Frontend Store

### Task 5.1: Create Types
- [x] Define SelectedFeature interface
- [x] Define SteeringResult interface
- [x] Define PromptTemplate interface

**Files:**
- `frontend/src/types/steering.ts`
- `frontend/src/types/promptTemplate.ts`

### Task 5.2: Create API Client
- [x] generate() function
- [x] compare() function
- [x] sweep() function
- [x] Template CRUD functions

**Files:**
- `frontend/src/api/steering.ts`
- `frontend/src/api/promptTemplates.ts`

### Task 5.3: Create Steering Store
- [x] Selected features state
- [x] Prompt state
- [x] Generation config state
- [x] Results state
- [x] Actions for all operations

**Files:**
- `frontend/src/stores/steeringStore.ts`

---

## Phase 6: UI Components

### Task 6.1: Create StrengthSlider
- [x] Range -10 to +10
- [x] Color gradient (red/gray/green)
- [x] Current value display

**Files:**
- `frontend/src/components/steering/StrengthSlider.tsx`

### Task 6.2: Create SelectedFeatureCard
- [x] Feature info display
- [x] Strength slider integration
- [x] Remove button
- [x] View details button

**Files:**
- `frontend/src/components/steering/SelectedFeatureCard.tsx`

### Task 6.3: Create FeatureBrowser Integration
- [x] Add to steering action
- [x] Modal in steering panel
- [x] Search and filter

**Files:**
- `frontend/src/components/steering/FeatureBrowser.tsx`

### Task 6.4: Create ComparisonResults
- [x] Side-by-side layout
- [x] Diff highlighting
- [x] Copy buttons
- [x] Export button

**Files:**
- `frontend/src/components/steering/ComparisonResults.tsx`

### Task 6.5: Create PromptListEditor
- [x] Multi-prompt input
- [x] Template loading
- [x] Variable handling

**Files:**
- `frontend/src/components/steering/PromptListEditor.tsx`

### Task 6.6: Create SteeringPanel
- [x] SAE selector
- [x] Feature selection area
- [x] Prompt input
- [x] Generation config
- [x] Results display
- [x] Comparison toggle

**Files:**
- `frontend/src/components/panels/SteeringPanel.tsx`

---

## Phase 7: Combined Multi-Feature Mode (FR-2.5)

### Task 7.1: Create Combined Steering Hook
- [ ] Create `CombinedSteeringHook` class in `forward_hooks.py`
- [ ] Implement `_compute_combined_vector()` method
- [ ] Pre-compute steering direction from SAE decoder weights
- [ ] Apply accumulated steering to residual stream
- [ ] Handle calibration factor per feature

**Files:**
- `backend/src/ml/forward_hooks.py`

### Task 7.2: Add Combined Generation Service Method
- [ ] Add `generate_combined()` method to `SteeringService`
- [ ] Support `include_baseline` flag for comparison
- [ ] Return combined output with all features applied
- [ ] Add calibration factor batching

**Files:**
- `backend/src/services/steering_service.py`

### Task 7.3: Create Combined API Endpoint
- [ ] Add `CombinedSteeringRequest` schema
- [ ] Add `CombinedSteeringResponse` schema
- [ ] Implement POST `/steering/combined` endpoint
- [ ] Add endpoint to router

**Files:**
- `backend/src/schemas/steering.py`
- `backend/src/api/v1/endpoints/steering.py`

### Task 7.4: Update Frontend API Client
- [ ] Add `generateCombined()` function to `steering.ts`
- [ ] Define `CombinedSteeringRequest` type
- [ ] Define `CombinedSteeringResponse` type

**Files:**
- `frontend/src/api/steering.ts`
- `frontend/src/types/steering.ts`

### Task 7.5: Update Steering Store
- [ ] Add `combinedMode: boolean` state
- [ ] Add `combinedResults: CombinedSteeringResult | null` state
- [ ] Add `setCombinedMode()` action
- [ ] Add `generateCombined()` action
- [ ] Modify `generate()` to branch based on mode

**Files:**
- `frontend/src/stores/steeringStore.ts`

### Task 7.6: Add Combined Mode UI
- [ ] Add "Combined Mode" checkbox to SteeringPanel
- [ ] Disable when < 2 features selected
- [ ] Show tooltip explaining combined mode
- [ ] Update Generate button text based on mode

**Files:**
- `frontend/src/components/panels/SteeringPanel.tsx`

### Task 7.7: Create Combined Results Display
- [ ] Create `CombinedResults.tsx` component
- [ ] Show baseline vs combined side-by-side (if comparison enabled)
- [ ] Display list of applied features with strengths
- [ ] Add diff highlighting
- [ ] Support export to JSON

**Files:**
- `frontend/src/components/steering/CombinedResults.tsx`

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `backend/src/ml/forward_hooks.py` | Hook infrastructure |
| `backend/src/services/steering_service.py` | Steering logic |
| `backend/src/schemas/steering.py` | Request/response schemas |
| `backend/src/api/v1/endpoints/steering.py` | API routes |
| `backend/src/models/prompt_template.py` | Template model |
| `backend/src/services/prompt_template_service.py` | Template service |
| `backend/src/api/v1/endpoints/prompt_templates.py` | Template API |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/types/steering.ts` | TypeScript types |
| `frontend/src/api/steering.ts` | API client |
| `frontend/src/stores/steeringStore.ts` | Zustand store |
| `frontend/src/components/steering/StrengthSlider.tsx` | Slider |
| `frontend/src/components/steering/SelectedFeatureCard.tsx` | Card |
| `frontend/src/components/steering/ComparisonResults.tsx` | Results |
| `frontend/src/components/steering/CombinedResults.tsx` | Combined mode results (Planned) |
| `frontend/src/components/panels/SteeringPanel.tsx` | Panel |

---

## Estimated Effort

### Phase 7: Combined Multi-Feature Mode
| Task | Estimate |
|------|----------|
| 7.1 Combined Steering Hook | 2-3 hours |
| 7.2 Service Method | 1-2 hours |
| 7.3 API Endpoint | 1 hour |
| 7.4 Frontend API Client | 30 min |
| 7.5 Store Updates | 1 hour |
| 7.6 UI Toggle | 30 min |
| 7.7 Combined Results Component | 2 hours |
| **Total** | **8-10 hours** |

---

*Related: [PRD](../prds/006_FPRD|Model_Steering.md) | [TDD](../tdds/006_FTDD|Model_Steering.md) | [TID](../tids/006_FTID|Model_Steering.md)*
