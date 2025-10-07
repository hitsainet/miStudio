# Mock UI Enhancement Update Plan

**Document ID:** Mock_UI_Enhancement_Update_Plan
**Created:** 2025-10-07
**Status:** Planning Phase
**Purpose:** Comprehensive plan to update all project documentation to incorporate Mock UI enhancements

---

## Executive Summary

The Mock UI (reference implementation) has received **6 major enhancements** that fundamentally expand the application's capabilities. These enhancements must be systematically incorporated into all 22 project documents to ensure the documentation accurately reflects the production requirements.

**Impact Scope:**
- 1 Architecture Decision Record (ADR)
- 6 Product Requirements Documents (PRDs)
- 5 Technical Design Documents (TDDs)
- 5 Technical Implementation Documents (TIDs)
- 5 Task Lists

**Estimated Effort:** 15-20 hours of focused documentation work

---

## Enhancement Inventory

### Enhancement 1: Training Template Management System

**Mock UI References:**
- Interface: Lines 597-609 (`TrainingTemplate`)
- State Management: Line 705
- Functions: Lines 1068-1150 (save, load, delete, favorite, export, import)
- UI Components: Lines 2285-2455 (collapsible section in TrainingPanel)

**Key Features:**
- Save/load/delete training hyperparameter configurations
- Export/import as JSON files (all three template types together)
- Auto-generated names: `{encoder}_{expansion}x_{steps}steps_{HHMM}`
- Favorite templates with star icon
- Collapsible "Saved Templates" section

**Database Schema Required:**
```sql
CREATE TABLE training_templates (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_id VARCHAR(255) REFERENCES models(id),  -- NULL = works with any model
    dataset_id VARCHAR(255) REFERENCES datasets(id),  -- NULL = works with any dataset
    encoder_type VARCHAR(50) NOT NULL,  -- 'sparse', 'skip', 'transcoder'
    hyperparameters JSONB NOT NULL,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**API Endpoints Required:**
- `GET /api/templates/training` - List all training templates
- `POST /api/templates/training` - Create new template
- `PUT /api/templates/training/:id` - Update template
- `DELETE /api/templates/training/:id` - Delete template
- `POST /api/templates/export` - Export all templates (training, extraction, steering)
- `POST /api/templates/import` - Import templates from JSON

**Documents Affected:**
- PRD: 003_FPRD|SAE_Training.md (add functional requirements)
- TDD: 003_FTDD|SAE_Training.md (add database schema, API design)
- TID: 003_FTID|SAE_Training.md (add implementation patterns)
- Tasks: 003_FTASKS|SAE_Training.md (add new parent tasks and sub-tasks)
- ADR: Add template management to data design section

---

### Enhancement 2: Extraction Template Management System

**Mock UI References:**
- Interface: Lines 611-621 (`ExtractionTemplate`)
- State Management: Line 706
- Functions: Lines 1152-1165 (save, delete, favorite)
- UI Components: Lines 1847-1971 (in ActivationExtractionConfig modal)

**Key Features:**
- Save/load/delete extraction configurations
- Export/import as JSON files
- Auto-generated names: `{type}_layers{min}-{max}_{samples}samples_{HHMM}`
- Favorite templates with star icon
- Collapsible "Saved Templates" section

**Database Schema Required:**
```sql
CREATE TABLE extraction_templates (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    layers INTEGER[] NOT NULL,  -- Array of layer indices
    hook_types VARCHAR(50)[] NOT NULL,  -- ['residual', 'mlp', 'attention']
    max_samples INTEGER,  -- NULL = no limit
    top_k_examples INTEGER NOT NULL DEFAULT 100,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**API Endpoints Required:**
- `GET /api/templates/extraction` - List extraction templates
- `POST /api/templates/extraction` - Create template
- `PUT /api/templates/extraction/:id` - Update template
- `DELETE /api/templates/extraction/:id` - Delete template

**Documents Affected:**
- PRD: 002_FPRD|Model_Management.md (add functional requirements for templates)
- TDD: 002_FTDD|Model_Management.md (add database schema, API design)
- TID: 002_FTID|Model_Management.md (add implementation patterns)
- Tasks: 002_FTASKS|Model_Management.md (add new parent tasks and sub-tasks)

---

### Enhancement 3: Steering Preset Management System

**Mock UI References:**
- Interface: Lines 666-677 (`SteeringPreset`)
- State Management: Line 707
- Functions: Lines 1181-1204 (save, delete, favorite)
- UI Components: Lines 4277-4445 (in SteeringPanel)

**Key Features:**
- Save/load/delete steering configurations (features + coefficients + layers + training_id)
- Export/import as JSON files
- Auto-generated names:
  - Single layer: `steering_{count}features_layer{N}_{HHMM}`
  - Multi-layer: `steering_{count}features_layers{min}-{max}_{HHMM}`
- Favorite presets with star icon
- Collapsible "Saved Presets" section
- Includes training_id reference (which training's features)

**Database Schema Required:**
```sql
CREATE TABLE steering_presets (
    id VARCHAR(255) PRIMARY KEY,
    training_id VARCHAR(255) NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    features JSONB NOT NULL,  -- [{"feature_id": 42, "coefficient": 2.0}, ...]
    intervention_layers INTEGER[] NOT NULL,  -- Array of layer indices
    temperature FLOAT NOT NULL DEFAULT 1.0,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**API Endpoints Required:**
- `GET /api/steering/presets?trainingId=:id` - List presets for training
- `POST /api/steering/presets` - Create preset
- `PUT /api/steering/presets/:id` - Update preset
- `DELETE /api/steering/presets/:id` - Delete preset
- `POST /api/steering/presets/:id/apply` - Apply preset to steering panel

**Documents Affected:**
- PRD: 005_FPRD|Model_Steering.md (add functional requirements)
- TDD: 005_FTDD|Model_Steering.md (add database schema, API design)
- TID: 005_FTID|Model_Steering.md (add implementation patterns)
- Tasks: 005_FTASKS|Model_Steering.md (add new parent tasks and sub-tasks)

---

### Enhancement 4: Multi-Layer Training Support

**Mock UI References:**
- Model Architecture Addition: Lines 354-358 (added `architecture` property to Model interface)
- Updated Model Data: Lines 773-801 (all models have architecture metadata)
- Hyperparameters Change: Line 377 (changed `trainingLayer: number` to `trainingLayers: number[]`)
- Default Config: Line 697 (changed from `trainingLayer: 6` to `trainingLayers: [6]`)
- Template Updates: Lines 820, 842, 864 (all templates use arrays)
- UI Component: Lines 2175-2236 (multi-layer checkbox grid, 8 columns)

**Key Changes:**
- **FROM:** Single layer selection (`trainingLayer: number`)
- **TO:** Multiple layer selection (`trainingLayers: number[]`)
- Enables training SAEs on multiple transformer layers simultaneously
- Dynamic layer selection based on model architecture metadata
- Visual grid of checkboxes (8 columns) with Select All/Clear All buttons
- Shows "Training Layers (N selected)" label

**Implementation Impact:**
1. **Backend Training Logic:**
   - Accept `trainingLayers: number[]` in training configuration
   - Modify SAE training loop to handle multiple layers
   - Extract activations from all specified layers
   - Train separate SAE instances per layer OR single SAE across layers (TBD)

2. **Database Schema:**
   - Change `trainings.hyperparameters.trainingLayer` (JSONB field) to array
   - Ensure backward compatibility or migrate existing data

3. **Frontend UI:**
   - Replace dropdown with multi-select checkbox grid
   - Dynamic generation based on `model.architecture.num_layers`
   - Select All / Clear All functionality

**Documents Affected:**
- PRD: 003_FPRD|SAE_Training.md (update functional requirements FR-1, FR-2)
- TDD: 003_FTDD|SAE_Training.md (update hyperparameters schema, training loop design)
- TDD: 002_FTDD|Model_Management.md (add architecture metadata to models table)
- TID: 003_FTID|SAE_Training.md (update training loop implementation)
- TID: 002_FTID|Model_Management.md (add architecture metadata extraction)
- Tasks: 003_FTASKS|SAE_Training.md (update UI tasks for multi-layer selection)
- Tasks: 002_FTASKS|Model_Management.md (add architecture metadata tasks)
- ADR: Document decision on multi-layer training approach

---

### Enhancement 5: Training Job Selector in Steering Tab

**Mock UI References:**
- Props Update: Lines 4157-4158 (added `trainings` prop to SteeringPanel)
- State Management: Line 4181 (`selectedTraining` state)
- Filtered List: Line 4208 (`completedTrainings` filter)
- UI Component: Lines 4611-4638 (dropdown with descriptive format)
- Preset Integration: Line 4341 (saves `selectedTraining` with preset)
- Preset Loading: Line 4255 (restores `selectedTraining` when loading preset)

**Key Features:**
- Dropdown selector at top of Steering tab
- Shows only completed trainings (status = 'completed')
- Descriptive format: `{encoderType} SAE • {modelName} • {datasetName} • Started {date}`
- Determines which training's features are available for steering
- Saved with steering presets for consistency

**Implementation Impact:**
1. **Steering Panel Workflow:**
   - User must select a completed training first
   - Feature selection dropdown is populated from selected training's features
   - Training selection saved with presets

2. **API Changes:**
   - Steering presets include `training_id` field (already in schema)
   - `GET /api/features?trainingId=:id` filters features by training

3. **UI Display:**
   - Consistent training job format across Features and Steering tabs
   - Dynamic lookup of model and dataset names from IDs

**Documents Affected:**
- PRD: 005_FPRD|Model_Steering.md (add training selector requirement)
- TDD: 005_FTDD|Model_Steering.md (add training filtering logic)
- TID: 005_FTID|Model_Steering.md (add training selector implementation)
- Tasks: 005_FTASKS|Model_Steering.md (add training selector UI tasks)

---

### Enhancement 6: Multi-Layer Steering Support

**Mock UI References:**
- SteeringPreset Interface: Line 672 (changed `intervention_layer: number` to `intervention_layers: number[]`)
- SteeringConfig Interface: Line 583 (changed `interventionLayer: number` to `interventionLayers: number[]`)
- API Documentation: Line 573 (updated to use `interventionLayers: number[]`)
- State Management: Line 4186 (changed to `interventionLayers` with default `[12]`)
- Mock Data: Lines 915, 930 (updated presets with arrays)
- Preset Functions: Lines 1189, 4256 (save/load arrays)
- UI Component: Lines 4643-4704 (multi-layer checkbox grid, 8 columns)

**Key Changes:**
- **FROM:** Single intervention layer (`interventionLayer: number`)
- **TO:** Multiple intervention layers (`interventionLayers: number[]`)
- Enables steering across multiple transformer layers simultaneously
- Same UI pattern as training layer selection (8-column checkbox grid)
- Dynamic based on selected model's architecture

**Implementation Impact:**
1. **Backend Steering Logic:**
   - Accept `interventionLayers: number[]` in steering configuration
   - Apply forward hooks to multiple layers during generation
   - Coefficient application at each specified layer
   - Combined effect of multi-layer interventions

2. **Database Schema:**
   - Change `steering_presets.intervention_layer` (JSONB or column) to array
   - Update existing presets to use arrays

3. **Frontend UI:**
   - Replace single layer slider with multi-select checkbox grid
   - Dynamic generation based on model architecture
   - Select All / Clear All functionality
   - Display layer range in preset names

**Documents Affected:**
- PRD: 005_FPRD|Model_Steering.md (update functional requirements for multi-layer)
- TDD: 005_FTDD|Model_Steering.md (update steering algorithm, schema)
- TID: 005_FTID|Model_Steering.md (update hook registration for multiple layers)
- Tasks: 005_FTASKS|Model_Steering.md (update UI and backend tasks)
- ADR: Document decision on multi-layer steering approach

---

## Document Update Strategy

### Phase 1: Foundation Updates (ADR + Project PRD)

**Documents:**
1. `000_PADR|miStudio.md` - ADR
2. `000_PPRD|miStudio.md` - Project PRD

**Updates:**
- ADR: Add template/preset management to data design decisions
- ADR: Document multi-layer training/steering architecture decisions
- ADR: Add model architecture metadata to technology stack
- Project PRD: Update feature descriptions to reflect enhancements

**Dependencies:** None (foundation documents)

---

### Phase 2: Feature PRD Updates

**Documents:**
1. `002_FPRD|Model_Management.md` - Add extraction templates
2. `003_FPRD|SAE_Training.md` - Add training templates + multi-layer
3. `005_FPRD|Model_Steering.md` - Add presets + multi-layer + training selector

**Updates:**
- Add new functional requirements for each enhancement
- Update user stories to include template/preset workflows
- Add acceptance criteria for new features
- Update success metrics

**Dependencies:** ADR and Project PRD must be updated first

---

### Phase 3: Technical Design Document (TDD) Updates

**Documents:**
1. `002_FTDD|Model_Management.md` - Architecture metadata + extraction templates
2. `003_FTDD|SAE_Training.md` - Training templates + multi-layer training design
3. `005_FTDD|Model_Steering.md` - Presets + multi-layer steering + training selector

**Updates:**
- Add database schemas for new tables
- Design API endpoints for template/preset management
- Update existing schemas for array fields (layers)
- Design multi-layer training/steering algorithms
- Update component architecture

**Dependencies:** Feature PRDs must be updated first

---

### Phase 4: Technical Implementation Document (TID) Updates

**Documents:**
1. `002_FTID|Model_Management.md` - Extraction template implementation
2. `003_FTID|SAE_Training.md` - Training template + multi-layer implementation
3. `005_FTID|Model_Steering.md` - Preset + multi-layer steering implementation

**Updates:**
- Add implementation code snippets for new features
- Update existing implementations for array handling
- Add UI component implementations (checkbox grids, collapsible sections)
- Add template/preset management service code
- Add auto-naming algorithms

**Dependencies:** TDDs must be updated first

---

### Phase 5: Task List Updates

**Documents:**
1. `002_FTASKS|Model_Management.md` - Add extraction template tasks
2. `003_FTASKS|SAE_Training.md` - Add training template + multi-layer tasks
3. `005_FTASKS|Model_Steering.md` - Add preset + multi-layer + selector tasks

**Updates:**
- Add new parent tasks for each enhancement
- Break down into detailed sub-tasks
- Reference exact Mock UI line numbers
- Add testing requirements for new features
- Update file lists with new components

**Dependencies:** TIDs must be updated first

---

## Detailed Task Breakdown by Enhancement

### Enhancement 1: Training Template Management

**New Database Tasks:**
- Create migration for `training_templates` table
- Add indexes (by user, by favorite, by created_at)
- Seed with example templates

**New Backend Tasks:**
- Create `TrainingTemplate` Pydantic schema
- Create `TemplateService` with CRUD operations
- Create template API routes (GET, POST, PUT, DELETE)
- Add export endpoint (combine all three types)
- Add import endpoint with validation
- Write unit tests for template service
- Write integration tests for template API

**New Frontend Tasks:**
- Create `trainingTemplatesStore` in Zustand
- Create template management UI in TrainingPanel (lines 2285-2455)
- Add collapsible "Saved Templates" section
- Implement save dialog with auto-generated name
- Implement load/delete/favorite actions
- Add export/import buttons with file handling
- Style template cards with favorite stars
- Write component tests

**Estimated Sub-tasks:** 25-30

---

### Enhancement 2: Extraction Template Management

**New Database Tasks:**
- Create migration for `extraction_templates` table
- Add indexes

**New Backend Tasks:**
- Create `ExtractionTemplate` Pydantic schema
- Create extraction template API routes
- Add to export/import endpoints
- Write unit tests
- Write integration tests

**New Frontend Tasks:**
- Create `extractionTemplatesStore` in Zustand
- Add template UI to ActivationExtractionConfig modal (lines 1847-1971)
- Implement save/load/delete/favorite
- Add auto-naming with layer range + samples
- Write component tests

**Estimated Sub-tasks:** 20-25

---

### Enhancement 3: Steering Preset Management

**New Database Tasks:**
- Create migration for `steering_presets` table (may already exist partially)
- Update schema if needed (ensure training_id, intervention_layers array)
- Add indexes

**New Backend Tasks:**
- Create/update `SteeringPreset` Pydantic schema
- Create preset API routes
- Add to export/import endpoints
- Write unit tests
- Write integration tests

**New Frontend Tasks:**
- Create `steeringPresetsStore` in Zustand
- Add preset UI to SteeringPanel (lines 4277-4445)
- Implement save/load/delete/favorite
- Add auto-naming with feature count + layer range
- Include training_id in presets
- Write component tests

**Estimated Sub-tasks:** 25-30

---

### Enhancement 4: Multi-Layer Training Support

**Database Migration Tasks:**
- Add `architecture` JSONB column to `models` table
- Migrate `trainings.hyperparameters.trainingLayer` to array
- Update existing training records

**Backend Tasks:**
- Update model loading to extract architecture metadata
- Update training configuration schema (array validation)
- Modify SAE training loop for multi-layer support
- Update checkpoint format if needed
- Write unit tests for multi-layer training
- Write integration tests

**Frontend Tasks:**
- Update TrainingPanel UI (replace dropdown with checkbox grid, lines 2175-2236)
- Add Select All / Clear All buttons
- Dynamic grid based on model.architecture.num_layers
- Update training config state to use array
- Update all template management to handle arrays
- Write component tests

**Estimated Sub-tasks:** 30-35

---

### Enhancement 5: Training Job Selector in Steering

**Backend Tasks:**
- Ensure training filtering endpoint exists (`GET /api/trainings?status=completed`)
- Update steering preset schema to include training_id

**Frontend Tasks:**
- Add training selector dropdown to SteeringPanel (lines 4611-4638)
- Filter trainings by status = 'completed'
- Display with descriptive format
- Connect to feature selection (filter by training)
- Save training_id with presets
- Restore training selection when loading preset
- Write component tests

**Estimated Sub-tasks:** 10-12

---

### Enhancement 6: Multi-Layer Steering Support

**Database Migration Tasks:**
- Migrate `steering_presets.intervention_layer` to array
- Update existing presets

**Backend Tasks:**
- Update steering configuration schema (array validation)
- Modify forward hook registration for multiple layers
- Update steering algorithm documentation
- Write unit tests for multi-layer steering
- Write integration tests

**Frontend Tasks:**
- Add multi-layer checkbox grid to SteeringPanel (lines 4643-4704)
- Add Select All / Clear All buttons
- Dynamic grid based on model architecture
- Update steering state to use array
- Update preset auto-naming for layer ranges
- Write component tests

**Estimated Sub-tasks:** 25-30

---

## Total New Tasks Summary

| Enhancement | Database | Backend | Frontend | Testing | Total Est. |
|-------------|----------|---------|----------|---------|-----------|
| Training Templates | 3 | 10 | 12 | 5 | 30 |
| Extraction Templates | 2 | 8 | 10 | 5 | 25 |
| Steering Presets | 3 | 10 | 12 | 5 | 30 |
| Multi-Layer Training | 3 | 12 | 15 | 5 | 35 |
| Training Selector | 0 | 2 | 8 | 2 | 12 |
| Multi-Layer Steering | 2 | 10 | 13 | 5 | 30 |
| **TOTALS** | **13** | **52** | **70** | **27** | **162** |

**Total New Sub-tasks Across All Features:** ~162 sub-tasks

---

## Document Update Checklist

### ADR (000_PADR|miStudio.md)

- [ ] Add template/preset management to data design section
- [ ] Document multi-layer training architecture decision
- [ ] Document multi-layer steering architecture decision
- [ ] Add model architecture metadata to technology stack
- [ ] Update database schema overview with new tables
- [ ] Add export/import functionality to API design principles

### Project PRD (000_PPRD|miStudio.md)

- [ ] Update SAE Training feature description (templates + multi-layer)
- [ ] Update Model Management feature description (extraction templates)
- [ ] Update Model Steering feature description (presets + multi-layer + selector)
- [ ] Add template/preset management to success metrics

### Model Management PRD (002_FPRD|Model_Management.md)

- [ ] Add FR-X.X: Extraction template CRUD operations
- [ ] Add FR-X.X: Template save with auto-generated names
- [ ] Add FR-X.X: Template favorite functionality
- [ ] Add FR-X.X: Template export/import
- [ ] Add FR-X.X: Model architecture metadata extraction
- [ ] Update user stories for template workflows
- [ ] Add acceptance criteria for templates

### SAE Training PRD (003_FPRD|SAE_Training.md)

- [ ] Add FR-X.X: Training template CRUD operations
- [ ] Add FR-X.X: Template save with auto-generated names
- [ ] Add FR-X.X: Template favorite functionality
- [ ] Add FR-X.X: Template export/import
- [ ] Update FR-1.X: Change single layer to multi-layer selection
- [ ] Update FR-2.X: Multi-layer training execution
- [ ] Update user stories for template workflows
- [ ] Update user stories for multi-layer training
- [ ] Add acceptance criteria for templates and multi-layer

### Model Steering PRD (005_FPRD|Model_Steering.md)

- [ ] Add FR-X.X: Steering preset CRUD operations
- [ ] Add FR-X.X: Preset save with auto-generated names
- [ ] Add FR-X.X: Preset favorite functionality
- [ ] Add FR-X.X: Preset export/import
- [ ] Add FR-X.X: Training job selector requirement
- [ ] Update FR-X.X: Change single layer to multi-layer intervention
- [ ] Update FR-X.X: Multi-layer steering execution
- [ ] Update user stories for preset workflows
- [ ] Update user stories for training selector
- [ ] Update user stories for multi-layer steering
- [ ] Add acceptance criteria for all new features

### Model Management TDD (002_FTDD|Model_Management.md)

- [ ] Add `extraction_templates` table schema
- [ ] Add `models.architecture` JSONB field
- [ ] Add API endpoints for extraction templates
- [ ] Add export/import endpoints
- [ ] Update component architecture for templates
- [ ] Add validation rules for template fields

### SAE Training TDD (003_FTDD|SAE_Training.md)

- [ ] Add `training_templates` table schema
- [ ] Update `trainings.hyperparameters` schema (trainingLayers array)
- [ ] Add API endpoints for training templates
- [ ] Update training algorithm for multi-layer support
- [ ] Update component architecture for templates
- [ ] Add validation rules for template fields and layer arrays

### Model Steering TDD (005_FTDD|Model_Steering.md)

- [ ] Add/update `steering_presets` table schema
- [ ] Update steering configuration schema (interventionLayers array)
- [ ] Add API endpoints for steering presets
- [ ] Update steering algorithm for multi-layer hooks
- [ ] Update component architecture for presets and selector
- [ ] Add validation rules for preset fields and layer arrays

### Model Management TID (002_FTID|Model_Management.md)

- [ ] Add extraction template service implementation
- [ ] Add template API route implementation
- [ ] Add template UI component implementation
- [ ] Add auto-naming algorithm for extraction templates
- [ ] Add model architecture metadata extraction code
- [ ] Add export/import file handling

### SAE Training TID (003_FTID|SAE_Training.md)

- [ ] Add training template service implementation
- [ ] Add template API route implementation
- [ ] Add template UI component implementation (lines 2285-2455)
- [ ] Add auto-naming algorithm for training templates
- [ ] Add multi-layer training loop implementation
- [ ] Add multi-layer selection UI (lines 2175-2236)
- [ ] Update hyperparameters handling for arrays

### Model Steering TID (005_FTID|Model_Steering.md)

- [ ] Add steering preset service implementation
- [ ] Add preset API route implementation
- [ ] Add preset UI component implementation (lines 4277-4445)
- [ ] Add auto-naming algorithm for steering presets
- [ ] Add training selector UI implementation (lines 4611-4638)
- [ ] Add multi-layer steering hook implementation
- [ ] Add multi-layer selection UI (lines 4643-4704)
- [ ] Update steering configuration handling for arrays

### Model Management Tasks (002_FTASKS|Model_Management.md)

- [ ] Add parent task: Extraction Template Database Schema
- [ ] Add parent task: Extraction Template Backend Service
- [ ] Add parent task: Extraction Template API Routes
- [ ] Add parent task: Extraction Template Frontend UI
- [ ] Add parent task: Extraction Template Testing
- [ ] Add parent task: Model Architecture Metadata
- [ ] Add ~25 new sub-tasks with Mock UI line references
- [ ] Update relevant files list

### SAE Training Tasks (003_FTASKS|SAE_Training.md)

- [ ] Add parent task: Training Template Database Schema
- [ ] Add parent task: Training Template Backend Service
- [ ] Add parent task: Training Template API Routes
- [ ] Add parent task: Training Template Frontend UI
- [ ] Add parent task: Training Template Testing
- [ ] Add parent task: Multi-Layer Training Backend Implementation
- [ ] Add parent task: Multi-Layer Selection UI
- [ ] Add parent task: Hyperparameters Migration for Arrays
- [ ] Add ~30-35 new sub-tasks with Mock UI line references
- [ ] Update relevant files list

### Model Steering Tasks (005_FTASKS|Model_Steering.md)

- [ ] Add parent task: Steering Preset Database Schema
- [ ] Add parent task: Steering Preset Backend Service
- [ ] Add parent task: Steering Preset API Routes
- [ ] Add parent task: Steering Preset Frontend UI
- [ ] Add parent task: Steering Preset Testing
- [ ] Add parent task: Training Job Selector UI
- [ ] Add parent task: Multi-Layer Steering Backend Implementation
- [ ] Add parent task: Multi-Layer Selection UI
- [ ] Add parent task: Steering Configuration Migration for Arrays
- [ ] Add ~30 new sub-tasks with Mock UI line references
- [ ] Update relevant files list

---

## Implementation Priority

### Critical Path Items (Implement First):
1. **Model Architecture Metadata** - Required for dynamic UI generation
2. **Multi-Layer Training Support** - Fundamental capability expansion
3. **Training Template Management** - High user value
4. **Multi-Layer Steering Support** - Fundamental capability expansion

### High Priority (Implement Second):
5. **Training Job Selector** - UX improvement for steering
6. **Steering Preset Management** - High user value

### Standard Priority (Implement Third):
7. **Extraction Template Management** - Nice-to-have for power users

---

## Risk Assessment

### Low Risk:
- Template/preset management (additive features, no breaking changes)
- Training job selector (additive feature)

### Medium Risk:
- Multi-layer training (requires training loop refactoring)
- Multi-layer steering (requires hook system refactoring)
- Database migrations for array fields (need backward compatibility)

### High Risk:
- Model architecture metadata extraction (external dependency on model files)
- Multi-layer training performance (GPU memory constraints on Jetson)

---

## Validation Strategy

After all updates, validate by:

1. **Cross-Document Consistency Check:**
   - Verify all PRDs reference the same features
   - Verify all TDDs implement all PRD requirements
   - Verify all TIDs provide implementation for all TDD designs
   - Verify all Tasks cover all TID implementations

2. **Mock UI Reference Check:**
   - Every UI component in docs matches exact Mock UI lines
   - All line references are accurate (re-verify after updates)
   - All styling matches Mock UI specifications

3. **Technical Feasibility Check:**
   - Database schemas support all features
   - API endpoints provide all required operations
   - Implementation guidance is complete and actionable

4. **Completeness Check:**
   - All 6 enhancements addressed in all relevant documents
   - No orphaned references to old single-layer approach
   - All new sub-tasks estimated and sized

---

## Next Steps

1. **Review this plan with stakeholder** ✅ (Current Step)
2. Get approval to proceed
3. Begin Phase 1: Update ADR and Project PRD
4. Proceed through phases 2-5 systematically
5. Perform validation checks after each phase
6. Create final change summary document

---

## Estimated Timeline

- **Phase 1** (ADR + Project PRD): 2-3 hours
- **Phase 2** (3 Feature PRDs): 4-5 hours
- **Phase 3** (3 TDDs): 4-5 hours
- **Phase 4** (3 TIDs): 4-5 hours
- **Phase 5** (3 Task Lists): 3-4 hours
- **Validation**: 1-2 hours

**Total: 18-24 hours of focused work**

---

## Questions for Stakeholder

1. **Architecture Decision for Multi-Layer Training:**
   - Should we train separate SAE instances per layer?
   - OR train a single larger SAE across multiple layers?
   - Implications for feature discovery and steering

2. **Backward Compatibility:**
   - How should we handle existing trainings/presets with single layer?
   - Auto-migrate to arrays `[layer]` OR maintain dual format?

3. **Template/Preset Storage:**
   - Store in database (as planned) OR local file system?
   - Export/import format finalized (combined JSON)?

4. **Priority Adjustment:**
   - Is the suggested priority order acceptable?
   - Any enhancements needed sooner/later?

---

**Document End**
**Status:** Awaiting Stakeholder Approval
**Next Action:** Stakeholder review and Q&A
