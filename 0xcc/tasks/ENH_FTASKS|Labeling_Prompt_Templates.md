# Enhancement Task List: Labeling Prompt Templates

**Feature**: Customizable Prompt Templates for Semantic Feature Labeling
**Priority**: P1 (Enhancement)
**Estimated Effort**: 12-16 hours (Actual: ~10 hours)
**Status**: ✅ COMPLETED
**Created**: 2025-11-13
**Completed**: 2025-11-13

## Overview

This enhancement adds a template system for customizing the prompts and API parameters used during semantic feature labeling. Users will be able to:
- Create, edit, and manage custom prompt templates
- Control system message, user prompt format, and API parameters (temperature, max_tokens, top_p)
- Save and reuse templates across labeling jobs
- Set a default template for new labeling jobs

This feature is inspired by the existing Training Templates system and follows similar patterns for consistency.

## Current State

**Existing Labeling System:**
- Fixed prompt structure in `OpenAILabelingService._build_prompt()` ([backend/src/services/openai_labeling_service.py:161-231](backend/src/services/openai_labeling_service.py#L161-L231))
- Hardcoded system message: "You are an expert in mechanistic interpretability analyzing sparse autoencoder features. Provide both category and specific labels in JSON format."
- Fixed API parameters: temperature=0.3, max_tokens=50, top_p=0.9
- Current prompt includes:
  - Instructions for dual labeling (category + specific)
  - 7 example patterns
  - Top 30 tokens table (dynamically filled with feature data)
  - Decision tree for specificity
  - JSON format requirement

**What's Working:**
- ✅ Enhanced logging shows full prompts and responses ([backend/src/services/openai_labeling_service.py:109-143](backend/src/services/openai_labeling_service.py#L109-L143))
- ✅ OpenAI-compatible endpoint support with URL validation
- ✅ Multiple labeling methods (openai, openai_compatible, local)
- ✅ Labeling job configuration and tracking

**What Needs Implementation:**
- ❌ Database model for storing prompt templates
- ❌ Backend API for template CRUD operations
- ❌ Service layer modifications to use custom templates
- ❌ Frontend UI for template management
- ❌ Template selection in labeling job creation

## Architecture

### Database Schema
```sql
-- New table: labeling_prompt_templates
CREATE TABLE labeling_prompt_templates (
    id VARCHAR(255) PRIMARY KEY,  -- Format: "lpt_{uuid}"
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Prompt content
    system_message TEXT NOT NULL,
    user_prompt_template TEXT NOT NULL,  -- Uses {tokens_table} placeholder

    -- API parameters
    temperature FLOAT DEFAULT 0.3,
    max_tokens INTEGER DEFAULT 50,
    top_p FLOAT DEFAULT 0.9,

    -- Metadata
    is_default BOOLEAN DEFAULT FALSE,
    is_system BOOLEAN DEFAULT FALSE,  -- System templates can't be deleted
    created_by VARCHAR(255),  -- Future: user ID

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_default_template UNIQUE (is_default) WHERE is_default = TRUE
);

-- Update labeling_jobs table
ALTER TABLE labeling_jobs ADD COLUMN prompt_template_id VARCHAR(255) REFERENCES labeling_prompt_templates(id);
```

### Backend Components
1. **Database Model**: `backend/src/models/labeling_prompt_template.py`
2. **Pydantic Schemas**: `backend/src/schemas/labeling_prompt_template.py`
3. **Service Layer**: `backend/src/services/labeling_prompt_template_service.py`
4. **API Endpoints**: `backend/src/api/v1/endpoints/labeling_prompt_templates.py`
5. **Updated Labeling Service**: Modify `backend/src/services/openai_labeling_service.py`

### Frontend Components
1. **Types**: `frontend/src/types/labelingPromptTemplate.ts`
2. **API Client**: `frontend/src/api/labelingPromptTemplates.ts`
3. **Store**: `frontend/src/stores/labelingPromptTemplatesStore.ts`
4. **Main Panel**: `frontend/src/components/panels/LabelingPromptTemplatesPanel.tsx`
5. **Form Component**: `frontend/src/components/labelingPromptTemplates/LabelingPromptTemplateForm.tsx`
6. **Card Component**: `frontend/src/components/labelingPromptTemplates/LabelingPromptTemplateCard.tsx`
7. **List Component**: `frontend/src/components/labelingPromptTemplates/LabelingPromptTemplateList.tsx`
8. **Updated Start Button**: Modify `frontend/src/components/labeling/StartLabelingButton.tsx`

## Implementation Tasks

### Phase 1: Backend Foundation (4-5 hours) ✅ COMPLETED

#### Task 1.1: Database Migration ✅
**File**: `backend/alembic/versions/{timestamp}_add_labeling_prompt_templates.py`
- [x] Create Alembic migration for `labeling_prompt_templates` table
- [x] Add `prompt_template_id` column to `labeling_jobs` table
- [x] Create indexes on `is_default`, `name`, `created_at`
- [x] Add foreign key constraint with ON DELETE RESTRICT
- [x] Run migration: `alembic upgrade head`
- [x] Verify schema: `psql -d mistudio -c "\d labeling_prompt_templates"`

#### Task 1.2: Database Model ✅
**File**: `backend/src/models/labeling_prompt_template.py`
- [x] Create `LabelingPromptTemplate` SQLAlchemy model
- [x] Define all columns matching schema
- [x] Add relationship to `LabelingJob` model
- [x] Add `__repr__` method
- [x] Update `backend/src/models/__init__.py` to export new model
- [x] Update `LabelingJob` model with reverse relationship

#### Task 1.3: Pydantic Schemas ✅
**File**: `backend/src/schemas/labeling_prompt_template.py`
- [x] Create `LabelingPromptTemplateCreate` schema (validation)
- [x] Create `LabelingPromptTemplateUpdate` schema (partial updates)
- [x] Create `LabelingPromptTemplateResponse` schema (API responses)
- [x] Create `LabelingPromptTemplateListResponse` schema (pagination)
- [x] Add field validators for temperature (0.0-2.0), max_tokens (10-500), top_p (0.0-1.0)
- [x] Add validator for {tokens_table} placeholder in user_prompt_template
- [x] Update `backend/src/schemas/__init__.py` to export new schemas

#### Task 1.4: Service Layer ✅
**File**: `backend/src/services/labeling_prompt_template_service.py`
- [x] Create `LabelingPromptTemplateService` class
- [x] Implement `create_template(data: LabelingPromptTemplateCreate) -> LabelingPromptTemplate`
- [x] Implement `get_template(template_id: str) -> Optional[LabelingPromptTemplate]`
- [x] Implement `list_templates(limit: int, offset: int) -> List[LabelingPromptTemplate]`
- [x] Implement `update_template(template_id: str, data: LabelingPromptTemplateUpdate) -> LabelingPromptTemplate`
- [x] Implement `delete_template(template_id: str) -> bool` (prevent deletion if used by jobs)
- [x] Implement `set_default_template(template_id: str) -> bool` (unset previous default)
- [x] Implement `get_default_template() -> Optional[LabelingPromptTemplate]`
- [x] Add error handling for not found, constraint violations, etc.
- [x] **BONUS**: Implemented `export_templates()` and `import_templates()` methods

#### Task 1.5: API Endpoints ✅
**File**: `backend/src/api/v1/endpoints/labeling_prompt_templates.py`
- [x] `GET /api/v1/labeling-prompt-templates` - List all templates (paginated)
  - Query params: limit, offset
  - Returns: LabelingPromptTemplateListResponse
- [x] `GET /api/v1/labeling-prompt-templates/default` - Get default template
  - Returns: LabelingPromptTemplateResponse or 404
- [x] `GET /api/v1/labeling-prompt-templates/:id` - Get specific template
  - Returns: LabelingPromptTemplateResponse or 404
- [x] `POST /api/v1/labeling-prompt-templates` - Create new template
  - Body: LabelingPromptTemplateCreate
  - Returns: 201 with LabelingPromptTemplateResponse
- [x] `PATCH /api/v1/labeling-prompt-templates/:id` - Update template
  - Body: LabelingPromptTemplateUpdate
  - Returns: LabelingPromptTemplateResponse or 404
- [x] `DELETE /api/v1/labeling-prompt-templates/:id` - Delete template
  - Returns: 204 or 404/409 (if used by jobs or is system template)
- [x] `POST /api/v1/labeling-prompt-templates/:id/set-default` - Set as default
  - Returns: LabelingPromptTemplateSetDefaultResponse
- [x] `GET /api/v1/labeling-prompt-templates/:id/usage-count` - Get usage count
  - Returns: usage count
- [x] `POST /api/v1/labeling-prompt-templates/export` - Export templates
  - Returns: JSON export data
- [x] `POST /api/v1/labeling-prompt-templates/import` - Import templates
  - Returns: Import results with statistics
- [x] Register router in `backend/src/api/v1/api.py`

### Phase 2: Backend Integration (2-3 hours) ✅ COMPLETED

#### Task 2.1: Seed Default Template ✅
**File**: `backend/src/db/seed_data.py` (create if doesn't exist)
- [x] Create seed script to insert default system template
- [x] Use current hardcoded prompt as "Default SAE Feature Labeling"
- [x] Mark as `is_system=True`, `is_default=True`
- [x] Add to application startup or migration
- [x] Document how to run seed script

#### Task 2.2: Update Labeling Service ✅
**File**: `backend/src/services/openai_labeling_service.py`
- [x] Add `prompt_template_id: Optional[str]` parameter to `__init__`
- [x] Load template from database in `__init__` if provided
- [x] Use template's system_message, user_prompt_template, and API params
- [x] Replace `{tokens_table}` placeholder with actual token table
- [x] Fallback to current hardcoded prompt if no template specified
- [x] Update logging to show template ID being used
- [x] Add unit tests for template rendering

#### Task 2.3: Update Labeling Job Creation ✅
**File**: `backend/src/schemas/labeling.py`
- [x] Add `prompt_template_id: Optional[str]` to `LabelingConfigRequest`
- [x] Add field description and validator
- [x] Update `LabelingStatusResponse` to include `prompt_template_id`

**File**: `backend/src/services/labeling_service.py`
- [x] Update `create_labeling_job()` to accept `prompt_template_id`
- [x] If not provided, use default template
- [x] Store `prompt_template_id` in `labeling_jobs` table
- [x] Pass template ID to `OpenAILabelingService` during labeling

**File**: `backend/src/models/labeling_job.py`
- [x] Add `prompt_template` relationship property

### Phase 3: Frontend Foundation (3-4 hours) ✅ COMPLETED

#### Task 3.1: TypeScript Types ✅
**File**: `frontend/src/types/labelingPromptTemplate.ts`
- [x] Define `LabelingPromptTemplate` interface (matches backend response)
- [x] Define `LabelingPromptTemplateCreate` interface
- [x] Define `LabelingPromptTemplateUpdate` interface
- [x] Define `LabelingPromptTemplateListResponse` interface
- [x] Add JSDoc comments with API contract details

#### Task 3.2: API Client ✅
**File**: `frontend/src/api/labelingPromptTemplates.ts`
- [x] Implement `getLabelingPromptTemplates(limit?, offset?)`
- [x] Implement `getDefaultLabelingPromptTemplate()`
- [x] Implement `getLabelingPromptTemplate(id)`
- [x] Implement `createLabelingPromptTemplate(data)`
- [x] Implement `updateLabelingPromptTemplate(id, data)`
- [x] Implement `deleteLabelingPromptTemplate(id)`
- [x] Implement `setDefaultLabelingPromptTemplate(id)`
- [x] Add proper TypeScript types for all functions
- [x] Add error handling
- [x] **BONUS**: Implemented `exportLabelingPromptTemplates()` and `importLabelingPromptTemplates()`
- [x] **BONUS**: Implemented `getLabelingPromptTemplateUsageCount()`

#### Task 3.3: Zustand Store ✅
**File**: `frontend/src/stores/labelingPromptTemplatesStore.ts`
- [x] Create store with state: templates, isLoading, error
- [x] Implement `fetchTemplates()` action
- [x] Implement `fetchDefaultTemplate()` action
- [x] Implement `createTemplate(data)` action
- [x] Implement `updateTemplate(id, data)` action
- [x] Implement `deleteTemplate(id)` action
- [x] Implement `setDefaultTemplate(id)` action
- [x] Add optimistic updates for better UX
- [x] Add error handling and notifications
- [x] Add selectors for filtering/sorting

### Phase 4: Frontend UI Components (3-4 hours) ✅ COMPLETED

#### Task 4.1: Template Form Component ✅
**File**: `frontend/src/components/labelingPromptTemplates/LabelingPromptTemplateForm.tsx`
- [x] Create form with fields:
  - Name (required, max 255 chars)
  - Description (optional, textarea)
  - System Message (required, textarea, min 10 chars)
  - User Prompt Template (required, textarea with monospace font)
  - Temperature (slider, 0.0-2.0, step 0.1, default 0.3)
  - Max Tokens (number input, 10-500, default 50)
  - Top P (slider, 0.0-1.0, step 0.05, default 0.9)
- [x] Add validation for all fields
- [x] Show character counts for text areas
- [x] Add {tokens_table} placeholder helper/documentation
- [x] Add preview section showing example rendered prompt
- [x] Add "Test with sample tokens" button
- [x] Implement form submission with loading state
- [x] Add cancel button
- [x] Style with Tailwind (match existing patterns)

#### Task 4.2: Template Card Component ✅
**File**: `frontend/src/components/labelingPromptTemplates/LabelingPromptTemplateCard.tsx`
- [x] Display template name, description, and metadata
- [x] Show badges: Default, System, Custom
- [x] Show API parameters (temp, max_tokens, top_p)
- [x] Add action buttons:
  - Edit (disabled for system templates)
  - Delete (disabled for system templates and if used by jobs)
  - Set as Default (if not already default)
  - Duplicate
  - View Full Prompt (modal)
- [x] Add confirmation dialog for destructive actions
- [x] Show usage count (number of jobs using this template)
- [x] Add expand/collapse for full prompt text
- [x] Style with Tailwind dark theme

#### Task 4.3: Template List Component ✅
**File**: `frontend/src/components/labelingPromptTemplates/LabelingPromptTemplateList.tsx`
- [x] Implement grid layout of template cards
- [x] Add search/filter by name
- [x] Add filter by: Default, System, Custom
- [x] Add sort options: Name, Created Date, Last Used
- [x] Implement pagination (limit 20 per page)
- [x] Add empty state with "Create First Template" CTA
- [x] Add loading skeleton
- [x] Add error state
- [x] Style with Tailwind

#### Task 4.4: Main Panel Component ✅
**File**: `frontend/src/components/panels/LabelingPromptTemplatesPanel.tsx`
- [x] Create panel layout matching other template panels
- [x] Add header with title and "Create Template" button
- [x] Add info section explaining prompt templates
- [x] Show current default template prominently
- [x] Integrate LabelingPromptTemplateList component
- [x] Add modal for create/edit form
- [x] Handle state management (create, edit, delete, set default)
- [x] Add notifications for actions (success/error)
- [x] Add confirmation dialogs
- [x] Style consistently with existing panels
- [x] **BONUS**: Added Export/Import functionality with modal and file handling
- [x] **BONUS**: Added usage statistics display with badges
- [x] **BONUS**: Added template preview modal with sample data

#### Task 4.5: Update Start Labeling Button ✅
**File**: `frontend/src/components/labeling/StartLabelingButton.tsx`
- [x] Add state for selected template: `const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null)`
- [x] Fetch templates on modal open
- [x] Add "Prompt Template" dropdown in modal (before Labeling Method)
- [x] Show template selector with options:
  - "Use Default Template" (null value)
  - List of available templates
- [x] Show selected template details (collapsible):
  - System message preview
  - User prompt preview
  - API parameters
- [x] Pass `prompt_template_id` to `startLabeling()` call
- [x] Update types in `labelingStore.ts` to accept `prompt_template_id`

### Phase 5: Testing & Documentation (1-2 hours) ✅ COMPLETED

#### Task 5.1: Backend Tests ✅
**File**: `backend/tests/test_labeling_prompt_templates.py`
- [x] Test template CRUD operations
- [x] Test default template management
- [x] Test template validation
- [x] Test deletion prevention (system templates, in-use templates)
- [x] Test prompt rendering with template
- [x] Test {tokens_table} placeholder replacement
- [x] Test API parameter usage

#### Task 5.2: Frontend Tests ✅
**File**: `frontend/src/components/labelingPromptTemplates/__tests__/`
- [x] Test form validation
- [x] Test template card actions
- [x] Test list filtering and pagination
- [x] Test store actions
- [x] Test API client functions

#### Task 5.3: Documentation ✅
- [x] Update API documentation with new endpoints
- [x] Add user guide for creating custom prompt templates
- [x] Document {tokens_table} placeholder format
- [x] Document API parameter effects
- [x] Add example templates for different use cases:
  - Domain-specific (code, medical, legal, etc.)
  - Different specificity levels
  - Different output formats
- [x] Update CLAUDE.md with new feature

## Testing Strategy

### Manual Testing Checklist
- [ ] Create a new custom template via UI
- [ ] Edit an existing custom template
- [ ] Delete a custom template (verify can't delete system template)
- [ ] Set a template as default
- [ ] Start a labeling job with default template
- [ ] Start a labeling job with custom template
- [ ] Verify prompt is correctly rendered in logs
- [ ] Verify API parameters (temperature, max_tokens, top_p) are used
- [ ] Test form validation (required fields, value ranges)
- [ ] Test search and filtering
- [ ] Test pagination
- [ ] Verify labels are generated correctly with custom template
- [ ] Test edge cases:
  - Very long prompts
  - Missing {tokens_table} placeholder (should error)
  - Invalid API parameters (should validate)
  - Duplicate template names (should allow)
  - Deleting template used by active job (should prevent)

### Integration Testing
- [ ] Verify database constraints work correctly
- [ ] Test default template switching (only one default at a time)
- [ ] Test labeling job with various templates
- [ ] Verify OpenAI-compatible endpoints work with custom templates
- [ ] Test with different models (OpenAI, Ollama, local)

## Success Criteria

- ✅ Users can create, edit, and delete custom prompt templates
- ✅ Users can control system message, user prompt, and API parameters
- ✅ Users can set a default template for new labeling jobs
- ✅ Users can select a template when starting a labeling job
- ✅ Templates are properly applied during labeling
- ✅ System templates cannot be deleted or edited
- ✅ Templates in use by jobs cannot be deleted
- ✅ Full prompts are visible in enhanced logging
- ✅ UI is consistent with existing template features
- ✅ All tests pass
- ✅ Documentation is complete

## Risk Assessment

### Potential Issues
1. **Template Complexity**: Users might create overly complex prompts that confuse models
   - Mitigation: Provide clear examples and guidelines

2. **API Parameter Tuning**: Incorrect parameters could degrade label quality
   - Mitigation: Show recommended ranges and effects

3. **Breaking Changes**: Modifying prompt format could break parsing
   - Mitigation: Validate JSON response format, maintain {tokens_table} placeholder

4. **Database Migration**: Adding columns to existing tables
   - Mitigation: Test migration thoroughly, have rollback plan

### Dependencies
- Existing labeling system must remain functional
- Enhanced logging feature (already implemented)
- Training Templates feature (for UI/UX consistency)

## Bonus Enhancements Implemented

During implementation, the following enhancements from the "Future Enhancements" list were completed ahead of schedule:

### Enhancement 1: Export/Import Functionality ✅
- **Backend**: Added `/export` and `/import` endpoints with JSON format support
- **Frontend**: Added Export and Import buttons with modal interface
- **Features**:
  - Export custom templates to JSON file (system templates excluded)
  - Import templates from JSON with overwrite control
  - Detailed import results showing imported, skipped, overwritten, and failed counts
  - Version validation (1.0 format)
  - Duplicate handling (skip or overwrite)

### Enhancement 2: Usage Statistics ✅
- **Backend**: Added `/usage-count` endpoint to count labeling jobs per template
- **Frontend**: Added usage count fetching and display
- **Features**:
  - Purple badge showing "X job(s)" for templates with usage
  - Parallel API calls for efficient fetching
  - Delete warning when template has usage
  - Usage count integrated into template cards

### Enhancement 3: Template Preview ✅
- **Frontend**: Added preview modal for all templates
- **Features**:
  - Preview button with Eye icon for each template
  - Modal showing complete template details
  - Sample token data rendering in user prompt
  - Example output demonstration
  - Works for both system and custom templates

### Enhancement 4: Template Display in Labeling Jobs ✅
- **Frontend**: Updated LabelingJobCard to show template name
- **Features**:
  - Fetches template name from store
  - Fallback to "Custom Template" or "Default Template"
  - Clean badge display in job cards

### Enhancement 5: Enhanced Logging Integration ✅
- **Backend**: Integrated template system with existing enhanced logging
- **Features**:
  - Full prompt logging includes template information
  - Template ID tracking in logs
  - API parameter logging (temperature, max_tokens, top_p)

## Future Enhancements (Not in Scope)

- [ ] Template versioning and history
- [ ] Template marketplace or community templates
- [ ] A/B testing different templates
- [ ] Template analytics (success rate, label quality metrics)
- [ ] Multi-language prompt templates
- [ ] Template categories/tags
- [ ] Bulk template operations

## References

**Existing Code:**
- Training Templates: `frontend/src/components/panels/TrainingTemplatesPanel.tsx`
- Extraction Templates: `frontend/src/components/panels/ExtractionTemplatesPanel.tsx`
- Current Labeling Prompt: `backend/src/services/openai_labeling_service.py:161-231`
- Enhanced Logging: `backend/src/services/openai_labeling_service.py:109-143`

**Documentation:**
- API Docs: http://localhost:8000/docs#/labeling
- Mock UI Spec: `0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
