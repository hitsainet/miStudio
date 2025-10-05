# MechInterp Studio (miStudio) - Template System Specification

**Document Type**: Technical Specification
**Last Updated**: 2025-10-05
**Purpose**: Comprehensive guidance on the template/preset system for saving and reusing activity configurations

---

## Table of Contents

1. [Overview](#overview)
2. [Template Types](#template-types)
3. [Use Cases](#use-cases)
4. [Data Structures](#data-structures)
5. [API Endpoints](#api-endpoints)
6. [Database Schema](#database-schema)
7. [Frontend Implementation](#frontend-implementation)
8. [Backend Implementation](#backend-implementation)
9. [Template Lifecycle](#template-lifecycle)
10. [Best Practices](#best-practices)
11. [Security Considerations](#security-considerations)
12. [Future Enhancements](#future-enhancements)

---

## Overview

The Template System allows users to save, manage, and reuse configurations across different activities in miStudio. This feature enables:

- **Quick Recreation**: Restore exact conditions from previous sessions
- **Starting Point**: Use saved configurations as templates for new experiments
- **Best Practices**: Share proven configurations across team members
- **Experimentation**: Compare different configuration approaches

### Core Principles

1. **Flexibility**: Templates can be generic (work with any model/dataset) or specific
2. **Composability**: Templates focus on hyperparameters, not data dependencies
3. **Discoverability**: Favorite/unfavorite system helps users find important templates
4. **Persistence**: Templates survive deletion of associated models/datasets

---

## Template Types

### 1. Training Templates

**Purpose**: Store SAE (Sparse Autoencoder) training configurations

**Key Properties**:
- Training hyperparameters (learning rate, batch size, L1 coefficient, etc.)
- Encoder type (sparse, skip, transcoder)
- Optional model/dataset bindings
- Favorite flag for quick access

**Example Use Cases**:
- "Fast Prototyping" - Quick training for testing ideas
- "High Quality SAE" - Production-quality training configuration
- "Large Expansion" - High expansion factor for detailed analysis

### 2. Extraction Templates

**Purpose**: Store feature extraction configurations

**Key Properties**:
- Target layers for extraction
- Hook types (residual, mlp, attention)
- Sample limits for testing
- Top-K examples per feature

**Example Use Cases**:
- "Quick Scan" - Fast exploration of early layers
- "Full Analysis" - Comprehensive extraction across all layers
- "MLP Focus" - Target only MLP activations

### 3. Steering Presets

**Purpose**: Store feature steering configurations for model behavior modification

**Key Properties**:
- Feature IDs and coefficients
- Intervention layer
- Temperature setting
- Training ID association

**Example Use Cases**:
- "Positive Sentiment" - Enhance positive sentiment features
- "Formal Tone" - Increase formality in generations
- "Technical Language" - Boost technical vocabulary features

---

## Use Cases

### Use Case 1: Iterative Experimentation

**Scenario**: A researcher wants to test different L1 coefficients while keeping other hyperparameters constant.

**Workflow**:
1. Create initial training template "Baseline Config"
2. Start training with template
3. Observe results, adjust L1 coefficient
4. Save new template "Baseline + Higher L1"
5. Compare results across multiple training runs

**Benefits**:
- Consistent baseline across experiments
- Easy comparison of results
- Quick iteration on specific parameters

### Use Case 2: Team Collaboration

**Scenario**: A team wants to share proven configurations for different model sizes.

**Workflow**:
1. Senior researcher creates templates for common scenarios
2. Templates are marked as favorites for visibility
3. Team members apply templates to their experiments
4. Results are more comparable across team

**Benefits**:
- Standardized approaches
- Knowledge sharing
- Reduced setup time for new team members

### Use Case 3: Quick Prototyping

**Scenario**: A user wants to quickly test a new dataset with minimal configuration.

**Workflow**:
1. Select "Fast Prototyping" training template
2. Choose new dataset
3. Start training immediately
4. Adjust parameters if needed, save as new template

**Benefits**:
- Minimal friction for new experiments
- Established baseline configurations
- Progressive refinement

### Use Case 4: Model-Specific Optimization

**Scenario**: A user has found optimal settings for a specific model architecture.

**Workflow**:
1. Experiment with different configurations for TinyLlama
2. Find optimal hyperparameters
3. Save as "TinyLlama Optimized" with model_id set
4. Template automatically filters to only show for TinyLlama

**Benefits**:
- Architecture-specific optimizations preserved
- Reduced clutter in template list
- Best practices for specific models captured

---

## Data Structures

### Training Template

```typescript
interface TrainingTemplate {
  id: string;                    // Unique identifier (e.g., "tt_abc123")
  name: string;                  // User-friendly name (max 500 chars)
  description?: string;          // Optional description
  model_id?: string | null;      // null = works with any model
  dataset_id?: string | null;    // null = works with any dataset
  encoder_type: 'sparse' | 'skip' | 'transcoder';
  hyperparameters: Hyperparameters;  // Full hyperparameter object
  is_favorite: boolean;          // Quick access flag
  created_at: string;            // ISO 8601 timestamp
  updated_at: string;            // ISO 8601 timestamp
}
```

### Extraction Template

```typescript
interface ExtractionTemplate {
  id: string;                    // Unique identifier (e.g., "et_abc123")
  name: string;                  // User-friendly name (max 500 chars)
  description?: string;          // Optional description
  layers: number[];              // Which transformer layers to extract from
  hook_types: string[];          // ['residual', 'mlp', 'attention']
  max_samples?: number;          // Limit samples for testing (null = all)
  top_k_examples: number;        // Max-activating examples per feature
  is_favorite: boolean;          // Quick access flag
  created_at: string;            // ISO 8601 timestamp
  updated_at: string;            // ISO 8601 timestamp
}
```

### Steering Preset

```typescript
interface SteeringPreset {
  id: string;                    // Unique identifier (e.g., "sp_abc123")
  training_id: string;           // Associated training job
  name: string;                  // User-friendly name
  description?: string;          // Optional description
  features: Array<{              // Feature configurations
    feature_id: number;
    coefficient: number;         // -5 to +5 range
  }>;
  intervention_layer: number;    // Which layer to intervene at
  temperature: number;           // Generation temperature (0-2)
  is_favorite: boolean;          // Quick access flag
  created_at: string;            // ISO 8601 timestamp
  updated_at: string;            // ISO 8601 timestamp
}
```

### Hyperparameters

```typescript
interface Hyperparameters {
  learning_rate: number;         // 1e-6 to 1e-2
  batch_size: number;            // Power of 2, typically 32-512
  l1_coefficient: number;        // 1e-5 to 1e-1 (sparsity penalty)
  expansion_factor: number;      // 1-32 (hidden layer expansion)
  training_steps: number;        // 1000-1000000
  optimizer: 'AdamW' | 'Adam' | 'SGD';
  lr_schedule: 'constant' | 'cosine' | 'linear' | 'exponential';
  ghost_grad_penalty: boolean;   // Enable ghost gradient for dead neurons
}
```

---

## API Endpoints

### Training Templates

```
GET    /api/templates/training              List all training templates
POST   /api/templates/training              Create new training template
GET    /api/templates/training/:id          Get specific template
PUT    /api/templates/training/:id          Update template
DELETE /api/templates/training/:id          Delete template
POST   /api/templates/training/:id/apply    Apply template to current config
```

**Query Parameters** (GET list):
- `is_favorite` (boolean) - Filter by favorite status
- `model_id` (string) - Filter by model compatibility
- `dataset_id` (string) - Filter by dataset compatibility
- `page` (integer) - Pagination page number
- `limit` (integer) - Results per page

**Apply Endpoint**:
The apply endpoint allows overriding the template's model_id and dataset_id:

```json
POST /api/templates/training/:id/apply
{
  "model_id": "m_xyz789",      // Override template model
  "dataset_id": "ds_abc456"    // Override template dataset
}
```

### Extraction Templates

```
GET    /api/templates/extraction           List all extraction templates
POST   /api/templates/extraction           Create new extraction template
GET    /api/templates/extraction/:id       Get specific template
PUT    /api/templates/extraction/:id       Update template
DELETE /api/templates/extraction/:id       Delete template
POST   /api/templates/extraction/:id/apply Apply template configuration
```

**Query Parameters** (GET list):
- `is_favorite` (boolean) - Filter by favorite status
- `page` (integer) - Pagination page number
- `limit` (integer) - Results per page

### Steering Presets

Steering presets use existing endpoints with minor enhancements:

```
GET    /api/steering/presets               List all steering presets
POST   /api/steering/presets               Create new steering preset
DELETE /api/steering/presets/:id           Delete steering preset
```

---

## Database Schema

### training_templates Table

```sql
CREATE TABLE training_templates (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    model_id VARCHAR(255) REFERENCES models(id) ON DELETE SET NULL,
    dataset_id VARCHAR(255) REFERENCES datasets(id) ON DELETE SET NULL,
    encoder_type VARCHAR(50) NOT NULL,
    hyperparameters JSONB NOT NULL,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT training_templates_encoder_type_check
        CHECK (encoder_type IN ('sparse', 'skip', 'transcoder'))
);

-- Indexes for performance
CREATE INDEX idx_training_templates_is_favorite
    ON training_templates(is_favorite) WHERE is_favorite = TRUE;

CREATE INDEX idx_training_templates_created_at
    ON training_templates(created_at DESC);

CREATE INDEX idx_training_templates_model_id
    ON training_templates(model_id) WHERE model_id IS NOT NULL;

CREATE INDEX idx_training_templates_dataset_id
    ON training_templates(dataset_id) WHERE dataset_id IS NOT NULL;
```

**Key Design Decisions**:
- `ON DELETE SET NULL` for model_id/dataset_id - templates persist when models/datasets deleted
- Partial indexes on `is_favorite` - only index favorites for performance
- JSONB for hyperparameters - flexible schema for future parameter additions
- Updated timestamp triggers for automatic tracking

### extraction_templates Table

```sql
CREATE TABLE extraction_templates (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    layers INTEGER[] NOT NULL,
    hook_types VARCHAR(50)[] NOT NULL,
    max_samples INTEGER,
    top_k_examples INTEGER NOT NULL DEFAULT 100,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_extraction_templates_is_favorite
    ON extraction_templates(is_favorite) WHERE is_favorite = TRUE;

CREATE INDEX idx_extraction_templates_created_at
    ON extraction_templates(created_at DESC);
```

**Key Design Decisions**:
- PostgreSQL array types for layers and hook_types
- No foreign key constraints - fully standalone templates
- Minimal indexes for performance on common queries

### Updated steering_presets Table

The existing steering_presets table is enhanced to include:

```sql
-- Add missing columns if not present
ALTER TABLE steering_presets ADD COLUMN IF NOT EXISTS is_favorite BOOLEAN DEFAULT FALSE;
ALTER TABLE steering_presets ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE steering_presets ADD COLUMN IF NOT EXISTS temperature NUMERIC(3,2) DEFAULT 0.7;

-- Add index for favorites
CREATE INDEX IF NOT EXISTS idx_steering_presets_is_favorite
    ON steering_presets(is_favorite) WHERE is_favorite = TRUE;
```

---

## Frontend Implementation

### Component Structure

```
frontend/src/components/templates/
├── TrainingTemplates/
│   ├── TrainingTemplates.tsx         # Main template list component
│   └── index.ts
├── ExtractionTemplates/
│   ├── ExtractionTemplates.tsx       # Extraction template list
│   └── index.ts
├── TemplateCard/
│   ├── TemplateCard.tsx              # Reusable template card
│   └── index.ts
└── TemplateEditor/
    ├── TemplateEditor.tsx            # Template creation/editing
    └── index.ts
```

### State Management

Templates are loaded on application initialization:

```typescript
// In main application component
const [trainingTemplates, setTrainingTemplates] = useState<TrainingTemplate[]>([]);
const [extractionTemplates, setExtractionTemplates] = useState<ExtractionTemplate[]>([]);
const [steeringPresets, setSteeringPresets] = useState<SteeringPreset[]>([]);

useEffect(() => {
  loadDatasets();
  loadModels();
  loadTemplates();  // Load all templates on init
}, []);
```

### Template Application Flow

```typescript
// User clicks "Apply Template" on a training template
const applyTrainingTemplate = async (templateId: string) => {
  try {
    // Fetch template configuration
    const config = await api.templates.training.apply(templateId, {
      model_id: selectedModel?.id,    // Override if user selected model
      dataset_id: selectedDataset?.id // Override if user selected dataset
    });

    // Update UI state with template configuration
    setEncoderType(config.encoder_type);
    setHyperparameters(config.hyperparameters);

    // Optionally update model/dataset if template has preferences
    if (config.model_id && !selectedModel) {
      setSelectedModel(models.find(m => m.id === config.model_id));
    }

  } catch (error) {
    showError("Failed to apply template");
  }
};
```

### Template Persistence

```typescript
// Save current configuration as template
const saveAsTemplate = async () => {
  try {
    const template: TrainingTemplate = {
      name: templateName,
      description: templateDescription,
      model_id: bindToModel ? selectedModel?.id : null,
      dataset_id: bindToDataset ? selectedDataset?.id : null,
      encoder_type: currentEncoderType,
      hyperparameters: currentHyperparameters,
      is_favorite: false
    };

    const created = await api.templates.training.create(template);
    setTrainingTemplates([...trainingTemplates, created]);
    showSuccess("Template saved successfully");

  } catch (error) {
    showError("Failed to save template");
  }
};
```

### UI/UX Best Practices

1. **Clear Labeling**: Template cards show name, description, and metadata
2. **Favorite System**: Star icon for quick access to frequently used templates
3. **Filter/Search**: Allow filtering by name, favorite status, model compatibility
4. **Visual Indicators**: Show whether template is generic or model/dataset-specific
5. **Confirmation Dialogs**: Confirm before deleting templates
6. **Loading States**: Show loading indicators during template operations

---

## Backend Implementation

### Service Layer

```python
# backend/app/services/template_service.py

from typing import List, Optional
from app.db.repositories.template_repo import TemplateRepository
from app.schemas.template import (
    TrainingTemplateCreate,
    TrainingTemplateUpdate,
    TrainingTemplate
)

class TemplateService:
    def __init__(self, template_repo: TemplateRepository):
        self.template_repo = template_repo

    async def create_training_template(
        self,
        template: TrainingTemplateCreate
    ) -> TrainingTemplate:
        """Create new training template with validation"""

        # Validate hyperparameters
        self._validate_hyperparameters(template.hyperparameters)

        # Generate unique ID
        template_id = f"tt_{generate_id()}"

        # Create in database
        created = await self.template_repo.create_training_template(
            id=template_id,
            **template.dict()
        )

        return created

    async def apply_training_template(
        self,
        template_id: str,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None
    ) -> dict:
        """Apply template with optional overrides"""

        template = await self.template_repo.get_training_template(template_id)

        if not template:
            raise TemplateNotFoundError(template_id)

        # Build configuration response
        config = {
            "encoder_type": template.encoder_type,
            "hyperparameters": template.hyperparameters,
            "model_id": model_id or template.model_id,
            "dataset_id": dataset_id or template.dataset_id
        }

        return config

    def _validate_hyperparameters(self, params: dict):
        """Validate hyperparameter ranges"""

        if not (1e-6 <= params["learning_rate"] <= 1e-2):
            raise ValidationError("learning_rate out of range")

        if params["batch_size"] not in [32, 64, 128, 256, 512, 1024, 2048]:
            raise ValidationError("batch_size must be power of 2")

        # Additional validations...
```

### Repository Layer

```python
# backend/app/db/repositories/template_repo.py

from typing import List, Optional
from sqlalchemy import select
from app.db.models.training_template import TrainingTemplate
from app.db.repositories.base import BaseRepository

class TemplateRepository(BaseRepository):

    async def create_training_template(self, **kwargs) -> TrainingTemplate:
        """Create new training template"""
        template = TrainingTemplate(**kwargs)
        self.session.add(template)
        await self.session.commit()
        await self.session.refresh(template)
        return template

    async def get_training_templates(
        self,
        is_favorite: Optional[bool] = None,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        page: int = 1,
        limit: int = 50
    ) -> List[TrainingTemplate]:
        """List training templates with filters"""

        query = select(TrainingTemplate)

        if is_favorite is not None:
            query = query.where(TrainingTemplate.is_favorite == is_favorite)

        if model_id:
            query = query.where(
                (TrainingTemplate.model_id == model_id) |
                (TrainingTemplate.model_id.is_(None))
            )

        if dataset_id:
            query = query.where(
                (TrainingTemplate.dataset_id == dataset_id) |
                (TrainingTemplate.dataset_id.is_(None))
            )

        # Add pagination
        query = query.offset((page - 1) * limit).limit(limit)
        query = query.order_by(
            TrainingTemplate.is_favorite.desc(),
            TrainingTemplate.created_at.desc()
        )

        result = await self.session.execute(query)
        return result.scalars().all()
```

---

## Template Lifecycle

### Creation

1. User configures settings in UI (training, extraction, or steering)
2. User clicks "Save as Template"
3. Modal prompts for:
   - Template name (required)
   - Description (optional)
   - Model/dataset binding (optional checkboxes)
   - Favorite flag (optional)
4. Frontend sends POST request to appropriate endpoint
5. Backend validates configuration
6. Template saved to database with timestamps
7. Frontend updates template list
8. Success notification shown to user

### Discovery

1. User opens template panel or dropdown
2. Templates loaded from API (cached in frontend state)
3. Templates displayed sorted by:
   - Favorites first
   - Then by created_at (newest first)
4. User can filter by:
   - Name search
   - Favorite status
   - Model/dataset compatibility (for training templates)
5. Template cards show key metadata

### Application

1. User selects template from list
2. User clicks "Apply Template"
3. Frontend fetches template configuration
4. For training templates with model/dataset bindings:
   - If current selection matches, use template values
   - If no current selection, suggest template preferences
   - If conflict, allow user to override
5. UI updates with template values
6. User can modify values before starting job
7. Optionally, user can "Save as New Template" with modifications

### Modification

1. User selects template
2. User clicks "Edit Template"
3. Modal shows current template values
4. User modifies name, description, or marks as favorite
5. Hyperparameters/configuration generally not editable (save as new instead)
6. Frontend sends PUT request
7. Template updated in database
8. Frontend refreshes template list

### Deletion

1. User selects template
2. User clicks "Delete Template"
3. Confirmation dialog appears
4. If confirmed, DELETE request sent to API
5. Template removed from database
6. Frontend removes from template list
7. Success notification shown

**Note**: Deletion is permanent. Templates are not soft-deleted.

---

## Best Practices

### Template Naming

**Good Names**:
- "Fast Prototyping - Low Expansion"
- "TinyLlama Optimized"
- "Full Layer Scan"
- "Production SAE - High Quality"

**Bad Names**:
- "Template 1"
- "My Config"
- "Test"
- "asdf"

**Naming Convention Recommendation**:
`[Purpose] - [Key Characteristic] - [Optional Model/Dataset]`

Examples:
- "Exploration - Quick Scan - GPT2"
- "Production - High L1 - TinyLlama"
- "Testing - Small Batch"

### Template Organization

1. **Use Favorites Sparingly**: Mark only 3-5 most commonly used templates as favorites
2. **Descriptive Descriptions**: Include key parameter values in description
3. **Generic vs Specific**: Create generic templates for common patterns, specific ones for proven combinations
4. **Regular Cleanup**: Periodically review and delete unused templates
5. **Team Standards**: Agree on naming conventions within teams

### Configuration Management

1. **Start Generic**: Begin with generic templates, specialize as needed
2. **Incremental Refinement**: Save variations as new templates during experimentation
3. **Document Assumptions**: Use description field to note important context
4. **Version Control**: Include version indicators in names if iterating ("v2", "updated")

### Performance Optimization

1. **Lazy Loading**: Load templates on demand rather than all at once
2. **Caching**: Cache template lists in frontend state
3. **Pagination**: Use pagination for large template collections
4. **Indexes**: Leverage database indexes for filtered queries

---

## Security Considerations

### Access Control

Currently, the template system does not implement user-level access control. All templates are shared across the application. Future enhancements should include:

1. **User Ownership**: Templates belong to users who created them
2. **Sharing Permissions**: Private, team-shared, or public templates
3. **Read/Write Permissions**: Separate permissions for viewing and modifying

### Input Validation

1. **Name Length**: Enforce 500 character limit on names
2. **Description Sanitization**: Prevent XSS in description fields
3. **Hyperparameter Ranges**: Validate all numeric parameters
4. **Array Bounds**: Validate layer arrays are within model bounds

### Data Integrity

1. **Foreign Key Handling**: Use ON DELETE SET NULL for optional references
2. **JSON Validation**: Validate JSONB hyperparameters match schema
3. **Constraint Checks**: Use database constraints for enum values
4. **Atomic Operations**: Use transactions for multi-step operations

---

## Future Enhancements

### Version 1.1 - Enhanced Discovery

- [ ] Template tags/categories for better organization
- [ ] Search by hyperparameter values
- [ ] Template usage statistics (how often applied)
- [ ] "Recently used" section

### Version 1.2 - Collaboration Features

- [ ] User ownership and permissions
- [ ] Template sharing with specific users/teams
- [ ] Template comments and ratings
- [ ] Import/export templates as JSON files

### Version 1.3 - Intelligence Features

- [ ] Template recommendations based on model/dataset
- [ ] Performance metrics linked to templates (which templates produce best results)
- [ ] Auto-save current configuration as draft template
- [ ] Template diff view (compare two templates)

### Version 2.0 - Advanced Features

- [ ] Template versioning (track changes over time)
- [ ] Template inheritance (templates derived from other templates)
- [ ] Conditional templates (different configs based on context)
- [ ] Template validation against model architecture
- [ ] Bulk template operations (export all, import set)

---

## Related Documentation

- [Folder Structure Specification](./001_SPEC|Folder_File_Details.md)
- [PostgreSQL Database Specification](./003_SPEC|Postgres_Usecase_Details_and_Guidance.md)
- [OpenAPI Specification](./openapi.yaml)
- [Training Types](./src/types/training.types.ts)
- [Mock UI Implementation](./Mock-embedded-interp-ui.tsx)

---

## Appendix: Example Templates

### Example Training Template: Fast Prototyping

```json
{
  "id": "tt_fast_proto",
  "name": "Fast Prototyping",
  "description": "Quick SAE training for testing ideas. Low training steps, small batch size, moderate expansion factor.",
  "model_id": null,
  "dataset_id": null,
  "encoder_type": "sparse",
  "hyperparameters": {
    "learning_rate": 0.0003,
    "batch_size": 128,
    "l1_coefficient": 0.001,
    "expansion_factor": 4,
    "training_steps": 5000,
    "optimizer": "AdamW",
    "lr_schedule": "constant",
    "ghost_grad_penalty": true
  },
  "is_favorite": true,
  "created_at": "2025-10-05T10:00:00Z",
  "updated_at": "2025-10-05T10:00:00Z"
}
```

### Example Extraction Template: Quick Scan

```json
{
  "id": "et_quick_scan",
  "name": "Quick Scan",
  "description": "Fast feature extraction for initial exploration. Early layers only, residual stream, limited samples.",
  "layers": [0, 4, 8],
  "hook_types": ["residual"],
  "max_samples": 1000,
  "top_k_examples": 10,
  "is_favorite": false,
  "created_at": "2025-10-05T10:15:00Z",
  "updated_at": "2025-10-05T10:15:00Z"
}
```

### Example Steering Preset: Positive Sentiment

```json
{
  "id": "sp_positive",
  "training_id": "tr_abc123",
  "name": "Positive Sentiment",
  "description": "Enhance positive sentiment features for cheerful text generation",
  "features": [
    {"feature_id": 42, "coefficient": 2.5},
    {"feature_id": 137, "coefficient": 1.8},
    {"feature_id": 891, "coefficient": 3.0}
  ],
  "intervention_layer": 8,
  "temperature": 0.7,
  "is_favorite": true,
  "created_at": "2025-10-05T10:30:00Z",
  "updated_at": "2025-10-05T10:30:00Z"
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-05
**Maintained By**: MechInterp Studio Team
