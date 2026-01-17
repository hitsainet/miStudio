# Bug Fix: Multi-SAE Import from Training

## Issue Summary
When training produces multiple SAEs (e.g., 2 layers × 3 hook_types = 6 SAEs), the `import_from_training` function only imports the first layer, ignoring all other layers and hook_types.

**Root Cause:** [sae_manager_service.py:222-224](backend/src/services/sae_manager_service.py#L222-L224)
```python
training_layers = hyperparams.get("training_layers", [])
layer = training_layers[0] if training_layers else hyperparams.get("target_layer")
```
Only the first layer is extracted; subsequent layers and hook_types are ignored.

## Expected Behavior
- User can select which SAEs to import (by layer + hook_type combination)
- User can choose to import all SAEs at once
- Each imported SAE gets its own entry in the ExternalSAE table

---

## Implementation Tasks

### Phase 1: Backend - Schema Updates

#### Task 1.1: Update SAEImportFromTrainingRequest Schema
**File:** [backend/src/schemas/sae.py:126-131](backend/src/schemas/sae.py#L126-L131)

- [ ] Add optional `layers` field (List[int]) to specify which layers to import
- [ ] Add optional `hook_types` field (List[str]) to specify which hook types to import
- [ ] Add `import_all` boolean field (default: True for backward compatibility)
- [ ] Add docstring explaining the selection behavior

**New Schema:**
```python
class SAEImportFromTrainingRequest(BaseModel):
    """Schema for importing SAE(s) from a completed training job.

    For multi-layer/multi-hook trainings, can import:
    - All SAEs: import_all=True (default)
    - Specific SAEs: import_all=False with layers and/or hook_types specified
    """
    training_id: str = Field(..., description="Training job ID to import from")
    name: Optional[str] = Field(None, max_length=255, description="Display name prefix")
    description: Optional[str] = Field(None, max_length=2000, description="Optional description")
    import_all: bool = Field(True, description="Import all available SAEs")
    layers: Optional[List[int]] = Field(None, description="Specific layers to import (if import_all=False)")
    hook_types: Optional[List[str]] = Field(None, description="Specific hook types to import (if import_all=False)")
```

#### Task 1.2: Create SAEImportFromTrainingResponse Schema
**File:** [backend/src/schemas/sae.py](backend/src/schemas/sae.py) (add after line 131)

- [ ] Create response schema for multi-SAE import
- [ ] Include list of created SAE IDs
- [ ] Include summary of what was imported

```python
class SAEImportFromTrainingResponse(BaseModel):
    """Response schema for importing SAEs from training."""
    imported_count: int = Field(..., description="Number of SAEs imported")
    sae_ids: List[str] = Field(..., description="List of created SAE IDs")
    saes: List[SAEResponse] = Field(..., description="List of created SAE objects")
    training_id: str = Field(..., description="Source training ID")
    message: str = Field(..., description="Summary message")
```

#### Task 1.3: Create AvailableSAEInfo Schema for Preview
**File:** [backend/src/schemas/sae.py](backend/src/schemas/sae.py)

- [ ] Create schema to describe available SAEs in a training
- [ ] Include layer, hook_type, path, and size info

```python
class AvailableSAEInfo(BaseModel):
    """Info about an available SAE in a training checkpoint."""
    layer: int = Field(..., description="Layer index")
    hook_type: str = Field(..., description="Hook type (e.g., 'hook_resid_pre')")
    path: str = Field(..., description="Path within checkpoint")
    size_bytes: Optional[int] = Field(None, description="SAE file size")

class TrainingAvailableSAEsResponse(BaseModel):
    """Response listing available SAEs in a completed training."""
    training_id: str = Field(..., description="Training ID")
    available_saes: List[AvailableSAEInfo] = Field(..., description="Available SAEs")
    total_count: int = Field(..., description="Total available SAE count")
```

---

### Phase 2: Backend - Service Updates

#### Task 2.1: Add Helper to Enumerate Available SAEs
**File:** [backend/src/services/sae_manager_service.py](backend/src/services/sae_manager_service.py)

- [ ] Create `get_available_saes_from_training()` method
- [ ] Scan community_format directory for layer_* subdirectories
- [ ] Extract hook_type from directory structure or cfg.json
- [ ] Return list of (layer, hook_type, path) tuples

```python
async def get_available_saes_from_training(
    self, db: AsyncSession, training_id: str
) -> List[AvailableSAEInfo]:
    """Get list of available SAEs from a completed training.

    Scans the community_format directory for layer subdirectories
    and extracts layer/hook_type information.
    """
    # Implementation scans community_format/layer_*/hook_*/
    # or community_format/layer_*/ with hook_type in cfg.json
```

#### Task 2.2: Update import_from_training to Handle Multiple SAEs
**File:** [backend/src/services/sae_manager_service.py:196-320](backend/src/services/sae_manager_service.py#L196-L320)

- [ ] Modify to accept updated SAEImportFromTrainingRequest
- [ ] If import_all=True, enumerate and import all available SAEs
- [ ] If import_all=False, filter by specified layers/hook_types
- [ ] Create separate ExternalSAE record for each imported SAE
- [ ] Generate unique names with layer/hook suffix (e.g., "SAE from model (L12-resid_pre)")
- [ ] Return SAEImportFromTrainingResponse with all created SAEs

**Key Changes:**
```python
async def import_from_training(
    self, db: AsyncSession, request: SAEImportFromTrainingRequest
) -> SAEImportFromTrainingResponse:
    # 1. Get available SAEs
    available = await self.get_available_saes_from_training(db, request.training_id)

    # 2. Filter if not import_all
    if not request.import_all:
        available = [s for s in available
                     if (not request.layers or s.layer in request.layers)
                     and (not request.hook_types or s.hook_type in request.hook_types)]

    # 3. Import each SAE
    created_saes = []
    for sae_info in available:
        sae = await self._import_single_sae(db, request, sae_info)
        created_saes.append(sae)

    return SAEImportFromTrainingResponse(
        imported_count=len(created_saes),
        sae_ids=[s.id for s in created_saes],
        saes=created_saes,
        training_id=request.training_id,
        message=f"Imported {len(created_saes)} SAE(s)"
    )
```

#### Task 2.3: Extract _import_single_sae Helper Method
**File:** [backend/src/services/sae_manager_service.py](backend/src/services/sae_manager_service.py)

- [ ] Refactor current single-SAE import logic into helper method
- [ ] Accept layer and hook_type as parameters
- [ ] Generate appropriate name suffix based on layer/hook_type

---

### Phase 3: Backend - API Updates

#### Task 3.1: Add Endpoint to List Available SAEs from Training
**File:** [backend/src/api/v1/endpoints/saes.py](backend/src/api/v1/endpoints/saes.py)

- [ ] Add `GET /api/v1/saes/training/{training_id}/available` endpoint
- [ ] Return TrainingAvailableSAEsResponse
- [ ] Validate training exists and is COMPLETED

```python
@router.get("/training/{training_id}/available", response_model=TrainingAvailableSAEsResponse)
async def get_available_saes_from_training(
    training_id: str,
    db: AsyncSession = Depends(get_async_session),
) -> TrainingAvailableSAEsResponse:
    """List available SAEs in a completed training for import."""
    sae_manager = SAEManagerService()
    available = await sae_manager.get_available_saes_from_training(db, training_id)
    return TrainingAvailableSAEsResponse(
        training_id=training_id,
        available_saes=available,
        total_count=len(available)
    )
```

#### Task 3.2: Update Import Endpoint Response Type
**File:** [backend/src/api/v1/endpoints/saes.py](backend/src/api/v1/endpoints/saes.py)

- [ ] Update `/api/v1/saes/import/training` response model to SAEImportFromTrainingResponse
- [ ] Ensure backward compatibility (single SAE import still works)

---

### Phase 4: Frontend - Types and API

#### Task 4.1: Update TypeScript Types
**File:** [frontend/src/types/sae.ts](frontend/src/types/sae.ts)

- [ ] Add `AvailableSAEInfo` interface
- [ ] Add `TrainingAvailableSAEsResponse` interface
- [ ] Update `SAEImportFromTrainingRequest` interface
- [ ] Add `SAEImportFromTrainingResponse` interface

```typescript
interface AvailableSAEInfo {
  layer: number;
  hook_type: string;
  path: string;
  size_bytes?: number;
}

interface TrainingAvailableSAEsResponse {
  training_id: string;
  available_saes: AvailableSAEInfo[];
  total_count: number;
}

interface SAEImportFromTrainingRequest {
  training_id: string;
  name?: string;
  description?: string;
  import_all?: boolean;
  layers?: number[];
  hook_types?: string[];
}

interface SAEImportFromTrainingResponse {
  imported_count: number;
  sae_ids: string[];
  saes: SAE[];
  training_id: string;
  message: string;
}
```

#### Task 4.2: Add API Function to Get Available SAEs
**File:** [frontend/src/api/saes.ts](frontend/src/api/saes.ts)

- [ ] Add `getAvailableSAEsFromTraining(trainingId: string)` function
- [ ] Returns `TrainingAvailableSAEsResponse`

---

### Phase 5: Frontend - UI Components

#### Task 5.1: Create SAEImportModal Component
**File:** `frontend/src/components/training/SAEImportModal.tsx` (new file)

- [ ] Create modal dialog for SAE import selection
- [ ] Fetch available SAEs on mount using new API endpoint
- [ ] Display grid/list of available SAEs with checkboxes
- [ ] Show layer and hook_type for each SAE
- [ ] Include "Select All" / "Deselect All" buttons
- [ ] Show total size to be imported
- [ ] Import button with loading state

**Component Structure:**
```tsx
interface SAEImportModalProps {
  training: Training;
  isOpen: boolean;
  onClose: () => void;
  onImportComplete: (response: SAEImportFromTrainingResponse) => void;
}

const SAEImportModal: React.FC<SAEImportModalProps> = ({
  training, isOpen, onClose, onImportComplete
}) => {
  const [availableSAEs, setAvailableSAEs] = useState<AvailableSAEInfo[]>([]);
  const [selectedSAEs, setSelectedSAEs] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);
  const [isImporting, setIsImporting] = useState(false);

  // Fetch available SAEs on mount
  // Render selection grid
  // Handle import with selected SAEs
};
```

#### Task 5.2: Update TrainingCard to Use Modal
**File:** [frontend/src/components/training/TrainingCard.tsx:252-268](frontend/src/components/training/TrainingCard.tsx#L252-L268)

- [ ] Add state for modal open/close
- [ ] Change "Import to SAEs" button to open modal instead of direct import
- [ ] Handle import completion callback (refresh SAE list, show success)
- [ ] Update imported state to track count of imported SAEs

**Changes:**
```tsx
// Add state
const [isImportModalOpen, setIsImportModalOpen] = useState(false);
const [importedCount, setImportedCount] = useState(0);

// Change button handler
const handleImportClick = () => {
  setIsImportModalOpen(true);
};

// Handle import completion
const handleImportComplete = (response: SAEImportFromTrainingResponse) => {
  setImportedCount(response.imported_count);
  setSaeImported(true);
  setIsImportModalOpen(false);
};

// Render modal
{isImportModalOpen && (
  <SAEImportModal
    training={training}
    isOpen={isImportModalOpen}
    onClose={() => setIsImportModalOpen(false)}
    onImportComplete={handleImportComplete}
  />
)}
```

#### Task 5.3: Update Button Text to Show Import Count
**File:** [frontend/src/components/training/TrainingCard.tsx](frontend/src/components/training/TrainingCard.tsx)

- [ ] Show "Imported (N)" instead of just checkmark when SAEs imported
- [ ] Update tooltip to show how many SAEs were imported

---

### Phase 6: Testing

#### Task 6.1: Backend Unit Tests
**File:** `backend/tests/unit/test_sae_import.py` (new file)

- [ ] Test get_available_saes_from_training with single layer
- [ ] Test get_available_saes_from_training with multiple layers
- [ ] Test import_from_training with import_all=True
- [ ] Test import_from_training with specific layers filter
- [ ] Test import_from_training with specific hook_types filter
- [ ] Test backward compatibility (old request format still works)

#### Task 6.2: Backend Integration Tests
**File:** `backend/tests/integration/test_sae_import_integration.py` (new file)

- [ ] Test full workflow: training → get available → import selected
- [ ] Test API endpoint responses
- [ ] Test error handling (training not found, not completed, etc.)

#### Task 6.3: Frontend Component Tests
**File:** `frontend/src/components/training/SAEImportModal.test.tsx` (new file)

- [ ] Test modal renders with available SAEs
- [ ] Test selection/deselection
- [ ] Test import flow
- [ ] Test error states

---

## Directory Structure Reference

Training checkpoint structure:
```
/data/trainings/{training_id}/
├── community_format/
│   ├── layer_0/
│   │   ├── hook_resid_pre/
│   │   │   ├── cfg.json
│   │   │   ├── sae_weights.safetensors
│   │   │   └── ...
│   │   ├── hook_resid_post/
│   │   │   └── ...
│   │   └── hook_mlp_out/
│   │       └── ...
│   └── layer_1/
│       ├── hook_resid_pre/
│       │   └── ...
│       └── ...
└── checkpoints/
    └── ...
```

---

## Acceptance Criteria

1. [ ] User can view list of available SAEs before importing
2. [ ] User can select specific SAEs by layer and/or hook_type
3. [ ] User can import all SAEs with one click
4. [ ] Each imported SAE gets a unique name with layer/hook suffix
5. [ ] Import progress/status is clearly shown
6. [ ] All existing single-SAE imports continue to work (backward compat)
7. [ ] All tests pass
8. [ ] No regressions in SAE functionality

---

## Files Modified Summary

| Phase | File | Type | Description |
|-------|------|------|-------------|
| 1 | backend/src/schemas/sae.py | Edit | Add new schemas |
| 2 | backend/src/services/sae_manager_service.py | Edit | Multi-SAE import logic |
| 3 | backend/src/api/v1/endpoints/saes.py | Edit | New endpoint + update response |
| 4 | frontend/src/types/sae.ts | Edit | TypeScript interfaces |
| 4 | frontend/src/api/saes.ts | Edit | New API function |
| 5 | frontend/src/components/training/SAEImportModal.tsx | New | Selection modal |
| 5 | frontend/src/components/training/TrainingCard.tsx | Edit | Use modal |
| 6 | backend/tests/unit/test_sae_import.py | New | Unit tests |
| 6 | backend/tests/integration/test_sae_import_integration.py | New | Integration tests |
| 6 | frontend/src/components/training/SAEImportModal.test.tsx | New | Component tests |
