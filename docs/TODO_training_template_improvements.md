# Training UX Improvements

**Created:** 2025-12-06
**Status:** ✅ Completed
**Priority:** Medium
**Completed:** 2025-12-06

## Overview

Improve the Training user experience in these areas:
1. Show selected template name in the Load Template dropdown
2. Auto-generate informative template names and descriptions when saving
3. Fix Live Metrics charts showing empty/no data
4. Fix Training Logs not persisting when panel is reopened

---

## Issue 1: Load Template Dropdown Resets After Selection

### Current Behavior
- User selects a template from "Load Template" dropdown
- Template is loaded and form fields are populated
- Dropdown immediately resets back to "Select a template... (N available)"
- User has no visual indicator of which template they loaded

### Desired Behavior
- After selecting a template, the dropdown should show the selected template name
- User can clearly see which template is currently loaded
- Selecting a different template updates the display

### Files to Modify

1. **`frontend/src/components/training/TemplateSelector.tsx`**
   - Line 64-69: Remove the reset of `e.target.value = ''` after selection
   - Add state to track the selected template ID
   - Change from uncontrolled to controlled component

   ```tsx
   // Add state for selected template
   const [selectedTemplateId, setSelectedTemplateId] = useState<string>('');

   // Change select to be controlled
   <select
     value={selectedTemplateId}
     onChange={(e) => {
       if (e.target.value) {
         setSelectedTemplateId(e.target.value);
         handleTemplateSelect(e.target.value);
       }
     }}
   >
   ```

2. **`frontend/src/components/panels/TrainingPanel.tsx`**
   - Line 713-722: Pass a callback to reset template selection when config changes significantly
   - Consider adding a "Clear Template" button or visual indicator

### Tasks

- [x] 1.1 Add `selectedTemplateId` state to `TemplateSelector.tsx`
- [x] 1.2 Convert `<select>` from uncontrolled to controlled component
- [x] 1.3 Update the default option text to show selected template name or placeholder
- [x] 1.4 Add clear button to reset selection
- [x] 1.5 Test template selection persists visually after loading

---

## Issue 2: Auto-Generate Informative Template Names and Descriptions

### Current Behavior
- When user clicks "Save as Template", they see empty name and description fields
- Placeholder text provides examples but user must type everything manually
- No automatic generation based on current configuration

### Desired Behavior
- Pre-populate template name with a pattern like:
  ```
  {ModelName}_{DatasetName}_{Architecture}_L{Layer(s)}
  ```
  Example: `Phi-4-mini_small-the-pile_JumpReLU_L28`

- Pre-populate description with detailed configuration:
  ```
  Hidden: 3072 → Latent: 24576 (8x) | L1: 0.0001 | LR: 0.00027 | Batch: 4096 | Steps: 50k | Layers: 28
  ```

### Files to Modify

1. **`frontend/src/components/panels/TrainingPanel.tsx`**

   **Add helper function (after line ~450):**
   ```tsx
   // Generate default template name and description based on current config
   const generateTemplateDefaults = useMemo(() => {
     const model = models.find(m => m.id === config.model_id);
     const dataset = datasets.find(d => d.id === config.dataset_id);

     // Extract short model name (e.g., "Phi-4-mini-instruct" -> "Phi-4-mini")
     const modelShort = model?.name?.split('-').slice(0, 3).join('-') || 'Model';

     // Extract short dataset name
     const datasetShort = dataset?.name?.replace(/_/g, '-') || 'Dataset';

     // Architecture short name
     const archMap: Record<string, string> = {
       'standard': 'Std',
       'skip': 'Skip',
       'transcoder': 'Trans',
       'jumprelu': 'JumpReLU',
     };
     const archShort = archMap[config.architecture_type] || config.architecture_type;

     // Layers
     const layers = config.training_layers?.length === 1
       ? `L${config.training_layers[0]}`
       : `L${config.training_layers?.[0]}-${config.training_layers?.[config.training_layers.length - 1]}`;

     const name = `${modelShort}_${datasetShort}_${archShort}_${layers}`;

     // Detailed description
     const multiplier = Math.round(config.latent_dim / config.hidden_dim);
     const stepsK = config.total_steps >= 1000 ? `${Math.round(config.total_steps / 1000)}k` : config.total_steps;
     const layerList = config.training_layers?.join(', ') || '0';

     const description = [
       `Hidden: ${config.hidden_dim} → Latent: ${config.latent_dim} (${multiplier}x)`,
       `L1: ${config.l1_alpha}`,
       `LR: ${config.learning_rate}`,
       `Batch: ${config.batch_size}`,
       `Steps: ${stepsK}`,
       `Layers: ${layerList}`,
     ].join(' | ');

     return { name, description };
   }, [config, models, datasets]);
   ```

   **Update Save Template Modal open handler (around line 1350):**
   ```tsx
   onClick={() => {
     const defaults = generateTemplateDefaults;
     setTemplateName(defaults.name);
     setTemplateDescription(defaults.description);
     setShowSaveTemplateModal(true);
   }}
   ```

### Information to Include in Auto-Generated Name

| Component | Source | Example |
|-----------|--------|---------|
| Model name | `selectedModel.name` (shortened) | `Phi-4-mini` |
| Dataset name | `selectedDataset.name` | `small-the-pile` |
| Architecture | `config.architecture_type` | `JumpReLU` |
| Layer(s) | `config.training_layers` | `L28` or `L0-31` |

### Information to Include in Auto-Generated Description

| Parameter | Format | Example |
|-----------|--------|---------|
| Hidden → Latent | `Hidden: {dim} → Latent: {dim} ({mult}x)` | `Hidden: 3072 → Latent: 24576 (8x)` |
| L1 Alpha | `L1: {value}` | `L1: 0.0001` |
| Learning Rate | `LR: {value}` | `LR: 0.00027` |
| Batch Size | `Batch: {value}` | `Batch: 4096` |
| Total Steps | `Steps: {value}k` | `Steps: 50k` |
| Layers | `Layers: {list}` | `Layers: 28` or `Layers: 0, 14, 28` |
| Target L0 (if set) | `L0: {value}` | `L0: 0.05` |
| Top-K (if set) | `TopK: {value}%` | `TopK: 5%` |

### Additional Considerations for JumpReLU

When architecture is JumpReLU, also include in description:
- Sparsity Coeff: `SparsityCoeff: {value}`
- Initial Threshold: `Thresh: {value}`
- Bandwidth: `BW: {value}`

### Tasks

- [x] 2.1 Create `generateTemplateDefaults` useMemo hook in TrainingPanel.tsx
- [x] 2.2 Extract short model name from full model name
- [x] 2.3 Extract short dataset name (handle underscores/hyphens)
- [x] 2.4 Map architecture type to short display name
- [x] 2.5 Format layer selection (single layer vs range)
- [x] 2.6 Generate compact but informative description string
- [x] 2.7 Handle JumpReLU-specific parameters in description
- [x] 2.8 Update "Save as Template" button onClick to pre-populate fields
- [x] 2.9 User can still edit the auto-generated values before saving
- [x] 2.10 Test with various model/dataset/architecture combinations

---

## Testing Checklist

- [ ] Load a template and verify dropdown shows selected template name
- [ ] Change model/dataset and verify template selector behavior
- [ ] Open "Save as Template" modal and verify auto-populated name
- [ ] Verify description includes all key hyperparameters
- [ ] Test with JumpReLU architecture for additional parameters
- [ ] Test with single layer vs multiple layers
- [ ] Verify user can edit auto-generated values
- [ ] Test with long model/dataset names (truncation)

---

## Related Files

| File | Purpose |
|------|---------|
| `frontend/src/components/training/TemplateSelector.tsx` | Template dropdown component |
| `frontend/src/components/panels/TrainingPanel.tsx` | Main training panel with save modal |
| `frontend/src/types/trainingTemplate.ts` | Type definitions |
| `frontend/src/stores/trainingTemplatesStore.ts` | Template state management |
| `frontend/src/components/training/TrainingCard.tsx` | Live metrics display (Issues 3 & 4) |
| `frontend/src/api/trainings.ts` | Training API functions (to add metrics fetch) |
| `backend/src/api/v1/endpoints/trainings.py` | Backend training endpoints |

---

## Issue 3: Live Metrics Shows Empty Charts

### Current Behavior
- User clicks "Show Live Metrics" on an active training job
- Charts appear but show no data or only partial data
- Data only starts appearing after WebSocket updates arrive
- If training has been running for a while, historical data is lost

### Root Cause
The `metricsHistory` state in `TrainingCard.tsx` (lines 94-100) only populates from WebSocket updates:

```tsx
// Current implementation - only captures new updates
useEffect(() => {
  if (training.current_loss !== undefined && training.current_loss !== null) {
    setMetricsHistory((prev) => {
      const newLoss = [...prev.loss, training.current_loss!];
      const newL0 = [...prev.l0_sparsity, training.current_l0_sparsity ?? 0];
      const newTimestamps = [...prev.timestamps, new Date().toISOString()];
      return {
        loss: newLoss.slice(-20),
        l0_sparsity: newL0.slice(-20),
        timestamps: newTimestamps.slice(-20),
      };
    });
  }
}, [training.current_loss, training.current_l0_sparsity, training.current_step]);
```

**Problem:** No historical data is fetched when the panel is opened.

### Desired Behavior
- When "Show Live Metrics" is clicked, fetch recent metrics history from backend
- Pre-populate charts with last N data points (e.g., last 20 metrics)
- Continue updating via WebSocket for new data points
- Charts should never appear empty for an active training with metrics

### Files to Modify

1. **`frontend/src/components/training/TrainingCard.tsx`**
   - Add API call to fetch metrics history when `showLiveMetrics` becomes true
   - Populate `metricsHistory` state with historical data before WebSocket updates

   ```tsx
   // Add effect to fetch historical metrics when panel opens
   useEffect(() => {
     if (showLiveMetrics && training.status === 'running') {
       // Fetch last 20 metrics from backend
       fetchTrainingMetrics(training.id, { limit: 20 })
         .then((metrics) => {
           setMetricsHistory({
             loss: metrics.map(m => m.loss),
             l0_sparsity: metrics.map(m => m.l0_sparsity ?? 0),
             timestamps: metrics.map(m => m.timestamp),
           });
         })
         .catch(console.error);
     }
   }, [showLiveMetrics, training.id, training.status]);
   ```

2. **`frontend/src/api/trainings.ts`** (or create if needed)
   - Add `fetchTrainingMetrics(trainingId, options)` API function

3. **`backend/src/api/v1/endpoints/trainings.py`**
   - Verify endpoint exists: `GET /api/v1/trainings/{training_id}/metrics`
   - Should return recent metrics with configurable limit

### Backend Endpoint Check

Need to verify the backend has an endpoint like:
```
GET /api/v1/trainings/{training_id}/metrics?limit=20&order=desc
```

Response format:
```json
{
  "data": [
    {
      "step": 1000,
      "loss": 0.45,
      "l0_sparsity": 0.02,
      "timestamp": "2025-12-06T14:30:00Z"
    },
    ...
  ]
}
```

### Tasks

- [x] 3.1 Verify backend metrics endpoint exists and returns historical data
- [x] 3.2 Add `fetchTrainingMetrics` API function to frontend
- [x] 3.3 Add useEffect to fetch metrics when `showLiveMetrics` becomes true
- [x] 3.4 Populate `metricsHistory` with fetched data
- [x] 3.5 Ensure WebSocket updates append to (not replace) historical data
- [x] 3.6 Handle loading state while fetching metrics
- [x] 3.7 Handle error state if metrics fetch fails
- [x] 3.8 Test with training that has been running for a while

---

## Issue 4: Training Logs Not Persisting

### Current Behavior
- Training logs only show entries that arrive via WebSocket after panel is opened
- If training has been running, previous log entries are not visible
- Closing and reopening the panel loses all accumulated logs

### Desired Behavior
- Fetch recent log entries when panel opens
- Show last N log messages (e.g., last 50)
- Continue appending new logs via WebSocket

### Tasks

- [x] 4.1 Verify backend has log retrieval endpoint (uses same metrics endpoint)
- [x] 4.2 Fetch recent logs when `showLiveMetrics` becomes true (done via metrics fetch)
- [x] 4.3 Display historical logs before WebSocket-delivered logs
- [x] 4.4 Test log continuity when reopening panel

---

## Estimated Effort

- Issue 1 (Dropdown persistence): ~30 minutes
- Issue 2 (Auto-generate name/description): ~1 hour
- Issue 3 (Live Metrics historical data): ~1-2 hours
- Issue 4 (Training logs persistence): ~1 hour
- Testing: ~1 hour

**Total: ~5-6 hours**
