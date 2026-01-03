# Task: Labeling Prompt Template Variable Insertion UI

## Overview
Add a dropdown and "Insert" button to the Labeling Prompt Template editor modal that allows users to easily insert template variables at the cursor position in the System Message or User Prompt Template textareas.

## Target File
`frontend/src/components/panels/LabelingPromptTemplatesPanel.tsx`

## Available Template Variables
These variables are substituted by the backend labeling service when generating prompts:

| Variable | Description |
|----------|-------------|
| `{examples_block}` | Activation examples with `<< >>` markers around prime tokens |
| `{tokens_table}` | NLP token frequency table (token → count) |
| `{analysis_block}` | NLP statistical analysis summary section |
| `{feature_id}` | The feature/neuron ID |
| `{neuron_index}` | Neuron index number |
| `{layer_name}` | Layer name |
| `{top_boosted_tokens}` | Logit lens - tokens boosted by this feature |
| `{top_suppressed_tokens}` | Logit lens - tokens suppressed by this feature |

## Implementation Tasks

### 1. Define TEMPLATE_VARIABLES Constant
- [ ] Add constant array at top of file with all variables and descriptions
```typescript
const TEMPLATE_VARIABLES = [
  { value: '{examples_block}', label: 'Examples Block', description: 'Activation examples with << >> markers around prime tokens' },
  { value: '{tokens_table}', label: 'Tokens Table', description: 'NLP token frequency table' },
  { value: '{analysis_block}', label: 'Analysis Block', description: 'NLP statistical analysis summary' },
  { value: '{feature_id}', label: 'Feature ID', description: 'The feature/neuron ID' },
  { value: '{neuron_index}', label: 'Neuron Index', description: 'Neuron index number' },
  { value: '{layer_name}', label: 'Layer Name', description: 'Layer name' },
  { value: '{top_boosted_tokens}', label: 'Top Boosted', description: 'Logit lens - tokens boosted by feature' },
  { value: '{top_suppressed_tokens}', label: 'Top Suppressed', description: 'Logit lens - tokens suppressed by feature' },
];
```

### 2. Add State Variables
- [ ] `selectedVariable: string` - Currently selected variable in dropdown
- [ ] `activeTextField: 'system' | 'user' | null` - Which textarea is active
- [ ] `cursorPosition: number` - Last known cursor position in active textarea

### 3. Add Textarea Refs
- [ ] `systemMessageRef: React.RefObject<HTMLTextAreaElement>`
- [ ] `userPromptRef: React.RefObject<HTMLTextAreaElement>`

### 4. Add Event Handlers
- [ ] `onFocus` handlers on both textareas to set `activeTextField`
- [ ] `onSelect` handlers on both textareas to capture `selectionStart` as `cursorPosition`
- [ ] `onClick` on textareas should also update cursor position

### 5. Create insertVariable() Function
```typescript
const insertVariable = () => {
  if (!selectedVariable || !activeTextField) return;

  const ref = activeTextField === 'system' ? systemMessageRef : userPromptRef;
  const field = activeTextField === 'system' ? 'system_message' : 'user_prompt_template';
  const currentValue = formData[field] || '';
  const pos = cursorPosition ?? currentValue.length;

  const newValue = currentValue.slice(0, pos) + selectedVariable + currentValue.slice(pos);
  setFormData({ ...formData, [field]: newValue });

  // Restore focus and set cursor after inserted variable
  setTimeout(() => {
    if (ref.current) {
      ref.current.focus();
      const newPos = pos + selectedVariable.length;
      ref.current.setSelectionRange(newPos, newPos);
    }
  }, 0);
};
```

### 6. Add UI Components
- [ ] Add row below User Prompt Template textarea in BOTH modals (Create New and Edit)
- [ ] Dropdown select with TEMPLATE_VARIABLES options
- [ ] "Insert" button next to dropdown

```tsx
{/* Template Variable Insertion */}
<div className="flex items-center gap-3 mt-2">
  <select
    value={selectedVariable}
    onChange={(e) => setSelectedVariable(e.target.value)}
    className="flex-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 text-sm focus:outline-none focus:border-emerald-500"
  >
    <option value="">Select variable to insert...</option>
    {TEMPLATE_VARIABLES.map((v) => (
      <option key={v.value} value={v.value} title={v.description}>
        {v.label} - {v.value}
      </option>
    ))}
  </select>
  <button
    type="button"
    onClick={insertVariable}
    disabled={!selectedVariable || !activeTextField}
    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white text-sm rounded transition-colors"
  >
    Insert
  </button>
</div>
<p className="text-xs text-slate-500 mt-1">
  Click in a text field above, then select a variable and click Insert
</p>
```

### 7. Styling
- [ ] Match existing modal styling (slate-800 backgrounds, slate-700 borders)
- [ ] Emerald accent for Insert button
- [ ] Disabled state when no variable selected or no active field

### 8. Testing
- [ ] Test insertion in Create New Template modal
- [ ] Test insertion in Edit Template modal
- [ ] Verify cursor position is correctly tracked after clicks
- [ ] Verify inserted variable appears at correct position
- [ ] Verify focus returns to textarea after insertion

## Notes
- The component has TWO modal forms: Create New (around line 550-640) and Edit (around line 700-780)
- Both need the insertion UI added
- Consider extracting the insertion UI into a small sub-component to avoid duplication

## Related Files
- `backend/src/services/openai_labeling_service.py` - Where variables are substituted (lines 807-990)
- `backend/src/services/local_labeling_service.py` - Alternative labeling service

## Priority
Low - Quality of life improvement for template editing

## Estimated Effort
~1-2 hours
